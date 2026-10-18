/**
 * `neurarch-mcp hook`: a Claude Code PreToolUse hook that sizes a training
 * launch before the Bash call runs it.
 *
 * Claude Code pipes the pending tool call to the hook as JSON on stdin. When
 * the call is a Bash command that launches training (`python train.py`,
 * `accelerate launch`, `torchrun`, `deepspeed`, `sky launch`) and the model,
 * batch and card can be read off the command line or a `.neurarch.yml`
 * `job:` block, the job is sized exactly as size_training_job sizes it. If it
 * will not fit the card it declares, the hook exits 2 with the reason on
 * stderr, which blocks the call and hands the reason to the agent. Every other
 * case exits 0 and prints nothing.
 *
 * It fails open, always. A hook that blocks a launch it could not size, or
 * that crashes the agent's turn, is worse than no hook: so a command it does
 * not recognise, a model it cannot load, a card it cannot parse, a payload it
 * cannot read, or any exception at all is exit 0 and silence. It also stays
 * out of the way of jobs the estimate does not model (gradient checkpointing,
 * LoRA / PEFT, 4- and 8-bit loading, CPU offload): the number is a full
 * fine-tune's peak, and quoting it at an adapter run would refuse a job that
 * fits. `NEURARCH_ALLOW_OOM=1` anywhere in the command lets a launch through
 * that the estimate refuses, because the estimate is a fit check and is
 * sometimes wrong at the edge of a card.
 *
 * No memory logic here: `sizeJob` is the plan bundle's `buildPlan`.
 */
import { readFile, stat } from 'node:fs/promises';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import type { ModelArchitecture } from './lib/types.js';
import { sizeJob, parseGpu, type Attention } from './lib/sizeJob.js';
import { parseYamlSubset, MAX_SEARCH_DEPTH } from './lib/policyFile.js';
import { loadModelCached } from './models.js';
import { loadHFModel } from './sources.js';
import { convertHFConfigToModel } from './vendor/engine.bundle.mjs';
import { looksLikeHFConfig } from './vendor/plan.bundle.mjs';

export interface HookIO {
  out: (s: string) => void;
  err: (s: string) => void;
}

/** What a launch line says about the job. Every field is optional: the job block fills the rest. */
export interface LaunchSpec {
  launcher: 'python' | 'accelerate' | 'torchrun' | 'deepspeed' | 'sky';
  model?: string;
  /** A global batch (`--batch`, `--batch_size`, `--train_batch_size`). */
  batch?: number;
  /** A per-process batch (`--per_device_train_batch_size`), multiplied by the process count. */
  perDeviceBatch?: number;
  /** Processes, from `--nproc_per_node`, `--num_processes` or `--num_gpus`. */
  processes?: number;
  seqLen?: number;
  gpu?: string;
  attention?: Attention;
  /** A flag that means the full fine-tune peak does not describe this job. */
  unmodelled?: string;
}

/** The `job:` block of a .neurarch.yml. */
export interface JobBlock {
  model?: string;
  gpu?: string;
  batch?: number;
  seq_len?: number;
  attention?: Attention;
}

/** A shell line split into words, quotes respected; `&&`, `;`, `|` become their own words. */
export function shellWords(cmd: string): string[] {
  const out: string[] = [];
  let cur = '';
  let quote: '"' | "'" | null = null;
  let has = false;
  for (let i = 0; i < cmd.length; i++) {
    const ch = cmd[i];
    if (quote) {
      if (ch === quote) quote = null;
      else if (ch === '\\' && quote === '"' && i + 1 < cmd.length) cur += cmd[++i];
      else cur += ch;
      continue;
    }
    if (ch === '"' || ch === "'") { quote = ch; has = true; continue; }
    if (ch === '\\' && i + 1 < cmd.length) { cur += cmd[++i]; has = true; continue; }
    if (/\s/.test(ch)) { if (has || cur) out.push(cur); cur = ''; has = false; continue; }
    if (ch === ';' || ch === '|' || ch === '&') {
      if (has || cur) out.push(cur);
      cur = ''; has = false;
      let op = ch;
      while (i + 1 < cmd.length && (cmd[i + 1] === '&' || cmd[i + 1] === '|')) op += cmd[++i];
      out.push(op);
      continue;
    }
    cur += ch;
    has = true;
  }
  if (has || cur) out.push(cur);
  return out;
}

const SEPARATORS = new Set([';', '&&', '||', '|', '&']);

const BATCH_FLAGS = ['--batch', '--batch_size', '--batch-size', '--train_batch_size', '--train-batch-size', '--global_batch_size', '--global-batch-size'];
const PER_DEVICE_FLAGS = ['--per_device_train_batch_size', '--per-device-train-batch-size', '--micro_batch_size', '--micro-batch-size'];
const SEQ_FLAGS = ['--max_seq_length', '--max-seq-length', '--max_seq_len', '--seq_len', '--seq-len', '--max_length', '--max-length', '--block_size', '--block-size', '--sequence_len', '--context_length'];
const MODEL_FLAGS = ['--model', '--model_name_or_path', '--model-name-or-path', '--model_name', '--model-name', '--model_path', '--model-path', '--model_id', '--base_model'];
const PROC_FLAGS = ['--nproc_per_node', '--nproc-per-node', '--num_processes', '--num-processes', '--num_gpus', '--num-gpus'];
const GPU_FLAGS = ['--gpu', '--gpus'];
const ATTN_FLAGS = ['--attn_implementation', '--attn-implementation', '--attention'];
/** Flags under which a full fine-tune's peak is the wrong number to refuse with. */
const UNMODELLED = /^--(gradient[_-]checkpointing|use[_-]peft|peft|lora[\w-]*|use[_-]lora|qlora|load[_-]in[_-](4|8)bit|quantization|bits|offload[\w-]*|cpu[_-]offload|zero[_-]?3?[_-]offload[\w-]*|activation[_-]checkpointing)$/;

const positiveInt = (s: string | undefined): number | undefined => {
  if (s == null || !/^\d+$/.test(s)) return undefined;
  const n = Number(s);
  return n >= 1 ? n : undefined;
};

/** `sky launch --gpus A100-80GB:8` and `--gpus H100:8` into the card syntax parseGpu reads. */
function skyGpu(v: string): string {
  const m = /^([^:]+)(?::(\d+))?$/.exec(v.trim());
  if (!m) return v;
  const name = m[1].replace(/gb$/i, '');
  return m[2] && m[2] !== '1' ? `${m[2]}x${name}` : name;
}

/**
 * The first training launch in a Bash command, or null when there is none.
 * Recognised: `python[3] [-m mod] <something train something>.py`,
 * `accelerate launch`, `torchrun`, `python -m torch.distributed.run`,
 * `deepspeed`, `sky launch`.
 */
export function parseLaunch(command: string): LaunchSpec | null {
  const w = shellWords(command);
  for (let i = 0; i < w.length; i++) {
    const t = w[i];
    const base = t.split('/').pop() ?? t;
    let launcher: LaunchSpec['launcher'] | null = null;
    let argsFrom = i + 1;
    if (base === 'accelerate' && w[i + 1] === 'launch') { launcher = 'accelerate'; argsFrom = i + 2; }
    else if (base === 'torchrun') launcher = 'torchrun';
    else if (base === 'deepspeed') launcher = 'deepspeed';
    else if (base === 'sky' && w[i + 1] === 'launch') { launcher = 'sky'; argsFrom = i + 2; }
    else if (/^python(\d(\.\d+)?)?$/.test(base)) {
      if (w[i + 1] === '-m' && /^torch\.distributed\.(run|launch)$/.test(w[i + 2] ?? '')) { launcher = 'torchrun'; argsFrom = i + 3; }
      else {
        // The script: the first word after the interpreter that is not an interpreter flag.
        // `-c` runs a string, not a script, so it is never a launch.
        let j = i + 1;
        while (j < w.length && /^-[A-Za-z]$/.test(w[j]) && w[j] !== '-m' && w[j] !== '-c') j++;
        const isModule = w[j] === '-m';
        const script = isModule ? w[j + 1] ?? '' : w[j] === '-c' ? '' : w[j] ?? '';
        const named = /train|finetune|fine_tune|sft|pretrain/i.test(script.split('/').pop() ?? '');
        if (named && (isModule || /\.py$/.test(script))) { launcher = 'python'; argsFrom = j + (isModule ? 2 : 1); }
      }
    }
    if (!launcher) continue;
    let end = argsFrom;
    while (end < w.length && !SEPARATORS.has(w[end])) end++;
    return { launcher, ...readFlags(w.slice(argsFrom, end), launcher) };
  }
  return null;
}

function readFlags(args: string[], launcher: LaunchSpec['launcher']): Omit<LaunchSpec, 'launcher'> {
  const spec: Omit<LaunchSpec, 'launcher'> = {};
  for (let k = 0; k < args.length; k++) {
    const a = args[k];
    if (!a.startsWith('--')) continue;
    const eq = a.indexOf('=');
    const name = eq === -1 ? a : a.slice(0, eq);
    const next = args[k + 1];
    const value = eq === -1 ? (next != null && !next.startsWith('--') ? next : undefined) : a.slice(eq + 1);
    if (UNMODELLED.test(name) && value !== 'false' && value !== 'False' && value !== '0') {
      spec.unmodelled ??= name;
      continue;
    }
    if (BATCH_FLAGS.includes(name)) spec.batch = positiveInt(value) ?? spec.batch;
    else if (PER_DEVICE_FLAGS.includes(name)) spec.perDeviceBatch = positiveInt(value) ?? spec.perDeviceBatch;
    else if (SEQ_FLAGS.includes(name)) spec.seqLen = positiveInt(value) ?? spec.seqLen;
    else if (MODEL_FLAGS.includes(name) && value) spec.model = value;
    else if (PROC_FLAGS.includes(name)) spec.processes = positiveInt(value) ?? spec.processes;
    else if (GPU_FLAGS.includes(name) && value) spec.gpu = launcher === 'sky' ? skyGpu(value) : value;
    else if (ATTN_FLAGS.includes(name) && value) {
      const v = value.toLowerCase();
      spec.attention = v === 'eager' ? 'eager' : /flash/.test(v) ? 'flash' : v === 'sdpa' ? 'sdpa' : spec.attention;
    }
  }
  return spec;
}

/** The nearest .neurarch.yml or .neurarch.yaml at or above `dir`, as a parsed object, or null. */
export async function findJobBlock(dir: string): Promise<{ path: string; job: JobBlock } | null> {
  let d = resolve(dir);
  for (let depth = 0; depth <= MAX_SEARCH_DEPTH; depth++) {
    for (const name of ['.neurarch.yml', '.neurarch.yaml']) {
      const p = join(d, name);
      const st = await stat(p).catch(() => null);
      if (!st?.isFile()) continue;
      const cfg = parseYamlSubset(await readFile(p, 'utf-8'));
      const raw = cfg.job;
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
      const j = raw as Record<string, unknown>;
      const int = (v: unknown) => (typeof v === 'number' && Number.isInteger(v) && v >= 1 ? v : typeof v === 'string' ? positiveInt(v) : undefined);
      const str = (v: unknown) => (typeof v === 'string' && v.trim() ? v.trim() : typeof v === 'number' ? String(v) : undefined);
      const attn = str(j.attention);
      return {
        path: p,
        job: {
          model: str(j.model),
          gpu: str(j.gpu),
          batch: int(j.batch ?? j.batch_size),
          seq_len: int(j.seq_len),
          attention: attn === 'eager' || attn === 'flash' || attn === 'sdpa' ? attn : undefined,
        },
      };
    }
    const up = dirname(d);
    if (up === d) break;
    d = up;
  }
  return null;
}

/**
 * A model spec off a command line or a job block, as a graph. A local
 * directory holding a config.json (a checkpoint) is read through the engine;
 * a path, zoo:<id> or hf:<org/name> goes through the server's own loader; a
 * bare org/name that is not a path on disk is a Hugging Face id.
 */
export async function loadJobModel(spec: string, cwd: string): Promise<ModelArchitecture> {
  const s = spec.trim();
  if (/^(zoo|hf):/i.test(s)) {
    if (/^hf:/i.test(s)) return (await loadHFModel(s.slice(3).trim())).model;
    return loadModelCached(s);
  }
  const p = isAbsolute(s) ? s : resolve(cwd, s);
  const st = await stat(p).catch(() => null);
  if (st?.isDirectory()) {
    const cfg = JSON.parse(await readFile(join(p, 'config.json'), 'utf-8')) as unknown;
    if (!looksLikeHFConfig(cfg)) throw new Error(`${p}/config.json is not a Hugging Face config`);
    const name = p.split('/').pop() || 'model';
    const { components, connections } = convertHFConfigToModel(name, cfg as never);
    return { id: `dir-${name}`, name, components, connections } as ModelArchitecture;
  }
  if (st?.isFile()) return loadModelCached(p);
  if (/^[\w.-]+\/[\w.-]+$/.test(s)) return (await loadHFModel(s)).model;
  throw new Error(`no model at ${s}`);
}

export interface HookDecision {
  block: boolean;
  /** The text for stderr when blocking; empty otherwise. */
  reason: string;
}

const NO = (): HookDecision => ({ block: false, reason: '' });

/** The decision for one hook payload. Never throws: any failure is "do not block". */
export async function decide(payload: unknown): Promise<HookDecision> {
  try {
    if (!payload || typeof payload !== 'object') return NO();
    const p = payload as { tool_name?: unknown; tool_input?: { command?: unknown }; cwd?: unknown };
    if (p.tool_name !== 'Bash') return NO();
    const command = p.tool_input?.command;
    if (typeof command !== 'string' || !command.trim()) return NO();
    if (/\bNEURARCH_ALLOW_OOM=1\b/.test(command)) return NO();
    const launch = parseLaunch(command);
    if (!launch || launch.unmodelled) return NO();
    const cwd = typeof p.cwd === 'string' && p.cwd ? p.cwd : process.cwd();
    const block = await findJobBlock(cwd).catch(() => null);
    const job = block?.job ?? {};

    const modelSpec = launch.model ?? job.model;
    let card = launch.gpu ?? job.gpu;
    const processes = launch.processes;
    let batch = launch.batch;
    if (batch == null && launch.perDeviceBatch != null) batch = launch.perDeviceBatch * (processes ?? 1);
    batch ??= job.batch;
    if (!modelSpec || !card || !batch) return NO();

    // A bare card name with a process count is that many of it.
    const parsed = parseGpu(card);
    if ('error' in parsed) return NO();
    if (processes && processes > 1 && parsed.count === 1) card = `${processes}x${parsed.card.id}`;

    const model = await loadJobModel(modelSpec, cwd);
    const seqLen = launch.seqLen ?? job.seq_len;
    const attention = launch.attention ?? job.attention;
    const r = sizeJob(model, { batchSize: batch, gpu: card, seqLen, attention });
    if (!r.sized || r.fits !== false || !r.memory) return NO();

    const label = r.job.gpu ?? card;
    const one = r.multiCard == null;
    const largest = one && r.maxBatch ? r.maxBatch[parsed.card.id] : r.multiCard?.sharded.maxGlobalBatch ?? null;
    const lines = [
      `neurarch: this training job will not fit ${label}.`,
      `  ${r.verdict}.`,
      `  peak ~${r.memory.peakGb} GB (${r.memory.needGb} GB with headroom), card ${label}${one ? ` (${parsed.card.gb} GB)` : ''}.`,
      largest != null
        ? `  largest batch that fits ${label}: ${largest}${r.job.seqLen ? ` at seq ${r.job.seqLen}` : ''}.`
        : `  no batch fits ${label}: weights and optimizer state alone are ${r.memory.weightsAndOptimizerGb} GB.`,
      `  smallest card that holds it at batch ${batch}: ${r.smallestFitGpu ?? (r.twoCardSplit ? `two ${r.twoCardSplit.gpu}` : 'none listed')}.`,
      `  job read from: ${[`the command (${launch.launcher})`, block ? block.path : null].filter(Boolean).join(' and ')}; model ${modelSpec}, batch ${batch}${r.job.seqLen ? `, seq ${r.job.seqLen}` : ''}, attention ${r.job.attention}.`,
      '  A calibrated estimate, not a measurement: at the edge of a card it is a fit check. If you are sure it fits, prefix the command with NEURARCH_ALLOW_OOM=1.',
    ];
    return { block: true, reason: lines.join('\n') + '\n' };
  } catch {
    return NO();
  }
}

/** Read stdin, decide, exit. 2 blocks the Bash call in Claude Code; 0 lets it run. */
export async function runHookCommand(stdinText: string, io: HookIO): Promise<number> {
  let payload: unknown;
  try { payload = JSON.parse(stdinText); } catch { return 0; }
  const d = await decide(payload);
  if (!d.block) return 0;
  io.err(d.reason);
  return 2;
}

export async function readStdin(): Promise<string> {
  if (process.stdin.isTTY) return '';
  const chunks: Buffer[] = [];
  for await (const c of process.stdin) chunks.push(c as Buffer);
  return Buffer.concat(chunks).toString('utf-8');
}
