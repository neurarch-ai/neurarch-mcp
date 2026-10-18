/**
 * `neurarch-mcp hook`: a Claude Code PreToolUse hook that sizes a training
 * launch before the Bash call runs it.
 *
 * Claude Code pipes the pending tool call to the hook as JSON on stdin. When
 * the call is a Bash command that launches training (`python train.py`,
 * `accelerate launch`, `torchrun`, `deepspeed`, `sky launch`) and the model,
 * batch and card can be read off the command line or a `.neurarch.yml`
 * `job:` block, the job is sized exactly as size_training_job sizes it.
 *
 * The `job:` block is the one the neurarch-bot Action reads (its
 * scripts/job.py), with the same semantics: a top-level `job:` and a model
 * entry's own `job:` merged over it key by key; keys gpu, batch, seq_len,
 * grad_accum, attention, epochs, samples, each a value or a `file:key`
 * pointer into a YAML or JSON training config read relative to the
 * repository root (the directory of the .neurarch.yml), dotted keys allowed;
 * unknown keys ignored. `batch` is the per-step batch the memory is sized at,
 * and `grad_accum` only sets the effective batch the reason quotes. One key is
 * the hook's own: `model`, because the bot takes the model from its traced
 * file list and the hook has no trace to read it from. If it
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
 * A vLLM or SGLang launch (`vllm serve`, `python -m vllm.entrypoints...`,
 * `python -m sglang.launch_server`) gets the serving pre-check instead, the
 * same `checkServeConfig` check_serving_config runs, on the model the command
 * names (a checkpoint directory, or a Hugging Face id read under --hf). It
 * blocks only on a `block` finding read off the model's own config.json and
 * the flags: TP that does not divide the heads, a served length past the
 * derived context, a --quantization that contradicts the checkpoint, auto tool
 * choice with no parser. The card is not on a serve command line, so nothing
 * that depends on it (memory, quant support) can block here. A family-template
 * config never blocks, the engine's own override variables
 * (VLLM_ALLOW_LONG_MAX_MODEL_LEN=1, SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1)
 * clear the context finding, and `NEURARCH_ALLOW_SERVE=1` lets the launch
 * through. Fail-open applies unchanged.
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
import { looksLikeHFConfig, parseServeCommand, checkServeConfig } from './vendor/plan.bundle.mjs';
import { resolveServeSide } from './lib/serveSide.js';

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
  /** Gradient accumulation steps (`--gradient_accumulation_steps`), for the effective batch. */
  gradAccum?: number;
  gpu?: string;
  attention?: Attention;
  /** A flag that means the full fine-tune peak does not describe this job. */
  unmodelled?: string;
}

/** The `job:` block of a .neurarch.yml, after `file:key` pointers are read. */
export interface JobBlock {
  /** Hook-only: the bot reads the model from its traced file list. */
  model?: string;
  gpu?: string;
  batch?: number;
  seq_len?: number;
  grad_accum?: number;
  attention?: Attention;
  epochs?: number;
  samples?: number;
}

/** The keys neurarch-bot's scripts/job.py reads, plus `model`. Anything else is ignored. */
export const JOB_KEYS = ['gpu', 'batch', 'seq_len', 'grad_accum', 'attention', 'epochs', 'samples', 'model'] as const;
const NUMERIC_JOB_KEYS = new Set(['batch', 'seq_len', 'grad_accum', 'epochs', 'samples']);
/** The bot's `file:key` pointer: a .yml / .yaml / .json path, a colon, a dotted key. */
const JOB_REF = /^(?<file>[^:\s]+\.(?:ya?ml|json)):(?<key>[A-Za-z0-9_.\-]+)$/;

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
const ACCUM_FLAGS = ['--gradient_accumulation_steps', '--gradient-accumulation-steps', '--grad_accum', '--grad-accum', '--accumulation_steps'];
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
    else if (ACCUM_FLAGS.includes(name)) spec.gradAccum = positiveInt(value) ?? spec.gradAccum;
    else if (PROC_FLAGS.includes(name)) spec.processes = positiveInt(value) ?? spec.processes;
    else if (GPU_FLAGS.includes(name) && value) spec.gpu = launcher === 'sky' ? skyGpu(value) : value;
    else if (ATTN_FLAGS.includes(name) && value) {
      const v = value.toLowerCase();
      spec.attention = v === 'eager' ? 'eager' : /flash/.test(v) ? 'flash' : v === 'sdpa' ? 'sdpa' : spec.attention;
    }
  }
  return spec;
}

const isMap = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

function dig(data: unknown, key: string): unknown {
  let d = data;
  for (const part of key.split('.')) {
    if (!isMap(d) || !(part in d)) throw new Error(`key ${key} not found`);
    d = d[part];
  }
  return d;
}

/** One `file:key` pointer read at `root`, or throws. JSON by extension, else the YAML subset. */
async function readRef(root: string, file: string, key: string): Promise<unknown> {
  const text = await readFile(isAbsolute(file) ? file : join(root, file), 'utf-8');
  return dig(file.endsWith('.json') ? JSON.parse(text) : parseYamlSubset(text), key);
}

/**
 * A raw job block resolved the way neurarch-bot resolves it: pointers read,
 * numeric keys coerced to positive whole numbers. A key that cannot be read
 * or is not a number is dropped and named in `problems`, never guessed.
 */
export async function resolveJob(raw: Record<string, unknown>, root: string): Promise<{ job: JobBlock; sources: Record<string, string>; problems: string[] }> {
  const job: Record<string, unknown> = {};
  const sources: Record<string, string> = {};
  const problems: string[] = [];
  for (const k of JOB_KEYS) {
    if (!(k in raw) || raw[k] == null) continue;
    let v: unknown = raw[k];
    const m = typeof v === 'string' ? JOB_REF.exec(v.trim()) : null;
    if (m?.groups) {
      try {
        v = await readRef(root, m.groups.file, m.groups.key);
        sources[k] = `${m.groups.file}:${m.groups.key}`;
      } catch (e) {
        problems.push(`job.${k} reads ${m.groups.file}:${m.groups.key}: ${(e as Error).message}`);
        continue;
      }
    }
    if (NUMERIC_JOB_KEYS.has(k)) {
      const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN;
      if (!(Number.isInteger(n) && n > 0)) { problems.push(`job.${k} is ${JSON.stringify(v)}, not a positive whole number`); continue; }
      job[k] = n;
    } else if (k === 'attention') {
      const a = String(v).trim().toLowerCase();
      if (a === 'sdpa' || a === 'flash' || a === 'eager') job[k] = a;
      else problems.push(`job.attention is ${JSON.stringify(v)}, not sdpa, flash or eager`);
    } else if (typeof v === 'string' || typeof v === 'number') {
      job[k] = String(v).trim();
    }
  }
  return { job: job as JobBlock, sources, problems };
}

/** The file part of a model reference: `models/doc_lm.py:DocLM` -> `models/doc_lm.py`. */
const fileOf = (ref: string) => ref.replace(/:[A-Za-z_][\w.]*$/, '');

/**
 * Which job block applies, by the bot's merge rule: the top-level `job:`,
 * then one model entry's own `job:` over it, key by key. The entry is the one
 * whose `path` (or hook-only `job.model`) names the model being launched;
 * with no such match, the only entry that has a `job:`, if there is exactly
 * one. Several candidates and no match: the top level alone, rather than a
 * guess.
 */
export function pickJob(cfg: Record<string, unknown>, modelHint: string | undefined, root: string): Record<string, unknown> | null {
  const top = isMap(cfg.job) ? cfg.job : {};
  const entries = (Array.isArray(cfg.models) ? cfg.models : []).filter(isMap).filter(e => isMap(e.job));
  const norm = (x: string) => resolve(root, fileOf(x.trim()));
  let own: Record<string, unknown> | undefined;
  if (modelHint) {
    const hint = /^(zoo|hf):/i.test(modelHint) ? modelHint : norm(modelHint);
    own = entries.find(e => {
      const j = e.job as Record<string, unknown>;
      if (typeof j.model === 'string' && (j.model === modelHint || (!/^(zoo|hf):/i.test(j.model) && norm(j.model) === hint))) return true;
      return typeof e.path === 'string' && norm(e.path) === hint;
    })?.job as Record<string, unknown> | undefined;
  }
  if (!own && entries.length === 1) own = entries[0].job as Record<string, unknown>;
  const merged = { ...top, ...(own ?? {}) };
  return Object.keys(merged).length ? merged : null;
}

/**
 * The nearest .neurarch.yml or .neurarch.yaml at or above `dir`, its job for
 * the model being launched, resolved. Null when there is no file or no job.
 */
export async function findJobBlock(dir: string, modelHint?: string): Promise<{ path: string; root: string; job: JobBlock; sources: Record<string, string>; problems: string[] } | null> {
  let d = resolve(dir);
  for (let depth = 0; depth <= MAX_SEARCH_DEPTH; depth++) {
    for (const name of ['.neurarch.yml', '.neurarch.yaml']) {
      const p = join(d, name);
      const st = await stat(p).catch(() => null);
      if (!st?.isFile()) continue;
      const raw = pickJob(parseYamlSubset(await readFile(p, 'utf-8')), modelHint, d);
      if (!raw) return null;
      return { path: p, root: d, ...(await resolveJob(raw, d)) };
    }
    const up = dirname(d);
    if (up === d) break;
    d = up;
  }
  return null;
}

/**
 * (micro, steps) for a target effective batch under a per-step cap: the
 * bot's `accumulation_split`, so the hook and the PR comment offer the same
 * fix. A divisor is preferred so the effective batch does not move; when the
 * only divisors are far below the cap, the cap and the rounded-up steps.
 */
export function accumulationSplit(target: number, maxBatch: number): [number, number] {
  if (maxBatch >= target) return [target, 1];
  let best = 1;
  for (let d = 1; d <= maxBatch; d++) if (target % d === 0) best = d;
  if (best * 2 >= maxBatch) return [best, target / best];
  return [maxBatch, Math.ceil(target / maxBatch)];
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

/** A vLLM or SGLang server launch: the engine and the words of its command, env prefixes dropped. */
export interface ServeLaunch {
  engine: 'vllm' | 'sglang';
  words: string[];
}

const LAUNCH_PREFIXES = new Set(['env', 'exec', 'nohup', 'time', 'sudo']);

/**
 * The first vLLM or SGLang server launch in a Bash command, or null. Only the
 * command word counts: `echo "vllm serve x"` and `pip install vllm` are not
 * launches. llama.cpp is left alone: its model is a .gguf file, not a config.
 */
export function parseServeLaunch(command: string): ServeLaunch | null {
  const w = shellWords(command);
  let start = 0;
  for (let i = 0; i <= w.length; i++) {
    if (i < w.length && !SEPARATORS.has(w[i])) continue;
    const seg = w.slice(start, i);
    start = i + 1;
    let j = 0;
    while (j < seg.length && (/^[A-Za-z_][A-Za-z0-9_]*=/.test(seg[j]) || LAUNCH_PREFIXES.has(seg[j]))) j++;
    const c = seg.slice(j);
    const base = (c[0] ?? '').split('/').pop() ?? '';
    let engine: ServeLaunch['engine'] | null = null;
    if (base === 'vllm' && c[1] === 'serve') engine = 'vllm';
    else if (base === 'sglang' && c[1] === 'serve') engine = 'sglang';
    else if (/^python(\d(\.\d+)?)?$/.test(base) && c[1] === '-m') {
      if (/^vllm\.entrypoints\./.test(c[2] ?? '')) engine = 'vllm';
      else if (c[2] === 'sglang.launch_server') engine = 'sglang';
    }
    if (engine) return { engine, words: c };
  }
  return null;
}

/** The env variable that lets each engine serve past the derived context. */
const LONG_CONTEXT_OVERRIDE: Record<ServeLaunch['engine'], string> = {
  vllm: 'VLLM_ALLOW_LONG_MAX_MODEL_LEN',
  sglang: 'SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN',
};

/** The serving pre-check for one Bash command. Throws on anything unexpected; decide() catches. */
export async function decideServe(command: string, cwd: string): Promise<HookDecision> {
  if (/\bNEURARCH_ALLOW_SERVE=1\b/.test(command)) return NO();
  const launch = parseServeLaunch(command);
  if (!launch) return NO();
  const parsed = parseServeCommand(launch.words, launch.engine);
  const ref = parsed.args.model;
  if (typeof ref !== 'string' || !ref.trim()) return NO();
  const resolved = await resolveServeSide(ref, cwd);
  // A guessed config (a family template) can be wrong about heads or context.
  if (!resolved.exact) return NO();
  const r = checkServeConfig({ ...resolved.side, engine: launch.engine, args: parsed.args });
  const envVar = LONG_CONTEXT_OVERRIDE[launch.engine];
  const overridden = new RegExp(`\\b${envVar}=(1|true)\\b`, 'i').test(command) || /^(1|true)$/i.test(process.env[envVar] ?? '');
  const blocks = r.findings.filter(f => f.severity === 'block' && !(overridden && f.rule === 'context-over-derived'));
  if (!blocks.length) return NO();
  const engineName = launch.engine === 'vllm' ? 'vLLM' : 'SGLang';
  const lines = [
    `neurarch: this ${engineName} launch has ${blocks.length === 1 ? 'a blocking finding' : `${blocks.length} blocking findings`} for ${ref}.`,
  ];
  for (const f of blocks) {
    lines.push(`  ${f.title}.`);
    if (f.detail) lines.push(`    ${f.detail}`);
    if (f.fix) lines.push(`    fix: ${f.fix}`);
  }
  lines.push(`  Read from ${resolved.source} and the flags, before anything loads; check_serving_config with the card (gpu, gpus) also checks memory, KV concurrency and quantization support. If you are sure, prefix the command with NEURARCH_ALLOW_SERVE=1.`);
  return { block: true, reason: lines.join('\n') + '\n' };
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
    const cwd = typeof p.cwd === 'string' && p.cwd ? p.cwd : process.cwd();
    const launch = parseLaunch(command);
    if (!launch) return await decideServe(command, cwd);
    if (launch.unmodelled) return NO();
    const block = await findJobBlock(cwd, launch.model).catch(() => null);
    const job = block?.job ?? {};

    const modelSpec = launch.model ?? job.model;
    const modelCwd = launch.model ? cwd : block?.root ?? cwd;
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

    const model = await loadJobModel(modelSpec, modelCwd);
    const seqLen = launch.seqLen ?? job.seq_len;
    const attention = launch.attention ?? job.attention;
    const accum = launch.gradAccum ?? job.grad_accum ?? 1;
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
        ? `  largest batch that fits ${label}: ${largest}${r.job.seqLen ? ` at seq ${r.job.seqLen}` : ''}.${accumLine(batch * accum, largest)}`
        : `  no batch fits ${label}: weights and optimizer state alone are ${r.memory.weightsAndOptimizerGb} GB.`,
      `  smallest card that holds it at batch ${batch}: ${r.smallestFitGpu ?? (r.twoCardSplit ? `two ${r.twoCardSplit.gpu}` : 'none listed')}.`,
      `  job read from: ${[`the command (${launch.launcher})`, block ? block.path : null, ...Object.entries(block?.sources ?? {}).map(([k, v]) => `${k} from ${v}`)].filter(Boolean).join(', ')}; model ${modelSpec}, batch ${batch}${accum > 1 ? ` x ${accum} accumulation steps (effective ${batch * accum})` : ''}${r.job.seqLen ? `, seq ${r.job.seqLen}` : ''}, attention ${r.job.attention}.`,
      '  A calibrated estimate, not a measurement: at the edge of a card it is a fit check. If you are sure it fits, prefix the command with NEURARCH_ALLOW_OOM=1.',
    ];
    return { block: true, reason: lines.join('\n') + '\n' };
  } catch {
    return NO();
  }
}

/** " Batch 4 with 8 gradient-accumulation steps keeps the effective batch at 32." or ''. */
function accumLine(target: number, largest: number): string {
  if (!(target > 0) || largest < 1) return '';
  const [micro, steps] = accumulationSplit(target, largest);
  if (steps <= 1) return '';
  const eff = micro * steps;
  return ` Batch ${micro} with ${steps} gradient-accumulation steps keeps the effective batch at ${eff}${eff === target ? '' : ` (the job had ${target})`}.`;
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
