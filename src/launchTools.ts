/**
 * Two checks on how a model is about to be run, before anything is loaded:
 *
 *   check_parallel_layout  is this TP / PP / DP / CP / EP split of a training
 *                          job legal, balanced, and worth running?
 *   check_serving_config   will this vLLM / SGLang / llama.cpp launch start,
 *                          hold what it was asked to hold, and serve the model
 *                          the way it was published?
 *
 * No logic here. The checks are the vendored plan bundle's
 * `checkParallelLayout` and `checkServeConfig` (the app's
 * src/utils/parallelLayout.ts and serveConfigCheck.ts, neurarch-ai/neurarch
 * #319, #322, #323, #325), the same functions behind `npx neurarch layout`,
 * `npx neurarch serve-check` and the `layout` / `serve` fields of
 * POST /api/v1/check, and the inputs are read by the bundle's own
 * `readLayout` / `readServeSpec`, so a key this server accepts means what it
 * means there. This file adds where the model and its Hugging Face files come
 * from (the same --hf rule every tool follows) and a one-line verdict.
 *
 * The layout report's times are PREDICTED from the graph: matmul FLOPs at an
 * MFU anchor plus memory-bound bytes at the card's bandwidth. Nothing was
 * timed. The result says so in `predicted.label` and again in `basis`, and the
 * verdict carries the word, so a number lifted out of it keeps its status.
 */
import type { ModelArchitecture } from './lib/types.js';
import type { ToolDef } from './tools.js';
import { loadHFModel } from './sources.js';
import { resolveServeSide, hfServeSide, type ResolvedServeSide } from './lib/serveSide.js';
import {
  readLayout, checkParallelLayout, readServeSpec, checkServeConfig,
  type LayoutReport, type ServeReport, type ServeEngine,
} from './vendor/plan.bundle.mjs';

const pct = (f: number) => `${(f * 100).toFixed(1)}%`;
const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? '' : 's'}`;

/** A model named by hf_id, else the one the server (or model_path) handed over. */
async function layoutSubject(hfId: string | undefined, model: ModelArchitecture | undefined): Promise<{ model: ModelArchitecture; source: string }> {
  if (hfId) {
    const id = hfId.replace(/^hf:/i, '').trim();
    const loaded = await loadHFModel(id);
    return { model: loaded.model, source: `hf:${id} (${loaded.configSource === 'fallback' ? 'a family template, not the real config' : 'config.json'})` };
  }
  if (!model) {
    throw new Error('check_parallel_layout: name a model. Pass model_path (a path, zoo:<id> or hf:<org/name>), model_source, or hf_id.');
  }
  return { model, source: 'model' };
}

// ── check_parallel_layout ────────────────────────────────────────────────────

/** The layout keys this tool forwards to readLayout; everything else in args is the tool's own. */
const LAYOUT_KEYS = [
  'tp', 'pp', 'dp', 'cp', 'ep', 'micro_batch', 'global_batch', 'grad_accum', 'seq_len', 'gpu', 'schedule',
  'virtual_stages', 'layers_per_stage', 'zero_stage', 'activation_checkpointing', 'seq_len_histogram',
] as const;

type LayoutArgs = Partial<Record<(typeof LAYOUT_KEYS)[number], unknown>> & { hf_id?: string };

export function layoutVerdict(r: LayoutReport): string {
  const blocks = r.findings.filter(f => f.severity === 'block');
  if (blocks.length) {
    return `will not run: ${blocks[0].title}${blocks.length > 1 ? ` (and ${plural(blocks.length - 1, 'more blocking finding')})` : ''}`;
  }
  const res = r.resolved;
  const parts = [`legal on ${res.worldSize} x ${res.gpu ?? 'an unknown card'}`];
  const im = r.imbalance;
  if (im) {
    parts.push(`predicted ${pct(im.estStepSlowdown)} slower per step than a balanced split (slowest stage ${im.slowestStage})`);
    if (im.rebalanced) parts.push(`split ${im.rebalanced.layersPerStage.join(',')} is predicted ${pct(im.rebalanced.estStepSpeedup)} faster`);
  }
  if (r.bubble) parts.push(`bubble ${pct(r.bubble.fraction)}`);
  const memory = r.stages.filter(s => s.fitsGpu === 'does_not_fit').map(s => s.index);
  if (memory.length) parts.push(`stage${memory.length > 1 ? 's' : ''} ${memory.join(', ')} will not fit the card`);
  const warns = r.findings.filter(f => f.severity === 'warn').length;
  if (warns) parts.push(plural(warns, 'warning'));
  return parts.join('; ');
}

export const checkParallelLayoutTool: ToolDef = {
  name: 'check_parallel_layout',
  description:
    'Call this BEFORE launching a multi-GPU training job split by tensor, pipeline, data, context or expert parallelism '
    + '(Megatron, NeMo, DeepSpeed, torchtitan). It answers three questions from the model graph and the layout, before the '
    + 'job takes a reservation. Legal: does tp x pp x dp x cp multiply out, does TP divide the attention heads, the KV heads, '
    + 'the hidden and FFN widths, does EP divide the experts, does layers_per_stage sum to the blocks, does the batch '
    + 'arithmetic (micro_batch x grad_accum x dp = global_batch) close; a failure here crashes or refuses to start. Balanced: '
    + 'a per-stage table (blocks, what is glued on: embedding, LM head and loss; params; FLOPs per token; predicted ms per '
    + 'microbatch; saved activations; per-GPU peak and its calibrated upper bound; fits / borderline / does_not_fit on the card), '
    + 'the predicted step slowdown against a perfectly balanced split, from a 1F1B schedule simulated op by op (a port of '
    + 'Primus\'s simulator) at this layout\'s microbatch count, and the contiguous split that minimises it. '
    + 'Worth it: the pipeline bubble, and with seq_len_histogram ([{len, count}] or "len,count" CSV rows) the expected '
    + 'slowdown from uneven sequence lengths across DP ranks. The last pipeline stage carrying the LM head and the loss is the '
    + 'usual straggler (ByteDance, OSDI 2025: the last stage causes most of the slowdown in 39.3% of straggling jobs). '
    + 'Every time is PREDICTED, not measured: matmul time from a per-card efficiency table fitted on public GEMM '
    + 'measurements plus memory-bound bytes against bandwidth; on 2 x H100 the predicted gain of a split landed within 1.5 '
    + 'points of measured and against vTrain\'s published step times the error is 9.0%, while absolute step time is less '
    + 'reliable than the gain. The result labels them under `predicted` and in `basis`. Required: micro_batch and seq_len; '
    + 'tp, pp, dp default to 1; gpu takes an id or prefix (h100, h200, b200, a100-80g, l40s, l4; default h100-sxm); '
    + 'zero_stage 0-3; schedule 1f1b (default) or interleaved with virtual_stages. The model is the server\'s, or '
    + 'model_path (a .py, a .neurarch.json, zoo:<id>, hf:<org/name>), or hf_id. Offline; an hf: model needs --hf the first time.',
  modelOptional: true,
  inputSchema: {
    type: 'object',
    properties: {
      tp: { type: 'integer' },
      pp: { type: 'integer' },
      dp: { type: 'integer' },
      cp: { type: 'integer' },
      ep: { type: 'integer' },
      micro_batch: { type: 'integer' },
      global_batch: { type: 'integer' },
      grad_accum: { type: 'integer' },
      seq_len: { type: 'integer' },
      gpu: { type: 'string' },
      schedule: { type: 'string', enum: ['1f1b', 'interleaved'] },
      virtual_stages: { type: 'integer' },
      layers_per_stage: { type: 'array', items: { type: 'integer' } },
      zero_stage: { type: 'integer', enum: [0, 1, 2, 3] },
      activation_checkpointing: { type: 'boolean' },
      seq_len_histogram: { type: ['array', 'string'] },
    },
    required: ['micro_batch', 'seq_len'],
    additionalProperties: false,
  },
  handler: async (args: LayoutArgs, model: ModelArchitecture | undefined) => {
    const raw: Record<string, unknown> = {};
    for (const k of LAYOUT_KEYS) if (args[k] != null) raw[k] = args[k];
    // Refused before any model is loaded, so a typo never hides behind a fetch.
    const layout = readLayout(raw);
    if ('error' in layout) throw new Error(`check_parallel_layout: ${layout.error.replace(/`layout\./g, '`')}`);
    const subject = await layoutSubject(args.hf_id, model);
    const r = checkParallelLayout(subject.model, layout);
    const im = r.imbalance;
    return {
      verdict: layoutVerdict(r),
      legal: !r.findings.some(f => f.severity === 'block'),
      predicted: {
        label: 'Predicted from the graph, not measured. See basis.',
        stepSlowdown: im ? im.estStepSlowdown : null,
        slowestStage: im ? im.slowestStage : null,
        slowestOverMean: im ? im.maxOverMean : null,
        stepMs: im ? { current: im.step.currentMs, balanced: im.step.balancedMs, recommended: im.rebalanced?.stepMs ?? null } : null,
        microbatches: im ? im.step.microbatches : null,
        formula: im ? im.step.formula : null,
        recommendedSplit: im?.rebalanced?.layersPerStage ?? null,
        recommendedSpeedup: im?.rebalanced?.estStepSpeedup ?? null,
        bubble: r.bubble ? r.bubble.fraction : null,
        dpSeqImbalance: r.dpSeqImbalance ?? null,
      },
      findings: r.findings,
      stages: r.stages,
      resolved: r.resolved,
      recommendations: r.recommendations,
      checked: r.checked,
      skipped: r.skipped,
      basis: r.basis,
      model: subject.model.name ?? null,
      source: subject.source,
    };
  },
};

// ── check_serving_config ─────────────────────────────────────────────────────

export function serveVerdict(r: ServeReport): string {
  const blocks = r.findings.filter(f => f.severity === 'block');
  if (blocks.length) {
    return `blocked: ${blocks[0].title}${blocks.length > 1 ? ` (and ${plural(blocks.length - 1, 'more blocking finding')})` : ''}`;
  }
  const parts = ['no blocking finding in the checks that ran'];
  if (r.memory) parts.push(`at most ${Math.floor(r.memory.maxConcurrentFullLength)} concurrent full-length sequences (an upper bound)`);
  const warns = r.findings.filter(f => f.severity === 'warn').length;
  if (warns) parts.push(plural(warns, 'warning'));
  return parts.join('; ');
}

interface ServeArgsIn {
  hf_id?: string;
  engine?: string;
  gpu?: string;
  gpus?: number;
  command?: string;
  args?: Record<string, unknown>;
  against?: string;
}

export const checkServingConfigTool: ToolDef = {
  name: 'check_serving_config',
  description:
    'Call this BEFORE launching or changing a vLLM, SGLang or llama.cpp server for a model (a new checkpoint, a new card, a '
    + 'new --max-model-len or quantization). Pass the launch command as you would run it (command), or already-parsed engine '
    + 'arguments (args, camelCase: tp, maxModelLen, maxNumSeqs, gpuMemoryUtilization, quantization, kvCacheDtype, ...), plus '
    + 'the card (gpu: h100, a100-80g, l40s, l4, ...) and the GPU count (gpus). It reads config.json, tokenizer_config.json, '
    + 'chat_template.jinja and generation_config.json and returns findings in six areas: parallel (TP divides the attention '
    + 'and KV heads, per-rank FFN / expert width against a block-quantized checkpoint, TP x PP x DP against the GPUs, expert '
    + 'parallel), context (the served length against the derived maximum with rope_scaling, learned positions), memory '
    + '(weights per GPU plus KV per token at the cache dtype against card x gpu_memory_utilization, and how many '
    + 'full-length sequences fit concurrently, an upper bound because the engine\'s own reserve is not counted), quant '
    + '(dtype and method against the card\'s compute capability), template and sampler (template present or overridden, '
    + 'known reference mismatches, sampler defaults against generation_config.json, pad == eos, tool and reasoning '
    + 'parsers), and with against (the model this deployment replaces) which config keys changed and what that does to '
    + 'memory, context and the template. Each finding is block (refuses to start or serves wrongly), warn or info, with a '
    + 'fix; `checked` and `skipped` say which checks ran. The model is hf_id, else the model the command names, else '
    + 'model_path. Derived from files and flags, nothing is loaded or run; an hf: model needs --hf the first time.',
  modelOptional: true,
  inputSchema: {
    type: 'object',
    properties: {
      command: { type: 'string', description: 'The launch line.' },
      args: { type: 'object', description: 'Parsed flags (camelCase).' },
      engine: { type: 'string', enum: ['vllm', 'sglang', 'llamacpp'] },
      gpu: { type: 'string' },
      gpus: { type: 'integer' },
      against: { type: 'string', description: 'Model it replaces.' },
    },
    additionalProperties: false,
  },
  handler: async (a: ServeArgsIn, model: ModelArchitecture | undefined) => {
    if (a.command == null && a.args == null) {
      throw new Error('check_serving_config: pass command (the launch line) or args (parsed engine flags).');
    }
    const spec = readServeSpec({ engine: a.engine, command: a.command, args: a.args, gpu: a.gpu, gpus: a.gpus });
    if ('error' in spec) throw new Error(`check_serving_config: ${spec.error.replace(/`serve\./g, '`')}`);

    // hf_id, then the model the command names, then the server's own (or model_path).
    const caveats: string[] = [];
    const named = a.hf_id ? `hf:${a.hf_id.replace(/^hf:/i, '').trim()}` : typeof spec.args.model === 'string' && spec.args.model.trim() ? spec.args.model : null;
    let resolved: ResolvedServeSide | null = null;
    if (named) {
      try {
        resolved = await resolveServeSide(named);
      } catch (e) {
        // A command that names a .gguf or a path on another machine: the
        // server's model, if there is one, is still worth checking against.
        if (a.hf_id || !model) throw new Error(`check_serving_config: ${(e as Error).message}. Pass hf_id.`);
        caveats.push(`The command names ${named}, which could not be read (${(e as Error).message}); checked against the server's model instead.`);
      }
    }
    if (!resolved) {
      if (!model) throw new Error('check_serving_config: name a model. Pass hf_id, a command that names one, or model_path.');
      const hf = /^hf-([\w.-]+\/[\w.-]+)$/.exec(model.id ?? '');
      resolved = hf ? await hfServeSide(hf[1]) : { side: { model, config: null, modelId: model.name }, source: 'model', filesRead: false, exact: false };
    }
    if (!resolved.side.config) {
      caveats.push('No config.json for this model, so the checks that read it (heads, context, quantization, rope scaling) had the graph alone: pass hf_id or a checkpoint directory for the published files.');
    } else if (!resolved.filesRead) {
      caveats.push('tokenizer_config.json, chat_template.jinja and generation_config.json were not read (not cached, and the server was not started with --hf), so the template and sampler checks did not run.');
    }
    const side = resolved.side;
    const source = resolved.source;
    const against = a.against ? (await resolveServeSide(a.against)).side : undefined;

    const r = checkServeConfig({
      ...side,
      engine: spec.engine as ServeEngine,
      args: spec.args,
      gpu: spec.gpu,
      gpus: spec.gpus,
      against,
    });
    return {
      verdict: serveVerdict(r),
      blocked: r.findings.some(f => f.severity === 'block'),
      engine: r.engine,
      findings: r.findings,
      memory: r.memory,
      resolved: r.resolved,
      change: r.change,
      checked: r.checked,
      skipped: r.skipped,
      ...(spec.ignored.length ? { ignoredFlags: spec.ignored } : {}),
      caveats,
      basis: r.basis,
      source,
    };
  },
};

export const LAUNCH_TOOLS: ToolDef[] = [checkParallelLayoutTool, checkServingConfigTool];
