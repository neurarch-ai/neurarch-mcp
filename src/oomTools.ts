/**
 * diagnose_oom: a CUDA out-of-memory error, read next to the plan's own
 * arithmetic for the job that hit it.
 *
 * The reader of an OOM is increasingly an agent, and an agent that guesses
 * halves the batch until the error goes away. The message already says how
 * much was allocated, how much was reserved and empty, and how big the failing
 * request was; next to the predicted peak that tells "too big for this card"
 * from "the allocator is fragmented" from "another process holds the card" from
 * "something the graph does not declare is holding memory". Only the first of
 * those is a smaller batch.
 *
 * No logic here: `diagnoseOom` and `oomFactsForModel` are the app's
 * (src/utils/oomDiagnosis.ts and oomFacts.ts, the code behind `npx neurarch
 * oom`), vendored in plan.bundle.mjs. This file loads the model the same way
 * size_training_job does and refuses to size a graph sizeJob refuses to size.
 * Offline: an hf_id is read from the config cache when it is there, and needs
 * --hf only for a fetch.
 */
import type { ModelArchitecture } from './lib/types.js';
import type { ToolDef } from './tools.js';
import { untrustworthy, type Attention } from './lib/sizeJob.js';
import { loadHFModel } from './sources.js';
import { diagnoseOom, oomFactsForModel, DEFAULT_GEN_TOKENS, type JobFacts } from './vendor/plan.bundle.mjs';

interface DiagnoseArgs {
  error: string;
  hf_id?: string;
  batch?: number;
  seq_len?: number;
  step?: number;
  attention?: Attention;
  inference?: boolean;
  gen_tokens?: number;
}

const r3 = (n: number | null | undefined) => (n == null ? n : Math.round(n * 1000) / 1000);

function intArg(name: string, v: unknown, min: number): number | undefined {
  if (v == null) return undefined;
  if (!Number.isInteger(v) || (v as number) < min) throw new Error(`diagnose_oom: ${name} must be an integer >= ${min}.`);
  return v as number;
}

export const diagnoseOomTool: ToolDef = {
  name: 'diagnose_oom',
  description:
    'Call this after a CUDA out-of-memory error, before changing batch size. Pass the error text as printed (either PyTorch '
    + 'wording) and, when you can, the model and the job: model_path (a .py, a .neurarch.json from neurarch-trace, zoo:<id>, '
    + 'hf:<org/name>) or hf_id, plus batch, seq_len, the step it failed at, and attention. It reads the allocated, '
    + 'reserved-but-empty, other-process and failing-request numbers out of the message, compares them with the calibrated '
    + 'predicted peak for that job (the same one size_training_job uses), and returns ONE cause: too-big-for-card, '
    + 'fragmentation, other-process, large-logits, undeclared-memory (eval pass, eager attention, a longer sequence than '
    + 'planned), grows-during-run (a leak or one long batch), or unknown. With it: a one-sentence summary, the evidence '
    + 'lines the cause was read from, fixes ordered least invasive first (each flagged changesTraining when it changes what '
    + 'is trained, or the outputs under inference), and the job facts used. A smaller batch is offered only when the job is '
    + 'too big for the card, and then with the gradient accumulation that keeps the effective batch. inference: true sizes '
    + 'generation instead (weights plus KV cache at seq_len prompt tokens plus gen_tokens, default 256, batch default 1). '
    + 'Without a model the message alone is read and the result says what it could not compare. Offline; an hf_id is read '
    + 'from the local config cache, and needs the server started with --hf only when it was never fetched.',
  modelOptional: true,
  inputSchema: {
    type: 'object',
    properties: {
      error: { type: 'string', description: 'The OOM text as printed.' },
      hf_id: { type: 'string', description: 'Hugging Face org/name.' },
      batch: { type: 'integer', minimum: 1 },
      seq_len: { type: 'integer', minimum: 1, description: 'Tokens per sample (the prompt, if inference).' },
      step: { type: 'integer', minimum: 0 },
      attention: { type: 'string', enum: ['sdpa', 'flash', 'eager'] },
      inference: { type: 'boolean', description: 'Generation, not training.' },
      gen_tokens: { type: 'integer', minimum: 0, description: 'Default 256.' },
    },
    required: ['error'],
    additionalProperties: false,
  },
  handler: async (args: DiagnoseArgs, model: ModelArchitecture | undefined) => {
    if (typeof args.error !== 'string' || !args.error.trim()) {
      throw new Error('diagnose_oom: error must be the out-of-memory text, as printed.');
    }
    if (args.attention && !['sdpa', 'flash', 'eager'].includes(args.attention)) {
      throw new Error(`diagnose_oom: attention "${args.attention}" is not one of sdpa, flash, eager.`);
    }
    const batch = intArg('batch', args.batch, 1);
    const seqLen = intArg('seq_len', args.seq_len, 1);
    const step = intArg('step', args.step, 0);
    const genTokens = intArg('gen_tokens', args.gen_tokens, 0);
    const inference = args.inference === true;
    const caveats: string[] = [];
    if (!inference && genTokens != null) caveats.push('gen_tokens was ignored: it applies only with inference: true.');
    if (inference && args.attention) caveats.push('attention was ignored: an inference peak is weights plus KV cache and keeps no attention scores.');

    let subject = model;
    let source = model ? 'model' : 'none';
    if (args.hf_id) {
      const id = args.hf_id.replace(/^hf:/i, '').trim();
      const loaded = await loadHFModel(id);
      subject = loaded.model;
      source = `hf:${id} (${loaded.configSource === 'fallback' ? 'a family template, not the real config' : loaded.configSource === 'cache' ? 'cached config.json' : 'config.json'})`;
    }

    let facts: JobFacts = { ...(inference ? { mode: 'inference' as const } : {}), batch: batch ?? null, step: step ?? null };
    const distrust = subject ? untrustworthy(subject) : null;
    if (!subject) {
      caveats.push('No model, so the message was read alone: pass model_path or hf_id, with batch and seq_len, to compare it with the predicted peak.');
    } else if (distrust) {
      caveats.push(`The model was not sized: ${distrust}`);
      source += ' (not sized)';
    } else {
      facts = oomFactsForModel(subject, {
        batch, seqLen, step, attention: args.attention,
        inference: inference ? { genTokens } : null,
      });
    }

    const d = diagnoseOom(args.error, facts);
    return {
      cause: d.cause,
      summary: d.summary,
      evidence: d.evidence,
      fixes: d.fixes,
      mode: d.mode,
      job: {
        source,
        ...(subject ? { model: subject.name } : {}),
        batch: facts.batch ?? null,
        seqLen: facts.seqLen ?? null,
        step: facts.step ?? null,
        ...(inference ? { genTokens: genTokens ?? DEFAULT_GEN_TOKENS } : { attention: args.attention ?? 'sdpa' }),
        predictedPeakGb: r3(facts.predictedPeakGb) ?? null,
        [inference ? 'weightsGb' : 'weightsAndOptimizerGb']: r3(facts.footprintGb) ?? null,
        [inference ? 'kvCachePerSequenceGb' : 'activationPerSampleGb']: r3(facts.activationPerSampleGb) ?? null,
        vocabSize: facts.vocabSize ?? null,
      },
      message: d.message,
      caveats,
    };
  },
};

export const OOM_TOOLS: ToolDef[] = [diagnoseOomTool];
