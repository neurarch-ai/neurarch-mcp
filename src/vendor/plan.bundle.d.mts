/**
 * Types for the vendored plan bundle (plan.bundle.mjs), hand-written because
 * the bundle ships as compiled JavaScript with no declarations.
 *
 * plan.bundle.mjs is a byte-for-byte copy of the Neurarch app's
 * lib/modelPlan.bundle.mjs (built there by `npm run build:plan` and covered by
 * that repository's bundle-freshness gate). It is the code behind
 * POST /api/v1/plan, the neurarch-bot PR comment and `neurarch-trace --plan`,
 * so the memory numbers size_training_job prints are the same numbers those
 * print, computed here offline.
 *
 * Only what this server calls is declared. plan.contract.test.ts asserts the
 * real module against this list.
 */
import type { ModelArchitecture } from '../lib/types.js';

export interface PlanFinding {
  title: string;
  detail?: string;
  fix?: string;
  stage?: string;
}

export interface DeviceSplit { card: number; gb: number; firstLayer: string; lastLayer: string; layers: number }
export interface DeviceMap { splits: DeviceSplit[]; fits: boolean }
export interface TwoCardSplit extends DeviceMap { gpu: string; gbPerCard: number }

export interface ModelPlan {
  version: 1;
  summary: string;
  model: {
    name: string;
    layers: number;
    params: number;
    inputShape: number[] | null;
    outputShape: number[] | null;
    fingerprint: string;
  };
  run: { legal: boolean | null; blockers: PlanFinding[]; warnings: PlanFinding[]; withheld?: string };
  cost: {
    estCostUsd: number | null;
    estTrainSec: number | null;
    refusedRunUsd: number | null;
    fitsGpu: string | null;
    gpuFits: { T4: boolean | null; A100_40: boolean | null; H100_80: boolean | null };
    /** Training peak at the plan's batch, GB. Includes stored attention scores only when `attention` is eager. */
    peakGb: number;
    /** The activation term alone, GB; null when no shape propagated. */
    activationGb: number | null;
    smallestFitGpu: string | null;
    maxBatch: { T4: number | null; A100_40: number | null; H100_80: number | null };
    twoCardSplit: TwoCardSplit | null;
    /** The attention mode the peak was computed for (the bundle defaults to sdpa). */
    attention: PlanAttention;
    /** GB of stored attention scores at the plan's batch; counted in peakGb only under eager. */
    attentionScoresGb: number | null;
    /** Tokens per sample a token model was sized at, or null for a non-token input. */
    seqLen: number | null;
    assumptions: string;
  };
  [key: string]: unknown;
}

export type PlanAttention = 'sdpa' | 'flash' | 'eager';

export interface BuildPlanOptions {
  base?: unknown;
  /** Default sdpa (no stored scores). Pass eager for a peak that includes them. */
  attention?: PlanAttention;
  /** Tokens per sample for a token model; 'default' applies the bundle's 2048-token default. Omitted: the graph's own input. */
  seqLen?: number | 'default';
  policy?: unknown;
  workload?: { samples: number; source: string };
  /** `gpu` must be one of the bundle's card names, e.g. "A100 (40GB)"; anything else prices on an A10G. */
  run?: { gpu?: string; epochs?: number; batchSize?: number };
  spec?: unknown;
}

export function buildPlan(model: unknown, opts?: BuildPlanOptions): ModelPlan;

/** GB of stored attention scores at `batchSize`: what SDPA or flash attention would not keep. */
export function attentionScoresGbFor(model: ModelArchitecture, batchSize: number): number;

/** The largest batch whose training peak fits `gb` with 1.15x headroom, or null (never 0). */
export function maxBatchFor(
  f: { footprintGb: number; activationPerSampleGb: number | null; activationGbAt?: (batch: number) => number },
  gb: number,
): number | null;

/** A greedy contiguous split of the graph across cards of the given sizes. Counts stored attention scores only under eager (default sdpa). */
export function deviceMapFor(model: unknown, gpus: number[], opts?: { batchSize?: number; attention?: PlanAttention }): DeviceMap;

/** The three fitted factors of the training-peak model. */
export const OPTIMIZER_K: number;
export const OUTPUT_K: number;
export const ATTENTION_K: number;

// ── the out-of-memory diagnosis (diagnose_oom) ──────────────────────────────
// The app's src/utils/oomDiagnosis.ts, oomFacts.ts and inferenceMemory.ts,
// bundled into lib/modelPlan.bundle.mjs through src/utils/planBundle.entry.ts
// (neurarch-ai/neurarch #297). The same functions behind `npx neurarch oom`.

/** The numbers PyTorch prints with a CUDA OOM, in GB. */
export interface OomMessage {
  triedGb: number | null;
  totalGb: number | null;
  freeGb: number | null;
  processInUseGb: number | null;
  allocatedGb: number | null;
  reservedUnallocatedGb: number | null;
  reservedGb: number | null;
  otherProcessesGb: number;
  format: 'torch2' | 'torch1' | 'unknown';
}

/** What the plan knows about the job. Under inference the peak is weights plus KV cache. */
export interface JobFacts {
  mode?: 'train' | 'inference';
  predictedPeakGb?: number | null;
  footprintGb?: number | null;
  activationPerSampleGb?: number | null;
  batch?: number | null;
  step?: number | null;
  vocabSize?: number | null;
  seqLen?: number | null;
}

export type OomCause = 'too-big-for-card' | 'fragmentation' | 'other-process' | 'large-logits' | 'undeclared-memory' | 'grows-during-run' | 'unknown';
export interface OomFix { action: string; because: string; changesTraining: boolean }
export interface OomDiagnosis {
  cause: OomCause;
  summary: string;
  evidence: string[];
  fixes: OomFix[];
  message: OomMessage | null;
  mode: 'train' | 'inference';
}

export type InferencePrecision = 'bf16' | 'fp16' | 'fp32' | 'int8' | 'int4';
export interface OomJobOptions {
  batch?: number | null;
  seqLen?: number | null;
  step?: number | null;
  attention?: PlanAttention;
  /** Size as generation: weights + KV cache. `seqLen` is then the prompt. */
  inference?: { genTokens?: number; precision?: InferencePrecision } | null;
}

export function parseOomMessage(text: string): OomMessage | null;
export function diagnoseOom(text: string, facts?: JobFacts): OomDiagnosis;
/** The text `npx neurarch oom` prints. */
export function renderOomText(d: OomDiagnosis): string;
/** Job facts for one model: a training peak (default batch 32, sdpa) or, with `inference`, weights + KV cache (default batch 1). */
export function oomFactsForModel(model: unknown, opts?: OomJobOptions): JobFacts;
export const DEFAULT_OOM_TRAIN_BATCH: number;
export const DEFAULT_GEN_TOKENS: number;
export const DEFAULT_INFERENCE_BATCH: number;
