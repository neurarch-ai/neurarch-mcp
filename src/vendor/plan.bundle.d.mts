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
  /** Stable rule id on findings from a catalogued check (edit checks, the adapter match). */
  rule?: string;
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
    /** The conservative peak: peakGb x upperBound.multiplier, at least 1 GB; 0 with nothing to size (neurarch #312). */
    peakGbUpper: number;
    upperBound: UpperBound;
    /** Three answers per card: the upper bound fits, only the point estimate does, or neither. */
    gpuVerdict: { T4: FitVerdict | null; A100_40: FitVerdict | null; H100_80: FitVerdict | null };
    /** The cheapest listed card whose verdict is `fits`. */
    smallestSafeGpu: string | null;
    /** The largest batch the upper bound fits with headroom. */
    maxBatchSafe: { T4: number | null; A100_40: number | null; H100_80: number | null };
  };
  /** Edits that load, run and are wrong, read against the base graph and/or the base config.json. */
  edit: { checked: Array<'graph' | 'config'>; findings: EditFinding[] } | null;
  [key: string]: unknown;
}

export type PlanAttention = 'sdpa' | 'flash' | 'eager';

/** `fits`: the upper bound with 1.15x headroom fits. `borderline`: it does not, the point estimate is inside the card. */
export type FitVerdict = 'fits' | 'borderline' | 'does_not_fit';
export interface UpperBound { multiplier: number; coverage: number; runs: number; multiplier90: number }

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
  /** The changed model's Hugging Face config.json and its base's; compared only as a pair. */
  config?: unknown;
  baseConfig?: unknown;
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

/**
 * The six fitted factors of the training peak (neurarch #306):
 * OPTIMIZER_K x 16 bytes per parameter, plus, per sample in bf16, OUTPUT_K x
 * every layer output, ATTENTION_K x stored attention scores, INTERNAL_K x each
 * layer's weight-matrix intermediates, RECURRENT_K x recurrent gate states and
 * TRANSIENT_K x the largest single tensor.
 */
export const OPTIMIZER_K: number;
export const OUTPUT_K: number;
export const ATTENTION_K: number;
export const INTERNAL_K: number;
export const RECURRENT_K: number;
export const TRANSIENT_K: number;
/** The output term's factor, kept under the old single-factor name. */
export const ACTIVATION_K: number;
/** Headroom every card check uses (1.15). */
export const FIT_HEADROOM: number;
export const DEFAULT_SEQ_LEN: number;
export const DEFAULT_ATTENTION: PlanAttention;

/** What the fit was fitted on and how it does held out; the app's /memory-calibration.html pins it. */
export const MEMORY_CALIBRATION: {
  readonly fittedRuns: number;
  readonly architectures: number;
  readonly heldOutMedianPct: number;
  readonly flagged: number;
  readonly noFit: number;
  readonly afterFitRuns: number;
  readonly afterFitMedianPct: number;
  /** Held-out measured / predicted peak at the 90th and 95th percentile (neurarch #312). */
  readonly upperK90: number;
  readonly upperK95: number;
  /** Held-out runs of 1 GB or more the two multipliers were read from. */
  readonly upperRuns: number;
};
/** The fit in one sentence, written from MEMORY_CALIBRATION. */
export function calibrationSentence(): string;
/** The cheapest listed card a training peak fits with headroom, by the plan's own fit rule. */
export function smallestCardFor(peakGb: number, params: number): string | null;

// ── the conservative peak and the three-way verdict (neurarch #312) ─────────
/** The multiplier the upper bound ships with (MEMORY_CALIBRATION.upperK95), and the held-out share it covers. */
export const UPPER_K: number;
export const UPPER_COVERAGE: number;
/** The upper bound of any job with parameters is at least this many GB. */
export const UPPER_FLOOR_GB: number;
/** peakGb x UPPER_K, never under UPPER_FLOOR_GB; 0 when there is nothing to size. */
export function upperPeakGb(peakGb: number): number;
/** fits / borderline / does_not_fit for one card of `gb`; null when params <= 0. */
export function fitVerdict(peakGb: number, params: number, gb: number): FitVerdict | null;
/** The cheapest listed card whose verdict is `fits`. */
export function smallestSafeCardFor(peakGb: number, params: number): string | null;

// ── does the data fit what the config accepts (neurarch #314) ───────────────
// The app's src/utils/configDataFit.ts. Pure: it reads no dataset, only what
// the caller states, and lists which checks ran.

export interface ModelInputContract {
  vocabSize: number | null;
  maxPositions: number | null;
  positionKind: 'learned' | 'rotary' | 'declared' | null;
  numLabels: number | null;
  imageSize: number | null;
  channels: number | null;
  source: Partial<Record<'vocabSize' | 'maxPositions' | 'numLabels' | 'imageSize' | 'channels', string>>;
}
export interface DataSpec { seqLen?: number; tokenizerSize?: number; numClasses?: number; imageSize?: number; channels?: number }
export interface DataFitFinding {
  rule: 'seq-over-positions' | 'seq-over-context' | 'tokenizer-over-vocab' | 'labels-over-head' | 'labels-under-head' | 'image-size-mismatch' | 'channels-mismatch';
  severity: 'block' | 'warn';
  title: string;
  detail: string;
  fix: string;
}
export interface DataFitReport {
  checked: string[];
  skipped: Array<{ check: string; reason: string }>;
  findings: DataFitFinding[];
  contract: ModelInputContract;
}
/** What the model accepts; the label count only from a config whose `architectures` names a classification head. */
export function inputContractOf(model: unknown, config?: Record<string, unknown> | null): ModelInputContract;
/** Reads `{ seq_len, tokenizer_size, num_classes, image_size, channels }` (snake or camel case); a present field that is not a positive integer is an error. */
export function readDataSpec(raw: unknown): DataSpec | { error: string };
export function checkDataFit(contract: ModelInputContract, data: DataSpec): DataFitReport;

// ── edits that load, run and are wrong (check_edit) ─────────────────────────
// The app's src/utils/editChecks.ts (neurarch-ai/neurarch), bundled through
// modelPlan.ts. Pure and deterministic.

export type EditRuleId =
  | 'vocab-resize-incomplete' | 'weight-tying-lost' | 'attention-config-drift' | 'rope-context-drift'
  | 'lora-target-no-match' | 'lora-target-partial' | 'lora-target-wrapper';
export interface EditFinding {
  /** An edit rule id, or one of two existing shape ids when a config edit breaks divisibility. */
  rule: EditRuleId | 'head-dim-divisibility' | 'gqa-head-divisibility';
  severity: 'block' | 'warn';
  title: string;
  detail: string;
  /** The tensors, config keys or modules the finding is about. */
  subjects: string[];
}
export interface LoraMatch {
  targets: string[] | string;
  matched: Array<{ name: string; type: string }>;
  perTarget: Array<{ target: string; count: number }>;
  wrappers: Array<{ name: string; type: string }>;
  byConvention: boolean;
  findings: EditFinding[];
}
export type LoraRole = 'q_proj' | 'k_proj' | 'v_proj' | 'o_proj' | 'gate' | 'up' | 'down' | 'fc' | 'head' | 'other';
export interface LoraModule {
  name: string; module: string; layer: string; type: string; role: LoraRole;
  inFeatures: number; outFeatures: number; params: number; lora: { r8: number; r16: number };
}
export interface LoraPreset {
  targetModules: string[]; modules: number;
  loraParams: { r8: number; r16: number }; pctOfBase: { r8: number; r16: number };
}
export interface LoraTargets {
  baseParams: number;
  modules: LoraModule[];
  byRole: Partial<Record<LoraRole, { count: number; params: number }>>;
  presets: { attention: LoraPreset; allLinear: LoraPreset };
  uncovered: Array<{ name: string; type: string; params: number }>;
}

export const EDIT_RULE_IDS: readonly EditRuleId[];
/** Public upstream reports of each failure happening (not a frequency). */
export const EDIT_RULE_SEEN_IN: Record<EditRuleId, string[]>;
/** Two graphs: token-indexed tensors left at the old vocabulary, tied heads that stopped being tied. */
export function checkGraphEdit(base: unknown, head: unknown): EditFinding[];
/** Two Hugging Face config.json objects: the same two questions plus keys that must move together. */
export function checkConfigEdit(base: unknown, head: unknown): EditFinding[];
/** A PEFT target_modules matched against the graph the way PEFT matches it. */
export function checkLoraTargets(model: unknown, targets: string[] | string, inventory: LoraTargets): LoraMatch;
/** True when an object reads like a Hugging Face model config. */
export function looksLikeHFConfig(v: unknown): boolean;
export function loraTargetsFor(model: unknown): LoraTargets;
export function renderEditFindingsMarkdown(findings: EditFinding[], heading?: string): string;

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
