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
    /** Training peak at the plan's batch, GB, with eager attention's stored scores included. */
    peakGb: number;
    /** The activation term alone, GB; null when no shape propagated. */
    activationGb: number | null;
    smallestFitGpu: string | null;
    maxBatch: { T4: number | null; A100_40: number | null; H100_80: number | null };
    twoCardSplit: TwoCardSplit | null;
    assumptions: string;
  };
  [key: string]: unknown;
}

export interface BuildPlanOptions {
  base?: unknown;
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

/** A greedy contiguous split of the graph across cards of the given sizes. Counts stored attention scores. */
export function deviceMapFor(model: unknown, gpus: number[], opts?: { batchSize?: number }): DeviceMap;

/** The three fitted factors of the training-peak model. */
export const OPTIMIZER_K: number;
export const OUTPUT_K: number;
export const ATTENTION_K: number;
