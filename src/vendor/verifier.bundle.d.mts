/**
 * Types for the vendored verifier bundle (verifier.bundle.mjs), hand-written
 * because the bundle ships as compiled JavaScript with no declarations.
 * verifier.contract.test.ts asserts the real module against this list.
 */
import type { ModelArchitecture } from '../lib/types.js';

export interface DesignCheckFinding {
  stage: string;
  severity: 'block' | 'warn';
  title: string;
  detail?: string;
  fix?: string;
}

export interface DesignCheck {
  verdict: 'ok' | 'warn' | 'block' | 'ask';
  outcome: 'complete' | 'blocked' | 'needs_input';
  summary: string;
  stoppedAt: string | null;
  findings: DesignCheckFinding[];
  stages: Array<{ stage: string; status: string; headline: string; data: Record<string, unknown> }>;
  decision?: { question: string; because: string; options: Array<{ label: string; value: string; hint?: string }> };
}

/** The measured evidence behind a rule, or absent when there is none. */
export interface RuleProvenance {
  /** One sentence quoting the real numbers. */
  evidence: string;
  /** Public URL where the measurement is documented. */
  source: string;
}

/** Pure and synchronous: the five pipeline stages over the graph, no I/O. */
export function checkDesign(model: ModelArchitecture): DesignCheck;

/** Published measurements for the given rule ids. Rules without one are omitted. */
export function provenanceFor(ruleIds: Iterable<string>): Record<string, RuleProvenance>;

export const RULE_PROVENANCE: Record<string, RuleProvenance>;

/** Rule ids whose provenance is a trained outcome, the only ones that may order two legal designs. */
export const OUTCOME_RULE_IDS: readonly string[];

/** What the ranking is worth, measured. Shipped inside every rank result. */
export const RANK_CALIBRATION: {
  exclusion: { claim: string; evidence: string };
  ordering: {
    claim: string;
    pairwiseAccuracy: number;
    coverage: number;
    sampleSize: number;
    basis: string;
    comparators: Record<string, number>;
  };
  [key: string]: unknown;
};

export interface CandidateSignals {
  id: string;
  blocking: number;
  warnings: number;
  otherStageBlockers?: Array<{ stage: string; title: string }>;
  outcomeFlags: string[];
  params: number | null;
  estCostUsd: number | null;
  fitsGpu: string | null;
  summary?: string;
}

export interface RankedCandidate extends CandidateSignals {
  /** Position after the caller's tieBreak, if any. */
  rank: number;
  /** The rank on measured evidence alone. Equal to `rank` without a tieBreak. */
  measuredRank: number;
  tier: 'legal' | 'blocked';
  tiedWith: number;
  reasons: string[];
}

/** How a caller may break a tie the verifier will not: cheapest first, or smallest first. */
export type TieBreak = 'none' | 'cost' | 'params';
export interface RankOptions { tieBreak?: TieBreak }

export interface RankResult {
  ranked: RankedCandidate[];
  recommended: string | null;
  recommendation: string;
  budget: { candidates: number; blocked: number; legal: number; wouldNotRun: string[]; reclaimed: number };
  /** The caller's rule, echoed, with how many positions it decided rather than a measurement. */
  tieBreak: { rule: TieBreak; decided: number; note: string };
  calibration: typeof RANK_CALIBRATION;
}

/** Extract the ranker's inputs from a verdict. Null when the verdict cannot be read; never a clean default. */
export function signalsFromCheck(
  id: string,
  check: unknown,
  firedRuleIds: Iterable<string>,
  outcomeRuleIds: Iterable<string>,
): CandidateSignals | null;

/** Order candidates: blocked last, outcome rules only, ties stay ties unless the caller's tieBreak says otherwise inside a measured tie. */
export function rankCandidates(candidates: CandidateSignals[], opts?: RankOptions): RankResult;

/** Resolve name-keyed connections to ids and fill missing ids, the way the hosted endpoint does. */
export function normalizeGraphForVerification<T>(graph: T): T;
