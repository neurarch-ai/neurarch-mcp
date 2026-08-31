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
