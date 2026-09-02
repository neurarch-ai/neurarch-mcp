/**
 * rank_designs: order k candidate graphs for an agent deciding which one to
 * spend GPU time on. The same contract as POST /api/v1/rank, computed here.
 *
 * What it will and will not say is the whole point, and it is the library's
 * decision, not this file's (src/vendor/verifier.bundle.mjs carries
 * rankCandidates from the main repo): a candidate with a pre-flight blocker
 * ranks last because it will not forward-pass, which is the one claim with a
 * measurement behind it. Two legal candidates are ordered only by rules whose
 * provenance is a trained outcome, and when nothing separates them the result
 * says so and `recommended` is null. Parameter count and cost come back per
 * candidate for the caller to break ties on its own budget; they never order
 * on their own. `tieBreak: 'cost' | 'params'` lets the caller state that rule,
 * and it acts only inside a measured tie among legal candidates: the result
 * echoes it, `measuredRank` keeps the verifier's own answer, and every reason
 * it decided says it was the caller's budget, not a measurement.
 *
 * `calibration` is inside every result so the ordering cannot be read as a
 * quality prediction by a caller that never saw the README.
 */
import type { ModelArchitecture } from './types.js';
import { lintModelGraph } from '../vendor/engine.bundle.mjs';
import {
  checkDesign, rankCandidates, signalsFromCheck, normalizeGraphForVerification, OUTCOME_RULE_IDS,
  type RankResult, type CandidateSignals, type TieBreak,
} from '../vendor/verifier.bundle.mjs';
import { MAX_COMPONENTS } from './checkDesign.js';

export const MAX_CANDIDATES = 32;

export interface NamedCandidate { id: string; model: ModelArchitecture }

export interface RankDesignsResult extends RankResult {
  /** Per candidate: the pre-flight findings that decided its tier, so the agent can fix rather than discard. */
  details: Record<string, { summary?: string; blocking: string[]; warnings: string[]; lintRules: string[] }>;
}

export function rankDesigns(candidates: NamedCandidate[], opts: { tieBreak?: TieBreak } = {}): RankDesignsResult {
  if (candidates.length === 0) throw new Error('rank_designs needs at least one candidate.');
  if (candidates.length > MAX_CANDIDATES) {
    throw new Error(`rank_designs takes at most ${MAX_CANDIDATES} candidates per call (got ${candidates.length}). Rank in batches.`);
  }
  const seen = new Set<string>();
  const signals: CandidateSignals[] = [];
  const details: RankDesignsResult['details'] = {};
  for (const c of candidates) {
    if (seen.has(c.id)) throw new Error(`Duplicate candidate id "${c.id}".`);
    seen.add(c.id);
    const n = c.model?.components?.length ?? 0;
    if (n > MAX_COMPONENTS) throw new Error(`Candidate "${c.id}" has ${n} layers, over the ${MAX_COMPONENTS} cap.`);
    const graph = normalizeGraphForVerification(c.model);
    const verdict = checkDesign(graph);
    const findings = lintModelGraph(graph);
    const sig = signalsFromCheck(c.id, verdict, findings.map(f => f.rule), OUTCOME_RULE_IDS);
    // Null means the verdict could not be read. Defaulting that to "no
    // blockers" would tell the agent to train a graph nothing verified.
    if (!sig) throw new Error(`Could not verify candidate "${c.id}"; it is not ranked.`);
    signals.push(sig);
    const pre = verdict.findings.filter(f => f.stage === 'preflight');
    details[c.id] = {
      summary: verdict.summary,
      blocking: pre.filter(f => f.severity === 'block').map(f => f.title),
      warnings: pre.filter(f => f.severity === 'warn').map(f => f.title),
      lintRules: [...new Set(findings.map(f => f.rule))],
    };
  }
  return { ...rankCandidates(signals, { tieBreak: opts.tieBreak ?? 'none' }), details };
}
