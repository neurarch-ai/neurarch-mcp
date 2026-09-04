/**
 * check_design — Neurarch's whole verdict on this graph, computed here.
 *
 * Every other tool in this server answers a question about the model from the
 * file in front of it: how many parameters, what does this layer touch, is
 * there a cycle. Useful, and all of it is inspection. None of it answers the
 * question an agent actually needs answered before it edits someone's
 * architecture, which is "is this design sound, what will it cost to train, and
 * where can it actually run".
 *
 * ── Why this stopped being a network call ──────────────────────────────────
 * It used to POST to /api/v1/check with an API key. Nothing about it needed to.
 * `checkDesign` is a pure, synchronous function of the graph: five pipeline
 * stages, ~13ms, no I/O of any kind. The server around it did three things the
 * key paid for, and none of them were the verdict: it validated the key,
 * capped the graph at 2000 components, and rewrote the error text. The corpus
 * row that would have justified metering is written by a separate endpoint that
 * requires no key at all.
 *
 * So the key bought no compute and captured no measurement. All it did was put
 * a signup between a stranger and the one tool this product is built around,
 * while the other eighteen tools in this server ran offline on first install.
 * The verifier is vendored now (src/vendor/verifier.bundle.mjs), and this file
 * is a function call.
 *
 * ── The network promise ────────────────────────────────────────────────────
 * This server's README promises no network calls unless you opt in. That
 * promise used to carry an exception for this tool. It no longer does: nothing
 * here opens a socket, ever, and the graph never leaves the machine. The single
 * remaining opt-in is NEURARCH_REPORT=1 (corpus reporting), whose payload is
 * structurally incapable of carrying the model.
 *
 * The hosted endpoint still exists, for callers who are not running this
 * server: https://www.neurarch.com/api/v1/check, same code path, same answer.
 */
import { checkDesign as runVerifier, provenanceFor } from '../vendor/verifier.bundle.mjs';
import type { ModelArchitecture } from './types.js';

export interface DesignCheckFinding {
  stage: string;
  severity: 'block' | 'warn';
  /**
   * The check that raised this, where the finding is a check.
   *
   * This is what makes `provenanceForRules` reachable from a verdict: without
   * it an agent holds our sentence about a blocker and no way to look up the
   * measurement behind it. Absent on findings that are not checks, and most of
   * the later stages are not: "no test cases" and "never trained" are states of
   * a workflow rather than verdicts on the graph.
   */
  ruleId?: string;
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

/**
 * The size cap the endpoint enforced, kept rather than dropped.
 *
 * Not for our sake any more (there is no server to protect), but for the
 * agent's: the stages are near-linear in layer count, and a runaway generated
 * graph should come back with a sentence rather than stall the tool call it was
 * made from. The number is the endpoint's, so both surfaces refuse the same
 * inputs.
 */
export const MAX_COMPONENTS = 2000;

export function checkDesign(model: ModelArchitecture): DesignCheck | { error: string } {
  const n = model?.components?.length ?? 0;
  if (n > MAX_COMPONENTS) {
    return {
      error: `check_design: this graph has ${n} layers, over the ${MAX_COMPONENTS} cap. `
        + 'Check a subgraph, or the block-level model rather than the expanded one.',
    };
  }
  try {
    return runVerifier(model) as DesignCheck;
  } catch (e) {
    // A verifier that throws is a bug in the verifier, not a broken model, and
    // saying so is more useful to the agent than a stack trace it cannot act on.
    return {
      error: `check_design could not evaluate this graph: ${(e as Error).message}. `
        + 'The graph loaded, so this is a gap in the verifier rather than a problem with your model. '
        + 'validate_model and lint_model still work on it.',
    };
  }
}

/**
 * The published measurement behind each rule that fired.
 *
 * Bundled rather than fetched, and free rather than metered, for the same
 * reason as the verdict: the measurements are already public at
 * https://www.neurarch.com/docs/structural-checks, and charging for a number
 * you can read on the website is theatre. It is also the give-back that makes
 * NEURARCH_REPORT=1 an exchange rather than a donation.
 *
 * Rules with no published measurement behind them are simply absent, which is
 * the honest shape: an empty object means "nothing here is backed by a number
 * we can show you", not "no rules fired".
 */
export function provenanceForRules(ruleIds: string[]): Record<string, unknown> {
  return provenanceFor(ruleIds) as Record<string, unknown>;
}
