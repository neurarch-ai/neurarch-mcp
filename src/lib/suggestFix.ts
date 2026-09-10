/**
 * suggest_fix's types and one re-export.
 *
 * The implementation used to live here. It is now `src/utils/suggestFix.ts` in
 * the Neurarch repo and arrives through the vendored engine bundle, for the
 * same reason `lintModelGraph` does: the VS Code extension's lightbulb and
 * `POST /api/v1/fix` offer the same edits, and three copies of "the exact fix
 * for head-dim-divisibility" is three answers as soon as one of them is
 * touched. The rules, the tests and the fixture live beside the engine now.
 *
 * Refresh with `npm run check:mcp-vendor` in the Neurarch checkout.
 */
import type { ModelArchitecture } from './types.js';
import type { EngineFinding } from '../vendor/engine.bundle.mjs';
import { suggestFixes as vendored } from '../vendor/engine.bundle.mjs';

export interface Fix {
  rule: string;
  layer?: string;
  confidence: 'exact' | 'proposal';
  summary: string;
  /** Unified diff against the source file. Empty when the fix is words only. */
  diff: string;
  /** The whole file with just this fix applied (what an editor replaces the buffer with). */
  newText: string;
  /** 1-based source lines the diff touches. */
  lines: number[];
  /** What the agent still has to do by hand, when the diff is not the whole fix. */
  followUp?: string;
}

export interface SuggestFixResult {
  path: string;
  fixes: Fix[];
  notFixable: Array<{ rule: string; layer?: string; reason: string }>;
  /** The whole file after every exact fix, so an agent can write it in one go. */
  patchedSource?: string;
}

export function suggestFixes(
  model: ModelArchitecture,
  findings: EngineFinding[],
  source: string,
  path: string,
): SuggestFixResult {
  return vendored(model as never, findings as never, source, path) as SuggestFixResult;
}
