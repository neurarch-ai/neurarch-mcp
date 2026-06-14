/**
 * Guarded RegExp construction for user-supplied pattern sources.
 *
 * Patterns arrive from MCP tool arguments (find_layers, layer_impact, the
 * /slash/ form of resolveTargets), i.e. ultimately from an LLM or a human.
 * Two cheap guards keep a hostile or accidental pattern from hanging the
 * server: a hard length cap, and a heuristic reject for nested unbounded
 * quantifiers — the classic catastrophic-backtracking shape like `(a+)+`.
 *
 * This is not a full ReDoS analyzer; it is a pragmatic backstop. Valid layer
 * name patterns (`^encoder\.\d+`, `block_\d`, `attn|mlp`) pass untouched.
 */

export const MAX_PATTERN_LENGTH = 200;

// Nested quantifier on a group, e.g. (x+)+ / (x*)* / (x+)* — the shapes that
// blow up exponentially. Deliberately narrow to avoid false positives.
const CATASTROPHIC = /\([^)]*[+*]\)[+*]/;

/**
 * Compile a user pattern into a RegExp, throwing a clear Error when the source
 * is too long or matches a known catastrophic shape. Throwing (rather than
 * returning null) lets each caller decide how to surface it.
 */
export function compileUserRegExp(source: string, flags?: string): RegExp {
  if (typeof source !== 'string') {
    throw new Error('pattern must be a string');
  }
  if (source.length > MAX_PATTERN_LENGTH) {
    throw new Error(`pattern too long (max ${MAX_PATTERN_LENGTH} chars)`);
  }
  if (CATASTROPHIC.test(source)) {
    throw new Error('pattern rejected: nested unbounded quantifier risks catastrophic backtracking');
  }
  return new RegExp(source, flags);
}

/** Non-throwing variant: returns null instead of throwing. */
export function tryCompileUserRegExp(source: string, flags?: string): RegExp | null {
  try {
    return compileUserRegExp(source, flags);
  } catch {
    return null;
  }
}
