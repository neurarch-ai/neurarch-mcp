/**
 * Types for the vendored engine bundle (engine.bundle.mjs), hand-written
 * because the bundle ships as compiled JavaScript with no declarations.
 *
 * Only what this server actually calls is declared. Adding a declaration for
 * something the bundle does not export would compile fine and fail at runtime,
 * so engine.contract.test.ts asserts the real module against this list.
 *
 * ModelArchitecture is deliberately this package's own type rather than the
 * app's: the two are the same shape on the wire (that is the whole point of
 * .neurarch.json), and importing the app's would drag its type graph in here.
 */
import type { ModelArchitecture } from '../lib/types.js';

export type EngineSeverity = 'block' | 'warn' | 'info';

export interface EngineFinding {
  /** Stable rule id, e.g. 'high-dropout' or 'head-dim-divisibility'. */
  rule: string;
  severity: EngineSeverity;
  message: string;
  /** The offending layer, when the rule pins one. Absent for whole-model rules. */
  componentName?: string;
  componentType?: string;
}

/**
 * Reconstruct a model graph from PyTorch source. Returns null when the file has
 * no recognizable model class, or parses to nothing but an input/output stub.
 */
export function graphFromPyTorchSource(code: string, name?: string): ModelArchitecture | null;

/** Run the structural rule set over a graph. */
export function lintModelGraph(model: ModelArchitecture): EngineFinding[];

/** Rule ids in the advisor set, for coverage assertions. */
export const ADVISOR_RULE_IDS: readonly string[];

/** Rule ids in the shape set. */
export const SHAPE_RULE_IDS: readonly string[];
