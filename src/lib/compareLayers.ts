/**
 * Structural diff of two layers.
 *
 * Answers "are these two layers the same, and if not, what differs?" — the
 * question an agent asks before deciding whether two blocks can share an
 * implementation, or whether an edit to one should mirror to the other. The
 * comparison is symmetric: `a` and `b` are interchangeable except in the
 * labelling of which side has which value.
 */
import type { MLComponent, ModelArchitecture } from './types.js';
import { resolveTargets } from './modelImpact.js';
import { estimateLayerParams, fmtParams } from './paramEstimator.js';

export interface ComparedLayer {
  name: string;
  type: string;
  paramCount: number;
  inputShape: number[] | null;
  outputShape: number[] | null;
  scope: string | null;
  augmentations: string[];
}

export interface ParamValueDiff {
  key: string;
  a: unknown;
  b: unknown;
}

export interface LayerComparison {
  a: ComparedLayer;
  b: ComparedLayer;
  sameType: boolean;
  /** b.paramCount - a.paramCount. */
  paramDelta: number;
  paramDeltaFormatted: string;
  inputShapeMatch: boolean;
  outputShapeMatch: boolean;
  /** Param keys present on a but not b, and vice-versa. */
  paramKeysOnlyInA: string[];
  paramKeysOnlyInB: string[];
  /** Keys present on both but with different values. */
  paramValueDiffs: ParamValueDiff[];
  /** True when type, params, and both shapes are identical. */
  identical: boolean;
}

const shapesEqual = (x?: number[], y?: number[]): boolean => {
  if (!x || !y) return !x && !y;
  return x.length === y.length && x.every((v, i) => v === y[i]);
};

function view(c: MLComponent): ComparedLayer {
  return {
    name: c.name,
    type: c.type,
    paramCount: estimateLayerParams(c.type, c.params, c.inputShape ?? []),
    inputShape: c.inputShape ?? null,
    outputShape: c.outputShape ?? null,
    scope: c.scope ?? null,
    augmentations: c.augmentations ?? [],
  };
}

/** Compare two layers by name or id. Returns null when either does not resolve. */
export function compareLayers(model: ModelArchitecture, aName: string, bName: string): LayerComparison | null {
  const { resolved: ra } = resolveTargets(model, [aName]);
  const { resolved: rb } = resolveTargets(model, [bName]);
  const ca = ra[0];
  const cb = rb[0];
  if (!ca || !cb) return null;

  const pa = ca.params ?? {};
  const pb = cb.params ?? {};
  const keysA = Object.keys(pa);
  const keysB = Object.keys(pb);
  const paramKeysOnlyInA = keysA.filter(k => !(k in pb));
  const paramKeysOnlyInB = keysB.filter(k => !(k in pa));
  const paramValueDiffs: ParamValueDiff[] = keysA
    .filter(k => k in pb)
    .filter(k => JSON.stringify(pa[k]) !== JSON.stringify(pb[k]))
    .map(k => ({ key: k, a: pa[k], b: pb[k] }));

  const a = view(ca);
  const b = view(cb);
  const sameType = a.type === b.type;
  const inputShapeMatch = shapesEqual(ca.inputShape, cb.inputShape);
  const outputShapeMatch = shapesEqual(ca.outputShape, cb.outputShape);
  const paramsIdentical =
    paramKeysOnlyInA.length === 0 && paramKeysOnlyInB.length === 0 && paramValueDiffs.length === 0;

  const delta = b.paramCount - a.paramCount;
  return {
    a,
    b,
    sameType,
    paramDelta: delta,
    paramDeltaFormatted: delta === 0 ? '0' : `${delta > 0 ? '+' : '-'}${fmtParams(Math.abs(delta))}`,
    inputShapeMatch,
    outputShapeMatch,
    paramKeysOnlyInA,
    paramKeysOnlyInB,
    paramValueDiffs,
    identical: sameType && paramsIdentical && inputShapeMatch && outputShapeMatch,
  };
}
