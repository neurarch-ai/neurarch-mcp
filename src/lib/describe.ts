/**
 * One-call architecture orientation.
 *
 * `describe_architecture` exists so an agent can ground itself in a single
 * round-trip instead of chaining get_model_summary -> param_count_by_block ->
 * flops_by_block -> validate_model. It returns a topologically-ordered pipeline,
 * model depth, IO shapes, the heaviest layers by params and by compute, and a
 * validation rollup.
 */
import type { MLComponent, ModelArchitecture } from './types.js';
import { estimateLayerParams, fmtParams } from './paramEstimator.js';
import { estimateLayerFlops, fmtFlops } from './flopsEstimator.js';
import { validateModel } from './validation.js';

export interface Hotspot {
  name: string;
  type: string;
  value: number;
  valueFormatted: string;
  pctOfTotal: number; // 0–100, one decimal
}

export interface ArchitectureDescription {
  name: string;
  layerCount: number;
  connectionCount: number;
  blockCount: number;
  /** Longest directed path length (edge count). null when the graph has a cycle. */
  depth: number | null;
  inputShape: number[] | null;
  outputShape: number[] | null;
  /** Topologically-ordered layer names (capped). */
  pipeline: string[];
  pipelineTruncated: boolean;
  totalParameters: number;
  totalParametersFormatted: string;
  totalMacs: number;
  totalMacsFormatted: string;
  paramHotspots: Hotspot[];
  computeHotspots: Hotspot[];
  validation: { ok: boolean; errors: number; warnings: number };
}

const PIPELINE_CAP = 80;
const HOTSPOT_COUNT = 5;

/**
 * Kahn topological order. Returns { order, acyclic }. When a cycle prevents a
 * full ordering, any leftover nodes are appended in array order and acyclic is
 * false so the caller can decline to report depth.
 */
function topoOrder(model: ModelArchitecture): { order: MLComponent[]; acyclic: boolean } {
  const indeg = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const c of model.components) { indeg.set(c.id, 0); adj.set(c.id, []); }
  for (const e of model.connections) {
    if (!indeg.has(e.from) || !indeg.has(e.to)) continue;
    adj.get(e.from)!.push(e.to);
    indeg.set(e.to, (indeg.get(e.to) ?? 0) + 1);
  }
  const byId = new Map(model.components.map(c => [c.id, c]));
  const queue = model.components.filter(c => (indeg.get(c.id) ?? 0) === 0).map(c => c.id);
  const order: MLComponent[] = [];
  const placed = new Set<string>();
  while (queue.length) {
    const id = queue.shift()!;
    if (placed.has(id)) continue;
    placed.add(id);
    order.push(byId.get(id)!);
    for (const next of adj.get(id) ?? []) {
      indeg.set(next, (indeg.get(next) ?? 0) - 1);
      if ((indeg.get(next) ?? 0) === 0) queue.push(next);
    }
  }
  const acyclic = order.length === model.components.length;
  if (!acyclic) {
    for (const c of model.components) if (!placed.has(c.id)) order.push(c);
  }
  return { order, acyclic };
}

/** Longest path (edge count) over a DAG given a valid topological order. */
function longestPath(model: ModelArchitecture, order: MLComponent[]): number {
  const adj = new Map<string, string[]>();
  for (const c of model.components) adj.set(c.id, []);
  for (const e of model.connections) adj.get(e.from)?.push(e.to);
  const dist = new Map<string, number>();
  for (const c of model.components) dist.set(c.id, 0);
  let max = 0;
  for (const c of order) {
    const d = dist.get(c.id) ?? 0;
    for (const next of adj.get(c.id) ?? []) {
      if ((dist.get(next) ?? 0) < d + 1) {
        dist.set(next, d + 1);
        if (d + 1 > max) max = d + 1;
      }
    }
  }
  return max;
}

function topHotspots(
  model: ModelArchitecture,
  metric: (c: MLComponent) => number,
  fmt: (n: number) => string,
): { total: number; hotspots: Hotspot[] } {
  const scored = model.components
    .map(c => ({ c, v: metric(c) }))
    .filter(x => x.v > 0);
  const total = scored.reduce((s, x) => s + x.v, 0);
  const hotspots = scored
    .sort((a, b) => b.v - a.v)
    .slice(0, HOTSPOT_COUNT)
    .map(({ c, v }) => ({
      name: c.name,
      type: c.type,
      value: v,
      valueFormatted: fmt(v),
      pctOfTotal: total > 0 ? Math.round((v / total) * 1000) / 10 : 0,
    }));
  return { total, hotspots };
}

export function describeArchitecture(model: ModelArchitecture): ArchitectureDescription {
  const { order, acyclic } = topoOrder(model);

  const params = topHotspots(
    model,
    c => estimateLayerParams(c.type, c.params, c.inputShape ?? []),
    fmtParams,
  );
  const compute = topHotspots(
    model,
    c => estimateLayerFlops(c.type, c.params, c.inputShape ?? [], c.outputShape ?? []),
    fmtFlops,
  );

  const inputLayer = model.components.find(c => c.type === 'input');
  const outputLayer = model.components.find(c => c.type === 'output');
  const v = validateModel(model);

  const names = order.map(c => c.name);
  return {
    name: model.name,
    layerCount: model.components.length,
    connectionCount: model.connections.length,
    blockCount: model.groups?.length ?? 0,
    depth: acyclic ? longestPath(model, order) : null,
    inputShape: inputLayer?.outputShape ?? inputLayer?.inputShape ?? null,
    outputShape: outputLayer?.inputShape ?? outputLayer?.outputShape ?? null,
    pipeline: names.slice(0, PIPELINE_CAP),
    pipelineTruncated: names.length > PIPELINE_CAP,
    totalParameters: params.total,
    totalParametersFormatted: fmtParams(params.total),
    totalMacs: compute.total,
    totalMacsFormatted: fmtFlops(compute.total),
    paramHotspots: params.hotspots,
    computeHotspots: compute.hotspots,
    validation: { ok: v.ok, errors: v.totals.errors, warnings: v.totals.warnings },
  };
}
