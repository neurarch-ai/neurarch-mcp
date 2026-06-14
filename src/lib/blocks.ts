/**
 * Drill into a single block.
 *
 * `list_blocks` enumerates the model's groups; `get_block` zooms into one of
 * them — its member layers with per-layer params/FLOPs, the block totals, and
 * the edges that cross the block boundary (what feeds it, what it feeds). The
 * boundary edges are the useful part: they tell an agent how a block connects
 * to the rest of the graph before it recommends extracting or replacing it.
 *
 * Resolution order: explicit group by name, then group by id, then — when no
 * group matches — every component whose `scope` starts with the given prefix.
 */
import type { MLComponent, ModelArchitecture } from './types.js';
import { estimateLayerParams, fmtParams } from './paramEstimator.js';
import { estimateLayerFlops, fmtFlops } from './flopsEstimator.js';

export interface BlockMember {
  name: string;
  type: string;
  paramCount: number;
  paramCountFormatted: string;
  flops: number;
  flopsFormatted: string;
}

export interface BlockBoundaryEdge {
  from: string;
  to: string;
  label: string | null;
}

export interface BlockReport {
  source: 'group' | 'scope';
  name: string;
  memberCount: number;
  members: BlockMember[];
  paramCount: number;
  paramCountFormatted: string;
  flops: number;
  flopsFormatted: string;
  /** Edges entering the block (source outside, target inside). */
  inputs: BlockBoundaryEdge[];
  /** Edges leaving the block (source inside, target outside). */
  outputs: BlockBoundaryEdge[];
}

const paramsOf = (c: MLComponent) => estimateLayerParams(c.type, c.params, c.inputShape ?? []);
const flopsOf = (c: MLComponent) => estimateLayerFlops(c.type, c.params, c.inputShape ?? [], c.outputShape ?? []);

/** Returns the block report, or null when nothing resolves to `name`. */
export function getBlock(model: ModelArchitecture, name: string): BlockReport | null {
  if (!name) return null;

  const groups = model.groups ?? [];
  const group = groups.find(g => g.name === name) ?? groups.find(g => g.id === name);

  let source: 'group' | 'scope';
  let blockName: string;
  let members: MLComponent[];

  if (group) {
    source = 'group';
    blockName = group.name;
    const ids = new Set(group.componentIds);
    members = model.components.filter(c => ids.has(c.id));
  } else {
    // Scope-prefix fallback: "encoder" matches "encoder.layer.0", "encoder", …
    members = model.components.filter(
      c => c.scope === name || (c.scope?.startsWith(name + '.') ?? false),
    );
    if (!members.length) return null;
    source = 'scope';
    blockName = name;
  }

  const memberIds = new Set(members.map(c => c.id));
  const byId = new Map(model.components.map(c => [c.id, c]));
  const nameOf = (id: string) => byId.get(id)?.name ?? id;

  const inputs: BlockBoundaryEdge[] = [];
  const outputs: BlockBoundaryEdge[] = [];
  for (const e of model.connections) {
    const fromIn = memberIds.has(e.from);
    const toIn = memberIds.has(e.to);
    if (toIn && !fromIn) inputs.push({ from: nameOf(e.from), to: nameOf(e.to), label: e.label ?? null });
    else if (fromIn && !toIn) outputs.push({ from: nameOf(e.from), to: nameOf(e.to), label: e.label ?? null });
  }

  const totalParams = members.reduce((s, c) => s + paramsOf(c), 0);
  const totalFlops = members.reduce((s, c) => s + flopsOf(c), 0);

  return {
    source,
    name: blockName,
    memberCount: members.length,
    members: members.map(c => {
      const params = paramsOf(c);
      const flops = flopsOf(c);
      return {
        name: c.name,
        type: c.type,
        paramCount: params,
        paramCountFormatted: fmtParams(params),
        flops,
        flopsFormatted: fmtFlops(flops),
      };
    }),
    paramCount: totalParams,
    paramCountFormatted: fmtParams(totalParams),
    flops: totalFlops,
    flopsFormatted: fmtFlops(totalFlops),
    inputs,
    outputs,
  };
}
