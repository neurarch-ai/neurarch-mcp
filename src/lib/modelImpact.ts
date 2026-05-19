/**
 * Model Impact Analyzer
 *
 * Given a target set of layers (by id or name) and a model graph, compute the
 * upstream + downstream "blast radius": every layer whose shape, parameters,
 * or training behaviour can be affected by changing the target set.
 *
 * Used to render a dry-run preview before the agent applies destructive
 * actions (delete / scale / replace), so the user can see "this rewires 8
 * downstream layers" before confirming.
 */
import type { ComponentType, MLComponent, ModelArchitecture } from './types.js';

export interface ImpactNode {
  id: string;
  name: string;
  type: ComponentType;
  distance: number;                       // 1 = direct neighbor of a target
  direction: 'upstream' | 'downstream';
  shapeSensitive: boolean;                // does this layer's behaviour depend on input shape?
}

export interface ImpactReport {
  targets:     MLComponent[];             // resolved target components
  unresolved:  string[];                  // names that did not match any component
  upstream:    ImpactNode[];              // ordered by distance asc
  downstream:  ImpactNode[];              // ordered by distance asc
  totals: {
    targets:           number;
    upstream:          number;
    downstream:        number;
    shapeChanging:     number;            // downstream layers whose output shape will likely change
    paramCarrying:     number;            // downstream layers that hold weights (rebuild required)
    totalAffected:     number;            // targets + upstream + downstream
  };
  /** Markdown summary suitable for an AgentPanel preview card. */
  summary: string;
}

// Layer types that derive their internal shape from input shape — changing an
// upstream layer forces these to be re-instantiated (not just reconnected).
const SHAPE_SENSITIVE_TYPES = new Set<ComponentType>([
  'linear', 'conv2d', 'conv3d', 'conv1d', 'depthwiseConv2d', 'separableConv2d',
  'transposeConv2d', 'audioConv', 'dilatedConv2d', 'depthwiseConv1d',
  'embedding', 'embeddingBag', 'segmentEmbedding', 'positionalEncoding',
  'learnedPositionalEmbedding', 'lmHead',
  'multiHeadAttention', 'groupedQueryAttention', 'causalAttention',
  'selfAttention', 'crossAttention', 'attention', 'transformerBlock',
  'lstm', 'gru', 'rnn', 'bidirectionalLSTM', 'bidirectionalGRU',
  'batchNorm', 'layerNorm', 'instanceNorm', 'groupNorm', 'rmsNorm',
  'mamba', 'conformerBlock',
]);

// Layers that carry trainable weights — semantic indicator for the user.
const PARAM_CARRYING_TYPES = new Set<ComponentType>([
  'linear', 'conv2d', 'conv3d', 'conv1d', 'depthwiseConv2d', 'separableConv2d',
  'transposeConv2d', 'audioConv', 'dilatedConv2d', 'depthwiseConv1d',
  'embedding', 'embeddingBag', 'segmentEmbedding',
  'lstm', 'gru', 'rnn', 'bidirectionalLSTM', 'bidirectionalGRU',
  'multiHeadAttention', 'groupedQueryAttention', 'causalAttention',
  'selfAttention', 'crossAttention', 'attention', 'transformerBlock',
  'feedForward', 'mamba', 'conformerBlock', 'lmHead',
  'batchNorm', 'layerNorm', 'instanceNorm', 'groupNorm', 'rmsNorm',
  'tabnet', 'graphConv', 'graphAttention', 'graphSAGE', 'gcn',
]);

function buildAdjacency(model: ModelArchitecture) {
  const downstream = new Map<string, string[]>();
  const upstream = new Map<string, string[]>();
  for (const c of model.components) {
    downstream.set(c.id, []);
    upstream.set(c.id, []);
  }
  for (const conn of model.connections) {
    downstream.get(conn.from)?.push(conn.to);
    upstream.get(conn.to)?.push(conn.from);
  }
  return { downstream, upstream };
}

function bfs(
  startIds: string[],
  adj: Map<string, string[]>,
  direction: 'upstream' | 'downstream',
  compById: Map<string, MLComponent>,
  excluded: Set<string>,
): ImpactNode[] {
  const visited = new Set<string>(startIds);
  const out: ImpactNode[] = [];
  let frontier = startIds.map(id => ({ id, distance: 0 }));
  while (frontier.length) {
    const next: Array<{ id: string; distance: number }> = [];
    for (const { id, distance } of frontier) {
      for (const neighbor of adj.get(id) ?? []) {
        if (visited.has(neighbor) || excluded.has(neighbor)) continue;
        visited.add(neighbor);
        const comp = compById.get(neighbor);
        if (!comp) continue;
        out.push({
          id: neighbor,
          name: comp.name,
          type: comp.type,
          distance: distance + 1,
          direction,
          shapeSensitive: SHAPE_SENSITIVE_TYPES.has(comp.type),
        });
        next.push({ id: neighbor, distance: distance + 1 });
      }
    }
    frontier = next;
  }
  return out;
}

export interface ImpactOptions {
  /** If true, "upstream" branch is computed too (default: true). */
  includeUpstream?: boolean;
  /** Limit search radius — undefined means full reachability. */
  maxDistance?: number;
}

/**
 * Resolve a list of layer names (with optional regex) and/or ids to component
 * objects. Names are matched exactly first, then by case-insensitive substring
 * if no exact hit. Patterns starting with "/" and ending with "/" are treated
 * as regex sources.
 */
export function resolveTargets(
  model: ModelArchitecture,
  needles: string[],
): { resolved: MLComponent[]; unresolved: string[] } {
  const resolved: MLComponent[] = [];
  const unresolved: string[] = [];
  const seen = new Set<string>();
  for (const needle of needles) {
    const hits: MLComponent[] = [];

    const direct = model.components.find(c => c.id === needle);
    if (direct) hits.push(direct);

    const exact = model.components.filter(c => c.name === needle);
    if (exact.length) hits.push(...exact);

    if (!hits.length && needle.startsWith('/') && needle.length > 2 && needle.endsWith('/')) {
      try {
        const re = new RegExp(needle.slice(1, -1));
        hits.push(...model.components.filter(c => re.test(c.name)));
      } catch { /* fall through */ }
    }

    if (!hits.length) {
      const lower = needle.toLowerCase();
      hits.push(...model.components.filter(c => c.name.toLowerCase().includes(lower)));
    }

    if (!hits.length) {
      unresolved.push(needle);
      continue;
    }
    for (const h of hits) {
      if (seen.has(h.id)) continue;
      seen.add(h.id);
      resolved.push(h);
    }
  }
  return { resolved, unresolved };
}

/** Match every component whose name satisfies a regex source (no slashes). */
export function resolveByPattern(
  model: ModelArchitecture,
  pattern: string,
): MLComponent[] {
  try {
    const re = new RegExp(pattern);
    return model.components.filter(c => re.test(c.name));
  } catch {
    return [];
  }
}

export function analyzeImpact(
  model: ModelArchitecture,
  targets: MLComponent[],
  opts: ImpactOptions = {},
): ImpactReport {
  const { includeUpstream = true, maxDistance } = opts;
  const compById = new Map(model.components.map(c => [c.id, c]));
  const { downstream: dAdj, upstream: uAdj } = buildAdjacency(model);
  const targetIds = new Set(targets.map(t => t.id));

  let downstreamNodes = bfs(targets.map(t => t.id), dAdj, 'downstream', compById, new Set());
  let upstreamNodes = includeUpstream
    ? bfs(targets.map(t => t.id), uAdj, 'upstream', compById, new Set())
    : [];

  if (typeof maxDistance === 'number') {
    downstreamNodes = downstreamNodes.filter(n => n.distance <= maxDistance);
    upstreamNodes   = upstreamNodes.filter(n => n.distance <= maxDistance);
  }

  // De-dup: a node might be reachable from a target both up and down (cycles
  // are rare in NN graphs but possible via residual structure). Prefer
  // downstream as the more user-visible direction.
  const dIds = new Set(downstreamNodes.map(n => n.id));
  upstreamNodes = upstreamNodes.filter(n => !dIds.has(n.id) && !targetIds.has(n.id));

  downstreamNodes.sort((a, b) => a.distance - b.distance);
  upstreamNodes.sort((a, b) => a.distance - b.distance);

  const shapeChanging = downstreamNodes.filter(n => n.shapeSensitive).length;
  const paramCarrying = downstreamNodes.filter(n => PARAM_CARRYING_TYPES.has(n.type)).length;

  return {
    targets,
    unresolved: [],
    upstream: upstreamNodes,
    downstream: downstreamNodes,
    totals: {
      targets:        targets.length,
      upstream:       upstreamNodes.length,
      downstream:     downstreamNodes.length,
      shapeChanging,
      paramCarrying,
      totalAffected:  targets.length + upstreamNodes.length + downstreamNodes.length,
    },
    summary: formatImpactSummary(targets, downstreamNodes, upstreamNodes, shapeChanging, paramCarrying),
  };
}

function formatImpactSummary(
  targets: MLComponent[],
  downstream: ImpactNode[],
  upstream: ImpactNode[],
  shapeChanging: number,
  paramCarrying: number,
): string {
  if (!targets.length) return 'No matching components.';
  const lines: string[] = [];
  const tNames = targets.map(t => t.name).slice(0, 4);
  const tSuffix = targets.length > 4 ? `, +${targets.length - 4} more` : '';
  lines.push(`**Target:** ${tNames.join(', ')}${tSuffix} (${targets.length})`);
  if (downstream.length) {
    lines.push(`**Downstream affected:** ${downstream.length} layer${downstream.length === 1 ? '' : 's'}`);
    if (shapeChanging) lines.push(`  - ${shapeChanging} may need shape recompute`);
    if (paramCarrying) lines.push(`  - ${paramCarrying} carry weights (rebuild needed)`);
  }
  if (upstream.length) {
    lines.push(`**Upstream context:** ${upstream.length} layer${upstream.length === 1 ? '' : 's'}`);
  }
  if (!downstream.length && !upstream.length) {
    lines.push('No connected neighbors. Safe to apply.');
  }
  return lines.join('\n');
}
