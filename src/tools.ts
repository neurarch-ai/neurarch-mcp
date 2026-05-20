import type { MLComponent, ModelArchitecture } from './lib/types.js';
import { analyzeImpact, resolveTargets, resolveByPattern } from './lib/modelImpact.js';
import { estimateLayerParams, fmtParams } from './lib/paramEstimator.js';
import { estimateLayerFlops, fmtFlops, fmtBytes } from './lib/flopsEstimator.js';
import { renderMermaid } from './mermaid.js';

export interface ToolContext {
  modelPath: string;
}

export interface ToolDef {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  handler: (args: any, model: ModelArchitecture, ctx: ToolContext) => unknown | Promise<unknown>;
}

// ── get_model_summary ────────────────────────────────────────────────────────
const getModelSummary: ToolDef = {
  name: 'get_model_summary',
  description: 'Return a one-shot overview of the Neurarch model: layer count, connection count, total parameters, dominant layer types, and the input/output shape. Use this first to orient yourself before drilling into specific layers.',
  inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  handler: (_args, model) => {
    const totalParams = model.components.reduce(
      (s, c) => s + estimateLayerParams(c.type, c.params, c.inputShape ?? []),
      0,
    );
    const typeCounts: Record<string, number> = {};
    for (const c of model.components) typeCounts[c.type] = (typeCounts[c.type] ?? 0) + 1;
    const dominantTypes = Object.entries(typeCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([type, count]) => ({ type, count }));
    const inputLayer = model.components.find(c => c.type === 'input');
    const outputLayer = model.components.find(c => c.type === 'output');
    return {
      name: model.name,
      layerCount: model.components.length,
      connectionCount: model.connections.length,
      groupCount: model.groups?.length ?? 0,
      totalParameters: totalParams,
      totalParametersFormatted: fmtParams(totalParams),
      inputShape: inputLayer?.outputShape ?? inputLayer?.inputShape ?? null,
      outputShape: outputLayer?.inputShape ?? outputLayer?.outputShape ?? null,
      dominantTypes,
    };
  },
};

// ── get_layer ────────────────────────────────────────────────────────────────
const getLayer: ToolDef = {
  name: 'get_layer',
  description: 'Return the full definition of a single layer (params, shapes, notes, connections). Looks up by exact name first, then by id, then by case-insensitive substring. Returns null when no match.',
  inputSchema: {
    type: 'object',
    properties: {
      name: { type: 'string', description: 'Layer name or id. Substring match if no exact hit.' },
    },
    required: ['name'],
    additionalProperties: false,
  },
  handler: ({ name }: { name: string }, model) => {
    const { resolved } = resolveTargets(model, [name]);
    const comp = resolved[0];
    if (!comp) return null;
    const params = estimateLayerParams(comp.type, comp.params, comp.inputShape ?? []);
    return {
      id: comp.id,
      name: comp.name,
      type: comp.type,
      inputShape: comp.inputShape ?? null,
      outputShape: comp.outputShape ?? null,
      params: comp.params,
      paramCount: params,
      paramCountFormatted: fmtParams(params),
      notes: comp.notes ?? null,
      scope: comp.scope ?? null,
      upstream: model.connections.filter(c => c.to === comp.id).map(c => c.from),
      downstream: model.connections.filter(c => c.from === comp.id).map(c => c.to),
    };
  },
};

// ── find_layers ──────────────────────────────────────────────────────────────
const findLayers: ToolDef = {
  name: 'find_layers',
  description: 'Search layers by type and/or name regex. Use this to answer "where are all the convolutions?" or "find every layer matching ^encoder_". Returns names + types — call get_layer for details.',
  inputSchema: {
    type: 'object',
    properties: {
      type:         { type: 'string', description: 'Exact component type (e.g. "conv2d", "linear", "multiHeadAttention").' },
      namePattern:  { type: 'string', description: 'Regex source matched against layer name.' },
      limit:        { type: 'number', description: 'Max results (default 100).' },
    },
    additionalProperties: false,
  },
  handler: ({ type, namePattern, limit }: { type?: string; namePattern?: string; limit?: number }, model) => {
    let matches: MLComponent[] = model.components;
    if (type) matches = matches.filter(c => c.type === type);
    if (namePattern) {
      try {
        const re = new RegExp(namePattern);
        matches = matches.filter(c => re.test(c.name));
      } catch (e) {
        return { error: `Invalid regex: ${(e as Error).message}` };
      }
    }
    const cap = typeof limit === 'number' && limit > 0 ? limit : 100;
    return {
      count: matches.length,
      truncated: matches.length > cap,
      layers: matches.slice(0, cap).map(c => ({
        name: c.name,
        type: c.type,
        outputShape: c.outputShape ?? null,
      })),
    };
  },
};

// ── layer_impact ─────────────────────────────────────────────────────────────
const layerImpact: ToolDef = {
  name: 'layer_impact',
  description: 'Compute the blast radius of changing a layer (or a regex-matched set): every upstream and downstream layer reachable through connections, flagged for shape sensitivity and weight rebuild requirements. Use this BEFORE recommending a destructive edit so you can warn the user about cascading effects. Mirrors the GitNexus impact tool.',
  inputSchema: {
    type: 'object',
    properties: {
      names:        { type: 'array', items: { type: 'string' }, description: 'Layer names or ids to use as targets.' },
      namePattern:  { type: 'string', description: 'Alternative to names — regex source matching layer names.' },
      maxDistance:  { type: 'number', description: 'Limit BFS depth. Omit for full reachability.' },
      includeUpstream: { type: 'boolean', description: 'Default true. Set false to only return downstream.' },
    },
    additionalProperties: false,
  },
  handler: (
    { names, namePattern, maxDistance, includeUpstream }: { names?: string[]; namePattern?: string; maxDistance?: number; includeUpstream?: boolean },
    model,
  ) => {
    let targets: MLComponent[] = [];
    if (Array.isArray(names) && names.length) {
      const { resolved, unresolved } = resolveTargets(model, names);
      targets = resolved;
      if (unresolved.length) {
        return { error: `Unresolved names: ${unresolved.join(', ')}` };
      }
    } else if (namePattern) {
      targets = resolveByPattern(model, namePattern);
      if (!targets.length) return { error: `No layers match /${namePattern}/` };
    } else {
      return { error: 'Provide either "names" or "namePattern".' };
    }
    const report = analyzeImpact(model, targets, { maxDistance, includeUpstream });
    return {
      targets:    report.targets.map(t => ({ name: t.name, type: t.type })),
      downstream: report.downstream.map(n => ({ name: n.name, type: n.type, distance: n.distance, shapeSensitive: n.shapeSensitive })),
      upstream:   report.upstream.map(n => ({ name: n.name, type: n.type, distance: n.distance })),
      totals:     report.totals,
      summary:    report.summary,
    };
  },
};

// ── param_count_by_block ─────────────────────────────────────────────────────
const paramCountByBlock: ToolDef = {
  name: 'param_count_by_block',
  description: 'Group parameter counts by block (named group), scope (dotted module path), or layer type. Answers "where do my parameters live?" — useful for finding bloated subnets.',
  inputSchema: {
    type: 'object',
    properties: {
      groupBy: { type: 'string', enum: ['block', 'scope', 'type'], description: 'Default "block" (collapsed groups). Falls back to scope when no groups exist.' },
    },
    additionalProperties: false,
  },
  handler: ({ groupBy }: { groupBy?: 'block' | 'scope' | 'type' }, model) => {
    const mode = groupBy ?? 'block';
    return bucketize(model, mode, (c) => estimateLayerParams(c.type, c.params, c.inputShape ?? []), fmtParams);
  },
};

// ── flops_by_block ───────────────────────────────────────────────────────────
const flopsByBlock: ToolDef = {
  name: 'flops_by_block',
  description: 'Group MAC counts (FLOPs ÷ 2) by block, scope, or type. Answers "where is compute spent?" — pairs with param_count_by_block to find parameter-light but compute-heavy regions (attention) versus the inverse (large MLPs).',
  inputSchema: {
    type: 'object',
    properties: {
      groupBy: { type: 'string', enum: ['block', 'scope', 'type'] },
    },
    additionalProperties: false,
  },
  handler: ({ groupBy }: { groupBy?: 'block' | 'scope' | 'type' }, model) => {
    const mode = groupBy ?? 'block';
    return bucketize(
      model,
      mode,
      (c) => estimateLayerFlops(c.type, c.params, c.inputShape ?? [], c.outputShape ?? []),
      fmtFlops,
    );
  },
};

// ── mermaid_diagram ──────────────────────────────────────────────────────────
const mermaidDiagram: ToolDef = {
  name: 'mermaid_diagram',
  description: 'Render the model as Mermaid flowchart syntax (top-down). The caller wraps the result in ```mermaid``` fences. Use sparingly — it gets noisy past ~30 layers; pair with namePattern via find_layers to give the user a clearer mental model.',
  inputSchema: {
    type: 'object',
    properties: {
      maxLayers: { type: 'number', description: 'Truncate after N layers (default 60). Returns a note when truncated.' },
    },
    additionalProperties: false,
  },
  handler: ({ maxLayers }: { maxLayers?: number }, model) => {
    const cap = typeof maxLayers === 'number' && maxLayers > 0 ? maxLayers : 60;
    if (model.components.length <= cap) {
      return { mermaid: renderMermaid(model), truncated: false };
    }
    const truncated: ModelArchitecture = {
      ...model,
      components: model.components.slice(0, cap),
      connections: model.connections.filter(c => {
        const a = model.components.findIndex(x => x.id === c.from);
        const b = model.components.findIndex(x => x.id === c.to);
        return a < cap && b < cap;
      }),
    };
    return {
      mermaid: renderMermaid(truncated),
      truncated: true,
      shown: cap,
      total: model.components.length,
    };
  },
};

// ── list_blocks ──────────────────────────────────────────────────────────────
const listBlocks: ToolDef = {
  name: 'list_blocks',
  description: 'List the model\'s collapsed groups ("blocks"): name, member count, params, FLOPs. Returns scope-derived blocks (dotted module path prefixes) when no explicit groups exist. Use this to talk about the architecture at a high level instead of layer-by-layer.',
  inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  handler: (_args, model) => {
    const groups = model.groups ?? [];
    if (groups.length) {
      return {
        source: 'groups',
        blocks: groups.map(g => {
          const members = model.components.filter(c => g.componentIds.includes(c.id));
          const params  = members.reduce((s, c) => s + estimateLayerParams(c.type, c.params, c.inputShape ?? []), 0);
          const flops   = members.reduce((s, c) => s + estimateLayerFlops(c.type, c.params, c.inputShape ?? [], c.outputShape ?? []), 0);
          return {
            name: g.name,
            id: g.id,
            memberCount: members.length,
            members: members.map(m => m.name),
            paramCount: params,
            paramCountFormatted: fmtParams(params),
            flops,
            flopsFormatted: fmtFlops(flops),
            collapsed: g.collapsed,
          };
        }),
      };
    }
    const bucket = bucketize(
      model,
      'scope',
      (c) => estimateLayerParams(c.type, c.params, c.inputShape ?? []),
      fmtParams,
    );
    return { ...bucket, source: 'scope-derived' };
  },
};

// ── Shared bucketing helper ──────────────────────────────────────────────────
function bucketize(
  model: ModelArchitecture,
  mode: 'block' | 'scope' | 'type',
  metric: (c: MLComponent) => number,
  fmt: (n: number) => string,
): { source: string; buckets: Array<{ key: string; layerCount: number; value: number; valueFormatted: string }>; total: number; totalFormatted: string } {
  const buckets = new Map<string, { value: number; count: number }>();
  const groups = model.groups ?? [];

  function bucketKeyFor(c: MLComponent): string {
    if (mode === 'type') return c.type;
    if (mode === 'block') {
      if (groups.length) {
        const g = groups.find(g => g.componentIds.includes(c.id));
        if (g) return g.name;
      }
      // fall through to scope
    }
    if (c.scope) {
      const top = c.scope.split('.').slice(0, 2).join('.');
      return top || '(root)';
    }
    return '(ungrouped)';
  }

  for (const c of model.components) {
    const key = bucketKeyFor(c);
    const v = metric(c);
    const b = buckets.get(key) ?? { value: 0, count: 0 };
    b.value += v;
    b.count += 1;
    buckets.set(key, b);
  }
  const total = [...buckets.values()].reduce((s, b) => s + b.value, 0);
  const arr = [...buckets.entries()]
    .map(([key, b]) => ({ key, layerCount: b.count, value: b.value, valueFormatted: fmt(b.value) }))
    .sort((a, b) => b.value - a.value);
  return {
    source: mode === 'type' ? 'type' : groups.length ? 'groups' : 'scope-derived',
    buckets: arr,
    total,
    totalFormatted: fmt(total),
  };
}

export const TOOLS: ToolDef[] = [
  getModelSummary,
  getLayer,
  findLayers,
  layerImpact,
  paramCountByBlock,
  flopsByBlock,
  mermaidDiagram,
  listBlocks,
];

// Suppress unused-export TS warning when not destructured elsewhere
export { fmtBytes };
