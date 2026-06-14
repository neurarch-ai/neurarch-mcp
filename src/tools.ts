import type { MLComponent, ModelArchitecture } from './lib/types.js';
import { analyzeImpact, resolveTargets, resolveByPattern } from './lib/modelImpact.js';
import { estimateLayerParams, fmtParams } from './lib/paramEstimator.js';
import { estimateLayerFlops, fmtFlops, fmtBytes } from './lib/flopsEstimator.js';
import { validateModel } from './lib/validation.js';
import { describeArchitecture } from './lib/describe.js';
import { getBlock } from './lib/blocks.js';
import { compileUserRegExp } from './lib/regexGuard.js';
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

// ── describe_architecture ────────────────────────────────────────────────────
const describeArchitectureTool: ToolDef = {
  name: 'describe_architecture',
  description: 'One-call orientation: topologically-ordered layer pipeline, model depth (longest path), input/output shapes, total params and MACs, the top-5 heaviest layers by parameters AND by compute, and a validation rollup. Use this instead of chaining get_model_summary + param_count_by_block + flops_by_block + validate_model — it answers "what is this model and where is the budget" in a single call.',
  inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  handler: (_args, model) => describeArchitecture(model),
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
  description: 'Search layers by type, name regex, scope prefix, and/or applied augmentation, then optionally rank by parameter count. Answers "where are all the convolutions?", "which layers under encoder.layer.3 are frozen?", or "what are the 5 biggest layers?". Returns name, type, scope, param count, and any augmentations — call get_layer for full detail.',
  inputSchema: {
    type: 'object',
    properties: {
      type:         { type: 'string', description: 'Exact component type (e.g. "conv2d", "linear", "multiHeadAttention").' },
      namePattern:  { type: 'string', description: 'Regex source matched against layer name.' },
      scope:        { type: 'string', description: 'Scope prefix, e.g. "encoder.layer.3" (matches that scope and anything nested under it).' },
      augmentation: { type: 'string', description: 'Only layers carrying this augmentation overlay, e.g. "freeze", "quantize_int8", "gradient_checkpoint", "amp".' },
      sortByParams: { type: 'boolean', description: 'When true, rank results by estimated parameter count (largest first) before applying limit.' },
      limit:        { type: 'number', description: 'Max results (default 100).' },
    },
    additionalProperties: false,
  },
  handler: (
    { type, namePattern, scope, augmentation, sortByParams, limit }:
      { type?: string; namePattern?: string; scope?: string; augmentation?: string; sortByParams?: boolean; limit?: number },
    model,
  ) => {
    let matches: MLComponent[] = model.components;
    if (type) matches = matches.filter(c => c.type === type);
    if (scope) matches = matches.filter(c => c.scope === scope || (c.scope?.startsWith(scope + '.') ?? false));
    if (augmentation) matches = matches.filter(c => c.augmentations?.includes(augmentation) ?? false);
    if (namePattern) {
      try {
        const re = compileUserRegExp(namePattern);
        matches = matches.filter(c => re.test(c.name));
      } catch (e) {
        return { error: `Invalid regex: ${(e as Error).message}` };
      }
    }
    const withParams = matches.map(c => ({
      c,
      params: estimateLayerParams(c.type, c.params, c.inputShape ?? []),
    }));
    if (sortByParams) withParams.sort((a, b) => b.params - a.params);
    const cap = typeof limit === 'number' && limit > 0 ? limit : 100;
    return {
      count: withParams.length,
      truncated: withParams.length > cap,
      layers: withParams.slice(0, cap).map(({ c, params }) => ({
        name: c.name,
        type: c.type,
        scope: c.scope ?? null,
        outputShape: c.outputShape ?? null,
        paramCount: params,
        paramCountFormatted: fmtParams(params),
        augmentations: c.augmentations ?? [],
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
    const keptComponents = model.components.slice(0, cap);
    const keptIds = new Set(keptComponents.map(c => c.id));
    const truncated: ModelArchitecture = {
      ...model,
      components: keptComponents,
      connections: model.connections.filter(c => keptIds.has(c.from) && keptIds.has(c.to)),
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

// ── get_block ─────────────────────────────────────────────────────────────────
const getBlockTool: ToolDef = {
  name: 'get_block',
  description: 'Drill into one block (named group, or a scope prefix like "encoder.layer.0"): its member layers with per-layer params and FLOPs, the block totals, and the edges crossing the block boundary — what feeds the block and what it feeds. Use after list_blocks to understand how a block connects to the rest of the graph before recommending extracting, replacing, or freezing it. Returns null when the name resolves to no group or scope.',
  inputSchema: {
    type: 'object',
    properties: {
      name: { type: 'string', description: 'Group name or id, or a scope prefix (e.g. "encoder" matches "encoder.layer.0").' },
    },
    required: ['name'],
    additionalProperties: false,
  },
  handler: ({ name }: { name: string }, model) => getBlock(model, name),
};

// ── validate_model ───────────────────────────────────────────────────────────
const validateModelTool: ToolDef = {
  name: 'validate_model',
  description: 'Run structural invariants over the model: cycles, dangling connection refs, duplicate ids/names, and orphan layers (no upstream / no downstream). Call before recommending a destructive edit so you can flag pre-existing issues separate from your change. Returns a list of findings with severity error|warn.',
  inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  handler: (_args, model) => validateModel(model),
};

// ── find_path ────────────────────────────────────────────────────────────────
const findPath: ToolDef = {
  name: 'find_path',
  description: 'BFS shortest directed path from one layer to another. Answers "does encoder.0.attention reach lm_head?" — returns the ordered list of layer names along the path, or null when unreachable. Useful before recommending an edit to confirm two layers are actually on the same flow.',
  inputSchema: {
    type: 'object',
    properties: {
      from: { type: 'string', description: 'Source layer name or id.' },
      to:   { type: 'string', description: 'Target layer name or id.' },
    },
    required: ['from', 'to'],
    additionalProperties: false,
  },
  handler: ({ from, to }: { from: string; to: string }, model) => {
    const { resolved: fromList, unresolved: fromUnresolved } = resolveTargets(model, [from]);
    if (fromUnresolved.length || !fromList[0]) {
      return { error: `Cannot find "from" layer "${from}".` };
    }
    const { resolved: toList, unresolved: toUnresolved } = resolveTargets(model, [to]);
    if (toUnresolved.length || !toList[0]) {
      return { error: `Cannot find "to" layer "${to}".` };
    }
    const src = fromList[0];
    const dst = toList[0];

    const adj = new Map<string, string[]>();
    for (const c of model.components) adj.set(c.id, []);
    for (const e of model.connections) adj.get(e.from)?.push(e.to);

    const prev = new Map<string, string | null>();
    prev.set(src.id, null);
    const queue: string[] = [src.id];
    while (queue.length) {
      const id = queue.shift()!;
      if (id === dst.id) break;
      for (const n of adj.get(id) ?? []) {
        if (prev.has(n)) continue;
        prev.set(n, id);
        queue.push(n);
      }
    }
    if (!prev.has(dst.id)) {
      return { reachable: false, path: null };
    }
    const ids: string[] = [];
    let cur: string | null = dst.id;
    while (cur !== null) {
      ids.unshift(cur);
      cur = prev.get(cur) ?? null;
    }
    const compById = new Map(model.components.map(c => [c.id, c]));
    return {
      reachable: true,
      length: ids.length - 1,
      path: ids.map(id => {
        const c = compById.get(id)!;
        return { name: c.name, type: c.type };
      }),
    };
  },
};

// ── list_connections ─────────────────────────────────────────────────────────
const listConnections: ToolDef = {
  name: 'list_connections',
  description: 'Return every connection as {from, to, label?}, with optional filters by source or target layer name. Useful when get_layer\'s upstream/downstream is not enough and the agent needs a flat edge list (e.g. for "what are all the residual links?").',
  inputSchema: {
    type: 'object',
    properties: {
      from:  { type: 'string', description: 'Optional source layer name or id filter.' },
      to:    { type: 'string', description: 'Optional target layer name or id filter.' },
      limit: { type: 'number', description: 'Max results (default 200).' },
    },
    additionalProperties: false,
  },
  handler: ({ from, to, limit }: { from?: string; to?: string; limit?: number }, model) => {
    const compById = new Map(model.components.map(c => [c.id, c]));
    let edges = model.connections.map(e => ({
      from: compById.get(e.from)?.name ?? e.from,
      to:   compById.get(e.to)?.name ?? e.to,
      label: e.label ?? null,
    }));
    if (from) {
      const { resolved } = resolveTargets(model, [from]);
      const names = new Set(resolved.map(r => r.name));
      edges = edges.filter(e => names.has(e.from));
    }
    if (to) {
      const { resolved } = resolveTargets(model, [to]);
      const names = new Set(resolved.map(r => r.name));
      edges = edges.filter(e => names.has(e.to));
    }
    const cap = typeof limit === 'number' && limit > 0 ? limit : 200;
    return {
      count: edges.length,
      truncated: edges.length > cap,
      connections: edges.slice(0, cap),
    };
  },
};

// ── list_hyperparams ─────────────────────────────────────────────────────────
const listHyperparams: ToolDef = {
  name: 'list_hyperparams',
  description: 'Dump the model-level hyperparameter table (learning rate, batch size, dropout, etc.) the user has set in the Neurarch hyperparams panel. Empty object when none defined. These are training-config knobs, NOT per-layer params.',
  inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  handler: (_args, model) => ({
    count: Object.keys(model.hyperparams ?? {}).length,
    hyperparams: model.hyperparams ?? {},
  }),
};

// ── get_design_notes ─────────────────────────────────────────────────────────
const getDesignNotes: ToolDef = {
  name: 'get_design_notes',
  description: 'Return the model\'s pinned design rationale: notes the user promoted from advisor warnings or agent replies, or typed manually. Each note has source/title/body/createdAt and optional affected layer ids. Use this to ground recommendations in the user\'s stated intent for the architecture before suggesting a change.',
  inputSchema: {
    type: 'object',
    properties: {
      layer: { type: 'string', description: 'Optional layer name or id. When set, return only notes affecting that layer.' },
    },
    additionalProperties: false,
  },
  handler: ({ layer }: { layer?: string }, model) => {
    const notes = model.designNotes ?? [];
    if (!layer) {
      return { count: notes.length, notes };
    }
    const { resolved } = resolveTargets(model, [layer]);
    if (!resolved.length) return { count: 0, notes: [], error: `Cannot find layer "${layer}".` };
    const ids = new Set(resolved.map(c => c.id));
    const filtered = notes.filter(n => n.affectedIds?.some(id => ids.has(id)));
    return { count: filtered.length, notes: filtered };
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
  describeArchitectureTool,
  getLayer,
  findLayers,
  layerImpact,
  paramCountByBlock,
  flopsByBlock,
  mermaidDiagram,
  listBlocks,
  getBlockTool,
  validateModelTool,
  findPath,
  listConnections,
  listHyperparams,
  getDesignNotes,
];

// Suppress unused-export TS warning when not destructured elsewhere
export { fmtBytes };
