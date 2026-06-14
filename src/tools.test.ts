import { describe, it, expect } from 'vitest';
import { TOOLS, type ToolContext } from './tools.js';
import { makeModel } from './test/fixtures.js';
import type { ModelArchitecture } from './lib/types.js';

const CTX: ToolContext = { modelPath: '/tmp/fixture.json' };

function call(name: string, args: Record<string, unknown>, model: ModelArchitecture = makeModel()): any {
  const tool = TOOLS.find(t => t.name === name);
  if (!tool) throw new Error(`tool ${name} not registered`);
  return tool.handler(args, model, CTX);
}

describe('tool registry', () => {
  it('exposes uniquely-named read tools, each with an object input schema', () => {
    const names = TOOLS.map(t => t.name);
    expect(new Set(names).size).toBe(names.length);
    for (const t of TOOLS) {
      expect(t.description.length).toBeGreaterThan(0);
      expect((t.inputSchema as any).type).toBe('object');
    }
  });
});

describe('get_model_summary', () => {
  it('summarizes counts, params, and dominant types', () => {
    const s = call('get_model_summary', {});
    expect(s.layerCount).toBe(7);
    expect(s.connectionCount).toBe(7);
    expect(s.groupCount).toBe(1);
    expect(s.totalParameters).toBeGreaterThan(0);
    expect(s.dominantTypes.find((d: any) => d.type === 'transformerBlock').count).toBe(2);
    expect(s.inputShape).toEqual([128]);
    expect(s.outputShape).toEqual([128, 32000]);
  });
});

describe('describe_architecture', () => {
  it('returns a one-call orientation with pipeline, depth, hotspots, validation', () => {
    const d = call('describe_architecture', {});
    expect(d.pipeline[0]).toBe('input');
    expect(d.depth).toBe(6);
    expect(d.paramHotspots[0].name).toBe('embed');
    expect(d.validation.ok).toBe(true);
  });
});

describe('get_layer', () => {
  it('returns full detail with upstream/downstream', () => {
    const l = call('get_layer', { name: 'final_norm' });
    expect(l.type).toBe('layerNorm');
    expect(l.upstream.length).toBeGreaterThan(0);
    expect(l.downstream.length).toBeGreaterThan(0);
  });
  it('returns null when nothing matches', () => {
    expect(call('get_layer', { name: 'definitely_absent' })).toBeNull();
  });
});

describe('find_layers', () => {
  it('filters by type', () => {
    const r = call('find_layers', { type: 'transformerBlock' });
    expect(r.count).toBe(2);
  });
  it('filters by name regex', () => {
    const r = call('find_layers', { namePattern: '^block_' });
    expect(r.count).toBe(2);
  });
  it('returns an error object for an invalid regex', () => {
    const r = call('find_layers', { namePattern: '(' });
    expect(r.error).toMatch(/Invalid regex/);
  });
});

describe('layer_impact', () => {
  it('resolves by names and returns downstream', () => {
    const r = call('layer_impact', { names: ['block_0'] });
    expect(r.downstream.map((n: any) => n.name)).toContain('block_1');
  });
  it('errors when neither names nor namePattern provided', () => {
    const r = call('layer_impact', {});
    expect(r.error).toBeDefined();
  });
  it('errors on unresolved names', () => {
    const r = call('layer_impact', { names: ['ghost_xyz'] });
    expect(r.error).toMatch(/Unresolved/);
  });
});

describe('find_path', () => {
  it('finds the shortest directed path', () => {
    const r = call('find_path', { from: 'input', to: 'output' });
    expect(r.reachable).toBe(true);
    expect(r.path[0].name).toBe('input');
    expect(r.path[r.path.length - 1].name).toBe('output');
    expect(r.length).toBe(r.path.length - 1);
  });
  it('reports unreachable when going against the flow', () => {
    const r = call('find_path', { from: 'output', to: 'input' });
    expect(r.reachable).toBe(false);
    expect(r.path).toBeNull();
  });
  it('errors on an unknown endpoint', () => {
    const r = call('find_path', { from: 'ghost_xyz', to: 'output' });
    expect(r.error).toMatch(/from/);
  });
});

describe('list_connections', () => {
  it('returns the full edge list with names', () => {
    const r = call('list_connections', {});
    expect(r.count).toBe(7);
    expect(r.connections.some((e: any) => e.label === 'residual')).toBe(true);
  });
  it('filters by source', () => {
    const r = call('list_connections', { from: 'block_0' });
    expect(r.connections.every((e: any) => e.from === 'block_0')).toBe(true);
  });
});

describe('param_count_by_block / flops_by_block', () => {
  it('param_count groups by block by default with a positive total', () => {
    const r = call('param_count_by_block', {});
    expect(Array.isArray(r.buckets)).toBe(true);
    expect(r.total).toBeGreaterThan(0);
  });
  it('flops_by_block groups by type when asked', () => {
    const r = call('flops_by_block', { groupBy: 'type' });
    expect(r.source).toBe('type');
  });
});

describe('list_blocks', () => {
  it('lists explicit groups when present', () => {
    const r = call('list_blocks', {});
    expect(r.source).toBe('groups');
    const enc = r.blocks.find((b: any) => b.name === 'encoder');
    expect(enc.memberCount).toBe(2);
    expect(enc.members.sort()).toEqual(['block_0', 'block_1']);
  });
});

describe('validate_model tool', () => {
  it('passes the fixture', () => {
    expect(call('validate_model', {}).ok).toBe(true);
  });
});

describe('mermaid_diagram', () => {
  it('renders without truncation for a small model', () => {
    const r = call('mermaid_diagram', {});
    expect(r.truncated).toBe(false);
    expect(r.mermaid).toMatch(/^flowchart TD/);
  });
  it('truncates and reports the cap', () => {
    const r = call('mermaid_diagram', { maxLayers: 3 });
    expect(r.truncated).toBe(true);
    expect(r.shown).toBe(3);
    expect(r.total).toBe(7);
  });
});

describe('list_hyperparams / get_design_notes', () => {
  it('dumps hyperparams', () => {
    const r = call('list_hyperparams', {});
    expect(r.count).toBe(2);
    expect(r.hyperparams.learningRate.value).toBe(0.0003);
  });
  it('returns all design notes', () => {
    expect(call('get_design_notes', {}).count).toBe(1);
  });
  it('filters design notes by affected layer', () => {
    const r = call('get_design_notes', { layer: 'block_0' });
    expect(r.count).toBe(1);
  });
});
