import { describe, it, expect } from 'vitest';
import { resolveTargets, resolveByPattern, analyzeImpact } from './modelImpact.js';
import { makeModel } from '../test/fixtures.js';

describe('resolveTargets', () => {
  it('matches by exact name', () => {
    const { resolved, unresolved } = resolveTargets(makeModel(), ['block_0']);
    expect(resolved.map(c => c.id)).toEqual(['blk0']);
    expect(unresolved).toEqual([]);
  });

  it('matches by id', () => {
    const { resolved } = resolveTargets(makeModel(), ['blk1']);
    expect(resolved.map(c => c.name)).toEqual(['block_1']);
  });

  it('falls back to case-insensitive substring', () => {
    const { resolved } = resolveTargets(makeModel(), ['BLOCK']);
    expect(resolved.map(c => c.id).sort()).toEqual(['blk0', 'blk1']);
  });

  it('supports /regex/ slash form', () => {
    const { resolved } = resolveTargets(makeModel(), ['/^block_/']);
    expect(resolved.map(c => c.id).sort()).toEqual(['blk0', 'blk1']);
  });

  it('reports unresolved needles', () => {
    const { resolved, unresolved } = resolveTargets(makeModel(), ['does_not_exist']);
    expect(resolved).toEqual([]);
    expect(unresolved).toEqual(['does_not_exist']);
  });

  it('de-dupes across needles', () => {
    const { resolved } = resolveTargets(makeModel(), ['block_0', 'blk0']);
    expect(resolved).toHaveLength(1);
  });
});

describe('resolveByPattern', () => {
  it('returns components whose name matches the regex source', () => {
    const matches = resolveByPattern(makeModel(), '^block_\\d$');
    expect(matches.map(c => c.id).sort()).toEqual(['blk0', 'blk1']);
  });

  it('returns [] for an invalid regex', () => {
    expect(resolveByPattern(makeModel(), '(')).toEqual([]);
  });
});

describe('analyzeImpact', () => {
  it('computes downstream and upstream blast radius', () => {
    const m = makeModel();
    const { resolved } = resolveTargets(m, ['block_0']);
    const report = analyzeImpact(m, resolved);
    const down = report.downstream.map(n => n.name).sort();
    expect(down).toEqual(['block_1', 'final_norm', 'lm_head', 'output']);
    const up = report.upstream.map(n => n.name).sort();
    expect(up).toEqual(['embed', 'input']);
  });

  it('flags shape-sensitive and param-carrying downstream layers', () => {
    const m = makeModel();
    const { resolved } = resolveTargets(m, ['block_0']);
    const report = analyzeImpact(m, resolved);
    // block_1, final_norm, lm_head are all shape-sensitive + param-carrying; output is neither
    expect(report.totals.shapeChanging).toBe(3);
    expect(report.totals.paramCarrying).toBe(3);
  });

  it('honors includeUpstream=false', () => {
    const m = makeModel();
    const { resolved } = resolveTargets(m, ['block_0']);
    const report = analyzeImpact(m, resolved, { includeUpstream: false });
    expect(report.upstream).toEqual([]);
  });

  it('honors maxDistance', () => {
    const m = makeModel();
    const { resolved } = resolveTargets(m, ['block_0']);
    const report = analyzeImpact(m, resolved, { maxDistance: 1 });
    expect(report.downstream.map(n => n.name).sort()).toEqual(['block_1', 'final_norm']);
  });

  it('returns a non-empty markdown summary', () => {
    const m = makeModel();
    const { resolved } = resolveTargets(m, ['block_0']);
    const report = analyzeImpact(m, resolved);
    expect(report.summary).toMatch(/Target/);
  });
});
