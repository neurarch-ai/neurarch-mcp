import { describe, it, expect } from 'vitest';
import { diffModels } from './diffModels.js';
import { makeModel } from '../test/fixtures.js';

describe('diffModels', () => {
  it('reports an identical model as identical', () => {
    const d = diffModels(makeModel(), makeModel());
    expect(d.identical).toBe(true);
    expect(d.layers.onlyInCurrent).toEqual([]);
    expect(d.layers.modified).toEqual([]);
    expect(d.summary).toMatch(/identical/);
  });

  it('detects an added layer (only in current)', () => {
    const current = makeModel();
    const other = makeModel();
    other.components = other.components.filter(c => c.name !== 'final_norm');
    const d = diffModels(current, other);
    expect(d.layers.onlyInCurrent.map(l => l.name)).toContain('final_norm');
    expect(d.identical).toBe(false);
  });

  it('detects a removed layer (only in other)', () => {
    const current = makeModel();
    current.components = current.components.filter(c => c.name !== 'lm_head');
    const d = diffModels(current, makeModel());
    expect(d.layers.onlyInOther.map(l => l.name)).toContain('lm_head');
  });

  it('reports field-level modifications phrased other -> current', () => {
    const current = makeModel();
    current.components.find(c => c.name === 'block_0')!.params.ffDim = 4096;
    current.components.find(c => c.name === 'block_0')!.type = 'feedForward';
    const d = diffModels(current, makeModel());
    const mod = d.layers.modified.find(m => m.name === 'block_0')!;
    expect(mod.changes).toContain('type: transformerBlock -> feedForward');
    expect(mod.changes.some(c => c.includes('ffDim') && c.includes('4096'))).toBe(true);
  });

  it('diffs connections by endpoint name', () => {
    const current = makeModel();
    // drop the residual block_0 -> final_norm edge
    current.connections = current.connections.filter(c => c.id !== 'c4');
    const d = diffModels(current, makeModel());
    expect(d.connections.onlyInOther).toEqual([{ from: 'block_0', to: 'final_norm' }]);
    expect(d.connections.onlyInCurrent).toEqual([]);
  });

  it('flags duplicate names as ambiguous instead of matching them', () => {
    const current = makeModel();
    current.components[1].name = current.components[2].name; // create a dup name
    const d = diffModels(current, makeModel());
    expect(d.ambiguousNames.current.length).toBeGreaterThan(0);
    expect(d.identical).toBe(false);
  });
});
