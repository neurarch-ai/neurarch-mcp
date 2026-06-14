import { describe, it, expect } from 'vitest';
import { compareLayers } from './compareLayers.js';
import { makeModel } from '../test/fixtures.js';

describe('compareLayers', () => {
  it('reports two structurally identical layers as identical', () => {
    const c = compareLayers(makeModel(), 'block_0', 'block_1')!;
    expect(c.sameType).toBe(true);
    expect(c.identical).toBe(true);
    expect(c.paramDelta).toBe(0);
    expect(c.paramValueDiffs).toEqual([]);
    expect(c.inputShapeMatch).toBe(true);
    expect(c.outputShapeMatch).toBe(true);
  });

  it('detects a differing param value', () => {
    const m = makeModel();
    m.components.find(x => x.name === 'block_1')!.params.ffDim = 2048;
    const c = compareLayers(m, 'block_0', 'block_1')!;
    expect(c.identical).toBe(false);
    expect(c.paramValueDiffs).toEqual([{ key: 'ffDim', a: 1024, b: 2048 }]);
    expect(c.paramDelta).not.toBe(0);
  });

  it('diffs layers of different type with disjoint param keys', () => {
    const c = compareLayers(makeModel(), 'block_0', 'final_norm')!;
    expect(c.sameType).toBe(false);
    expect(c.identical).toBe(false);
    expect(c.paramKeysOnlyInA.sort()).toEqual(['embedDim', 'ffDim']);
    expect(c.paramKeysOnlyInB).toEqual(['normalizedShape']);
    expect(c.paramDeltaFormatted).toMatch(/^[+-]/);
  });

  it('returns null when either layer cannot be resolved', () => {
    expect(compareLayers(makeModel(), 'block_0', 'ghost_xyz')).toBeNull();
    expect(compareLayers(makeModel(), 'ghost_xyz', 'block_0')).toBeNull();
  });
});
