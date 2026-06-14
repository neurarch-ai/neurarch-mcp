import { describe, it, expect } from 'vitest';
import { getBlock } from './blocks.js';
import { makeModel } from '../test/fixtures.js';

describe('getBlock', () => {
  it('resolves an explicit group by name with members and totals', () => {
    const b = getBlock(makeModel(), 'encoder')!;
    expect(b.source).toBe('group');
    expect(b.memberCount).toBe(2);
    expect(b.members.map(m => m.name).sort()).toEqual(['block_0', 'block_1']);
    expect(b.paramCount).toBeGreaterThan(0);
    expect(b.paramCountFormatted).toMatch(/[0-9]/);
  });

  it('reports the edges crossing the block boundary', () => {
    const b = getBlock(makeModel(), 'encoder')!;
    // embed -> block_0 enters the block; block_1 -> final_norm and the
    // block_0 -> final_norm residual leave it.
    expect(b.inputs.some(e => e.from === 'embed' && e.to === 'block_0')).toBe(true);
    expect(b.outputs.some(e => e.from === 'block_1' && e.to === 'final_norm')).toBe(true);
    expect(b.outputs.some(e => e.from === 'block_0' && e.to === 'final_norm' && e.label === 'residual')).toBe(true);
    // internal block_0 -> block_1 edge is NOT a boundary edge
    expect(b.inputs.concat(b.outputs).some(e => e.from === 'block_0' && e.to === 'block_1')).toBe(false);
  });

  it('falls back to scope-prefix matching when no group matches', () => {
    const b = getBlock(makeModel(), 'encoder.layer.0')!;
    expect(b.source).toBe('scope');
    expect(b.members.map(m => m.name)).toEqual(['block_0']);
  });

  it('returns null for an unknown name', () => {
    expect(getBlock(makeModel(), 'does_not_exist')).toBeNull();
    expect(getBlock(makeModel(), '')).toBeNull();
  });
});
