import { describe, it, expect } from 'vitest';
import { describeArchitecture } from './describe.js';
import { makeModel, makeCyclicModel } from '../test/fixtures.js';

describe('describeArchitecture', () => {
  it('orders the pipeline topologically from input to output', () => {
    const d = describeArchitecture(makeModel());
    expect(d.pipeline[0]).toBe('input');
    expect(d.pipeline[d.pipeline.length - 1]).toBe('output');
    // embed must come before the blocks, blocks before the head
    expect(d.pipeline.indexOf('embed')).toBeLessThan(d.pipeline.indexOf('block_0'));
    expect(d.pipeline.indexOf('block_1')).toBeLessThan(d.pipeline.indexOf('lm_head'));
    expect(d.pipelineTruncated).toBe(false);
  });

  it('computes depth as the longest directed path edge count', () => {
    // input -> embed -> block_0 -> block_1 -> final_norm -> lm_head -> output = 6 edges
    expect(describeArchitecture(makeModel()).depth).toBe(6);
  });

  it('reports IO shapes and counts', () => {
    const d = describeArchitecture(makeModel());
    expect(d.inputShape).toEqual([128]);
    expect(d.outputShape).toEqual([128, 32000]);
    expect(d.layerCount).toBe(7);
    expect(d.blockCount).toBe(1);
  });

  it('ranks param and compute hotspots with percentages', () => {
    const d = describeArchitecture(makeModel());
    expect(d.paramHotspots.length).toBeGreaterThan(0);
    // the embedding table (32000 x 256 = 8.19M) dominates parameters
    expect(d.paramHotspots[0].name).toBe('embed');
    expect(d.paramHotspots[0].pctOfTotal).toBeGreaterThan(0);
    // hotspots are sorted descending
    for (let i = 1; i < d.paramHotspots.length; i++) {
      expect(d.paramHotspots[i - 1].value).toBeGreaterThanOrEqual(d.paramHotspots[i].value);
    }
    expect(d.computeHotspots.length).toBeGreaterThan(0);
  });

  it('embeds the validation rollup', () => {
    const d = describeArchitecture(makeModel());
    expect(d.validation.ok).toBe(true);
    expect(d.validation.errors).toBe(0);
  });

  it('returns null depth for a cyclic graph without throwing', () => {
    const d = describeArchitecture(makeCyclicModel());
    expect(d.depth).toBeNull();
    expect(d.pipeline.length).toBe(3); // still lists every node
  });
});
