import { describe, it, expect } from 'vitest';
import { estimateLayerParams, fmtParams } from './paramEstimator.js';
import { estimateLayerFlops, fmtFlops, fmtBytes } from './flopsEstimator.js';

describe('estimateLayerParams', () => {
  it('linear = in*out + out (bias)', () => {
    expect(estimateLayerParams('linear', { inFeatures: 256, outFeatures: 10 })).toBe(256 * 10 + 10);
  });

  it('linear infers inFeatures from last input dim', () => {
    expect(estimateLayerParams('linear', { outFeatures: 10 }, [256])).toBe(256 * 10 + 10);
  });

  it('conv2d = inC*outC*k*k + outC', () => {
    expect(estimateLayerParams('conv2d', { inChannels: 3, outChannels: 16, kernelSize: 3 })).toBe(3 * 16 * 9 + 16);
  });

  it('embedding = vocab * dim, no bias', () => {
    expect(estimateLayerParams('embedding', { vocabSize: 1000, embeddingDim: 64 })).toBe(64000);
  });

  it('layerNorm = 2 * features (scale + bias)', () => {
    expect(estimateLayerParams('layerNorm', { normalizedShape: 256 })).toBe(512);
  });

  it('lmHead = embedDim * vocabSize, no bias by default', () => {
    expect(estimateLayerParams('lmHead', { embedDim: 256, vocabSize: 32000 })).toBe(256 * 32000);
    expect(estimateLayerParams('lmHead', { embedDim: 256, vocabSize: 32000, bias: true })).toBe(256 * 32000 + 32000);
  });

  it('conv2d divides input channels by groups (matches FLOPs convention)', () => {
    const dense = estimateLayerParams('conv2d', { inChannels: 64, outChannels: 64, kernelSize: 3 });
    const grouped = estimateLayerParams('conv2d', { inChannels: 64, outChannels: 64, kernelSize: 3, groups: 2 });
    expect(dense).toBe(64 * 64 * 9 + 64);
    expect(grouped).toBe(32 * 64 * 9 + 64); // inC/groups
  });

  it('bidirectionalLSTM counts stacked layers; L=1 matches the legacy formula', () => {
    const single = estimateLayerParams('bidirectionalLSTM', { hiddenSize: 128, inputSize: 64 });
    expect(single).toBe(2 * (4 * (64 * 128 + 128 * 128 + 2 * 128)));
    const stacked = estimateLayerParams('bidirectionalLSTM', { hiddenSize: 128, inputSize: 64, numLayers: 3 });
    expect(stacked).toBeGreaterThan(single);
  });

  it('lmHead returns 0 new params when weight-tied to the embedding', () => {
    expect(estimateLayerParams('lmHead', { embedDim: 256, vocabSize: 32000, weightTied: true })).toBe(0);
    expect(estimateLayerParams('lmHead', { embedDim: 256, vocabSize: 32000, tied: true })).toBe(0);
    expect(estimateLayerParams('lmHead', { embedDim: 256, vocabSize: 32000 })).toBe(256 * 32000);
  });

  it('parameter-free ops return 0', () => {
    expect(estimateLayerParams('relu', {})).toBe(0);
    expect(estimateLayerParams('maxpool2d', {})).toBe(0);
    expect(estimateLayerParams('dropout', { p: 0.5 })).toBe(0);
  });

  it('coerces invalid params to a safe fallback rather than NaN', () => {
    const out = estimateLayerParams('linear', { inFeatures: 'oops', outFeatures: 10 }, [128]);
    expect(Number.isFinite(out)).toBe(true);
  });
});

describe('fmtParams', () => {
  it('formats magnitudes with suffixes', () => {
    expect(fmtParams(0)).toBe('—');
    expect(fmtParams(512)).toBe('512');
    expect(fmtParams(1500)).toBe('1.5K');
    expect(fmtParams(2_000_000)).toBe('2.00M');
    expect(fmtParams(3_000_000_000)).toBe('3.00B');
  });

  it('returns ? for NaN', () => {
    expect(fmtParams(NaN)).toBe('?');
  });
});

describe('estimateLayerFlops', () => {
  it('linear MACs = in*out*batch', () => {
    expect(estimateLayerFlops('linear', { inFeatures: 256, outFeatures: 10 }, [256], [10])).toBe(2560);
  });

  it('conv2d MACs scale with output spatial size', () => {
    const macs = estimateLayerFlops('conv2d', { inChannels: 3, outChannels: 16, kernelSize: 3 }, [3, 32, 32], [16, 32, 32]);
    expect(macs).toBe((3 / 1) * 16 * 3 * 3 * 32 * 32);
  });

  it('bidirectionalLSTM FLOPs scale with numLayers; L=1 matches legacy', () => {
    const single = estimateLayerFlops('bidirectionalLSTM', { hiddenSize: 128, inputSize: 64 }, [10, 64], [10, 256]);
    expect(single).toBe(2 * 4 * 10 * (64 * 128 + 128 * 128));
    const stacked = estimateLayerFlops('bidirectionalLSTM', { hiddenSize: 128, inputSize: 64, numLayers: 2 }, [10, 64], [10, 256]);
    expect(stacked).toBeGreaterThan(single);
  });

  it('activations and embeddings are cheap / free', () => {
    expect(estimateLayerFlops('embedding', {}, [128], [128, 64])).toBe(0);
    expect(estimateLayerFlops('relu', {}, [10], [10])).toBe(10);
  });
});

describe('fmtFlops / fmtBytes', () => {
  it('fmtFlops uses G/M/K/T suffixes and — for zero', () => {
    expect(fmtFlops(0)).toBe('—');
    expect(fmtFlops(1500)).toBe('1.5 K');
    expect(fmtFlops(2e9)).toBe('2.00 G');
    expect(fmtFlops(5e12)).toBe('5.00 T');
  });

  it('fmtBytes uses KB/MB/GB suffixes', () => {
    expect(fmtBytes(0)).toBe('—');
    expect(fmtBytes(2048)).toBe('2.0 KB');
    expect(fmtBytes(5 * 1_048_576)).toBe('5.0 MB');
  });
});
