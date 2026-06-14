/**
 * Shared test fixtures. `makeModel()` returns a fresh deep clone each call so a
 * mutation test can't bleed into the next test. The base model is a tiny but
 * structurally representative graph: input -> embedding -> two transformer
 * blocks (grouped) -> layerNorm -> lmHead -> output, with a residual skip and a
 * couple of hyperparams / design notes so every tool has something to chew on.
 */
import type { ModelArchitecture } from '../lib/types.js';

const BASE: ModelArchitecture = {
  id: 'model-1',
  name: 'tiny-transformer',
  description: 'A fixture transformer for tests.',
  components: [
    { id: 'in',   type: 'input',       name: 'input',   position: { x: 0, y: 0 },   params: {}, inputs: [], outputs: ['emb'], outputShape: [128] },
    { id: 'emb',  type: 'embedding',   name: 'embed',   position: { x: 0, y: 100 }, params: { vocabSize: 32000, embeddingDim: 256 }, inputs: ['in'], outputs: ['blk0'], scope: 'encoder.embed', outputShape: [128, 256] },
    { id: 'blk0', type: 'transformerBlock', name: 'block_0', position: { x: 0, y: 200 }, params: { embedDim: 256, ffDim: 1024 }, inputs: ['emb'], outputs: ['blk1', 'norm'], scope: 'encoder.layer.0', outputShape: [128, 256] },
    { id: 'blk1', type: 'transformerBlock', name: 'block_1', position: { x: 0, y: 300 }, params: { embedDim: 256, ffDim: 1024 }, inputs: ['blk0'], outputs: ['norm'], scope: 'encoder.layer.1', outputShape: [128, 256] },
    { id: 'norm', type: 'layerNorm',    name: 'final_norm', position: { x: 0, y: 400 }, params: { normalizedShape: 256 }, inputs: ['blk1', 'blk0'], outputs: ['head'], scope: 'encoder.norm', outputShape: [128, 256] },
    { id: 'head', type: 'lmHead',       name: 'lm_head', position: { x: 0, y: 500 }, params: { embedDim: 256, vocabSize: 32000 }, inputs: ['norm'], outputs: ['out'], outputShape: [128, 32000] },
    { id: 'out',  type: 'output',       name: 'output',  position: { x: 0, y: 600 }, params: {}, inputs: ['head'], outputs: [], inputShape: [128, 32000] },
  ],
  connections: [
    { id: 'c0', from: 'in',   to: 'emb',  fromPort: 'bottom', toPort: 'top' },
    { id: 'c1', from: 'emb',  to: 'blk0', fromPort: 'bottom', toPort: 'top' },
    { id: 'c2', from: 'blk0', to: 'blk1', fromPort: 'bottom', toPort: 'top' },
    { id: 'c3', from: 'blk1', to: 'norm', fromPort: 'bottom', toPort: 'top' },
    { id: 'c4', from: 'blk0', to: 'norm', fromPort: 'bottom', toPort: 'top', label: 'residual' },
    { id: 'c5', from: 'norm', to: 'head', fromPort: 'bottom', toPort: 'top' },
    { id: 'c6', from: 'head', to: 'out',  fromPort: 'bottom', toPort: 'top' },
  ],
  groups: [
    { id: 'g0', name: 'encoder', componentIds: ['blk0', 'blk1'], collapsed: false },
  ],
  hyperparams: {
    learningRate: { value: 0.0003, type: 'float', description: 'AdamW LR' },
    batchSize: { value: 32, type: 'int' },
  },
  designNotes: [
    { id: 'n0', source: 'manual', title: 'Why two blocks', body: 'Depth-2 keeps the fixture cheap.', createdAt: '2026-01-01T00:00:00Z', affectedIds: ['blk0', 'blk1'] },
  ],
};

/** Fresh deep clone of the base model — safe to mutate in a test. */
export function makeModel(): ModelArchitecture {
  return structuredClone(BASE);
}

/** A model with a deliberate cycle (a -> b -> c -> a). */
export function makeCyclicModel(): ModelArchitecture {
  return {
    id: 'cyc', name: 'cyclic', components: [
      { id: 'a', type: 'linear', name: 'a', position: { x: 0, y: 0 }, params: {}, inputs: ['c'], outputs: ['b'] },
      { id: 'b', type: 'linear', name: 'b', position: { x: 0, y: 0 }, params: {}, inputs: ['a'], outputs: ['c'] },
      { id: 'c', type: 'linear', name: 'c', position: { x: 0, y: 0 }, params: {}, inputs: ['b'], outputs: ['a'] },
    ],
    connections: [
      { id: 'e0', from: 'a', to: 'b', fromPort: 'bottom', toPort: 'top' },
      { id: 'e1', from: 'b', to: 'c', fromPort: 'bottom', toPort: 'top' },
      { id: 'e2', from: 'c', to: 'a', fromPort: 'bottom', toPort: 'top' },
    ],
  };
}

/** A model with a dangling edge, a duplicate name, and an orphan. */
export function makeBrokenModel(): ModelArchitecture {
  return {
    id: 'brk', name: 'broken', components: [
      { id: 'x', type: 'input',  name: 'dup', position: { x: 0, y: 0 }, params: {}, inputs: [], outputs: ['y'] },
      { id: 'y', type: 'linear', name: 'dup', position: { x: 0, y: 0 }, params: {}, inputs: ['x'], outputs: [] },
      { id: 'z', type: 'linear', name: 'lonely', position: { x: 0, y: 0 }, params: {}, inputs: [], outputs: [] },
    ],
    connections: [
      { id: 'e0', from: 'x', to: 'y', fromPort: 'bottom', toPort: 'top' },
      { id: 'e1', from: 'x', to: 'ghost', fromPort: 'bottom', toPort: 'top' },
    ],
  };
}
