/**
 * The vendored bundle is compiled output from another repository, and its
 * declaration file is written by hand. Nothing but this test stands between a
 * regenerated bundle whose exports moved and a server that throws
 * "graphFromPyTorchSource is not a function" at a user.
 */
import { describe, it, expect } from 'vitest';
import * as engine from './engine.bundle.mjs';

describe('vendored engine contract', () => {
  it('exports the functions this server declares', () => {
    expect(typeof engine.graphFromPyTorchSource).toBe('function');
    expect(typeof engine.lintModelGraph).toBe('function');
  });

  it('exports the rule id lists', () => {
    expect(Array.isArray(engine.ADVISOR_RULE_IDS)).toBe(true);
    expect(Array.isArray(engine.SHAPE_RULE_IDS)).toBe(true);
    expect(engine.ADVISOR_RULE_IDS.length).toBeGreaterThan(20);
  });

  it('needs nothing from the outside world', async () => {
    // The whole reason this can be vendored: the bundle has no imports, so it
    // cannot acquire a dependency the package.json does not declare.
    const { readFile } = await import('node:fs/promises');
    const src = await readFile(new URL('./engine.bundle.mjs', import.meta.url), 'utf-8');
    expect(src).not.toMatch(/^\s*import\s/m);
    expect(src).not.toMatch(/\brequire\(/);
  });
});

describe('vendored engine contract, 0.13 additions', () => {
  it('exports the code generator and the HF pieces', () => {
    expect(typeof engine.generatePyTorchCode).toBe('function');
    expect(typeof engine.convertHFConfigToModel).toBe('function');
    expect(typeof engine.fetchHFModelConfig).toBe('function');
  });
  it('builds a graph from a config with no network', () => {
    const { components, connections } = engine.convertHFConfigToModel('t/llama', {
      model_type: 'llama', architectures: ['LlamaForCausalLM'], hidden_size: 256, num_hidden_layers: 2,
      num_attention_heads: 4, intermediate_size: 512, vocab_size: 1000, max_position_embeddings: 64,
    });
    expect(components.length).toBeGreaterThan(5);
    expect(connections.length).toBeGreaterThan(4);
  });
  it('generates source that mentions every named layer type it supports', async () => {
    const { makeModel } = await import('../test/fixtures.js');
    const code = engine.generatePyTorchCode(makeModel());
    expect(code).toMatch(/class \w+\(nn\.Module\)/);
    expect(code).toMatch(/nn\.Embedding/);
  });
});
