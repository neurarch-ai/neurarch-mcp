import { describe, it, expect, afterEach } from 'vitest';
import { parseModelRef, listZoo, loadZooModel, modelFromText, loadHFModel, setHfEnabled } from './sources.js';
import { makeModel } from './test/fixtures.js';

afterEach(() => setHfEnabled(false));

describe('parseModelRef', () => {
  it('routes zoo: and hf: prefixes and leaves everything else a file', () => {
    expect(parseModelRef('zoo:bert-base')).toEqual({ kind: 'zoo', id: 'bert-base' });
    expect(parseModelRef('HF:Qwen/Qwen2.5-7B')).toEqual({ kind: 'hf', id: 'Qwen/Qwen2.5-7B' });
    expect(parseModelRef('./model.py')).toEqual({ kind: 'file', path: './model.py' });
    // A Windows drive letter is not a scheme.
    expect(parseModelRef('C:\\models\\m.json').kind).toBe('file');
  });
});

describe('zoo', () => {
  it('ships the whole library, every entry loadable as a valid graph', async () => {
    const entries = await listZoo();
    expect(entries.length).toBeGreaterThanOrEqual(80);
    for (const e of entries) {
      const m = await loadZooModel(e.id);
      expect(Array.isArray(m.components)).toBe(true);
      expect(m.components.length).toBe(e.nodes);
      expect(m.components.every(c => typeof c.id === 'string')).toBe(true);
    }
  });

  it('names the nearest entries on a miss', async () => {
    await expect(loadZooModel('qwen')).rejects.toThrow(/Closest: .*qwen/);
    await expect(loadZooModel('zzz-nothing')).rejects.toThrow(/list_architectures/);
  });
});

describe('modelFromText', () => {
  it('reads a .neurarch.json document', () => {
    const m = modelFromText(JSON.stringify(makeModel()));
    expect(m.components.length).toBe(7);
  });
  it('reads PyTorch source', () => {
    const m = modelFromText('import torch.nn as nn\nclass M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, 4)\n    def forward(self, x):\n        return self.fc(x)\n');
    expect(m.components.some(c => c.type === 'linear')).toBe(true);
  });
  it('says which of the two it failed as', () => {
    expect(() => modelFromText('{ not json')).toThrow(/looks like JSON/);
    expect(() => modelFromText('{"a":1}')).toThrow(/components/);
    expect(() => modelFromText('print("hi")')).toThrow(/No PyTorch model/);
  });
});

describe('hf', () => {
  it('refuses without --hf, naming the flag, before touching the network', async () => {
    await expect(loadHFModel('Qwen/Qwen2.5-7B')).rejects.toThrow(/--hf/);
  });
  it('rejects a malformed repo id before touching the network', async () => {
    setHfEnabled(true);
    await expect(loadHFModel('not a repo')).rejects.toThrow(/org\/name/);
  });
});
