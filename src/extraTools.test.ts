import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, writeFile, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { TOOLS, type ToolContext } from './tools.js';
import { EXTRA_TOOLS, HF_TOOLS, hfToolsIfEnabled } from './extraTools.js';
import { makeModel } from './test/fixtures.js';
import { setHfEnabled } from './sources.js';
import { clearModelCache } from './models.js';
import type { ModelArchitecture } from './lib/types.js';

let dir: string;
beforeEach(async () => { dir = await mkdtemp(join(tmpdir(), 'nx-extra-')); clearModelCache(); });
afterEach(async () => { await rm(dir, { recursive: true, force: true }); setHfEnabled(false); });

function call(name: string, args: Record<string, unknown>, ctx: Partial<ToolContext> = {}, model: ModelArchitecture = makeModel()): any {
  const tool = [...TOOLS, ...HF_TOOLS].find(t => t.name === name);
  if (!tool) throw new Error(`tool ${name} not registered`);
  return tool.handler(args, model, { modelPath: join(dir, 'model.neurarch.json'), ...ctx });
}

// The unit fixture is deliberately incomplete (no input shape, no numHeads) and
// pre-flight blocks it, so the ranker tests use the shipped examples: tiny-gpt
// is legal, tiny-vit.py carries a planted head-dim crash.
const GPT = resolve(__dirname, '../examples/tiny-gpt.neurarch.json');
const VIT = resolve(__dirname, '../examples/tiny-vit.py');
async function legalModel(): Promise<ModelArchitecture> { return JSON.parse(await readFile(GPT, 'utf-8')); }

describe('registry', () => {
  it('lists the 0.13 tools after the original ones, and load_hf_model only behind --hf', () => {
    for (const t of EXTRA_TOOLS) expect(TOOLS.includes(t)).toBe(true);
    expect(TOOLS.some(t => t.name === 'load_hf_model')).toBe(false);
    expect(hfToolsIfEnabled()).toEqual([]);
    setHfEnabled(true);
    expect(hfToolsIfEnabled().map(t => t.name)).toEqual(['load_hf_model']);
    expect(HF_TOOLS[0].annotations?.openWorldHint).toBe(true);
  });
});

describe('rank_designs', () => {
  it('puts a blocked candidate last, keeps a tie a tie, and ships calibration', async () => {
    const legal = await legalModel();
    const r = await call('rank_designs', {
      candidates: [
        { id: 'a', model: legal },
        { id: 'broken', model_path: VIT },
      ],
      include_current: true,
    }, {}, legal);
    const order = r.ranked.map((c: any) => c.id);
    expect(order[order.length - 1]).toBe('broken');
    expect(r.ranked.find((c: any) => c.id === 'broken').tier).toBe('blocked');
    expect(r.budget.wouldNotRun).toEqual(['broken']);
    // 'a' and 'current' are the same graph: nothing separates them.
    expect(r.recommended).toBeNull();
    expect(r.ranked.find((c: any) => c.id === 'a').rank).toBe(r.ranked.find((c: any) => c.id === 'current').rank);
    expect(r.calibration.ordering.pairwiseAccuracy).toBeLessThan(0.6);
    expect(r.details.broken.blocking.length).toBeGreaterThan(0);
  });

  it('loads candidates from paths and zoo refs, and labels them by path when no id is given', async () => {
    const p = join(dir, 'variant.neurarch.json');
    await writeFile(p, JSON.stringify(makeModel()));
    const r = await call('rank_designs', { candidates: [{ model_path: p }, { id: 'ref', model_path: 'zoo:bert-base' }] });
    expect(r.ranked.map((c: any) => c.id).sort()).toEqual([p, 'ref'].sort());
  });

  it('refuses a candidate with both or neither source, and duplicate ids', async () => {
    await expect(call('rank_designs', { candidates: [{ id: 'x' }] })).rejects.toThrow(/neither/);
    await expect(call('rank_designs', { candidates: [{ id: 'x', model: makeModel(), model_path: 'zoo:bert-base' }] })).rejects.toThrow(/not both/);
    await expect(call('rank_designs', { candidates: [{ id: 'x', model: makeModel() }, { id: 'x', model: makeModel() }] })).rejects.toThrow(/Duplicate/);
  });
});

describe('export_pytorch', () => {
  it('returns source and lists the layer types it had no template for', async () => {
    const r = await call('export_pytorch', {});
    expect(r.language).toBe('python');
    expect(r.code).toMatch(/import torch/);
    expect(r.code).toMatch(/nn\.Module/);
    expect(Array.isArray(r.unsupportedLayers)).toBe(true);
  });
  it('writes only under --write, and never over the source file', async () => {
    const out = join(dir, 'gen.py');
    const refused = await call('export_pytorch', { save_to: out });
    expect(refused.saveRefused).toMatch(/--write/);
    const ok = await call('export_pytorch', { save_to: out }, { writeEnabled: true });
    expect(ok.savedTo).toBe(out);
    expect(await readFile(out, 'utf-8')).toMatch(/nn\.Module/);
    await expect(call('export_pytorch', { save_to: join(dir, 'model.neurarch.json') }, { writeEnabled: true })).rejects.toThrow(/refuses to overwrite/);
  });
});

describe('list_architectures / load_architecture', () => {
  it('searches by substring, domain and attention', async () => {
    const all = await call('list_architectures', {});
    expect(all.total).toBeGreaterThanOrEqual(80);
    expect(all.matched).toBe(all.total);
    const q = await call('list_architectures', { query: 'qwen' });
    expect(q.matched).toBeGreaterThan(0);
    expect(q.architectures.every((a: any) => a.modelPath.startsWith('zoo:'))).toBe(true);
    const gqa = await call('list_architectures', { attention: 'groupedQueryAttention' });
    expect(gqa.matched).toBeGreaterThan(0);
    expect(gqa.matched).toBeLessThan(all.total);
    const d = await call('list_architectures', { domain: all.domains[0] });
    expect(d.architectures.every((a: any) => a.domain === all.domains[0])).toBe(true);
  });
  it('describes an entry and saves it only under --write', async () => {
    const r = await call('load_architecture', { id: 'bert-base' });
    expect(r.modelPath).toBe('zoo:bert-base');
    expect(r.layerCount).toBeGreaterThan(10);
    expect(r.pipeline.length).toBeGreaterThan(0);
    const out = join(dir, 'bert.neurarch.json');
    expect((await call('load_architecture', { id: 'bert-base', save_to: out })).saveRefused).toMatch(/--write/);
    const saved = await call('load_architecture', { id: 'bert-base', save_to: out }, { writeEnabled: true });
    expect(saved.savedTo).toBe(out);
    expect(JSON.parse(await readFile(out, 'utf-8')).components.length).toBe(r.layerCount);
  });
});

describe('find_models', () => {
  it('finds parseable source, flags unparseable nn.Modules, skips junk dirs', async () => {
    await writeFile(join(dir, 'net.py'), 'import torch.nn as nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(8, 16)\n        self.act = nn.ReLU()\n        self.fc2 = nn.Linear(16, 8)\n        self.act2 = nn.ReLU()\n        self.fc3 = nn.Linear(8, 4)\n    def forward(self, x):\n        return self.fc3(self.act2(self.fc2(self.act(self.fc1(x)))))\n');
    await writeFile(join(dir, 'thin.py'), 'import torch.nn as nn\nclass Thin(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, 4)\n        self.act = nn.ReLU()\n    def forward(self, x):\n        return self.act(self.fc(x))\n');
    await writeFile(join(dir, 'dynamic.py'), 'import torch.nn as nn\nclass Dyn(nn.Module):\n    def __init__(self, cfg):\n        super().__init__()\n        self.body = build(cfg)\n');
    await writeFile(join(dir, 'train.py'), 'def train():\n    pass\n');
    await writeFile(join(dir, 'saved.neurarch.json'), JSON.stringify(makeModel()));
    const { mkdir } = await import('node:fs/promises');
    await mkdir(join(dir, 'node_modules', 'x'), { recursive: true });
    await writeFile(join(dir, 'node_modules', 'x', 'evil.py'), 'class E(nn.Module): pass\n');
    const r = await call('find_models', { dir });
    const byPath = Object.fromEntries(r.models.map((m: any) => [m.path, m]));
    expect(byPath['net.py'].status).toBe('parsed');
    expect(byPath['net.py'].classes).toEqual(['Net']);
    expect(byPath['thin.py'].status).toBe('partial');
    expect(byPath['thin.py'].detail).toMatch(/neurarch-trace/);
    expect(byPath['dynamic.py'].status).toBe('no-model');
    expect(byPath['dynamic.py'].detail).toMatch(/neurarch-trace/);
    expect(byPath['saved.neurarch.json'].kind).toBe('neurarch-json');
    expect(byPath['train.py']).toBeUndefined();
    expect(Object.keys(byPath).some(p => p.includes('node_modules'))).toBe(false);
    expect(r.models[0].status).toBe('parsed');
  });
  it('defaults to the directory of the server model', async () => {
    await writeFile(join(dir, 'a.neurarch.json'), JSON.stringify(makeModel()));
    const r = await call('find_models', {});
    expect(r.dir).toBe(dir);
    expect(r.models.length).toBe(1);
  });
});
