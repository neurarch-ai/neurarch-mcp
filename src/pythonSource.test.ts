/**
 * Pointing the server at a `.py` file.
 *
 * The interesting cases are not "does it parse" (the vendored engine's job,
 * tested where it lives) but the boundaries around it: what a partial graph is
 * allowed to claim, and what the write path is forbidden to do to source code.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadModelFile, sourceKindFor } from './loader.js';
import { WRITE_TOOLS } from './writeTools.js';
import { TOOLS } from './tools.js';
import { makeModel } from './test/fixtures.js';

const MLP = `
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.norm = nn.LayerNorm(256)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.9)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.norm(self.fc1(x)))))
`;

let dir: string;
const write = async (name: string, body: string) => {
  const path = join(dir, name);
  await writeFile(path, body, 'utf-8');
  return path;
};

beforeEach(async () => { dir = await mkdtemp(join(tmpdir(), 'nx-mcp-py-')); });
afterEach(async () => { await rm(dir, { recursive: true, force: true }); });

describe('sourceKindFor', () => {
  it('routes on extension, case-insensitively', () => {
    expect(sourceKindFor('/a/model.py')).toBe('pytorch-source');
    expect(sourceKindFor('/a/Model.PY')).toBe('pytorch-source');
    expect(sourceKindFor('/a/model.neurarch.json')).toBe('neurarch-json');
    // A pyi stub is not source we can parse, and must not be treated as such.
    expect(sourceKindFor('/a/model.pyi')).toBe('neurarch-json');
  });
});

describe('loading a .py model', () => {
  it('reads layers, types, params and wiring out of the source', async () => {
    const model = await loadModelFile(await write('net.py', MLP));
    const byName = Object.fromEntries(model.components.map(c => [c.name, c]));
    expect(Object.keys(byName)).toEqual(
      expect.arrayContaining(['fc1', 'norm', 'act', 'drop', 'fc2']),
    );
    expect(byName.fc1.type).toBe('linear');
    expect(byName.fc1.params).toMatchObject({ inFeatures: 784, outFeatures: 256 });
    expect(byName.drop.type).toBe('dropout');
    expect(model.connections.length).toBeGreaterThan(0);
  });

  it('names the model after the file', async () => {
    const model = await loadModelFile(await write('resnet_tiny.py', MLP));
    expect(model.name).toBe('resnet_tiny');
  });

  it('counts parameters, which do not need shapes', async () => {
    const model = await loadModelFile(await write('net.py', MLP));
    const summary = TOOLS.find(t => t.name === 'get_model_summary')!;
    const out = await summary.handler({}, model, { modelPath: 'net.py' }) as {
      totalParameters: number; layerCount: number;
    };
    // 784*256 + 256 + 256*10 + 10 = 203,530, plus LayerNorm's 512.
    expect(out.totalParameters).toBeGreaterThan(200_000);
    expect(out.layerCount).toBe(7);
  });

  it('explains itself when the file holds no model', async () => {
    const path = await write('train.py', 'def main():\n    print("training")\n');
    await expect(loadModelFile(path)).rejects.toThrow(/No PyTorch model found/);
  });

  it('points a malformed .json at the .py path instead of guessing', async () => {
    const path = await write('broken.json', '{ not json');
    await expect(loadModelFile(path)).rejects.toThrow(/give it a \.py extension/);
  });
});

describe('lint_model', () => {
  const lint = TOOLS.find(t => t.name === 'lint_model')!;
  const ctx = { modelPath: 'x.json' };

  it('runs offline on a graph parsed from source and catches the planted bug', async () => {
    const model = await loadModelFile(await write('net.py', MLP));
    const out = await lint.handler({}, model, ctx) as {
      clean: boolean; counts: Record<string, number>;
      findings: Array<{ rule: string; severity: string }>;
    };
    // Dropout at 0.9 is the rule this fixture exists to trip.
    expect(out.findings.map(f => f.rule)).toContain('high-dropout');
    expect(out.clean).toBe(false);
  });

  it('filters by severity and says that it filtered', async () => {
    const model = await loadModelFile(await write('net.py', MLP));
    const all = await lint.handler({}, model, ctx) as { findings: unknown[] };
    const blocks = await lint.handler({ severity: 'block' }, model, ctx) as {
      findings: Array<{ severity: string }>; reportedSeverityFloor: string; totalBeforeFilter: number;
    };
    expect(blocks.findings.every(f => f.severity === 'block')).toBe(true);
    expect(blocks.reportedSeverityFloor).toBe('block');
    expect(blocks.totalBeforeFilter).toBe(all.findings.length);
  });

  it('reports counts on a hand-built graph too', async () => {
    const out = await lint.handler({}, makeModel(), ctx) as {
      counts: { block: number; warn: number; info: number };
    };
    expect(out.counts).toEqual(expect.objectContaining({
      block: expect.any(Number), warn: expect.any(Number), info: expect.any(Number),
    }));
  });
});

describe('save_model', () => {
  const save = WRITE_TOOLS.find(t => t.name === 'save_model')!;

  it('refuses to write JSON over a .py file', async () => {
    const path = await write('net.py', MLP);
    await expect(
      save.handler({ path }, makeModel(), { modelPath: path }),
    ).rejects.toThrow(/refusing to write JSON over/);
    // And the source is untouched.
    const { readFile } = await import('node:fs/promises');
    expect(await readFile(path, 'utf-8')).toBe(MLP);
  });

  it('still writes a .neurarch.json target', async () => {
    const path = join(dir, 'out.neurarch.json');
    const out = await save.handler({ path }, makeModel(), { modelPath: path }) as { ok: boolean };
    expect(out.ok).toBe(true);
    expect((await loadModelFile(path)).name).toBe('tiny-transformer');
  });
});
