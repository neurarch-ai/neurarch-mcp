import { describe, it, expect } from 'vitest';
import { isUnresolved, parseQuality } from './parseQuality.js';
import { graphFromPyTorchSource } from '../vendor/engine.bundle.mjs';
import { makeModel } from '../test/fixtures.js';
import { TOOLS } from '../tools.js';

describe('isUnresolved', () => {
  it('flags source text and leaves numbers and enums alone', () => {
    for (const v of ['config.hidden_size', 'dims[0]', 'self.inplanes', '1 if aa else 2', 'd_model', 'n * 4']) expect(isUnresolved(v), v).toBe(true);
    for (const v of [256, '256', '0.1', 'relu', 'same', true, undefined, [3, 3], '']) expect(isUnresolved(v), String(v)).toBe(false);
  });
});

describe('parseQuality', () => {
  it('grades the fixture full and a config-driven source partial', () => {
    expect(parseQuality(makeModel()).grade).toBe('full');
    const src = `import torch.nn as nn
class M(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size, 10)
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(self.emb(x)))))
`;
    const g = graphFromPyTorchSource(src, 'm')!;
    const q = parseQuality(g);
    expect(q.grade).toBe('partial');
    expect(q.unresolvedParams).toBeGreaterThan(0);
    expect(q.note).toMatch(/neurarch-trace/);
    // And lint_model holds back the dimension rules on those layers, saying so.
    const lint = TOOLS.find(t => t.name === 'lint_model')!.handler({}, g, { modelPath: '' }) as any;
    expect(lint.parseQuality.grade).toBe('partial');
    expect(lint.findings.some((f: any) => f.rule === 'invalid-output-shape')).toBe(false);
    const describe_ = TOOLS.find(t => t.name === 'describe_architecture')!.handler({}, g, { modelPath: '' }) as any;
    expect(describe_.parseQuality.grade).toBe('partial');
  });
  it('grades a one-layer graph thin', () => {
    const g = graphFromPyTorchSource('import torch.nn as nn\nclass M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.head = nn.Linear(8, 2)\n    def forward(self, x):\n        return self.head(x)\n', 'm')!;
    expect(parseQuality(g).grade).toBe('thin');
  });
  it('keeps the planted head-dim block on tiny-vit, whose dims are numbers', async () => {
    const { readFile } = await import('node:fs/promises');
    const g = graphFromPyTorchSource(await readFile(new URL('../../examples/tiny-vit.py', import.meta.url), 'utf-8'), 'vit')!;
    const lint = TOOLS.find(t => t.name === 'lint_model')!.handler({}, g, { modelPath: '' }) as any;
    expect(lint.counts.block).toBe(1);
  });
});
