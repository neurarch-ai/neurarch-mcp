import { describe, it, expect } from 'vitest';
import { readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadModelFile } from './loader.js';
import { validateModel } from './lib/validation.js';
import { lintModelGraph } from './vendor/engine.bundle.mjs';

// examples/ lives at the repo root, two levels up from this file (src/).
const EXAMPLES_DIR = join(fileURLToPath(new URL('.', import.meta.url)), '..', 'examples');

describe('shipped example models', () => {
  it('loads and validates every examples/*.neurarch.json with zero errors', async () => {
    const files = (await readdir(EXAMPLES_DIR)).filter(f => f.endsWith('.neurarch.json'));
    expect(files.length).toBeGreaterThan(0);
    for (const file of files) {
      const model = await loadModelFile(join(EXAMPLES_DIR, file));
      const report = validateModel(model);
      expect(report.totals.errors, `${file} should have no validation errors`).toBe(0);
      expect(report.totals.warnings, `${file} should have no validation warnings`).toBe(0);
    }
  });

  it('parses every examples/*.py, and the ViT still carries its planted blocker', async () => {
    const files = (await readdir(EXAMPLES_DIR)).filter(f => f.endsWith('.py'));
    expect(files.length).toBeGreaterThan(0);
    for (const file of files) {
      const model = await loadModelFile(join(EXAMPLES_DIR, file));
      expect(model.components.length, `${file} should parse into layers`).toBeGreaterThan(2);
    }
    // The example exists to demonstrate lint_model finding something real. If a
    // registry change ever stops this from firing, the example is a dud and the
    // README is telling users to expect an answer they will not get.
    const vit = await loadModelFile(join(EXAMPLES_DIR, 'tiny-vit.py'));
    const rules = lintModelGraph(vit);
    expect(rules.find(f => f.rule === 'head-dim-divisibility')?.severity).toBe('block');
  });
});
