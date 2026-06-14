import { describe, it, expect } from 'vitest';
import { readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadModelFile } from './loader.js';
import { validateModel } from './lib/validation.js';

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
});
