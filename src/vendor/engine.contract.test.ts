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
