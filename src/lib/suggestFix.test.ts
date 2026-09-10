/**
 * The vendored `suggestFixes` on a real file.
 *
 * The rules themselves are tested next to the engine (src/utils/suggestFix.test.ts
 * in the Neurarch repo); this is the end-to-end check that the bundle this
 * server ships actually produces them, which is the thing a refreshed vendor
 * bundle can silently break.
 */
import { describe, it, expect } from 'vitest';
import { readFile } from 'node:fs/promises';
import { suggestFixes } from './suggestFix.js';
import { graphFromPyTorchSource, lintModelGraph } from '../vendor/engine.bundle.mjs';

const VIT = new URL('../../examples/tiny-vit.py', import.meta.url);

describe('suggestFixes on tiny-vit.py', () => {
  it('fixes the head-dim crash on every line that shares the width, exactly', async () => {
    const src = await readFile(VIT, 'utf-8');
    const g = graphFromPyTorchSource(src, 'vit')!;
    const r = suggestFixes(g, lintModelGraph(g), src, 'tiny-vit.py');
    const hd = r.fixes.find(f => f.rule === 'head-dim-divisibility')!;
    expect(hd.confidence).toBe('exact');
    expect(hd.lines.length).toBe(7);
    expect(hd.diff).toMatch(/-        self\.attn = nn\.MultiheadAttention\(embed_dim=258, num_heads=8\)/);
    expect(hd.diff).toMatch(/\+        self\.attn = nn\.MultiheadAttention\(embed_dim=256, num_heads=8\)/);
    // The patched file has no 258 left and lints clean of the block.
    expect(r.patchedSource).not.toMatch(/258/);
    const g2 = graphFromPyTorchSource(r.patchedSource!, 'vit')!;
    expect(lintModelGraph(g2).some(f => f.rule === 'head-dim-divisibility')).toBe(false);
  });
  it('swaps two consecutive forward() statements for an ordering rule', async () => {
    const src = await readFile(VIT, 'utf-8');
    const g = graphFromPyTorchSource(src, 'vit')!;
    const r = suggestFixes(g, lintModelGraph(g).filter(f => f.rule === 'dropout-before-bn'), src, 'tiny-vit.py');
    expect(r.fixes[0].confidence).toBe('exact');
    expect(r.fixes[0].diff).toMatch(/^-        x = self\.(drop|norm2)\(x\)$/m);
    expect(r.fixes[0].diff).toMatch(/^\+        x = self\.(drop|norm2)\(x\)$/m);
    expect(r.patchedSource!.indexOf('self.norm2(x)')).toBeLessThan(r.patchedSource!.indexOf('self.drop(x)'));
  });
  it('labels insertions as proposals and shape findings as not fixable', async () => {
    const src = await readFile(VIT, 'utf-8');
    const g = graphFromPyTorchSource(src, 'vit')!;
    const r = suggestFixes(g, lintModelGraph(g), src, 'tiny-vit.py');
    expect(r.fixes.find(f => f.rule === 'attention-no-pe')?.confidence).toBe('proposal');
    expect(r.fixes.find(f => f.rule === 'consecutive-linear-no-activation')?.diff).toMatch(/\+        self\.proj_act = nn\.GELU\(\)/);
    expect(r.notFixable.map(n => n.rule)).toContain('invalid-output-shape');
  });
  it('every diff applies to the original independently (line numbers do not stack)', async () => {
    const src = await readFile(VIT, 'utf-8');
    const g = graphFromPyTorchSource(src, 'vit')!;
    const r = suggestFixes(g, lintModelGraph(g), src, 'tiny-vit.py');
    const lines = src.split('\n');
    for (const f of r.fixes) {
      for (const m of f.diff.matchAll(/^-(.*)$/gm)) {
        if (m[1].startsWith('--')) continue;
        expect(lines, `${f.rule}: removed line must exist in the original`).toContain(m[1]);
      }
    }
  });
});
