/**
 * vendorContract — the vendored verifier is the one this checkout's sibling builds.
 *
 * `src/vendor/verifier.bundle.mjs` is built in the neurarch repo and copied here
 * by hand. Nothing in this repo checked it, and on 2026-09-04 it was found four
 * commits behind: findings had lost their stable `ruleId` (so `provenanceForRules`
 * could not be joined to a finding at all) and the pre-flight stage had stopped
 * carrying the memory footprint. Thirty test files and 265 assertions passed
 * throughout, because every one of them asserted on fields that had not moved.
 *
 * These assertions are the fields an agent reads and cannot get anywhere else.
 * They are deliberately about the SHAPE the bundle returns rather than about
 * verifier behaviour, which is tested in the repo that builds it: this file's
 * whole job is to fail when the copy is stale.
 *
 * Proven to fail on the bundle it replaced, which is the only reason to trust
 * its green (see docs/DISTRIBUTION.md, "refreshing the vendored bundles").
 */
import { describe, expect, it } from 'vitest';
import { checkDesign, provenanceForRules } from './checkDesign.js';
import type { ModelArchitecture } from './types.js';

/** 12 channels over 5 heads: head-dim-divisibility blocks, by construction. */
const badHeads: ModelArchitecture = {
  id: 'vendor-contract', name: 'vendor contract', components: [
    { id: 'a', type: 'input', name: 'in', params: { shape: [16, 12] }, position: { x: 0, y: 0 } },
    { id: 'b', type: 'attention', name: 'attn', params: { embedDim: 12, numHeads: 5 }, position: { x: 0, y: 1 } },
  ],
  connections: [{ id: 'c', source: 'a', target: 'b' }],
} as unknown as ModelArchitecture;

describe('vendored verifier contract', () => {
  it('gives every finding a stable rule id that provenance can be joined on', () => {
    const r = checkDesign(badHeads);
    expect('error' in r).toBe(false);
    const check = r as Exclude<typeof r, { error: string }>;
    const blocking = check.findings.filter(f => f.severity === 'block');
    expect(blocking.length).toBeGreaterThan(0);
    for (const f of blocking) {
      // Not `f.id`: that one bakes in the component name, so it is unique per
      // graph and joins to nothing.
      expect(typeof f.ruleId, `finding "${f.title}" carries no ruleId`).toBe('string');
      expect(f.ruleId).not.toMatch(/vendor-contract|attn/);
    }
    const ids = blocking.map(f => f.ruleId!).filter(Boolean);
    expect(ids).toContain('head-dim-divisibility');
    // The join the ruleId exists for.
    expect(Object.keys(provenanceForRules(ids)).length).toBeGreaterThan(0);
  });

  it('reports the memory a run would need, not just the time', () => {
    const ok: ModelArchitecture = {
      id: 'm', name: 'm', components: [
        { id: 'a', type: 'input', name: 'in', params: { shape: [1, 28, 28] }, position: { x: 0, y: 0 } },
        { id: 'b', type: 'flatten', name: 'flat', params: {}, position: { x: 0, y: 1 } },
        { id: 'c', type: 'linear', name: 'fc', params: { outFeatures: 10 }, position: { x: 0, y: 2 } },
      ],
      connections: [
        { id: 'c1', source: 'a', target: 'b' },
        { id: 'c2', source: 'b', target: 'c' },
      ],
    } as unknown as ModelArchitecture;
    const r = checkDesign(ok);
    expect('error' in r).toBe(false);
    const check = r as Exclude<typeof r, { error: string }>;
    const pre = check.stages.find(s => s.stage === 'preflight');
    expect(pre, 'no preflight stage').toBeTruthy();
    // A footprint with no provenance is a claim; `samplesAssumed` is the flag
    // that says whether the sample count behind it was read or guessed.
    for (const key of ['weightBytes', 'trainFootprintBytes', 'samplesAssumed']) {
      expect(key in pre!.data, `preflight stage data has no ${key}`).toBe(true);
    }
  });
});
