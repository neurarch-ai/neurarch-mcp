/**
 * The verifier bundle is compiled output from another repository and its
 * declaration file is written by hand. This is what stands between a
 * regenerated bundle whose exports moved and check_design throwing at a user.
 */
import { describe, it, expect } from 'vitest';
import * as verifier from './verifier.bundle.mjs';
import { makeModel } from '../test/fixtures.js';

describe('vendored verifier contract', () => {
  it('exports the functions this server declares', () => {
    expect(typeof verifier.checkDesign).toBe('function');
    expect(typeof verifier.provenanceFor).toBe('function');
    expect(typeof verifier.RULE_PROVENANCE).toBe('object');
  });

  it('returns a verdict synchronously, with no I/O', () => {
    const out = verifier.checkDesign(makeModel());
    expect(out).not.toBeInstanceOf(Promise);
    expect(['ok', 'warn', 'block', 'ask']).toContain(out.verdict);
  });

  it('walks the five stages in order, and stops where it says it stopped', () => {
    // The walk halts at the first blocked or needs-input stage rather than
    // reporting confident verdicts past a graph that cannot forward-pass, so
    // the contract is "a prefix of the five, in order", not "always five".
    const ORDER = ['preflight', 'data', 'train', 'evaluate', 'deploy'];
    const out = verifier.checkDesign(makeModel());
    const walked = out.stages.map(s => s.stage);
    expect(walked.length).toBeGreaterThan(0);
    expect(walked).toEqual(ORDER.slice(0, walked.length));
    if (walked.length < ORDER.length) {
      expect(out.stoppedAt).toBe(walked[walked.length - 1]);
    }
  });

  it('needs nothing from the outside world', async () => {
    const { readFile } = await import('node:fs/promises');
    const src = await readFile(new URL('./verifier.bundle.mjs', import.meta.url), 'utf-8');
    expect(src).not.toMatch(/^\s*import\s/m);
    expect(src).not.toMatch(/\brequire\(/);
    // The whole basis of the offline promise: a verifier that could open a
    // socket would make "no network calls by default" unverifiable.
    expect(src).not.toMatch(/\bfetch\(/);
  });

  it('has provenance only for rules with a published measurement', () => {
    const some = verifier.provenanceFor(['head-dim-divisibility', 'not-a-real-rule']);
    expect(some['head-dim-divisibility'].source).toMatch(/^https:\/\//);
    expect(some['not-a-real-rule']).toBeUndefined();
  });
});

describe('vendored verifier contract, 0.13 additions', () => {
  it('exports the ranker with its calibration and the outcome rule list', () => {
    expect(typeof verifier.rankCandidates).toBe('function');
    expect(typeof verifier.signalsFromCheck).toBe('function');
    expect(typeof verifier.normalizeGraphForVerification).toBe('function');
    expect(Array.isArray(verifier.OUTCOME_RULE_IDS)).toBe(true);
    expect(verifier.RANK_CALIBRATION.ordering.pairwiseAccuracy).toBeGreaterThan(0.4);
    expect(verifier.RANK_CALIBRATION.ordering.coverage).toBeLessThan(0.2);
  });
  it('signalsFromCheck returns null, not a clean default, for an unreadable verdict', () => {
    expect(verifier.signalsFromCheck('x', { nonsense: true }, [], [])).toBeNull();
    const sig = verifier.signalsFromCheck('x', verifier.checkDesign(makeModel()), [], []);
    expect(sig?.id).toBe('x');
  });
});
