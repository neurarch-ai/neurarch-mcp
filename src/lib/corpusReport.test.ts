import { describe, it, expect } from 'vitest';
import { buildCorpusReport, reportingEnabled } from './corpusReport.js';
import type { ModelArchitecture } from './types.js';
import type { ValidationReport } from './validation.js';

const model = {
  id: 'm', name: 'm',
  components: [
    { id: 'i', name: 'in', type: 'input', params: {} },
    { id: 'l1', name: 'fc1', type: 'linear', params: { inFeatures: 8, outFeatures: 8 } },
    { id: 'r', name: 'act', type: 'relu', params: {} },
    { id: 'l2', name: 'fc2', type: 'linear', params: { inFeatures: 8, outFeatures: 2 } },
    { id: 'o', name: 'out', type: 'output', params: {} },
  ],
  connections: [
    { from: 'i', to: 'l1' }, { from: 'l1', to: 'r' }, { from: 'r', to: 'l2' }, { from: 'l2', to: 'o' },
  ],
} as unknown as ModelArchitecture;

const validation: ValidationReport = {
  ok: false,
  findings: [
    { rule: 'cycle', severity: 'error', message: 'A cycle exists through fc1.' },
    { rule: 'orphan', severity: 'warn', message: 'act has no downstream.', componentNames: ['act'] },
  ],
  totals: { errors: 1, warnings: 1 },
};

describe('buildCorpusReport', () => {
  it('hashes byte-identically to the app/server fingerprint (cross-repo lock vector)', () => {
    // fnv1a('input:1|linear:2|output:1|relu:1#e4') computed against the main
    // repo's lib/archFingerprint.ts. If this fails, the hash drifted and the
    // server will reject every row this package sends.
    const r = buildCorpusReport(model, validation);
    expect(r.typeHistogram).toBe('input:1|linear:2|output:1|relu:1');
    expect(r.connectionCount).toBe(4);
    expect(r.fingerprint).toBe('951e5874');
  });

  it('maps validator severities into the corpus vocabulary (error -> block)', () => {
    const r = buildCorpusReport(model, validation);
    expect(r.findings).toEqual([
      { ruleId: 'cycle', severity: 'block' },
      { ruleId: 'orphan', severity: 'warn' },
    ]);
  });

  it('carries ONLY structure and rule ids: no names, messages, params, or paths', () => {
    const r = buildCorpusReport(model, validation);
    expect(Object.keys(r).sort()).toEqual(
      ['channel', 'connectionCount', 'findings', 'fingerprint', 'layerCount', 'typeHistogram'],
    );
    const wire = JSON.stringify(r);
    expect(wire).not.toContain('fc1');       // layer names
    expect(wire).not.toContain('act');       // names from findings
    expect(wire).not.toContain('cycle exists'); // messages
    expect(wire).not.toContain('inFeatures');   // params
  });

  it('is off unless NEURARCH_REPORT=1', () => {
    const prev = process.env.NEURARCH_REPORT;
    try {
      delete process.env.NEURARCH_REPORT;
      expect(reportingEnabled()).toBe(false);
      process.env.NEURARCH_REPORT = '0';
      expect(reportingEnabled()).toBe(false);
      process.env.NEURARCH_REPORT = '1';
      expect(reportingEnabled()).toBe(true);
    } finally {
      if (prev === undefined) delete process.env.NEURARCH_REPORT;
      else process.env.NEURARCH_REPORT = prev;
    }
  });
});
