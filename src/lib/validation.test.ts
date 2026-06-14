import { describe, it, expect } from 'vitest';
import { validateModel } from './validation.js';
import { makeModel, makeCyclicModel, makeBrokenModel } from '../test/fixtures.js';

describe('validateModel', () => {
  it('passes a well-formed model with no findings', () => {
    const report = validateModel(makeModel());
    expect(report.ok).toBe(true);
    expect(report.totals.errors).toBe(0);
    expect(report.findings).toEqual([]);
  });

  it('detects a cycle and reports it as an error', () => {
    const report = validateModel(makeCyclicModel());
    expect(report.ok).toBe(false);
    const cycles = report.findings.filter(f => f.rule === 'cycle');
    expect(cycles.length).toBeGreaterThanOrEqual(1);
    expect(cycles[0].severity).toBe('error');
  });

  it('reports the same cycle only once', () => {
    const report = validateModel(makeCyclicModel());
    expect(report.findings.filter(f => f.rule === 'cycle')).toHaveLength(1);
  });

  it('flags dangling connections, duplicate names, and orphans', () => {
    const report = validateModel(makeBrokenModel());
    const rules = report.findings.map(f => f.rule);
    expect(rules).toContain('dangling-connection');
    expect(rules).toContain('duplicate-name');
    expect(rules).toContain('orphan');
    // dangling edge is an error, so the model is not ok
    expect(report.ok).toBe(false);
  });

  it('flags a fully disconnected node as an orphan', () => {
    const report = validateModel(makeBrokenModel());
    const lonely = report.findings.find(
      f => f.rule === 'orphan' && f.componentNames?.includes('lonely'),
    );
    expect(lonely).toBeDefined();
    expect(lonely?.severity).toBe('warn');
  });

  it('does not flag orphans on a single-component model', () => {
    const report = validateModel({
      id: 'm', name: 'solo',
      components: [{ id: 'a', type: 'input', name: 'a', position: { x: 0, y: 0 }, params: {}, inputs: [], outputs: [] }],
      connections: [],
    });
    expect(report.findings.filter(f => f.rule === 'orphan')).toHaveLength(0);
  });
});
