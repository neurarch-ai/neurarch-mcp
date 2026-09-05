import { describe, it, expect } from 'vitest';
import {
  buildCorpusReport, reportingEnabled, DEFAULT_REPORT_URL,
  sendCorpusReport, flushCorpusReports, pendingCorpusReports,
} from './corpusReport.js';
import type { ModelArchitecture } from './types.js';

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

/** A graph the design rules actually fire on: 8 heads do not divide 100. */
const badHeads = {
  id: 'm', name: 'm',
  components: [
    { id: 'i', name: 'in', type: 'input', params: { shape: [16, 100] } },
    { id: 'a', name: 'attn', type: 'multiHeadAttention', params: { embedDim: 100, numHeads: 8 } },
    { id: 'o', name: 'out', type: 'output', params: {} },
  ],
  connections: [{ from: 'i', to: 'a' }, { from: 'a', to: 'o' }],
} as unknown as ModelArchitecture;

describe('buildCorpusReport', () => {
  it('hashes byte-identically to the app/server fingerprint (cross-repo lock vector)', () => {
    // fnv1a('input:1|linear:2|output:1|relu:1#e4') computed against the main
    // repo's lib/archFingerprint.ts. If this fails, the hash drifted and the
    // server will reject every row this package sends.
    const r = buildCorpusReport(model);
    expect(r.typeHistogram).toBe('input:1|linear:2|output:1|relu:1');
    expect(r.connectionCount).toBe(4);
    expect(r.fingerprint).toBe('951e5874');
  });

  it('grades with the rule ids every other channel uses, not the validator\'s', () => {
    // The point of the change this test pins: an `mcp` row and a `ci` row must
    // name the same rule for the same graph, or the two cannot be pooled.
    const r = buildCorpusReport(badHeads);
    const ids = r.findings.map(f => f.ruleId);
    expect(ids).toContain('head-dim-divisibility');
    expect(r.findings.find(f => f.ruleId === 'head-dim-divisibility')?.severity).toBe('block');
    // The validator's own vocabulary must not appear: those ids exist in no
    // other channel, and a row spelled in them is a row nobody can join.
    expect(ids).not.toContain('cycle');
    expect(ids).not.toContain('orphan');
  });

  it('carries only warn and block: info has no corpus severity to be', () => {
    for (const r of [buildCorpusReport(model), buildCorpusReport(badHeads)]) {
      for (const f of r.findings) expect(['warn', 'block']).toContain(f.severity);
    }
  });

  it('is a property of the graph, so the same graph gives the same row', () => {
    // Three tools send this row now. If it depended on which one called, the
    // server would be pooling three different things under one channel.
    expect(buildCorpusReport(model)).toEqual(buildCorpusReport(model));
  });

  it('carries ONLY structure and rule ids: no names, messages, params, or paths', () => {
    const r = buildCorpusReport(model);
    expect(Object.keys(r).sort()).toEqual(
      ['channel', 'connectionCount', 'findings', 'fingerprint', 'layerCount', 'typeHistogram'],
    );
    const wire = JSON.stringify(r);
    expect(wire).not.toContain('fc1');       // layer names
    expect(wire).not.toContain('act');       // names from findings
    expect(wire).not.toContain('divide');    // rule messages
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
  it('posts to a host that does not redirect', () => {
    // The row carries no Authorization header, so a redirect would not lose
    // anything, but it would burn a round trip inside a 5s fire-and-forget
    // budget. Same apex-307 trap as check_design; same answer.
    expect(DEFAULT_REPORT_URL).toBe('https://www.neurarch.com/api/v1/report');
    expect(new URL(DEFAULT_REPORT_URL).hostname).not.toBe('neurarch.com');
  });
});

/**
 * The row that was sent and never arrived.
 *
 * `sendCorpusReport` is fire-and-forget, which is right at the call site: a
 * corpus row must never slow or fail a verdict. It was also the whole story,
 * so a process that exited before the socket flushed dropped the row with no
 * error anywhere. Under a chat client the server is long-lived and this never
 * shows; under an agent that spawns it, asks once and closes the pipe, it
 * always does. A channel that reports nothing looks the same whether it is
 * broken or unused, which is why this is worth a test rather than a comment.
 */
describe('flushCorpusReports', () => {
  const payload = () => buildCorpusReport(model);

  it('resolves immediately when nothing is in flight', async () => {
    await flushCorpusReports(50);
    expect(pendingCorpusReports()).toBe(0);
  });

  it('waits for a send that is still in the socket', async () => {
    const realFetch = globalThis.fetch;
    let release!: () => void;
    const landed = new Promise<void>(r => { release = r; });
    let arrived = false;
    globalThis.fetch = (async () => {
      await landed;
      arrived = true;
      return new Response('', { status: 201 });
    }) as typeof fetch;
    try {
      sendCorpusReport(payload());
      expect(pendingCorpusReports()).toBe(1);
      // The exit path: the process is going away and the POST has not landed.
      setTimeout(release, 10);
      await flushCorpusReports(2000);
      expect(arrived, 'flush returned before the row was sent').toBe(true);
      expect(pendingCorpusReports()).toBe(0);
    } finally {
      globalThis.fetch = realFetch;
    }
  });

  it('never throws when the endpoint is down', async () => {
    const realFetch = globalThis.fetch;
    globalThis.fetch = (() => Promise.reject(new Error('ECONNREFUSED'))) as typeof fetch;
    try {
      sendCorpusReport(payload());
      await expect(flushCorpusReports(500)).resolves.toBeUndefined();
      expect(pendingCorpusReports()).toBe(0);
    } finally {
      globalThis.fetch = realFetch;
    }
  });

  // Last on purpose. Its stub never settles and ignores the abort signal, so
  // its row stays in the pending set for the rest of the file. A real fetch
  // cannot do that (the 5s AbortController settles it either way), but a later
  // test would wait on it and read its own flush as slow.
  it('gives up rather than holding a shutdown open', async () => {
    // The same rule that says a row may not fail a tool call says it may not
    // hang an exit. A reporting endpoint that never answers costs the deadline
    // and nothing more.
    const realFetch = globalThis.fetch;
    globalThis.fetch = (() => new Promise(() => { /* never settles */ })) as typeof fetch;
    try {
      sendCorpusReport(payload());
      const t0 = Date.now();
      await flushCorpusReports(80);
      expect(Date.now() - t0).toBeLessThan(1000);
    } finally {
      globalThis.fetch = realFetch;
    }
  });
});
