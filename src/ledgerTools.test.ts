/**
 * The two tools that reach the network, driven against a stub of it.
 *
 * Three properties are worth more than the rest and each has a test that fails
 * if it stops holding:
 *
 *  1. `history` sends the fingerprint and nothing else. The graph is the thing
 *     someone is trusting this server with, and the ledger question can be
 *     asked with an 8-character hash, so it is.
 *  2. Without a key, `history` says the ledger was not read. An empty array
 *     would read as "never trained", which is a different and much more
 *     expensive answer to get wrong.
 *  3. `plan` returns the server's `text` byte for byte. It is the same card the
 *     CI bot posts; an agent and a reviewer disagreeing about what it said
 *     would be this feature failing at its only job.
 */
import { describe, it, expect, beforeEach, afterEach, beforeAll, afterAll } from 'vitest';
import { createServer, type Server } from 'node:http';
import { AddressInfo } from 'node:net';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { TOOLS, type ToolContext } from './tools.js';
import { LEDGER_TOOLS, historyTool, planTool } from './ledgerTools.js';
import { DEFAULT_API } from './lib/neurarchApi.js';
import { structuralFingerprint } from './lib/corpusReport.js';
import { makeModel } from './test/fixtures.js';
import { clearModelCache } from './models.js';
import type { ModelArchitecture } from './lib/types.js';

interface Seen { method: string; url: string; auth?: string; contentType?: string; body: any }

let server: Server;
let base: string;
let seen: Seen[] = [];
/** What the stub answers next, so a test can stage an empty ledger or a 401. */
let reply: { status: number; body: unknown } = { status: 200, body: {} };

const PLAN_TEXT = `Plan: MnistNet   10 layers · 20.6K params · fingerprint 7d154c61
  shape     [1,28,28] -> [10]
  will run  yes (0 blockers, 0 warnings)
  history   Last time this structure trained here: valAcc 0.9858, 3 epochs, 1m, 2026-09-04 (+1 earlier)`;

const ROWS = [
  { at: '2026-09-04T02:51:12.448+00:00', metric: 'valAcc', value: 0.9865, epochs: 3, wallSec: 34, estCostUsd: 0.0075, source: 'via=v1/train pr=2 kind=bot job=job_a' },
  { at: '2026-08-30T09:00:00.000+00:00', metric: 'valAcc', value: 0.981, epochs: 3, wallSec: 210, estCostUsd: 1.5, source: 'via=v1/train repo=neurarch-ai/model-ci-example pr=8' },
];

beforeAll(async () => {
  server = createServer((req, res) => {
    let raw = '';
    req.on('data', d => { raw += d; });
    req.on('end', () => {
      seen.push({
        method: req.method ?? '',
        url: req.url ?? '',
        auth: req.headers.authorization,
        contentType: req.headers['content-type'],
        body: raw ? JSON.parse(raw) : null,
      });
      res.writeHead(reply.status, { 'content-type': 'application/json' });
      res.end(JSON.stringify(reply.body));
    });
  });
  await new Promise<void>(r => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
afterAll(async () => { await new Promise<void>(r => { server.close(() => r()); }); });

let dir: string;
let prevApi: string | undefined;
let prevKey: string | undefined;
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'nx-ledger-'));
  clearModelCache();
  seen = [];
  prevApi = process.env.NEURARCH_API;
  prevKey = process.env.NEURARCH_API_KEY;
  process.env.NEURARCH_API = base;
  delete process.env.NEURARCH_API_KEY;
});
afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
  if (prevApi === undefined) delete process.env.NEURARCH_API; else process.env.NEURARCH_API = prevApi;
  if (prevKey === undefined) delete process.env.NEURARCH_API_KEY; else process.env.NEURARCH_API_KEY = prevKey;
});

function call(name: string, args: Record<string, unknown>, ctx: Partial<ToolContext> = {}, model?: ModelArchitecture): any {
  const tool = LEDGER_TOOLS.find(t => t.name === name);
  if (!tool) throw new Error(`tool ${name} not registered`);
  return tool.handler(args, (model ?? makeModel()) as ModelArchitecture, { modelPath: join(dir, 'model.neurarch.json'), ...ctx });
}

describe('registry', () => {
  it('registers both tools, open-world, with plan listed first', () => {
    expect(LEDGER_TOOLS.map(t => t.name)).toEqual(['plan', 'history']);
    for (const t of LEDGER_TOOLS) {
      expect(TOOLS.includes(t)).toBe(true);
      expect(t.annotations?.openWorldHint).toBe(true);
    }
    // Only history can answer without a graph, and only because a fingerprint
    // is a graph's name. plan must never be given that licence: it sends one.
    expect(historyTool.modelOptional).toBe(true);
    expect(planTool.modelOptional).toBeUndefined();
  });

  it('defaults to www, because the apex redirect drops the Authorization header', () => {
    // A 307 to www strips the bearer token, and the answer that comes back is
    // an empty ledger, which is indistinguishable from "nothing trained".
    expect(DEFAULT_API).toBe('https://www.neurarch.com');
    expect(new URL(DEFAULT_API).hostname).not.toBe('neurarch.com');
  });
});

describe('history without a key', () => {
  it('says the ledger was not read, and opens no socket at all', async () => {
    const r = await call('history', {});
    expect(seen).toHaveLength(0);
    expect(r.ledger).toBe('no-key');
    // Not [] and not 0: an unread ledger is not an empty one.
    expect(r.rows).toBeNull();
    expect(r.runs).toBeNull();
    expect(r.summary).toMatch(/per organisation/);
    expect(r.summary).toMatch(/not the same as "never trained"/);
    expect(r.summary).toMatch(/NEURARCH_API_KEY/);
    expect(r.summary).toMatch(/neurarch\.com\/developer/);
  });

  it('still reports the fingerprint, so the caller can ask for a key and retry', async () => {
    const model = makeModel();
    const r = await call('history', {}, {}, model);
    expect(r.fingerprint).toBe(structuralFingerprint(model));
    expect(r.fingerprintFrom).toBe('model');
  });
});

describe('history with a key', () => {
  beforeEach(() => { process.env.NEURARCH_API_KEY = 'nrk_test_key'; });

  it('sends the fingerprint and the bearer token, and nothing else', async () => {
    reply = { status: 200, body: { fingerprint: 'deadbeef', rows: ROWS } };
    const model = makeModel();
    await call('history', {}, {}, model);
    expect(seen).toHaveLength(1);
    const [req] = seen;
    expect(req.method).toBe('GET');
    expect(req.auth).toBe('Bearer nrk_test_key');
    expect(req.url).toBe(`/api/v1/history?fingerprint=${structuralFingerprint(model)}`);
    // The graph is the thing being trusted to this server. It must not be on
    // the wire for a question a hash can ask.
    expect(req.body).toBeNull();
    for (const c of model.components) expect(req.url).not.toContain(c.name);
  });

  it('turns rows into a sentence a person would say, plus a table', async () => {
    reply = { status: 200, body: { fingerprint: 'x', rows: ROWS } };
    const r = await call('history', { fingerprint: '7D154C61' });
    expect(r.fingerprint).toBe('7d154c61');
    expect(r.fingerprintFrom).toBe('argument');
    expect(r.ledger).toBe('read');
    expect(r.runs).toBe(2);
    expect(r.summary).toContain('98.65%');
    expect(r.summary).toContain('3 epochs');
    expect(r.summary).toContain('34s');
    expect(r.summary).toContain('under a cent');
    expect(r.summary).toContain('2 runs');
    expect(r.table.split('\n')).toHaveLength(3);
    expect(r.table).toMatch(/^when\s+result\s+epochs\s+wall\s+cost\s+source$/m);
    expect(r.rows).toEqual(ROWS);
  });

  it('an empty ledger is reported as empty, and as "not recorded here"', async () => {
    reply = { status: 200, body: { fingerprint: 'x', rows: [] } };
    const r = await call('history', {});
    expect(r.ledger).toBe('read');
    expect(r.runs).toBe(0);
    expect(r.rows).toEqual([]);
    // "nothing here" is the claim; "never trained" is the claim it must refuse
    // to make, and the sentence has to draw that line out loud.
    expect(r.summary).toMatch(/not recorded here/);
    expect(r.summary).toMatch(/rather than "never trained"/);
  });

  it('names the fix when the key is refused or the rate limit is hit', async () => {
    reply = { status: 401, body: { error: 'unknown key' } };
    await expect(call('history', {})).rejects.toThrow(/refused the key: unknown key.*NEURARCH_API_KEY/s);
    reply = { status: 429, body: { error: 'slow down' } };
    await expect(call('history', {})).rejects.toThrow(/rate-limited/);
  });

  it('rejects a malformed fingerprint before it reaches the wire', async () => {
    await expect(call('history', { fingerprint: 'nope' })).rejects.toThrow(/not a fingerprint/);
    await expect(call('history', { fingerprint: '7d154c6' })).rejects.toThrow(/8 hex characters/);
    expect(seen).toHaveLength(0);
  });

  it('asks for a subject rather than guessing when the server holds no model', async () => {
    const tool = LEDGER_TOOLS.find(t => t.name === 'history')!;
    await expect(tool.handler({}, undefined as never, { modelPath: '' })).rejects.toThrow(/name a structure/);
    expect(seen).toHaveLength(0);
  });
});

describe('plan', () => {
  const okReply = { status: 200, body: { text: PLAN_TEXT, markdown: '**card**', plan: { model: { fingerprint: '7d154c61' }, policy: { applied: true, violations: 2 }, history: { rows: ROWS } } } };

  it('posts the graph, pins share to false, and returns text byte for byte', async () => {
    reply = okReply;
    const model = makeModel();
    const r = await call('plan', {}, {}, model);
    expect(seen).toHaveLength(1);
    const [req] = seen;
    expect(req.method).toBe('POST');
    expect(req.url).toBe('/api/v1/plan');
    expect(req.contentType).toBe('application/json');
    expect(req.body.model.components).toHaveLength(model.components.length);
    // An agent must not be able to publish someone's design.
    expect(req.body.share).toBe(false);
    expect(req.body.source.kind).toBe('mcp');
    expect(req.body.source.tool).toMatch(/^neurarch-mcp\/\d+\.\d+\.\d+/);
    expect(r.text).toBe(PLAN_TEXT);
    expect(r.markdown).toBe('**card**');
    expect(r.sentTo).toBe(`${base}/api/v1/plan`);
  });

  it('sends no Authorization header when there is no key, and says the card has no history line', async () => {
    reply = okReply;
    const r = await call('plan', {});
    expect(seen[0].auth).toBeUndefined();
    expect(r.history.included).toBe(false);
    expect(r.history.runs).toBeNull();
    expect(r.history.note).toMatch(/per organisation/);
  });

  it('sends the key when there is one, and reports how many runs the card carried', async () => {
    process.env.NEURARCH_API_KEY = 'nrk_test_key';
    reply = okReply;
    const r = await call('plan', {});
    expect(seen[0].auth).toBe('Bearer nrk_test_key');
    expect(r.history).toEqual({ included: true, runs: 2 });
  });

  it('reads the repository .neurarch.yml, merges the model\'s own lines, and says it did', async () => {
    reply = okReply;
    await mkdir(join(dir, 'models'), { recursive: true });
    const modelFile = join(dir, 'models', 'small_cnn.py');
    await writeFile(modelFile, '# model\n');
    await writeFile(join(dir, '.neurarch.yml'),
      'policy:\n  forbid_types: [softmax]\n\nmodels:\n  - path: models/small_cnn.py:SmallCNN\n    input: 1,3,32,32\n    policy:\n      max_params: 500K\n');
    const r = await call('plan', {}, { currentPath: modelFile });
    expect(seen[0].body.policy).toEqual({ forbid_types: ['softmax'], max_params: '500K' });
    expect(r.policy.source).toBe('file');
    expect(r.policy.path).toBe(join(dir, '.neurarch.yml'));
    expect(r.policy.matchedModel).toBe('models/small_cnn.py:SmallCNN');
    expect(r.policy.lines.sort()).toEqual(['forbid_types', 'max_params']);
    expect(r.policy.violations).toBe(2);
    expect(r.policy.note).toContain('.neurarch.yml');
  });

  it('sends no policy and says so when the repository has no config', async () => {
    reply = okReply;
    const r = await call('plan', {});
    expect(seen[0].body.policy).toBeUndefined();
    expect(r.policy.source).toBe('none');
    expect(r.policy.path).toBeNull();
    expect(r.policy.lines).toEqual([]);
    expect(r.policy.note).toMatch(/No \.neurarch\.yml found/);
  });

  it('an explicit policy replaces the file rather than merging with it', async () => {
    reply = okReply;
    await writeFile(join(dir, '.neurarch.yml'), 'policy:\n  forbid_types: [softmax]\n');
    const r = await call('plan', { policy: { max_layers: 8 } });
    expect(seen[0].body.policy).toEqual({ max_layers: 8 });
    expect(r.policy.source).toBe('argument');
    expect(r.policy.note).toMatch(/no \.neurarch\.yml was read/);
  });

  it('sends the base graph when base_path names one', async () => {
    reply = okReply;
    const basePath = join(dir, 'base.neurarch.json');
    await writeFile(basePath, JSON.stringify(makeModel()));
    const r = await call('plan', { base_path: basePath });
    expect(seen[0].body.base.components.length).toBeGreaterThan(0);
    expect(r.comparedWith).toBe(basePath);
  });

  it('refuses to plan while ignoring a .neurarch.yml it cannot read', async () => {
    reply = okReply;
    await writeFile(join(dir, '.neurarch.yml'), 'policy:\n  note: |\n    unreadable\n');
    await expect(call('plan', {})).rejects.toThrow(/could not be read/);
    expect(seen).toHaveLength(0);
  });

  it('reports an unreachable API as one sentence, not a stack trace', async () => {
    process.env.NEURARCH_API = 'http://127.0.0.1:1';
    await expect(call('plan', {})).rejects.toThrow(/could not be reached.*works offline/s);
  });
});
