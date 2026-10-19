/**
 * One factory, two protocol eras.
 *
 * The 2026-07-28 revision drops initialize and sessions and turns every
 * server-to-client request into an input_required result the client answers
 * by retrying the call. A 2025-era client still opens with initialize. Both
 * reach the same createMcpServer factory, over stdio (serveStdio) and over
 * HTTP (createMcpHandler), so the same call must give the same answer on
 * either. This starts the real HTTP server on a loopback port and drives it
 * once with a legacy client and once with a client pinned to 2026-07-28, so
 * the era routing in http.ts is what is under test, not a stand-in.
 *
 * The elicitation case is the one where the eras genuinely differ on the
 * wire: a 2025 connection gets elicitation/create pushed mid-call, a 2026
 * request gets input_required and a retry. The answer the tool reports has to
 * be the same either way.
 */
import { describe, it, expect, beforeAll, afterAll, afterEach, vi } from 'vitest';
import { readFile } from 'node:fs/promises';
import { Client, StreamableHTTPClientTransport } from '@modelcontextprotocol/client';
import { MODERN_PROTOCOL } from './server.js';
import { startHttpServer, stopHttpServer } from './http.js';
import { makeModel } from './test/fixtures.js';
import { feedFetch } from './test/feedStub.js';
import type { ModelArchitecture } from './lib/types.js';

type Era = 'legacy' | 'modern';

const PORT = 18947;
let current: ModelArchitecture = makeModel();
const clients: Client[] = [];

beforeAll(() => {
  startHttpServer({ getModel: () => current, ctx: { modelPath: '' }, writeEnabled: false, version: '0.0.0-test', host: '127.0.0.1', port: PORT, toolSet: 'full' });
});
afterEach(async () => {
  await Promise.all(clients.splice(0).map(c => c.close().catch(() => undefined)));
  current = makeModel();
  vi.unstubAllGlobals();
});

/**
 * The client and the server share this process, so the feed stub may only
 * answer www.neurarch.com: the client's own requests to the loopback server
 * have to reach it.
 */
const realFetch = globalThis.fetch;
function stubFeed(f: typeof fetch) {
  vi.stubGlobal('fetch', ((u: string | URL | Request, init?: RequestInit) =>
    (String(u instanceof Request ? u.url : u).startsWith('https://www.neurarch.com') ? f(u, init) : realFetch(u, init))) as typeof fetch);
}
afterAll(async () => { await stopHttpServer(); });

async function connect(era: Era, opts: { model?: ModelArchitecture; elicitation?: boolean } = {}): Promise<Client> {
  if (opts.model) current = opts.model;
  const transport = new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${PORT}/mcp`));
  const client = new Client(
    { name: 'eras', version: '0' },
    {
      capabilities: opts.elicitation ? { elicitation: {} } : {},
      ...(era === 'modern' ? { versionNegotiation: { mode: { pin: MODERN_PROTOCOL } } } : {}),
    },
  );
  await client.connect(transport);
  clients.push(client);
  return client;
}

/** What a call answers, without the per-request envelope a 2026 result may carry. */
function answer(r: { content?: unknown; structuredContent?: unknown; isError?: unknown }) {
  return { content: r.content, structuredContent: r.structuredContent, isError: r.isError ?? false };
}

describe('both protocol eras from one factory', () => {
  it('negotiates the era each client asked for', async () => {
    expect((await connect('legacy')).getNegotiatedProtocolVersion?.() ?? '').not.toBe(MODERN_PROTOCOL);
    expect((await connect('modern')).getNegotiatedProtocolVersion?.()).toBe(MODERN_PROTOCOL);
  });

  it('lists the same tools, prompts and resources', async () => {
    const [a, b] = [await connect('legacy'), await connect('modern')];
    const names = async (c: Client) => ({
      tools: (await c.listTools()).tools.map(t => t.name),
      prompts: (await c.listPrompts()).prompts.map(p => p.name),
      resources: (await c.listResources()).resources.map(r => r.uri),
    });
    const [na, nb] = [await names(a), await names(b)];
    expect(na.tools.length).toBeGreaterThan(10);
    expect(nb).toEqual(na);
  });

  it('answers the same calls with the same results', async () => {
    const [a, b] = [await connect('legacy'), await connect('modern')];
    const calls = [
      { name: 'lint_model', arguments: {} },
      { name: 'describe_architecture', arguments: {} },
      { name: 'check_design', arguments: {} },
      { name: 'get_layer', arguments: { name: 'nope' } },
    ];
    for (const call of calls) {
      expect(answer(await b.callTool(call)), call.name).toEqual(answer(await a.callTool(call)));
    }
    const res = async (c: Client) => (await c.readResource({ uri: 'neurarch://rules/catalogue' })).contents;
    expect(await res(b)).toEqual(await res(a));
  });

  it('check_design asks the human on both eras and reports the same answer', async () => {
    const model = JSON.parse(await readFile(new URL('../examples/tiny-gpt.neurarch.json', import.meta.url), 'utf-8'));
    const outcomes: Record<Era, unknown> = { legacy: undefined, modern: undefined };
    for (const era of ['legacy', 'modern'] as const) {
      const client = await connect(era, { model, elicitation: true });
      let asked = 0;
      client.setRequestHandler('elicitation/create', async (req) => {
        asked++;
        const choices = ((req.params as any).requestedSchema as { properties: { choice: { enum: string[] } } }).properties.choice.enum;
        return { action: 'accept', content: { choice: choices[choices.length - 1] } };
      });
      const r = await client.callTool({ name: 'check_design', arguments: { ask_user: true } });
      expect(r.isError, era).toBeFalsy();
      expect(asked, era).toBe(1);
      outcomes[era] = (r.structuredContent as any).decision.answer;
    }
    expect(outcomes.modern).toEqual({ action: 'accept', value: 'synthetic' });
    expect(outcomes.modern).toEqual(outcomes.legacy);
  });

  it('a declined question is reported as declined on the 2026 era', async () => {
    const model = JSON.parse(await readFile(new URL('../examples/tiny-gpt.neurarch.json', import.meta.url), 'utf-8'));
    const client = await connect('modern', { model, elicitation: true });
    client.setRequestHandler('elicitation/create', async () => ({ action: 'decline' }));
    const r = await client.callTool({ name: 'check_design', arguments: { ask_user: true } });
    expect((r.structuredContent as any).decision.answer).toEqual({ action: 'decline' });
  });

  it('sends progress on the 2026 era too, in order', async () => {
    stubFeed(feedFetch());
    const c = await connect('modern');
    const seen: number[] = [];
    const r = await c.callTool({ name: 'model_changes', arguments: { hf_id: 'Qwen/Qwen3-8B' } }, { onprogress: (p) => { seen.push(p.progress); } });
    expect(r.isError).toBeFalsy();
    expect(seen).toEqual([0, 1, 2]);
  });

  it('a cancelled 2026 call aborts the fetch it was waiting on', async () => {
    const f = feedFetch({ delayMs: 5_000 });
    stubFeed(f);
    const c = await connect('modern');
    const ctrl = new AbortController();
    const pending = c.callTool({ name: 'model_changes', arguments: { hf_id: 'Qwen/Qwen3-8B' } }, { signal: ctrl.signal });
    await new Promise(r => setTimeout(r, 100));
    ctrl.abort('user pressed stop');
    await expect(pending).rejects.toThrow();
    await new Promise(r => setTimeout(r, 100));
    expect(f.urls.length).toBe(1);
  });
});
