/**
 * The 0.20 protocol surface, over a real client: instructions, server title,
 * argument errors an agent can act on, `{ error }` results as isError,
 * resource links, completions, progress notifications and cancellation.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { readFile } from 'node:fs/promises';
import { Client } from '@modelcontextprotocol/client';
import { InMemoryTransport } from '@modelcontextprotocol/server';
import { createMcpServer, SERVER_INSTRUCTIONS, GPU_HINTS } from './server.js';
import { TOOLS } from './tools.js';
import { makeModel } from './test/fixtures.js';
import { clearChangeCache } from './changeTools.js';
import { feedFetch } from './test/feedStub.js';

async function connect(opts: { writeEnabled?: boolean } = {}): Promise<Client> {
  const model = makeModel();
  const server = createMcpServer({ getModel: () => model, ctx: { modelPath: '/tmp/nx/model.neurarch.json' }, writeEnabled: opts.writeEnabled ?? false, version: '0.0.0-test', toolSet: 'core' });
  const c = new Client({ name: 't', version: '0' }, { capabilities: {} });
  const [a, b] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(b), c.connect(a)]);
  return c;
}

const text = (r: unknown) => (((r as { content?: unknown }).content as Array<{ type: string; text?: string }>)[0]?.text ?? '');

afterEach(() => {
  vi.unstubAllGlobals();
  clearChangeCache();
});

describe('initialize', () => {
  it('sends instructions that name only real tools, and a title', async () => {
    const c = await connect();
    expect(c.getInstructions()).toBe(SERVER_INSTRUCTIONS);
    const names = new Set(TOOLS.map(t => t.name));
    for (const m of SERVER_INSTRUCTIONS.matchAll(/\b([a-z]+(?:_[a-z]+)+)\b/g)) expect(names.has(m[1]), m[1]).toBe(true);
    expect(SERVER_INSTRUCTIONS).not.toMatch(/\b(best|better|optimal|guarantee)\b/i);
    // The one mention of accuracy is the disclaimer.
    expect(SERVER_INSTRUCTIONS).toContain('No tool here predicts accuracy or model quality.');
    expect(c.getServerVersion()).toMatchObject({ name: 'neurarch-mcp', title: 'Neurarch' });
  });

  it('sends a PNG icon as a data URI, and it is the same bytes as icon.png', async () => {
    const { readFileSync } = await import('node:fs');
    const icons = (await connect()).getServerVersion()?.icons ?? [];
    expect(icons).toHaveLength(1);
    expect(icons[0].mimeType).toBe('image/png');
    const b64 = icons[0].src.replace(/^data:image\/png;base64,/, '');
    expect(Buffer.from(b64, 'base64').equals(readFileSync(new URL('../icon.png', import.meta.url)))).toBe(true);
  });

  it('lists every tool with a top-level title', async () => {
    const c = await connect({ writeEnabled: true });
    for (const t of (await c.listTools()).tools) expect(t.title, t.name).toEqual(expect.any(String));
  });
});

describe('argument errors', () => {
  it('names a missing required argument instead of crashing inside the handler', async () => {
    // Both of these answered "Cannot read properties of undefined" before 0.20.
    const c = await connect();
    for (const [tool, arg] of [['rank_designs', 'candidates'], ['load_architecture', 'id']] as const) {
      const r = await c.callTool({ name: tool, arguments: {} });
      expect(r.isError, tool).toBe(true);
      expect(text(r)).toContain(`missing required "${arg}"`);
      expect(text(r)).toContain(`neurarch://docs/${tool}`);
    }
  });

  it('names an unknown argument and lists the accepted ones', async () => {
    const r = await (await connect()).callTool({ name: 'lint_model', arguments: { severty: 'warn' } });
    expect(r.isError).toBe(true);
    expect(text(r)).toMatch(/unknown argument "severty".*Accepted: severity, response_format, model_path/s);
  });

  it('names a wrong type and an out-of-set value', async () => {
    const c = await connect();
    const t = await c.callTool({ name: 'size_training_job', arguments: { batch_size: 'eight' } });
    expect(text(t)).toContain('"batch_size" must be');
    const e = await c.callTool({ name: 'lint_model', arguments: { severity: 'fatal' } });
    expect(text(e)).toContain('"severity" must be one of "block", "warn", "info"');
  });

  it('suggests the tool an unknown name was probably reaching for', async () => {
    const r = await (await connect()).callTool({ name: 'check_serving', arguments: {} });
    expect(r.isError).toBe(true);
    expect(text(r)).toMatch(/Did you mean: .*check_serving_config/);
  });

  it('reports a handler\'s { error } as isError, and a missing layer with its nearest names', async () => {
    const c = await connect();
    const bad = await c.callTool({ name: 'find_layers', arguments: { namePattern: '(' } });
    expect(bad.isError).toBe(true);
    expect(text(bad)).toMatch(/^find_layers: Invalid regex/);
    const missing = await c.callTool({ name: 'get_layer', arguments: { name: 'final_nrom' } });
    expect(missing.isError).toBe(true);
    expect(text(missing)).toMatch(/Closest: final_norm/);
  });
});

describe('resource links and completions', () => {
  it('list_architectures links each entry to its neurarch://zoo resource', async () => {
    const r = await (await connect()).callTool({ name: 'list_architectures', arguments: { query: 'qwen' } });
    const links = (r.content as Array<{ type: string; uri?: string }>).filter(x => x.type === 'resource_link');
    expect(links.length).toBe((r.structuredContent as { architectures: unknown[] }).architectures.length);
    expect(links[0].uri).toMatch(/^neurarch:\/\/zoo\/qwen/);
  });

  it('completes zoo ids, tool names, rule ids and card ids', async () => {
    const c = await connect();
    const zoo = await c.complete({ ref: { type: 'ref/resource', uri: 'neurarch://zoo/{id}' }, argument: { name: 'id', value: 'qwen' } });
    expect(zoo.completion.values.length).toBeGreaterThan(0);
    expect(zoo.completion.values.every(v => v.includes('qwen'))).toBe(true);
    const doc = await c.complete({ ref: { type: 'ref/resource', uri: 'neurarch://docs/{tool}' }, argument: { name: 'tool', value: 'check_s' } });
    expect(doc.completion.values).toContain('check_serving_config');
    const rule = await c.complete({ ref: { type: 'ref/prompt', name: 'explain_finding' }, argument: { name: 'rule', value: 'head' } });
    expect(rule.completion.values).toContain('head-dim-divisibility');
    const arch = await c.complete({ ref: { type: 'ref/prompt', name: 'compare_with_reference' }, argument: { name: 'architecture', value: 'bert' } });
    expect(arch.completion.values).toContain('bert-base');
    const gpu = await c.complete({ ref: { type: 'ref/prompt', name: 'precheck_serving' }, argument: { name: 'gpu', value: 'h1' } });
    expect(gpu.completion.values).toContain('h100-sxm');
  });

  it('offers only card ids the plan bundle actually has', async () => {
    const bundle = await readFile(new URL('./vendor/plan.bundle.mjs', import.meta.url), 'utf-8');
    for (const id of GPU_HINTS) expect(bundle, id).toContain(`id: "${id}"`);
  });

  it('serves the rule catalogue with every source represented', async () => {
    const r = await (await connect()).readResource({ uri: 'neurarch://rules/catalogue' });
    const cat = JSON.parse((r.contents[0] as { text: string }).text) as { rules: Array<{ id: string; source: string; provenance?: unknown }> };
    expect(new Set(cat.rules.map(x => x.source))).toEqual(new Set(['shape', 'lint', 'edit', 'pin']));
    expect(cat.rules.find(x => x.id === 'head-dim-divisibility')?.provenance).toBeDefined();
    expect(cat.rules.find(x => x.id === 'pin/hf-unpinned')?.provenance).toBeUndefined();
  });
});

describe('progress and cancellation', () => {
  it('sends progress notifications when the call carries a progressToken', async () => {
    vi.stubGlobal('fetch', feedFetch());
    const seen: Array<{ progress: number; total?: number }> = [];
    const r = await (await connect()).callTool(
      { name: 'model_changes', arguments: { hf_id: 'Qwen/Qwen3-8B' } },
      { onprogress: (p) => { seen.push({ progress: p.progress, total: p.total }); } },
    );
    expect(r.isError).toBeFalsy();
    expect(seen.map(s => s.progress)).toEqual([0, 1, 2]);
    expect(seen.every(s => s.total === 2)).toBe(true);
  });

  it('a cancelled call aborts the fetch it was waiting on', async () => {
    const f = feedFetch({ delayMs: 5_000 });
    vi.stubGlobal('fetch', f);
    const c = await connect();
    const ctrl = new AbortController();
    const pending = c.callTool({ name: 'model_changes', arguments: { hf_id: 'Qwen/Qwen3-8B' } }, { signal: ctrl.signal });
    await new Promise(r => setTimeout(r, 50));
    ctrl.abort('user pressed stop');
    await expect(pending).rejects.toThrow();
    expect(f.urls.length).toBe(1); // the index fetch began, the second never started
  });
});
