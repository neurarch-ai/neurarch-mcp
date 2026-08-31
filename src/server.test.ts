/**
 * End-to-end over a real MCP client and an in-memory transport.
 *
 * server.ts stopped being a pass-through the moment it grew `model_path`
 * routing and structured results, and the parts that matter are the parts only
 * a real client sees: what ListTools advertises, and what a call actually
 * returns on the wire. Testing the handlers directly would miss all of it.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, writeFile, rm, utimes } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { createMcpServer } from './server.js';
import { makeModel } from './test/fixtures.js';
import { clearModelCache } from './models.js';
import { MODEL_PATH_PARAM } from './cli.js';
import type { ModelArchitecture } from './lib/types.js';

let dir: string;

async function connect(opts: { writeEnabled?: boolean; model?: ModelArchitecture } = {}) {
  const model = opts.model ?? makeModel();
  const server = createMcpServer({
    getModel: () => model,
    ctx: { modelPath: join(dir, 'model.neurarch.json') },
    writeEnabled: opts.writeEnabled ?? false,
    version: '0.0.0-test',
  });
  const client = new Client({ name: 'test', version: '0.0.0' }, { capabilities: {} });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  return client;
}

/** A second model on disk, structurally distinguishable from the fixture. */
async function writeOtherModel(name = 'other.neurarch.json'): Promise<string> {
  const path = join(dir, name);
  await writeFile(path, JSON.stringify({
    id: 'other', name: 'other-model',
    components: [
      { id: 'i', type: 'input', name: 'input', position: { x: 0, y: 0 }, params: {}, inputs: [], outputs: ['l'], outputShape: [8] },
      { id: 'l', type: 'linear', name: 'fc', position: { x: 0, y: 1 }, params: { inFeatures: 8, outFeatures: 4 }, inputs: ['i'], outputs: [], inputShape: [8], outputShape: [4] },
    ],
    connections: [{ id: 'e', from: 'i', to: 'l', fromPort: 'bottom', toPort: 'top' }],
  }), 'utf-8');
  return path;
}

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'nx-mcp-server-'));
  clearModelCache();
});
afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

describe('ListTools', () => {
  it('annotates every read tool as read-only and non-destructive', async () => {
    const client = await connect();
    const { tools } = await client.listTools();
    expect(tools.length).toBeGreaterThan(10);
    for (const t of tools) {
      expect(t.annotations, `${t.name} has no annotations`).toBeDefined();
      expect(t.annotations!.readOnlyHint, t.name).toBe(true);
      expect(t.annotations!.destructiveHint, t.name).toBe(false);
    }
  });

  it('marks check_design as the one read tool that leaves the machine', async () => {
    const client = await connect();
    const { tools } = await client.listTools();
    const check = tools.find(t => t.name === 'check_design')!;
    expect(check.annotations!.openWorldHint).toBe(true);
    expect(check.annotations!.readOnlyHint).toBe(true);
    // Everything else is closed-world: it only ever reads the file.
    const others = tools.filter(t => t.name !== 'check_design');
    expect(others.every(t => t.annotations!.openWorldHint === false)).toBe(true);
  });

  it('flags the destructive write tools and only those', async () => {
    const client = await connect({ writeEnabled: true });
    const { tools } = await client.listTools();
    const byName = Object.fromEntries(tools.map(t => [t.name, t.annotations!]));
    for (const name of ['delete_layer', 'delete_connection', 'save_model', 'modify_layer']) {
      expect(byName[name].destructiveHint, name).toBe(true);
      expect(byName[name].readOnlyHint, name).toBe(false);
    }
    for (const name of ['add_layer', 'add_connection']) {
      expect(byName[name].destructiveHint, name).toBe(false);
      expect(byName[name].readOnlyHint, name).toBe(false);
    }
  });

  it('offers model_path on read tools and withholds it from write tools', async () => {
    const client = await connect({ writeEnabled: true });
    const { tools } = await client.listTools();
    const props = (t: { inputSchema: unknown }) =>
      ((t.inputSchema as { properties?: Record<string, unknown> }).properties ?? {});
    expect(props(tools.find(t => t.name === 'get_model_summary')!)).toHaveProperty(MODEL_PATH_PARAM);
    expect(props(tools.find(t => t.name === 'add_layer')!)).not.toHaveProperty(MODEL_PATH_PARAM);
    // diff_models has a `path` of its own meaning "the other file"; adding
    // model_path must not disturb it.
    const diff = props(tools.find(t => t.name === 'diff_models')!);
    expect(diff).toHaveProperty('path');
    expect(diff).toHaveProperty(MODEL_PATH_PARAM);
  });
});

describe('CallTool', () => {
  it('returns structuredContent alongside the JSON text', async () => {
    const client = await connect();
    const res = await client.callTool({ name: 'get_model_summary', arguments: {} }) as any;
    expect(res.structuredContent.name).toBe('tiny-transformer');
    expect(res.structuredContent.layerCount).toBe(7);
    // The text half stays, for clients that predate structured results.
    expect(JSON.parse(res.content[0].text).layerCount).toBe(7);
  });

  it('answers about another file when model_path is given', async () => {
    const path = await writeOtherModel();
    const client = await connect();
    const mine = await client.callTool({ name: 'get_model_summary', arguments: {} }) as any;
    const other = await client.callTool({
      name: 'get_model_summary',
      arguments: { [MODEL_PATH_PARAM]: path },
    }) as any;
    expect(mine.structuredContent.name).toBe('tiny-transformer');
    expect(other.structuredContent.name).toBe('other-model');
    expect(other.structuredContent.layerCount).toBe(2);
  });

  it('keeps the tool\'s own arguments working alongside model_path', async () => {
    const path = await writeOtherModel();
    const client = await connect();
    const res = await client.callTool({
      name: 'get_layer',
      arguments: { name: 'fc', [MODEL_PATH_PARAM]: path },
    }) as any;
    expect(res.structuredContent.type).toBe('linear');
  });

  it('refuses model_path on a write tool rather than editing the wrong file', async () => {
    const path = await writeOtherModel();
    const client = await connect({ writeEnabled: true });
    const res = await client.callTool({
      name: 'delete_layer',
      arguments: { name: 'fc', [MODEL_PATH_PARAM]: path },
    }) as any;
    expect(res.isError).toBe(true);
    expect(res.content[0].text).toMatch(/does not accept model_path/);
  });

  it('reports an unreadable model_path as a tool error, not a crash', async () => {
    const client = await connect();
    const res = await client.callTool({
      name: 'get_model_summary',
      arguments: { [MODEL_PATH_PARAM]: join(dir, 'nope.json') },
    }) as any;
    expect(res.isError).toBe(true);
    expect(res.content[0].text).toMatch(/Cannot read model file/);
  });

  it('picks up an edit to the other file on the next call', async () => {
    const path = await writeOtherModel();
    const client = await connect();
    const before = await client.callTool({
      name: 'get_model_summary', arguments: { [MODEL_PATH_PARAM]: path },
    }) as any;
    expect(before.structuredContent.layerCount).toBe(2);

    const model = JSON.parse(await import('node:fs/promises').then(fs => fs.readFile(path, 'utf-8')));
    model.components.push({
      id: 'r', type: 'relu', name: 'act', position: { x: 0, y: 2 },
      params: {}, inputs: ['l'], outputs: [],
    });
    await writeFile(path, JSON.stringify(model), 'utf-8');
    // mtime resolution can be coarse enough that two writes in the same
    // millisecond look identical; push it forward so the test is testing the
    // cache rule and not the filesystem clock.
    const future = new Date(Date.now() + 2000);
    await utimes(path, future, future);

    const after = await client.callTool({
      name: 'get_model_summary', arguments: { [MODEL_PATH_PARAM]: path },
    }) as any;
    expect(after.structuredContent.layerCount).toBe(3);
  });

  it('still hides write tools without --write', async () => {
    const client = await connect();
    const res = await client.callTool({ name: 'delete_layer', arguments: { name: 'block_0' } }) as any;
    expect(res.isError).toBe(true);
    expect(res.content[0].text).toMatch(/--write/);
  });
});
