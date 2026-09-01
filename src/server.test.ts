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

async function connect(opts: { writeEnabled?: boolean; model?: ModelArchitecture; toolSet?: 'core' | 'full' } = {}) {
  const model = opts.model ?? makeModel();
  const server = createMcpServer({
    getModel: () => model,
    ctx: { modelPath: join(dir, 'model.neurarch.json') },
    writeEnabled: opts.writeEnabled ?? false,
    version: '0.0.0-test',
    toolSet: opts.toolSet ?? 'full',
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

  it('marks every read tool closed-world, check_design included', async () => {
    // check_design was the exception until the verifier was vendored: it POSTed
    // the graph to an endpoint, so its answer tracked a service rather than the
    // file, and it was annotated openWorld/non-idempotent to say so. It is a
    // local function call now, and an annotation that still claimed otherwise
    // would tell a client to expect network failures that cannot happen.
    const client = await connect();
    const { tools } = await client.listTools();
    for (const t of tools) {
      expect(t.annotations!.openWorldHint, t.name).toBe(false);
    }
    const check = tools.find(t => t.name === 'check_design')!;
    expect(check.annotations!.readOnlyHint).toBe(true);
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
    const client = await connect({ writeEnabled: true, toolSet: 'full' });
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

  it('answers about a PyTorch file given as model_path', async () => {
    const path = join(dir, 'net.py');
    await writeFile(path, [
      'import torch.nn as nn',
      '',
      'class Net(nn.Module):',
      '    def __init__(self):',
      '        super().__init__()',
      '        self.fc = nn.Linear(16, 4)',
      '',
      '    def forward(self, x):',
      '        return self.fc(x)',
      '',
    ].join('\n'), 'utf-8');
    const client = await connect();
    const res = await client.callTool({
      name: 'get_model_summary',
      arguments: { [MODEL_PATH_PARAM]: path },
    }) as any;
    expect(res.structuredContent.name).toBe('net');
    expect(res.structuredContent.totalParameters).toBe(16 * 4 + 4);
  });

  it('still hides write tools without --write', async () => {
    const client = await connect();
    const res = await client.callTool({ name: 'delete_layer', arguments: { name: 'block_0' } }) as any;
    expect(res.isError).toBe(true);
    expect(res.content[0].text).toMatch(/--write/);
  });
});

describe('prompts and resources', () => {
  it('advertises prompts and renders one with its argument', async () => {
    const client = await connect();
    const { prompts } = await client.listPrompts();
    expect(prompts.map(p => p.name)).toContain('review_design');
    expect(prompts.map(p => p.name)).toContain('shrink_for_target');
    const got = await client.getPrompt({ name: 'shrink_for_target', arguments: { target: 'under 10M params' } });
    const text = (got.messages[0].content as { text: string }).text;
    expect(text).toMatch(/under 10M params/);
    expect(text).toMatch(/rank_designs/);
    await expect(client.getPrompt({ name: 'shrink_for_target' })).rejects.toThrow(/target/);
  });

  it('serves the model, its diagram, its source, the zoo and the rules as resources', async () => {
    const client = await connect();
    const { resources } = await client.listResources();
    expect(resources.map(r => r.uri)).toEqual(expect.arrayContaining(['neurarch://model', 'neurarch://model/mermaid', 'neurarch://model/pytorch', 'neurarch://zoo', 'neurarch://rules']));
    const model = await client.readResource({ uri: 'neurarch://model' });
    expect(JSON.parse((model.contents[0] as { text: string }).text).components.length).toBe(7);
    const mermaid = await client.readResource({ uri: 'neurarch://model/mermaid' });
    expect((mermaid.contents[0] as { text: string }).text).toMatch(/^flowchart TD/);
    const py = await client.readResource({ uri: 'neurarch://model/pytorch' });
    expect((py.contents[0] as { text: string }).text).toMatch(/nn\.Module/);
    const { resourceTemplates } = await client.listResourceTemplates();
    expect(resourceTemplates[0].uriTemplate).toBe('neurarch://zoo/{id}');
    const bert = await client.readResource({ uri: 'neurarch://zoo/bert-base' });
    expect(JSON.parse((bert.contents[0] as { text: string }).text).components.length).toBeGreaterThan(10);
    await expect(client.readResource({ uri: 'neurarch://nope' })).rejects.toThrow(/Unknown resource/);
  });
});

describe('model_source and zoo: routing', () => {
  it('answers about inline PyTorch source, and about a zoo entry by model_path', async () => {
    const client = await connect();
    const inline = await client.callTool({
      name: 'get_model_summary',
      arguments: { model_source: 'import torch.nn as nn\nclass M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, 4)\n    def forward(self, x):\n        return self.fc(x)\n' },
    });
    expect((inline.structuredContent as any).name).toBe('model');
    const zoo = await client.callTool({ name: 'get_model_summary', arguments: { [MODEL_PATH_PARAM]: 'zoo:bert-base' } });
    expect((zoo.structuredContent as any).layerCount).toBeGreaterThan(10);
    const both = await client.callTool({ name: 'get_model_summary', arguments: { [MODEL_PATH_PARAM]: 'zoo:bert-base', model_source: '{}' } });
    expect(both.isError).toBe(true);
  });

  it('hosted: a server with no model says so until a call names one', async () => {
    const server = createMcpServer({
      getModel: () => { throw new Error('This server was started without a model.'); },
      ctx: { modelPath: '' }, writeEnabled: false, version: '0.0.0-test',
    });
    const client = new Client({ name: 'test', version: '0.0.0' }, { capabilities: {} });
    const [ct, st] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(st), client.connect(ct)]);
    const bare = await client.callTool({ name: 'get_model_summary', arguments: {} });
    expect(bare.isError).toBe(true);
    expect((bare.content as any)[0].text).toMatch(/without a model/);
    const named = await client.callTool({ name: 'get_model_summary', arguments: { [MODEL_PATH_PARAM]: 'zoo:resnet-50' } });
    expect(named.isError).toBeFalsy();
  });

  it('--tools=core advertises the core set but still answers a full-set tool by name', async () => {
    const model = makeModel();
    const server = createMcpServer({ getModel: () => model, ctx: { modelPath: '' }, writeEnabled: false, version: '0.0.0-test', toolSet: 'core' });
    const client = new Client({ name: 'test', version: '0.0.0' }, { capabilities: {} });
    const [ct, st] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(st), client.connect(ct)]);
    const { tools } = await client.listTools();
    expect(tools.length).toBeLessThan(15);
    expect(tools.map(t => t.name)).toContain('rank_designs');
    expect(tools.map(t => t.name)).not.toContain('list_connections');
    const r = await client.callTool({ name: 'list_connections', arguments: {} });
    expect(r.isError).toBeFalsy();
  });
});

describe('docs resources and elicitation', () => {
  it('serves every tool\'s long description as a resource, including the ones core does not list', async () => {
    const client = await connect({ toolSet: 'core' });
    const { tools } = await client.listTools();
    expect(tools.map(t => t.name)).not.toContain('list_connections');
    const all = await client.readResource({ uri: 'neurarch://docs' });
    const text = (all.contents[0] as { text: string }).text;
    expect(text).toMatch(/## list_connections/);
    expect(text).toMatch(/## save_model \(write tool/);
    const one = await client.readResource({ uri: 'neurarch://docs/rank_designs' });
    expect((one.contents[0] as { text: string }).text).toMatch(/calibration/);
    await expect(client.readResource({ uri: 'neurarch://docs/nope' })).rejects.toThrow(/No tool named/);
    // The listed description is the short one, and it carries an output schema hint.
    const cd = tools.find(t => t.name === 'check_design')!;
    expect(cd.description!.length).toBeLessThan(220);
    expect((cd as { outputSchema?: unknown }).outputSchema).toBeDefined();
  });

  it('check_design asks the human through elicitation when the client can answer', async () => {
    const model = JSON.parse(await (await import('node:fs/promises')).readFile(new URL('../examples/tiny-gpt.neurarch.json', import.meta.url), 'utf-8'));
    const server = createMcpServer({ getModel: () => model, ctx: { modelPath: '' }, writeEnabled: false, version: '0.0.0-test' });
    const client = new Client({ name: 'test', version: '0.0.0' }, { capabilities: { elicitation: {} } });
    const { ElicitRequestSchema } = await import('@modelcontextprotocol/sdk/types.js');
    let asked: string | undefined;
    client.setRequestHandler(ElicitRequestSchema, async (req) => {
      asked = req.params.message;
      const choices = ((req.params as any).requestedSchema as { properties: { choice: { enum: string[] } } }).properties.choice.enum;
      return { action: 'accept', content: { choice: choices[choices.length - 1] } };
    });
    const [ct, st] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(st), client.connect(ct)]);
    const r = await client.callTool({ name: 'check_design', arguments: { ask_user: true } });
    const out = r.structuredContent as any;
    expect(asked).toMatch(/data/i);
    expect(out.decision.answer.action).toBe('accept');
    expect(out.decision.answer.value).toBe('synthetic');
    // Default output is the trimmed one.
    expect(out.stages[0].data).toBeUndefined();
    const verbose = await client.callTool({ name: 'check_design', arguments: { verbose: true } });
    expect((verbose.structuredContent as any).stages[0].data).toBeDefined();
  });
});
