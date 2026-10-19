/**
 * Every declared outputSchema, checked the way a client checks it.
 *
 * The SDK's Client validates structuredContent against the tool's outputSchema
 * on every callTool and throws on a mismatch (and on a schema'd tool that sends
 * no structuredContent). So calling each schema'd tool through a real client,
 * on models of different shapes, is the test: a handler that drifts from its
 * schema fails here, not in somebody's editor.
 *
 * The coverage guard at the bottom fails when a tool gains a schema and no
 * call here, so a new schema cannot ship unexercised.
 */
import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { createMcpServer } from './server.js';
import { listedTools, declaresOutputSchema } from './cli.js';
import { loadZooModel } from './sources.js';
import { makeModel } from './test/fixtures.js';
import { clearChangeCache } from './changeTools.js';
import type { ModelArchitecture } from './lib/types.js';
import { feedFetch } from './test/feedStub.js';

let dir: string;
beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), 'nx-mcp-out-'));
  await writeFile(join(dir, 'train.py'), 'from transformers import AutoModel\nm = AutoModel.from_pretrained("Qwen/Qwen3-8B")\n');
  vi.stubGlobal('fetch', feedFetch());
  clearChangeCache();
});
afterAll(async () => {
  vi.unstubAllGlobals();
  await rm(dir, { recursive: true, force: true });
});

async function connect(model: ModelArchitecture): Promise<Client> {
  const server = createMcpServer({ getModel: () => model, ctx: { modelPath: join(dir, 'model.neurarch.json') }, writeEnabled: false, version: 't', toolSet: 'full' });
  const c = new Client({ name: 't', version: '0' }, { capabilities: {} });
  const [a, b] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(b), c.connect(a)]);
  await c.listTools(); // the client caches each tool's output validator from this
  return c;
}

const CALLS: Array<[string, Record<string, unknown>]> = [
  ['size_training_job', { batch_size: 8 }],
  ['size_training_job', { batch_size: 8, gpu: 'l4' }],
  ['check_edit', { base_model_path: 'zoo:bert-base' }],
  ['check_parallel_layout', { micro_batch: 1, seq_len: 512 }],
  ['check_parallel_layout', { micro_batch: 1, seq_len: 512, tp: 2, pp: 2 }],
  ['check_serving_config', { engine: 'vllm', args: { tp: 1 }, gpu: 'h100' }],
  ['check_pins', { source: 'x = AutoModel.from_pretrained("a/b")', filename: 't.py' }],
  ['check_pins', { paths: ['.'] }],
  ['model_changes', { hf_id: 'Qwen/Qwen3-8B' }],
  ['model_changes', { hf_id: 'Qwen/Qwen3-8B', since_revision: '895c8d171bc0', detail: 'full' }],
  ['model_changes', { hf_id: 'nobody/untracked' }],
  ['diagnose_oom', { error: 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.15 GiB of which 1.2 GiB is free.', batch: 8 }],
  ['check_design', {}],
  ['lint_model', {}],
  ['lint_model', { severity: 'block' }],
  ['validate_model', {}],
  ['rank_designs', { candidates: [{ model_path: 'zoo:bert-base' }], include_current: true }],
  ['suggest_fix', {}],
  ['describe_architecture', {}],
  ['get_model_summary', {}],
  ['layer_impact', { namePattern: '.' }],
  ['find_layers', {}],
  ['find_layers', { limit: 2 }],
  ['list_connections', {}],
  ['export_pytorch', {}],
  ['list_architectures', {}],
  ['list_architectures', { query: 'qwen', detail: 'full' }],
  ['find_models', { dir: '.' }],
];

/** Tools whose results depend on a service; their schemas declare no required keys and are covered in ledgerTools.test.ts. */
const NETWORK_ONLY = new Set(['plan', 'history']);

describe('outputSchema conformance', () => {
  it('every schema\'d tool returns structuredContent the SDK client accepts, on four models', async () => {
    const models = [makeModel(), await loadZooModel('bert-base'), await loadZooModel('qwen2.5-7b'), await loadZooModel('resnet-50')];
    for (const model of models) {
      const c = await connect(model);
      for (const [name, args] of CALLS) {
        // callTool throws when structuredContent does not match outputSchema.
        const r = await c.callTool({ name, arguments: args });
        expect(r.isError, `${model.name} ${name}: ${JSON.stringify(r.content).slice(0, 300)}`).toBeFalsy();
        expect(r.structuredContent, `${model.name} ${name}`).toBeDefined();
      }
    }
  }, 120_000);

  it('the server-side guard knows exactly the tools that list an outputSchema', () => {
    // server.ts turns a non-object result from these into isError rather than
    // dropping structuredContent; the guard and the listing must agree.
    for (const t of listedTools(true, 'full')) {
      expect(declaresOutputSchema(t.name), t.name).toBe(!!t.outputSchema);
    }
  });

  it('every tool that declares an outputSchema is exercised above', () => {
    const called = new Set(CALLS.map(c => c[0]));
    const declared = listedTools(false, 'full').filter(t => t.outputSchema).map(t => t.name);
    for (const name of declared) {
      if (NETWORK_ONLY.has(name)) continue;
      expect(called.has(name), `${name} has an outputSchema and no conformance call`).toBe(true);
    }
  });

  it('declares required keys only where they are typed, and every schema is an object schema', () => {
    for (const t of listedTools(false, 'full')) {
      if (!t.outputSchema) continue;
      expect(t.outputSchema.type, t.name).toBe('object');
      const req = (t.outputSchema.required as string[] | undefined) ?? [];
      const props = (t.outputSchema.properties ?? {}) as Record<string, unknown>;
      for (const k of req) expect(props, `${t.name}.${k}`).toHaveProperty(k);
    }
  });
});
