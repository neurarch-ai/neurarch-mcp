import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { TOOLS } from './tools.js';
import { CORE_TOOLS } from './cli.js';
import { setHfEnabled } from './sources.js';
import { convertHFConfigToModel } from './vendor/engine.bundle.mjs';
import type { ModelArchitecture } from './lib/types.js';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { createMcpServer } from './server.js';

const tool = TOOLS.find(t => t.name === 'size_training_job')!;
const call = (args: Record<string, unknown>, model?: ModelArchitecture) =>
  tool.handler(args, model as ModelArchitecture, { modelPath: '' }) as Promise<any>;

async function template(name: string): Promise<ModelArchitecture> {
  return JSON.parse(await readFile(new URL(`./test/templates/${name}.neurarch.json`, import.meta.url), 'utf-8'));
}

let cache: string;
const realCache = process.env.NEURARCH_MCP_CACHE;
beforeEach(async () => { cache = await mkdtemp(join(tmpdir(), 'nx-size-')); process.env.NEURARCH_MCP_CACHE = cache; });
afterEach(async () => {
  setHfEnabled(false);
  vi.restoreAllMocks();
  if (realCache === undefined) delete process.env.NEURARCH_MCP_CACHE; else process.env.NEURARCH_MCP_CACHE = realCache;
  await rm(cache, { recursive: true, force: true });
});

describe('size_training_job, the tool', () => {
  it('is the first tool an agent sees, and its description says to call it before launching', () => {
    expect(CORE_TOOLS[0]).toBe('size_training_job');
    expect(tool.description).toMatch(/BEFORE submitting or launching any training job/);
    expect(tool.description).toMatch(/calibrated estimate.*AdamW with bf16 autocast/);
    expect(tool.description).toMatch(/median\s+error 8%/);
  });

  it('answers on the model it is handed', async () => {
    const r = await call({ batch_size: 32, gpu: 'A100-40' }, await template('phi3-mini'));
    expect(r.verdict).toMatch(/^fits an A100-40 with \d+% headroom/);
    expect(r.source).toBe('model');
    expect(r.lastRun.how).toMatch(/history/);
  });

  it('sizes a Hugging Face id from the config cache with no network', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    const id = 'Qwen/Qwen2.5-0.5B';
    const { components, connections } = convertHFConfigToModel(id, {
      model_type: 'qwen2', architectures: ['Qwen2ForCausalLM'], hidden_size: 896, num_hidden_layers: 24,
      num_attention_heads: 14, num_key_value_heads: 2, intermediate_size: 4864, vocab_size: 151936,
      max_position_embeddings: 32768, tie_word_embeddings: true,
    });
    await mkdir(cache, { recursive: true });
    await writeFile(join(cache, `${id.replace('/', '__')}.json`), JSON.stringify({
      model: { id: `hf-${id}`, name: 'Qwen2.5-0.5B', components, connections },
      configSource: 'config', realParamCount: 494032768, cached: false,
    }));
    setHfEnabled(true);

    const r = await call({ hf_id: id, batch_size: 8, gpu: 'L4' });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(r.sized).toBe(true);
    expect(r.source).toMatch(/^hf:Qwen\/Qwen2.5-0.5B/);
    expect(r.publishedParams).toBe(494032768);
    // Sized at 2048 tokens, not at the 32,768-token context the config declares.
    expect(r.job.seqLen).toBe(2048);
    expect(r.memory.peakGb).toBeGreaterThan(5);
    expect(r.memory.peakGb).toBeLessThan(40);
    const longer = await call({ hf_id: id, batch_size: 8, gpu: 'L4', seq_len: 4096 });
    expect(longer.memory.peakGb).toBeGreaterThan(r.memory.peakGb);
  });

  it('an hf_id without --hf is refused, and names the flag', async () => {
    await expect(call({ hf_id: 'Qwen/Qwen2.5-0.5B', batch_size: 8 })).rejects.toThrow(/--hf/);
  });

  it('with no model and no hf_id, says what to pass', async () => {
    await expect(call({ batch_size: 8 })).rejects.toThrow(/name a model/);
  });

  it('rejects a batch that is not a positive integer', async () => {
    await expect(call({ batch_size: 0 }, await template('gpt2'))).rejects.toThrow(/batch_size/);
  });

  it('answers over MCP on a hosted server with no model of its own, given model_path', async () => {
    const server = createMcpServer({
      getModel: () => { throw new Error('no model'); },
      ctx: { modelPath: '' }, writeEnabled: false, version: '0.0.0-test',
    });
    const client = new Client({ name: 'test', version: '0.0.0' }, { capabilities: {} });
    const [ct, st] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(st), client.connect(ct)]);
    const path = new URL('./test/templates/gpt2.neurarch.json', import.meta.url).pathname;
    const r = await client.callTool({ name: 'size_training_job', arguments: { model_path: path, batch_size: 32, gpu: 'T4' } });
    expect(r.isError).toBeFalsy();
    expect((r.structuredContent as { verdict: string }).verdict).toMatch(/^fits a T4/);
  });
});
