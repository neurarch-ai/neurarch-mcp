import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { TOOLS } from './tools.js';
import { CORE_TOOLS, listedTools } from './cli.js';
import { setHfEnabled } from './sources.js';
import { convertHFConfigToModel } from './vendor/engine.bundle.mjs';
import { diagnoseOom, oomFactsForModel } from './vendor/plan.bundle.mjs';
import type { ModelArchitecture } from './lib/types.js';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { createMcpServer } from './server.js';

const tool = TOOLS.find(t => t.name === 'diagnose_oom')!;
const call = (args: Record<string, unknown>, model?: ModelArchitecture) =>
  tool.handler(args, model as ModelArchitecture, { modelPath: '' }) as Promise<any>;

async function template(name: string): Promise<ModelArchitecture> {
  return JSON.parse(await readFile(new URL(`./test/templates/${name}.neurarch.json`, import.meta.url), 'utf-8'));
}

// PyTorch >= 2.1 wording, a 24 GB card nearly full of live tensors.
const FULL = 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total capacity of 22.05 GiB of which 412.12 MiB is free. Including non-PyTorch memory, this process has 21.63 GiB memory in use. Of the allocated memory 20.88 GiB is allocated by PyTorch, and 452.18 MiB is reserved by PyTorch but unallocated.';
// An 80 GB card with 17 GB reserved and empty: the request failed for want of one contiguous block.
const FRAGMENTED = 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.15 GiB of which 1.50 GiB is free. Including non-PyTorch memory, this process has 77.60 GiB memory in use. Of the allocated memory 60.00 GiB is allocated by PyTorch, and 17.00 GiB is reserved by PyTorch but unallocated.';

let cache: string;
const realCache = process.env.NEURARCH_MCP_CACHE;
beforeEach(async () => { cache = await mkdtemp(join(tmpdir(), 'nx-oom-')); process.env.NEURARCH_MCP_CACHE = cache; });
afterEach(async () => {
  setHfEnabled(false);
  vi.restoreAllMocks();
  if (realCache === undefined) delete process.env.NEURARCH_MCP_CACHE; else process.env.NEURARCH_MCP_CACHE = realCache;
  await rm(cache, { recursive: true, force: true });
});

describe('diagnose_oom, the tool', () => {
  it('is in the core listing and tells an agent when to call it', () => {
    expect(CORE_TOOLS).toContain('diagnose_oom');
    expect(tool.description).toMatch(/after a CUDA out-of-memory error, before changing batch size/);
    const listed = listedTools(false, 'core').find(t => t.name === 'diagnose_oom')!;
    expect(listed.description).toMatch(/before changing batch size/);
    expect(listed.inputSchema.required).toEqual(['error']);
  });

  it('calls a job too big for its card too big, and keeps the effective batch in the first fix', async () => {
    const r = await call({ error: FULL, batch: 64, seq_len: 2048 }, await template('phi3-mini'));
    expect(r.cause).toBe('too-big-for-card');
    expect(r.summary).toMatch(/bigger than this card/);
    expect(r.evidence.some((e: string) => /plan predicts/.test(e))).toBe(true);
    expect(r.fixes[0].action).toMatch(/^batch \d+ with gradient accumulation of \d+ steps$/);
    expect(r.fixes[0].changesTraining).toBe(false);
    expect(r.job).toMatchObject({ source: 'model', batch: 64, seqLen: 2048, attention: 'sdpa' });
    expect(r.job.predictedPeakGb).toBeGreaterThan(22);
    expect(r.mode).toBe('train');
  });

  it('is the bundle\'s answer, not a second computation', async () => {
    const m = await template('phi3-mini');
    const r = await call({ error: FULL, batch: 64, seq_len: 2048 }, m);
    const d = diagnoseOom(FULL, oomFactsForModel(m, { batch: 64, seqLen: 2048 }));
    expect({ cause: r.cause, summary: r.summary, evidence: r.evidence, fixes: r.fixes }).toEqual(
      { cause: d.cause, summary: d.summary, evidence: d.evidence, fixes: d.fixes },
    );
  });

  it('does not blame the batch when the allocator is fragmented', async () => {
    const r = await call({ error: FRAGMENTED, batch: 8, step: 0 }, await template('gpt2'));
    expect(r.cause).toBe('fragmentation');
    expect(r.fixes[0].action).toMatch(/expandable_segments:True/);
    expect(r.fixes.some((f: { action: string }) => /^batch /.test(f.action))).toBe(false);
  });

  it('with no model reads the message alone and says what it could not compare', async () => {
    const r = await call({ error: FRAGMENTED });
    expect(r.cause).toBe('fragmentation');
    expect(r.job.source).toBe('none');
    expect(r.job.predictedPeakGb).toBeNull();
    expect(r.caveats.join(' ')).toMatch(/model_path or hf_id/);
  });

  it('says so when the text is not an out-of-memory error', async () => {
    const r = await call({ error: 'RuntimeError: shape mismatch' }, await template('gpt2'));
    expect(r.cause).toBe('unknown');
    expect(r.message).toBeNull();
    expect(r.fixes).toEqual([]);
  });

  it('sizes an hf_id from the config cache with no --hf and no network, as an inference job', async () => {
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

    const r = await call({ error: FULL, hf_id: id, batch: 256, seq_len: 8192, inference: true, gen_tokens: 1024 });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(r.mode).toBe('inference');
    expect(r.job.source).toBe(`hf:${id} (cached config.json)`);
    expect(r.job).toMatchObject({ batch: 256, seqLen: 8192, genTokens: 1024 });
    expect(r.job.weightsGb).toBeGreaterThan(0.5);
    expect(r.job.kvCachePerSequenceGb).toBeGreaterThan(0);
    expect(r.cause).toBe('too-big-for-card');
    expect(r.fixes[0].action).toMatch(/^generate at batch \d+/);
  });

  it('an hf_id never fetched, without --hf, is refused and names the flag', async () => {
    await expect(call({ error: FULL, hf_id: 'Qwen/Qwen2.5-0.5B' })).rejects.toThrow(/--hf/);
  });

  it('rejects an empty error and a non-integer batch', async () => {
    await expect(call({ error: '  ' })).rejects.toThrow(/out-of-memory text/);
    await expect(call({ error: FULL, batch: 0 })).rejects.toThrow(/batch/);
  });

  it('answers over MCP on a hosted server with no model of its own, given model_path', async () => {
    const server = createMcpServer({
      getModel: () => { throw new Error('no model'); },
      ctx: { modelPath: '' }, writeEnabled: false, version: '0.0.0-test',
    });
    const client = new Client({ name: 'test', version: '0.0.0' }, { capabilities: {} });
    const [ct, st] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(st), client.connect(ct)]);
    const path = new URL('./test/templates/phi3-mini.neurarch.json', import.meta.url).pathname;
    const r = await client.callTool({ name: 'diagnose_oom', arguments: { model_path: path, error: FULL, batch: 64 } });
    expect(r.isError).toBeFalsy();
    expect((r.structuredContent as { cause: string }).cause).toBe('too-big-for-card');
  });
});
