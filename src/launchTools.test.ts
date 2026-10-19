import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Client } from '@modelcontextprotocol/client';
import { InMemoryTransport } from '@modelcontextprotocol/server';
import { TOOLS } from './tools.js';
import { CORE_TOOLS, listedTools } from './cli.js';
import { createMcpServer } from './server.js';
import { setHfEnabled, loadHFModel } from './sources.js';
import { checkParallelLayout, readLayout, checkServeConfig, readServeSpec, LAYOUT_BASIS, SERVE_BASIS } from './vendor/plan.bundle.mjs';
import { stubHuggingFace, fixtureFile, QWEN3_8B, QWEN3_8B_PARAMS } from './test/hfStub.js';
import type { ModelArchitecture } from './lib/types.js';

const layoutTool = TOOLS.find(t => t.name === 'check_parallel_layout')!;
const serveTool = TOOLS.find(t => t.name === 'check_serving_config')!;
const layout = (args: Record<string, unknown>, model?: ModelArchitecture) => layoutTool.handler(args, model as ModelArchitecture, { modelPath: '' }) as Promise<any>;
const serve = (args: Record<string, unknown>, model?: ModelArchitecture) => serveTool.handler(args, model as ModelArchitecture, { modelPath: '' }) as Promise<any>;

async function template(name: string): Promise<ModelArchitecture> {
  return JSON.parse(await readFile(new URL(`./test/templates/${name}.neurarch.json`, import.meta.url), 'utf-8'));
}

/** An MCP client on a hosted-style server with no model of its own, so model_path is the only subject. */
async function client(): Promise<Client> {
  const server = createMcpServer({ getModel: () => { throw new Error('no model'); }, ctx: { modelPath: '' }, writeEnabled: false, version: '0.0.0-test' });
  const c = new Client({ name: 'test', version: '0.0.0' }, { capabilities: {} });
  const [ct, st] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(st), c.connect(ct)]);
  return c;
}

let cache: string;
let fetchSpy: ReturnType<typeof stubHuggingFace>;
const realCache = process.env.NEURARCH_MCP_CACHE;
beforeEach(async () => {
  cache = await mkdtemp(join(tmpdir(), 'nx-launch-'));
  process.env.NEURARCH_MCP_CACHE = cache;
  fetchSpy = stubHuggingFace();
  setHfEnabled(true);
});
afterEach(async () => {
  setHfEnabled(false);
  vi.restoreAllMocks();
  if (realCache === undefined) delete process.env.NEURARCH_MCP_CACHE; else process.env.NEURARCH_MCP_CACHE = realCache;
  await rm(cache, { recursive: true, force: true });
});

const fetchedUrls = () => fetchSpy.mock.calls.map(c => String(c[0]));

describe('the two launch checks in the listing', () => {
  it('are core tools, listed after check_edit, each saying when to call it', () => {
    const at = (n: string) => CORE_TOOLS.indexOf(n);
    expect(at('check_parallel_layout')).toBe(at('check_edit') + 1);
    expect(at('check_serving_config')).toBe(at('check_edit') + 2);
    const listed = listedTools(false, 'core');
    const l = listed.find(t => t.name === 'check_parallel_layout')!;
    const s = listed.find(t => t.name === 'check_serving_config')!;
    expect(l.description).toMatch(/^Call BEFORE a TP\/PP\/DP training launch/);
    expect(l.description).toMatch(/predicted, not measured/);
    expect(s.description).toMatch(/^Call BEFORE a vLLM\/SGLang\/llama\.cpp launch/);
    expect(Object.keys(l.inputSchema.properties as object)).toEqual([
      'tp', 'pp', 'dp', 'cp', 'ep', 'micro_batch', 'global_batch', 'grad_accum', 'seq_len', 'gpu', 'schedule',
      'virtual_stages', 'layers_per_stage', 'zero_stage', 'activation_checkpointing', 'seq_len_histogram', 'model_path',
    ]);
    expect(l.inputSchema.required).toEqual(['micro_batch', 'seq_len']);
    expect(Object.keys(s.inputSchema.properties as object)).toEqual(['command', 'args', 'engine', 'gpu', 'gpus', 'against', 'model_path']);
    expect(l.annotations.readOnlyHint).toBe(true);
    expect(s.annotations.readOnlyHint).toBe(true);
  });
});

describe('check_parallel_layout on Qwen3-8B (real config.json, served offline)', () => {
  const split = { tp: 4, pp: 4, dp: 1, micro_batch: 1, global_batch: 32, seq_len: 4096, gpu: 'h100' };

  it('is the bundle\'s report, with the predicted numbers labelled and a better split', async () => {
    const c = await client();
    const r = await c.callTool({ name: 'check_parallel_layout', arguments: { model_path: `hf:${QWEN3_8B}`, ...split } });
    expect(r.isError).toBeFalsy();
    const out = r.structuredContent as any;
    expect(fetchedUrls().every(u => u.startsWith('https://huggingface.co/'))).toBe(true);

    // The same numbers the bundle gives on the same graph and layout: nothing recomputed here.
    const model = (await loadHFModel(QWEN3_8B)).model;
    const direct = checkParallelLayout(model, readLayout(split) as never);
    expect(out.stages).toEqual(direct.stages);
    expect(out.findings).toEqual(direct.findings);
    expect(out.basis).toBe(LAYOUT_BASIS);

    expect(out.legal).toBe(true);
    expect(out.resolved).toMatchObject({ worldSize: 16, blocks: 36, microbatchesPerStep: 32, layersPerStage: [9, 9, 9, 9] });
    expect(out.stages.map((s: any) => s.layers.extras)).toEqual([['embedding'], [], [], ['head+loss']]);
    // The last stage carries the LM head and the loss, and is the slowest.
    expect(out.predicted.label).toMatch(/^Predicted from the graph, not measured/);
    expect(out.predicted.slowestStage).toBe(3);
    expect(out.predicted.stepSlowdown).toBeCloseTo(0.1553, 3);
    expect(out.predicted.recommendedSplit).toEqual([10, 10, 9, 7]);
    expect(out.predicted.recommendedSpeedup).toBeCloseTo(0.1143, 3);
    expect(out.predicted.microbatches).toBe(32);
    expect(out.predicted.formula).toMatch(/^1F1B schedule simulated op by op/);
    expect(out.findings.map((f: any) => `${f.severity}:${f.rule}`)).toContain('warn:stage-imbalance');
    expect(out.verdict).toBe(
      'legal on 16 x h100-sxm; predicted 15.5% slower per step than a balanced split (slowest stage 3); '
      + 'split 10,10,9,7 is predicted 11.4% faster; bubble 8.6%; 1 warning',
    );
    expect(out.source).toBe(`model`);
  });

  it('takes the recommended split back and reports it balanced', async () => {
    const m = (await loadHFModel(QWEN3_8B)).model;
    const out = await layout({ ...split, layers_per_stage: [10, 10, 9, 7] }, m);
    expect(out.resolved.layersPerStage).toEqual([10, 10, 9, 7]);
    // 36 blocks over 4 stages move a whole layer at a time, so the best split still sits ~5% over the mean.
    expect(out.predicted.slowestOverMean).toBeLessThan(1.05);
    expect(out.predicted.recommendedSplit).toBeNull();
    // At the best whole-block split the rule drops to info and says nothing contiguous does better.
    expect(out.findings.filter((f: any) => f.rule === 'stage-imbalance').map((f: any) => f.severity)).not.toContain('warn');
  });

  it('says a TP that does not divide the 32 heads will not run', async () => {
    const m = (await loadHFModel(QWEN3_8B)).model;
    const out = await layout({ tp: 3, micro_batch: 1, grad_accum: 8, seq_len: 4096 }, m);
    expect(out.legal).toBe(false);
    expect(out.verdict).toMatch(/^will not run: TP 3 does not divide 32 attention heads \(and 2 more blocking findings\)$/);
    expect(out.findings.filter((f: any) => f.severity === 'block').map((f: any) => f.rule)).toEqual(['tp-heads', 'tp-kv-heads', 'tp-hidden']);
  });

  it('reads a sequence-length histogram and predicts the DP imbalance', async () => {
    const m = (await loadHFModel(QWEN3_8B)).model;
    const out = await layout({ tp: 8, dp: 4, micro_batch: 2, grad_accum: 4, seq_len: 8192, seq_len_histogram: '512,700\n2048,200\n8192,100' }, m);
    expect(out.checked).toContain('dp-seq-imbalance');
    expect(out.predicted.dpSeqImbalance.expectedMaxOverMean).toBeGreaterThan(1);
  });

  it('refuses an unreadable layout before loading anything, naming the key as the tool spells it', async () => {
    fetchSpy.mockClear();
    await expect(layout({ micro_batch: 0, seq_len: 4096, model_path: 'x' })).rejects.toThrow(/`micro_batch` must be a positive integer/);
    await expect(layout({ micro_batch: 1, seq_len: 4096, schedule: 'gpipe' })).rejects.toThrow(/`schedule` must be "1f1b" or "interleaved"/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('answers on a non-HF graph too (the gpt2 template), and asks for a model when there is none', async () => {
    const out = await layout({ pp: 2, micro_batch: 4, global_batch: 64, seq_len: 1024, gpu: 'l4' }, await template('gpt2'));
    expect(out.stages).toHaveLength(2);
    expect(out.source).toBe('model');
    await expect(layout({ micro_batch: 1, seq_len: 4096 })).rejects.toThrow(/name a model/);
  });
});

describe('check_serving_config on Qwen3-8B (real config, tokenizer and generation config, served offline)', () => {
  const tooLong = 'vllm serve Qwen/Qwen3-8B --tensor-parallel-size 4 --max-model-len 131072 --max-num-seqs 64';

  it('reads the model off the command, fetches the four files, and returns the bundle\'s findings', async () => {
    const out = await serve({ command: tooLong, gpu: 'l4', gpus: 4 });
    const urls = fetchedUrls();
    for (const f of ['config.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json']) {
      expect(urls).toContain(`https://huggingface.co/${QWEN3_8B}/resolve/main/${f}`);
    }
    expect(out.source).toBe(`hf:${QWEN3_8B} (config.json)`);
    expect(out.engine).toBe('vllm');
    expect(out.blocked).toBe(true);
    expect(out.verdict).toBe('blocked: Served length 131072 is past the model\'s 40960-token context');
    const rules = out.findings.map((f: any) => `${f.severity}:${f.rule}`);
    expect(rules).toEqual(expect.arrayContaining(['block:context-over-derived', 'warn:seqs-over-capacity', 'info:reasoning-parser-missing', 'info:pad-is-eos']));
    expect(out.resolved).toMatchObject({ tp: 4, gpus: 4, servedLen: 131072, derivedMaxLen: 40960 });
    expect(out.memory.maxConcurrentFullLength).toBeGreaterThanOrEqual(3);
    expect(out.memory.maxConcurrentFullLength).toBeLessThan(4);
    expect(out.memory.weightsGbPerGpu).toBeCloseTo((QWEN3_8B_PARAMS * 2) / 4 / 1e9, 0);
    expect(out.checked).toEqual(expect.arrayContaining(['tp-heads', 'context', 'memory', 'chat-template', 'sampler']));
    expect(out.caveats).toEqual([]);
    expect(out.basis).toBe(SERVE_BASIS);

    // Identical to the bundle on the same inputs.
    const cfg = (await loadHFModel(QWEN3_8B)).config;
    const spec = readServeSpec({ command: tooLong, gpu: 'l4', gpus: 4 }) as any;
    const direct = checkServeConfig({
      ...spec, model: (await loadHFModel(QWEN3_8B)).model, config: cfg, modelId: QWEN3_8B,
      tokenizerConfig: JSON.parse((await fixtureFile('Qwen3-8B', 'tokenizer_config.json'))!),
      generationConfig: JSON.parse((await fixtureFile('Qwen3-8B', 'generation_config.json'))!),
      chatTemplateFile: null,
    });
    expect(out.findings).toEqual(direct.findings);
    expect(out.memory).toEqual(direct.memory);
  });

  it('a clean launch: no blocking finding, and the concurrency in the verdict', async () => {
    const out = await serve({ command: 'vllm serve Qwen/Qwen3-8B --max-model-len 32768 --max-num-seqs 8 --reasoning-parser qwen3', gpu: 'h100', gpus: 1 });
    expect(out.blocked).toBe(false);
    expect(out.verdict).toMatch(/^no blocking finding in the checks that ran; at most \d+ concurrent full-length sequences \(an upper bound\)/);
  });

  it('blocks TP 3 on SGLang too, from parsed args with the model named by model_path', async () => {
    const c = await client();
    const r = await c.callTool({ name: 'check_serving_config', arguments: { model_path: `hf:${QWEN3_8B}`, engine: 'sglang', args: { tp: 3 } } });
    expect(r.isError).toBeFalsy();
    const out = r.structuredContent as any;
    expect(out.engine).toBe('sglang');
    expect(out.findings.filter((f: any) => f.severity === 'block').map((f: any) => f.rule)).toContain('tp-heads');
    expect(out.source).toBe(`hf:${QWEN3_8B} (config.json)`);
  });

  it('diffs against the model it replaces: a local checkpoint with YaRN to 131072', async () => {
    const dir = join(cache, 'qwen3-8b-long');
    await mkdir(dir, { recursive: true });
    const cfg = JSON.parse((await fixtureFile('Qwen3-8B', 'config.json'))!);
    cfg.rope_scaling = { rope_type: 'yarn', factor: 4.0, original_max_position_embeddings: 32768 };
    await writeFile(join(dir, 'config.json'), JSON.stringify(cfg));
    await writeFile(join(dir, 'tokenizer_config.json'), (await fixtureFile('Qwen3-8B', 'tokenizer_config.json'))!);
    await writeFile(join(dir, 'generation_config.json'), (await fixtureFile('Qwen3-8B', 'generation_config.json'))!);
    const out = await serve({ command: `vllm serve ${dir} --max-model-len 131072`, gpu: 'h100', against: QWEN3_8B });
    expect(out.source).toBe(`${dir} (config.json on disk)`);
    expect(out.findings.map((f: any) => f.rule)).not.toContain('context-over-derived');
    expect(out.change.changed.map((c: any) => c.key)).toContain('rope_scaling');
  });

  it('serves a second call from the cache with no network and no --hf', async () => {
    await serve({ command: tooLong, gpu: 'l4', gpus: 4 });
    setHfEnabled(false);
    fetchSpy.mockClear();
    const out = await serve({ command: tooLong, gpu: 'l4', gpus: 4 });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(out.checked).toContain('chat-template');
    expect(out.caveats).toEqual([]);
  });

  it('without --hf and nothing cached, names the flag', async () => {
    setHfEnabled(false);
    await expect(serve({ command: tooLong })).rejects.toThrow(/--hf/);
  });

  it('refuses a call with neither command nor args, and an unknown card, before any fetch', async () => {
    fetchSpy.mockClear();
    await expect(serve({ gpu: 'h100' })).rejects.toThrow(/pass command/);
    await expect(serve({ command: tooLong, gpu: 'tpu-v9' })).rejects.toThrow(/`gpu` "tpu-v9" is not a card id/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
