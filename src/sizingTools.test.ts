import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { TOOLS } from './tools.js';
import { CORE_TOOLS } from './cli.js';
import { setHfEnabled } from './sources.js';
import { convertHFConfigToModel } from './vendor/engine.bundle.mjs';
import { MEMORY_CALIBRATION } from './vendor/plan.bundle.mjs';
import * as plan from './vendor/plan.bundle.mjs';
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
    expect(tool.description).toMatch(new RegExp(`flagged ${MEMORY_CALIBRATION.flagged} of the ${MEMORY_CALIBRATION.noFit}`));
    expect(tool.description).toMatch(new RegExp(`median miss is ${MEMORY_CALIBRATION.afterFitMedianPct}%`));
    expect(tool.description).not.toMatch(/flagged 54|42%/);
    expect(tool.description).not.toMatch(/98%|median\s+error 8%/);
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

// neurarch #312: three answers per card against the conservative peak, the
// same fitVerdict the CLI's "fits" line and the plan card print.
describe('size_training_job: fits / borderline / does not fit, and the upper bound', () => {
  it('says borderline where only the point estimate fits, and names the card that holds the upper bound', async () => {
    const r = await call({ batch_size: 12, gpu: 'T4' }, await template('phi3-mini'));
    expect(r.fits).toBe(true);
    expect(r.gpuVerdict.T4).toBe('borderline');
    expect(r.gpuVerdict.L4).toBe('fits');
    expect(r.smallestFitGpu).toBe('T4');
    expect(r.smallestSafeGpu).toBe('L4');
    expect(r.verdict).toMatch(/^fits a T4 with \d+% headroom/);
    expect(r.verdict).toContain('borderline on a T4: the estimate fits, the upper bound with 1.15x headroom does not: measure once (one step at this batch) before reserving');
    expect(r.verdict).toMatch(/an L4 holds the upper bound \(~\d+ GB\)/);
  });

  it('every per-card verdict is the plan\'s own fitVerdict on this job\'s peak', async () => {
    const m = await template('phi3-mini');
    const r = await call({ batch_size: 32 }, m);
    const params = r.model.params;
    // memory.peakGb is rounded to 2 decimals; recompute from the unrounded upper bound instead.
    expect(r.upperBound.multiplier).toBe(plan.UPPER_K);
    expect(r.upperBound.runs).toBe(MEMORY_CALIBRATION.upperRuns);
    expect(r.upperBound.peakGb).toBeCloseTo(r.memory.peakGb * plan.UPPER_K, 1);
    expect(r.upperBound.needGb).toBeCloseTo(r.upperBound.peakGb * 1.15, 1);
    for (const [card, gb] of [['T4', 16], ['L4', 24], ['L40S', 48], ['A100-40', 40], ['A100-80', 80], ['H100-80', 80]] as const) {
      expect(r.gpuVerdict[card], card).toBe(plan.fitVerdict(r.memory.peakGb, params, gb));
    }
    // The 48 GB L40S holds the upper bound that the 40 GB A100 does not.
    expect(r.gpuVerdict['A100-40']).toBe('borderline');
    expect(r.smallestSafeGpu).toBe('L40S');
    expect(r.upperBound.sentence).toBe('the estimate x 1.43: held out by architecture, 95% of 91 measured runs peaked at or under it');
  });

  it('a clean fit carries no borderline clause, and the safe batch is below the point-estimate batch', async () => {
    const r = await call({ batch_size: 12, gpu: 'T4' }, await template('gpt2'));
    expect(r.gpuVerdict.T4).toBe('fits');
    expect(r.verdict).not.toMatch(/borderline/);
    expect(r.smallestSafeGpu).toBe('T4');
    for (const card of Object.keys(r.maxBatch)) {
      if (r.maxBatch[card] != null) expect(r.maxBatchSafe[card], card).toBeLessThanOrEqual(r.maxBatch[card]);
    }
    expect(r.maxBatchSafe.T4).toBeLessThan(r.maxBatch.T4);
  });

  it('a multi-card request reports the bound for the reading that decided it', async () => {
    const r = await call({ batch_size: 64, gpu: '2xA100-40' }, await template('phi3-mini'));
    expect(r.multiCard.ddp.fits).toBe(true);
    expect(r.multiCard.ddp.verdict).toBe('borderline');
    expect(r.multiCard.sharded.verdict).toBe('fits');
    expect(r.verdict).toMatch(/even under plain DDP.*; borderline per A100-40 under DDP: .*or shard it/);
  });

  it('an unsized graph has no verdicts rather than empty ones', async () => {
    const r = await call({ batch_size: 8 }, { id: 'x', name: 'x', components: [], connections: [] } as unknown as ModelArchitecture);
    expect(r.sized).toBe(false);
    expect(r.gpuVerdict).toBeNull();
    expect(r.upperBound).toBeNull();
    expect(r.smallestSafeGpu).toBeNull();
  });
});

// neurarch #314: the data the job feeds against what the model accepts.
const BERT_CLS = {
  model_type: 'bert', architectures: ['BertForSequenceClassification'], hidden_size: 768, num_hidden_layers: 2,
  num_attention_heads: 12, intermediate_size: 3072, vocab_size: 30522, max_position_embeddings: 512, type_vocab_size: 2,
  id2label: { 0: 'NEGATIVE', 1: 'POSITIVE' },
};

async function cacheHF(dir: string, id: string, config: Record<string, unknown>, keepConfig = true): Promise<ModelArchitecture> {
  const { components, connections } = convertHFConfigToModel(id, config);
  const model = { id: `hf-${id}`, name: id.split('/').pop()!, components, connections } as ModelArchitecture;
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, `${id.replace('/', '__')}.json`), JSON.stringify({
    model, configSource: 'config', cached: false, ...(keepConfig ? { config } : {}),
  }));
  return model;
}

describe('size_training_job: does the data fit the config', () => {
  it('blocks the three first-batch crashes on an hf: model, label count from the config\'s head, with no network', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    await cacheHF(cache, 'acme/bert-cls', BERT_CLS);
    const r = await call({ hf_id: 'acme/bert-cls', batch_size: 8, data: { seq_len: 1024, tokenizer_size: 30524, num_classes: 5 } });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(r.dataFit.checked).toEqual(['sequence length', 'tokenizer size', 'label count']);
    expect(r.dataFit.findings.map((f: { rule: string }) => f.rule)).toEqual(['seq-over-positions', 'tokenizer-over-vocab', 'labels-over-head']);
    expect(r.dataFit.findings[0].title).toBe('Sequences of 1024 tokens are longer than the 512-row position table');
    expect(r.dataFit.findings[1].fix).toBe('Call model.resize_token_embeddings(len(tokenizer)) before training, and budget for the new rows.');
    expect(r.dataFit.contract.numLabels).toBe(2);
    expect(r.dataFit.contract.source.numLabels).toBe('the BertForSequenceClassification head');
    expect(r.verdict).toMatch(/^the data will not load: Sequences of 1024 tokens are longer than the 512-row position table \(and 2 more blocking data-fit findings\); memory: fits/);
  });

  it('reads the config for a model given as model_path hf:, the same as hf_id', async () => {
    const model = await cacheHF(cache, 'acme/bert-cls', BERT_CLS);
    const r = await call({ batch_size: 8, data: { num_classes: 2 } }, model);
    expect(r.dataFit.checked).toEqual(['label count']);
    expect(r.dataFit.findings).toEqual([]);
    expect(r.verdict).toMatch(/^fits/);
  });

  it('a cache entry from before 0.17.3 has no config: the label check says it was skipped, not passed', async () => {
    await cacheHF(cache, 'acme/bert-old', BERT_CLS, false);
    const r = await call({ hf_id: 'acme/bert-old', batch_size: 8, data: { num_classes: 5 } });
    expect(r.dataFit.checked).toEqual([]);
    expect(r.dataFit.skipped[0].check).toBe('label count');
  });

  it('seq_len alone is checked against the position scheme, as the CLI does with --seq-len', async () => {
    await cacheHF(cache, 'acme/bert-cls', BERT_CLS);
    const r = await call({ hf_id: 'acme/bert-cls', batch_size: 8, seq_len: 512 });
    expect(r.dataFit.checked).toEqual(['sequence length']);
    expect(r.dataFit.findings).toEqual([]);
    // data.seq_len is the data's own and wins over the sizing length.
    const r2 = await call({ hf_id: 'acme/bert-cls', batch_size: 8, seq_len: 256, data: { seq_len: 600 } });
    expect(r2.dataFit.findings[0].rule).toBe('seq-over-positions');
  });

  it('no data and no seq_len: no dataFit key, the result is what it was', async () => {
    const r = await call({ batch_size: 32, gpu: 'T4' }, await template('gpt2'));
    expect(r).not.toHaveProperty('dataFit');
  });

  it('refuses a data field that is not a positive integer rather than ignoring it', async () => {
    await expect(call({ batch_size: 8, data: { seq_len: 'long' } }, await template('gpt2'))).rejects.toThrow(/data\.seq_len.*positive integer/);
  });
});

describe('check_design: data and config', () => {
  const check = TOOLS.find(t => t.name === 'check_design')!;
  const run = (args: Record<string, unknown>, model: ModelArchitecture) =>
    check.handler(args, model, { modelPath: '' }) as Promise<any>;

  it('adds dataFit beside the verdict, with the head read from a posted config', async () => {
    const { components, connections } = convertHFConfigToModel('bert-base-uncased', BERT_CLS);
    const model = { id: 'local', name: 'bert', components, connections } as unknown as ModelArchitecture;
    const without = await run({ data: { num_classes: 5 } }, model);
    expect(without.dataFit.skipped[0].check).toBe('label count');
    const r = await run({ data: { num_classes: 5, tokenizer_size: 30522 }, config: BERT_CLS }, model);
    expect(r.verdict).toBeDefined();
    expect(r.dataFit.checked).toEqual(['tokenizer size', 'label count']);
    expect(r.dataFit.findings.map((f: { rule: string }) => f.rule)).toEqual(['labels-over-head']);
  });

  it('reads the config itself for an hf: model', async () => {
    const model = await cacheHF(cache, 'acme/bert-cls', BERT_CLS);
    const r = await run({ data: { num_classes: 3 } }, model);
    expect(r.dataFit.findings[0].rule).toBe('labels-over-head');
  });

  it('without data it returns what it did before', async () => {
    const r = await run({}, await template('gpt2'));
    expect(r).not.toHaveProperty('dataFit');
  });
});
