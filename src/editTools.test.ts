import { describe, it, expect } from 'vitest';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { TOOLS } from './tools.js';
import { CORE_TOOLS, listedTools } from './cli.js';
import { sizeJob, ACCURACY_SENTENCE } from './lib/sizeJob.js';
import { checkGraphEdit, checkConfigEdit, EDIT_RULE_SEEN_IN } from './vendor/plan.bundle.mjs';
import type { ModelArchitecture } from './lib/types.js';

const tool = TOOLS.find(t => t.name === 'check_edit')!;
const call = (args: Record<string, unknown>, model?: ModelArchitecture) =>
  tool.handler(args, model as ModelArchitecture, { modelPath: '' }) as Promise<any>;

async function template(name: string): Promise<ModelArchitecture> {
  return JSON.parse(await readFile(new URL(`./test/templates/${name}.neurarch.json`, import.meta.url), 'utf-8'));
}

/** gpt2 with the input embedding resized by 8 tokens and the head left behind, the transformers#45276 shape. */
async function resizedGpt2(): Promise<ModelArchitecture> {
  const m = structuredClone(await template('gpt2'));
  const emb = m.components.find(c => c.type === 'embedding')!;
  (emb.params as Record<string, number>).numEmbeddings += 8;
  return m;
}

const LLAMA = {
  model_type: 'llama', architectures: ['LlamaForCausalLM'], vocab_size: 32000, hidden_size: 2048,
  num_attention_heads: 16, num_key_value_heads: 4, intermediate_size: 5632, num_hidden_layers: 4,
  max_position_embeddings: 4096, tie_word_embeddings: false,
};

describe('check_edit, the tool', () => {
  it('is in the core listing and says when to call it', () => {
    expect(CORE_TOOLS).toContain('check_edit');
    expect(tool.description).toMatch(/^Call after editing a model's config or architecture and before training/);
    const listed = listedTools(false, 'core').find(t => t.name === 'check_edit')!;
    expect(listed.description).toMatch(/^Call after editing a model's config or architecture, before training/);
    expect(Object.keys(listed.inputSchema.properties as object)).toEqual(expect.arrayContaining(['base_model_path', 'base_config', 'config', 'batch', 'seq_len', 'gpu', 'model_path']));
  });

  it('names a vocabulary resize that left the head behind, with the bundle\'s own finding and where it was reported', async () => {
    const base = await template('gpt2');
    const head = await resizedGpt2();
    const dir = await mkdtemp(join(tmpdir(), 'nx-edit-'));
    try {
      const basePath = join(dir, 'base.neurarch.json');
      await writeFile(basePath, JSON.stringify(base));
      const r = await call({ base_model_path: basePath, batch: 8, gpu: 'T4' }, head);
      expect(r.checked).toEqual(['graph']);
      const expected = checkGraphEdit(base, head);
      expect(expected.map(f => f.rule)).toEqual(['vocab-resize-incomplete']);
      expect(r.findings.map((f: { rule: string }) => f.rule)).toEqual(['vocab-resize-incomplete']);
      expect(r.findings[0].subjects).toEqual(['lm_head']);
      expect(r.findings[0].severity).toBe('block');
      expect(r.findings[0].seenIn).toEqual(EDIT_RULE_SEEN_IN['vocab-resize-incomplete']);
      expect(r.ok).toBe(false);
      expect(r.verdict).toMatch(/^1 edit finding \(1 block\); peak /);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('reports the memory delta as sizeJob\'s two peaks, and says when the edit pushes the job off the card', async () => {
    const base = await template('gpt2');
    // The largest batch at which the base fits a T4; the edit doubles the context (1024 -> 2048 tokens).
    const mb = sizeJob(base, { batchSize: 1, gpu: 'T4' }).maxBatch!.T4!;
    const head = structuredClone(base);
    const inp = head.components.find(c => c.type === 'input')!;
    const shape = [...((inp.params as { shape: number[] }).shape)];
    shape[shape.length - 1] *= 2;
    inp.params = { ...inp.params, shape };
    (inp as { inputShape?: number[] }).inputShape = shape;
    const before = sizeJob(base, { batchSize: mb, gpu: 'T4' });
    const after = sizeJob(head, { batchSize: mb, gpu: 'T4' });
    expect(before.fits).toBe(true);
    expect(after.fits).toBe(false);
    const dir = await mkdtemp(join(tmpdir(), 'nx-edit-'));
    try {
      const basePath = join(dir, 'base.neurarch.json');
      await writeFile(basePath, JSON.stringify(base));
      const out = await call({ base_model_path: basePath, batch: mb, gpu: 'T4' }, head);
      expect(out.findings).toEqual([]);
      expect(out.memory.gpu).toBe('T4');
      expect(out.memory.before.peakGb).toBe(before.memory!.peakGb);
      expect(out.memory.after.peakGb).toBe(after.memory!.peakGb);
      expect(out.memory.deltaGb).toBeCloseTo(after.memory!.peakGb - before.memory!.peakGb, 2);
      expect(out.memory.after.maxBatchOnCard).toBe(after.maxBatch!.T4);
      expect(out.memory.summary).toMatch(new RegExp(`no longer fits T4, batch <= ${after.maxBatch!.T4} does`));
      expect(out.ok).toBe(false);
      expect(out.accuracy).toBe(ACCURACY_SENTENCE);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('reads a config pair: head drift, divisibility under the lint id, RoPE without scaling', async () => {
    const changed = { ...LLAMA, num_attention_heads: 12, max_position_embeddings: 16384 };
    const r = await call({ base_config: LLAMA, config: JSON.stringify(changed), batch: 4, seq_len: 1024 });
    expect(r.checked).toEqual(['config']);
    const rules = r.findings.map((f: { rule: string }) => f.rule);
    expect(rules).toEqual(checkConfigEdit(LLAMA, changed).map(f => f.rule));
    expect(rules).toEqual(expect.arrayContaining(['head-dim-divisibility', 'attention-config-drift', 'rope-context-drift']));
    expect(r.findings.find((f: { rule: string }) => f.rule === 'head-dim-divisibility').seenIn).toEqual([]);
    expect(r.memory.before.peakGb).toBeGreaterThan(0);
    expect(r.memory.seqLen).toBe(1024);
    expect(r.ok).toBe(false);
  });

  it('a clean config edit returns no findings and still compares memory', async () => {
    const changed = { ...LLAMA, num_hidden_layers: 8 };
    const r = await call({ base_config: LLAMA, config: changed, batch: 4 });
    expect(r.findings).toEqual([]);
    expect(r.ok).toBe(true);
    expect(r.memory.after.peakGb).toBeGreaterThan(r.memory.before.peakGb);
    expect(r.verdict).toMatch(/^no edit findings in config; peak /);
  });

  it('reads a config.json from a path, and refuses JSON that is not a Hugging Face config', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'nx-edit-'));
    try {
      const p = join(dir, 'config.json');
      await writeFile(p, JSON.stringify(LLAMA));
      const r = await call({ base_config: p, config: { ...LLAMA, vocab_size: 32008 } });
      expect(r.checked).toEqual(['config']);
      await expect(call({ base_config: { foo: 1 }, config: LLAMA })).rejects.toThrow(/does not read as a Hugging Face config/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('matches LoRA target_modules the way PEFT does', async () => {
    const r = await call({ base_config: LLAMA, config: LLAMA, lora_target_modules: ['q_proj', 'qkv_bogus'] });
    expect(r.checked).toEqual(['config', 'lora']);
    expect(r.findings.map((f: { rule: string }) => f.rule)).toEqual(['lora-target-partial']);
    expect(r.findings[0].subjects).toEqual(['qkv_bogus']);
    expect(r.lora.matched).toBeGreaterThan(0);
  });

  it('needs both sides', async () => {
    await expect(call({ config: LLAMA })).rejects.toThrow(/before the edit/);
    await expect(call({ base_config: LLAMA })).rejects.toThrow(/changed model/);
    await expect(call({ base_config: LLAMA, config: LLAMA, batch: 0 })).rejects.toThrow(/batch must be a positive integer/);
  });
});
