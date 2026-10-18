/**
 * The plan bundle is a copy of the Neurarch app's lib/modelPlan.bundle.mjs,
 * and size_training_job prints its numbers as calibrated measurements. So this
 * file pins the numbers, not only the exports: a peak computed by the app's own
 * `buildPlan` (src/utils/modelPlan.ts at neurarch-ai/neurarch 11c02b72, the
 * 141-run refit) on two of the app's templates (public/templates/gpt2 and
 * phi3-mini), copied into src/test/templates/ with only the dashes in their
 * prose notes changed. A stale copy fails here: checked against the pre-refit
 * bundle from f0acf8a0^, which has no attentionScoresGbFor and puts gpt2 at
 * 13.38 GB and phi3-mini at 77.05 GB, so every test in this file but the
 * import check went red.
 *
 * Since neurarch #285 (582aafe3) the plan defaults to SDPA attention, which
 * stores no score matrix. The 11c02b72 numbers are eager numbers, so they are
 * pinned under attention: 'eager', where the bundle at 96254170 reproduces
 * them to the last digit; the SDPA defaults are pinned beside them, and each
 * is the eager peak minus attentionScoresGbFor at the same batch.
 */
import { describe, it, expect } from 'vitest';
import { readFile } from 'node:fs/promises';
import * as plan from './plan.bundle.mjs';

async function template(name: string): Promise<unknown> {
  return JSON.parse(await readFile(new URL(`../test/templates/${name}.neurarch.json`, import.meta.url), 'utf-8'));
}

describe('vendored plan contract', () => {
  it('exports the functions this server declares', () => {
    for (const fn of ['buildPlan', 'attentionScoresGbFor', 'maxBatchFor', 'deviceMapFor'] as const) {
      expect(typeof plan[fn], fn).toBe('function');
    }
    expect([plan.OPTIMIZER_K, plan.OUTPUT_K, plan.ATTENTION_K]).toEqual([1.26, 1.57, 3.47]);
    for (const fn of ['parseOomMessage', 'diagnoseOom', 'renderOomText', 'oomFactsForModel'] as const) {
      expect(typeof plan[fn], fn).toBe('function');
    }
    expect([plan.DEFAULT_OOM_TRAIN_BATCH, plan.DEFAULT_GEN_TOKENS, plan.DEFAULT_INFERENCE_BATCH]).toEqual([32, 256, 1]);
  });

  it('needs nothing from the outside world', async () => {
    const src = await readFile(new URL('./plan.bundle.mjs', import.meta.url), 'utf-8');
    expect(src).not.toMatch(/^\s*import\s/m);
    expect(src).not.toMatch(/\brequire\(/);
    expect(src).not.toMatch(/\bfetch\(/);
  });

  // Expected values: buildPlan(template, { run: { batchSize: 32 } }) in the app
  // repository at 11c02b72 (eager by then, eager by name now), printed to full
  // precision.
  it('matches the app\'s buildPlan on the gpt2 template under eager attention (fits a T4)', async () => {
    const p = plan.buildPlan(await template('gpt2'), { run: { batchSize: 32 }, attention: 'eager' });
    expect(p.model.params).toBe(84331345);
    expect(p.cost.peakGb).toBeCloseTo(10.376738319040001, 9);
    expect(p.cost.activationGb).toBeCloseTo(8.676618403840001, 9);
    expect(p.cost.smallestFitGpu).toBe('T4 (16GB)');
    expect(p.cost.maxBatch).toEqual({ T4: 45, A100_40: 122, H100_80: 250 });
    expect(p.cost.twoCardSplit).toBeNull();
  });

  it('matches the app\'s buildPlan on the phi3-mini template under eager attention (does not fit a T4)', async () => {
    const m = await template('phi3-mini');
    const p = plan.buildPlan(m, { run: { batchSize: 32 }, attention: 'eager' });
    expect(p.model.params).toBe(310288704);
    expect(p.cost.peakGb).toBeCloseTo(47.71804469248001, 9);
    expect(p.cost.activationGb).toBeCloseTo(41.462624419840004, 9);
    expect(plan.attentionScoresGbFor(m as never, 32)).toBeCloseTo(29.807073034240002, 9);
    expect(p.cost.smallestFitGpu).toBe('A100 (80GB)');
    expect(p.cost.gpuFits.T4).toBe(false);
    expect(p.cost.maxBatch).toEqual({ T4: 5, A100_40: 22, H100_80: 48 });
    expect(p.cost.twoCardSplit?.gpu).toBe('A100 (40GB)');
    expect(p.cost.twoCardSplit?.fits).toBe(true);
  });

  // Expected values: buildPlan(template, { run: { batchSize: 32 } }) in the app
  // repository at 96254170, printed to full precision. SDPA is the default.
  it('defaults to SDPA: the eager peak minus the plan\'s own attention-score term', async () => {
    const g = await template('gpt2');
    const pg = plan.buildPlan(g, { run: { batchSize: 32 } });
    expect(pg.cost.attention).toBe('sdpa');
    expect(pg.cost.seqLen).toBe(1024);
    expect(pg.cost.peakGb).toBeCloseTo(7.582325222080001, 9);
    expect(pg.cost.peakGb).toBeCloseTo(10.376738319040001 - plan.attentionScoresGbFor(g as never, 32), 9);
    expect(pg.cost.maxBatch).toEqual({ T4: 66, A100_40: 179, H100_80: 369 });

    const m = await template('phi3-mini');
    const pm = plan.buildPlan(m, { run: { batchSize: 32 } });
    expect(pm.cost.attention).toBe('sdpa');
    expect(pm.cost.seqLen).toBe(2048);
    expect(pm.cost.peakGb).toBeCloseTo(17.91097165824, 9);
    expect(pm.cost.peakGb).toBeCloseTo(47.71804469248001 - plan.attentionScoresGbFor(m as never, 32), 9);
    expect(pm.cost.activationGb).toBeCloseTo(11.6555513856, 9);
    expect(pm.cost.gpuFits.T4).toBe(false);
    expect(pm.cost.smallestFitGpu).toBe('L4 (24GB)');
    expect(pm.cost.maxBatch).toEqual({ T4: 21, A100_40: 78, H100_80: 173 });
  });

  it('prices on the card it is given, so the server\'s card names match the bundle\'s', async () => {
    const m = await template('gpt2');
    const t4 = plan.buildPlan(m, { run: { batchSize: 32, gpu: 'T4 (16GB)' } }).cost.estCostUsd;
    const h100 = plan.buildPlan(m, { run: { batchSize: 32, gpu: 'H100 (80GB)' } }).cost.estCostUsd;
    const unknown = plan.buildPlan(m, { run: { batchSize: 32, gpu: 'no such card' } }).cost.estCostUsd;
    const a10g = plan.buildPlan(m, { run: { batchSize: 32, gpu: 'A10G (24GB)' } }).cost.estCostUsd;
    expect(t4).not.toBe(h100);
    // An unknown name silently prices on the A10G; the server must never send one.
    expect(unknown).toBe(a10g);
  });
});
