/**
 * The plan bundle is a copy of the Neurarch app's lib/modelPlan.bundle.mjs,
 * and size_training_job prints its numbers as calibrated measurements. So this
 * file pins the numbers, not only the exports: a peak computed by the app's own
 * `buildPlan` (src/utils/modelPlan.ts at neurarch-ai/neurarch a1bac74e, the
 * six-term refit on 178 runs, #306) on two of the app's templates
 * (public/templates/gpt2 and phi3-mini), copied into src/test/templates/ with
 * only the dashes in their prose notes changed. Every expected value below was
 * printed by running that source and this bundle on the same file; the two
 * agreed to the last digit. A stale copy fails here: the three-term bundle
 * this replaced puts gpt2 at 10.38 GB under eager and phi3-mini at 47.72 GB.
 *
 * The plan defaults to SDPA attention (neurarch #285), which stores no score
 * matrix, so the eager numbers are pinned under attention: 'eager' and the
 * SDPA defaults beside them, each the eager peak minus attentionScoresGbFor
 * at the same batch.
 */
import { describe, it, expect } from 'vitest';
import { readFile } from 'node:fs/promises';
import * as plan from './plan.bundle.mjs';

async function template(name: string): Promise<unknown> {
  return JSON.parse(await readFile(new URL(`../test/templates/${name}.neurarch.json`, import.meta.url), 'utf-8'));
}

describe('vendored plan contract', () => {
  it('exports the functions this server declares', () => {
    for (const fn of ['buildPlan', 'attentionScoresGbFor', 'maxBatchFor', 'deviceMapFor', 'calibrationSentence'] as const) {
      expect(typeof plan[fn], fn).toBe('function');
    }
    expect([plan.OPTIMIZER_K, plan.OUTPUT_K, plan.ATTENTION_K, plan.INTERNAL_K, plan.RECURRENT_K, plan.TRANSIENT_K])
      .toEqual([1.2, 0.88, 3.64, 1.34, 5.2, 2.02]);
    expect(plan.FIT_HEADROOM).toBe(1.15);
    expect(plan.MEMORY_CALIBRATION).toEqual({
      fittedRuns: 178, architectures: 41, heldOutMedianPct: 7, flagged: 113, noFit: 131, afterFitRuns: 25, afterFitMedianPct: 14,
      upperK90: 1.17, upperK95: 1.43, upperRuns: 91,
    });
    for (const fn of ['upperPeakGb', 'fitVerdict', 'smallestSafeCardFor', 'inputContractOf', 'readDataSpec', 'checkDataFit'] as const) {
      expect(typeof plan[fn], fn).toBe('function');
    }
    expect([plan.UPPER_K, plan.UPPER_COVERAGE, plan.UPPER_FLOOR_GB]).toEqual([1.43, 0.95, 1]);
    for (const fn of ['checkGraphEdit', 'checkConfigEdit', 'checkLoraTargets', 'looksLikeHFConfig', 'loraTargetsFor'] as const) {
      expect(typeof plan[fn], fn).toBe('function');
    }
    expect(plan.EDIT_RULE_IDS).toEqual([
      'vocab-resize-incomplete', 'weight-tying-lost', 'attention-config-drift', 'rope-context-drift',
      'lora-target-no-match', 'lora-target-partial', 'lora-target-wrapper',
    ]);
    for (const id of plan.EDIT_RULE_IDS) expect(Array.isArray(plan.EDIT_RULE_SEEN_IN[id]), id).toBe(true);
    for (const fn of ['parseOomMessage', 'diagnoseOom', 'renderOomText', 'oomFactsForModel'] as const) {
      expect(typeof plan[fn], fn).toBe('function');
    }
    expect([plan.DEFAULT_OOM_TRAIN_BATCH, plan.DEFAULT_GEN_TOKENS, plan.DEFAULT_INFERENCE_BATCH]).toEqual([32, 256, 1]);
    for (const fn of ['readLayout', 'readHistogram', 'checkParallelLayout', 'renderLayoutText', 'readServeSpec', 'parseServeCommand', 'checkServeConfig', 'renderServeText'] as const) {
      expect(typeof plan[fn], fn).toBe('function');
    }
    // The sentences check_parallel_layout and check_serving_config return as `basis`:
    // a stale bundle (before neurarch #325's bandwidth term) says FLOPs only.
    expect(plan.LAYOUT_BASIS).toMatch(/^Derived from the graph, not measured\./);
    expect(plan.LAYOUT_BASIS).toMatch(/80% of its memory bandwidth/);
    expect(plan.SERVE_BASIS).toMatch(/^Predicted from config\.json, the engine flags and the card, not measured\./);
    expect(Array.isArray(plan.QUANT_SUPPORT)).toBe(true);
  });

  it('needs nothing from the outside world', async () => {
    const src = await readFile(new URL('./plan.bundle.mjs', import.meta.url), 'utf-8');
    expect(src).not.toMatch(/^\s*import\s/m);
    expect(src).not.toMatch(/\brequire\(/);
    expect(src).not.toMatch(/\bfetch\(/);
  });

  // Expected values: buildPlan(template, { run: { batchSize: 32 }, attention })
  // in the app repository at a1bac74e, printed to full precision.
  it('matches the app\'s buildPlan on the gpt2 template under eager attention', async () => {
    const m = await template('gpt2');
    const p = plan.buildPlan(m, { run: { batchSize: 32 }, attention: 'eager' });
    expect(p.model.params).toBe(84331345);
    expect(p.cost.peakGb).toBeCloseTo(15.181297531306669, 9);
    expect(p.cost.activationGb).toBeCloseTo(13.562135707306668, 9);
    expect(plan.attentionScoresGbFor(m as never, 32)).toBeCloseTo(2.9313151795200003, 9);
    expect(p.cost.smallestFitGpu).toBe('L4 (24GB)');
    expect(p.cost.gpuFits.T4).toBe(false);
    expect(p.cost.maxBatch).toEqual({ T4: 29, A100_40: 78, H100_80: 160 });
    expect(p.cost.twoCardSplit?.gpu).toBe('T4 (16GB)');
    expect(p.cost.twoCardSplit?.fits).toBe(true);
  });

  it('matches the app\'s buildPlan on the phi3-mini template under eager attention (needs an 80 GB card)', async () => {
    const m = await template('phi3-mini');
    const p = plan.buildPlan(m, { run: { batchSize: 32 }, attention: 'eager' });
    expect(p.model.params).toBe(310288704);
    expect(p.cost.peakGb).toBeCloseTo(66.50700668927999, 9);
    expect(p.cost.activationGb).toBeCloseTo(60.54946357247999, 9);
    expect(plan.attentionScoresGbFor(m as never, 32)).toBeCloseTo(40.129590722559996, 9);
    expect(p.cost.smallestFitGpu).toBe('A100 (80GB)');
    expect(p.cost.gpuFits).toEqual({ T4: false, A100_40: false, H100_80: true });
    expect(p.cost.maxBatch).toEqual({ T4: 4, A100_40: 15, H100_80: 33 });
    expect(p.cost.twoCardSplit).toBeNull();
  });

  it('defaults to SDPA: the eager peak minus the plan\'s own attention-score term', async () => {
    const g = await template('gpt2');
    const pg = plan.buildPlan(g, { run: { batchSize: 32 } });
    expect(pg.cost.attention).toBe('sdpa');
    expect(pg.cost.seqLen).toBe(1024);
    expect(pg.cost.peakGb).toBeCloseTo(12.249982351786668, 9);
    expect(pg.cost.peakGb).toBeCloseTo(15.181297531306669 - plan.attentionScoresGbFor(g as never, 32), 9);
    expect(pg.cost.activationGb).toBeCloseTo(10.630820527786668, 9);
    expect(pg.cost.smallestFitGpu).toBe('T4 (16GB)');
    expect(pg.cost.maxBatch).toEqual({ T4: 37, A100_40: 99, H100_80: 204 });

    const m = await template('phi3-mini');
    const pm = plan.buildPlan(m, { run: { batchSize: 32 } });
    expect(pm.cost.attention).toBe('sdpa');
    expect(pm.cost.seqLen).toBe(2048);
    expect(pm.cost.peakGb).toBeCloseTo(26.377415966719997, 9);
    expect(pm.cost.peakGb).toBeCloseTo(66.50700668927999 - plan.attentionScoresGbFor(m as never, 32), 9);
    expect(pm.cost.activationGb).toBeCloseTo(20.419872849919997, 9);
    expect(pm.cost.gpuFits.T4).toBe(false);
    expect(pm.cost.smallestFitGpu).toBe('L40S (48GB)');
    expect(pm.cost.maxBatch).toEqual({ T4: 12, A100_40: 45, H100_80: 99 });
    expect(pm.cost.twoCardSplit?.gpu).toBe('L4 (24GB)');
  });

  it('the footprint term is OPTIMIZER_K x 16 bytes per parameter, so sizeJob\'s linear split holds', async () => {
    for (const name of ['gpt2', 'phi3-mini']) {
      const p = plan.buildPlan(await template(name), { run: { batchSize: 32 }, attention: 'eager' });
      expect(p.cost.peakGb - p.cost.activationGb!, name).toBeCloseTo((p.model.params * 16 * plan.OPTIMIZER_K) / 1e9, 6);
      const p64 = plan.buildPlan(await template(name), { run: { batchSize: 64 }, attention: 'eager' });
      expect(p64.cost.activationGb!, name).toBeCloseTo(2 * p.cost.activationGb!, 6);
    }
  });

  it('calibrationSentence is written from MEMORY_CALIBRATION', () => {
    const c = plan.MEMORY_CALIBRATION;
    const s = plan.calibrationSentence();
    for (const v of [c.fittedRuns, c.architectures, `${c.heldOutMedianPct}%`, `${c.flagged} of the ${c.noFit}`, c.afterFitRuns, `${c.afterFitMedianPct}%`]) {
      expect(s).toContain(String(v));
    }
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
