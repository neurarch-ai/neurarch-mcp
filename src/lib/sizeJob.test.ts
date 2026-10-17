import { describe, it, expect } from 'vitest';
import { readFile } from 'node:fs/promises';
import { sizeJob, parseGpu, withSeqLen, CARDS, ACCURACY_SENTENCE } from './sizeJob.js';
import { buildPlan } from '../vendor/plan.bundle.mjs';
import { loadZooModel } from '../sources.js';
import type { ModelArchitecture } from './types.js';

async function template(name: string): Promise<ModelArchitecture> {
  return JSON.parse(await readFile(new URL(`../test/templates/${name}.neurarch.json`, import.meta.url), 'utf-8'));
}

describe('sizeJob is the plan, not a second computation', () => {
  it('under eager attention it returns the plan\'s own peak, max batch and split', async () => {
    for (const name of ['gpt2', 'phi3-mini']) {
      const m = await template(name);
      const p = buildPlan(m, { run: { batchSize: 32 } });
      const s = sizeJob(m, { batchSize: 32, attention: 'eager' });
      expect(s.memory!.peakGb, name).toBeCloseTo(p.cost.peakGb, 2);
      expect(s.maxBatch!.T4).toBe(p.cost.maxBatch.T4);
      expect(s.maxBatch!['A100-40']).toBe(p.cost.maxBatch.A100_40);
      expect(s.maxBatch!['H100-80']).toBe(p.cost.maxBatch.H100_80);
      expect(CARDS.find(c => c.id === s.smallestFitGpu)?.planName).toBe(p.cost.smallestFitGpu);
      expect(s.twoCardSplit?.gpu ?? null).toBe(p.cost.twoCardSplit ? CARDS.find(c => c.planName === p.cost.twoCardSplit!.gpu)!.id : null);
    }
  });

  it('every card name the server sends is one the bundle prices (an unknown name silently prices an A10G)', async () => {
    const m = await template('gpt2');
    const fallback = buildPlan(m, { run: { batchSize: 32, gpu: 'no such card' } }).cost.estCostUsd;
    for (const c of CARDS) {
      if (c.id === 'A10G') continue;
      expect(buildPlan(m, { run: { batchSize: 32, gpu: c.planName } }).cost.estCostUsd, c.id).not.toBe(fallback);
    }
  });
});

describe('a template that fits a T4 and one that does not', () => {
  it('gpt2 at batch 32 fits a T4, and says how much headroom', async () => {
    const s = sizeJob(await template('gpt2'), { batchSize: 32, gpu: 'T4' });
    expect(s.sized).toBe(true);
    expect(s.fits).toBe(true);
    expect(s.verdict).toMatch(/^fits a T4 with \d+% headroom at batch 32, seq 1024/);
    expect(s.smallestFitGpu).toBe('T4');
    expect(s.twoCardSplit).toBeNull();
    expect(s.cost?.gpu).toBe('T4');
    expect(s.cost?.usd).toBeGreaterThan(0);
    expect(s.accuracy).toBe(ACCURACY_SENTENCE);
    expect(s.accuracy).toMatch(/141 measured runs across 44 architectures.*median error is 8%.*98%/);
  });

  it('phi3-mini at batch 32 does not fit a T4, and names the card and the batch that would', async () => {
    const s = sizeJob(await template('phi3-mini'), { batchSize: 32, gpu: 'T4' });
    expect(s.fits).toBe(false);
    expect(s.verdict).toMatch(/^will not fit a T4: needs ~\d+ GB with headroom/);
    expect(s.verdict).toContain(`use ${s.smallestFitGpu}`);
    expect(s.verdict).toContain(`batch <= ${s.maxBatch!.T4}`);
    expect(s.maxBatch!.T4).toBeLessThan(32);
  });

  it('the batch it offers fits, and one more does not', async () => {
    const m = await template('phi3-mini');
    const mb = sizeJob(m, { batchSize: 32, gpu: 'T4' }).maxBatch!.T4!;
    expect(sizeJob(m, { batchSize: mb, gpu: 'T4' }).fits).toBe(true);
    expect(sizeJob(m, { batchSize: mb + 1, gpu: 'T4' }).fits).toBe(false);
  });

  it('under eager attention phi3-mini needs more than an A100-40 and gets the plan\'s two-card split', async () => {
    const s = sizeJob(await template('phi3-mini'), { batchSize: 32, gpu: 'A100-40', attention: 'eager' });
    expect(s.fits).toBe(false);
    expect(s.verdict).toMatch(/^will not fit an A100-40: needs ~55 GB with headroom \(peak 48 GB\).*use A100-80 or batch <= 22$/);
    expect(s.twoCardSplit?.gpu).toBe('A100-40');
    expect(s.twoCardSplit?.fits).toBe(true);
  });
});

describe('the attention flag', () => {
  it('sdpa and flash drop the stored attention scores, eager keeps them', async () => {
    const m = await template('phi3-mini');
    const eager = sizeJob(m, { batchSize: 32, attention: 'eager' });
    const sdpa = sizeJob(m, { batchSize: 32 });
    const flash = sizeJob(m, { batchSize: 32, attention: 'flash' });
    expect(sdpa.job.attention).toBe('sdpa');
    expect(eager.memory!.peakGb - sdpa.memory!.peakGb).toBeCloseTo(29.81, 1);
    expect(sdpa.memory!.attentionScoresDroppedGb).toBeCloseTo(29.81, 1);
    expect(eager.memory!.attentionScoresGb).toBeCloseTo(29.81, 1);
    expect(flash.memory!.peakGb).toBe(sdpa.memory!.peakGb);
    // It changes the answer, not just the number: an A100-80 under eager, an L4 under SDPA.
    expect(eager.smallestFitGpu).toBe('A100-80');
    expect(sdpa.smallestFitGpu).toBe('L4');
    expect(sdpa.caveats.join(' ')).toMatch(/Under eager attention the same job keeps another 30 GB/);
  });
});

describe('sequence length, multi-card requests, and graphs it refuses', () => {
  it('seq_len overrides a token model\'s input and lowers the peak', async () => {
    const m = await template('gpt2');
    const full = sizeJob(m, { batchSize: 32 });
    const half = sizeJob(m, { batchSize: 32, seqLen: 512 });
    expect(half.job.seqLen).toBe(512);
    expect(half.memory!.peakGb).toBeLessThan(full.memory!.peakGb);
    expect(withSeqLen(m, null).seqLen).toBe(1024);
  });

  it('seq_len on a vision model is ignored and says so', () => {
    const vision = {
      id: 'v', name: 'v',
      components: [
        { id: 'i', type: 'input', name: 'in', params: { shape: [3, 32, 32] }, position: { x: 0, y: 0 } },
        { id: 'c', type: 'conv2d', name: 'conv', params: { inChannels: 3, outChannels: 16, kernelSize: 3, padding: 1 }, position: { x: 0, y: 1 } },
        { id: 'o', type: 'output', name: 'out', params: {}, position: { x: 0, y: 2 } },
      ],
      connections: [{ id: 'e1', from: 'i', to: 'c' }, { id: 'e2', from: 'c', to: 'o' }],
    } as unknown as ModelArchitecture;
    const s = sizeJob(vision, { batchSize: 8, seqLen: 512 });
    expect(s.job.seqLen).toBeNull();
    expect(s.caveats.join(' ')).toMatch(/seq_len was ignored/);
  });

  it('reads an N-card request both as plain DDP and as sharded', async () => {
    const s = sizeJob(await template('phi3-mini'), { batchSize: 256, gpu: '8xH100' });
    expect(s.job.gpu).toBe('8xH100-80');
    expect(s.multiCard?.ddp.perDeviceBatch).toBe(32);
    expect(s.multiCard?.ddp.fits).toBe(true);
    expect(s.multiCard?.sharded.capacityGb).toBe(640);
    expect(s.verdict).toMatch(/^fits 8xH100-80 even under plain DDP/);
    expect(s.cost?.cards).toBe(8);
  });

  it('refuses to size a graph that feeds a hidden-state tensor into a token embedding', async () => {
    // The bundled gpt2-small reference declares its input as [1, 512, 768] and
    // feeds it to an embedding; the plan would put it at over 2 TB.
    const s = sizeJob(await loadZooModel('gpt2-small'), { batchSize: 8 });
    expect(s.sized).toBe(false);
    expect(s.memory).toBeNull();
    expect(s.verdict).toMatch(/^cannot size this graph: .*token embedding/);
  });

  it('parses the ways a scheduler writes a card, and refuses a bare A100', () => {
    expect(parseGpu('8xH100')).toMatchObject({ card: { id: 'H100-80' }, count: 8 });
    expect(parseGpu('A100-40')).toMatchObject({ card: { id: 'A100-40', gb: 40 }, count: 1 });
    expect(parseGpu('2 x a100_80gb')).toMatchObject({ card: { id: 'A100-80' }, count: 2 });
    expect(parseGpu('A100')).toEqual({ error: expect.stringMatching(/ambiguous/) });
    expect(parseGpu('H200')).toEqual({ error: expect.stringMatching(/unknown card/) });
    expect(() => sizeJob({ id: 'x', name: 'x', components: [], connections: [] } as unknown as ModelArchitecture, { batchSize: 1, gpu: 'A100' }))
      .toThrow(/ambiguous/);
  });
});
