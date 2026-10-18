/**
 * Sizing one training job before it takes a GPU reservation.
 *
 * Every memory number here comes from the vendored plan bundle (the app's
 * `buildPlan`, the code behind POST /api/v1/plan), so this file does no memory
 * arithmetic of its own beyond two things the plan does not do for a caller:
 *
 *   1. Removing the stored-attention-score term when the job runs SDPA or flash
 *      attention. The plan is asked for eager attention (its `peakGb` then
 *      includes the scores); SDPA (the PyTorch
 *      and Hugging Face default) and flash keep no score matrix for backward.
 *      The term removed is the plan's own (`attentionScoresGbFor`), so the two
 *      cannot drift.
 *   2. Answering for the card the caller named, at the batch the caller named,
 *      including N-card requests.
 *
 * The sequence-length override, the refusal of a graph that feeds an image
 * tensor into a token embedding, and the multi-card reading are the same rules
 * the app's `npx neurarch rightsize` applies (scripts/rightsize.ts there), so
 * the agent and the platform lead's report agree on one job.
 */
import type { MLComponent, ModelArchitecture } from './types.js';
import {
  buildPlan,
  attentionScoresGbFor,
  maxBatchFor,
  deviceMapFor,
  OPTIMIZER_K,
  OUTPUT_K,
  ATTENTION_K,
  INTERNAL_K,
  RECURRENT_K,
  TRANSIENT_K,
  calibrationSentence,
  FIT_HEADROOM,
  DEFAULT_SEQ_LEN as PLAN_DEFAULT_SEQ_LEN,
  type ModelPlan,
} from '../vendor/plan.bundle.mjs';

export type Attention = 'sdpa' | 'flash' | 'eager';

/** Same headroom every card check in the plan uses: allocator fragmentation and the CUDA context. */
export const HEADROOM = FIT_HEADROOM;

/** Sequence length a token model is sized at when the caller does not say (its context, if shorter). */
export const DEFAULT_SEQ_LEN = PLAN_DEFAULT_SEQ_LEN;

export interface Card {
  /** What the agent writes and what the result says. */
  id: string;
  gb: number;
  /** The bundle's own name for the card, which is what prices a run on it. */
  planName: string;
}

/**
 * The cards the plan can price, in the plan's own order: cheapest per hour
 * first. "Smallest card that fits" means the first card in this list that
 * holds the peak, which is what the plan's `smallestFitGpu` means too.
 */
export const CARDS: Card[] = [
  { id: 'T4', gb: 16, planName: 'T4 (16GB)' },
  { id: 'L4', gb: 24, planName: 'L4 (24GB)' },
  { id: 'A10G', gb: 24, planName: 'A10G (24GB)' },
  { id: 'L40S', gb: 48, planName: 'L40S (48GB)' },
  { id: 'A100-40', gb: 40, planName: 'A100 (40GB)' },
  { id: 'A100-80', gb: 80, planName: 'A100 (80GB)' },
  { id: 'H100-80', gb: 80, planName: 'H100 (80GB)' },
];

const CARD_ALIASES: Array<[RegExp, string]> = [
  [/^(nvidia)?t4(16(gb)?)?$/, 'T4'],
  [/^(nvidia)?l4(24(gb)?)?$/, 'L4'],
  [/^(nvidia)?a10g?(24(gb)?)?$/, 'A10G'],
  [/^(nvidia)?l40s?(48(gb)?)?$/, 'L40S'],
  [/^(nvidia)?a100(sxm|pcie)?40(gb)?$/, 'A100-40'],
  [/^(nvidia)?a100(sxm|pcie)?80(gb)?$/, 'A100-80'],
  [/^(nvidia)?h100(sxm|pcie|nvl)?(80(gb)?)?$/, 'H100-80'],
];

/**
 * "A100-40", "8xH100", "2 x a100_80gb" -> { card, count }. A bare "A100" is
 * refused rather than guessed: the two A100s differ by 2x in memory, which is
 * the whole question being asked.
 */
export function parseGpu(raw: string): { card: Card; count: number } | { error: string } {
  const s = String(raw ?? '').trim();
  if (!s) return { error: 'gpu is empty' };
  let count = 1;
  let name = s;
  const m = /^(\d+)\s*[x×*]\s*(.+)$/i.exec(s) ?? /^(.+?)\s*[x×*]\s*(\d+)$/i.exec(s);
  if (m) {
    if (/^\d+$/.test(m[1])) { count = Number(m[1]); name = m[2]; } else { name = m[1]; count = Number(m[2]); }
  }
  if (!Number.isInteger(count) || count < 1) return { error: `card count in "${s}" is not a positive integer` };
  const norm = name.toLowerCase().replace(/[^a-z0-9]/g, '');
  if (/^(nvidia)?a100(sxm|pcie)?$/.test(norm)) return { error: `"${name}" is ambiguous: say A100-40 or A100-80` };
  const hit = CARD_ALIASES.find(([re]) => re.test(norm));
  if (!hit) return { error: `unknown card "${name}" (known: ${CARDS.map(c => c.id).join(', ')})` };
  return { card: CARDS.find(c => c.id === hit[1])!, count };
}

/** "a T4", "an A100-40", "an H100-80", "an L4". */
function article(id: string): string {
  return /^[AEFHILMNORSX]/i.test(id) ? 'an' : 'a';
}

const r2 = (n: number) => Math.round(n * 100) / 100;
const gbText = (n: number) => (n >= 10 ? n.toFixed(0) : n >= 1 ? n.toFixed(1) : n.toFixed(2));

/** Overwrite the sequence axis of a token model's input. Vision and tabular inputs are left alone. */
export function withSeqLen(model: ModelArchitecture, seqLen: number | null): { model: ModelArchitecture; seqLen: number | null; applies: boolean } {
  const inp = model.components.find(c => c.type === 'input');
  const shape = (inp?.params as { shape?: unknown } | undefined)?.shape;
  const hasEmbedding = model.components.some(c => /embed/i.test(c.type));
  if (!inp || !Array.isArray(shape) || shape.length > 2 || !hasEmbedding) return { model, seqLen: null, applies: false };
  const current = Number(shape[shape.length - 1]);
  const L = seqLen ?? Math.min(current, DEFAULT_SEQ_LEN);
  if (L === current) return { model, seqLen: L, applies: true };
  const next = [...shape.slice(0, -1), L] as number[];
  const components = model.components.map((c): MLComponent => (c === inp
    ? { ...c, params: { ...c.params, shape: next }, inputShape: next } as MLComponent
    : c));
  return { model: { ...model, components }, seqLen: L, applies: true };
}

/**
 * A graph whose activation sizes cannot be believed, named, or null. One case:
 * the input feeds a token embedding a tensor of rank 3 or more (a hidden-state
 * or image shape read as token ids), which multiplies every activation by the
 * hidden width or the pixel count and would turn a 16 GB job into a 2 TB one.
 * A number about a graph like that is not a weak number, it is a wrong one.
 */
export function untrustworthy(model: ModelArchitecture): string | null {
  const inp = model.components.find(c => c.type === 'input');
  const shape = (inp?.params as { shape?: unknown } | undefined)?.shape;
  if (!inp || !Array.isArray(shape) || shape.length < 3) return null;
  const fed = new Set((model.connections ?? []).filter(e => e.from === inp.id).map(e => e.to));
  const emb = model.components.find(c => fed.has(c.id) && c.type === 'embedding');
  if (!emb) return null;
  return `the input is a ${shape.length}-D tensor ${JSON.stringify(shape)} fed straight into a token embedding (${emb.name}), `
    + 'so every activation size would be multiplied by a dimension that is not a token. Trace the model '
    + '(trace_model, or NEURARCH_TRACE=1 python train.py) or load it as hf:<org/name> for a graph whose input is token ids.';
}

/**
 * How many of the same 131 held-out did-not-fit card checks `accelerate
 * estimate-memory` flagged. The one number in the accuracy sentence that is
 * not in the bundle's MEMORY_CALIBRATION: it is the comparison the app quotes
 * beside it (neurarch-ai/neurarch README and /memory-calibration.html at
 * a1bac74e). Check it there when MEMORY_CALIBRATION moves.
 */
export const ACCELERATE_FLAGGED_AT_MOST = 8;

/**
 * The sentence every result carries. The factors and the calibration are the
 * bundle's own (`calibrationSentence()` is written from MEMORY_CALIBRATION),
 * so a refit that ships in a new bundle changes this sentence with it; the
 * held-out median is never quoted without the 14% after-fit miss beside it.
 */
export const ACCURACY_SENTENCE =
  'Calibrated estimate of the training peak under AdamW with bf16 autocast, not a measurement: '
  + `${OPTIMIZER_K} x 16 bytes per parameter (weights, gradients, optimizer state), plus, per sample in bf16, `
  + `${OUTPUT_K} x every layer output, ${ATTENTION_K} x stored attention scores (eager attention only; SDPA and flash keep none), `
  + `${INTERNAL_K} x each layer's weight-matrix intermediates, ${RECURRENT_K} x recurrent gate states and ${TRANSIENT_K} x the largest single tensor. `
  + `${calibrationSentence()} On the same held-out checks accelerate estimate-memory flagged at most ${ACCELERATE_FLAGGED_AT_MOST}. `
  + 'At the edge of a card treat it as a fit check, not a precise number. '
  + `A card "fits" when it holds the peak x ${HEADROOM} (allocator fragmentation and the CUDA context).`;

export interface SizeJobInput {
  batchSize: number;
  gpu?: string;
  seqLen?: number;
  attention?: Attention;
  epochs?: number;
  samples?: number;
}

export interface SizeJobResult {
  sized: boolean;
  verdict: string;
  fits: boolean | null;
  model: { name: string; params: number; layers: number; fingerprint: string };
  job: { batchSize: number; seqLen: number | null; attention: Attention; gpu: string | null; epochs: number; samples: number | null };
  memory: {
    peakGb: number;
    /** peakGb x 1.15: what a card (or N cards) must hold. */
    needGb: number;
    weightsAndOptimizerGb: number;
    /** Activations counted in peakGb, including attention scores when eager. */
    activationGb: number | null;
    /** Stored attention scores counted in peakGb (0 unless eager). */
    attentionScoresGb: number;
    /** Stored attention scores this job does NOT keep because it runs SDPA/flash (0 when eager). */
    attentionScoresDroppedGb: number;
  } | null;
  smallestFitGpu: string | null;
  maxBatch: Record<string, number | null> | null;
  multiCard: {
    cards: number;
    card: string;
    /** Each card holds a full copy at the per-device batch (plain DDP). */
    ddp: { perDeviceBatch: number; peakGbPerCard: number; fits: boolean };
    /** One model sharded across the cards (FSDP / ZeRO-3), the peak at the global batch against summed capacity. */
    sharded: { peakGb: number; capacityGb: number; fits: boolean; maxGlobalBatch: number | null };
  } | null;
  twoCardSplit: (NonNullable<ModelPlan['cost']['twoCardSplit']> & { conservative: boolean; note?: string }) | null;
  cost: { gpu: string; cards: number; usd: number | null; trainSec: number | null; assumptions: string } | null;
  forwardPasses: { legal: boolean | null; blockers: string[] };
  lastRun: { fingerprint: string; how: string };
  caveats: string[];
  accuracy: string;
}

interface Sizer {
  /** Peak GB at batch b for this job's attention mode. */
  peakAt: (b: number) => number;
  footprintGb: number;
  perSampleGb: number | null;
}

function sizerFor(plan: ModelPlan, model: ModelArchitecture, batch: number, attention: Attention): { sizer: Sizer; attnCounted: number; attnDropped: number } {
  const act = plan.cost.activationGb;
  const footprintGb = plan.cost.peakGb - (act ?? 0);
  if (act == null) return { sizer: { peakAt: () => footprintGb, footprintGb, perSampleGb: null }, attnCounted: 0, attnDropped: 0 };
  const attnAtBatch = attentionScoresGbFor(model, batch);
  const keptAct = attention === 'eager' ? act : Math.max(0, act - attnAtBatch);
  const perSampleGb = keptAct / batch;
  return {
    sizer: { peakAt: (b: number) => footprintGb + perSampleGb * b, footprintGb, perSampleGb },
    attnCounted: attention === 'eager' ? attnAtBatch : 0,
    attnDropped: attention === 'eager' ? 0 : attnAtBatch,
  };
}

function maxBatchOn(s: Sizer, gb: number): number | null {
  if (s.perSampleGb == null) return null;
  const per = s.perSampleGb;
  return maxBatchFor({ footprintGb: s.footprintGb, activationPerSampleGb: per, activationGbAt: (b: number) => per * b }, gb);
}

/** The plan's two-card rule, on this job's peak: the smallest card size one of cannot hold and two can. */
function twoCardSplit(model: ModelArchitecture, peakGb: number, batch: number, attention: Attention, attnDropped: number, only?: Card): SizeJobResult['twoCardSplit'] {
  const sizes = only ? [only.gb] : [...new Set(CARDS.map(c => c.gb))].sort((a, b) => a - b);
  for (const gb of sizes) {
    if (peakGb * HEADROOM <= gb) return null;
    if (peakGb * HEADROOM > 2 * gb) continue;
    // The bundle's per-layer cut counts eager attention scores. Under SDPA or
    // flash that over-states every card's share, so a cut it says fits
    // certainly fits; one it says does not is not reported, rather than
    // reported wrong.
    const map = deviceMapFor(model, [gb, gb], { batchSize: batch, attention: 'eager' });
    const conservative = attention !== 'eager' && attnDropped > 0.05;
    if (map.fits) {
      const gpu = only?.id ?? CARDS.find(c => c.gb === gb)!.id;
      return {
        gpu,
        gbPerCard: gb,
        splits: map.splits.map(sp => ({ ...sp, gb: r2(sp.gb) })),
        fits: map.fits,
        conservative,
        ...(conservative
          ? { note: 'Each card\'s share counts stored attention scores this job does not keep, so the shares are over-stated; the cut is safe.' }
          : {}),
      };
    }
  }
  return null;
}

/**
 * Size one job. Never throws on a graph: a graph it cannot size comes back with
 * `sized: false` and the reason as the verdict.
 */
export function sizeJob(input: ModelArchitecture, opts: SizeJobInput): SizeJobResult {
  const batch = Math.max(1, Math.floor(opts.batchSize));
  const attention: Attention = opts.attention ?? 'sdpa';
  const epochs = Math.max(1, Math.floor(opts.epochs ?? 10));
  const caveats: string[] = [];

  let requested: { card: Card; count: number } | null = null;
  if (opts.gpu != null && String(opts.gpu).trim() !== '') {
    const g = parseGpu(opts.gpu);
    if ('error' in g) throw new Error(`size_training_job: ${g.error}.`);
    requested = g;
  }
  const label = requested ? (requested.count === 1 ? requested.card.id : `${requested.count}x${requested.card.id}`) : null;

  const seq = withSeqLen(input, opts.seqLen ?? null);
  if (opts.seqLen != null && !seq.applies) caveats.push('seq_len was ignored: this graph\'s input is not a token sequence.');
  const model = seq.model;

  // Pinned to eager: sizerFor removes the score term itself, from the plan's
  // own attentionScoresGbFor. The bundle has defaulted to SDPA since neurarch
  // #285, and an SDPA plan here would have the scores removed twice.
  const plan = buildPlan(model, { attention: 'eager', run: { batchSize: batch, epochs }, ...(opts.samples ? { workload: { samples: opts.samples, source: 'size_training_job samples argument' } } : {}) });
  const head = {
    model: { name: plan.model.name, params: plan.model.params, layers: plan.model.layers, fingerprint: plan.model.fingerprint },
    job: { batchSize: batch, seqLen: seq.seqLen, attention, gpu: label, epochs, samples: opts.samples ?? null },
    forwardPasses: { legal: plan.run.legal, blockers: plan.run.blockers.map(b => b.title) },
    lastRun: {
      fingerprint: plan.model.fingerprint,
      how: plan.model.fingerprint
        ? `Call history with fingerprint "${plan.model.fingerprint}" for what this structure measured the last time it trained in your organisation (needs NEURARCH_API_KEY; sends the fingerprint, never the graph).`
        : 'No fingerprint: the graph could not be read.',
    },
    accuracy: ACCURACY_SENTENCE,
  };
  const unsized = (verdict: string): SizeJobResult => ({
    sized: false, verdict, fits: null, ...head, memory: null, smallestFitGpu: null, maxBatch: null,
    multiCard: null, twoCardSplit: null, cost: null, caveats,
  });

  const distrust = untrustworthy(model);
  if (distrust) return unsized(`cannot size this graph: ${distrust}`);
  if (plan.model.params <= 0) return unsized(`cannot size this graph: ${plan.run.blockers[0]?.title ?? 'it has no parameters to size'}.`);

  const { sizer, attnCounted, attnDropped } = sizerFor(plan, model, batch, attention);
  const peakGb = sizer.peakAt(batch);
  const needGb = peakGb * HEADROOM;
  const actCounted = plan.cost.activationGb == null ? null : peakGb - sizer.footprintGb;
  if (plan.cost.activationGb == null) {
    caveats.push('No shape propagated through this graph, so activations are not counted: the peak is weights and optimizer state only, a floor. Trace the model for real shapes.');
  }
  if (attention !== 'eager' && attnDropped > 0.05) {
    caveats.push(`Sized for ${attention} attention. Under eager attention the same job keeps another ${gbText(attnDropped)} GB of attention scores.`);
  }

  const fitsOn = (gb: number) => needGb <= gb;
  const smallest = CARDS.find(c => fitsOn(c.gb)) ?? null;
  const maxBatch: Record<string, number | null> = {};
  for (const c of CARDS) maxBatch[c.id] = maxBatchOn(sizer, c.gb);

  // Multi-card: both readings, because the job's parallelism decides which applies.
  let multiCard: SizeJobResult['multiCard'] = null;
  if (requested && requested.count > 1) {
    const n = requested.count;
    const perDevice = Math.max(1, Math.ceil(batch / n));
    const ddpPeak = sizer.peakAt(perDevice);
    const cap = requested.card.gb * n;
    multiCard = {
      cards: n,
      card: requested.card.id,
      ddp: { perDeviceBatch: perDevice, peakGbPerCard: r2(ddpPeak), fits: ddpPeak * HEADROOM <= requested.card.gb },
      sharded: { peakGb: r2(peakGb), capacityGb: cap, fits: needGb <= cap, maxGlobalBatch: maxBatchOn(sizer, cap) },
    };
  }

  // The verdict line, for the card asked about or the cheapest that holds it.
  let fits: boolean;
  let verdict: string;
  const headroomPct = (gb: number, peak: number) => Math.round((1 - peak / gb) * 100);
  const atBatch = `at batch ${batch}${seq.seqLen ? `, seq ${seq.seqLen}` : ''}`;
  if (requested && requested.count === 1) {
    const c = requested.card;
    fits = fitsOn(c.gb);
    if (fits) {
      verdict = `fits ${article(c.id)} ${c.id} with ${headroomPct(c.gb, peakGb)}% headroom ${atBatch} (peak ~${gbText(peakGb)} GB of ${c.gb} GB)`;
    } else {
      const alt = smallest && smallest.gb > c.gb ? smallest.id : CARDS.filter(x => x.gb > c.gb).find(x => fitsOn(x.gb))?.id;
      const mb = maxBatch[c.id];
      const options = [
        alt ? `use ${alt}` : null,
        mb != null ? `batch <= ${mb}` : null,
      ].filter(Boolean);
      verdict = `will not fit ${article(c.id)} ${c.id}: needs ~${gbText(needGb)} GB with headroom (peak ${gbText(peakGb)} GB) ${atBatch}`
        + (options.length ? `, ${options.join(' or ')}` : '')
        + (mb == null && sizer.footprintGb * HEADROOM > c.gb ? `; no batch fits, weights and optimizer state alone are ${gbText(sizer.footprintGb)} GB` : '');
    }
  } else if (requested && multiCard) {
    const { card, ddp, sharded } = multiCard;
    fits = sharded.fits;
    if (ddp.fits) {
      verdict = `fits ${label} even under plain DDP: a full copy per card at per-device batch ${ddp.perDeviceBatch} peaks at ~${gbText(ddp.peakGbPerCard)} GB of ${requested.card.gb} GB (${headroomPct(requested.card.gb, ddp.peakGbPerCard)}% headroom)`;
    } else if (sharded.fits) {
      verdict = `fits ${label} only sharded (FSDP / ZeRO-3): ~${gbText(needGb)} GB with headroom across ${sharded.capacityGb} GB; under plain DDP each ${card} would need ~${gbText(ddp.peakGbPerCard * HEADROOM)} GB`;
    } else {
      verdict = `will not fit ${label} even sharded: needs ~${gbText(needGb)} GB with headroom against ${sharded.capacityGb} GB`
        + (sharded.maxGlobalBatch != null ? `; global batch <= ${sharded.maxGlobalBatch} fits sharded` : '');
    }
  } else {
    fits = !!smallest;
    if (smallest) {
      verdict = `fits ${article(smallest.id)} ${smallest.id} (the cheapest listed card that holds it) with ${headroomPct(smallest.gb, peakGb)}% headroom ${atBatch} (peak ~${gbText(peakGb)} GB)`;
    } else {
      const big = CARDS[CARDS.length - 1];
      const mb = maxBatch[big.id];
      verdict = `will not fit any single listed card (largest ${big.gb} GB): needs ~${gbText(needGb)} GB with headroom ${atBatch}`
        + (mb != null ? `; batch <= ${mb} fits ${article(big.id)} ${big.id}` : '; no batch fits one card');
    }
  }

  const split = requested && requested.count === 1
    ? (fits ? null : twoCardSplit(model, peakGb, batch, attention, attnDropped, requested.card))
    : requested ? null : twoCardSplit(model, peakGb, batch, attention, attnDropped);
  if (split && !requested && !smallest) verdict += `; or split across two ${split.gpu}`;
  if (!split && requested && requested.count === 1 && !fits && needGb <= 2 * requested.card.gb && attention !== 'eager' && attnDropped > 0.05) {
    caveats.push(`Two ${requested.card.id} cards hold the total, but no per-layer cut was confirmed: the cut counts the attention scores ${attention} does not keep, and with them it does not fit.`);
  }

  // One run's cost: on the card asked for, else the cheapest that holds it, else the split's card.
  const priceCard = requested?.card ?? smallest ?? (split ? CARDS.find(c => c.id === split.gpu) ?? null : null);
  const priceCount = requested?.count ?? (smallest ? 1 : split ? 2 : 1);
  let cost: SizeJobResult['cost'] = null;
  if (priceCard) {
    const priced = buildPlan(model, {
      attention: 'eager',
      run: { gpu: priceCard.planName, batchSize: batch, epochs },
      ...(opts.samples ? { workload: { samples: opts.samples, source: 'size_training_job samples argument' } } : {}),
    });
    const pricing = priced.cost.assumptions.split('. GPU fit')[0];
    cost = {
      gpu: priceCount === 1 ? priceCard.id : `${priceCount}x${priceCard.id}`,
      cards: priceCount,
      usd: priced.cost.estCostUsd == null ? null : Math.round(priced.cost.estCostUsd * 100) / 100,
      trainSec: priced.cost.estTrainSec == null ? null : Math.round(priced.cost.estTrainSec / priceCount),
      assumptions: `${pricing.replace('on the priced GPU', `on ${priceCard.id}`)}.`
        + (priceCount > 1 ? ` Across ${priceCount} cards the card-hours are assumed to scale ideally, so the price is one card's and the wall time is divided by ${priceCount}; real scaling is worse.` : '')
        + (opts.samples ? '' : ' Pass samples and epochs for a price on your real workload.'),
    };
  }

  return {
    sized: true,
    verdict,
    fits,
    ...head,
    memory: {
      peakGb: r2(peakGb),
      needGb: r2(needGb),
      weightsAndOptimizerGb: r2(sizer.footprintGb),
      activationGb: actCounted == null ? null : r2(actCounted),
      attentionScoresGb: r2(attnCounted),
      attentionScoresDroppedGb: r2(attnDropped),
    },
    smallestFitGpu: smallest?.id ?? null,
    maxBatch,
    multiCard,
    twoCardSplit: split,
    cost,
    caveats,
  };
}
