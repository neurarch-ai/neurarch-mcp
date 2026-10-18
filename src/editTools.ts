/**
 * check_edit: an edit to a model's config or architecture, read before the
 * training run that would be the first place it shows.
 *
 * The edits it catches do not fail. A `resize_token_embeddings` that misses a
 * tensor, a head count that drifts away from its kv heads, a context length
 * raised with no RoPE scaling, a LoRA `target_modules` entry that matches
 * nothing: each loads, runs and trains, and nothing raises for an agent to
 * iterate on. So the tool is called on the pair (what the model was, what it
 * is now) and answers two things: the edit findings, and what the edit did to
 * the training peak at the job's batch and card.
 *
 * No logic of its own. The findings are the app's `checkGraphEdit`,
 * `checkConfigEdit` and `checkLoraTargets` (src/utils/editChecks.ts in
 * neurarch-ai/neurarch, vendored in plan.bundle.mjs, the same code behind the
 * plan's `edit` block on the CI bot's comment); the memory is `sizeJob`, which
 * is `buildPlan`; a config with no graph beside it is read into one by the
 * engine's `convertHFConfigToModel`. Offline: a model named as hf:<org/name>
 * follows the same --hf rule as every other tool.
 */
import { readFile } from 'node:fs/promises';
import type { ModelArchitecture } from './lib/types.js';
import type { ToolDef } from './tools.js';
import { sizeJob, ACCURACY_SENTENCE, type Attention, type SizeJobResult } from './lib/sizeJob.js';
import { loadHFModel } from './sources.js';
import { loadModelCached } from './models.js';
import { convertHFConfigToModel } from './vendor/engine.bundle.mjs';
import {
  checkGraphEdit,
  checkConfigEdit,
  checkLoraTargets,
  loraTargetsFor,
  looksLikeHFConfig,
  EDIT_RULE_IDS,
  EDIT_RULE_SEEN_IN,
  DEFAULT_OOM_TRAIN_BATCH,
  type EditFinding,
  type EditRuleId,
} from './vendor/plan.bundle.mjs';

export interface CheckEditArgs {
  base_model_path?: string;
  base_config?: unknown;
  config?: unknown;
  lora_target_modules?: string[] | string;
  batch?: number;
  seq_len?: number;
  gpu?: string;
  attention?: Attention;
}

type Side = { model: ModelArchitecture; source: string; fromConfig: boolean };

const r2 = (n: number) => Math.round(n * 100) / 100;

/** A config argument: an object, JSON text, or a path to a config.json. */
async function readConfig(name: string, v: unknown): Promise<Record<string, unknown> | undefined> {
  if (v == null) return undefined;
  let obj: unknown = v;
  if (typeof v === 'string') {
    const t = v.trim();
    if (t.startsWith('{')) {
      try { obj = JSON.parse(t); } catch (e) { throw new Error(`check_edit: ${name} looks like JSON but does not parse: ${(e as Error).message}`); }
    } else {
      const text = await readFile(t, 'utf-8').catch(() => { throw new Error(`check_edit: ${name} is neither JSON nor a readable file: ${t}`); });
      try { obj = JSON.parse(text); } catch (e) { throw new Error(`check_edit: ${name} file ${t} does not parse as JSON: ${(e as Error).message}`); }
    }
  }
  if (!looksLikeHFConfig(obj)) {
    throw new Error(`check_edit: ${name} does not read as a Hugging Face config.json (no model_type or architectures).`);
  }
  return obj as Record<string, unknown>;
}

async function fromHf(id: string): Promise<Side> {
  const clean = id.replace(/^hf:/i, '').trim();
  const loaded = await loadHFModel(clean);
  const how = loaded.configSource === 'fallback' ? 'a family template, not the real config' : loaded.configSource === 'cache' ? 'cached config.json' : 'config.json';
  return { model: loaded.model, source: `hf:${clean} (${how})`, fromConfig: false };
}

function fromConfig(label: string, cfg: Record<string, unknown>): Side {
  const name = typeof cfg.model_type === 'string' ? cfg.model_type : label;
  const { components, connections } = convertHFConfigToModel(name, cfg as never);
  return {
    model: { id: `${label}-config`, name: `${name} (${label})`, components, connections } as ModelArchitecture,
    source: `${label} config.json`,
    fromConfig: true,
  };
}

interface MemorySide {
  source: string;
  sized: boolean;
  peakGb: number | null;
  needGb: number | null;
  fits: boolean | null;
  smallestFitGpu: string | null;
  maxBatchOnCard: number | null;
  verdict: string;
}

function memorySide(s: Side, r: SizeJobResult, card: string | null): MemorySide {
  return {
    source: s.source,
    sized: r.sized,
    peakGb: r.memory?.peakGb ?? null,
    needGb: r.memory?.needGb ?? null,
    fits: r.fits,
    smallestFitGpu: r.smallestFitGpu,
    maxBatchOnCard: card && r.maxBatch ? r.maxBatch[card] ?? null : null,
    verdict: r.verdict,
  };
}

/** The memory sentence: before and after, and whether the edit moved the job across a card's line. */
function memorySentence(before: MemorySide, after: MemorySide, card: string | null, batch: number): string {
  if (before.peakGb == null || after.peakGb == null) {
    return `memory not compared: ${!before.sized ? `the base ${before.verdict}` : `the changed model ${after.verdict}`}`;
  }
  const d = after.peakGb - before.peakGb;
  const delta = `${d >= 0 ? '+' : ''}${r2(d)} GB`;
  const head = `peak ${r2(before.peakGb)} -> ${r2(after.peakGb)} GB (${delta}) at batch ${batch}`;
  if (card) {
    if (before.fits && !after.fits) return `${head}: no longer fits ${card}${after.maxBatchOnCard != null ? `, batch <= ${after.maxBatchOnCard} does` : ''}${after.smallestFitGpu ? `, or use ${after.smallestFitGpu}` : ''}`;
    if (!before.fits && after.fits) return `${head}: now fits ${card}`;
    return `${head}: ${after.fits ? 'still fits' : 'still does not fit'} ${card}`;
  }
  if (before.smallestFitGpu !== after.smallestFitGpu) {
    return `${head}: the smallest card that holds it moved ${before.smallestFitGpu ?? 'none'} -> ${after.smallestFitGpu ?? 'none'}`;
  }
  return `${head}: the smallest card that holds it is still ${after.smallestFitGpu ?? 'none'}`;
}

const seenIn = (rule: string): string[] => (EDIT_RULE_IDS as readonly string[]).includes(rule) ? EDIT_RULE_SEEN_IN[rule as EditRuleId] : [];

export const checkEditTool: ToolDef = {
  name: 'check_edit',
  description:
    'Call after editing a model\'s config or architecture and before training. Give the pair: the base (base_model_path: '
    + 'a .py, a .neurarch.json, zoo:<id> or hf:<org/name>; or base_config) and the changed model (the server\'s model, '
    + 'model_path in the same forms, model_source, or config). base_config and config take a Hugging Face '
    + 'config.json as an object, as JSON text, or as a path. It returns the edits that load, run and are wrong, each with '
    + 'a stable rule id: vocab-resize-incomplete (resize_token_embeddings left a token-indexed tensor at the old vocabulary), '
    + 'weight-tying-lost, attention-config-drift (heads, head_dim, hidden_size and kv heads that must move together and did '
    + 'not), rope-context-drift (context length raised with no RoPE scaling), and, with lora_target_modules, '
    + 'lora-target-no-match, lora-target-partial and lora-target-wrapper, matched the way PEFT matches them. Config '
    + 'divisibility is reported under the existing lint ids head-dim-divisibility and gqa-head-divisibility. Then the memory '
    + 'delta: the training peak before and after at batch (default 32) and seq_len, and whether the job still fits gpu '
    + '(or the smallest card that holds it, before and after), from the same calibrated estimate size_training_job uses. '
    + 'block findings should be fixed before the run. It never says an edit is a good idea: a deliberate change is '
    + 'reported as one the config and the weights have to agree on. Offline; an hf: model needs --hf unless its config is cached.',
  modelOptional: true,
  inputSchema: {
    type: 'object',
    properties: {
      base_model_path: { type: 'string', description: 'The model before the edit.' },
      base_config: { description: 'config.json before: object, JSON or path.' },
      config: { description: 'config.json after.' },
      lora_target_modules: { type: ['array', 'string'], items: { type: 'string' } },
      batch: { type: 'integer', minimum: 1 },
      seq_len: { type: 'integer', minimum: 1 },
      gpu: { type: 'string' },
      attention: { type: 'string', enum: ['sdpa', 'flash', 'eager'] },
    },
    additionalProperties: false,
  },
  handler: async (args: CheckEditArgs, model: ModelArchitecture | undefined) => {
    for (const k of ['batch', 'seq_len'] as const) {
      const v = args[k];
      if (v != null && (!Number.isInteger(v) || v < 1)) throw new Error(`check_edit: ${k} must be a positive integer.`);
    }
    if (args.attention && !['sdpa', 'flash', 'eager'].includes(args.attention)) {
      throw new Error(`check_edit: attention "${args.attention}" is not one of sdpa, flash, eager.`);
    }

    const baseConfig = await readConfig('base_config', args.base_config);
    const config = await readConfig('config', args.config);
    const caveats: string[] = [];

    let base: Side | null = null;
    if (args.base_model_path) {
      base = /^hf:/i.test(args.base_model_path)
        ? await fromHf(args.base_model_path)
        : { model: await loadModelCached(args.base_model_path), source: args.base_model_path, fromConfig: false };
    } else if (baseConfig) base = fromConfig('base', baseConfig);

    let head: Side | null = null;
    if (model) head = { model, source: 'model', fromConfig: false };
    else if (config) head = fromConfig('changed', config);

    if (!base && !baseConfig) {
      throw new Error('check_edit: name the model before the edit: base_model_path (a path, zoo:<id> or hf:<org/name>) or base_config.');
    }
    if (!head && !config) {
      throw new Error('check_edit: name the changed model: model_path (a path, zoo:<id> or hf:<org/name>), model_source or config, or start the server on it.');
    }

    const findings: EditFinding[] = [];
    const checked: Array<'graph' | 'config' | 'lora'> = [];
    if (baseConfig && config) {
      checked.push('config');
      findings.push(...checkConfigEdit(baseConfig, config));
    } else if (baseConfig || config) {
      caveats.push(`Only ${baseConfig ? 'base_config' : 'config'} was given, so config keys were not compared; pass both to compare them.`);
    }
    // Two graphs read out of the same two configs would report the config's
    // vocabulary finding a second time under graph names, so the graph pair is
    // compared only when at least one side is a real graph.
    if (base && head && !(base.fromConfig && head.fromConfig)) {
      checked.push('graph');
      findings.push(...checkGraphEdit(base.model, head.model));
    }
    let lora: { matched: number; perTarget: Array<{ target: string; count: number }>; byConvention: boolean } | null = null;
    if (args.lora_target_modules != null) {
      if (!head) {
        caveats.push('lora_target_modules was not matched: there is no changed graph to match it against.');
      } else {
        checked.push('lora');
        const m = checkLoraTargets(head.model, args.lora_target_modules, loraTargetsFor(head.model));
        findings.push(...m.findings);
        lora = { matched: m.matched.length, perTarget: m.perTarget, byConvention: m.byConvention };
        if (m.byConvention) caveats.push('Some LoRA matches were decided on a projection this graph names by convention (inside a composite layer), not on a module name read from a checkpoint.');
      }
    }

    const batch = args.batch ?? DEFAULT_OOM_TRAIN_BATCH;
    let memory: {
      batch: number; seqLen: number | null; attention: Attention; gpu: string | null;
      before: MemorySide; after: MemorySide; deltaGb: number | null; summary: string;
    } | null = null;
    if (base && head) {
      const opts = { batchSize: batch, gpu: args.gpu, seqLen: args.seq_len, attention: args.attention };
      const rb = sizeJob(base.model, opts);
      const rh = sizeJob(head.model, opts);
      const card = rh.job.gpu;
      const before = memorySide(base, rb, card);
      const after = memorySide(head, rh, card);
      memory = {
        batch,
        seqLen: rh.job.seqLen,
        attention: rh.job.attention,
        gpu: card,
        before,
        after,
        deltaGb: before.peakGb != null && after.peakGb != null ? r2(after.peakGb - before.peakGb) : null,
        summary: memorySentence(before, after, card, batch),
      };
    } else {
      caveats.push('Memory was not compared: one side has no graph.');
    }

    const blocks = findings.filter(f => f.severity === 'block').length;
    const head1 = findings.length
      ? `${findings.length} edit finding${findings.length === 1 ? '' : 's'} (${blocks} block)`
      : `no edit findings in ${checked.join(', ') || 'nothing compared'}`;
    return {
      verdict: memory ? `${head1}; ${memory.summary}` : head1,
      // Not ok: a blocking finding, or the edit is what pushed the job off the card.
      ok: blocks === 0 && !(memory && memory.before.fits !== false && memory.after.fits === false),
      checked,
      findings: findings.map(f => ({ ...f, seenIn: seenIn(f.rule) })),
      ...(lora ? { lora } : {}),
      memory,
      base: base?.source ?? 'base_config',
      changed: head?.source ?? 'config',
      ...(memory ? { accuracy: ACCURACY_SENTENCE } : {}),
      caveats,
    };
  },
};

export const EDIT_TOOLS: ToolDef[] = [checkEditTool];
