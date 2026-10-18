/**
 * The model side of a serving check: the graph plus the four files a serving
 * engine reads (config.json, tokenizer_config.json, chat_template.jinja,
 * generation_config.json), from wherever the launch names the model.
 *
 * Shared by check_serving_config and the PreToolUse hook, so the tool and the
 * hook read a model the same way. Three places a model can be:
 *
 *   a local checkpoint directory  the four files on disk; no socket.
 *   a Hugging Face repo id        config.json through loadHFModel and the other
 *                                 three through loadHFServeFiles: the cache at
 *                                 any age, a fetch only under --hf.
 *   a graph file / zoo:<id>       the graph alone, no config.
 */
import { readFile, stat } from 'node:fs/promises';
import { isAbsolute, join, resolve } from 'node:path';
import type { ModelArchitecture } from './types.js';
import type { ServeModelSide } from '../vendor/plan.bundle.mjs';
import { looksLikeHFConfig } from '../vendor/plan.bundle.mjs';
import { convertHFConfigToModel } from '../vendor/engine.bundle.mjs';
import { loadHFModel, loadHFServeFiles } from '../sources.js';
import { loadModelCached } from '../models.js';

export interface ResolvedServeSide {
  side: ServeModelSide;
  /** Where the model came from, as a sentence fragment. */
  source: string;
  /** False when the tokenizer / template / generation files could not be read at all. */
  filesRead: boolean;
  /** True when the config is the model's own config.json, not a family template or nothing. */
  exact: boolean;
}

const HF_ID = /^[\w.-]+\/[\w.-]+$/;

async function readJson(path: string): Promise<Record<string, unknown> | null> {
  try {
    const v = JSON.parse(await readFile(path, 'utf-8')) as unknown;
    return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

/** A checkpoint directory holding a Hugging Face config.json, read from disk. */
export async function localServeSide(dir: string): Promise<ResolvedServeSide> {
  const config = await readJson(join(dir, 'config.json'));
  if (!config || !looksLikeHFConfig(config)) throw new Error(`${dir}/config.json is not a Hugging Face config`);
  const name = dir.replace(/\/+$/, '').split('/').pop() || 'model';
  const { components, connections } = convertHFConfigToModel(name, config as never);
  const model = { id: `dir-${name}`, name, components, connections } as ModelArchitecture;
  const tmpl = await readFile(join(dir, 'chat_template.jinja'), 'utf-8').catch(() => null);
  return {
    side: {
      model, config, modelId: name,
      tokenizerConfig: await readJson(join(dir, 'tokenizer_config.json')),
      chatTemplateFile: tmpl && tmpl.trim() ? tmpl : null,
      generationConfig: await readJson(join(dir, 'generation_config.json')),
    },
    source: `${dir} (config.json on disk)`,
    filesRead: true,
    exact: true,
  };
}

/** A Hugging Face repo: config.json through the --hf path, the other three beside it. */
export async function hfServeSide(id: string): Promise<ResolvedServeSide> {
  const loaded = await loadHFModel(id);
  const files = await loadHFServeFiles(id);
  return {
    side: { model: loaded.model, config: loaded.config ?? null, modelId: id, ...(files ?? {}) },
    source: `hf:${id} (${loaded.configSource === 'fallback' ? 'a family template, not the real config' : 'config.json'})`,
    filesRead: !!files,
    exact: loaded.configSource !== 'fallback' && !!loaded.config,
  };
}

/**
 * Resolve a model reference as a launch line names it. A path that exists on
 * disk wins over the org/name reading of the same string, because
 * `vllm serve models/qwen` with a local `models/qwen/` serves the directory.
 */
export async function resolveServeSide(ref: string, cwd = process.cwd()): Promise<ResolvedServeSide> {
  const s = ref.trim();
  if (/^hf:/i.test(s)) return hfServeSide(s.slice(3).trim());
  if (/^zoo:/i.test(s)) return { side: { model: await loadModelCached(s), config: null, modelId: s }, source: s, filesRead: false, exact: false };
  const p = isAbsolute(s) ? s : resolve(cwd, s);
  const st = await stat(p).catch(() => null);
  if (st?.isDirectory()) return localServeSide(p);
  if (st?.isFile()) return { side: { model: await loadModelCached(p), config: null, modelId: s }, source: s, filesRead: false, exact: false };
  if (HF_ID.test(s)) return hfServeSide(s);
  throw new Error(`no model at ${s}: pass a Hugging Face id (org/name), a checkpoint directory, or a graph file`);
}
