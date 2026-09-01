/**
 * Where a model can come from, beyond a path on disk.
 *
 * Every read tool takes `model_path`. For most of this server's life that was
 * a file: a .neurarch.json the app wrote, later a .py the parser could read.
 * Two more forms are worth the same argument, and are spelled as prefixes so
 * an agent that already knows `model_path` needs to learn nothing new:
 *
 *   zoo:<id>       one of the reference architectures bundled with this
 *                  package (zoo/index.json lists them). Offline, always.
 *   hf:<org/name>  a Hugging Face repo, built from its config.json. This one
 *                  opens a socket, so it is refused unless the server was
 *                  started with --hf. The refusal names the flag.
 *
 * Inline text (`model_source`) is the third form, for the hosted case where
 * there is no shared filesystem at all; see server.ts.
 */
import { readFile, mkdir, writeFile, stat } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { homedir } from 'node:os';
import { join } from 'node:path';
import type { ModelArchitecture } from './lib/types.js';
import { graphFromPyTorchSource, fetchHFModelConfig, convertHFConfigToModel } from './vendor/engine.bundle.mjs';

export type ModelRef =
  | { kind: 'file'; path: string }
  | { kind: 'zoo'; id: string }
  | { kind: 'hf'; id: string };

export function parseModelRef(spec: string): ModelRef {
  const s = spec.trim();
  if (/^zoo:/i.test(s)) return { kind: 'zoo', id: s.slice(4).trim() };
  if (/^hf:/i.test(s)) return { kind: 'hf', id: s.slice(3).trim() };
  return { kind: 'file', path: s };
}

// ── zoo ──────────────────────────────────────────────────────────────────────

export interface ZooEntry {
  id: string;
  title: string;
  domain: string;
  type: string;
  params: string;
  layers: string;
  nodes: number;
  attention: string[];
  paramCheck: { estimate: number; real: number; deviationPct: number } | null;
  urls: Record<string, string>;
}

/**
 * zoo/ sits next to dist/ in the published package and next to src/ in a
 * checkout, so one relative URL from this module finds it in both. tsup
 * bundles src/*.ts into dist/index.js, one directory deep either way.
 */
const ZOO_DIR = new URL('../zoo/', import.meta.url);

let zooIndex: ZooEntry[] | null = null;

export async function listZoo(): Promise<ZooEntry[]> {
  if (zooIndex) return zooIndex;
  const raw = await readFile(new URL('index.json', ZOO_DIR), 'utf-8');
  zooIndex = (JSON.parse(raw) as { architectures: ZooEntry[] }).architectures;
  return zooIndex;
}

export async function loadZooModel(id: string): Promise<ModelArchitecture> {
  const entries = await listZoo();
  const hit = entries.find(e => e.id === id);
  if (!hit) {
    const near = entries.filter(e => e.id.includes(id.toLowerCase()) || e.title.toLowerCase().includes(id.toLowerCase()))
      .slice(0, 5).map(e => e.id);
    throw new Error(
      `No reference architecture named "${id}". `
      + (near.length ? `Closest: ${near.join(', ')}. ` : '')
      + 'Call list_architectures for the full set.',
    );
  }
  const gz = await readFile(new URL(`${id}.json.gz`, ZOO_DIR));
  const model = JSON.parse(gunzipSync(gz).toString('utf-8')) as ModelArchitecture;
  if (!model.name) model.name = hit.title;
  return model;
}

// ── hugging face ─────────────────────────────────────────────────────────────

let hfEnabled = false;
export function setHfEnabled(on: boolean): void { hfEnabled = on; }
export function isHfEnabled(): boolean { return hfEnabled; }

/** Where fetched configs are kept so the second question about a model is offline. */
export function hfCacheDir(): string {
  return process.env.NEURARCH_MCP_CACHE || join(homedir(), '.cache', 'neurarch-mcp', 'hf');
}

const HF_CACHE_TTL_MS = 24 * 60 * 60 * 1000;

export interface HFLoadResult {
  model: ModelArchitecture;
  /** 'config' is the real config.json; 'fallback' is a family template and says so. */
  configSource: 'config' | 'diffusers' | 'fallback' | 'cache';
  realParamCount?: number;
  cached: boolean;
}

export async function loadHFModel(modelId: string): Promise<HFLoadResult> {
  if (!hfEnabled) {
    throw new Error(
      `hf:${modelId} needs the server started with --hf. That flag is the switch for the one `
      + 'network call this server can make on your behalf (huggingface.co, to read config.json); '
      + 'without it, no tool opens a socket. Set HF_TOKEN for gated repos.',
    );
  }
  if (!/^[\w.-]+\/[\w.-]+$/.test(modelId)) {
    throw new Error(`"${modelId}" is not a Hugging Face repo id. Expected the form org/name, e.g. Qwen/Qwen2.5-7B.`);
  }
  const cacheFile = join(hfCacheDir(), `${modelId.replace('/', '__')}.json`);
  const cachedStat = await stat(cacheFile).catch(() => null);
  if (cachedStat && Date.now() - cachedStat.mtimeMs < HF_CACHE_TTL_MS) {
    const cached = JSON.parse(await readFile(cacheFile, 'utf-8')) as HFLoadResult;
    return { ...cached, configSource: 'cache', cached: true };
  }

  const config = await fetchHFModelConfig(modelId);
  if (!config) {
    throw new Error(
      `Could not build a graph for ${modelId}: config.json was not readable and the model id matches no known family. `
      + 'If the repo is gated, set HF_TOKEN and accept the licence on huggingface.co first.',
    );
  }
  const { components, connections } = convertHFConfigToModel(modelId, config);
  const model: ModelArchitecture = {
    id: `hf-${modelId}`,
    name: modelId.split('/').pop() || modelId,
    description: `Built from huggingface.co/${modelId} config.json (${config._configSource ?? 'config'}).`,
    components,
    connections,
  };
  const result: HFLoadResult = {
    model,
    configSource: config._configSource ?? 'config',
    realParamCount: typeof config._realParamCount === 'number' ? config._realParamCount : undefined,
    cached: false,
  };
  await mkdir(hfCacheDir(), { recursive: true }).catch(() => undefined);
  await writeFile(cacheFile, JSON.stringify(result)).catch(() => undefined);
  return result;
}

// ── inline text ──────────────────────────────────────────────────────────────

/**
 * Parse model text handed over the wire rather than read from disk. JSON is
 * tried first because a .neurarch.json is unambiguous; anything else is
 * treated as PyTorch source. The hosted server has no filesystem the client
 * can see, so this is the only way in there.
 */
export function modelFromText(text: string, name = 'model'): ModelArchitecture {
  const trimmed = text.trim();
  if (trimmed.startsWith('{')) {
    let parsed: unknown;
    try { parsed = JSON.parse(trimmed); } catch (e) {
      throw new Error(`model_source looks like JSON but does not parse: ${(e as Error).message}`);
    }
    const m = parsed as Partial<ModelArchitecture>;
    if (!Array.isArray(m.components) || !Array.isArray(m.connections)) {
      throw new Error('model_source JSON needs "components" and "connections" arrays (a .neurarch.json).');
    }
    return m as ModelArchitecture;
  }
  const model = graphFromPyTorchSource(trimmed, name);
  if (!model) {
    throw new Error(
      'No PyTorch model found in model_source. The parser looks for an nn.Module subclass whose layers '
      + 'are assigned in __init__. Pass a .neurarch.json instead if the model is built dynamically.',
    );
  }
  return model;
}
