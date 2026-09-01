/**
 * Loading models other than the one this server was launched with.
 *
 * The server takes a model path on the command line, and for a long time that
 * was the whole story: one process, one model. That is wrong for the shape of
 * the work. An agent is usually sitting in a repository that holds several
 * architectures (a baseline and two variants, or one model per experiment), and
 * making the user register five MCP servers to ask five questions is a tax with
 * no upside. Every read tool therefore takes an optional `model_path`.
 *
 * Two properties this cache has to keep:
 *
 * 1. **Freshness beats speed.** The file on disk is the truth, and the agent may
 *    be editing it between calls. So every hit re-stats and compares mtime; only
 *    an unchanged file is served from memory. Parsing is milliseconds, and a
 *    stale graph would make the agent confidently wrong, which is the failure
 *    mode this whole server exists to prevent.
 * 2. **Bounded memory.** A long-lived server pointed at a big repo could
 *    otherwise accumulate every model it was ever asked about. The map is
 *    insertion-ordered, so evicting the oldest key is one line.
 */
import { resolve } from 'node:path';
import { stat } from 'node:fs/promises';
import { loadModelFile } from './loader.js';
import type { ModelArchitecture } from './lib/types.js';
import { parseModelRef, loadZooModel, loadHFModel } from './sources.js';

/** Small on purpose: this is a working set, not a database. */
const MAX_CACHED = 8;

const cache = new Map<string, { mtimeMs: number; model: ModelArchitecture }>();

/**
 * Load a model by path, reusing the parse when the file has not changed.
 *
 * Errors are the loader's own, unwrapped: "cannot read", "not valid JSON",
 * "missing components/connections" are all more useful to an agent than a
 * generic cache-layer message would be.
 */
export async function loadModelCached(path: string): Promise<ModelArchitecture> {
  // `zoo:` and `hf:` are not files. The zoo is read from the package (fresh by
  // construction, it only changes with a release); the HF loader keeps its own
  // on-disk cache with a TTL, because a config.json can change upstream.
  const ref = parseModelRef(path);
  if (ref.kind === 'zoo') return loadZooModel(ref.id);
  if (ref.kind === 'hf') return (await loadHFModel(ref.id)).model;

  const abs = resolve(ref.path);

  // A failed stat is not cached, and not swallowed either: loadModelFile
  // produces the good message for a missing or unreadable path, so let it.
  const mtimeMs = await stat(abs).then(s => s.mtimeMs).catch(() => null);
  const hit = mtimeMs === null ? undefined : cache.get(abs);
  if (hit && hit.mtimeMs === mtimeMs) return hit.model;

  const model = await loadModelFile(abs);
  if (mtimeMs !== null) {
    cache.set(abs, { mtimeMs, model });
    if (cache.size > MAX_CACHED) {
      const oldest = cache.keys().next().value;
      if (oldest !== undefined) cache.delete(oldest);
    }
  }
  return model;
}

/** Test seam. Nothing in the running server needs to drop the cache. */
export function clearModelCache(): void {
  cache.clear();
}

/** Current cache size. Exists so a test can assert eviction actually happens. */
export function modelCacheSize(): number {
  return cache.size;
}
