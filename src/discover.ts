/**
 * find_models: what in this repository is a model at all.
 *
 * The server is started on one file, and `model_path` lets a tool ask about
 * another, but both assume the agent already knows the path. In a repository
 * it has just been dropped into, it does not. This walks the tree, keeps the
 * files that look like they define an nn.Module or are a saved graph, and
 * tries the parser on each so the answer is "these parse, these do not, and
 * this is why", not a list of filenames.
 */
import { readdir, readFile, stat } from 'node:fs/promises';
import { join, relative, resolve } from 'node:path';
import { graphFromPyTorchSource } from './vendor/engine.bundle.mjs';
import { parseQuality } from './lib/parseQuality.js';

const SKIP_DIRS = new Set([
  'node_modules', '.git', '.hg', '.svn', '__pycache__', '.venv', 'venv', 'env',
  'site-packages', 'dist', 'build', '.mypy_cache', '.pytest_cache', '.tox', 'checkpoints', 'wandb', 'runs',
]);

export interface DiscoveredModel {
  path: string;
  kind: 'pytorch-source' | 'neurarch-json';
  /** 'partial': a graph came back but it is thin or carries unevaluated params; trust it less than 'parsed'. */
  status: 'parsed' | 'partial' | 'no-model' | 'error';
  /** The nn.Module subclasses declared in the file, parsed or not. */
  classes: string[];
  layers?: number;
  connections?: number;
  detail?: string;
}

export interface DiscoverOptions {
  dir: string;
  /** Hard cap on files opened, so a monorepo does not turn one call into a minute. */
  maxFiles?: number;
  maxDepth?: number;
}

const MODULE_CLASS = /class\s+(\w+)\s*\([^)]*\b(?:nn\.Module|Module|LightningModule|PreTrainedModel)\b[^)]*\)/g;

export async function discoverModels(opts: DiscoverOptions): Promise<{ dir: string; scanned: number; models: DiscoveredModel[]; truncated: boolean }> {
  const root = resolve(opts.dir);
  const maxFiles = opts.maxFiles ?? 400;
  const maxDepth = opts.maxDepth ?? 6;
  const info = await stat(root).catch(() => null);
  if (!info || !info.isDirectory()) throw new Error(`Not a directory: ${root}`);

  const files: string[] = [];
  let truncated = false;
  async function walk(dir: string, depth: number): Promise<void> {
    if (depth > maxDepth || truncated) return;
    const entries = await readdir(dir, { withFileTypes: true }).catch(() => []);
    for (const e of entries) {
      if (truncated) return;
      if (e.isDirectory()) {
        if (!SKIP_DIRS.has(e.name) && !e.name.startsWith('.')) await walk(join(dir, e.name), depth + 1);
      } else if (e.isFile() && (e.name.endsWith('.py') || e.name.endsWith('.neurarch.json'))) {
        if (files.length >= maxFiles) { truncated = true; return; }
        files.push(join(dir, e.name));
      }
    }
  }
  await walk(root, 0);

  const models: DiscoveredModel[] = [];
  for (const file of files) {
    const rel = relative(root, file);
    if (file.endsWith('.neurarch.json')) {
      try {
        const g = JSON.parse(await readFile(file, 'utf-8'));
        if (Array.isArray(g.components) && Array.isArray(g.connections)) {
          models.push({ path: rel, kind: 'neurarch-json', status: 'parsed', classes: [], layers: g.components.length, connections: g.connections.length });
        }
      } catch (e) {
        models.push({ path: rel, kind: 'neurarch-json', status: 'error', classes: [], detail: (e as Error).message });
      }
      continue;
    }
    const code = await readFile(file, 'utf-8').catch(() => '');
    const classes = [...code.matchAll(MODULE_CLASS)].map(m => m[1]);
    if (classes.length === 0) continue;
    try {
      const g = graphFromPyTorchSource(code, rel.replace(/\.py$/, ''));
      if (g && g.components.length > 2) {
        const q = parseQuality(g);
        models.push({
          path: rel, kind: 'pytorch-source', status: q.grade === 'full' ? 'parsed' : 'partial', classes,
          layers: g.components.length, connections: g.connections.length, ...(q.note ? { detail: q.note } : {}),
        });
      } else {
        models.push({
          path: rel, kind: 'pytorch-source', status: 'no-model', classes,
          detail: 'Declares an nn.Module but the static parser found no layer assignments it could read. '
            + 'Trace it at runtime instead: pip install neurarch-trace, then neurarch-trace <module>:<class> --input ...',
        });
      }
    } catch (e) {
      models.push({ path: rel, kind: 'pytorch-source', status: 'error', classes, detail: (e as Error).message });
    }
  }
  models.sort((a, b) => (a.status === 'parsed' ? 0 : 1) - (b.status === 'parsed' ? 0 : 1) || (b.layers ?? 0) - (a.layers ?? 0));
  return { dir: root, scanned: files.length, models, truncated };
}
