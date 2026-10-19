/**
 * check_pins: Hugging Face downloads that are not pinned to a commit.
 *
 * A model id with no revision loads whatever `main` points at on the day the
 * job runs. Config, tokenizer, chat template and weights can all move under one
 * name, so two runs of one script can load two different models and nothing in
 * the repository records which. That is the "what changed" half of a pre-check
 * that no graph can answer, because the graph is downstream of the download.
 *
 * No logic here. The scanner is the vendored policy bundle's `scanPins` and
 * `fromPinCheck` (neurarch-ai/neurarch src/utils/policy/pinCheck.ts, #346), the
 * same functions behind `npx neurarch pin-check`, aligned with Bandit B615 and
 * flagging the same 15 lines Bandit 1.9.4 flags on Bandit's own examples. This
 * file walks the tree the way the CLI walks it and pages the result.
 *
 * Offline. The CLI's `--resolve` (ask the Hub for the current commit) is not
 * offered: it would make the one tool that reads the user's tree also the one
 * that tells huggingface.co which models the tree loads.
 */
import { readFile, readdir, stat } from 'node:fs/promises';
import { isAbsolute, join, relative, resolve, sep, dirname } from 'node:path';
import type { ToolDef, ToolContext } from './tools.js';
import { scanPins, pinFileKind, fromPinCheck, PIN_RULE_META, RENOVATE_PRESET_URL, type PinHit } from './vendor/policy.bundle.mjs';
import { pageOf, PAGE_PARAMS } from './paging.js';

/** The CLI's skip list, unchanged: a virtualenv is not the user's code. */
const SKIP_DIRS = new Set(['.git', '.hg', '.svn', 'node_modules', '.venv', 'venv', 'env', '__pycache__', '.tox', '.nox', '.mypy_cache', '.pytest_cache', '.ruff_cache', 'site-packages', '.ipynb_checkpoints']);
const MAX_BYTES = 5 * 1024 * 1024;
/** A walk that reads more than this many candidate files stops and says so. */
export const MAX_FILES = 5000;

const posix = (p: string) => p.split(sep).join('/');

async function* walk(abs: string): AsyncGenerator<string> {
  const st = await stat(abs).catch(() => null);
  if (!st) return;
  if (st.isFile()) {
    if (pinFileKind(posix(abs)) && st.size <= MAX_BYTES) yield abs;
    return;
  }
  if (!st.isDirectory()) return;
  const entries = await readdir(abs, { withFileTypes: true }).catch(() => []);
  for (const e of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    if (e.isSymbolicLink()) continue;
    if (e.isDirectory() && SKIP_DIRS.has(e.name)) continue;
    yield* walk(join(abs, e.name));
  }
}

/** Where a relative path resolves: next to the server's model file, else the working directory. */
function baseDir(ctx: ToolContext): string {
  const p = ctx.currentPath || ctx.modelPath;
  return p ? dirname(resolve(p)) : process.cwd();
}

interface PinArgs {
  paths?: string[];
  source?: string;
  filename?: string;
  include_dynamic?: boolean;
  limit?: number;
  cursor?: string;
}

export const checkPinsTool: ToolDef = {
  name: 'check_pins',
  description:
    'Find Hugging Face downloads that are not pinned to a commit, before a job or a deployment loads whatever `main` '
    + 'points at that day. Reads Python (from_pretrained, load_dataset, hf_hub_download, snapshot_download, pipeline), '
    + 'notebooks, shell scripts and Dockerfiles (vllm serve, the SGLang server, huggingface-cli / hf download) and YAML '
    + 'model keys. Three rules: pin/hf-unpinned (no revision, warn), pin/hf-floating-revision (a branch or tag such as '
    + '"main" or "v1.0", warn) and pin/hf-unpinned-dynamic (repo id in a variable, info; hidden unless include_dynamic). '
    + 'A literal commit of 7+ hex characters, a local path and `# nosec B615` are not flagged, as in Bandit B615. Each '
    + 'finding has the file, line, and the fix spelled for that file (revision="<sha>", --revision <sha>, '
    + 'model_revision:). Pass paths (files or directories; relative to the model\'s directory, default ".") or source '
    + 'with filename for text the server cannot read. Same scanner as `npx neurarch pin-check`. Offline: it never asks '
    + 'the Hub for the current commit. Not for checking what changed in a model: use model_changes for that.',
  modelOptional: true,
  inputSchema: {
    type: 'object',
    properties: {
      paths: { type: 'array', items: { type: 'string' }, description: 'Files or dirs. Default ".".' },
      source: { type: 'string', description: 'File text, instead of paths.' },
      filename: { type: 'string', description: 'For source, e.g. train.py.' },
      include_dynamic: { type: 'boolean', description: 'List ids held in variables.' },
      ...PAGE_PARAMS,
    },
    additionalProperties: false,
  },
  handler: async (a: PinArgs, _model, ctx) => {
    const hits: PinHit[] = [];
    let files = 0;
    let capped = false;
    if (a.source != null) {
      if (!a.filename) throw new Error('check_pins: source needs filename, whose extension says how to read it (train.py, serve.sh, config.yaml, Dockerfile, nb.ipynb).');
      if (!pinFileKind(posix(a.filename))) throw new Error(`check_pins: ${a.filename} is not a file kind the scanner reads. Use a .py, .ipynb, .sh/.bash/.slurm, .yaml/.yml or Dockerfile name.`);
      files = 1;
      hits.push(...scanPins(posix(a.filename), a.source));
    } else {
      const base = baseDir(ctx);
      const roots = a.paths?.length ? a.paths : ['.'];
      for (const p of roots) {
        const abs = isAbsolute(p) ? p : resolve(base, p);
        if (!(await stat(abs).catch(() => null))) throw new Error(`check_pins: ${p}: no such file or directory (resolved against ${base}).`);
        for await (const f of walk(abs)) {
          if (files >= MAX_FILES) { capped = true; break; }
          if (ctx.signal?.aborted) throw new Error('check_pins: cancelled.');
          const text = await readFile(f, 'utf-8').catch(() => null);
          if (text == null) continue;
          files++;
          if (ctx.progress && files % 250 === 0) await ctx.progress(files, undefined, `read ${files} files`);
          const rel = posix(relative(base, f)) || posix(f);
          hits.push(...scanPins(rel, text));
        }
        if (capped) break;
      }
    }
    const all = fromPinCheck(hits);
    const shown = a.include_dynamic ? all : all.filter(f => f.rule !== 'pin/hf-unpinned-dynamic');
    const counts = { warn: 0, info: 0 };
    for (const f of all) if (f.severity === 'warn' || f.severity === 'info') counts[f.severity]++;
    const page = pageOf(shown, a, 50);
    const unpinned = counts.warn;
    const verdict = unpinned
      ? `${unpinned} download${unpinned === 1 ? '' : 's'} not pinned to a commit in ${files} file${files === 1 ? '' : 's'}`
      : `every Hugging Face download found in ${files} file${files === 1 ? '' : 's'} is pinned to a commit, or none was found`;
    return {
      verdict,
      filesRead: files,
      ...(capped ? { capped: `Stopped after ${MAX_FILES} files; pass narrower paths.` } : {}),
      counts,
      findings: page.items.map(f => ({
        rule: f.rule,
        code: PIN_RULE_META[f.rule as keyof typeof PIN_RULE_META]?.code ?? null,
        severity: f.severity,
        path: f.path ?? null,
        line: f.line ?? null,
        model: f.model ?? null,
        title: f.title,
        // The CLI prints the pin and the Renovate sentence separately; one
        // preset URL at the top level says the second half once.
        fix: (f.fix ?? '').split(' Then extend ')[0],
      })),
      ...(a.include_dynamic || !counts.info ? {} : { hiddenDynamic: counts.info }),
      total: page.total,
      nextCursor: page.nextCursor,
      renovatePreset: RENOVATE_PRESET_URL,
      basis: 'Static scan of the files, aligned with Bandit B615; nothing was downloaded and the Hub was not asked.',
    };
  },
};

export const PIN_TOOLS: ToolDef[] = [checkPinsTool];
