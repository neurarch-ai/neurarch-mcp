/**
 * The tools that arrived with 0.13: the ones that let an agent decide and act,
 * where the original nineteen let it look.
 *
 *   rank_designs        which of k candidate graphs deserves the GPU budget
 *   export_pytorch      the graph back as runnable source
 *   list_architectures  the bundled reference library, searchable
 *   load_architecture   one entry of it, as a graph any tool can then read
 *   load_hf_model       a Hugging Face repo as a graph (behind --hf)
 *   find_models         what in this repository is a model at all
 */
import { writeFile, readFile } from 'node:fs/promises';
import { resolve, basename } from 'node:path';
import type { ModelArchitecture } from './lib/types.js';
import type { ToolDef, ToolContext } from './tools.js';
import { rankDesigns, MAX_CANDIDATES, type NamedCandidate } from './lib/rank.js';
import { describeArchitecture } from './lib/describe.js';
import { loadModelCached } from './models.js';
import { listZoo, loadZooModel, loadHFModel, isHfEnabled } from './sources.js';
import { discoverModels } from './discover.js';
import { generatePyTorchCode, lintModelGraph } from './vendor/engine.bundle.mjs';
import { suggestFixes } from './lib/suggestFix.js';
import { pageOf, PAGE_PARAMS } from './paging.js';

/**
 * Creating a file is a write, even when the file is new. Tools that can save
 * take `save_to` and honour it only under --write, for the same reason
 * save_model is gated: an agent in a read-only session must not be able to
 * leave anything on disk.
 */
async function maybeSave(ctx: ToolContext, saveTo: string | undefined, text: string): Promise<{ savedTo?: string; saveRefused?: string }> {
  if (!saveTo) return {};
  if (!ctx.writeEnabled) {
    return { saveRefused: `save_to ignored: the server is read-only. Restart with --write to let tools create files. (${saveTo})` };
  }
  const abs = resolve(saveTo);
  await writeFile(abs, text, 'utf-8');
  return { savedTo: abs };
}

// ── rank_designs ─────────────────────────────────────────────────────────────
export const rankDesignsTool: ToolDef = {
  name: 'rank_designs',
  description:
    'Order several candidate designs for the one decision an agent cannot make by inspection: which of them to '
    + 'spend a training run on. Candidates are model files (model_path), zoo:<id> or hf:<id> references, or inline '
    + 'graphs; include_current adds the model this server holds. Blocked candidates (a pre-flight finding that means '
    + 'the graph will not forward-pass) rank last and are reported as budget you can reclaim; that part is measured '
    + '(96/96 blocked graphs crashed, 80/80 passes ran). Legal candidates are ordered only by rules with a trained '
    + 'outcome behind them, and a tie stays a tie: `recommended` is null when nothing measured separates the top '
    + 'candidates, which is the common and honest answer. Params, cost and GPU fit are returned per candidate for '
    + 'you to break ties on your own budget, or pass tie_break to have that done and labelled as yours. '
    + '`calibration` is inside every result.',
  inputSchema: {
    type: 'object',
    properties: {
      candidates: {
        type: 'array',
        minItems: 1,
        maxItems: MAX_CANDIDATES,
        items: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Label in the result.' },
            model_path: { type: 'string', description: 'File, zoo:<id>, hf:<org/name>' },
            model: { type: 'object', description: 'Inline graph instead of a path.' },
          },
          additionalProperties: false,
        },
        description: 'Designs to order; each needs model_path or model.',
      },
      include_current: { type: 'boolean', description: 'Also rank this server\'s model as "current".' },
      tie_break: {
        type: 'string',
        enum: ['none', 'cost', 'params'],
        description: 'Break a measured tie only.',
      },
    },
    required: ['candidates'],
    additionalProperties: false,
  },
  handler: async (args: { candidates: Array<{ id?: string; model_path?: string; model?: ModelArchitecture }>; include_current?: boolean; tie_break?: 'none' | 'cost' | 'params' }, model) => {
    const named: NamedCandidate[] = [];
    if (args.include_current) named.push({ id: 'current', model });
    for (const [i, c] of args.candidates.entries()) {
      if (c.model && c.model_path) throw new Error(`candidates[${i}]: give model_path or model, not both.`);
      if (c.model) {
        named.push({ id: c.id ?? `candidate-${i}`, model: c.model });
      } else if (c.model_path) {
        named.push({ id: c.id ?? c.model_path, model: await loadModelCached(c.model_path) });
      } else {
        throw new Error(`candidates[${i}] has neither model_path nor model.`);
      }
    }
    return rankDesigns(named, { tieBreak: args.tie_break });
  },
};

// ── export_pytorch ───────────────────────────────────────────────────────────
export const exportPytorchTool: ToolDef = {
  name: 'export_pytorch',
  description:
    'Render the current graph as a runnable PyTorch nn.Module, the same generator behind the app\'s Export panel. '
    + 'Use it after editing the graph with the write tools, or to hand a zoo:/hf: architecture to the user as code. '
    + 'The source is returned; pass save_to to also write it to a file (honoured only under --write, and never over '
    + 'the .py this server was started from: the graph was derived from that file and regenerating it would drop '
    + 'everything the parser could not read). Layer types the generator has no template for are left as a comment.',
  inputSchema: {
    type: 'object',
    properties: {
      save_to: { type: 'string', description: 'Write the .py here (--write).' },
    },
    additionalProperties: false,
  },
  handler: async ({ save_to }: { save_to?: string }, model, ctx) => {
    if (save_to && resolve(save_to) === resolve(ctx.modelPath)) {
      throw new Error('export_pytorch refuses to overwrite the file this server was started from. Choose another path.');
    }
    const code = generatePyTorchCode(model);
    const gaps = [...code.matchAll(/# (?:TODO|No layer code for component type)[^\n]*/g)].map(m => m[0]);
    return {
      language: 'python',
      lines: code.split('\n').length,
      code,
      unsupportedLayers: gaps,
      ...(await maybeSave(ctx, save_to, code)),
    };
  },
};

// ── list_architectures ───────────────────────────────────────────────────────
export const listArchitecturesTool: ToolDef = {
  name: 'list_architectures',
  description:
    'Search the reference library bundled with this server: verified graphs of published architectures '
    + '(DeepSeek-V3, Qwen2.5, Llama, Mixtral, Whisper, CLIP, BERT, ViT, ResNet and more), each with real dimensions '
    + 'from the model\'s config and a parameter count checked against the published one. Offline. Any read tool '
    + 'can then be asked about an entry by passing model_path: "zoo:<id>", so you can compare the user\'s design '
    + 'against a known-good one without leaving the conversation. Paged (20 per page, nextCursor); detail "full" adds '
    + 'attention kinds, the parameter check and paper / Hugging Face links per entry. Each entry is also a resource at '
    + 'neurarch://zoo/<id>.',
  inputSchema: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'Substring of id, title, type or domain.' },
      domain: { type: 'string', description: 'e.g. "Language models".' },
      attention: { type: 'string', description: 'e.g. "groupedQueryAttention".' },
      detail: { type: 'string', enum: ['concise', 'full'] },
      ...PAGE_PARAMS,
    },
    additionalProperties: false,
  },
  links: (r: { architectures?: Array<{ id: string; title: string }> }) => (r.architectures ?? []).map(e => ({
    uri: `neurarch://zoo/${e.id}`, name: e.id, title: e.title, mimeType: 'application/json',
  })),
  handler: async ({ query, domain, attention, detail, limit, cursor }: { query?: string; domain?: string; attention?: string; detail?: 'concise' | 'full'; limit?: number; cursor?: string }) => {
    const all = await listZoo();
    const q = query?.toLowerCase();
    const hits = all.filter(e =>
      (!q || [e.id, e.title, e.type, e.domain].some(s => s.toLowerCase().includes(q)))
      && (!domain || e.domain === domain)
      && (!attention || e.attention.includes(attention)),
    );
    const domains = [...new Set(all.map(e => e.domain))].sort();
    const page = pageOf(hits, { limit, cursor }, 20);
    return {
      total: all.length,
      matched: hits.length,
      domains,
      nextCursor: page.nextCursor,
      architectures: page.items.map(e => ({
        id: e.id, title: e.title, domain: e.domain, type: e.type, params: e.params, layers: e.layers,
        modelPath: `zoo:${e.id}`,
        ...(detail === 'full' ? { attention: e.attention, paramCheck: e.paramCheck, paper: e.urls.paper, huggingface: e.urls.huggingface } : {}),
      })),
    };
  },
};

// ── load_architecture ────────────────────────────────────────────────────────
export const loadArchitectureTool: ToolDef = {
  name: 'load_architecture',
  description:
    'Open one reference architecture from the bundled library and describe it: pipeline, depth, parameter and '
    + 'compute hotspots, validation. Pass save_to to write it out as a .neurarch.json the user can edit in the app '
    + 'or start this server on (requires --write). For further questions, pass model_path: "zoo:<id>" to any tool.',
  inputSchema: {
    type: 'object',
    properties: {
      id: { type: 'string', description: 'Entry id from list_architectures, e.g. "qwen2.5-7b".' },
      save_to: { type: 'string', description: 'Write a .neurarch.json copy (--write).' },
    },
    required: ['id'],
    additionalProperties: false,
  },
  handler: async ({ id, save_to }: { id: string; save_to?: string }, _model, ctx) => {
    const model = await loadZooModel(id);
    const entry = (await listZoo()).find(e => e.id === id);
    return {
      id,
      modelPath: `zoo:${id}`,
      title: entry?.title,
      paper: entry?.urls.paper,
      huggingface: entry?.urls.huggingface,
      paramCheck: entry?.paramCheck,
      ...describeArchitecture(model),
      ...(await maybeSave(ctx, save_to, JSON.stringify(model, null, 2) + '\n')),
    };
  },
};

// ── load_hf_model ────────────────────────────────────────────────────────────
export const loadHfModelTool: ToolDef = {
  name: 'load_hf_model',
  description:
    'Build a graph for a Hugging Face model from its config.json and describe it. This is the one tool that '
    + 'reaches the network (huggingface.co only; HF_TOKEN is sent for gated repos), which is why it is listed only '
    + 'when the server runs with --hf. The result says whether it came from the real config or a generic family '
    + 'template (configSource), and quotes the parameter count HF publishes so you can see how close the graph is. '
    + 'Cached for a day under ~/.cache/neurarch-mcp. Afterwards any tool accepts model_path: "hf:<org/name>".',
  annotations: { openWorldHint: true, idempotentHint: false },
  inputSchema: {
    type: 'object',
    properties: {
      model_id: { type: 'string', description: 'Repo id, e.g. "Qwen/Qwen2.5-7B" or "meta-llama/Llama-3.1-8B".' },
      save_to: { type: 'string', description: 'Write a .neurarch.json copy (--write).' },
    },
    required: ['model_id'],
    additionalProperties: false,
  },
  handler: async ({ model_id, save_to }: { model_id: string; save_to?: string }, _model, ctx) => {
    const r = await loadHFModel(model_id);
    return {
      modelId: model_id,
      modelPath: `hf:${model_id}`,
      configSource: r.configSource,
      approximate: r.configSource === 'fallback',
      realParamCount: r.realParamCount ?? null,
      ...describeArchitecture(r.model),
      ...(await maybeSave(ctx, save_to, JSON.stringify(r.model, null, 2) + '\n')),
    };
  },
};

// ── find_models ──────────────────────────────────────────────────────────────
export const findModelsTool: ToolDef = {
  name: 'find_models',
  description:
    'Walk a directory for model definitions: .py files declaring an nn.Module subclass and saved .neurarch.json '
    + 'graphs. Each .py is tried against the parser, so the answer distinguishes files this server can read now from '
    + 'ones that need a runtime trace (neurarch-trace). Use it on arrival in an unfamiliar repository, then pass the '
    + 'path you want as model_path to any other tool. Skips node_modules, virtualenvs, checkpoints and the like.',
  inputSchema: {
    type: 'object',
    properties: {
      dir: { type: 'string', description: 'Default: the server model\'s directory.' },
      max_files: { type: 'integer', minimum: 1, maximum: 5000, description: 'Cap on files opened (default 400).' },
    },
    additionalProperties: false,
  },
  handler: async ({ dir, max_files }: { dir?: string; max_files?: number }, _model, ctx) => {
    const root = dir ?? resolve(ctx.modelPath, '..');
    return discoverModels({ dir: root, maxFiles: max_files });
  },
};

/** Read tools added in 0.13, in the order they are listed. */
export const EXTRA_TOOLS: ToolDef[] = [
  rankDesignsTool,
  exportPytorchTool,
  listArchitecturesTool,
  loadArchitectureTool,
  findModelsTool,
];

/** Listed only when --hf is on: a tool that can never succeed should not be advertised. */
export const HF_TOOLS: ToolDef[] = [loadHfModelTool];

export function hfToolsIfEnabled(): ToolDef[] {
  return isHfEnabled() ? HF_TOOLS : [];
}

// ── suggest_fix ──────────────────────────────────────────────────────────────
export const suggestFixTool: ToolDef = {
  name: 'suggest_fix',
  description:
    'Turn lint findings into edits to the source file: a unified diff per finding, labelled exact (a number or '
    + 'an order the rule pins, changed on every line that shares it) or proposal (a missing layer inserted with a '
    + 'note on what forward() still needs). Works on a .py; for a .neurarch.json use the write tools. Findings with '
    + 'no mechanical fix come back under notFixable with the reason. Apply the diff, then lint_model again.',
  inputSchema: {
    type: 'object',
    properties: {
      rule: { type: 'string', description: 'Only this rule id.' },
      layer: { type: 'string', description: 'Only this layer.' },
      severity: { type: 'string', enum: ['block', 'warn', 'info'], description: 'Minimum. Default warn.' },
      save_to: { type: 'string', description: 'Write the patched file (exact fixes, --write).' },
    },
    additionalProperties: false,
  },
  handler: async ({ rule, layer, severity, save_to }: { rule?: string; layer?: string; severity?: 'block' | 'warn' | 'info'; save_to?: string }, model, ctx) => {
    const path = ctx.currentPath ?? ctx.modelPath;
    if (!path || !/\.py$/i.test(path)) {
      return {
        fixes: [], notFixable: [],
        note: 'suggest_fix edits PyTorch source. This model did not come from a .py (pass model_path to a .py, or start the server on one); for a graph file use modify_layer / add_layer / delete_layer under --write.',
      };
    }
    const source = await readFile(path, 'utf-8');
    const order = { block: 0, warn: 1, info: 2 } as const;
    const floor = order[severity ?? 'warn'];
    const findings = lintModelGraph(model).filter(f =>
      order[f.severity] <= floor && (!rule || f.rule === rule) && (!layer || f.componentName === layer));
    const result = suggestFixes(model, findings, source, basename(path));
    if (save_to && resolve(save_to) === resolve(path)) throw new Error('suggest_fix will not overwrite the source file; choose another save_to or apply the diff yourself.');
    const saved = result.patchedSource ? await maybeSave(ctx, save_to, result.patchedSource) : {};
    const { patchedSource: _omit, ...rest } = result;
    return { ...rest, ...(save_to ? saved : {}), considered: findings.length };
  },
};
EXTRA_TOOLS.push(suggestFixTool);

// ── trace_model ──────────────────────────────────────────────────────────────
/**
 * The static parser reads source; this runs it. neurarch-trace (pip) builds
 * the model, runs one forward with hooks and writes a .neurarch.json with
 * real shapes, so every tool works on code the parser cannot follow. The
 * server shells out rather than embedding Python: the model's own environment
 * is the only place its imports resolve.
 */
export const traceModelTool: ToolDef = {
  name: 'trace_model',
  description:
    'Trace the model at runtime with neurarch-trace (pip install neurarch-trace) and get a graph with real shapes: '
    + 'instantiate it in Python, run one forward pass with hooks, write a .neurarch.json. Use when find_models or '
    + 'parseQuality says the static parse is thin or partial. target is "path/to/file.py:ClassOrFactory", '
    + '"module.path:attr" or "hf:<repo>"; input is the batch-first dims per model input. The graph is cached and '
    + 'returned as a model_path any tool accepts. Runs the user\'s Python (NEURARCH_PYTHON or python3).',
  annotations: { idempotentHint: false },
  inputSchema: {
    type: 'object',
    properties: {
      target: { type: 'string', description: '"path/to/model.py:ClassName", "pkg.module:factory", or "hf:org/name".' },
      input: { type: 'array', items: { type: 'string' }, minItems: 1, description: 'Input dims, batch first, one per input: ["1,3,224,224"]. ":long" for token ids.' },
      dtype: { type: 'string', description: 'Input dtype (float32; long for hf:).' },
      depth: { type: 'integer', minimum: 1, description: 'Default: leaves.' },
      name: { type: 'string' },
      save_to: { type: 'string', description: 'Write the graph here (--write).' },
      timeout_s: { type: 'integer', minimum: 5, maximum: 600, description: 'Seconds (default 120).' },
    },
    required: ['target', 'input'],
    additionalProperties: false,
  },
  handler: async (args: { target: string; input: string[]; dtype?: string; depth?: number; name?: string; save_to?: string; timeout_s?: number }, _model, ctx) => {
    const { spawn } = await import('node:child_process');
    const { mkdir, readFile: rf, copyFile } = await import('node:fs/promises');
    const { homedir } = await import('node:os');
    const { join, isAbsolute } = await import('node:path');
    const python = process.env.NEURARCH_PYTHON || 'python3';
    const dir = process.env.NEURARCH_MCP_CACHE ? join(process.env.NEURARCH_MCP_CACHE, 'traces') : join(homedir(), '.cache', 'neurarch-mcp', 'traces');
    await mkdir(dir, { recursive: true });
    const stem = (args.name ?? args.target.split(/[:/\\]/).pop() ?? 'model').replace(/[^\w.-]+/g, '_');
    const out = join(dir, `${stem}-${Date.now()}.neurarch.json`);
    // A file target relative to the server's model directory resolves there,
    // which is where an agent that just ran find_models expects it to.
    let target = args.target;
    const fileMatch = /^(.+\.py):(\w+)$/.exec(target);
    if (fileMatch && !isAbsolute(fileMatch[1])) target = `${resolve(ctx.modelPath ? join(ctx.modelPath, '..') : '.', fileMatch[1])}:${fileMatch[2]}`;
    const argv = ['-m', 'neurarch_trace', target, '-o', out];
    for (const i of args.input) argv.push('--input', i);
    if (args.dtype) argv.push('--dtype', args.dtype);
    if (args.depth) argv.push('--depth', String(args.depth));
    if (args.name) argv.push('--name', args.name);

    let cancelled = false;
    const result = await new Promise<{ code: number | null; stdout: string; stderr: string; timedOut: boolean }>(res => {
      const child = spawn(python, argv, { cwd: fileMatch ? join(target.split(':')[0], '..') : process.cwd() });
      let stdout = '', stderr = '', timedOut = false;
      child.stdout.on('data', d => { stdout += d; });
      child.stderr.on('data', d => { stderr += d; });
      const limitS = args.timeout_s ?? 120;
      const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, limitS * 1000);
      // A cancelled call kills the trace rather than leaving Python running
      // against a request nobody is waiting for.
      const onAbort = () => { cancelled = true; child.kill('SIGKILL'); };
      ctx.signal?.addEventListener('abort', onAbort, { once: true });
      // A trace can take a minute; a client that asked for progress hears the
      // elapsed seconds against the timeout every five, which is all that is
      // known from outside the Python process.
      const started = Date.now();
      const beat = ctx.progress
        ? setInterval(() => { const s = Math.round((Date.now() - started) / 1000); void ctx.progress!(s, limitS, `tracing ${args.target}: ${s}s of at most ${limitS}s`); }, 5000)
        : undefined;
      const done = () => { clearTimeout(timer); if (beat) clearInterval(beat); ctx.signal?.removeEventListener('abort', onAbort); };
      child.on('error', e => { done(); res({ code: null, stdout, stderr: stderr + e.message, timedOut }); });
      child.on('close', code => { done(); res({ code, stdout, stderr, timedOut }); });
    });

    if (cancelled) throw new Error('trace_model: cancelled by the client; the Python process was killed.');
    if (result.timedOut) throw new Error(`trace_model: the trace did not finish in ${args.timeout_s ?? 120}s and was killed. Try a smaller input or --depth.`);
    if (result.code !== 0) {
      const tail = result.stderr.trim().split('\n').slice(-6).join('\n');
      if (/No module named neurarch_trace/.test(result.stderr)) {
        throw new Error(`trace_model: neurarch-trace is not installed for ${python}. Run: ${python} -m pip install neurarch-trace (set NEURARCH_PYTHON to the interpreter that has torch and the model's imports).`);
      }
      if (/ENOENT|spawn .* ENOENT/.test(result.stderr)) throw new Error(`trace_model: ${python} not found. Set NEURARCH_PYTHON to a Python with torch and neurarch-trace installed.`);
      throw new Error(`trace_model failed (exit ${result.code}):\n${tail}`);
    }
    const graph = JSON.parse(await rf(out, 'utf-8')) as ModelArchitecture;
    const saved = args.save_to ? await maybeSave(ctx, args.save_to, JSON.stringify(graph, null, 2) + '\n') : {};
    void copyFile;
    return {
      modelPath: out,
      layers: graph.components.length,
      connections: graph.connections.length,
      ...describeArchitecture(graph),
      ...saved,
      stderrTail: result.stderr.trim().split('\n').slice(-3).join('\n') || undefined,
    };
  },
};
EXTRA_TOOLS.push(traceModelTool);
