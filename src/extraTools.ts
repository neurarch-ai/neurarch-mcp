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
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import type { ModelArchitecture } from './lib/types.js';
import type { ToolDef, ToolContext } from './tools.js';
import { rankDesigns, MAX_CANDIDATES, type NamedCandidate } from './lib/rank.js';
import { describeArchitecture } from './lib/describe.js';
import { loadModelCached } from './models.js';
import { listZoo, loadZooModel, loadHFModel, isHfEnabled } from './sources.js';
import { discoverModels } from './discover.js';
import { generatePyTorchCode } from './vendor/engine.bundle.mjs';

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
    + 'you to break ties on your own budget; they are never used to order. `calibration` is inside every result.',
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
            id: { type: 'string', description: 'Label for this candidate in the result. Defaults to the path or candidate-N.' },
            model_path: { type: 'string', description: 'A .py or .neurarch.json path, or zoo:<id> / hf:<org/name>.' },
            model: { type: 'object', description: 'An inline graph ({components, connections}) instead of a path.' },
          },
          additionalProperties: false,
        },
        description: 'The designs to order. Each needs either model_path or model.',
      },
      include_current: { type: 'boolean', description: 'Also rank the model this server was started with, as "current". Default false.' },
    },
    required: ['candidates'],
    additionalProperties: false,
  },
  handler: async (args: { candidates: Array<{ id?: string; model_path?: string; model?: ModelArchitecture }>; include_current?: boolean }, model) => {
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
    return rankDesigns(named);
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
      save_to: { type: 'string', description: 'Optional path to write the generated .py to. Requires --write.' },
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
    + 'against a known-good one without leaving the conversation.',
  inputSchema: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'Substring match on id, title, type or domain (case-insensitive).' },
      domain: { type: 'string', description: 'Exact domain filter, e.g. "Language models".' },
      attention: { type: 'string', description: 'Only entries using this attention type, e.g. "groupedQueryAttention".' },
    },
    additionalProperties: false,
  },
  handler: async ({ query, domain, attention }: { query?: string; domain?: string; attention?: string }) => {
    const all = await listZoo();
    const q = query?.toLowerCase();
    const hits = all.filter(e =>
      (!q || [e.id, e.title, e.type, e.domain].some(s => s.toLowerCase().includes(q)))
      && (!domain || e.domain === domain)
      && (!attention || e.attention.includes(attention)),
    );
    const domains = [...new Set(all.map(e => e.domain))].sort();
    return {
      total: all.length,
      matched: hits.length,
      domains,
      architectures: hits.map(e => ({
        id: e.id, title: e.title, domain: e.domain, type: e.type, params: e.params, layers: e.layers,
        attention: e.attention, paramCheck: e.paramCheck, paper: e.urls.paper, huggingface: e.urls.huggingface,
        modelPath: `zoo:${e.id}`,
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
      save_to: { type: 'string', description: 'Optional path for a .neurarch.json copy. Requires --write.' },
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
      save_to: { type: 'string', description: 'Optional path for a .neurarch.json copy. Requires --write.' },
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
      dir: { type: 'string', description: 'Directory to scan. Defaults to the directory of the model this server was started with.' },
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
