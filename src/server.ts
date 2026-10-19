/**
 * The MCP server wiring, transport-agnostic. Both the stdio path (index.ts) and
 * the HTTP path (http.ts) build a server here and connect their own transport,
 * so the tool list, dispatch, and error shaping live in exactly one place.
 *
 * The model is read through a getter rather than captured by value: --watch
 * reassigns the current model when the file changes on disk, and every tool
 * call must see the latest one. The getter may throw: a hosted server started
 * with no model has nothing to return until a call names one (model_path or
 * model_source), and that error is the right answer for a tool that did not.
 *
 * Three MCP surfaces, and why each is here:
 *
 *   tools      what the agent calls. See tools.ts, extraTools.ts, writeTools.ts.
 *   prompts    what the USER can pick from a menu: Claude Desktop and Cursor
 *              render these as slash commands, which is the only place a
 *              person discovers what this server is for without reading a
 *              README. Each one is a recipe over the tools, written so the
 *              agent climbs the ladder in order rather than guessing.
 *   resources  what the client can attach as context: the graph itself, its
 *              diagram, the generated source, the reference library index and
 *              the rule provenance table. A resource is fetched once and
 *              pinned; a tool result scrolls away.
 */
import { Server, inputRequired, type ServerContext, type Tool } from '@modelcontextprotocol/server';
import type { ModelArchitecture } from './lib/types.js';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { TOOLS, type ToolContext, type ToolDef } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';
import { HF_TOOLS } from './extraTools.js';
import { listedTools, resolveToolCall, isWriteTool, declaresOutputSchema, MODEL_PATH_PARAM, type ToolSetName } from './cli.js';
import { reportingEnabled, buildCorpusReport, sendCorpusReport } from './lib/corpusReport.js';
import { loadModelCached } from './models.js';
import { listZoo, loadZooModel, modelFromText } from './sources.js';
import { renderMermaid } from './mermaid.js';
import { generatePyTorchCode } from './vendor/engine.bundle.mjs';
import { RULE_PROVENANCE } from './vendor/verifier.bundle.mjs';
import { PROMPTS, renderPrompt } from './prompts.js';
import { ruleCatalogue } from './ruleCatalogue.js';
import { validateArgs } from './argCheck.js';
import { SERVER_ICONS } from './icon.js';

/**
 * The tools that grade a graph rather than describe one.
 *
 * A corpus row records "this shape got this verdict", so it is meaningful only
 * after something graded the shape. get_layer_info and the rest inspect; these
 * four judge.
 *
 * `plan` is here so that NEURARCH_REPORT means one thing across every tool
 * that grades, rather than a coverage gap an agent falls into by reaching for
 * the newest tool. The row is derived from the graph, not from the response,
 * so it is the same row the other three send and the server's fingerprint
 * dedupes it. `history` is deliberately absent: it grades nothing, it is
 * handed a fingerprint rather than a graph, and there is no verdict on it to
 * record.
 */
const GRADING_TOOLS = new Set(['validate_model', 'lint_model', 'check_design', 'plan']);

/** Argument name for "answer this about model text I am handing you inline". */
export const MODEL_SOURCE_PARAM = 'model_source';

export interface McpServerOptions {
  /** Reads the current model. A getter so --watch reloads are always seen. May throw when none is loaded. */
  getModel: () => ModelArchitecture;
  ctx: ToolContext;
  writeEnabled: boolean;
  version: string;
  toolSet?: ToolSetName;
}

/** Build a configured MCP Server (no transport attached yet). */
export function createMcpServer(opts: McpServerOptions): Server {
  const { getModel, ctx, writeEnabled, version } = opts;
  const toolSet = opts.toolSet ?? 'core';
  const withSource = (t: ReturnType<typeof listedTools>[number]) => ({
    ...t,
    inputSchema: isWriteTool(t.name) ? t.inputSchema : withModelSource(t.inputSchema),
  });
  const tools = listedTools(writeEnabled, toolSet).map(withSource);
  // What a call is validated against: every callable tool, listed or not, in
  // the exact shape a client would have been shown under --tools=full.
  const callable = new Map(listedTools(writeEnabled, 'full').map(withSource).map(t => [t.name, t]));

  const server = new Server(
    { name: 'neurarch-mcp', title: 'Neurarch', version, websiteUrl: 'https://www.neurarch.com/mcp', icons: SERVER_ICONS },
    { capabilities: { tools: {}, prompts: {}, resources: {}, completions: {} }, instructions: SERVER_INSTRUCTIONS },
  );

  server.setRequestHandler('tools/list', async () => ({ tools: tools as Tool[] }));

  server.setRequestHandler('tools/call', async (req, extra) => {
    const { tool, errorText } = resolveToolCall(req.params.name, writeEnabled);
    if (!tool) {
      return {
        isError: true,
        content: [{ type: 'text', text: errorText! }],
      };
    }

    // `model_path` lets one server answer about any model in the repository
    // instead of only the one it was launched with; `model_source` does the
    // same for text the client holds and this process cannot read from disk.
    // Both are stripped before the handler sees them: every schema is
    // additionalProperties:false, and a tool should not have to know they exist.
    const rawArgs = (req.params.arguments ?? {}) as Record<string, unknown>;
    const pathArg = rawArgs[MODEL_PATH_PARAM];
    const sourceArg = rawArgs[MODEL_SOURCE_PARAM];
    const wantsOtherModel = typeof pathArg === 'string' && pathArg.trim() !== '';
    const wantsInline = typeof sourceArg === 'string' && sourceArg.trim() !== '';

    if ((wantsOtherModel || wantsInline) && isWriteTool(tool.name)) {
      // Refused rather than ignored. Honouring it would let an agent mutate,
      // and then save over, a file nobody pointed this server at; silently
      // dropping it would mutate a different model than the one the agent
      // believes it is editing. Both are worse than an error that says so.
      return {
        isError: true,
        content: [{
          type: 'text',
          text: `${tool.name} does not accept ${MODEL_PATH_PARAM} or ${MODEL_SOURCE_PARAM}. Mutations always target the model file `
            + 'this server was started with. Start a second server pointed at the other file to edit it.',
        }],
      };
    }
    if (wantsOtherModel && wantsInline) {
      return { isError: true, content: [{ type: 'text', text: `${tool.name}: give ${MODEL_PATH_PARAM} or ${MODEL_SOURCE_PARAM}, not both.` }] };
    }

    // Arguments are checked against the advertised schema before a handler
    // sees them (after the model_path refusal above, whose message is the more
    // specific one). Without this a missing required argument reached the
    // handler as undefined and came back as "Cannot read properties of
    // undefined", which tells an agent nothing about what to send instead.
    const listed = callable.get(tool.name);
    const argError = listed ? validateArgs(tool.name, listed.inputSchema, rawArgs) : null;
    if (argError) return { isError: true, content: [{ type: 'text', text: argError }] };

    try {
      let model: ModelArchitecture;
      const { [MODEL_PATH_PARAM]: _p, [MODEL_SOURCE_PARAM]: _s, ...args } = rawArgs;
      if (wantsOtherModel) {
        try {
          model = await loadModelCached(pathArg as string);
        } catch (e) {
          return { isError: true, content: [{ type: 'text', text: `${tool.name}: ${(e as Error).message}` }] };
        }
      } else if (wantsInline) {
        try {
          model = modelFromText(sourceArg as string);
        } catch (e) {
          return { isError: true, content: [{ type: 'text', text: `${tool.name}: ${(e as Error).message}` }] };
        }
      } else {
        try {
          model = getModel();
        } catch (e) {
          // A tool that names its own subject still has an answer on a server
          // with no model: `history` takes a fingerprint. Every other tool is
          // a question about a graph, and the missing-model error is the right
          // answer for one that named none.
          if (!tool.modelOptional) throw e;
          model = undefined as unknown as ModelArchitecture;
        }
      }

      const base: ToolContext = wantsOtherModel && !/^(zoo|hf):/i.test(pathArg as string)
        ? { ...ctx, currentPath: resolve(pathArg as string) }
        : wantsOtherModel || wantsInline ? { ...ctx, currentPath: '' } : ctx;
      // Elicitation is offered only when the client said it can answer. On a
      // 2025-era connection the question is pushed (elicitation/create) and
      // the handler waits for it; on 2026-07-28 there is no server-to-client
      // request, so the call returns input_required and the client retries it
      // with the answer. See elicit() below for why a retry is safe here.
      const canAsk = !!server.getClientCapabilities()?.elicitation;
      const callCtx: ToolContext = {
        ...base,
        ...(canAsk ? { elicit: (q) => elicit(server, extra, q) } : {}),
        signal: extra.mcpReq.signal,
        ...progressReporter(extra.mcpReq._meta?.progressToken ?? req.params._meta?.progressToken, extra),
      };
      const result = await tool.handler(args, model, callCtx);
      // Opt-in corpus row (NEURARCH_REPORT=1). Fire and forget: it can neither
      // slow nor fail the tool call. Privacy scope in lib/corpusReport.ts.
      //
      // All three grading tools, not just validate_model. The old wiring made
      // the channel's coverage an accident of which tool an agent reached for,
      // and validate_model is the narrowest of the three: an agent that asked
      // for the design rules or the whole verdict recorded nothing at all. The
      // row is derived from the graph rather than from the result, so these
      // three send the same row and the server's fingerprint dedupes them.
      if (model && GRADING_TOOLS.has(tool.name) && reportingEnabled()) {
        sendCorpusReport(buildCorpusReport(model));
      }
      // Both shapes, deliberately. `structuredContent` (spec 2025-06-18) is what
      // a modern client should read; the JSON text stays because clients that
      // predate it would otherwise see an empty result, and because a human
      // reading a transcript can still see what came back.
      //
      // Tools with a stable result shape also declare an `outputSchema`
      // (cli.ts OUTPUT_SCHEMAS). A client validates structuredContent against
      // it and fails the call on a mismatch, so those schemas name only the
      // keys a handler always returns, and outputSchema.test.ts calls every
      // one of them through a real client, which runs that validation.
      const structured =
        result !== null && typeof result === 'object' && !Array.isArray(result)
          ? (result as Record<string, unknown>)
          : undefined;
      // A handler that answered { error } is reporting a failed call, and the
      // spec's place for that is isError: the agent then reads it as something
      // to fix, not as data. Kept as the handlers' convention because the
      // direct-call tests and the CLI read it the same way.
      if (structured && typeof structured.error === 'string') {
        return { isError: true, content: [{ type: 'text', text: `${tool.name}: ${structured.error}` }] };
      }
      // A declared outputSchema makes structuredContent a MUST on success.
      // Silently omitting it would hand a validating client a result it has
      // to reject, so a handler that broke its own contract fails loudly here.
      if (!structured && declaresOutputSchema(tool.name)) {
        return {
          isError: true,
          content: [{ type: 'text', text: `${tool.name}: internal error, the result is not an object although the tool declares an outputSchema` }],
        };
      }
      const links = structured && tool.links ? tool.links(structured) : [];
      return {
        content: [
          { type: 'text', text: JSON.stringify(result, null, 2) },
          ...links.map(l => ({ type: 'resource_link' as const, ...l })),
        ],
        ...(structured ? { structuredContent: structured } : {}),
      };
    } catch (e) {
      if (e instanceof InputNeeded) return inputRequired({ inputRequests: { [e.key]: inputRequired.elicit(e.params) } });
      return {
        isError: true,
        content: [{ type: 'text', text: `${tool.name} failed: ${(e as Error).message}` }],
      };
    }
  });

  // ── prompts ────────────────────────────────────────────────────────────────
  server.setRequestHandler('prompts/list', async () => ({
    prompts: PROMPTS.map(p => ({ name: p.name, title: p.title, description: p.description, arguments: p.arguments })),
  }));

  server.setRequestHandler('prompts/get', async (req) => {
    const prompt = PROMPTS.find(p => p.name === req.params.name);
    if (!prompt) throw new Error(`Unknown prompt: ${req.params.name}. Available: ${PROMPTS.map(p => p.name).join(', ')}.`);
    const args = (req.params.arguments ?? {}) as Record<string, string>;
    for (const a of prompt.arguments ?? []) {
      if (a.required && !args[a.name]) throw new Error(`Prompt ${prompt.name} needs argument "${a.name}": ${a.description}`);
    }
    return {
      description: prompt.description,
      messages: [{ role: 'user', content: { type: 'text', text: renderPrompt(prompt, args, { writeEnabled }) } }],
    };
  });

  // ── resources ──────────────────────────────────────────────────────────────
  const STATIC_RESOURCES = [
    { uri: 'neurarch://model', name: 'Current model graph', description: 'The model this server was started with, as .neurarch.json.', mimeType: 'application/json' },
    { uri: 'neurarch://model/mermaid', name: 'Current model diagram', description: 'The current model as a Mermaid flowchart.', mimeType: 'text/plain' },
    { uri: 'neurarch://model/pytorch', name: 'Current model as PyTorch', description: 'Runnable nn.Module source generated from the current graph.', mimeType: 'text/x-python' },
    { uri: 'neurarch://zoo', name: 'Reference architecture index', description: 'The bundled library of verified published architectures. Each entry is readable at neurarch://zoo/{id}.', mimeType: 'application/json' },
    { uri: 'neurarch://rules', name: 'Rule provenance', description: 'For every design rule with a published measurement behind it, the measurement and its source.', mimeType: 'application/json' },
    { uri: 'neurarch://rules/catalogue', name: 'Rule catalogue', title: 'Every rule id this server can report', description: 'Every rule id a tool here can return (lint, shape, edit, pin), with its source and, where one exists, the measurement behind it.', mimeType: 'application/json' },
    { uri: 'neurarch://docs', name: 'Tool documentation', description: 'The long-form contract of every tool, including the ones --tools=core does not list. One read instead of 7k tokens on every turn.', mimeType: 'text/markdown' },
  ];

  server.setRequestHandler('resources/list', async () => ({ resources: STATIC_RESOURCES }));

  server.setRequestHandler('resources/templates/list', async () => ({
    resourceTemplates: [
      { uriTemplate: 'neurarch://zoo/{id}', name: 'Reference architecture', description: 'One bundled architecture as .neurarch.json; ids come from neurarch://zoo.', mimeType: 'application/json' },
      { uriTemplate: 'neurarch://docs/{tool}', name: 'Tool documentation', description: 'The full description and input schema of one tool.', mimeType: 'text/markdown' },
    ],
  }));

  server.setRequestHandler('resources/read', async (req) => {
    const uri = req.params.uri;
    const text = async (): Promise<{ mimeType: string; text: string }> => {
      if (uri === 'neurarch://model') return { mimeType: 'application/json', text: JSON.stringify(getModel(), null, 2) };
      if (uri === 'neurarch://model/mermaid') return { mimeType: 'text/plain', text: renderMermaid(getModel()) };
      if (uri === 'neurarch://model/pytorch') return { mimeType: 'text/x-python', text: generatePyTorchCode(getModel()) };
      if (uri === 'neurarch://zoo') return { mimeType: 'application/json', text: JSON.stringify(await listZoo(), null, 2) };
      if (uri === 'neurarch://rules') return { mimeType: 'application/json', text: JSON.stringify(RULE_PROVENANCE, null, 2) };
      if (uri === 'neurarch://rules/catalogue') return { mimeType: 'application/json', text: JSON.stringify(ruleCatalogue(), null, 2) };
      const zoo = /^neurarch:\/\/zoo\/([\w.-]+)$/.exec(uri);
      if (zoo) return { mimeType: 'application/json', text: JSON.stringify(await loadZooModel(zoo[1]), null, 2) };
      if (uri === 'neurarch://docs') return { mimeType: 'text/markdown', text: renderToolDocs(allToolDefs()) };
      const doc = /^neurarch:\/\/docs\/([\w-]+)$/.exec(uri);
      if (doc) {
        const t = allToolDefs().find(x => x.name === doc[1]);
        if (!t) throw new Error(`No tool named ${doc[1]}. Read neurarch://docs for the list.`);
        return { mimeType: 'text/markdown', text: renderToolDocs([t]) };
      }
      throw new Error(`Unknown resource: ${uri}`);
    };
    const { mimeType, text: body } = await text();
    return { contents: [{ uri, mimeType, text: body }] };
  });

  // ── completions ────────────────────────────────────────────────────────────
  // The values a prompt argument or a resource template variable can take,
  // where this server knows the set: zoo ids, rule ids, tool names. A client
  // shows them as the person types, which beats a slash command that fails on
  // a misspelled id.
  server.setRequestHandler('completion/complete', async (req) => {
    const { ref, argument } = req.params;
    let pool: string[] = [];
    if (ref.type === 'ref/prompt') {
      if (argument.name === 'architecture') pool = (await listZoo()).map(e => e.id);
      else if (argument.name === 'rule') pool = ruleCatalogue().rules.map(r => r.id);
      else if (argument.name === 'gpu') pool = GPU_HINTS;
    } else if (ref.type === 'ref/resource') {
      if (ref.uri === 'neurarch://zoo/{id}' && argument.name === 'id') pool = (await listZoo()).map(e => e.id);
      else if (ref.uri === 'neurarch://docs/{tool}' && argument.name === 'tool') pool = allToolDefs().map(t => t.name);
    }
    const q = (argument.value ?? '').toLowerCase();
    const hits = pool.filter(v => v.toLowerCase().startsWith(q)).concat(pool.filter(v => !v.toLowerCase().startsWith(q) && v.toLowerCase().includes(q)));
    return { completion: { values: hits.slice(0, 100), total: hits.length, hasMore: hits.length > 100 } };
  });

  return server;
}

/**
 * Card ids from the vendored plan bundle's GPU table (server.test.ts checks
 * each one is there). Offered as completions only, never enforced: the tools
 * resolve their own `gpu` argument, prefixes included.
 */
export const GPU_HINTS = ['h100-sxm', 'h200', 'b200', 'gh200-96g', 'gh200-144g', 'a100-80g', 'a100-40g', 'l40s', 'a40', 'rtx-pro-6000', 'l4', 'a10g', 't4', 'v100', 'rtx-4090', 'rtx-3090'];

/**
 * Sent once at initialize (InitializeResult.instructions). Clients that read
 * it put it in the system prompt, so it is the one place to say which tool
 * answers which moment, and what kind of number comes back. Kept short: it is
 * paid on every conversation the server is attached to.
 */
export const SERVER_INSTRUCTIONS = [
  'Neurarch pre-checks a change to an open-weight model before it takes GPU time or traffic.',
  'Before a training launch: size_training_job (peak memory at your batch, smallest card, max batch, cost).',
  'Before a multi-GPU split: check_parallel_layout. Before a vLLM / SGLang / llama.cpp launch: check_serving_config.',
  'After editing a config or architecture: check_edit. After a CUDA OOM: diagnose_oom, before changing the batch.',
  'Hugging Face downloads: check_pins finds unpinned ones; model_changes says what changed upstream between revisions.',
  'Structure: lint_model, then check_design. For a .py file, trace_model first: the static parser is the weak path.',
  'Memory figures are calibrated estimates and times are predictions; each result labels its own, so quote the label.',
  'No tool here predicts accuracy or model quality.',
  'Offline except plan, history and model_changes (www.neurarch.com) and hf: models under --hf (huggingface.co).',
].join(' ');

/**
 * notifications/progress for this call, when the client asked for them by
 * sending a progressToken. Progress must increase, so a report that would not
 * is dropped rather than sent out of order.
 */
function progressReporter(
  token: string | number | undefined,
  extra: { mcpReq: { notify: (n: { method: 'notifications/progress'; params: { progressToken: string | number; progress: number; total?: number; message?: string } }) => Promise<void> } },
): Pick<ToolContext, 'progress'> {
  if (token === undefined) return {};
  let last = -Infinity;
  return {
    progress: async (progress, total, message) => {
      if (!(progress > last)) return;
      last = progress;
      await extra.mcpReq.notify({
        method: 'notifications/progress',
        params: { progressToken: token, progress, ...(total !== undefined ? { total } : {}), ...(message ? { message } : {}) },
      }).catch(() => undefined);
    },
  };
}

function allToolDefs(): ToolDef[] {
  return [...TOOLS, ...HF_TOOLS, ...WRITE_TOOLS];
}

function renderToolDocs(defs: ToolDef[]): string {
  return defs.map(t => {
    const gate = WRITE_TOOLS.includes(t) ? ' (write tool, needs --write)' : HF_TOOLS.includes(t) ? ' (needs --hf)' : '';
    const props = ((t.inputSchema as { properties?: Record<string, { description?: string; type?: string }> }).properties) ?? {};
    const args = Object.entries(props).map(([k, v]) => `- \`${k}\`${v.type ? ` (${v.type})` : ''}: ${v.description ?? ''}`).join('\n');
    return `## ${t.name}${gate}\n\n${t.description}\n\n${args ? `Arguments:\n${args}\n` : 'No arguments.\n'}`;
  }).join('\n');
}

/** The 2026-07-28 protocol revision: no server-to-client requests, input arrives as a retried call. */
export const MODERN_PROTOCOL = '2026-07-28';

/** Whether this connection (2025 era) or this request (2026 era) speaks the 2026-07-28 revision. */
function servesModernEra(server: Server): boolean {
  return (server.getNegotiatedProtocolVersion() ?? '') >= MODERN_PROTOCOL;
}

/**
 * Thrown by elicit() on a 2026-07-28 request whose answer has not arrived
 * yet, and turned into an input_required result by the tools/call handler.
 * Not an error the agent ever sees.
 */
class InputNeeded extends Error {
  constructor(readonly key: string, readonly params: ElicitForm) {
    super('input required');
  }
}

type ElicitForm = Parameters<typeof inputRequired.elicit>[0];

/** A stable input key per question, so the retried call finds its own answer. */
function questionKey(message: string): string {
  return `choice_${createHash('sha256').update(message).digest('hex').slice(0, 12)}`;
}

/**
 * One elicitation, shaped as a single-choice form. `value` comes back only on
 * accept; decline and cancel are answers too and are passed through, because
 * "the user did not want to decide" is information the tool should report.
 *
 * Two eras. A 2025-era connection gets the push request and the handler waits
 * on it. A 2026-07-28 request cannot push: the first entry throws InputNeeded
 * (the call answers input_required), the client asks the person and retries
 * the same call with the answer in inputResponses, and the handler runs again
 * from the top. Re-running is safe because the only tool that asks
 * (check_design with ask_user) does nothing before the question but read the
 * graph, so the retry recomputes the same decision. No requestState is
 * minted: nothing the first round learned is carried, so there is nothing a
 * client could forge.
 */
async function elicit(
  server: Server,
  ctx: ServerContext,
  q: { message: string; options: Array<{ value: string; label: string; hint?: string }> },
): Promise<{ action: 'accept' | 'decline' | 'cancel'; value?: string }> {
  const form = {
    message: q.message,
    requestedSchema: {
      type: 'object' as const,
      properties: {
        choice: {
          type: 'string' as const,
          title: 'Your choice',
          enum: q.options.map(o => o.value),
          enumNames: q.options.map(o => (o.hint ? `${o.label}: ${o.hint}` : o.label)),
        },
      },
      required: ['choice'],
    },
  };
  let res: { action: 'accept' | 'decline' | 'cancel'; content?: unknown };
  if (servesModernEra(server)) {
    const key = questionKey(q.message);
    const answer = ctx.mcpReq.inputResponses?.[key] as { action?: unknown; content?: unknown } | undefined;
    if (!answer || (answer.action !== 'accept' && answer.action !== 'decline' && answer.action !== 'cancel')) {
      throw new InputNeeded(key, form as ElicitForm);
    }
    res = { action: answer.action, content: answer.content };
  } else {
    res = await ctx.mcpReq.elicitInput(form);
  }
  const value = res.action === 'accept' ? String((res.content as Record<string, unknown> | undefined)?.choice ?? '') : undefined;
  return { action: res.action, ...(value ? { value } : {}) };
}

const MODEL_SOURCE_PROPERTY = {
  type: 'string',
  description: 'Optional: answer about model text passed inline (a .neurarch.json document or PyTorch source).',
};

function withModelSource(schema: Record<string, unknown>): Record<string, unknown> {
  const props = (schema.properties ?? {}) as Record<string, unknown>;
  if (MODEL_SOURCE_PARAM in props) return schema;
  return { ...schema, properties: { ...props, [MODEL_SOURCE_PARAM]: MODEL_SOURCE_PROPERTY } };
}
