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
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ReadResourceRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import type { ModelArchitecture } from './lib/types.js';
import { resolve } from 'node:path';
import { TOOLS, type ToolContext, type ToolDef } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';
import { HF_TOOLS } from './extraTools.js';
import { listedTools, resolveToolCall, isWriteTool, MODEL_PATH_PARAM, type ToolSetName } from './cli.js';
import { reportingEnabled, buildCorpusReport, sendCorpusReport } from './lib/corpusReport.js';
import { loadModelCached } from './models.js';
import { listZoo, loadZooModel, modelFromText } from './sources.js';
import { renderMermaid } from './mermaid.js';
import { generatePyTorchCode } from './vendor/engine.bundle.mjs';
import { RULE_PROVENANCE } from './vendor/verifier.bundle.mjs';
import { PROMPTS, renderPrompt } from './prompts.js';

/**
 * The tools that grade a graph rather than describe one.
 *
 * A corpus row records "this shape got this verdict", so it is meaningful only
 * after something graded the shape. get_layer_info and the rest inspect; these
 * three judge.
 */
const GRADING_TOOLS = new Set(['validate_model', 'lint_model', 'check_design']);

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
  const tools = listedTools(writeEnabled, toolSet).map(t => ({
    ...t,
    inputSchema: isWriteTool(t.name) ? t.inputSchema : withModelSource(t.inputSchema),
  }));

  const server = new Server(
    { name: 'neurarch-mcp', version },
    { capabilities: { tools: {}, prompts: {}, resources: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools }));

  server.setRequestHandler(CallToolRequestSchema, async (req) => {
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
        model = getModel();
      }

      const base: ToolContext = wantsOtherModel && !/^(zoo|hf):/i.test(pathArg as string)
        ? { ...ctx, currentPath: resolve(pathArg as string) }
        : wantsOtherModel || wantsInline ? { ...ctx, currentPath: '' } : ctx;
      // Elicitation is offered only when the client said it can answer.
      const canAsk = !!server.getClientCapabilities()?.elicitation;
      const callCtx: ToolContext = canAsk ? { ...base, elicit: (q) => elicit(server, q) } : base;
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
      if (GRADING_TOOLS.has(tool.name) && reportingEnabled()) {
        sendCorpusReport(buildCorpusReport(model));
      }
      // Both shapes, deliberately. `structuredContent` (spec 2025-06-18) is what
      // a modern client should read; the JSON text stays because clients that
      // predate it would otherwise see an empty result, and because a human
      // reading a transcript can still see what came back.
      //
      // No `outputSchema` is declared. A client validates structuredContent
      // against a declared schema and fails the call on a mismatch, so
      // declaring 30 hand-written schemas would convert every drift between
      // schema and handler into a broken tool. Sending the data without the
      // contract is the honest half of the feature; the contract can come when
      // the schemas are generated from the handlers rather than typed twice.
      const structured =
        result !== null && typeof result === 'object' && !Array.isArray(result)
          ? (result as Record<string, unknown>)
          : undefined;
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(structured ? { structuredContent: structured } : {}),
      };
    } catch (e) {
      return {
        isError: true,
        content: [{ type: 'text', text: `${tool.name} failed: ${(e as Error).message}` }],
      };
    }
  });

  // ── prompts ────────────────────────────────────────────────────────────────
  server.setRequestHandler(ListPromptsRequestSchema, async () => ({
    prompts: PROMPTS.map(p => ({ name: p.name, title: p.title, description: p.description, arguments: p.arguments })),
  }));

  server.setRequestHandler(GetPromptRequestSchema, async (req) => {
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
    { uri: 'neurarch://docs', name: 'Tool documentation', description: 'The long-form contract of every tool, including the ones --tools=core does not list. One read instead of 7k tokens on every turn.', mimeType: 'text/markdown' },
  ];

  server.setRequestHandler(ListResourcesRequestSchema, async () => ({ resources: STATIC_RESOURCES }));

  server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => ({
    resourceTemplates: [
      { uriTemplate: 'neurarch://zoo/{id}', name: 'Reference architecture', description: 'One bundled architecture as .neurarch.json; ids come from neurarch://zoo.', mimeType: 'application/json' },
      { uriTemplate: 'neurarch://docs/{tool}', name: 'Tool documentation', description: 'The full description and input schema of one tool.', mimeType: 'text/markdown' },
    ],
  }));

  server.setRequestHandler(ReadResourceRequestSchema, async (req) => {
    const uri = req.params.uri;
    const text = async (): Promise<{ mimeType: string; text: string }> => {
      if (uri === 'neurarch://model') return { mimeType: 'application/json', text: JSON.stringify(getModel(), null, 2) };
      if (uri === 'neurarch://model/mermaid') return { mimeType: 'text/plain', text: renderMermaid(getModel()) };
      if (uri === 'neurarch://model/pytorch') return { mimeType: 'text/x-python', text: generatePyTorchCode(getModel()) };
      if (uri === 'neurarch://zoo') return { mimeType: 'application/json', text: JSON.stringify(await listZoo(), null, 2) };
      if (uri === 'neurarch://rules') return { mimeType: 'application/json', text: JSON.stringify(RULE_PROVENANCE, null, 2) };
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

  return server;
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

/**
 * One elicitation, shaped as a single-choice form. `value` comes back only on
 * accept; decline and cancel are answers too and are passed through, because
 * "the user did not want to decide" is information the tool should report.
 */
async function elicit(
  server: Server,
  q: { message: string; options: Array<{ value: string; label: string; hint?: string }> },
): Promise<{ action: 'accept' | 'decline' | 'cancel'; value?: string }> {
  const res = await server.elicitInput({
    message: q.message,
    requestedSchema: {
      type: 'object',
      properties: {
        choice: {
          type: 'string',
          title: 'Your choice',
          enum: q.options.map(o => o.value),
          enumNames: q.options.map(o => (o.hint ? `${o.label}: ${o.hint}` : o.label)),
        },
      },
      required: ['choice'],
    },
  });
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
