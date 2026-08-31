/**
 * The MCP server wiring, transport-agnostic. Both the stdio path (index.ts) and
 * the HTTP path (http.ts) build a server here and connect their own transport,
 * so the tool list, dispatch, and error shaping live in exactly one place.
 *
 * The model is read through a getter rather than captured by value: --watch
 * reassigns the current model when the file changes on disk, and every tool
 * call must see the latest one.
 */
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import type { ModelArchitecture } from './lib/types.js';
import { type ToolContext } from './tools.js';
import { listedTools, resolveToolCall, isWriteTool, MODEL_PATH_PARAM } from './cli.js';
import { reportingEnabled, buildCorpusReport, sendCorpusReport } from './lib/corpusReport.js';

/**
 * The tools that grade a graph rather than describe one.
 *
 * A corpus row records "this shape got this verdict", so it is meaningful only
 * after something graded the shape. get_layer_info and the rest inspect; these
 * three judge.
 */
const GRADING_TOOLS = new Set(['validate_model', 'lint_model', 'check_design']);
import { loadModelCached } from './models.js';

export interface McpServerOptions {
  /** Reads the current model. A getter so --watch reloads are always seen. */
  getModel: () => ModelArchitecture;
  ctx: ToolContext;
  writeEnabled: boolean;
  version: string;
}

/** Build a configured MCP Server (no transport attached yet). */
export function createMcpServer(opts: McpServerOptions): Server {
  const { getModel, ctx, writeEnabled, version } = opts;
  const tools = listedTools(writeEnabled);

  const server = new Server(
    { name: 'neurarch-mcp', version },
    { capabilities: { tools: {} } },
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
    // instead of only the one it was launched with. It is stripped before the
    // handler sees it: every schema is additionalProperties:false, and a tool
    // should not have to know it exists.
    const rawArgs = (req.params.arguments ?? {}) as Record<string, unknown>;
    const pathArg = rawArgs[MODEL_PATH_PARAM];
    const wantsOtherModel = typeof pathArg === 'string' && pathArg.trim() !== '';

    if (wantsOtherModel && isWriteTool(tool.name)) {
      // Refused rather than ignored. Honouring it would let an agent mutate,
      // and then save over, a file nobody pointed this server at; silently
      // dropping it would mutate a different model than the one the agent
      // believes it is editing. Both are worse than an error that says so.
      return {
        isError: true,
        content: [{
          type: 'text',
          text: `${tool.name} does not accept ${MODEL_PATH_PARAM}. Mutations always target the model file `
            + 'this server was started with. Start a second server pointed at the other file to edit it.',
        }],
      };
    }

    try {
      let model = getModel();
      let args: Record<string, unknown> = rawArgs;
      if (wantsOtherModel) {
        try {
          model = await loadModelCached(pathArg as string);
        } catch (e) {
          return {
            isError: true,
            content: [{ type: 'text', text: `${tool.name}: ${(e as Error).message}` }],
          };
        }
        const { [MODEL_PATH_PARAM]: _dropped, ...rest } = rawArgs;
        args = rest;
      }

      const result = await tool.handler(args, model, ctx);
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
      // declaring 24 hand-written schemas would convert every drift between
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

  return server;
}
