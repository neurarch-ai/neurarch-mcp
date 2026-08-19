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
import type { ValidationReport } from './lib/validation.js';
import { type ToolContext } from './tools.js';
import { selectTools, resolveToolCall } from './cli.js';
import { reportingEnabled, buildCorpusReport, sendCorpusReport } from './lib/corpusReport.js';

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
  const tools = selectTools(writeEnabled);

  const server = new Server(
    { name: 'neurarch-mcp', version },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: tools.map(t => ({
      name: t.name,
      description: t.description,
      inputSchema: t.inputSchema,
    })),
  }));

  server.setRequestHandler(CallToolRequestSchema, async (req) => {
    const { tool, errorText } = resolveToolCall(req.params.name, writeEnabled);
    if (!tool) {
      return {
        isError: true,
        content: [{ type: 'text', text: errorText! }],
      };
    }
    try {
      const model = getModel();
      const result = await tool.handler(req.params.arguments ?? {}, model, ctx);
      // Opt-in corpus row (NEURARCH_REPORT=1), validate_model only. Fire and
      // forget: it can neither slow nor fail the tool call. Privacy scope in
      // lib/corpusReport.ts.
      if (tool.name === 'validate_model' && reportingEnabled()) {
        sendCorpusReport(buildCorpusReport(model, result as ValidationReport));
      }
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
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
