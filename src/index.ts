import { resolve } from 'node:path';
import { unwatchFile, watchFile } from 'node:fs';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { loadModelFile } from './loader.js';
import type { ModelArchitecture } from './lib/types.js';
import { TOOLS, type ToolContext } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';
import { parseFlags, selectTools, resolveToolCall } from './cli.js';
import pkg from '../package.json';

const VERSION: string = pkg.version;

const HELP = `neurarch-mcp — Model Context Protocol server for a Neurarch model file.

Usage:
  npx neurarch-mcp <path-to-model.neurarch.json> [--write] [--watch]

Flags:
  --version Print the neurarch-mcp version and exit (alias: -v).
  --write   Enable mutation tools (add_layer, modify_layer, add_connection,
            delete_layer, delete_connection, save_model). Default is read-only
            so accidental mutations cannot clobber the file you are editing in
            the app.
  --watch   Reload the model file from disk when it changes. Useful when you
            are editing in the Neurarch app and want the MCP to track your
            saves without restarting the server. Incompatible with in-memory
            edits from --write that have not been persisted: an external save
            will overwrite them.

Read tools (always available):
${TOOLS.map(t => `  - ${t.name}: ${t.description.split('.')[0]}`).join('\n')}

Write tools (only when --write is set):
${WRITE_TOOLS.map(t => `  - ${t.name}: ${t.description.split('.')[0]}`).join('\n')}

Example Claude Code config (~/.claude/mcp_servers.json):
{
  "mcpServers": {
    "neurarch": {
      "command": "npx",
      "args": ["-y", "neurarch-mcp", "/abs/path/to/model.neurarch.json", "--write"]
    }
  }
}
`;

async function main(): Promise<void> {
  const { versionRequested, helpRequested, writeEnabled, watchEnabled, modelArg } =
    parseFlags(process.argv.slice(2));

  if (versionRequested) {
    process.stdout.write(`neurarch-mcp ${VERSION}\n`);
    process.exit(0);
  }
  if (helpRequested || !modelArg) {
    process.stdout.write(HELP);
    process.exit(helpRequested ? 0 : 1);
  }

  const modelPath = resolve(modelArg);

  let currentModel: ModelArchitecture;
  try {
    currentModel = await loadModelFile(modelPath);
  } catch (e) {
    process.stderr.write(`neurarch-mcp: ${(e as Error).message}\n`);
    process.exit(1);
  }

  const tools = selectTools(writeEnabled);
  const ctx: ToolContext = { modelPath };

  if (writeEnabled) {
    process.stderr.write(
      `neurarch-mcp: write mode enabled. ${WRITE_TOOLS.length} mutation tools exposed.\n`,
    );
  }

  if (watchEnabled) {
    process.stderr.write(`neurarch-mcp: watch mode enabled. Polling ${modelPath} for changes.\n`);
    let reloading = false;
    watchFile(modelPath, { interval: 1000 }, async (curr, prev) => {
      if (curr.mtimeMs === prev.mtimeMs) return;
      if (reloading) return;
      reloading = true;
      try {
        const next = await loadModelFile(modelPath);
        currentModel = next;
        process.stderr.write(
          `neurarch-mcp: reloaded model (${next.components.length} layers, ${next.connections.length} connections).\n`,
        );
      } catch (e) {
        process.stderr.write(`neurarch-mcp: reload failed, keeping previous model: ${(e as Error).message}\n`);
      } finally {
        reloading = false;
      }
    });
    const stop = () => { unwatchFile(modelPath); process.exit(0); };
    process.once('SIGINT', stop);
    process.once('SIGTERM', stop);
  }

  const server = new Server(
    { name: 'neurarch-mcp', version: VERSION },
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
      const result = await tool.handler(req.params.arguments ?? {}, currentModel, ctx);
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

  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((e: unknown) => {
  process.stderr.write(`neurarch-mcp: fatal: ${(e as Error).stack ?? String(e)}\n`);
  process.exit(1);
});
