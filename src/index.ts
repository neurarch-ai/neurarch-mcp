import { resolve } from 'node:path';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { loadModelFile } from './loader.js';
import { TOOLS, type ToolContext, type ToolDef } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';

const HELP = `neurarch-mcp — Model Context Protocol server for a Neurarch model file.

Usage:
  npx neurarch-mcp <path-to-model.neurarch.json> [--write]

Flags:
  --write   Enable mutation tools (add_layer, modify_layer, add_connection,
            save_model). Default is read-only — writes are opt-in so accidental
            mutations cannot clobber the file you are editing in the app.

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
  const argv = process.argv.slice(2);
  const writeIdx = argv.indexOf('--write');
  const writeEnabled = writeIdx !== -1;
  if (writeEnabled) argv.splice(writeIdx, 1);

  const arg = argv[0];
  if (!arg || arg === '-h' || arg === '--help') {
    process.stdout.write(HELP);
    process.exit(arg ? 0 : 1);
  }

  const modelPath = resolve(arg);

  let model;
  try {
    model = await loadModelFile(modelPath);
  } catch (e) {
    process.stderr.write(`neurarch-mcp: ${(e as Error).message}\n`);
    process.exit(1);
  }

  const tools: ToolDef[] = writeEnabled ? [...TOOLS, ...WRITE_TOOLS] : TOOLS;
  const ctx: ToolContext = { modelPath };

  if (writeEnabled) {
    process.stderr.write(
      `neurarch-mcp: write mode enabled. ${WRITE_TOOLS.length} mutation tools exposed.\n`,
    );
  }

  const server = new Server(
    { name: 'neurarch-mcp', version: '0.2.0' },
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
    const tool = tools.find(t => t.name === req.params.name);
    if (!tool) {
      const isWriteTool = WRITE_TOOLS.some(t => t.name === req.params.name);
      const hint = isWriteTool && !writeEnabled
        ? ' Restart the MCP server with --write to enable mutation tools.'
        : '';
      return {
        isError: true,
        content: [{ type: 'text', text: `Unknown tool: ${req.params.name}.${hint}` }],
      };
    }
    try {
      const result = await tool.handler(req.params.arguments ?? {}, model, ctx);
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
