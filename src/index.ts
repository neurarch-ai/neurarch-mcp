import { resolve } from 'node:path';
import { unwatchFile, watchFile } from 'node:fs';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { loadModelFile } from './loader.js';
import type { ModelArchitecture } from './lib/types.js';
import { TOOLS, type ToolContext } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';
import { parseFlags } from './cli.js';
import { createMcpServer } from './server.js';
import {
  startHttpServer,
  isLoopbackHost,
  DEFAULT_HTTP_HOST,
  DEFAULT_HTTP_PORT,
} from './http.js';
import pkg from '../package.json';

const VERSION: string = pkg.version;

const HELP = `neurarch-mcp — Model Context Protocol server for a Neurarch model file.

Usage:
  npx neurarch-mcp <path-to-model.neurarch.json> [--write] [--watch] [--http[=PORT]]

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
  --http[=PORT]
            Serve over Streamable HTTP instead of stdio (default port 8787), so
            a remote or hosted agent can reach a model running on your machine
            (e.g. behind a Cloudflare/Tailscale tunnel). POST JSON-RPC to /mcp;
            GET /health for a liveness probe.
  --host=ADDR
            Bind address for --http. Defaults to 127.0.0.1 (loopback only). Use
            0.0.0.0 to expose it to a tunnel — but then a token is required if
            --write is on (see below).

Environment:
  NEURARCH_MCP_TOKEN  When set, --http requires 'Authorization: Bearer <token>'
                      on every request (constant-time checked). Required before
                      --write may bind to a non-loopback host.

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

Remote example (serve locally, drive from a cloud agent over a tunnel):
  NEURARCH_MCP_TOKEN=$(openssl rand -hex 16) \\
    npx neurarch-mcp /abs/path/to/model.neurarch.json --write --http --host=0.0.0.0
  # then point a tunnel (cloudflared / tailscale funnel) at 8787 and connect the
  # agent to https://<tunnel>/mcp with the same bearer token.
`;

async function main(): Promise<void> {
  const {
    versionRequested, helpRequested, writeEnabled, watchEnabled,
    httpEnabled, httpPort, httpHost, modelArg,
  } = parseFlags(process.argv.slice(2));

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

  // Read the model through a getter so --watch reloads (which reassign
  // currentModel) and in-place --write edits are both always seen.
  const getModel = () => currentModel;

  if (httpEnabled) {
    const host = httpHost ?? DEFAULT_HTTP_HOST;
    const port = httpPort ?? DEFAULT_HTTP_PORT;
    const token = process.env.NEURARCH_MCP_TOKEN || undefined;
    // Exposing write tools to anything but loopback without a token would let
    // any reachable client mutate (and save over) the model file. Refuse it.
    if (!isLoopbackHost(host) && writeEnabled && !token) {
      process.stderr.write(
        'neurarch-mcp: refusing to expose --write on a non-loopback host without NEURARCH_MCP_TOKEN. ' +
        'Set a token, drop --write, or bind to 127.0.0.1.\n',
      );
      process.exit(1);
    }
    startHttpServer({ getModel, ctx, writeEnabled, version: VERSION, host, port, token });
    return;
  }

  const server = createMcpServer({ getModel, ctx, writeEnabled, version: VERSION });
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((e: unknown) => {
  process.stderr.write(`neurarch-mcp: fatal: ${(e as Error).stack ?? String(e)}\n`);
  process.exit(1);
});
