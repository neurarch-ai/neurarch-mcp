import { resolve } from 'node:path';
import { unwatchFile, watchFile } from 'node:fs';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { loadModelFile, sourceKindFor } from './loader.js';
import type { ModelArchitecture } from './lib/types.js';
import { TOOLS, type ToolContext } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';
import { parseFlags, CORE_TOOLS } from './cli.js';
import { createMcpServer } from './server.js';
import { setHfEnabled } from './sources.js';
import { EXTRA_TOOLS, HF_TOOLS } from './extraTools.js';
import { LEDGER_TOOLS } from './ledgerTools.js';
import { PROMPTS } from './prompts.js';
import { runLintCommand, runCheckCommand } from './commands.js';
import {
  startHttpServer,
  isLoopbackHost,
  DEFAULT_HTTP_HOST,
  DEFAULT_HTTP_PORT,
} from './http.js';
import pkg from '../package.json';

const VERSION: string = pkg.version;

const ZOO_COUNT = 81;

const HELP = `neurarch-mcp: Model Context Protocol server for a neural network model.

Usage:
  npx neurarch-mcp <model.py | model.neurarch.json | zoo:<id> | hf:<org/name>> [flags]
  npx neurarch-mcp lint  <model...> [--json]     print the design-rule findings, exit 1 on a block
  npx neurarch-mcp check <model>    [--json]     print the full verdict, exit 1 on a block
  npx neurarch-mcp --http [--host=0.0.0.0]       serve with no model; callers pass model_path or model_source
  npx neurarch-mcp --hosted                      the same over stdio (a container with nothing mounted)

Model:
  A PyTorch .py file (parsed into a graph: layers, params and wiring exact,
  shapes unknown), a .neurarch.json saved from the app or written by
  neurarch-trace (everything), zoo:<id> for one of the ${ZOO_COUNT} bundled reference
  architectures (list_architectures), or hf:<org/name> for a Hugging Face repo
  (needs --hf). --write is refused on a .py file because the graph was derived
  from it and cannot be written back; use export_pytorch to emit new source.

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
            0.0.0.0 to expose it to a tunnel, but then a token is required if
            --write is on (see below).
  --hf      Allow hf:<org/name> model refs and list load_hf_model. This is the
            one switch that lets a tool open a socket on your behalf
            (huggingface.co, to read config.json). HF_TOKEN is sent for gated
            repos. Results are cached for a day under ~/.cache/neurarch-mcp.
  --tools=full
            Advertise every read tool. The default is the core set
            (${CORE_TOOLS.length}: ${CORE_TOOLS.join(', ')});
            every tool stays callable by name either way, and the full
            contract of each is one read away at neurarch://docs/<tool>.

Environment:
  NEURARCH_MCP_TOKEN  When set, --http requires 'Authorization: Bearer <token>'
                      on every request (constant-time checked). Required before
                      --write may bind to a non-loopback host.
  NEURARCH_REPORT     Set to 1 to opt in to anonymous corpus reporting: each
                      validate_model, lint_model, check_design or plan call
                      shares its structural fingerprint, layer-type histogram,
                      edge count, and (rule, severity) pairs. Never the graph,
                      params, names, or any identity. Off by default; errors
                      never affect the tool call.
  NEURARCH_API_KEY    Your organisation's key (nrk_...). Required by the
                      history tool, which reads what happened the last time a
                      structure trained inside your organisation; optional for
                      the plan tool, where it adds that same line to the card.
                      Read nowhere else, and sent nowhere but NEURARCH_API.
  NEURARCH_API        Override the API host (default https://www.neurarch.com).
                      The same variable neurarch-trace uses.

  Two tools reach the network when you call them, and they say so in their
  descriptions: plan POSTs the graph to /api/v1/plan for the same card the CI
  bot posts, and history sends an 8-character structural fingerprint (never the
  graph) to /api/v1/history. With --hf off, NEURARCH_REPORT unset and neither
  of those called, this server makes no network calls at all: every other tool,
  check_design included, runs locally against a vendored verifier.

Read tools (always available):
${TOOLS.filter(t => !EXTRA_TOOLS.includes(t) && !LEDGER_TOOLS.includes(t)).map(t => `  - ${t.name}: ${t.description.split('.')[0]}`).join('\n')}
${EXTRA_TOOLS.map(t => `  - ${t.name}: ${t.description.split('.')[0]}`).join('\n')}

Ledger tools (reach neurarch.com when called; see NEURARCH_API_KEY below):
${LEDGER_TOOLS.map(t => `  - ${t.name}: ${t.description.split('.')[0]}`).join('\n')}

Network tools (only with --hf):
${HF_TOOLS.map(t => `  - ${t.name}: ${t.description.split('.')[0]}`).join('\n')}

Write tools (only when --write is set):
${WRITE_TOOLS.map(t => `  - ${t.name}: ${t.description.split('.')[0]}`).join('\n')}

Prompts (slash commands in Claude Desktop, Cursor, VS Code):
${PROMPTS.map(p => `  - ${p.name}: ${p.description.split('.')[0]}`).join('\n')}

Resources: neurarch://model, neurarch://model/mermaid, neurarch://model/pytorch,
  neurarch://zoo, neurarch://zoo/{id}, neurarch://rules

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
    httpEnabled, httpPort, httpHost, modelArg, hfEnabled, toolSet, command, json, positional, hostedRequested,
  } = parseFlags(process.argv.slice(2));

  if (versionRequested) {
    process.stdout.write(`neurarch-mcp ${VERSION}\n`);
    process.exit(0);
  }
  setHfEnabled(hfEnabled);

  if (command) {
    const io = { out: (t: string) => process.stdout.write(t), err: (t: string) => process.stderr.write(t) };
    const code = command === 'lint'
      ? await runLintCommand(positional, json, io)
      : await runCheckCommand(positional, json, io);
    process.exit(code);
  }

  // A server with no model is the norm over HTTP, where every call can carry
  // model_path or model_source. On stdio the client usually shares our disk,
  // so a path up front costs nothing and catches typos at startup; the
  // exception is a container with nothing mounted (Docker MCP Toolkit runs
  // stdio inside the image), which says so with --hosted.
  const hosted = !modelArg && (httpEnabled || hostedRequested);
  if (helpRequested || (!modelArg && !hosted)) {
    process.stdout.write(HELP);
    process.exit(helpRequested ? 0 : 1);
  }

  // stdio IS the protocol channel. Anything in the vendored engine that logs
  // to stdout (the code generator warns about layer types it has no template
  // for) would corrupt a JSON-RPC frame, so route it to stderr for the life of
  // the process. Nothing in this server writes to stdout on purpose.
  if (!httpEnabled) {
    console.log = (...a: unknown[]) => process.stderr.write(a.map(String).join(' ') + '\n');
    console.info = console.log;
  }

  const modelPath = modelArg ? resolve(modelArg) : '';

  let currentModel: ModelArchitecture | null = null;
  if (modelArg) {
    try {
      currentModel = /^(zoo|hf):/i.test(modelArg)
        ? await (await import('./models.js')).loadModelCached(modelArg)
        : await loadModelFile(modelPath);
    } catch (e) {
      process.stderr.write(`neurarch-mcp: ${(e as Error).message}\n`);
      process.exit(1);
    }
  }

  const ctx: ToolContext = { modelPath, writeEnabled };

  // Mutations against a graph parsed out of Python have nowhere to go: writing
  // them back means regenerating source, which this server does not do, and
  // save_model would put JSON where the .py was. Refused at startup rather than
  // at the first edit, so the user finds out before an agent has built a plan
  // on top of tools that were never going to work.
  if (writeEnabled && (!modelArg || /^(zoo|hf):/i.test(modelArg))) {
    process.stderr.write(
      'neurarch-mcp: --write needs a .neurarch.json file to write to. zoo:/hf: references and the hosted mode '
      + 'have no file behind them; use load_architecture / load_hf_model with save_to first, then start on that file.\n',
    );
    process.exit(1);
  }
  if (writeEnabled && sourceKindFor(modelPath) === 'pytorch-source') {
    process.stderr.write(
      'neurarch-mcp: refusing --write on a PyTorch source file. The graph is derived from '
      + `${modelPath}, so edits cannot be written back to it. Export the model as .neurarch.json `
      + 'from the Neurarch app and point --write at that, or drop --write to use the read tools.\n',
    );
    process.exit(1);
  }

  if (writeEnabled) {
    process.stderr.write(
      `neurarch-mcp: write mode enabled. ${WRITE_TOOLS.length} mutation tools exposed.\n`,
    );
  }

  if (hfEnabled) {
    process.stderr.write('neurarch-mcp: --hf on. load_hf_model and hf:<org/name> refs may contact huggingface.co.\n');
  }

  if (watchEnabled && currentModel && !/^(zoo|hf):/i.test(modelArg ?? '')) {
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
  // currentModel) and in-place --write edits are both always seen. In hosted
  // mode there is none, and the error is the answer for a call that named no
  // model_path or model_source.
  const getModel = (): ModelArchitecture => {
    if (!currentModel) {
      throw new Error(
        'This server was started without a model. Pass model_path (a path, zoo:<id> or hf:<org/name>) '
        + 'or model_source (inline .neurarch.json or PyTorch source) with the call.',
      );
    }
    return currentModel;
  };

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
    startHttpServer({ getModel, ctx, writeEnabled, version: VERSION, host, port, token, toolSet, hosted });
    return;
  }

  const server = createMcpServer({ getModel, ctx, writeEnabled, version: VERSION, toolSet });
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((e: unknown) => {
  process.stderr.write(`neurarch-mcp: fatal: ${(e as Error).stack ?? String(e)}\n`);
  process.exit(1);
});
