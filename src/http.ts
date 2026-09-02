/**
 * Optional HTTP transport (`--http`). Serves the same MCP server over Streamable
 * HTTP instead of stdio, so a remote / hosted agent can reach a model running on
 * your machine — e.g. run the server locally, expose it through a Cloudflare or
 * Tailscale tunnel, and drive it from a phone or a cloud agent.
 *
 * Stateful sessions: a client POSTs `initialize`, gets an `Mcp-Session-Id`
 * header back, and carries it on every later request — the standard Streamable
 * HTTP flow every real MCP client (Claude Code, Cursor, ...) speaks. GET /mcp
 * opens the server->client SSE stream; DELETE /mcp tears a session down.
 * Zero runtime deps beyond the MCP SDK — Node's built-in http.
 *
 * Security posture (this exposes tools, including --write mutations, over a
 * socket):
 *   - Binds to 127.0.0.1 by default. Without a token, the Host header is checked
 *     against a loopback allowlist (DNS-rebinding protection) and no CORS
 *     headers are sent, so a browser cannot drive it cross-origin.
 *   - Set NEURARCH_MCP_TOKEN to require `Authorization: Bearer <token>` on every
 *     request (constant-time checked). Required (enforced in index.ts) before
 *     write tools may bind to a non-loopback host.
 */
import { createServer as createHttpServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { randomUUID, timingSafeEqual } from 'node:crypto';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { isInitializeRequest } from '@modelcontextprotocol/sdk/types.js';
import type { ModelArchitecture } from './lib/types.js';
import { type ToolContext } from './tools.js';
import { createMcpServer } from './server.js';
import type { ToolSetName } from './cli.js';

export const DEFAULT_HTTP_PORT = 8787;
export const DEFAULT_HTTP_HOST = '127.0.0.1';

/** Whether a bind address is loopback-only (safe to expose without a token). */
export function isLoopbackHost(host: string): boolean {
  return (
    host === '127.0.0.1' ||
    host === 'localhost' ||
    host === '::1' ||
    host === '::ffff:127.0.0.1'
  );
}

function bearerOk(req: IncomingMessage, token: string): boolean {
  const auth = req.headers['authorization'];
  if (typeof auth !== 'string') return false;
  const a = Buffer.from(auth);
  const b = Buffer.from(`Bearer ${token}`);
  return a.length === b.length && timingSafeEqual(a, b);
}

async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  if (chunks.length === 0) return undefined;
  return JSON.parse(Buffer.concat(chunks).toString('utf-8'));
}

function jsonRpcError(res: ServerResponse, status: number, code: number, message: string, headers: Record<string, string> = {}): void {
  res.writeHead(status, { 'content-type': 'application/json', ...headers });
  res.end(JSON.stringify({ jsonrpc: '2.0', error: { code, message }, id: null }));
}

function sessionHeader(req: IncomingMessage): string | undefined {
  const v = req.headers['mcp-session-id'];
  return Array.isArray(v) ? v[0] : v;
}

export interface HttpServerOptions {
  getModel: () => ModelArchitecture;
  toolSet?: ToolSetName;
  /** Started with no model: /health reports that instead of a model name. */
  hosted?: boolean;
  ctx: ToolContext;
  writeEnabled: boolean;
  version: string;
  host: string;
  port: number;
  /** When set, every request must carry `Authorization: Bearer <token>`. */
  token?: string;
}

let liveServer: import('node:http').Server | null = null;

/** Close the listener and every live session. A test seam; production exits by signal. */
export function stopHttpServer(): Promise<void> {
  return new Promise((res) => {
    const s = liveServer;
    liveServer = null;
    if (!s) return res();
    s.close(() => res());
    s.closeAllConnections?.();
  });
}

/** Start the Streamable HTTP transport and register signal handlers. */
export function startHttpServer(opts: HttpServerOptions): void {
  const { getModel, ctx, writeEnabled, version, host, port, token, toolSet, hosted } = opts;
  // A bearer token defeats browser CSRF (the page can't read the secret to
  // forge the header), so host-checking is redundant there and would only break
  // tunnels whose Host header is the tunnel domain. Without a token we lean on
  // the loopback Host allowlist instead.
  const protectDns = !token;
  // NEURARCH_MCP_PUBLIC_HOST opts a tokenless server into serving a public
  // hostname (comma-separated; with or without a port). This is the deliberate
  // "anyone may read" posture for a hosted deployment whose clients cannot
  // send a header (claude.ai custom connectors speak OAuth or nothing): the
  // read tools are open, --write stays refused on non-loopback without a
  // token, and the DNS-rebinding check still pins the exact hostnames named.
  const publicHosts = (process.env.NEURARCH_MCP_PUBLIC_HOST ?? '')
    .split(',').map(h => h.trim()).filter(Boolean)
    .flatMap(h => (h.includes(':') ? [h] : [h, `${h}:443`, `${h}:${port}`]));
  const allowedHosts = [`${host}:${port}`, `127.0.0.1:${port}`, `localhost:${port}`, ...publicHosts];

  // One transport per live session, keyed by the id the SDK mints on initialize.
  const sessions = new Map<string, StreamableHTTPServerTransport>();

  async function handleMcp(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const sid = sessionHeader(req);

    // GET opens the SSE stream, DELETE tears the session down — both need an
    // existing session; the transport does the rest.
    if (req.method === 'GET' || req.method === 'DELETE') {
      const transport = sid ? sessions.get(sid) : undefined;
      if (!transport) {
        jsonRpcError(res, 400, -32000, 'Missing or unknown Mcp-Session-Id.');
        return;
      }
      await transport.handleRequest(req, res);
      return;
    }

    if (req.method !== 'POST') {
      jsonRpcError(res, 405, -32000, 'Method not allowed.', { allow: 'POST, GET, DELETE' });
      return;
    }

    let body: unknown;
    try {
      body = await readJsonBody(req);
    } catch {
      jsonRpcError(res, 400, -32700, 'Parse error: request body is not valid JSON.');
      return;
    }

    // Reuse an established session.
    if (sid) {
      const transport = sessions.get(sid);
      if (!transport) {
        jsonRpcError(res, 404, -32001, 'Unknown session. Re-initialize.');
        return;
      }
      await transport.handleRequest(req, res, body);
      return;
    }

    // No session yet: only an initialize request may open one.
    if (!isInitializeRequest(body)) {
      jsonRpcError(res, 400, -32000, 'No session: send an initialize request first.');
      return;
    }

    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
      enableDnsRebindingProtection: protectDns,
      ...(protectDns ? { allowedHosts } : {}),
      onsessioninitialized: (id: string) => { sessions.set(id, transport); },
      onsessionclosed: (id: string) => { sessions.delete(id); },
    });
    transport.onclose = () => {
      if (transport.sessionId) sessions.delete(transport.sessionId);
    };
    const server = createMcpServer({ getModel, ctx, writeEnabled, version, toolSet });
    await server.connect(transport);
    await transport.handleRequest(req, res, body);
  }

  const httpServer = createHttpServer((req, res) => {
    const url = (req.url ?? '/').split('?')[0];

    // A person in a browser, not an MCP client: no session header and no SSE
    // accept. Tell them what this is instead of a bare JSON-RPC error.
    if (req.method === 'GET' && url === '/mcp' && !sessionHeader(req) && !String(req.headers.accept ?? '').includes('text/event-stream')) {
      res.writeHead(200, { 'content-type': 'text/plain; charset=utf-8' });
      res.end(
        `neurarch-mcp ${version}: a Model Context Protocol server, not a web page.\n\n`
        + 'Connect an MCP client to this URL (Streamable HTTP). Claude Code:\n'
        + `  claude mcp add --transport http neurarch <this URL>${token ? " --header \"Authorization: Bearer <token>\"" : ''}\n\n`
        + 'Liveness: GET /health. Docs: https://www.neurarch.com/mcp\n',
      );
      return;
    }

    if (req.method === 'GET' && url === '/health') {
      res.writeHead(200, { 'content-type': 'application/json' });
      if (hosted) {
        res.end(JSON.stringify({ ok: true, hosted: true, version, write: writeEnabled }));
        return;
      }
      const m = getModel();
      res.end(JSON.stringify({ ok: true, name: m.name, layers: m.components.length, write: writeEnabled }));
      return;
    }

    if (url !== '/mcp') {
      res.writeHead(404, { 'content-type': 'text/plain' });
      res.end('neurarch-mcp: POST JSON-RPC to /mcp (or GET /health).\n');
      return;
    }

    if (token && !bearerOk(req, token)) {
      jsonRpcError(res, 401, -32001, 'Unauthorized: set Authorization: Bearer <NEURARCH_MCP_TOKEN>.', { 'www-authenticate': 'Bearer' });
      return;
    }

    handleMcp(req, res).catch((e) => {
      if (!res.headersSent) jsonRpcError(res, 500, -32603, (e as Error).message);
    });
  });

  liveServer = httpServer;

  httpServer.listen(port, host, () => {
    const auth = token
      ? ' (bearer auth required)'
      : isLoopbackHost(host)
        ? ' (loopback only, no token)'
        : ' (WARNING: no token on a non-loopback host)';
    process.stderr.write(`neurarch-mcp: HTTP transport on http://${host}:${port}/mcp${auth}\n`);
  });

  const stop = () => { httpServer.close(); process.exit(0); };
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);
}
