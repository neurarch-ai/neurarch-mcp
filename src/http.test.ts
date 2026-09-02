/**
 * The HTTP transport's edges: the public-host opt-in, the browser answer on
 * /mcp, and the Host allowlist that stays closed without it. Started on an
 * ephemeral port with a raw http client so the Host header can lie, which is
 * exactly what the DNS-rebinding check exists to catch.
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { request as httpRequest } from 'node:http';
import { startHttpServer, stopHttpServer } from './http.js';
import { makeModel } from './test/fixtures.js';

function req(port: number, opts: { path: string; method?: string; headers?: Record<string, string>; body?: string }): Promise<{ status: number; body: string }> {
  return new Promise((res, rej) => {
    const r = httpRequest({ host: '127.0.0.1', port, path: opts.path, method: opts.method ?? 'GET', headers: opts.headers }, (resp) => {
      let body = '';
      resp.on('data', d => { body += d; });
      resp.on('end', () => res({ status: resp.statusCode ?? 0, body }));
    });
    r.on('error', rej);
    if (opts.body) r.write(opts.body);
    r.end();
  });
}

const INIT = JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '0' } } });
const MCP_HEADERS = { 'content-type': 'application/json', accept: 'application/json, text/event-stream' };

describe('public host opt-in', () => {
  const PORT = 18931;
  beforeAll(() => {
    process.env.NEURARCH_MCP_PUBLIC_HOST = 'mcp.example.dev';
    startHttpServer({ getModel: () => makeModel(), ctx: { modelPath: '' }, writeEnabled: false, version: '0.0.0-test', host: '127.0.0.1', port: PORT });
  });
  afterAll(async () => { delete process.env.NEURARCH_MCP_PUBLIC_HOST; await stopHttpServer(); });

  it('serves the named public host without a token, and still refuses others', async () => {
    const ok = await req(PORT, { path: '/mcp', method: 'POST', headers: { ...MCP_HEADERS, host: 'mcp.example.dev' }, body: INIT });
    expect(ok.status).toBe(200);
    expect(ok.body).toMatch(/serverInfo/);
    const rebind = await req(PORT, { path: '/mcp', method: 'POST', headers: { ...MCP_HEADERS, host: 'evil.attacker.dev' }, body: INIT });
    expect(rebind.status).toBeGreaterThanOrEqual(400);
  });

  it('answers a plain browser GET on /mcp with words, not a JSON-RPC error', async () => {
    const r = await req(PORT, { path: '/mcp', headers: { host: 'mcp.example.dev', accept: 'text/html' } });
    expect(r.status).toBe(200);
    expect(r.body).toMatch(/Model Context Protocol server, not a web page/);
    expect(r.body).toMatch(/claude mcp add/);
  });

  it('health stays open', async () => {
    const r = await req(PORT, { path: '/health', headers: { host: 'mcp.example.dev' } });
    expect(JSON.parse(r.body).ok).toBe(true);
  });
});
