# On the v2 TypeScript SDK (spec 2026-07-28)

Status: **done.** The server runs on `@modelcontextprotocol/server` 2.3.1 with
`@modelcontextprotocol/node` 2.1.1 for the HTTP adapter (2.1.1 declares
`@modelcontextprotocol/server ^2.3.0` as its peer; there is no 2.3.x of the
adapter and none is needed). Tests use `@modelcontextprotocol/client` 2.3.1.
v1 (`@modelcontextprotocol/sdk`) implements the spec only up to 2025-11-25.

## What changed, and where

| Where | v1 | v2 |
|---|---|---|
| `src/server.ts` handlers | `setRequestHandler(CallToolRequestSchema, (req, extra) => ...)` | `setRequestHandler('tools/call', (req, ctx) => ...)`, same for the other seven |
| cancellation and progress | `extra.signal`, `extra.sendNotification` | `ctx.mcpReq.signal`, `ctx.mcpReq.notify`; progress token read from `ctx.mcpReq._meta` |
| elicitation | `server.elicitInput` mid-call | 2025 era: `ctx.mcpReq.elicitInput` mid-call, unchanged on the wire. 2026-07-28: the call returns `inputRequired.elicit(...)` and the client retries with the answer |
| stdio (`src/index.ts`) | `new StdioServerTransport()` + `connect` | `serveStdio(factory)`, which negotiates the era per connection |
| HTTP (`src/http.ts`) | one session transport per `Mcp-Session-Id` | routed by `isLegacyRequest`: 2026-07-28 to `createMcpHandler(factory, { legacy: 'reject' })`, one instance per request; 2025 era to the session flow it had, on `NodeStreamableHTTPServerTransport` |
| tests | `client.setRequestHandler(ElicitRequestSchema, ...)`, `callTool(params, undefined, opts)` | `'elicitation/create'`, `callTool(params, opts)` |

## Three decisions worth knowing before changing this

1. **2025 HTTP keeps sessions.** The SDK's default for 2025 clients is a
   stateless per-request instance. That instance never sees the initialize in
   which the client declared `elicitation`, so `check_design { ask_user }`
   would silently stop asking over HTTP. `src/eras.test.ts` caught exactly
   that. The session path is the documented way to keep a sessionful 2025
   deployment behind a strict modern handler.
2. **No `requestState`.** The only tool that asks is `check_design`, and
   before the question it only reads the graph. A retry recomputes the same
   decision, and the answer is keyed by a hash of the question text, so there
   is nothing from round one to carry and nothing a client could forge.
3. **Host and token checks stay ours.** `createMcpHandler` validates neither.
   `http.ts` checks the bearer token and, without one, the Host allowlist
   (including `NEURARCH_MCP_PUBLIC_HOST`) before either path sees the request.

## How it is checked

- `src/eras.test.ts` starts the real HTTP server and drives it with a 2025
  client and a client pinned to 2026-07-28: same tools, prompts and resources,
  same results for the same calls, elicitation accept and decline, progress
  and cancellation on 2026.
- `src/stdio.smoke.test.ts` spawns the built server over stdio.
- By hand before release: a v1 1.29 client and a v2 client (legacy and pinned
  2026-07-28) over stdio, and a v1 client over HTTP, each `initialize` (or
  discover) + `tools/list` + `tools/call lint_model`.

## Hosted instance

The fly deploy runs `--http`, so it gets the era routing above. Probe after a
deploy with both a 2025 `initialize` and a pinned 2026 client.
