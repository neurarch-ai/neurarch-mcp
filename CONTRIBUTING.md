# Contributing to neurarch-mcp

Thanks for helping make AI coding agents model-aware. The highest-value contribution is a **new tool** that answers a question agents keep asking about a model graph.

## Setup

```bash
git clone https://github.com/neurarch-ai/neurarch-mcp
cd neurarch-mcp
npm install
npm run build                 # tsup -> dist/index.js
node dist/index.js --help     # confirm the bin runs
```

To try it end to end: export a model from the [Neurarch](https://neurarch.com) app (`File > Save JSON`), then point the built server at it from your MCP client (see the README Install section).

## How a tool is defined

Every tool is one `ToolDef` object (`src/tools.ts` for read tools, `src/writeTools.ts` for write tools):

```ts
interface ToolDef {
  name: string;
  description: string;                 // the agent reads this to decide when to call it
  inputSchema: Record<string, unknown>; // JSON Schema for the args
  handler: (args, model, ctx) => unknown | Promise<unknown>;
}
```

- Read tools live in the `TOOLS` array in `src/tools.ts`.
- Write tools live in `WRITE_TOOLS` in `src/writeTools.ts` and are only exposed when the server is started with `--write`.
- `src/index.ts` registers `ListTools` / `CallTool` and dispatches by `name`. Adding a tool to the right array is all the wiring you need.
- Pure model logic (params, FLOPs, impact, validation) lives in `src/lib/`. Reuse those estimators rather than recomputing.

## Add a read tool in 4 steps

1. **Define** a `ToolDef` in `src/tools.ts`. Write the `description` for the *agent*: say plainly when it should call the tool and what it gets back. Put non-trivial logic in a unit-testable helper under `src/lib/` (see `lib/describe.ts`, `lib/blocks.ts`).
2. **Register** it by adding the object to the `TOOLS` array.
3. **Test it.** Add a `*.test.ts` next to the source and a handler case to `src/tools.test.ts`, reusing `makeModel()` from `src/test/fixtures.ts`. Run `npm test`.
4. **Build and try it** (`npm run build`, then ask your agent a question that should trigger it). Add it to the README tool table.

A write tool is the same, in `WRITE_TOOLS`. Mutations must go through the helpers in `src/lib/writeOps.ts` so shape invalidation and id/name handling stay consistent, and must never write to disk except via `save_model`.

## Quality bar

- **Descriptions are the API.** The agent picks tools from the `description` alone. State the trigger and the return shape; skip implementation detail.
- **Read tools never mutate.** No disk writes, no model mutation.
- **Deterministic output.** Same model in, same JSON out. No timestamps or randomness in results.
- **Degrade gracefully.** Return `null` or an explanatory object, never throw, on a missing layer or empty graph.

## Pull requests

- Branch, commit, open a PR. One tool or one fix per PR.
- `npm run typecheck`, `npm run build`, and `npm test` must pass (CI runs all three on Node 20 and 22).
- Not sure if a tool belongs here or in the app? Open an issue and we will help scope it.

## Scope

This server reads a single saved `.neurarch.json` graph and exposes structure to MCP clients. The full rule engine and shape propagator live in the [Neurarch](https://neurarch.com) app; the CI lint lives in [neurarch-lint](https://github.com/neurarch-ai/neurarch-lint). Anything needing the live app session or the full propagator is out of scope for v1.
