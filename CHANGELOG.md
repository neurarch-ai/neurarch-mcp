# Changelog

All notable changes to `neurarch-mcp` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.10.0]

### Added
- **`check_design`** — the one tool here that returns a *verdict* rather than an
  inspection. Every other tool answers a question about the file in front of it
  (how many parameters, what does this layer touch, is there a cycle); this one
  answers the question an agent needs before it edits someone's architecture:
  is the design sound, what will a training run cost, which deployment target
  fits it and at what latency, and which decisions are still the human's.
  `validate_model` is the local structural subset of it.

  Requires `NEURARCH_API_KEY` (create one at neurarch.com under Settings →
  Developer API) and makes one network call. **With no key set it opens no
  socket and returns an explanation**, so the "no network calls by default"
  property is unchanged for anyone who has not configured one.

  Note that unlike corpus reporting, this **sends the model graph**, because a
  verdict about a graph cannot be computed without it. That difference is called
  out in its own README section rather than buried: if it is not acceptable for
  a given model, do not set a key and every other tool keeps working.

- `NEURARCH_API_URL` to point `check_design` at a self-hosted or test endpoint.

### Added in 0.9.1
- **Opt-in corpus reporting (`NEURARCH_REPORT=1`)** — when explicitly enabled,
  a `validate_model` call shares one anonymous structure+verdict row with the
  Neurarch corpus: structural fingerprint (8-char hash), layer-type histogram,
  edge count, and (rule id, severity) pairs. Never the graph, parameter
  values, layer names, or any identity; the payload shape cannot carry them.
  Off by default (no network calls at all without the flag); fire-and-forget
  with a 5s cap, so reporting can never slow or fail a tool call.

## [0.9.0]

### Added
- **Remote transport (`--http`)** — serve the same MCP tools over Streamable
  HTTP instead of stdio, so a hosted or phone-based agent can drive a model
  running on your machine (e.g. behind a Cloudflare/Tailscale tunnel). Stateful
  `Mcp-Session-Id` sessions, `GET /health` liveness probe, zero new runtime
  deps. `--host=ADDR` sets the bind address (default `127.0.0.1`).
  - **Auth & safety:** loopback-only by default with DNS-rebinding protection and
    no CORS; `NEURARCH_MCP_TOKEN` enables constant-time bearer auth and is
    required before `--write` may bind to a non-loopback host (the server
    refuses to start otherwise).

### Changed
- Server wiring extracted into a transport-agnostic `createMcpServer`
  (`src/server.ts`), shared by the stdio and HTTP paths; the model is now read
  through a getter so `--watch` reloads are always seen. `parseFlags` gains
  `--http` / `--host` parsing (unit-tested).

## [0.8.0]

### Added
- **`diff_models` tool** — structurally diff the current model against another
  `.neurarch.json` file: layers only in current / only in the other / modified
  (with field-level changes), plus added and removed connections. Layers are
  matched by name; duplicate names are reported as ambiguous rather than guessed.
  Use to review what an agent changed, or how the model differs from a checkpoint.
- **README "what it looks like"** transcript showing a real tool-grounded exchange.

### Changed
- CLI argument parsing and tool-dispatch gating extracted into a pure,
  unit-tested `src/cli.ts` (`parseFlags`, `selectTools`, `resolveToolCall`). The
  write-tool-gating hint ("restart with --write") is now covered by tests.

## [0.7.0]

### Added
- **`compare_layers` tool** — structural diff of two layers: same-type, the
  parameter-count delta, whether input/output shapes match, and exactly which
  param keys differ (only-in-a, only-in-b, changed values). Answers "are
  block_0 and block_1 identical?" before mirroring an edit or sharing an
  implementation.
- **`mermaid_diagram` renders groups as subgraphs** — grouped layers are wrapped
  in labelled Mermaid `subgraph` blocks so the diagram mirrors the model's
  blocks; ungrouped layers stay at the top level.

### Fixed
- `conv2d` parameter count now divides input channels by `groups`, matching the
  FLOPs estimator (grouped convs were overstated by `groups`×).
- `bidirectionalLSTM` params and FLOPs now honor `numLayers` (the unidirectional
  `lstm` case already did); `numLayers: 1` is unchanged.
- `lmHead` returns no new parameters when `weightTied`/`tied` is set, so
  weight-tied LLMs no longer inflate the headline parameter total.
- `mermaid_diagram` truncation keeps the topological head instead of an
  arbitrary file-order prefix, so the truncated diagram starts at the input.

## [0.6.0]

### Added
- **`find_layers` filters** — beyond type and name regex, the tool now filters
  by `scope` prefix (e.g. `encoder.layer.3` and anything nested under it) and by
  applied `augmentation` (e.g. `freeze`, `quantize_int8`), and can rank results
  by parameter count via `sortByParams`. Results now also carry each layer's
  scope, parameter count, and augmentations. Answers "which layers under X are
  frozen?" and "what are the biggest layers?" directly. Backward compatible.
- **`examples/resnet-mini.neurarch.json`** — a residual block with a skip/merge
  (`add`) node, giving the example set a branchier graph for the impact, path,
  and validation tools.

## [0.5.0]

### Added
- **`get_block` tool** — drill into one block (named group or scope prefix):
  member layers with per-layer params/FLOPs, block totals, and the edges
  crossing the block boundary (what feeds the block and what it feeds). Pairs
  with `list_blocks` for high-level → detail navigation.
- **Runnable example models** under `examples/` (`tiny-gpt`, `tiny-cnn`) so the
  server can be tried without exporting from the app. A test guards that every
  shipped example loads and validates with zero errors.
- **`--version` flag** (alias `-v`) prints the version and exits. The version is
  now sourced from `package.json`, so the CLI, the MCP handshake, and the
  package no longer drift.

## [0.4.0]

### Added
- **`describe_architecture` tool** — one-call orientation that returns the
  topologically-ordered layer pipeline, model depth (longest path), input/output
  shapes, total params and MACs, the top-5 heaviest layers by parameters **and**
  by compute, and a validation rollup. Replaces the common
  `get_model_summary` + `param_count_by_block` + `flops_by_block` +
  `validate_model` chain with a single round-trip.
- **Test suite (Vitest, ~100 unit tests)** covering validation, write ops,
  impact analysis, the param/FLOPs estimators, the loader, Mermaid rendering,
  the regex guard, and every tool handler. Run with `npm test`.
- **GitHub Actions CI** — `typecheck` + `build` + `test` on Node 20 and 22 for
  every push and pull request.
- `CONTRIBUTING.md` with a 3-step "add a tool" guide.
- `npm run typecheck` script (`tsc --noEmit`).

### Changed
- **Hardened user-supplied regex** (`find_layers`, `layer_impact`, the `/regex/`
  form of name resolution) behind a shared guard: caps pattern length and
  rejects nested-unbounded-quantifier shapes that risk catastrophic
  backtracking.
- **Stricter model loading** — files whose components lack a string `id` are now
  rejected at load time with a clear message instead of surfacing later as
  confusing "layer not found" errors.

### Fixed
- `mermaid_diagram` truncation no longer does an O(n²) `findIndex` scan per
  connection; it uses a precomputed id set.
- `lmHead` layers now report parameters (`embedDim × vocabSize`, optional bias)
  and FLOPs (`tokens × embedDim × vocabSize`); previously the estimators had no
  case for them and silently returned 0, undercounting LM parameter budgets.

## [0.3.0]
- `validate_model`, `find_path`, `list_connections`, hyperparameters and design
  notes tools, layer/connection delete tools, and `--watch` live reload.

## [0.2.0]
- Write tools: `add_layer`, `modify_layer`, `add_connection`, `save_model`.

## [0.1.1]
- Initial release: read-only structural tools over a Neurarch model file.
