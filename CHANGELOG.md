# Changelog

All notable changes to `neurarch-mcp` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.13.0]

### Added
- **`rank_designs`**: which of k candidate designs to spend a training run on,
  computed locally with the same contract as `POST /api/v1/rank`. Candidates
  are paths, `zoo:`/`hf:` refs or inline graphs; `include_current` adds the
  server's model. Blocked candidates (a pre-flight finding that means the graph
  will not forward-pass, the measured part: 96/96 crashed, 80/80 ran) rank
  last and come back as reclaimable budget. Legal candidates are ordered only
  by rules with a trained outcome behind them, and a tie stays a tie:
  `recommended` is null when nothing measured separates the top, which is the
  common answer and the honest one. Params, cost and GPU fit are returned per
  candidate and never used to order. `calibration` ships inside every result.

- **`export_pytorch`**: the graph as a runnable `nn.Module`, the app's own
  generator. `save_to` is honoured only under `--write` and never over the file
  the server was started from.

- **81 reference architectures bundled** (`zoo/`, 320KB gzipped, synced from
  github.com/neurarch-ai/awesome-llm-model-zoo): DeepSeek-V3, Qwen2.5, Llama,
  Mixtral, Gemma, Whisper, CLIP, BERT, ViT, ResNet and more, with real
  dimensions from each model's config and a parameter count checked against
  the published one. `list_architectures` searches them, `load_architecture`
  opens one, and any read tool accepts `model_path: "zoo:<id>"`. Offline.

- **Hugging Face input, behind `--hf`**: `load_hf_model` and `model_path:
  "hf:<org/name>"` build a graph from a repo's config.json. It is the one tool
  that opens a socket, which is why it is listed only when the flag is on;
  `HF_TOKEN` is sent for gated repos, results are cached for a day under
  `~/.cache/neurarch-mcp`. The result says whether it came from the real config
  or a family template (`configSource`) and quotes the parameter count HF
  publishes, so the agent can see how close the graph is (Qwen2.5-0.5B: graph
  494.00M against 494,032,768 published).

- **`find_models`**: walk a directory for nn.Module definitions and saved
  graphs, try the parser on each, and say which need a runtime trace instead.

- **`neurarch-trace`** (`python/neurarch-trace`, PyPI): instantiate the model,
  run one forward with hooks, write a `.neurarch.json` with real shapes. Covers
  what static parsing cannot: `from_pretrained`, timm, models spread across
  files, dynamic construction. `pip install neurarch-trace`, then
  `neurarch-trace my_pkg.model:build --input 1,3,224,224`.

- **MCP prompts**: `review_design`, `pre_train_checklist`, `shrink_for_target`,
  `compare_with_reference`, `explain_finding`. These render as slash commands
  in Claude Desktop, Cursor and VS Code, and each is the tool ladder written
  out in order, with the rule that every number comes from a tool result.

- **MCP resources**: `neurarch://model` (the graph), `neurarch://model/mermaid`,
  `neurarch://model/pytorch`, `neurarch://zoo`, `neurarch://zoo/{id}`,
  `neurarch://rules` (the provenance table). A resource is pinned as context;
  a tool result scrolls away.

- **`model_source`**: any read tool accepts inline model text (a
  `.neurarch.json` document or PyTorch source) instead of a path, for clients
  with no shared filesystem.

- **Hosted mode**: `--http` with no model argument serves a server that answers
  about whatever each call names (`model_path` zoo:/hf:, or `model_source`).
  `Dockerfile`, `fly.toml` and `docs/HOSTED.md` carry the deploy.

- **`neurarch-mcp lint <files...>` and `neurarch-mcp check <file>`**: the two
  grading tools as a CLI, exit 1 on a block, `--json` for CI. One package that
  is both the server and the linter; `neurarch-lint` on npm stays as an alias.

- **`--tools=core`**: advertise eleven tools instead of twenty-five. Every tool
  stays callable by name; this trims what rides along in the agent's context.

- **Claude Desktop bundle**: `manifest.json` and `npm run build:mcpb` produce
  the one-click `.mcpb`.

- **`skills/neurarch-skill`**: the evidence-gated edit loop skill now lives in
  this repo (`npx skills add neurarch-ai/neurarch-mcp`).

### Changed
- The README leads with a GIF (GitHub strips `<video>`), has Cursor and VS Code
  install buttons, and is written for the search terms people use.
- stdio mode routes `console.log` to stderr for the life of the process, so
  nothing in the vendored engine can corrupt a JSON-RPC frame.
- `ToolContext` carries `writeEnabled`; tools that can create files
  (`save_to`) refuse without it, the same way `save_model` is gated.

## [0.12.0]

### Changed
- **`check_design` no longer needs an API key, or a network call.** The verifier
  is vendored (`src/vendor/verifier.bundle.mjs`) and runs here, in about 13ms,
  against the graph already in memory.

  The key was never buying what a key is supposed to buy. `checkDesign` is a
  pure, synchronous function of the graph; the server around it validated the
  key, capped the graph at 2000 components, and rewrote the error text. The
  corpus row that would have justified metering is written by a different
  endpoint, which requires no key. So the key bought no compute and captured no
  measurement: it put a signup between a stranger and the one tool this product
  is built around, while the other eighteen tools ran offline on first install.

  The size cap is kept, because a runaway generated graph should come back with
  a sentence rather than stall the tool call. The hosted endpoint is unchanged
  for callers who are not running this server.

- **The network promise has no exceptions now.** Unset `NEURARCH_REPORT` and
  this server opens no socket, from any tool. `NEURARCH_API_KEY` and
  `NEURARCH_API_URL` are read nowhere in the package.

- **Corpus reporting covers all three grading tools**, not just
  `validate_model`, and grades with `lintModelGraph` rule ids rather than the
  validator's own vocabulary.

  Both halves were making the channel unusable. Coverage was an accident of
  which tool an agent happened to reach for: an agent that asked for the design
  rules or the whole verdict recorded nothing. And an `mcp` row said `cycle`
  where a `ci` row said `head-dim-divisibility`, so rows from the two channels
  could not be pooled even when they were about the same graph. The row is now
  derived from the graph rather than from the tool result, so the same graph
  produces the same row whichever tool asked, and the server's fingerprint
  dedupes rather than triple-counting.

### Added
- **`lint_model` returns `provenance`**: for each rule that fired and has a
  published measurement behind it, the measurement, quoted with its public
  source. Local, from a static table, no key. Rules with no number behind them
  are absent rather than dressed up, which is the same honesty contract the
  table has always carried.

## [0.11.0]

### Added
- **PyTorch source is now a first-class input.** `npx neurarch-mcp model.py` works.
  The server parses the file into the same graph the app and the CI action use,
  with the same parser, which removes the precondition that made this server a
  tool for people who were already Neurarch users: there is no longer a "first
  go open the app, draw your model and export it" step between installing this
  and getting an answer.

  What comes out of source is exact for layers, types, hyperparameters and
  wiring. Tensor shapes do not, because the source never says what goes in, so
  parameter counts still come out while FLOPs and shape contracts report as
  unknown rather than as a confident zero. `--write` is refused on a `.py` file
  and `save_model` refuses a `.py` target, because the graph was derived from
  the source and this server does not generate Python.

- **`lint_model`** — the design rules, offline and with no API key: attention
  head-dim and GQA divisibility, normalization and activation ordering, dropout
  and feature ranges, missing residuals in deep stacks, and the shape rules that
  can be decided statically. The same rule set the Neurarch CI action reports,
  so a clean result here is a clean CI run.

  It completes a ladder that was missing its middle rung. `validate_model` asks
  whether the thing is a well-formed graph at all; `lint_model` runs the design
  rules; `check_design` adds readiness, cost and deployment fit and is the only
  one that needs a key and a network call. An agent that reached for the network
  first was paying for an answer two thirds of which was computable locally.

- **`model_path` on every read tool.** One server now covers a repository
  instead of one file: ask about `baseline.py`, then about
  `variant_b.neurarch.json`, without registering a second server. Files are
  cached and re-read whenever their mtime moves, so an answer is never stale.
  Write tools refuse `model_path` outright: honouring it would let an agent
  mutate, and then save over, a file nobody pointed this server at, and ignoring
  it would edit a different model than the agent believes it is editing.

- **Tool annotations** (`readOnlyHint`, `destructiveHint`, `idempotentHint`,
  `openWorldHint`). All thirty tools were previously advertised as equally
  consequential, so a client had to confirm all of them or none. Now the read
  tools declare themselves read-only and closed-world, `check_design` declares
  the one network call, and `delete_layer` / `delete_connection` /
  `modify_layer` / `save_model` declare that they destroy something.

- **`structuredContent` on every result**, alongside the JSON text that clients
  predating it still read. No `outputSchema` is declared: a client fails a call
  whose structured result does not match a declared schema, so hand-writing two
  dozen schemas would convert every drift between schema and handler into a
  broken tool. The data is worth shipping without that; the contract can follow
  when the schemas are generated rather than typed twice.

- `examples/tiny-vit.py`, so the 30-second path needs no app and no export. It
  carries a real planted bug (`embed_dim=258`, `num_heads=8`) and a test asserts
  `lint_model` still finds it, because an example that quietly stops
  demonstrating anything is worse than no example.

### Changed
- The vendored Neurarch engine (`src/vendor/engine.bundle.mjs`) is compiled from
  the app repo and carries the component registry, the PyTorch parser and the
  rule set. A contract test asserts its exports and its lack of imports, so a
  regenerated bundle whose surface moved fails here rather than in a user's
  agent.
- `server.ts` had no test before this release, which is why the routing it now
  does is covered end to end through a real MCP client over an in-memory
  transport.

## [0.10.1]

### Fixed
- **`check_design` returned "Invalid or missing API key" to everyone who had a
  valid one.** 0.10.0 defaulted to `https://neurarch.com/api/v1/check`. The apex
  answers every `/api` route with a 307 to `www.neurarch.com`; Node's `fetch`
  follows it, keeps the method and body as a 307 requires, and then strips the
  `Authorization` header because the redirect crosses origins (this is the fetch
  spec, not a bug in undici). The request that landed carried no key, the server
  answered 401, and the error text sent the user off to re-create a key that was
  never the problem. The default is now `https://www.neurarch.com/api/v1/check`,
  and a test pins the host so the trap cannot be walked into again.

  No action needed beyond upgrading. Anyone who worked around this by setting
  `NEURARCH_API_URL` to the `www` host can drop the override.

- Corpus reporting (`NEURARCH_REPORT=1`) posted to the same apex host. It has no
  `Authorization` header to lose, so rows were still delivered, but each one paid
  a redirect inside a 5-second fire-and-forget budget. It now posts to `www`
  directly, and reads `NEURARCH_REPORT_URL` at call time rather than freezing it
  at import time, matching `check_design`.

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
