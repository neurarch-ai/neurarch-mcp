# Changelog

All notable changes to `neurarch-mcp` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

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
