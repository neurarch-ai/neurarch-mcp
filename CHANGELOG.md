# Changelog

All notable changes to `neurarch-mcp` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

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
