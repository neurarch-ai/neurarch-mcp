# Changelog

All notable changes to neurarch-mcp are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

- `CONTRIBUTING.md` with a 3-step "add a tool" guide.

## [0.2.0]

Current published release.

- 13 read tools: `get_model_summary`, `get_layer`, `find_layers`, `layer_impact`, `validate_model`, `find_path`, `list_connections`, `param_count_by_block`, `flops_by_block`, `mermaid_diagram`, `list_blocks`, `list_hyperparams`, `get_design_notes`.
- 6 write tools behind `--write`: `add_layer`, `modify_layer`, `add_connection`, `delete_layer`, `delete_connection`, `save_model`.
- `--watch` flag reloads the model file on change.
- Pure-TypeScript, no runtime dependency beyond the MCP SDK.

Earlier 0.1.x releases predate this changelog.
