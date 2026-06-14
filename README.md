# neurarch-mcp

[![CI](https://github.com/neurarch-ai/neurarch-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/neurarch-ai/neurarch-mcp/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/neurarch-mcp.svg)](https://www.npmjs.com/package/neurarch-mcp)
[![npm downloads](https://img.shields.io/npm/dm/neurarch-mcp.svg)](https://www.npmjs.com/package/neurarch-mcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Model Context Protocol](https://img.shields.io/badge/MCP-server-7c3aed.svg)](https://modelcontextprotocol.io)
[![GitHub stars](https://img.shields.io/github/stars/neurarch-ai/neurarch-mcp.svg?style=social)](https://github.com/neurarch-ai/neurarch-mcp/stargazers)
[![Try Neurarch](https://img.shields.io/badge/Neurarch-try_it-7c3aed)](https://neurarch.com)

Model Context Protocol server that exposes a [Neurarch](https://neurarch.com) model graph to Claude Code, Cursor, Windsurf, Codex, and any other MCP-aware AI agent.

The agent gets **structural awareness** of your neural network: layer list, parameter counts, FLOPs, blast-radius impact analysis, and Mermaid diagrams, without you pasting 200 lines of `nn.Module` into chat.

<!-- Demo (highest-converting element for stars): record a 15-30s clip of the agent answering a
     model-structure question inside Claude Code or Cursor, save it to docs/demo.gif, then uncomment:
![neurarch-mcp answering a model-structure question inside Claude Code](docs/demo.gif)
-->


## Why

When you ask Claude Code "rewrite the training loop for my new encoder block", it sees your `train.py` but not your model. It guesses at shapes and parameter counts. `neurarch-mcp` plugs the structured model graph into the same conversation so the agent can answer "what depends on `attn_8`?" or "where does the parameter budget actually live?" with one call instead of ten.

## Install

No install. Just point your MCP client at it via `npx`:

```jsonc
// ~/.claude/mcp_servers.json (Claude Code)
{
  "mcpServers": {
    "neurarch": {
      "command": "npx",
      "args": ["-y", "neurarch-mcp", "/abs/path/to/your-model.neurarch.json"]
    }
  }
}
```

Every MCP-aware client uses the same `command` + `args` shape, only the config file differs:

| Client | MCP config |
|---|---|
| Claude Code | `~/.claude/mcp_servers.json` |
| Cursor | `.cursor/mcp.json` |
| Windsurf | its MCP config (same `command` + `args`) |
| Codex | its MCP config (same `command` + `args`) |

To produce the model file: open your model in the [Neurarch](https://neurarch.com) app, then **File → Save (.json)**. The MCP server reads that file directly.

## Try it in 30 seconds (no app needed)

This repo ships runnable example models under [`examples/`](./examples). Point the server at one and your agent can immediately answer structural questions:

```jsonc
{
  "mcpServers": {
    "neurarch": {
      "command": "npx",
      "args": ["-y", "neurarch-mcp", "./examples/tiny-gpt.neurarch.json"]
    }
  }
}
```

- [`examples/tiny-gpt.neurarch.json`](./examples/tiny-gpt.neurarch.json) — a small GPT-style decoder (embedding, 2 transformer blocks, LM head).
- [`examples/tiny-cnn.neurarch.json`](./examples/tiny-cnn.neurarch.json) — a CIFAR-style CNN (2 conv stages + classifier).
- [`examples/resnet-mini.neurarch.json`](./examples/resnet-mini.neurarch.json) — a residual block with a skip/merge node (a branchier graph for impact and path tools).

Then ask: *"describe the architecture, and tell me where the parameter budget lives."*

## Tools

### Read (always available)

| Tool | What it does |
|---|---|
| `get_model_summary` | One-shot overview: layer count, total params, dominant types, input/output shape. |
| `describe_architecture` | One-call orientation: topo-ordered pipeline, depth, IO shapes, total params/MACs, top-5 param **and** compute hotspots, validation rollup. Replaces a 4-tool chain. |
| `get_layer` | Full definition of one layer by name: params, shapes, notes, upstream/downstream ids. |
| `compare_layers` | Structural diff of two layers: same-type, param-count delta, shape match, and exactly which param keys differ. |
| `find_layers` | Search layers by type, name regex, scope prefix, or augmentation (e.g. frozen layers); optionally rank by parameter count. |
| `layer_impact` | Blast radius of changing a layer or matched set. Flags shape-sensitive and weight-carrying downstream layers. |
| `validate_model` | Structural invariants: cycles, dangling connection refs, duplicate ids/names, orphan layers. |
| `find_path` | Shortest directed path between two layers, or `null` when unreachable. |
| `list_connections` | Flat edge list with optional `from` / `to` filters. |
| `param_count_by_block` | Parameter counts grouped by block / scope / type. |
| `flops_by_block` | MAC counts (FLOPs ÷ 2) grouped by block / scope / type. |
| `mermaid_diagram` | Render the model as Mermaid `flowchart TD` syntax; groups render as labelled subgraphs. Truncates past 60 layers (keeping the topological head). |
| `list_blocks` | List collapsed groups (or scope-derived blocks if none): members, params, FLOPs. |
| `get_block` | Drill into one block (group or scope prefix): per-layer params/FLOPs, totals, and the edges crossing the block boundary (what feeds it, what it feeds). |
| `list_hyperparams` | Model-level hyperparameters (learning rate, batch size, ...) the user set in the app. |
| `get_design_notes` | Pinned design rationale: agent / advisor / manual notes, optionally filtered by layer. |

### Write (opt in with `--write`)

| Tool | What it does |
|---|---|
| `add_layer` | Insert a new layer, optionally auto-wired downstream of an existing one. |
| `modify_layer` | Shallow-merge params, rename, or change scope. Returns a before/after diff. |
| `add_connection` | Wire two existing layers. Fails on self-loops and duplicate edges. |
| `delete_layer` | Remove a layer and every connection touching it. Invalidates downstream shapes. |
| `delete_connection` | Remove a single directed edge. Invalidates the target's cached shape. |
| `save_model` | Persist the in-memory model to disk. Call this after any mutation. |

`layer_impact` is the headline read tool. Before the agent recommends `delete every conv_X`, it can call `layer_impact` and tell the user "this rewires 8 downstream layers, 3 of which carry weights and will need rebuild." `validate_model` is the headline safety tool — call it before recommending a destructive edit to surface pre-existing issues separately from the change.

### Flags

- `--write` — expose mutation tools. Off by default so accidental writes can't clobber a file you're editing in the Neurarch app.
- `--watch` — poll the model file for changes and reload on save. Pair with the Neurarch app: edit visually, agent sees the latest graph without restarting the MCP server. Note: an external save will overwrite any unsaved in-memory edits made via `--write`.
- `--version` (alias `-v`) — print the version and exit. `--help` (`-h`) prints usage and the full tool list.

## Example prompt (Claude Code)

After wiring the server, in Claude Code:

> Look at the Neurarch model. Where do the parameters actually live, and which block would shrink the model fastest if I cut it in half?

The agent calls `describe_architecture` (one shot: pipeline, depth, param + compute hotspots, validation), then `layer_impact` on the heaviest block, and writes a recommendation grounded in the actual numbers from your model.

## What this is not

- **Not a generic codebase indexer.** This serves one `.neurarch.json` file. For codebase structure, use [GitNexus](https://github.com/abhigyanpatwari/GitNexus) or similar.
- **Not connected to your Neurarch workspace.** v1 reads a saved JSON file only. Live editing happens in the Neurarch web app.

## Issues & Feedback

This repo is the public home for both:

- **neurarch-mcp** (this MCP server): bugs, protocol changes, integration questions.
- **[Neurarch](https://neurarch.com)** (the app): canvas bugs, agent issues, linter rules, feature requests.

| | |
|---|---|
| 🐛 **[Report a bug](https://github.com/neurarch-ai/neurarch-mcp/issues/new?template=bug_report.yml)** | Something is broken or behaving unexpectedly. |
| 💡 **[Request a feature](https://github.com/neurarch-ai/neurarch-mcp/issues/new?template=feature_request.yml)** | An idea that would make Neurarch or the MCP server better. |
| ❓ **[Ask a question](https://github.com/neurarch-ai/neurarch-mcp/issues/new?template=question.yml)** | Something specific you can't figure out. |
| 💬 **[Start a discussion](https://github.com/neurarch-ai/neurarch-mcp/discussions)** | Open-ended ideas, design feedback, "how would you…". |

Please tag issues with `mcp`, `app`, `linter`, or `feature-request` so we can triage faster.

## Star this repo

If `neurarch-mcp` saved you from pasting an `nn.Module` into chat, a ⭐ helps other ML engineers find it. It is the lowest-effort way to support the project.

## Contributing

A new tool is a small, self-contained PR. See [CONTRIBUTING.md](./CONTRIBUTING.md) for the 3-step "add a tool" guide.

## Development

```bash
git clone https://github.com/neurarch-ai/neurarch-mcp
cd neurarch-mcp
npm install
npm run typecheck             # tsc --noEmit
npm run build                 # tsup → dist/index.js
npm test                      # vitest (≈100 unit tests)
node dist/index.js --help     # confirm bin works
```

CI runs `typecheck` + `build` + `test` on Node 20 and 22 for every push and PR.

The package vendors a small set of pure-TypeScript utilities (model types, parameter and FLOP estimators, impact analyzer) from the main Neurarch repo. They live under `src/lib/` and have no runtime dependencies beyond `@modelcontextprotocol/sdk`.

## License

MIT. See [LICENSE](./LICENSE).

## Links

- [Neurarch](https://neurarch.com) — the visual neural-network editor that produces the model files this server reads.
- [Model Context Protocol](https://modelcontextprotocol.io) — the spec this server implements.
- [npm](https://www.npmjs.com/package/neurarch-mcp) — package page.
