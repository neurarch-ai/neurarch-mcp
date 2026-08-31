# neurarch-mcp

[![CI](https://github.com/neurarch-ai/neurarch-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/neurarch-ai/neurarch-mcp/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/neurarch-mcp.svg)](https://www.npmjs.com/package/neurarch-mcp)
[![npm downloads](https://img.shields.io/npm/dm/neurarch-mcp.svg)](https://www.npmjs.com/package/neurarch-mcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Model Context Protocol](https://img.shields.io/badge/MCP-server-7c3aed.svg)](https://modelcontextprotocol.io)
[![GitHub stars](https://img.shields.io/github/stars/neurarch-ai/neurarch-mcp.svg?style=social)](https://github.com/neurarch-ai/neurarch-mcp/stargazers)
[![Try Neurarch](https://img.shields.io/badge/Neurarch-try_it-7c3aed)](https://neurarch.com)

Model Context Protocol server that exposes a [Neurarch](https://neurarch.com) model graph to Claude Code, Claude Desktop, Cursor, VS Code, Windsurf, Codex, and any other MCP-aware AI agent.

The agent gets **structural awareness** of your neural network: layer list, parameter counts, FLOPs, blast-radius impact analysis, offline design linting, and Mermaid diagrams, without you pasting 200 lines of `nn.Module` into chat. Point it at your `.py` and it reads the model straight out of the source.

<!-- For guaranteed inline autoplay on GitHub: drag docs/demo.webm into any GitHub
     issue or PR comment box, then replace the <video> src below with the resulting
     https://github.com/user-attachments/assets/... URL. The raw URL works as a
     poster + download fallback until then. -->
<video
  src="https://github.com/neurarch-ai/neurarch-mcp/raw/main/docs/demo.webm"
  poster="https://raw.githubusercontent.com/neurarch-ai/neurarch-mcp/main/docs/demo-poster.png"
  autoplay muted loop playsinline width="900">
  <a href="https://github.com/neurarch-ai/neurarch-mcp/raw/main/docs/demo.webm">
    <img src="https://raw.githubusercontent.com/neurarch-ai/neurarch-mcp/main/docs/demo-poster.png"
         alt="neurarch-mcp answering a model-structure question, grounded in real tool output" width="900">
  </a>
</video>

<sub>▶ [Watch the 13s demo](https://github.com/neurarch-ai/neurarch-mcp/raw/main/docs/demo.webm) — every number above is produced by the tools, not guessed by the model.</sub>

## Why

When you ask Claude Code "rewrite the training loop for my new encoder block", it sees your `train.py` but not your model. It guesses at shapes and parameter counts. `neurarch-mcp` plugs the structured model graph into the same conversation so the agent can answer "what depends on `attn_8`?" or "where does the parameter budget actually live?" with one call instead of ten.

## What it looks like

A real exchange against [`examples/tiny-gpt.neurarch.json`](./examples/tiny-gpt.neurarch.json):

```
You ▸ Where does the parameter budget live, and is it safe to drop block_1?

Claude ▸ (calls describe_architecture)
  tiny-gpt — 8 layers, depth 7, 57.8M params, 8.27 GMACs.
  Heaviest by params: token_embedding (25.7M, 44%), lm_head (25.7M), block_0/1 (3.1M each).
  Heaviest by compute: lm_head, then the two decoder blocks. Validation: clean.

  (calls layer_impact { names: ["block_1"] })
  Downstream of block_1: final_norm and lm_head (shape-sensitive, weight-carrying), then output.

  The budget is dominated by the embedding + LM head (~89%), not the decoder blocks,
  so dropping block_1 frees only ~3.1M params (~5%) while removing one of just two
  decoder layers. It is structurally safe (final_norm just re-reads block_0's output),
  but it is the wrong lever if your goal is shrinking the model — quantizing the
  embedding/head would do far more.
```

Numbers above are produced by the tools, not estimated by the model.

## Install

No install step. Every client below launches the server the same way:

```bash
npx -y neurarch-mcp /abs/path/to/model.py                    # your PyTorch source
npx -y neurarch-mcp /abs/path/to/your-model.neurarch.json    # or a file saved from the app
```

**Point it at a `.py` and it works.** The server parses PyTorch source into the same graph the app uses, with the same parser, so there is no "first go draw your model" step between installing this and getting an answer. What comes out of source is layers, types, hyperparameters and wiring, all exact. What does not is tensor shapes, because the source never says what goes in: parameter counts still come out, FLOPs and shape contracts report as unknown rather than as zero. `--write` is refused on a `.py` file, since the graph was derived from it and this server does not generate Python.

For the full picture, export from the [Neurarch](https://neurarch.com) app with **File → Save (.json)**: that file carries shapes, groups, hyperparameters and design notes, so every tool has everything. Add `--watch` so the agent sees app-side saves without a restart, and `--write` if you want the agent to be able to edit the model (off by default).

Use an **absolute** path to the model file in any global config: `npx` does not run from your project directory, so relative paths only work in project-scoped configs.

### Claude Code

One command:

```bash
claude mcp add neurarch -- npx -y neurarch-mcp /abs/path/to/your-model.neurarch.json --watch
```

Or commit a project-scoped `.mcp.json` at the repo root so every collaborator gets the server automatically:

```json
{
  "mcpServers": {
    "neurarch": {
      "command": "npx",
      "args": ["-y", "neurarch-mcp", "./model.neurarch.json", "--watch"]
    }
  }
}
```

### Claude Desktop

Open **Settings → Developer → Edit Config**, or edit the file directly:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "neurarch": {
      "command": "npx",
      "args": ["-y", "neurarch-mcp", "/abs/path/to/your-model.neurarch.json", "--watch"]
    }
  }
}
```

Fully quit and reopen Claude Desktop (the config is read at startup). The tools appear under the search-and-tools icon in the chat input.

### Cursor

Create `.cursor/mcp.json` in your project (or `~/.cursor/mcp.json` for all projects), then enable the server under **Settings → MCP**:

```json
{
  "mcpServers": {
    "neurarch": {
      "command": "npx",
      "args": ["-y", "neurarch-mcp", "./model.neurarch.json", "--watch"]
    }
  }
}
```

### VS Code (Copilot agent mode)

Create `.vscode/mcp.json` (note the `servers` key, not `mcpServers`):

```json
{
  "servers": {
    "neurarch": {
      "command": "npx",
      "args": ["-y", "neurarch-mcp", "${workspaceFolder}/model.neurarch.json", "--watch"]
    }
  }
}
```

Or from a shell: `code --add-mcp '{"name":"neurarch","command":"npx","args":["-y","neurarch-mcp","/abs/path/to/model.neurarch.json"]}'`

### Other clients (Windsurf, Codex, ...)

Same `command` + `args` shape; only the config file location differs. For clients that speak Streamable HTTP instead of stdio, run the server with `--http` and point the client at it:

```json
{
  "mcpServers": {
    "neurarch": {
      "type": "http",
      "url": "http://127.0.0.1:8787/mcp"
    }
  }
}
```

If you set `NEURARCH_MCP_TOKEN`, add `"headers": { "Authorization": "Bearer <token>" }`. See [Remote access](#remote-access) for tunnels and security.

### Verify it works

Ask the agent: *"List the Neurarch tools you can see."* You should get `describe_architecture`, `layer_impact`, `validate_model` and friends (18 read tools; 6 more with `--write`). From a shell, `npx -y neurarch-mcp --help` prints usage and the full tool list.

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

- [`examples/tiny-vit.py`](./examples/tiny-vit.py) — **plain PyTorch source**, no export step, with a real design bug planted in it.
- [`examples/tiny-gpt.neurarch.json`](./examples/tiny-gpt.neurarch.json) — a small GPT-style decoder (embedding, 2 transformer blocks, LM head).
- [`examples/tiny-cnn.neurarch.json`](./examples/tiny-cnn.neurarch.json) — a CIFAR-style CNN (2 conv stages + classifier).
- [`examples/resnet-mini.neurarch.json`](./examples/resnet-mini.neurarch.json) — a residual block with a skip/merge node (a branchier graph for impact and path tools).

Point it at the Python file and ask *"lint this model"*, and `lint_model` comes
back with five findings on a file nothing was exported from, one of them a
blocker: `head-dim-divisibility` on `attn`, because `embed_dim=258` is not
divisible by `num_heads=8`. That is a runtime crash sitting in source that reads
fine, found offline, with no key and no account.

Then ask:

> Look at the Neurarch model. Where do the parameters actually live, and which block would shrink the model fastest if I cut it in half?

The agent calls `describe_architecture` (one shot: pipeline, depth, param + compute hotspots, validation), then `layer_impact` on the heaviest block, and writes a recommendation grounded in the actual numbers from the model, like the transcript above.

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
| `lint_model` | Neurarch's structural **design rules**, offline and with no API key: attention head-dim and GQA divisibility, norm/activation ordering, dropout and feature ranges, missing residuals in deep stacks, and the statically decidable shape rules. Same rule set the Neurarch CI action reports, so a clean result here is a clean CI run. |
| `check_design` | Neurarch's **full verdict** on the model: readiness to train, parameter and cost estimates, the best deployment target and its latency, and any decision still left to the human. Broader than `validate_model`, which is the local structural subset. Requires `NEURARCH_API_KEY` and makes one network call. |
| `find_path` | Shortest directed path between two layers, or `null` when unreachable. |
| `list_connections` | Flat edge list with optional `from` / `to` filters. |
| `param_count_by_block` | Parameter counts grouped by block / scope / type. |
| `flops_by_block` | MAC counts (FLOPs ÷ 2) grouped by block / scope / type. |
| `mermaid_diagram` | Render the model as Mermaid `flowchart TD` syntax; groups render as labelled subgraphs. Truncates past 60 layers (keeping the topological head). |
| `list_blocks` | List collapsed groups (or scope-derived blocks if none): members, params, FLOPs. |
| `get_block` | Drill into one block (group or scope prefix): per-layer params/FLOPs, totals, and the edges crossing the block boundary (what feeds it, what it feeds). |
| `diff_models` | Structurally diff the current model against another `.neurarch.json` file: layers added / removed / modified (field-level) and connection changes. |
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

`layer_impact` is the headline read tool. Before the agent recommends `delete every conv_X`, it can call `layer_impact` and tell the user "this rewires 8 downstream layers, 3 of which carry weights and will need rebuild." `validate_model` is the headline safety tool — call it before recommending a destructive edit to surface pre-existing issues separately from the change. Three tools grade the model, and they are a ladder worth climbing in order:
`validate_model` asks whether it is a well-formed graph at all (cycles, dangling
refs, orphans), `lint_model` runs the design rules over it, and both are free,
offline and instant. `check_design` answers what the file cannot -- readiness,
training cost, deployment fit, and the decisions that are still the human's --
and it is the only one that needs a key and a network call. An agent that starts
at the top pays for an answer two thirds of which was computable on the machine
it was already standing on.

### Flags

- `--write` — expose mutation tools. Off by default so accidental writes can't clobber a file you're editing in the Neurarch app.
- `--watch` — poll the model file for changes and reload on save. Pair with the Neurarch app: edit visually, agent sees the latest graph without restarting the MCP server. Note: an external save will overwrite any unsaved in-memory edits made via `--write`.
- `--http[=PORT]` — serve over Streamable HTTP instead of stdio (default port `8787`). See [Remote access](#remote-access) below.
- `--host=ADDR` — bind address for `--http`. Defaults to `127.0.0.1` (loopback only).
- `--version` (alias `-v`) — print the version and exit. `--help` (`-h`) prints usage and the full tool list.

### One server, many models

Every read tool takes an optional **`model_path`**, so a single server covers a
whole repository instead of one file. Ask `get_model_summary` about
`baseline.py`, then about `variant_b.neurarch.json`, without registering a
second server or restarting anything. Files are cached and re-read whenever
their mtime moves, so the answer is never stale.

Write tools deliberately refuse `model_path`: mutations always target the file
passed on the command line, so an agent cannot edit, and then save over, a path
it invented.

### Tool annotations

Every tool declares what it does to your files. The read tools are marked
read-only and closed-world; `check_design` marks itself as the one that reaches
off the machine; `delete_layer`, `delete_connection`, `modify_layer` and
`save_model` mark themselves destructive. Clients that honour annotations can
therefore confirm the three tools that can actually destroy something instead of
prompting on all thirty. Results also carry `structuredContent` alongside the
JSON text, for clients that read it.

## Remote access

By default the server talks stdio, so the agent and the model file live on the same machine. `--http` serves the same tools over Streamable HTTP, so a **hosted or phone-based agent can drive a model running on your machine** — e.g. behind a Cloudflare or Tailscale tunnel.

```bash
# local only (safe default: loopback, no auth needed)
npx neurarch-mcp model.neurarch.json --http

# expose to a tunnel with a bearer token and write tools
NEURARCH_MCP_TOKEN=$(openssl rand -hex 16) \
  npx neurarch-mcp model.neurarch.json --write --http --host=0.0.0.0
# then point cloudflared / tailscale funnel at :8787 and connect the agent to
# https://<tunnel>/mcp with the same token.
```

`POST` JSON-RPC to `/mcp`; `GET /health` is a liveness probe. Sessions follow the standard Streamable HTTP handshake (`Mcp-Session-Id`), so any MCP-aware client connects unchanged.

**Security:**

- Binds to `127.0.0.1` by default. Without a token, the `Host` header is checked against a loopback allowlist (DNS-rebinding protection) and no CORS headers are sent.
- Set `NEURARCH_MCP_TOKEN` to require `Authorization: Bearer <token>` on every request (constant-time checked). It is **required** before `--write` may bind to a non-loopback host — the server refuses to start otherwise.

## Sharing results with the corpus (opt-in)

Set `NEURARCH_REPORT=1` to share one anonymous structure+verdict row per
`validate_model` call with the Neurarch corpus: the structural fingerprint
(8-char hash), the layer-type histogram and edge count that let the server
verify it, and the finding (rule id, severity) pairs. Never the graph,
parameter values, layer names, file paths, or any identity: the payload shape
cannot carry them, and the server rejects rows whose fingerprint does not
recompute from the histogram it came with.

Off by default. Reporting is fire-and-forget with a 5-second cap, so it can
never slow or fail a tool call. Policy:
[neurarch.com/rules.html#data](https://neurarch.com/rules.html#data).

## Network: two switches, both off by default

This server opens a socket for exactly two reasons, and neither happens unless
you turn it on:

| Switch | What it sends | When |
|---|---|---|
| `NEURARCH_REPORT=1` | An anonymous structure+verdict row (fingerprint, histogram, edge count, rule-id/severity pairs). **Structurally incapable of carrying the graph.** | After each `validate_model` call, fire and forget |
| `NEURARCH_API_KEY=nrk_...` | **The model graph itself**, because a verdict about a graph cannot be computed without it | Only when the agent calls `check_design` |

The second one is called out separately because it is materially different from
the first: it sends your architecture. If that is not acceptable for a given
model, do not set a key. Every other tool in this server keeps working, and
`validate_model` gives you the local structural subset of the same checks.

With neither set, this server makes **no network calls at all**.

## Troubleshooting

- **The server never appears in the client.** The model path must be absolute in any global config; `npx` does not run from your project directory. Relative paths only work in project-scoped configs (`.mcp.json`, `.cursor/mcp.json`, `.vscode/mcp.json`).
- **Read tools work but write tools are missing.** You did not pass `--write`. It is off by default so accidental writes can't clobber a file you're editing in the app.
- **`npx` fails on first run.** Node >= 20 is required (`node --version`).
- **Claude Desktop shows nothing after editing the config.** Fully quit and reopen the app; the config is only read at startup.
- **The agent sees a stale graph after you edit in the app.** Add `--watch`, or restart the server.

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
npm test                      # vitest (≈190 unit + end-to-end tests)
node dist/index.js --help     # confirm bin works
```

CI runs `typecheck` + `build` + `test` on Node 20 and 22 for every push and PR.

The package vendors two things from the main Neurarch repo, and they are vendored for the same reason: this server has to work with no network, no API key and no second install step.

- `src/lib/` — pure-TypeScript utilities (model types, parameter and FLOP estimators, impact analyzer), maintained as source here.
- `src/vendor/engine.bundle.mjs` — the compiled Neurarch engine: the component registry, the PyTorch parser behind `.py` support, and the rule set behind `lint_model`. Generated, never hand-edited; the header says how to regenerate it, and `src/vendor/engine.contract.test.ts` fails if its exports drift or it ever acquires an import.

Neither adds a runtime dependency: `@modelcontextprotocol/sdk` is still the only one.

## License

MIT. See [LICENSE](./LICENSE).

## Links

- [Neurarch](https://neurarch.com) — the visual neural-network editor that produces the model files this server reads.
- [Model Context Protocol](https://modelcontextprotocol.io) — the spec this server implements.
- [npm](https://www.npmjs.com/package/neurarch-mcp) — package page.
