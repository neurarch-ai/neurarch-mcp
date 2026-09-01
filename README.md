# neurarch-mcp

**PyTorch MCP server: lint, verify and rank a neural network design before you train it, from Claude Code, Cursor, Claude Desktop, VS Code, Windsurf or Codex.**

[![CI](https://github.com/neurarch-ai/neurarch-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/neurarch-ai/neurarch-mcp/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/neurarch-mcp.svg)](https://www.npmjs.com/package/neurarch-mcp)
[![npm downloads](https://img.shields.io/npm/dm/neurarch-mcp.svg)](https://www.npmjs.com/package/neurarch-mcp)
[![PyPI neurarch-trace](https://img.shields.io/pypi/v/neurarch-trace.svg?label=neurarch-trace)](https://pypi.org/project/neurarch-trace/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Model Context Protocol](https://img.shields.io/badge/MCP-server-7c3aed.svg)](https://modelcontextprotocol.io)
[![smithery badge](https://smithery.ai/badge/neurarch-ai/neurarch-mcp)](https://smithery.ai/servers/neurarch-ai/neurarch-mcp)
[![GitHub stars](https://img.shields.io/github/stars/neurarch-ai/neurarch-mcp.svg?style=social)](https://github.com/neurarch-ai/neurarch-mcp/stargazers)

Your coding agent reads your model as source text, so it guesses at shapes, parameter counts, and what an edit breaks. `neurarch-mcp` hands it the structured graph instead, plus the verifier: **the design rules, the full readiness / cost / deployment verdict, and which of several candidate designs is worth the GPU time.** Point it at a PyTorch `.py`, a saved graph, or a Hugging Face repo. Everything runs on your machine, with no API key, no account, and no network call unless you turn one on.

[![neurarch-mcp answering a model-structure question, every number from a tool call](https://raw.githubusercontent.com/neurarch-ai/neurarch-mcp/main/docs/demo.gif)](https://github.com/neurarch-ai/neurarch-mcp/raw/main/docs/demo.webm)

<sub>Every number above is produced by the tools, not guessed by the model. [13s webm](https://github.com/neurarch-ai/neurarch-mcp/raw/main/docs/demo.webm).</sub>

## Install

```bash
claude mcp add neurarch -- npx -y neurarch-mcp /abs/path/to/model.py        # Claude Code
```

[<img src="https://img.shields.io/badge/Add%20to-Cursor-000000?logo=cursor&logoColor=white" alt="Add to Cursor" height="28">](cursor://anysphere.cursor-deeplink/mcp/install?name=neurarch&config=eyJjb21tYW5kIjoibnB4IiwiYXJncyI6WyIteSIsIm5ldXJhcmNoLW1jcCIsIi4vbW9kZWwucHkiXX0=)
[<img src="https://img.shields.io/badge/Install%20in-VS%20Code-0098FF?logo=visualstudiocode&logoColor=white" alt="Install in VS Code" height="28">](https://vscode.dev/redirect/mcp/install?name=neurarch&config=%7B%22name%22%3A%20%22neurarch%22%2C%20%22command%22%3A%20%22npx%22%2C%20%22args%22%3A%20%5B%22-y%22%2C%20%22neurarch-mcp%22%2C%20%22%24%7BworkspaceFolder%7D%2Fmodel.py%22%5D%7D)
[<img src="https://img.shields.io/badge/Claude%20Desktop-.mcpb-D97757?logo=anthropic&logoColor=white" alt="Claude Desktop bundle" height="28">](https://github.com/neurarch-ai/neurarch-mcp/releases/latest)

Per-client config for Claude Desktop, Cursor, VS Code, Windsurf and Codex is [below](#client-setup). Node 20+ is the only requirement; there is no install step beyond `npx`.

## What it looks like

A real exchange against [`examples/tiny-gpt.neurarch.json`](./examples/tiny-gpt.neurarch.json):

```
You ▸ Where does the parameter budget live, and is it safe to drop block_1?

Claude ▸ (calls describe_architecture)
  tiny-gpt: 8 layers, depth 7, 57.8M params, 8.27 GMACs.
  Heaviest by params: token_embedding (25.7M, 44%), lm_head (25.7M), block_0/1 (3.1M each).
  Heaviest by compute: lm_head, then the two decoder blocks. Validation: clean.

  (calls layer_impact { names: ["block_1"] })
  Downstream of block_1: final_norm and lm_head (shape-sensitive, weight-carrying), then output.

  The budget is dominated by the embedding + LM head (~89%), not the decoder blocks,
  so dropping block_1 frees only ~3.1M params (~5%) while removing one of just two
  decoder layers. It is structurally safe (final_norm just re-reads block_0's output),
  but it is the wrong lever if your goal is shrinking the model. Quantizing the
  embedding/head would do far more.
```

And the one this server is built around. Point it at [`examples/tiny-vit.py`](./examples/tiny-vit.py), plain PyTorch source with a bug planted in it, and ask *"lint this model"*:

```
block  head-dim-divisibility [attn:multiHeadAttention]: embed dim (258) must be
       divisible by numHeads (8); head_dim would be 32.25
```

That is a runtime crash sitting in source that reads fine, found offline in milliseconds, with no key and no account. The same rule set the [Neurarch CI action](https://github.com/neurarch-ai/neurarch-lint) reports, so a clean result here is a clean CI run. The rules are measured (the crash rules: 96 of 96 blocked graphs crashed PyTorch forward, 80 of 80 passes ran); the static parser in front of them is the weaker half, and [we measured that too](#what-the-static-parser-can-read).

## Three ways in

| Input | What the agent gets | How |
|---|---|---|
| **A self-contained PyTorch `.py` file** | Layers, types, hyperparameters and wiring, when the file builds its own layers with literal sizes. Shapes and FLOPs are unknown, because the source never says what goes in, and are reported as unknown rather than as zero. **On real repositories this is the weak path**: see [what the parser can read](#what-the-static-parser-can-read). | `npx -y neurarch-mcp model.py` |
| **A traced graph** (`neurarch-trace`) | Everything, with real shapes. Handles what static parsing cannot: `from_pretrained`, timm, models spread across files, anything built dynamically. | `pip install neurarch-trace`<br>`neurarch-trace my_pkg.model:build --input 1,3,224,224` |
| **A Hugging Face repo** | The architecture from `config.json`, with the published parameter count next to the graph's so you can see how close it is (Qwen2.5-0.5B: 494.00M against 494,032,768). | `npx -y neurarch-mcp hf:Qwen/Qwen2.5-0.5B --hf` |

A `.neurarch.json` saved from the [Neurarch app](https://neurarch.com) (**File → Save**) is the fourth, and carries shapes, groups and design notes. Add `--watch` so the agent sees app-side saves without a restart.

### What the static parser can read

We measured it rather than describe it: [`docs/REAL_REPOS_STUDY.md`](./docs/REAL_REPOS_STUDY.md) runs the parser and linter over 116 model files from 59 popular repositories (nanoGPT, HF `modeling_*.py`, timm, torchvision, DiT, MAE, CLIP, Mamba, SAM, diffusers and more). The parser returned a graph for 63% of files, a graph a person would recognise as the model for 8% (DiT, MAE, CLIP, RWKV, x-transformers, Mamba, DETR and two others), and every one of the 103 findings it raised on real code was judged by hand to be its own artefact or a deliberate design choice: **zero real bugs found**. The causes are counted in the study (layers built through other classes, factories, loops, and dimensions like `config.hidden_size` the parser cannot evaluate).

Three consequences are in this release. A graph from source carries a `parseQuality` grade (`full`, `partial`, `thin`) on `describe_architecture` and `lint_model`, with the fix named. Dimension rules are held back on layers whose dimension is still source text, and the count is reported as `suppressed` rather than dropped. And `find_models` marks thin parses `partial` so an agent does not build a plan on two layers. For real repositories, use `neurarch-trace`: it reads the numbers at runtime, which is the only place they exist.

Every read tool also takes an optional `model_path`, so **one server covers a whole repository**: ask about `baseline.py`, then `variant_b.neurarch.json`, then `zoo:llama-3-8b`, without restarting anything. `find_models` tells the agent what is there.

## Tools

Three tools grade the model, and they are a ladder worth climbing in order. Each is free, offline and instant; `check_design` runs five pipeline stages, so an agent that starts at the top still pays for an answer two thirds of which a cheaper tool had.

| | Tool | Answers |
|---|---|---|
| 1 | `validate_model` | Is this a well-formed graph at all: cycles, dangling refs, duplicate names, orphans. |
| 2 | `lint_model` | Does it break a design rule: attention head-dim and GQA divisibility, norm/activation ordering, dropout and feature ranges, missing residuals in deep stacks, the shape rules decidable statically. Returns `provenance`: for every rule that fired and has a published measurement behind it, the measurement. A rule with no number is absent rather than dressed up. |
| 3 | `check_design` | Will it train, what will it cost, where can it run: readiness, parameter and cost estimates, the best deployment target and its latency, and the decisions still left to the human. Same code path as the app and `POST /api/v1/check`. |
| 4 | `rank_designs` | **Which of k candidate designs deserves the training budget.** Candidates are paths, `zoo:`/`hf:` refs or inline graphs. Blocked ones (a pre-flight finding that means the graph will not forward-pass) rank last and come back as reclaimable budget; that part is measured, 96 of 96 blocked graphs crashed PyTorch forward and 80 of 80 passes ran. Legal candidates are ordered only by rules with a trained outcome behind them, and **a tie stays a tie**: `recommended` is null when nothing measured separates the top, which is the common answer. Params, cost and GPU fit are returned per candidate for you to break ties on your own budget; they never order. `calibration` ships inside every result (pairwise accuracy 51.4% at 8.3% coverage, in-sample), so the ordering cannot be read as a quality prediction. |

### Inspect

| Tool | What it does |
|---|---|
| `describe_architecture` | One-call orientation: topo-ordered pipeline, depth, IO shapes, total params and MACs, top-5 param **and** compute hotspots, validation rollup. Start here. |
| `get_model_summary` | Layer count, total params, dominant types, input/output shape. |
| `get_layer` | One layer by name: params, shapes, notes, upstream/downstream. |
| `find_layers` | Search by type, name regex, scope prefix or augmentation; rank by parameter count. |
| `compare_layers` | Structural diff of two layers. |
| `layer_impact` | **Blast radius** of changing a layer or matched set: shape-sensitive and weight-carrying downstream layers. Call it before recommending an edit. |
| `find_path`, `list_connections` | Directed path between two layers; the edge list. |
| `param_count_by_block`, `flops_by_block` | Params and MACs grouped by block, scope or type. |
| `list_blocks`, `get_block` | Collapsed groups and what crosses their boundary. |
| `diff_models` | Structural diff against another `.neurarch.json`. |
| `mermaid_diagram` | The model as Mermaid `flowchart TD`. |
| `list_hyperparams`, `get_design_notes` | What the user set and wrote in the app. |

### Reference library and other models

| Tool | What it does |
|---|---|
| `list_architectures` | Search the **81 reference architectures bundled with this server** (DeepSeek-V3, Qwen2.5, Llama, Mixtral, Gemma, Whisper, CLIP, BERT, ViT, ResNet and more), each with real dimensions from the model's config and a parameter count checked against the published one. Offline. |
| `load_architecture` | Open one and describe it. Then `model_path: "zoo:<id>"` on any tool. |
| `load_hf_model` | A Hugging Face repo as a graph. Listed only under `--hf`, because it is the one tool that opens a socket. |
| `find_models` | Walk a directory for nn.Module definitions and saved graphs, try the parser on each, and say which need `neurarch-trace` instead. |
| `export_pytorch` | The graph as a runnable `nn.Module`, the app's own generator. `save_to` needs `--write` and never targets the file the server was started from. |

### Write (opt in with `--write`)

`add_layer`, `modify_layer`, `add_connection`, `delete_layer`, `delete_connection`, `save_model`. Mutations always target the file passed on the command line; write tools refuse `model_path`, so an agent cannot edit, and then save over, a path it invented. Refused on a `.py` (the graph was derived from it; use `export_pytorch` to emit new source).

Every tool declares what it does to your files (MCP annotations): read tools are read-only and closed-world, `load_hf_model` is open-world, the three that can destroy something say so. Results carry `structuredContent` alongside the JSON text.

## Prompts and resources

In Claude Desktop, Cursor and VS Code these show up as slash commands. Each is the tool ladder written out in order, with one rule on top: every number in the answer comes from a tool result, never from memory of similar models.

| Prompt | Argument | What it does |
|---|---|---|
| `/review_design` | `focus?` | Structured review: readiness, risks, where the budget lives, the edits worth making with their blast radius. |
| `/pre_train_checklist` | | Pass / fail / unknown per line, each backed by the tool that decided it, before you spend GPU time. |
| `/shrink_for_target` | `target` | Fit "under 100M params" or "a T4 with batch 32": find where the budget lives, propose variants, rank them. |
| `/compare_with_reference` | `architecture?` | The model next to a published one from the library, with the differences that would change training. |
| `/explain_finding` | `rule` | What a finding means for this model, the evidence behind it, the smallest edit that clears it. |

Resources a client can pin as context: `neurarch://model` (the graph), `neurarch://model/mermaid`, `neurarch://model/pytorch`, `neurarch://zoo`, `neurarch://zoo/{id}`, `neurarch://rules` (the provenance table).

## Also a CLI

```bash
npx -y neurarch-mcp lint model.py                 # findings, exit 1 on a block
npx -y neurarch-mcp lint a.py b.py --json         # for CI
npx -y neurarch-mcp check model.py                # the full verdict
npx -y neurarch-mcp check zoo:qwen2.5-7b          # on a reference architecture
```

## Client setup

Use an **absolute** path in any global config: `npx` does not run from your project directory. Relative paths work in project-scoped configs (`.mcp.json`, `.cursor/mcp.json`, `.vscode/mcp.json`).

<details open><summary><b>Claude Code</b></summary>

```bash
claude mcp add neurarch -- npx -y neurarch-mcp /abs/path/to/model.py
```

Or commit a project-scoped `.mcp.json` so every collaborator gets the server:

```json
{ "mcpServers": { "neurarch": { "command": "npx", "args": ["-y", "neurarch-mcp", "./model.py"] } } }
```
</details>

<details><summary><b>Claude Desktop</b></summary>

Download the `.mcpb` from [Releases](https://github.com/neurarch-ai/neurarch-mcp/releases/latest) and open it, or edit the config (**Settings → Developer → Edit Config**; macOS `~/Library/Application Support/Claude/claude_desktop_config.json`, Windows `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{ "mcpServers": { "neurarch": { "command": "npx", "args": ["-y", "neurarch-mcp", "/abs/path/to/model.py"] } } }
```

Fully quit and reopen Claude Desktop; the config is read at startup.
</details>

<details><summary><b>Cursor</b></summary>

Click **Add to Cursor** above, or create `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

```json
{ "mcpServers": { "neurarch": { "command": "npx", "args": ["-y", "neurarch-mcp", "./model.py"] } } }
```
</details>

<details><summary><b>VS Code (Copilot agent mode)</b></summary>

Click **Install in VS Code** above, or create `.vscode/mcp.json` (note the `servers` key):

```json
{ "servers": { "neurarch": { "command": "npx", "args": ["-y", "neurarch-mcp", "${workspaceFolder}/model.py"] } } }
```
</details>

<details><summary><b>Windsurf, Codex, and anything speaking Streamable HTTP</b></summary>

Same `command` + `args` shape; only the file location differs. For HTTP clients, run `npx neurarch-mcp model.py --http` and point the client at:

```json
{ "mcpServers": { "neurarch": { "type": "http", "url": "http://127.0.0.1:8787/mcp" } } }
```
</details>

**Verify:** ask the agent *"List the Neurarch tools you can see."* From a shell, `npx -y neurarch-mcp --help` prints usage and the full tool list.

### Flags

- `--write`: expose the six mutation tools. Off by default.
- `--watch`: reload the model file on change. Pair with the app.
- `--hf`: allow `hf:<org/name>` refs and list `load_hf_model`. The one network switch. `HF_TOKEN` is sent for gated repos; results are cached for a day under `~/.cache/neurarch-mcp`.
- `--tools=core`: advertise eleven tools instead of twenty-five (every tool stays callable by name). Trims what rides along in the agent's context on every turn.
- `--http[=PORT]`, `--host=ADDR`: serve over Streamable HTTP. See [Remote access](#remote-access).
- `--version`, `--help`.

## Network: one switch, off by default

| Switch | What it sends | When |
|---|---|---|
| `--hf` | A request to huggingface.co for a repo's `config.json` (and `HF_TOKEN`, if set). Never your model. | Only on `load_hf_model` or an `hf:` ref. |
| `NEURARCH_REPORT=1` | One anonymous structure+verdict row: structural fingerprint, layer-type histogram, edge count, (rule id, severity) pairs. **Structurally incapable of carrying the graph.** | After each `validate_model`, `lint_model` or `check_design` call, fire and forget, 5s cap. |

With both unset, this server makes **no network calls at all**, and no tool is an exception: the parser, the rule engine, the verifier, the ranker and the reference library are vendored into the package. Your model never leaves the machine. Corpus policy: [neurarch.com/rules.html#data](https://neurarch.com/rules.html#data).

## Remote access

By default the server talks stdio, so the agent and the model file live on the same machine. `--http` serves the same tools over Streamable HTTP, so a hosted or phone-based agent can drive a model running on your machine, for example behind a Cloudflare or Tailscale tunnel.

```bash
npx neurarch-mcp model.py --http                        # loopback, no auth needed
NEURARCH_MCP_TOKEN=$(openssl rand -hex 16) \
  npx neurarch-mcp model.neurarch.json --write --http --host=0.0.0.0
```

Binds to `127.0.0.1` by default with DNS-rebinding protection; `NEURARCH_MCP_TOKEN` requires `Authorization: Bearer <token>` on every request and is **required** before `--write` may bind to a non-loopback host.

**Hosted, with no model on disk:** `npx neurarch-mcp --http --hf` serves a server that answers about whatever each call names (`model_path: "zoo:..."` / `"hf:..."`, or `model_source` with the model text inline). `Dockerfile`, `fly.toml` and [docs/HOSTED.md](./docs/HOSTED.md) carry the deploy. A hosted server sees the model text a client sends it, which the local one never does; say so wherever a URL is published.

## Real shapes from real code: `neurarch-trace`

Static parsing stops at the source. [`python/neurarch-trace`](./python/neurarch-trace) instantiates the model, runs one forward pass with hooks, and writes a `.neurarch.json` with every shape filled in, functional residual adds and concats included:

```bash
pip install neurarch-trace
neurarch-trace models/resnet.py:ResNet18 --input 1,3,224,224 -o resnet18.neurarch.json
neurarch-trace hf:bert-base-uncased --input 1,128 --dtype long          # needs transformers
npx -y neurarch-mcp resnet18.neurarch.json
```

Shapes come out batchless (`[3,224,224]`, never `[1,3,224,224]`), the convention every tool here expects.

## Development

```bash
git clone https://github.com/neurarch-ai/neurarch-mcp && cd neurarch-mcp
npm install
npm run typecheck && npm run build && npm test     # vitest, 240+ tests
node dist/index.js --help
npm run build:mcpb                                 # the Claude Desktop bundle
```

CI runs typecheck, build and test on Node 20 and 22. The package vendors from the main Neurarch repo so that it works with no network, no key and no second install: `src/vendor/engine.bundle.mjs` (registry, PyTorch parser, rule set, code generator, HF config converter) and `src/vendor/verifier.bundle.mjs` (five pipeline stages, provenance table, ranker). Both are generated, contract-tested, and asserted to contain no `import` (and, for the verifier, no `fetch`). `@modelcontextprotocol/sdk` is the only runtime dependency. `zoo/` is synced from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo) with `npm run sync:zoo`.

A new tool is a small, self-contained PR: see [CONTRIBUTING.md](./CONTRIBUTING.md). The agent skill that teaches an evidence-gated edit loop over these tools lives in [`skills/`](./skills/neurarch-skill) (`npx skills add neurarch-ai/neurarch-mcp`).

## Troubleshooting

- **The server never appears in the client.** The model path must be absolute in any global config; `npx` does not run from your project directory. Relative paths only work in project-scoped configs (`.mcp.json`, `.cursor/mcp.json`, `.vscode/mcp.json`).
- **Read tools work but write tools are missing.** You did not pass `--write`. It is off by default so accidental writes can't clobber a file you're editing in the app.
- **`npx` fails on first run.** Node >= 20 is required (`node --version`).
- **Claude Desktop shows nothing after editing the config.** Fully quit and reopen the app; the config is only read at startup.
- **The agent sees a stale graph after you edit in the app.** Add `--watch`, or restart the server.

## What this is not

- **Not a generic codebase indexer.** It reads model definitions (`.py`, `.neurarch.json`, `zoo:`, `hf:`), not your whole tree. For codebase structure, use [GitNexus](https://github.com/abhigyanpatwari/GitNexus) or similar.
- **Not a trainer.** It tells you whether a run would start, what it would cost and where the result could be served; it never spends your GPU time. `check_design` says when a decision is yours.
- **Not connected to your Neurarch workspace.** It reads files. Live editing happens in the [Neurarch app](https://neurarch.com); `--watch` follows its saves.

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

The package vendors from the main Neurarch repo, and everything is vendored for the same reason: this server has to work with no network, no API key and no second install step.

- `src/lib/`: pure-TypeScript utilities (model types, parameter and FLOP estimators, impact analyzer), maintained as source here.
- `src/vendor/engine.bundle.mjs`: the compiled Neurarch engine: the component registry, the PyTorch parser behind `.py` support, and the rule set behind `lint_model`. Generated, never hand-edited; the header says how to regenerate it, and `src/vendor/engine.contract.test.ts` fails if its exports drift or it ever acquires an import.
- `src/vendor/verifier.bundle.mjs`: the compiled Neurarch verifier behind `check_design`: the five pipeline stages, plus the rule-provenance table. Same code path as the app and the hosted endpoint, so an agent here and a person in the app get the same answer. Same rules: generated, contract-tested (`verifier.contract.test.ts`), and asserted to contain no import and no `fetch`.

Neither adds a runtime dependency: `@modelcontextprotocol/sdk` is still the only one.

## License

MIT. See [LICENSE](./LICENSE).

## Links

- [Neurarch](https://neurarch.com): the visual neural-network editor that produces the model files this server reads.
- [Model Context Protocol](https://modelcontextprotocol.io): the spec this server implements.
- [npm](https://www.npmjs.com/package/neurarch-mcp): package page.
