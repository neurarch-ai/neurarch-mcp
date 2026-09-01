# neurarch-mcp tool map

Capability map for the official [neurarch-mcp](https://github.com/neurarch-ai/neurarch-mcp) server. The connected server's tool schemas are authoritative for exact payloads; this file tells you which tool answers which question and in what order to prefer them. MCP hosts may prefix tool names (for example `mcp__neurarch__get_layer`); match the terminal name.

## Orientation (call these first)

| Tool | Use for |
|:--|:--|
| `describe_architecture` | One-call orientation: topo-ordered pipeline, depth, IO shapes, total params and MACs, top param and compute hotspots, validation rollup. Prefer this over chaining four smaller reads. |
| `get_model_summary` | Lighter overview when you only need counts and IO shapes. |
| `validate_model` | Structural invariants: cycles, dangling connection refs, duplicate ids or names, orphan layers. Run once as a baseline before any edit and again after every mutation. |

## Targeted reads

| Tool | Use for |
|:--|:--|
| `get_layer` | Full definition of one layer by name: params, shapes, notes, upstream and downstream ids. The read-back tool after any mutation. |
| `find_layers` | Search by type, name regex, scope prefix, or augmentation (for example frozen); optionally rank by parameter count. The only correct way to expand "every conv layer" into an explicit target list. |
| `compare_layers` | Structural diff of two layers: type match, param-count delta, shape match, exact differing param keys. |
| `list_connections` | Flat edge list with optional `from` and `to` filters. Read-back tool for rewiring edits. |
| `find_path` | Shortest directed path between two layers, `null` when unreachable. Use to confirm or refute "does A feed B". |
| `layer_impact` | Blast radius of changing a layer or matched set: downstream layers, which are shape-sensitive, which carry weights. Mandatory before destructive or dimension-changing edits. |

## Aggregates and structure

| Tool | Use for |
|:--|:--|
| `param_count_by_block` | Parameter counts grouped by block, scope, or type. Before/after reads of this are the evidence for any "shrink/grow the model" claim. |
| `flops_by_block` | MAC counts (FLOPs divided by 2) with the same grouping. |
| `list_blocks` | Collapsed groups (or scope-derived blocks): members, params, FLOPs. |
| `get_block` | One block in depth: per-layer params and FLOPs, totals, and the edges crossing the block boundary. |
| `mermaid_diagram` | Mermaid `flowchart TD` of the graph; groups render as subgraphs; truncates past 60 layers keeping the topological head. For human-facing explanation, not for state discovery. |
| `diff_models` | Structural diff of the current model against another `.neurarch.json` file: layers added, removed, modified (field-level) and connection changes. The honest changelog between saved versions. |

## Context reads

| Tool | Use for |
|:--|:--|
| `list_hyperparams` | Model-level hyperparameters the user set in the app (learning rate, batch size, ...). |
| `get_design_notes` | Pinned design rationale (agent, advisor, manual notes), optionally filtered by layer. Read before proposing to undo something a note explains. |

## Write tools (present only when the server runs with `--write`)

| Tool | Use for |
|:--|:--|
| `add_layer` | Insert a new layer, optionally auto-wired downstream of an existing one. Prefer the auto-wire form over a separate `add_connection` when inserting into a pipeline. |
| `modify_layer` | Shallow-merge params, rename, or change scope. Returns a before/after diff; still confirm with `get_layer`. |
| `add_connection` | Wire two existing layers. Fails on self-loops and duplicate edges. |
| `delete_layer` | Remove a layer and every connection touching it. Invalidates downstream cached shapes. |
| `delete_connection` | Remove a single directed edge. Invalidates the target's cached shape. |
| `save_model` | Persist the in-memory model to disk. Required after any mutation you intend to keep; nothing is durable without it. |

If these six tools are absent, the server is read-only. Stop before mutating and ask the user to restart with `--write`; do not edit the JSON file directly.

## Server flags that change your world

- `--write`: exposes the write tools. Off by default.
- `--watch`: the server reloads the file when the Neurarch app (or anything else) saves it. An external save overwrites unsaved in-memory edits made through the write tools. If reads suddenly contradict your last verified state, assume a reload happened and re-discover.
- `--http[=PORT]` and `--host`: same tools over Streamable HTTP instead of stdio. Irrelevant to tool semantics; connection setup follows the neurarch-mcp README.

## Added in neurarch-mcp 0.13

| Question | Tool | Notes |
|:--|:--|:--|
| Which of these candidate designs deserves the training budget? | `rank_designs` | Blocked candidates last; ties stay ties; `recommended` is null when nothing measured separates the top. Read `calibration` before quoting the order. |
| Give me the graph as code | `export_pytorch` | Returns source; `save_to` only under `--write`, never over the file the server was started from. |
| What published architecture is this closest to? | `list_architectures`, `load_architecture` | 81 bundled graphs; then `model_path: "zoo:<id>"` on any read tool. |
| What does this Hugging Face model look like? | `load_hf_model` | Listed only under `--hf`; the one network call. `configSource: "fallback"` means a family template, not the real dimensions. |
| What in this repository is a model? | `find_models` | Distinguishes files the parser reads from ones that need `neurarch-trace`. |

Any read tool also accepts `model_source` (inline `.neurarch.json` or PyTorch
text) in place of `model_path`. Prompts (`review_design`, `pre_train_checklist`,
`shrink_for_target`, `compare_with_reference`, `explain_finding`) are the same
loop, pre-written; prefer them when the client shows them.
