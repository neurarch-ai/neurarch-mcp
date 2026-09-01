---
name: neurarch-skill
description: Run an evidence-gated edit loop over a Neurarch model graph through the neurarch-mcp MCP server. Use when an agent needs to inspect, explain, audit, extend, refactor, or debug a neural-network architecture stored as a .neurarch.json graph; summarize a model; locate parameter or compute hotspots; trace dependencies and blast radius before a change; diff two model versions; add, modify, rewire, or delete layers and connections; validate structural invariants; or persist an edited graph, without guessing shapes, parameter counts, layer names, or layer vocabulary from training code.
---

# Neurarch Graph Loop

Use the Neurarch model graph as the single source of truth for a network's structure. Move every architecture question or edit through one bounded, evidence-gated pass using only the neurarch-mcp tools exposed by the host. Numbers come from tools, not from the model's memory of similar architectures.

## Read the references

- Read [references/neurarch-mcp-tools.md](references/neurarch-mcp-tools.md) before the first neurarch-mcp operation in a session and whenever a required capability or payload is unclear.
- Read [references/graph-conventions.md](references/graph-conventions.md) before creating a layer or substantially rewiring a graph.

## Operating contract

1. Use only the connected neurarch-mcp server for graph reads and writes. Do not hand-edit the `.neurarch.json` file while the server is connected: the server holds an in-memory copy, and direct file edits either get overwritten by `save_model` or silently diverge from what the tools report.
2. Treat the connected tool schemas as authoritative. MCP hosts may namespace tool names; match the terminal tool name and semantics rather than assuming a prefix.
3. Discover the model's vocabulary at runtime: layer types, layer names, scopes, blocks, and hyperparameters come from the live graph, never from another model, a template, or memory of a similar architecture.
4. Frame one authorized edit at a time, including the evidence that will prove it landed and the condition that would stop the loop.
5. Make the smallest mutation that can complete that edit, then re-read the affected part of the graph.
6. Treat a write tool's success response as a claim. Only a follow-up read (`get_layer`, `list_connections`, `validate_model`) confirms the intended state.
7. Run `validate_model` before the first edit and record the result. Pre-existing issues belong to the baseline; only new issues belong to your change.
8. Call `save_model` after verified mutations. In-memory edits are lost on server restart, and with `--watch` an external save from the Neurarch app overwrites them.
9. Preserve the model's existing naming style, scope structure, and design notes unless the user explicitly asks to change them.

## Loop model

Run every operation as:

**Discover, Frame, Act, Verify, Persist, Continue or stop**

- **Discover:** rebuild current graph state from tools, not memory.
- **Frame:** select one bounded edit, its blast radius, and proof of success.
- **Act:** perform the minimum mutation in dependency order.
- **Verify:** compare fresh reads with the intended state.
- **Persist:** save the graph and report evidence-backed deltas.
- **Continue or stop:** route the next authorized edit, or stop at done, ambiguity, missing capability, validation regressions, or a human gate.

Do not collapse Act and Verify into one step. A successful `add_layer` response proves only that the server accepted the call, not that the wiring, shapes, or parameter totals ended up as intended.

## Core workflow

### 1. Discover the model and capabilities

Start every session (and every return to the loop after an external change) with:

1. `describe_architecture` for orientation: pipeline order, depth, IO shapes, total params and MACs, hotspots, validation rollup.
2. `validate_model` for the structural baseline: cycles, dangling refs, duplicate ids or names, orphans.
3. `get_layer` on every layer the task names, and `find_layers` when the user's wording does not match one layer exactly.

If write tools (`add_layer`, `modify_layer`, `add_connection`, `delete_layer`, `delete_connection`, `save_model`) are absent, the server was started read-only. Stop before any planned mutation and tell the user to restart with `--write`. Do not bypass the boundary by editing the JSON file.

If no neurarch-mcp server is connected, enter the bootstrap gate:

1. Detect the current MCP host and its supported configuration mechanism.
2. Configure `npx -y neurarch-mcp /abs/path/to/model.neurarch.json` as a stdio server (add `--write` only when the user wants edits, `--watch` when they are also editing in the Neurarch app).
3. Obtain any host-required approval before changing user or global configuration.
4. Reload tool discovery, verify the read tools answer against the intended file, then resume at Discover.

If the user has no `.neurarch.json`, they can export one from the Neurarch app (File, Save) or start from the runnable examples in the neurarch-mcp repository. Do not fabricate a graph file by hand.

### 2. Frame the next bounded edit

Define:

- current observed state (from Discover, not assumption);
- one requested or clearly authorized edit;
- the exact mutations needed, in dependency order;
- evidence that will prove success (expected param delta, expected new edges, validation staying clean);
- conditions that require stopping or asking.

Before any destructive or shape-relevant edit, call `layer_impact` on the target set and surface the blast radius to the user: how many downstream layers, which are shape-sensitive, which carry weights. For a set of layers, enumerate them with `find_layers` first and impact-check the matched set, not a guessed list of names.

Ask one concise question before mutating when:

- a name the user gave matches zero or multiple layers;
- the requested edit implies deleting weight-carrying layers the user did not name;
- the baseline `validate_model` already reports issues that the edit would touch.

### 3. Act once

Perform related mutations in dependency order: create a layer before wiring it, rewire survivors before deleting a bypassed layer, delete connections only when the server will not already remove them as part of `delete_layer`. Change only what the framed edit requires. Do not opportunistically rename, re-scope, or "clean up" untouched layers.

For mass edits ("delete every conv layer", "double every FFN width"), enumerate the exact target set with `find_layers`, state the count to the user, then act on that enumerated set. Never act on a pattern you have not expanded and read back.

### 4. Verify from fresh state

After each mutation:

1. `get_layer` on every created or modified layer; `list_connections` (filtered) on every rewired region.
2. `validate_model` again. Compare against the baseline: any new issue is yours to fix or to report before continuing.
3. When the edit was sized in parameters or compute, re-read `param_count_by_block` or `describe_architecture` and check the delta against the framed expectation. A 2x-off delta means the edit did not do what was framed; stop and re-discover.

If a multi-step edit partially fails, re-read the involved region and recover only the missing mutation. Retry only with new evidence or a changed payload; never replay an unchanged failing call.

### 5. Persist and report

1. `save_model` once the verified state matches the frame.
2. Report what changed with tool-sourced numbers: layers added or removed, edges changed, param and MAC deltas, validation status. Never report a number the tools did not produce.
3. If the session also touches training code, `diff_models` against the previous saved file is the honest changelog to cite in a commit or PR description.

There is no write tool for design notes. When a decision deserves durable context (why a layer was removed, why a width was chosen), say so explicitly in the report so the user can pin it in the Neurarch app; do not pretend it was persisted.

## Graph rules

### Names and vocabulary

Use the model's existing names and scope prefixes. When creating layers, follow the dominant naming pattern already in the graph (discover it with `find_layers`, do not assume `snake_case` or any other style). Never reuse an existing name.

### Structural components are nodes

Positional encodings, attention variants, normalization, and merge points are independent graph nodes, not parameters folded into an attention layer. When the user asks to "switch to RoPE" or "add ALiBi", frame it as node-level edits (add or replace the encoding node and rewire), not as a `modify_layer` on the attention node.

### Shape honesty

`validate_model` checks structural invariants, not full shape propagation. After edits that change dimensions, read the affected layers' shapes with `get_layer` and reason explicitly; if a downstream shape cannot be confirmed from tool output, say so instead of asserting compatibility. The Neurarch app and neurarch-lint are the deeper shape and lint gates; recommend them for final verification of dimension-sensitive changes.

### Frozen and augmented layers

`find_layers` can filter by augmentation (for example frozen layers). Before edits that assume trainability, check; do not silently modify or delete frozen layers without surfacing it.

## Failure handling

- On an unknown layer name, re-run `find_layers` with a broader pattern before concluding the layer is absent.
- On a write tool error, re-read the involved layers and connections before retrying; the graph may have partially changed.
- On a validation regression you cannot repair with one bounded follow-up edit, stop, report the baseline result, the new issues, and the exact mutations applied since the last `save_model`.
- On `--watch` conflicts (tool output suddenly disagrees with your last verified read), assume an external save from the app won. Re-discover from step 1; never overwrite the user's newer graph with your older intent.
- On a missing capability (no write tools, no such tool on this server version), stop with the exact flag or upgrade the user needs. Never emulate a missing tool by editing files.
