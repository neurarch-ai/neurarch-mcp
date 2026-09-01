# Neurarch graph conventions

Follow the open model's existing conventions first. Use the rules below only where the graph itself does not already answer the question. Everything here is discoverable at runtime; nothing should be hardcoded from this file into a claim about a specific model.

## The graph is nodes and directed edges, nothing hidden

A `.neurarch.json` model is a flat set of layers (nodes) plus directed connections (edges). There is no implicit sequencing: if two layers are not connected, nothing flows between them, whatever their names suggest. `describe_architecture` reports the topological pipeline; trust it over name order.

## Structural components are independent nodes

Positional encodings (learned, sinusoidal, RoPE, ALiBi), attention variants, normalization layers, activations, and merge points (add, concat) are each their own node. They are not parameters folded into an attention layer.

Consequences:

- "Switch to RoPE" means replace or insert an encoding node and rewire, not `modify_layer` on the attention node.
- "Add a residual connection" means an explicit merge node plus two incoming edges, not an attribute.
- When comparing to a paper, expect the graph to be more explicit than the paper's block diagram.

## Scopes and blocks

Layers may carry scope prefixes (for example `block_0/attn`) and the app may define collapsed groups. `list_blocks`, `get_block`, `param_count_by_block`, and `flops_by_block` aggregate along these. When creating layers inside an existing block, keep the scope prefix consistent with the block's members; discover the pattern with `find_layers` on the scope.

## Naming

- Never reuse an existing layer name; ids and names must stay unique (a `validate_model` invariant).
- Follow the dominant style already in the graph. Discover it; do not assume a convention.
- When inserting a numbered sibling (`block_2` after `block_1`), read the existing siblings first so numbering, scope, and params stay parallel.

## Editing etiquette

- Insert before you wire, and wire before you delete the bypassed path, so the graph never passes through a state with orphaned live layers.
- `delete_layer` removes all touching edges; plan replacement wiring before deleting, then re-wire survivors immediately after.
- Weight-carrying layers deserve extra care: deleting one discards trained state conceptually, even in a pure structure file. Surface it, then act.
- Frozen or otherwise augmented layers (discoverable via `find_layers`) usually encode a deliberate training decision; do not modify them silently.

## Evidence vocabulary for reports

When reporting a finished edit, cite tool-sourced numbers only, in this shape:

```
Edit: <what changed, layer and edge level>
Params: <before> -> <after> (<delta>), from param_count_by_block
Compute: <before> -> <after> GMACs, from flops_by_block or describe_architecture
Validation: clean | baseline issues unchanged (<list>) | NEW issues (<list>)
Saved: yes (save_model) | NO, in-memory only
```

Never write "params roughly halved" when a before/after read is one tool call away, and never claim "saved" without having called `save_model`.
