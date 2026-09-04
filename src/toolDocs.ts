/**
 * What a tool costs before it is ever called: its description rides along in
 * ListTools on every turn of every conversation the server is attached to.
 * Measured on 0.13: ~6.9k tokens for the full set, with rank_designs alone at
 * 859 characters. That is a tax paid whether or not the tool is used.
 *
 * So every tool has two texts. `SHORT` is what ListTools sends: one or two
 * sentences on when to reach for it. The long form each tool declares as its
 * `description` becomes documentation, served as `neurarch://docs/<tool>` for
 * an agent that wants the contract before calling. Nothing is lost; it moves
 * from "always in context" to "one read away".
 */
export const SHORT: Record<string, string> = {
  plan: 'Will it run, what it costs, which GPUs it fits, which of your .neurarch.yml rules it breaks, and what this structure scored last time it trained here. Sends the graph to /api/v1/plan. Start here.',
  history: 'What this exact structure scored last time it trained in your organisation, by fingerprint. Sends the fingerprint, never the graph. Needs NEURARCH_API_KEY.',
  get_model_summary: 'Layer count, total params, dominant layer types, input/output shape. Cheapest orientation.',
  describe_architecture: 'One-call orientation: pipeline order, depth, IO shapes, params and MACs, top hotspots, validation. Start here.',
  get_layer: 'Everything about one layer by name: type, params, shapes, upstream and downstream.',
  find_layers: 'Search layers by type, name regex, scope prefix or augmentation; optionally rank by params.',
  compare_layers: 'Structural diff of two layers: type, param delta, shape match, differing keys.',
  layer_impact: 'Blast radius of changing a layer or matched set: which downstream layers are shape-sensitive or carry weights. Call before proposing an edit.',
  param_count_by_block: 'Parameter counts grouped by block, scope or type.',
  flops_by_block: 'MAC counts grouped by block, scope or type.',
  mermaid_diagram: 'The model as a Mermaid flowchart (groups as subgraphs; truncates past 60 layers).',
  list_blocks: 'The collapsed groups (or scope-derived blocks): members, params, FLOPs.',
  get_block: 'One block in detail: per-layer params and FLOPs, and the edges crossing its boundary.',
  diff_models: 'Structural diff against another .neurarch.json: layers added, removed, modified; connection changes.',
  validate_model: 'Is this a well-formed graph: cycles, dangling refs, duplicate ids or names, orphans. First rung of the ladder.',
  lint_model: 'The structural design rules (head-dim and GQA divisibility, norm/activation order, dropout ranges, missing residuals) with provenance for the measured ones. Second rung. Offline.',
  check_design: 'The full verdict: will it train, what it costs, where it can run, and which decisions are still the human\'s. Third rung. Offline, ~10ms. verbose:true adds every stage\'s data.',
  find_path: 'Shortest directed path between two layers, or null.',
  list_connections: 'The edge list, optionally filtered by from/to.',
  list_hyperparams: 'Model-level hyperparameters the user set in the app.',
  get_design_notes: 'Pinned design rationale from the app, optionally per layer.',
  rank_designs: 'Order candidate designs for a training budget: blocked (will not forward-pass) last, legal ones only by outcome-backed rules, ties stay ties, calibration in every result.',
  export_pytorch: 'The graph as a runnable nn.Module. save_to needs --write and never targets the source file.',
  list_architectures: 'Search 81 bundled reference architectures (DeepSeek-V3, Qwen2.5, Llama, Whisper, CLIP, BERT, ViT, ResNet...). Then model_path "zoo:<id>" on any tool.',
  load_architecture: 'Open one reference architecture and describe it; save_to (with --write) writes it as .neurarch.json.',
  load_hf_model: 'A Hugging Face repo as a graph from its config.json (the one network call; listed only under --hf). Says whether it used the real config or a family template.',
  find_models: 'Walk a directory for nn.Module definitions and saved graphs; says which parse fully, partially, or need a runtime trace.',
  suggest_fix: 'Lint findings as unified diffs against the .py: exact where a number or order is pinned, proposal where a layer is missing. Apply, then lint again.',
  trace_model: 'Run the model once in Python with hooks (neurarch-trace) for a graph with real shapes; needs the input dims. The way in for a real repository: the static parser is recognisable on 41% of real files.',
  add_layer: 'Insert a new layer, optionally wired after an existing one.',
  modify_layer: 'Shallow-merge params, rename, or re-scope a layer. Returns a before/after diff.',
  add_connection: 'Wire two existing layers.',
  delete_layer: 'Remove a layer and every connection touching it.',
  delete_connection: 'Remove one directed edge.',
  save_model: 'Persist the in-memory model to disk. Call after any mutation.',
};

export const TITLES: Record<string, string> = {
  plan: 'Plan this design',
  history: 'Training history',
  get_model_summary: 'Model summary',
  describe_architecture: 'Describe architecture',
  get_layer: 'Get layer',
  find_layers: 'Find layers',
  compare_layers: 'Compare layers',
  layer_impact: 'Layer impact',
  param_count_by_block: 'Params by block',
  flops_by_block: 'FLOPs by block',
  mermaid_diagram: 'Mermaid diagram',
  list_blocks: 'List blocks',
  get_block: 'Get block',
  diff_models: 'Diff models',
  validate_model: 'Validate model',
  lint_model: 'Lint model',
  check_design: 'Check design',
  find_path: 'Find path',
  list_connections: 'List connections',
  list_hyperparams: 'List hyperparams',
  get_design_notes: 'Design notes',
  rank_designs: 'Rank designs',
  export_pytorch: 'Export PyTorch',
  list_architectures: 'List reference architectures',
  load_architecture: 'Load reference architecture',
  load_hf_model: 'Load Hugging Face model',
  find_models: 'Find models',
  suggest_fix: 'Suggest fix',
  trace_model: 'Trace model',
  add_layer: 'Add layer',
  modify_layer: 'Modify layer',
  add_connection: 'Add connection',
  delete_layer: 'Delete layer',
  delete_connection: 'Delete connection',
  save_model: 'Save model',
};

export function shortDescription(name: string, fallback: string): string {
  return SHORT[name] ?? fallback;
}
