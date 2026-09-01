/**
 * MCP prompts: the recipes a person can pick from a menu.
 *
 * A tool is something the agent decides to call. A prompt is something the
 * user asks for by name, and in Claude Desktop, Cursor and VS Code it shows up
 * as a slash command, which makes it the one part of this server a person can
 * discover without reading anything. So each prompt is written as a procedure
 * over the tools, in the order the tools are meant to be climbed, and says
 * what NOT to do (report numbers from memory, propose an edit before checking
 * its blast radius, call the verdict "done" without a read-back).
 */
export interface PromptArg { name: string; description: string; required?: boolean }

export interface PromptDef {
  name: string;
  title: string;
  description: string;
  arguments?: PromptArg[];
  /** Renders the user message. Argument values are already validated for presence. */
  render: (args: Record<string, string>, ctx: { writeEnabled: boolean }) => string;
}

const GROUND_RULES = `Ground rules for this session:
- Every number you state (parameters, FLOPs, shapes, layer counts) must come from a tool result in this conversation, never from memory of similar models.
- Climb the ladder in order and stop at the first rung that fails: validate_model (is it a graph), lint_model (does it break a design rule), check_design (will it train, what will it cost, where can it run).
- Before recommending any change to a layer, call layer_impact on it and say which downstream layers are shape-sensitive or carry weights.
- Quote the provenance lint_model returns for a rule when you cite it; a rule with no measurement behind it is a convention, and say so.`;

export const PROMPTS: PromptDef[] = [
  {
    name: 'review_design',
    title: 'Review this design',
    description: 'A structured design review of the current model: readiness, risks, parameter budget, and the edits worth making, every number from the tools.',
    arguments: [{ name: 'focus', description: 'Optional: what to review for, e.g. "training stability", "inference cost on a T4", "is the attention configured right".' }],
    render: ({ focus }) => `Review the model this server is attached to.${focus ? ` Focus: ${focus}.` : ''}

${GROUND_RULES}

Procedure:
1. describe_architecture, then validate_model.
2. lint_model. For every block or warn finding, explain what it means for this model specifically, and quote the provenance where there is one.
3. check_design. Report the verdict, the training cost and time estimate, the best deployment target and its latency, and any decision it says is still the human's.
4. Where the parameter or compute budget is concentrated (from describe_architecture's hotspots), say whether that is where it should be for this kind of model, and name the single edit with the best ratio of effect to blast radius (check it with layer_impact).
5. End with: a verdict in one sentence, then a numbered list of edits in priority order, each with its expected parameter delta and the downstream layers it touches.`,
  },
  {
    name: 'pre_train_checklist',
    title: 'Pre-training checklist',
    description: 'The checks worth running before spending GPU time: structural, design rules, cost, GPU fit, and what is still unknown about the graph.',
    render: () => `Run the pre-training checklist for the model this server is attached to.

${GROUND_RULES}

Produce a checklist with a pass / fail / unknown mark per line, each line backed by the tool that decided it:
- Graph is well-formed (validate_model)
- No blocking design rule (lint_model, severity block)
- No warning worth fixing first (lint_model, severity warn; list them)
- Forward pass will run (check_design preflight stage)
- Estimated parameters and training cost (check_design train stage)
- Fits the intended GPU (check_design; if the stage reports the smallest GPU it fits, say which)
- Shapes are known end to end (get_model_summary: if the model came from Python source, shapes are unknown and this line is "unknown", with the fix: trace it with neurarch-trace or export from the app)
- Decisions still owed by the human (check_design.decision)
Then say in one sentence whether you would start the run, and what you would change first if not.`,
  },
  {
    name: 'shrink_for_target',
    title: 'Shrink to fit a target',
    description: 'Find the edits that bring the model under a parameter, memory, latency or GPU budget with the least damage, and rank the resulting variants.',
    arguments: [{ name: 'target', description: 'The budget, e.g. "under 100M params", "fits a T4 with batch 32", "half the FLOPs", "latency under 50ms on CPU".', required: true }],
    render: ({ target }, { writeEnabled }) => `Bring the model this server is attached to within this budget: ${target}.

${GROUND_RULES}

Procedure:
1. describe_architecture and param_count_by_block / flops_by_block to find where the budget actually lives. Do not guess at it.
2. Propose at most three concrete variants (for example: fewer blocks, narrower embedding, fewer heads with the same head_dim, a lighter head). For each, call layer_impact on the layers it changes and state the blast radius.
3. ${writeEnabled
        ? 'Apply each variant with the write tools in turn, lint_model and check_design it, and record its parameter count and cost. Restore the original between variants (delete/re-add or modify_layer back). Do not save_model until the user picks one.'
        : 'This server is read-only, so build each variant as an inline graph (copy the current graph from the neurarch://model resource, apply the edit in the JSON) and pass them to rank_designs as candidates with include_current: true.'}
4. Call rank_designs over the variants. Report its ordering, and read the calibration it returns out loud: the ranking separates designs that will not run from those that will; it does not predict which legal design trains best.
5. Recommend one variant, with its parameter delta, cost delta and what it gives up. If rank_designs returned a tie, say so and choose on the budget the user gave, not on a preference the tools did not express.`,
  },
  {
    name: 'compare_with_reference',
    title: 'Compare with a reference architecture',
    description: 'Put the current model next to a published one from the bundled library and explain the structural differences that matter.',
    arguments: [{ name: 'architecture', description: 'A library id from list_architectures, e.g. "qwen2.5-7b", "llama-3-8b", "bert-base", "resnet-50". Leave blank to have the closest one picked.', required: false }],
    render: ({ architecture }) => `Compare the model this server is attached to with ${architecture ? `the reference architecture "${architecture}"` : 'the closest reference architecture in the bundled library'}.

${GROUND_RULES}

Procedure:
1. describe_architecture on the current model.
2. ${architecture
        ? `load_architecture with id "${architecture}".`
        : 'list_architectures (filter by the current model\'s domain or attention type), pick the closest match, and say why you picked it.'} Then describe_architecture with model_path "zoo:<id>".
3. Compare, with numbers from both results: depth, width (embedding or channel dims), attention configuration (heads, KV heads, head_dim), normalisation placement, parameter distribution between embedding / blocks / head, and total parameters.
4. lint_model on both. If a rule fires on the user's model and not on the reference, that is the first thing to report.
5. End with the two or three differences that would change training behaviour, and for each whether it looks deliberate or like an oversight.`,
  },
  {
    name: 'explain_finding',
    title: 'Explain a finding',
    description: 'What a lint rule or check_design finding means for this model, the evidence behind it, and the smallest edit that clears it.',
    arguments: [{ name: 'rule', description: 'The rule id or finding title, e.g. "head-dim-divisibility".', required: true }],
    render: ({ rule }) => `Explain the finding "${rule}" on the model this server is attached to.

${GROUND_RULES}

1. lint_model (and check_design if the finding is not a lint rule) to get the finding as it fires on this model, with the layer it names.
2. Say what the rule checks, in terms of this layer's actual parameters (get_layer). Quote the provenance if lint_model returned one for it; if it returned none, say the rule is a convention without a published measurement behind it.
3. Name the smallest edit that clears it, run layer_impact on that layer, and state the blast radius.
4. If the finding is a warning rather than a block, say plainly whether it is worth acting on for this model.`,
  },
];

export function renderPrompt(prompt: PromptDef, args: Record<string, string>, ctx: { writeEnabled: boolean }): string {
  return prompt.render(args, ctx);
}
