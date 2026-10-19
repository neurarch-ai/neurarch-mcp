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
- Every number you state (parameters, FLOPs, shapes, layer counts, memory) must come from a tool result in this conversation, never from memory of similar models.
- Before any training job is launched, size it with size_training_job (batch, card, attention) and quote its verdict; its memory number is a calibrated estimate, so say so.
- Time and memory figures from check_parallel_layout and check_serving_config are predictions or upper bounds; keep the label the tool gives them. None of these tools predicts accuracy or quality.
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
    description: 'The checks worth running before spending GPU time: memory and GPU fit at your batch, cost, what this structure measured last time, then whether it forward-passes.',
    render: () => `Run the pre-training checklist for the model this server is attached to.

${GROUND_RULES}

Produce a checklist with a pass / fail / unknown mark per line, each line backed by the tool that decided it:
- Fits the card the job will reserve at its batch (size_training_job: quote the verdict, the peak and its parts, the max batch on that card, and the smallest card that holds it; ask for the batch and card if you do not know them)
- One run's cost (size_training_job cost, with its assumptions)
- What this structure measured the last time it trained (history, with the fingerprint size_training_job returned; "not read" if there is no key)
- Graph is well-formed (validate_model)
- No blocking design rule (lint_model, severity block)
- No warning worth fixing first (lint_model, severity warn; list them)
- Forward pass will run (check_design preflight stage)
- Estimated parameters (check_design train stage)
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
1. describe_architecture and param_count_by_block / flops_by_block to find where the budget actually lives. Do not guess at it. If the budget is memory or a card, size_training_job gives the peak, its parts and the max batch per card first: a smaller batch or SDPA attention may meet it with no edit at all.
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
  {
    name: 'precheck_change',
    title: 'Pre-check a model change',
    description: 'Before a changed model takes a GPU or traffic: what changed against the base, what the job now needs, and whether the launch it is headed for is legal. One go / no-go table, every line from a tool.',
    arguments: [
      { name: 'base', description: 'The model before the change: a path, zoo:<id> or hf:<org/name>, e.g. "hf:Qwen/Qwen3-8B".', required: true },
      { name: 'launch', description: 'Optional: where it is going, e.g. "train on 8xH100 at batch 16", "tp 4 pp 2 micro 1 seq 4096", or a vllm serve line.' },
    ],
    render: ({ base, launch }) => `Pre-check the change from ${base} to the model this server is attached to${launch ? `, headed for: ${launch}` : ''}.

${GROUND_RULES}

Procedure:
1. check_edit with base_model_path "${base}" (and the batch and card if the launch names them). List every block finding first: these load, run and are wrong, so nothing later will catch them.
2. size_training_job on the changed model at the launch's batch and card, and say whether the change is what moved the job on or off its card.
3. ${launch ? 'If the launch is a multi-GPU training split, check_parallel_layout with its tp / pp / dp / micro_batch / seq_len. If it is a serving launch, check_serving_config with the command and against "' + base + '".' : 'If the user names a multi-GPU split or a serving launch, run check_parallel_layout or check_serving_config on it; otherwise say these were not checked.'}
4. If ${base} is a Hugging Face id, model_changes on it, so a change upstream is not mistaken for the user's own.
5. End with a table: check, verdict (pass / fail / not checked), the tool that decided it. Then one sentence: go or no-go, and the first thing to fix if no-go. Label every memory and time figure as the tool labels it (estimate, prediction, upper bound).`,
  },
  {
    name: 'precheck_serving',
    title: 'Pre-check a serving launch',
    description: 'Before a vLLM, SGLang or llama.cpp launch: will it start, how many full-length sequences fit, does it serve the model as published, and has the model moved upstream.',
    arguments: [
      { name: 'command', description: 'The launch line, e.g. "vllm serve Qwen/Qwen3-8B -tp 4 --max-model-len 32768".', required: true },
      { name: 'gpu', description: 'The card, e.g. "h100", "l4", "a100-80g".' },
    ],
    render: ({ command, gpu }) => `Pre-check this serving launch: ${command}${gpu ? ` on ${gpu}` : ''}.

${GROUND_RULES}

Procedure:
1. check_serving_config with command set to the launch line${gpu ? ` and gpu "${gpu}"` : ' (ask for the card and GPU count if the line does not say)'}. Report block findings first, then the memory line, quoting "upper bound" where the result does.
2. Read \`skipped\` and \`caveats\` out loud: a check that did not run is not a pass.
3. model_changes on the model the command serves. If the launch has no --revision, say which recorded changes it would pick up, and which of them are breaking.
4. End with: will it start (yes / no / not checked), will it serve what was asked (context, concurrency), and the one flag to change first.`,
  },
  {
    name: 'pin_downloads',
    title: 'Pin Hugging Face downloads',
    description: 'Find every Hugging Face download in this repository that is not pinned to a commit, and what each model has changed upstream, so the pins can be chosen rather than guessed.',
    arguments: [{ name: 'paths', description: 'Optional: directories or files to scan, comma separated. Default: the repository.' }],
    render: ({ paths }) => `Find the Hugging Face downloads in ${paths ? paths : 'this repository'} that are not pinned to a commit.

${GROUND_RULES}

Procedure:
1. check_pins${paths ? ` with paths ${JSON.stringify(paths.split(',').map(x => x.trim()).filter(Boolean))}` : ''}. Page through nextCursor until it is null.
2. For each distinct model id it names, model_changes. Report how many recorded changes there are and how many are breaking, and the commit main points at now.
3. For each finding, propose the edit the finding's fix spells for that file. Do not invent a commit sha: use one model_changes returned, or leave <sha> for the user.
4. Say which models are not in the change record: no record is not the same as no change.`,
  },
];

export function renderPrompt(prompt: PromptDef, args: Record<string, string>, ctx: { writeEnabled: boolean }): string {
  return prompt.render(args, ctx);
}
