# AGENTS.md

For the agent, not the reader. If you are a coding agent that has landed in this
repository or installed this package, this is the short version of what it gives
you and, more usefully, of when its answers are worth trusting.

## What this is

Two shipped artifacts from one repo:

- **`neurarch-mcp`** (npm, Node 20+): an MCP server that turns a PyTorch model
  into a structured graph and answers questions about it with computation
  instead of inference, first of all what a training job on it needs. Every
  tool but `plan` and `history` runs locally, needs no key and no account, and
  opens no socket.
- **`neurarch-trace`** (PyPI, torch >= 2.0): runs one forward pass and writes
  the graph with real per-layer shapes.

## Before you launch a training job

**Call `size_training_job` before you submit or launch any training job**, with
the batch size and the card you are about to reserve:

```
size_training_job { batch_size: 32, gpu: "A100-40" }            # the server's model
size_training_job { model_path: "Model.neurarch.json", batch_size: 256, gpu: "8xH100" }
size_training_job { hf_id: "Qwen/Qwen2.5-0.5B", batch_size: 8, seq_len: 4096 }   # needs --hf
```

It returns the training peak at that batch (weights and optimizer state,
activations, stored attention scores), the smallest card that holds it, the
largest batch each card holds, a two-card split when one card does not, one
run's cost, and a verdict line you can quote: `fits an A100-40 with 22%
headroom` or `will not fit an A100-40: needs ~76 GB ... use A100-80 or batch <=
15`. If the verdict says it will not fit, change the batch or the card before
launching, not after the OOM.

Pass `attention: "eager"` if the job uses eager attention; the default is SDPA,
which keeps no score matrix. The number is a calibrated estimate under AdamW
with bf16 autocast (fitted on 178 measured runs across 41 architectures; held
out, the median error is 7% and it flagged 113 of the 131 times a run did not
fit a card, where accelerate estimate-memory flagged at most 8; on 25 runs
measured after the fit the median miss is 14%), not a measurement; say so when
you quote it, and quote the 14% with the 7%.

## After editing a model's config or architecture

**Call `check_edit` after an edit and before training**, with the model before
and after:

```
check_edit { base_model_path: "hf:Qwen/Qwen2.5-0.5B", model_path: "Model.neurarch.json", batch: 16, gpu: "L4" }
check_edit { base_config: "base/config.json", config: "config.json", lora_target_modules: ["q_proj", "v_proj"] }
```

The edits it reports load, run and are wrong, so nothing else will tell you: a
`resize_token_embeddings` that left a tensor at the old vocabulary, a tied head
that came untied, attention keys that had to move together and did not, a
longer context with no RoPE scaling, LoRA targets that match nothing. Fix
every `block` finding before the run. It also returns the training peak before
and after the edit, and says when the edit is what pushed the job off its card.

## After a CUDA out-of-memory error

**Call `diagnose_oom` after a CUDA out-of-memory error, before changing batch
size**, with the error text and the job you ran:

```
diagnose_oom { error: "<the OOM text>", model_path: "Model.neurarch.json", batch: 64, seq_len: 2048, step: 0 }
diagnose_oom { error: "<the OOM text>", hf_id: "Qwen/Qwen2.5-0.5B", batch: 8, inference: true, gen_tokens: 512 }
```

It names one cause (too big for the card, a fragmented allocator, another
process on the card, a giant logits tensor, memory the graph does not declare,
or growth during the run) with the numbers it read it from, and fixes ordered
least invasive first. Apply the first fix. Halve the batch only when the cause
is `too-big-for-card`, and then keep the effective batch with the gradient
accumulation the fix names.

Then, in this order:

1. `history` with the fingerprint the sizing result returns: what this
   structure measured the last time it trained in this organisation.
2. Whether it forward-passes: `lint_model`, then `check_design`. Agents rarely
   write a structurally illegal model, so this is the second line, not the
   first; the failure we saw a structure-focused agent hit was an OOM.

## The rule that matters for findings

**Do not act on a finding from a `.py` file. Trace the model first.**

The static parser is the weakest path in this repo and we measured how weak:
over 116 model files from 59 popular repositories it returns a graph for 86% and
a graph a person would recognise as the model for 41%. Of every `block` and
`warn` it raised on those graphs, hand-judged, **none was a real defect**. The
cause is structural, not a bug list: the parser reads construction order as data
flow and never sees `forward()`, so residual adds, functional activations and
config-selected heads all read as missing or misordered.
[`docs/REAL_REPOS_STUDY.md`](./docs/REAL_REPOS_STUDY.md) is the study.

A traced graph has none of that failure mode, because the model has already been
built by the time we look at it. The rules themselves are measured on traced
graphs: across 264 graphs, every one of the 96 designs the verifier blocked
crashed in PyTorch forward, and all 80 it passed ran clean.

So: parse to orient, trace to decide.

## Getting a graph, best first

```bash
# 1. The script already runs the model: no target, no shapes, no edit.
pip install neurarch-trace
NEURARCH_TRACE=1 python train.py            # writes ./<ClassName>.neurarch.json

# 2. You can name the model and its input.
neurarch-trace my_pkg.model:build --input 1,3,224,224
neurarch-trace hf:Qwen/Qwen2.5-0.5B

# 3. Static, when neither is possible. Orient only.
npx -y neurarch-mcp model.py
```

Option 1 is the one to reach for inside someone's repository, because a real
training entry point rarely has a model you can name: it is behind a config, a
registry, a `from_pretrained` and an `if args.variant ==`. The running process
has already resolved all of that. It records the first forward pass, changes
nothing about the run, and removes itself afterwards. `NEURARCH_TRACE_SKIP=n`
steps past warmup passes.

The MCP server also exposes this as the `trace_model` tool, which shells out to
the same package.

## Before a multi-GPU training job or a serving launch

**Call `check_parallel_layout` before launching a job split by TP / PP / DP**, and
**`check_serving_config` before a vLLM, SGLang or llama.cpp launch**:

```
check_parallel_layout { model_path: "hf:Qwen/Qwen3-8B", tp: 4, pp: 4, micro_batch: 1, global_batch: 32, seq_len: 4096, gpu: "h100" }
check_serving_config { command: "vllm serve Qwen/Qwen3-8B -tp 4 --max-model-len 32768", gpu: "l4", gpus: 4 }
```

Fix every `block` finding before launching. The layout times are predicted
from the graph, not measured: say so when you quote the slowdown or the
recommended split. A serving concurrency is an upper bound (the engine's own
reserve is not counted).

## The tools, by the question they answer

| Question | Tool |
|---|---|
| What will this job need, and on which card? | `size_training_job` |
| Why did it run out of memory, and what is the smallest fix? | `diagnose_oom` |
| I edited the config or the architecture: is the edit silently wrong, and what did it do to memory? | `check_edit` |
| Is this TP / PP / DP split legal, and which stage is the straggler? | `check_parallel_layout` |
| Will this vLLM / SGLang / llama.cpp launch start and serve the model as published? | `check_serving_config` |
| What is this model? | `describe_architecture`, `get_model_summary`, `mermaid_diagram` |
| Where is the budget? | `param_count_by_block`, `flops_by_block`, `list_blocks`, `get_block` |
| Will it run? | `validate_model`, `lint_model`, `check_design` |
| What does this edit break? | `layer_impact`, `find_path`, `list_connections`, `diff_models` |
| How do I fix this finding? | `suggest_fix` |
| Order k candidates (blocked last, ties stay ties) | `rank_designs` |
| Should I spend the money? | `plan` |
| What happened last time this trained? | `history` |
| Detail on one layer | `get_layer`, `find_layers`, `compare_layers`, `list_hyperparams` |
| Change the graph | `add_layer`, `modify_layer`, `add_connection`, `delete_layer`, `delete_connection`, `save_model` |
| Get code out | `export_pytorch` |
| Find something to start from | `list_architectures`, `load_architecture`, `load_hf_model`, `find_models` |

`plan` and `history` are the only two that open a socket, and they are the only
two that can answer a question the model in front of you structurally cannot:
what happened the last time this exact structure trained in this organisation.
That is not in the weights and not in the repository.

## When you are asked "is this design worth training"

Answer with `size_training_job` and `plan`, not with prose. Sizing is offline
and says what the run needs and costs; `plan` is the card the CI bot posts, with
the repository's own rules and the history line. In CI or a training entry
point the same verdict is available without an agent:

```bash
NEURARCH_TRACE=1 NEURARCH_TRACE_PLAN=1 NEURARCH_TRACE_FAIL_ON_BLOCK=1 python train.py
```

which stops a design that will not forward-pass after one step, instead of at
the shape error two epochs in on a GPU that has been billing the whole time.

## Working in this repo

```bash
npm ci && npm run typecheck && npm run build && npm test    # the Node server
cd python/neurarch-trace && pytest tests -q                 # the tracer
```

- `src/vendor/*.bundle.mjs` are generated from the Neurarch app, not edited
  here. `engine` and `verifier` come from `npm run build:mcp-engine` /
  `build:mcp-verifier` there; `plan.bundle.mjs` is a byte-for-byte copy of its
  committed `lib/modelPlan.bundle.mjs` (`npm run build:plan`), and
  `plan.contract.test.ts` pins its peak on two app templates. A stale bundle is the failure mode that has cost the most time in
  this repo: the rules silently drift from the ones the app and CI enforce.
- `python/neurarch-trace/neurarch_autopatch.pth` must reach the root of
  site-packages or `NEURARCH_TRACE=1` does nothing and says nothing. The
  mechanism is in `setup.py`; the publish workflow refuses to ship a wheel
  without it.
- Claims in this README and in docs are expected to name the measurement behind
  them. If you cannot point at the script that produced a number, do not write
  the number.
