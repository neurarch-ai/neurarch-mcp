# AGENTS.md

For the agent, not the reader. If you are a coding agent that has landed in this
repository or installed this package, this is the short version of what it gives
you and, more usefully, of when its answers are worth trusting.

## What this is

Two shipped artifacts from one repo:

- **`neurarch-mcp`** (npm, Node 20+) — an MCP server that turns a PyTorch model
  into a structured graph and answers questions about it with computation
  instead of inference. Every tool but `plan` and `history` runs locally, needs
  no key and no account, and opens no socket.
- **`neurarch-trace`** (PyPI, torch >= 2.0) — runs one forward pass and writes
  the graph with real per-layer shapes.

## The rule that matters

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

## The tools, by the question they answer

| Question | Tool |
|---|---|
| What is this model? | `describe_architecture`, `get_model_summary`, `mermaid_diagram` |
| Where is the budget? | `param_count_by_block`, `flops_by_block`, `list_blocks`, `get_block` |
| Will it run? | `validate_model`, `lint_model`, `check_design` |
| What does this edit break? | `layer_impact`, `find_path`, `list_connections`, `diff_models` |
| How do I fix this finding? | `suggest_fix` |
| Which candidate deserves the GPU? | `rank_designs` |
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

Answer with `plan`, not with prose. It returns the parameter count, whether the
graph forward-passes, which GPU it fits, what a run costs, and the blockers. In
CI or a training entry point the same verdict is available without an agent:

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

- `src/vendor/verifier.bundle.mjs` is generated from the Neurarch app, not
  edited here. A stale bundle is the failure mode that has cost the most time in
  this repo: the rules silently drift from the ones the app and CI enforce.
- `python/neurarch-trace/neurarch_autopatch.pth` must reach the root of
  site-packages or `NEURARCH_TRACE=1` does nothing and says nothing. The
  mechanism is in `setup.py`; the publish workflow refuses to ship a wheel
  without it.
- Claims in this README and in docs are expected to name the measurement behind
  them. If you cannot point at the script that produced a number, do not write
  the number.
