# neurarch-trace

## One command, one card

```
pip install neurarch-trace
neurarch-trace models/resnet.py:ResNet18 --input 1,3,224,224 --plan --share
```

The trace runs locally and writes `ResNet18.neurarch.json` as always. `--plan`
then sends that graph to `POST https://www.neurarch.com/api/v1/plan` and prints
the card the server renders: what the model is, its parameter count, whether it
will run, which GPU it fits, what a training run costs, and every blocker the
verifier found. `--share` (which implies `--plan`) also stores the graph at a
public unguessable URL and prints it on the last line. Illustrative output (the
numbers below are made up; the server prints its own):

```
+------------------------------------------------------------------+
| ResNet18                                       models/resnet.py  |
| 18 layers, 11.7M params, 1.8 GFLOPs at 3x224x224                 |
+------------------------------------------------------------------+
| Runs            yes, shapes propagate end to end                 |
| GPU fit         T4 16 GB at batch 64 fp16 (5.1 GB peak)          |
| Training cost   ~$0.60 per epoch of ImageNet-1k on a T4          |
| Blockers        none                                             |
| Warnings        1: no dropout before the classifier              |
+------------------------------------------------------------------+
Share: https://www.neurarch.com/p/3f9k2q
```

stdout carries only the card and the `Share:` line, so the whole thing pastes
into a PR or a chat as is. Everything else (the "wrote ..." line, the notice that
the graph is being sent) goes to stderr.

Privacy: nothing leaves the machine unless you pass `--plan` or `--share`.
`--plan` sends the graph (layer types, shapes and hyperparameters, no weights, no
source) and the target string you typed, and the server keeps nothing addressable
by anyone. `--share` stores the graph at a public unguessable URL; anyone with the
link can open it, so do not pass it for a design you cannot show.

More flags on the same path: `--base other.neurarch.json` sends a second graph and
the card includes a diff against it, the design you are changing from.
`--fail-on-block` exits 2 when the card reports a blocker (a graph that would not
forward-pass), for CI; by default the exit code is 0 once the card printed.
`NEURARCH_API_KEY` (sent as a bearer token) lifts the per-IP rate limit, and
`NEURARCH_API` points the CLI at another deployment. The HTTP timeout is 30
seconds. On a network failure, a rate limit, or a server error the CLI prints one
line to stderr, exits 1, and the `.neurarch.json` is still on disk.

## No arguments at all

```
pip install neurarch-trace
NEURARCH_TRACE=1 python train.py
```

No target, no `--input`, no edit to `train.py`. The install puts one line in
site-packages, Python runs it before the first line of any script in the
environment, and with `NEURARCH_TRACE` set it waits for torch and records the
**first forward pass the process runs**. Your script built the model and handed
it a batch; there is nothing left to specify.

```
neurarch-trace: wrote ./GPT.neurarch.json (74 layers, 81 connections) from the
first forward pass of GPT
```

The reason to prefer this over pointing the CLI at a file is not convenience. A
real training entry point rarely has a model you can name: it is behind a config,
a registry, a `from_pretrained`, a wrapper, three imports and a `if
args.variant ==`. The running process has already resolved all of that.

Add the plan, and it becomes a gate:

```
NEURARCH_TRACE=1 NEURARCH_TRACE_PLAN=1 NEURARCH_TRACE_FAIL_ON_BLOCK=1 python train.py
```

A design that will not forward-pass now stops the process after one step, with
the blocker printed, instead of at the shape error two epochs in on a GPU that
has been billing the whole time.

**What it will not do to your run.** It reads the forward pass with the same
hooks the CLI uses and drives nothing: it does not switch the model to `eval()`,
does not enable grad, does not set `requires_grad` on your batch, and never runs
a second forward pass. After the one capture it puts `nn.Module.__call__` back,
so from step two there is no wrapper left in the stack. Nothing leaves the
machine unless `NEURARCH_TRACE_PLAN` or `NEURARCH_TRACE_SHARE` is set. Any
failure inside the tracer is one line on stderr and your run continues: a tracer
must never be the reason a training job dies.

The cost of that restraint is that under `torch.no_grad()` there is no autograd
graph to walk, so residual adds and concatenations are not recovered as `add` /
`concatenate` nodes. A training step has grad on and does not have this problem;
for an inference-only script, use the CLI.

| Variable | |
|---|---|
| `NEURARCH_TRACE=1` | arm it (nothing happens without this) |
| `NEURARCH_TRACE_OUT=path` | where the graph goes (default `./<name>.neurarch.json`) |
| `NEURARCH_TRACE_NAME=name` | graph name (default: the model's class name) |
| `NEURARCH_TRACE_DEPTH=n` | stop descending at module depth n |
| `NEURARCH_TRACE_SKIP=n` | ignore the first n forward passes (warmup, LR probes) |
| `NEURARCH_TRACE_PLAN=1` | send the graph for the plan and print the card |
| `NEURARCH_TRACE_SHARE=1` | implies `PLAN`; also store it at a public URL |
| `NEURARCH_TRACE_FAIL_ON_BLOCK=1` | exit 2 when the plan reports a blocker |

If you would rather not set an environment variable, `python -c "import
neurarch_trace.autopatch as a; a.install(force=True)"` arms the same thing from
inside a script, as long as it runs before the first forward pass.

After `pip install -e .` from a source checkout, run `python
scripts/dev_copy_pth.py` once: an editable install does not place the
site-packages line, and without it `NEURARCH_TRACE=1` does nothing and says
nothing. `--check` reports whether it is there.

## What it does

Run one forward pass over a PyTorch model and write a `.neurarch.json` graph with
the real input and output shape of every layer. Point
[neurarch-mcp](https://github.com/neurarch-ai/neurarch-mcp) at that file and every
tool works: parameter counts, FLOPs, shape contracts, `lint_model`, `check_design`.

neurarch-mcp can already read a `.py` file by parsing it statically. Static parsing
cannot see tensor shapes, and it cannot follow a model that is built at runtime:
`AutoModel.from_pretrained(...)`, a timm factory, an architecture spread across a
dozen files. `neurarch-trace` closes that gap by instantiating the model and
watching the tensors go through it.

## Install

```
pip install neurarch-trace          # torch >= 2.0
pip install 'neurarch-trace[hf]'    # adds transformers for hf: targets
```

## Usage

```
neurarch-trace <target> --input 1,3,224,224 [--input 1,128:long ...] [-o out.neurarch.json]
python -m neurarch_trace <target> ...     # same thing
```

A file and a class or factory in it:

```
neurarch-trace models/resnet.py:ResNet18 --input 1,3,224,224 -o resnet18.neurarch.json
```

An importable module and a factory function (called with no arguments):

```
neurarch-trace my_pkg.model:build_model --input 1,3,224,224
```

A Hugging Face checkpoint (defaults to `--input 1,16 --dtype long`, token ids):

```
neurarch-trace hf:prajjwal1/bert-tiny -o bert-tiny.neurarch.json
```

Then hand the graph to your agent:

```
npx -y neurarch-mcp ./resnet18.neurarch.json
```

`<target>` may name an `nn.Module` instance, an `nn.Module` subclass (instantiated
with no arguments), or a callable that returns one. `--input` is repeated for
multi-input forwards; a `:dtype` suffix (`1,128:long`) overrides `--dtype` for that
input. Random tensors are used, `torch.randn` for float dtypes and
`torch.randint(0, 1000, ...)` for integer ones.

Other flags: `--name` sets the graph name (default: the attribute or repo name),
`-o -` writes to stdout (not combinable with `--plan`), `--depth N` stops
descending at module depth N and records the modules there as single nodes,
`--verbose` shows the traceback on a failure (otherwise a failure is one line on
stderr and exit code 1). `--plan`, `--share`, `--base` and `--fail-on-block` are
described under "One command, one card" above.

## The shape convention

Shapes in the graph are written **without the batch dimension**: `[3, 224, 224]`
for an image, `[128, 768]` for a token sequence. The first dim of every `--input`
is the batch and is stripped from every recorded shape. Neurarch reads a leading
dimension as the channel axis, so a shape that still carries its batch of 1 would
be read as a one-channel tensor and every downstream number would be wrong.

The trace runs in `eval()` mode on CPU with autograd enabled.

## What ends up in the graph

- One node per leaf module call, typed with the Neurarch vocabulary (`conv2d`,
  `linear`, `layerNorm`, `multiHeadAttention`, ...) and carrying the same
  parameter keys the static parser emits, so the MCP estimators read them. A
  module called twice is two nodes. Modules the mapping table does not know
  become `customModule` with `className` and `paramCount` in their params.
- `scope` on every node is the dotted path of its parent module
  (`encoder.layer.0.attention`), which is what the MCP block tools group by.
- Edges follow actual data flow. A tensor a layer returns is matched by identity;
  anything produced by functional code in between (`x + residual`, `torch.cat`,
  reshapes, RoPE math) is traced back through autograd to the layers it came from.
  A residual add becomes an `add` node with two inputs, a concatenation a
  `concatenate` node.
- Exactly one `input` node per `--input` and one `output` node.

## Limits

- Functional ops other than add / cat / stack do not get nodes of their own; a
  `torch.flatten` or `F.softmax` between two modules is invisible, and the two
  modules are wired directly.
- Attribution runs on the autograd graph, so a forward wrapped in
  `torch.no_grad()` or an integer-only path with no learnable parameter behind it
  falls back to matching by dtype against the model inputs.
- The batch is assumed to be dim 0. Sequence-first layouts
  (`nn.MultiheadAttention` with `batch_first=False`) come out with the sequence
  length stripped instead.
- `NEURARCH_TRACE=1` captures the outermost `nn.Module.__call__` in the process,
  which is the model in every normal script but is a warmup or a loss module in
  a few; `NEURARCH_TRACE_SKIP` steps past those. A model already wrapped in
  `torch.compile` should be traced before the compile call, not after.
- The mapping table lives in `neurarch_trace/mapping.py` and mirrors
  `codeParser.ts` in the Neurarch app. If a torch module is missing there, it is
  recorded as `customModule` rather than guessed.
