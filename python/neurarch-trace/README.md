# neurarch-trace

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
`-o -` writes to stdout, `--depth N` stops descending at module depth N and
records the modules there as single nodes, `--verbose` shows the traceback on a
failure (otherwise a failure is one line on stderr and exit code 1).

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
- The mapping table lives in `neurarch_trace/mapping.py` and mirrors
  `codeParser.ts` in the Neurarch app. If a torch module is missing there, it is
  recorded as `customModule` rather than guessed.
