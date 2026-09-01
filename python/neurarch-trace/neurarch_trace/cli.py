"""`neurarch-trace <target> --input 1,3,224,224 [-o out.neurarch.json]`."""
import argparse
import importlib
import importlib.util
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from . import __version__
from .tracer import trace
from .writer import build_graph, write_graph

DTYPES = {
    "float32": torch.float32, "float": torch.float32, "float16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "float64": torch.float64, "long": torch.long, "int64": torch.long,
    "int": torch.int32, "int32": torch.int32, "bool": torch.bool,
}


class TraceError(Exception):
    """A failure the user can act on; printed as one line without a traceback."""


def load_target(target: str) -> Tuple[nn.Module, str]:
    """Resolve `module:attr`, `file.py:attr` or `hf:<repo>` into a model and a default name."""
    if target.startswith("hf:"):
        repo = target[3:]
        try:
            from transformers import AutoModel
        except ImportError:
            raise TraceError("hf: targets need transformers: pip install 'neurarch-trace[hf]'")
        return AutoModel.from_pretrained(repo), re.sub(r"[^A-Za-z0-9_.-]+", "-", repo.split("/")[-1])

    if ":" not in target:
        raise TraceError("target must be module.path:attr, path/to/file.py:attr or hf:<repo-id>, got %r" % target)
    modpath, attr = target.rsplit(":", 1)
    if modpath.endswith(".py") or os.sep in modpath:
        path = os.path.abspath(modpath)
        if not os.path.isfile(path):
            raise TraceError("no such file: %s" % path)
        sys.path.insert(0, os.path.dirname(path))
        spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        # A bare module path is resolved from the working directory first, the
        # way `python -m` would, so `models.resnet:ResNet18` works from a repo root.
        sys.path.insert(0, os.getcwd())
        mod = importlib.import_module(modpath)

    obj = mod
    for part in attr.split("."):
        if not hasattr(obj, part):
            raise TraceError("%s has no attribute %r" % (modpath, attr))
        obj = getattr(obj, part)

    if isinstance(obj, nn.Module):
        model = obj
    elif callable(obj):
        try:
            model = obj()
        except Exception as e:
            raise TraceError("%s() failed: %s: %s" % (attr, type(e).__name__, e))
        if not isinstance(model, nn.Module):
            raise TraceError("%s() returned %s, not an nn.Module" % (attr, type(model).__name__))
    else:
        raise TraceError("%s is %s; expected an nn.Module, an nn.Module subclass or a factory" % (attr, type(obj).__name__))
    return model, attr.split(".")[-1]


def build_inputs(specs: Sequence[str], default_dtype: str) -> List[torch.Tensor]:
    """`1,3,224,224` or `1,128:long`; the first dim is the batch."""
    torch.manual_seed(0)
    out = []
    for spec in specs:
        dims_text, _, dtype_name = spec.partition(":")
        dtype_name = dtype_name or default_dtype
        if dtype_name not in DTYPES:
            raise TraceError("unknown dtype %r (one of %s)" % (dtype_name, ", ".join(sorted(DTYPES))))
        try:
            dims = [int(d) for d in dims_text.split(",") if d.strip()]
        except ValueError:
            raise TraceError("bad --input %r: expected comma-separated integers like 1,3,224,224" % spec)
        if not dims:
            raise TraceError("bad --input %r: no dims" % spec)
        dtype = DTYPES[dtype_name]
        if dtype == torch.bool:
            out.append(torch.rand(dims) > 0.5)
        elif dtype.is_floating_point:
            out.append(torch.randn(dims, dtype=dtype))
        else:
            out.append(torch.randint(0, 1000, dims, dtype=dtype))
    return out


def run(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="neurarch-trace",
        description="Run one forward pass over a PyTorch model and write a .neurarch.json graph with real shapes.",
    )
    p.add_argument("target", help="module.path:attr, path/to/file.py:attr, or hf:<repo-id>")
    p.add_argument("--input", action="append", default=[], metavar="DIMS",
                   help="input dims, batch first, e.g. 1,3,224,224 or 1,128:long; repeat for multi-input forwards")
    p.add_argument("--dtype", default=None, help="dtype for inputs without a :dtype suffix (default float32; long for hf:)")
    p.add_argument("--name", default=None, help="graph name (default: the attr or repo name)")
    p.add_argument("-o", "--output", default=None, help="output path (default ./<name>.neurarch.json; - for stdout)")
    p.add_argument("--depth", type=int, default=None, help="stop descending at this module depth (default: leaf modules)")
    p.add_argument("--verbose", action="store_true", help="show the full traceback on failure")
    p.add_argument("--version", action="version", version="neurarch-trace " + __version__)
    args = p.parse_args(argv)

    is_hf = args.target.startswith("hf:")
    specs = args.input or (["1,16"] if is_hf else [])
    if not specs:
        raise TraceError("--input is required (e.g. --input 1,3,224,224); it defaults only for hf: targets")
    dtype = args.dtype or ("long" if is_hf else "float32")

    model, default_name = load_target(args.target)
    inputs = build_inputs(specs, dtype)
    try:
        nodes = trace(model, inputs, depth=args.depth)
    except TraceError:
        raise
    except Exception as e:
        raise TraceError("forward pass failed: %s: %s" % (type(e).__name__, e))

    name = args.name or default_name
    description = "Traced by neurarch-trace %s from %s with input %s" % (
        __version__, args.target, " ".join(specs))
    graph = build_graph(nodes, name, description)
    out = args.output or ("./%s.neurarch.json" % name)
    write_graph(graph, out)
    if out != "-":
        layers = sum(1 for c in graph["components"] if c["type"] not in ("input", "output"))
        print("wrote %s (%d layers, %d connections)" % (out, layers, len(graph["connections"])))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    verbose = "--verbose" in (argv if argv is not None else sys.argv[1:])
    try:
        return run(argv)
    except Exception as e:
        if verbose:
            raise
        print("neurarch-trace: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
