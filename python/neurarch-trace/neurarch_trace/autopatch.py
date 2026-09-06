"""Trace the model a script already runs, without editing the script.

    pip install neurarch-trace
    NEURARCH_TRACE=1 python train.py

There is no `--input` here and no import target, because there is nothing to
guess: the script constructs its own model and hands it its own batch. The first
outermost `nn.Module.__call__` in the process is the model, whatever built it,
and the tensors it was called with are the input. That is the whole reason this
path exists next to the CLI.

It is also why it is not a parser. Reading `train.py` as text means recovering
the graph from construction order and never seeing `forward()`; we measured that
and it returns a graph a person recognises for 41% of real model files. A hook
on the running module tree has no such failure mode, because the model has
already been built by the time we look at it.

## What it will not do to the run it is watching

The forward pass belongs to the caller. This module installs
`register_forward_pre_hook` / `register_forward_hook` and reads what goes past;
it does not switch the model to eval, does not enable grad, does not set
`requires_grad` on the batch, and does not run a second forward pass. It
captures once and then puts `nn.Module.__call__` back, so from the second step
on there is no wrapper left in the stack.

The cost of that restraint: under `torch.no_grad()` there is no autograd graph
to walk, so tensor identity is the only way home and functional joins (a
residual `x + y`, a `torch.cat`) are not recovered as `add` / `concatenate`
nodes. A training step has grad on and does not have this problem. An inference
script does; trace it with the CLI instead.

## Configuration, all through the environment

    NEURARCH_TRACE=1            enable (the .pth does nothing without it)
    NEURARCH_TRACE_OUT=path     where the graph goes (default ./<name>.neurarch.json)
    NEURARCH_TRACE_NAME=name    graph name (default: the model's class name)
    NEURARCH_TRACE_DEPTH=n      stop descending at module depth n
    NEURARCH_TRACE_SKIP=n       ignore the first n forwards (warmup, LR probes)
    NEURARCH_TRACE_PLAN=1       send the graph for the plan and print the card
    NEURARCH_TRACE_SHARE=1      implies PLAN; also store it at a public URL
    NEURARCH_TRACE_FAIL_ON_BLOCK=1   exit 2 when the plan reports a blocker

The last two lines together are the point of the whole file:

    NEURARCH_TRACE=1 NEURARCH_TRACE_PLAN=1 NEURARCH_TRACE_FAIL_ON_BLOCK=1 python train.py

stops a structurally broken design after one forward pass instead of at the
epoch boundary where its shape error would have surfaced, on a GPU that is
already billing. Nothing leaves the machine unless PLAN or SHARE is set.
"""
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

PREFIX = "neurarch-trace: "

# Per-thread nesting depth of nn.Module.__call__, so "the outermost call" has a
# meaning in a DataLoader thread and in the main thread independently.
_local = threading.local()

_state: Dict[str, Any] = {
    "installed": False,   # the torch import watcher is on sys.meta_path
    "patched": False,     # nn.Module.__call__ is ours
    "captured": False,    # the one capture has been attempted; never retry
    "skips": None,        # remaining warmup forwards to ignore
    "orig_call": None,
}


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """True when NEURARCH_TRACE asks for the autopatch. Read by the .pth file."""
    env = os.environ if env is None else env
    return _truthy(env.get("NEURARCH_TRACE"))


def _int_env(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        _warn("ignoring %s=%r: not an integer" % (name, raw))
        return default


def _warn(message: str) -> None:
    sys.stderr.write(PREFIX + message + "\n")


def _note(message: str) -> None:
    sys.stderr.write(PREFIX + message + "\n")


# ── installation ───────────────────────────────────────────────────────────
#
# The .pth runs before the script's first line, which is long before torch
# exists. Python has no post-import hook in the standard library (PEP 369 was
# rejected), so the finder below joins sys.meta_path, lets the real machinery
# produce torch's spec, and wraps that spec's loader so the patch lands the
# instant torch finishes importing. The loader object comes fresh out of
# find_spec, so wrapping it affects this import and nothing else.


class _TorchWatcher:
    """A meta-path finder that patches torch the moment it is imported."""

    def __init__(self) -> None:
        self._busy = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != "torch" or self._busy:
            return None
        import importlib.util
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            return None
        finally:
            self._busy = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        loader = spec.loader
        original = loader.exec_module

        def exec_module(module: Any) -> None:
            original(module)
            _on_torch_ready()

        try:
            loader.exec_module = exec_module  # type: ignore[method-assign]
        except (AttributeError, TypeError):
            return None
        return spec

    # Python 2-era API some tools still probe for; explicitly absent.
    find_module = None


def install(force: bool = False) -> bool:
    """Arm the autopatch. Returns True when it is armed after this call.

    Idempotent. Called by the .pth file when NEURARCH_TRACE is set, and usable
    by hand (`import neurarch_trace.autopatch as a; a.install(force=True)`) from
    a script that would rather not set an environment variable.
    """
    if _state["installed"]:
        return True
    if not force and not enabled():
        return False
    # Our own CLI drives its own forward pass and must not be wrapped by this.
    if os.path.basename(sys.argv[0] or "").startswith("neurarch-trace"):
        return False
    _state["installed"] = True
    _state["skips"] = max(0, _int_env("NEURARCH_TRACE_SKIP", 0) or 0)
    if "torch" in sys.modules:
        _on_torch_ready()
    else:
        sys.meta_path.insert(0, _TorchWatcher())
    return True


def _on_torch_ready() -> None:
    """Replace nn.Module.__call__ with the depth-counting wrapper."""
    if _state["patched"] or _state["captured"]:
        return
    try:
        import torch.nn as nn
    except Exception as exc:  # torch imported but broken; not our problem to raise
        _warn("could not reach torch.nn (%s); autopatch is off" % exc)
        return

    # torch binds `__call__ = _wrapped_call_impl` as a class attribute, so the
    # name to replace is __call__ itself; patching _wrapped_call_impl would be
    # invisible to every call site.
    original: Callable[..., Any] = nn.Module.__call__
    _state["orig_call"] = original
    _state["patched"] = True

    def __call__(self: Any, *args: Any, **kwargs: Any) -> Any:
        depth = getattr(_local, "depth", 0)
        _local.depth = depth + 1
        try:
            if depth == 0 and not _state["captured"]:
                return _outermost(self, original, args, kwargs)
            return original(self, *args, **kwargs)
        finally:
            _local.depth = depth

    nn.Module.__call__ = __call__  # type: ignore[method-assign]


def uninstall() -> None:
    """Put nn.Module.__call__ back. Called automatically after one capture."""
    if not _state["patched"]:
        return
    try:
        import torch.nn as nn
        nn.Module.__call__ = _state["orig_call"]  # type: ignore[method-assign]
    except Exception:
        pass
    _state["patched"] = False
    _state["orig_call"] = None


# ── the one capture ────────────────────────────────────────────────────────


def _outermost(model: Any, original: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """Called for the first forward pass of the process that nothing else wraps."""
    from .tracer import Tracer, flatten_tensors

    skips = _state["skips"] or 0
    if skips > 0:
        _state["skips"] = skips - 1
        return original(model, *args, **kwargs)

    inputs: List[Any] = flatten_tensors((args, kwargs))
    if not inputs:
        # A module called with no tensors is not the forward pass we want, and
        # it must not consume the single shot; the real one is still coming.
        return original(model, *args, **kwargs)

    # Set before the forward, not after: if this throws, the user gets their
    # own exception and their next step is not wrapped a second time.
    _state["captured"] = True

    try:
        tracer = Tracer(model, depth=_int_env("NEURARCH_TRACE_DEPTH"))
        tracer.begin(inputs, require_grad=False)
    except Exception as exc:
        _warn("could not attach to %s (%s); the run continues untraced"
              % (type(model).__name__, exc))
        uninstall()
        return original(model, *args, **kwargs)

    try:
        result = original(model, *args, **kwargs)
    finally:
        tracer.detach()
        uninstall()

    try:
        nodes = tracer.finish(result)
    except Exception as exc:
        _warn("the forward pass ran but the graph could not be closed (%s)" % exc)
        return result

    try:
        _emit(nodes, model, inputs)
    except SystemExit:
        raise
    except Exception as exc:
        _warn("traced the model but could not write the graph (%s)" % exc)
    return result


def _describe_inputs(inputs: Sequence[Any]) -> str:
    return " ".join(",".join(str(d) for d in tuple(t.shape)) for t in inputs)


def _emit(nodes: List[Any], model: Any, inputs: Sequence[Any]) -> None:
    from . import __version__
    from .writer import build_graph, write_graph

    name = os.environ.get("NEURARCH_TRACE_NAME") or type(model).__name__
    shapes = _describe_inputs(inputs)
    description = "Traced by neurarch-trace %s (autopatch) from %s with input %s" % (
        __version__, type(model).__name__, shapes)
    graph = build_graph(nodes, name, description)

    out = os.environ.get("NEURARCH_TRACE_OUT") or ("./%s.neurarch.json" % name)
    if out == "-":
        # stdout belongs to the script we are a guest in.
        raise ValueError("NEURARCH_TRACE_OUT=- is not supported by the autopatch; give it a path")
    write_graph(graph, out)
    layers = sum(1 for c in graph["components"] if c["type"] not in ("input", "output"))
    _note("wrote %s (%d layers, %d connections) from the first forward pass of %s"
          % (out, layers, len(graph["connections"]), type(model).__name__))

    share = _truthy(os.environ.get("NEURARCH_TRACE_SHARE"))
    if not (share or _truthy(os.environ.get("NEURARCH_TRACE_PLAN"))):
        return
    _plan(graph, shapes, layers, share)


def _plan(graph: Dict[str, Any], shapes: str, layers: int, share: bool) -> None:
    """The only path on which anything leaves the machine."""
    from . import __version__
    from .plan import PlanError, api_base, api_host, has_blocker, print_plan, request_plan

    api = api_base()
    _note("sending the graph (%d layers) to %s for the plan." % (layers, api_host(api)))
    source = {
        "kind": "trace",
        "target": "autopatch",
        "input": shapes,
        "tool": "neurarch-trace/%s (autopatch)" % __version__,
    }
    try:
        response = request_plan(
            graph, source, share=share, api=api,
            api_key=os.environ.get("NEURARCH_API_KEY") or None,
        )
    except PlanError as exc:
        _warn("%s; the graph was still written" % exc)
        return
    print_plan(response, share=share, out=sys.stderr)

    if not _truthy(os.environ.get("NEURARCH_TRACE_FAIL_ON_BLOCK")):
        return
    if not has_blocker(response):
        return
    # SystemExit rather than a return code: this is a guest in someone's
    # training script, and the whole value of stopping here is that it happens
    # before the run bills anything. A bare `except Exception` around the step
    # does not swallow it.
    _warn("the plan reports a blocker and NEURARCH_TRACE_FAIL_ON_BLOCK is set; "
          "stopping before the run costs anything")
    raise SystemExit(2)


def _reset_for_tests() -> None:
    """Undo everything, so a test can arm the autopatch more than once."""
    uninstall()
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _TorchWatcher)]
    _state.update(installed=False, patched=False, captured=False, skips=None, orig_call=None)
    _local.depth = 0


__all__ = ["install", "uninstall", "enabled"]
