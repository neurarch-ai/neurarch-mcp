"""One forward pass with hooks, turned into a Neurarch graph.

Data flow is recovered two ways, in this order:

1. Tensor identity. Every tensor a recorded module returns is remembered by
   `id()`; a consumer whose input is that same object is wired to it directly.
2. The autograd graph. When a consumer's input is not a recorded tensor (it came
   out of `x + residual`, `torch.cat`, a reshape, RoPE math, anything functional)
   its `grad_fn` chain is walked upward until recorded producers are found. An
   AddBackward or CatBackward met on the way with two or more producers under it
   becomes an explicit `add` / `concatenate` node, which is how the app draws a
   merge point.

Both need the traced tensors kept alive (so ids are not recycled) and grad
enabled on the forward (so the chain exists at all). Both are handled here.
"""
import sys
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .mapping import PASSTHROUGH_CLASSES, TORCH_CLASS_MAP, map_module

MERGE_TYPES = {"AddBackward0": "add", "CatBackward0": "concatenate", "StackBackward0": "stack"}


class Node:
    __slots__ = ("id", "type", "name", "params", "scope", "input_shape", "output_shape", "inputs")

    def __init__(self, id: str, type: str, name: str, params: Dict[str, Any], scope: Optional[str] = None):
        self.id, self.type, self.name, self.params, self.scope = id, type, name, params, scope
        self.input_shape: Optional[List[int]] = None
        self.output_shape: Optional[List[int]] = None
        self.inputs: List[str] = []


def batchless(t: torch.Tensor) -> List[int]:
    # The app reads shapes without the batch axis; a leading 1 would be taken
    # for the channel count and corrupt every downstream number.
    return list(t.shape[1:]) if t.dim() >= 1 else []


def flatten_tensors(obj: Any, out: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
    """Tensors inside nested tuples / lists / dicts (HF ModelOutput is a dict)."""
    if out is None:
        out = []
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            flatten_tensors(v, out)
    elif isinstance(obj, (tuple, list)):
        for v in obj:
            flatten_tensors(v, out)
    return out


def select_modules(model: nn.Module, depth: Optional[int]) -> List[Tuple[str, nn.Module]]:
    """(qualified name, module) pairs that become graph nodes."""
    picked: List[Tuple[str, nn.Module]] = []
    atomic: List[str] = []
    for qual, m in model.named_modules():
        if type(m).__name__ in PASSTHROUGH_CLASSES or any(qual.startswith(a + ".") for a in atomic):
            continue
        d = 0 if qual == "" else qual.count(".") + 1
        if depth is not None and d > depth:
            continue
        # A torch module the table knows is one layer even when it has children
        # (nn.MultiheadAttention owns an out_proj Linear); descending into it
        # would turn one attention node into a lone linear.
        known = type(m).__name__ in TORCH_CLASS_MAP
        leaf = known or next(m.children(), None) is None or (depth is not None and d == depth)
        if leaf:
            picked.append((qual, m))
            if known:
                atomic.append(qual)
    return picked


class Tracer:
    def __init__(self, model: nn.Module, depth: Optional[int] = None):
        self.model = model
        self.depth = depth
        self.nodes: List[Node] = []
        self.node_by_id: Dict[str, Node] = {}
        self.merge_ids: set = set()
        self.by_id: Dict[int, str] = {}
        self.by_gradfn: Dict[Any, str] = {}
        self.gradfn_at_register: Dict[int, Any] = {}
        self.memo: Dict[Any, List[str]] = {}
        self.keep: List[torch.Tensor] = []
        self.param_ids = {id(p) for p in model.parameters()} | {id(b) for b in model.buffers()}
        self.input_nodes: List[Tuple[torch.Tensor, str]] = []
        self.calls: Dict[str, int] = {}
        self.pending: List[Tuple[List[str], Optional[List[int]]]] = []
        # Scope of the module whose inputs are being resolved right now, so a
        # merge node lands inside the block that consumes it, as the app draws it.
        self.consumer_scope: Optional[str] = None

    # node bookkeeping

    def _add(self, node: Node) -> Node:
        base, k = node.id, 2
        while node.id in self.node_by_id:
            node.id = "%s_%d" % (base, k)
            k += 1
        self.node_by_id[node.id] = node
        self.nodes.append(node)
        return node

    def _register(self, t: torch.Tensor, node_id: str) -> None:
        self.keep.append(t)
        self.by_id[id(t)] = node_id
        # Remember the grad_fn the tensor had when the module returned it. An
        # in-place op afterwards (`out += identity`, torchvision's residual)
        # keeps the Python object and swaps the grad_fn, and that swap is the
        # only sign the tensor is no longer what the module produced.
        self.gradfn_at_register[id(t)] = t.grad_fn
        if t.grad_fn is not None:
            self.by_gradfn[t.grad_fn] = node_id

    # producer resolution

    def _producers_of(self, t: torch.Tensor) -> List[str]:
        hit = self.by_id.get(id(t))
        if hit is not None and t.grad_fn is self.gradfn_at_register.get(id(t)):
            return [hit]
        # Same object, different grad_fn: modified in place since it was
        # recorded (resnet's `out += identity`). Identity would wire the
        # consumer to the pre-add producer and drop the residual branch as a
        # dead end; the autograd walk below sees the AddBackward instead.
        fn = t.grad_fn
        if fn is not None:
            found = self._resolve(fn)
            merge = self.by_gradfn.get(fn)
            if merge in self.merge_ids and self.node_by_id[merge].output_shape is None:
                # The merged tensor is in hand only here; deeper merges keep None.
                self.node_by_id[merge].input_shape = self.node_by_id[merge].output_shape = batchless(t)
            return found
        if id(t) in self.param_ids:
            return []
        return self._leaf_input_guess(t)

    def _leaf_input_guess(self, t: torch.Tensor) -> List[str]:
        # Integer inputs carry no autograd history, so `input_ids.view(...)` or a
        # mask built from them arrives as a fresh leaf. Match it back to the
        # model input of the same dtype (and shape, when dtype is ambiguous).
        same = [(x, nid) for x, nid in self.input_nodes if x.dtype == t.dtype]
        if len(same) > 1:
            same = [(x, nid) for x, nid in same if x.shape == t.shape] or same[:1]
        return [same[0][1]] if same else []

    def _resolve(self, fn: Any) -> List[str]:
        hit = self.by_gradfn.get(fn)
        if hit is not None:
            return [hit]
        if fn in self.memo:
            return self.memo[fn]
        self.memo[fn] = []
        if hasattr(fn, "variable"):
            v = fn.variable
            result = [self.by_id[id(v)]] if id(v) in self.by_id else []
        else:
            found: List[str] = []
            for nf, _ in fn.next_functions:
                if nf is None:
                    continue
                for p in self._resolve(nf):
                    if p not in found:
                        found.append(p)
            kind = MERGE_TYPES.get(fn.name())
            if kind and len(found) >= 2:
                found = [self._merge_node(fn, kind, found)]
            result = found
        self.memo[fn] = result
        return result

    def _merge_node(self, fn: Any, kind: str, producers: List[str]) -> str:
        node = self._add(Node(kind, kind, kind, {}, self.consumer_scope))
        node.name = node.id  # the app's duplicate-name rule fires on two nodes both called "add"
        node.inputs = list(producers)
        self.by_gradfn[fn] = node.id
        self.merge_ids.add(node.id)
        return node.id

    # hooks

    def _pre(self, qual: str, module: nn.Module, args: tuple, kwargs: dict) -> None:
        self.consumer_scope = scope_of(qual)
        tensors = flatten_tensors((args, kwargs))
        producers: List[str] = []
        for t in tensors:
            for p in self._producers_of(t):
                if p not in producers:
                    producers.append(p)
        self.pending.append((producers, batchless(tensors[0]) if tensors else None))

    def _post(self, qual: str, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        producers, in_shape = self.pending.pop()
        typ, params = map_module(module)
        k = self.calls[qual] = self.calls.get(qual, 0) + 1
        display = qual or type(module).__name__
        name = display if k == 1 else "%s#%d" % (display, k)
        node = self._add(Node(display.replace(".", "_"), typ, name, params, scope_of(qual)))
        node.inputs, node.input_shape = producers, in_shape
        outs = flatten_tensors(output)
        node.output_shape = batchless(outs[0]) if outs else None
        for t in outs:
            self._register(t, node.id)

    # driver

    def run(self, inputs: Sequence[torch.Tensor]) -> List[Node]:
        for i, x in enumerate(inputs):
            if x.is_floating_point():
                x.requires_grad_(True)
            node = self._add(Node("input" if len(inputs) == 1 else "input_%d" % i, "input", "input", {"shape": batchless(x)}))
            node.output_shape = batchless(x)
            self.input_nodes.append((x, node.id))
            self._register(x, node.id)

        handles = []
        for qual, m in select_modules(self.model, self.depth):
            handles.append(m.register_forward_pre_hook(partial(self._pre, qual), with_kwargs=True))
            handles.append(m.register_forward_hook(partial(self._post, qual), with_kwargs=True))
        if not handles:
            raise RuntimeError("model has no modules to record (is it an nn.Module with layers?)")

        # Functional chains between two modules can be long (RoPE, masks), and
        # the walk is recursive; the default limit is too low for real LLMs.
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                result = self.model(*inputs)
        finally:
            for h in handles:
                h.remove()
            self.model.train(was_training)

        outs = flatten_tensors(result)
        self.consumer_scope = None
        out = self._add(Node("output", "output", "output", {}))
        for t in outs:
            for p in self._producers_of(t):
                if p not in out.inputs:
                    out.inputs.append(p)
        out.input_shape = batchless(outs[0]) if outs else None
        if not out.inputs:
            raise RuntimeError("forward returned nothing traceable (no tensor output reached a recorded layer)")
        return self.nodes


def scope_of(qual: str) -> Optional[str]:
    return qual.rsplit(".", 1)[0] if "." in qual else None


def trace(model: nn.Module, inputs: Sequence[torch.Tensor], depth: Optional[int] = None) -> List[Node]:
    return Tracer(model, depth).run(list(inputs))
