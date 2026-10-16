"""Traced nodes -> the `.neurarch.json` document the MCP server loads.

Field names follow `src/lib/types.ts` (ModelArchitecture / MLComponent /
ComponentConnection). Shapes that were never observed are omitted rather than
written as null so the estimators fall back the same way they do for a graph
drawn by hand.
"""
import json
import sys
from typing import Optional, Any, Dict, List, Sequence

from .tracer import Node


def _emitted_shape(p: Node, by_id: Dict[str, Node]) -> Optional[List[int]]:
    """The shape a producer actually emitted. A synthesized merge node (the
    residual add) often carries no shape of its own because the merged tensor
    was never in hand; an elementwise merge emits the shape of its inputs, so
    borrow the first known one. Without this the GPT's `ve_gate`, which reads
    x[..., :32] of the residual, kept its edge unmarked and the plan refused a
    model that trains."""
    if p.output_shape is not None:
        return list(p.output_shape)
    for src in p.inputs:
        q = by_id.get(src)
        if q is not None and q.output_shape is not None and q.type != "input":
            return list(q.output_shape)
    return None


def build_graph(nodes: Sequence[Node], name: str, description: str = "") -> Dict[str, Any]:
    outputs: Dict[str, List[str]] = {n.id: [] for n in nodes}
    for n in nodes:
        for src in n.inputs:
            outputs[src].append(n.id)

    components = []
    for i, n in enumerate(nodes):
        c: Dict[str, Any] = {
            "id": n.id,
            "type": n.type,
            "name": n.name,
            "position": {"x": 0, "y": i * 100},
            "params": n.params,
            "inputs": list(n.inputs),
            "outputs": outputs[n.id],
        }
        if n.scope:
            c["scope"] = n.scope
        if n.input_shape is not None and n.type != "input":
            c["inputShape"] = n.input_shape
        if n.output_shape is not None:
            c["outputShape"] = n.output_shape
        components.append(c)

    by_id = {n.id: n for n in nodes}
    connections = []
    for n in nodes:
        for src in n.inputs:
            edge: Dict[str, Any] = {
                "id": "e%d" % len(connections),
                "from": src,
                "to": n.id,
                "fromPort": "bottom",
                "toPort": "top",
            }
            # The producer was found by walking autograd, so the tensor that
            # reached this layer may have passed through a slice, a view, a
            # rotary embedding or a fused kernel the graph has no node for.
            # When the shape the layer actually received is not the shape the
            # producer actually emitted, say so: a propagated shape across this
            # edge is a claim about an op nobody drew, and the verifier drops
            # flow-dependent findings downstream of an inferred edge rather
            # than refusing a model that just ran. (Karpathy's GPT: ve_gate
            # reads x[..., :32] of a 512-wide residual; the unfiltered rule
            # refused the working baseline eight times.)
            p = by_id.get(src)
            if (p is not None and n.input_shape is not None and p.type != "input"
                    and _emitted_shape(p, by_id) is not None
                    and list(_emitted_shape(p, by_id)) != list(n.input_shape)):
                edge["inferred"] = "unmodelled-op"
            connections.append(edge)

    return {
        "id": "traced-" + name,
        "name": name,
        "description": description,
        "components": components,
        "connections": connections,
    }


def write_graph(graph: Dict[str, Any], out_path: str) -> None:
    text = json.dumps(graph, indent=2)
    if out_path == "-":
        sys.stdout.write(text + "\n")
        return
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
