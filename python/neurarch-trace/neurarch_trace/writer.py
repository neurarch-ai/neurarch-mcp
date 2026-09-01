"""Traced nodes -> the `.neurarch.json` document the MCP server loads.

Field names follow `src/lib/types.ts` (ModelArchitecture / MLComponent /
ComponentConnection). Shapes that were never observed are omitted rather than
written as null so the estimators fall back the same way they do for a graph
drawn by hand.
"""
import json
import sys
from typing import Any, Dict, List, Sequence

from .tracer import Node


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

    connections = []
    for n in nodes:
        for src in n.inputs:
            connections.append({
                "id": "e%d" % len(connections),
                "from": src,
                "to": n.id,
                "fromPort": "bottom",
                "toPort": "top",
            })

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
