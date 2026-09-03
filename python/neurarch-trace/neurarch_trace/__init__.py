"""neurarch-trace: one forward pass, one .neurarch.json with real shapes.

    from neurarch_trace import trace_model
    graph = trace_model(model, [torch.randn(1, 3, 224, 224)], name="resnet")
"""
from typing import Any, Dict, Optional, Sequence

__version__ = "0.1.3"


def trace_model(model, inputs: Sequence[Any], name: str = "model", depth: Optional[int] = None,
                description: str = "") -> Dict[str, Any]:
    """Trace `model` on `inputs` (batch first) and return the graph as a dict."""
    from .tracer import trace
    from .writer import build_graph
    return build_graph(trace(model, list(inputs), depth=depth), name, description)


__all__ = ["trace_model", "__version__"]
