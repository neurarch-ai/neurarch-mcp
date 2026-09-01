"""torch module class -> Neurarch component type and param keys.

Source of truth: `src/utils/codeParser.ts` in the Neurarch app (the static
parser that neurarch-mcp vendors). The MCP param and FLOPs estimators key off
the exact param names that parser emits, so every entry here mirrors what
`parseLayerDefinition` would produce for the same `nn.X(...)` call. Adding a
type that codeParser does not emit is fine as long as it exists in
`src/lib/types.ts`; renaming a param key is not.
"""
from typing import Any, Callable, Dict, Optional, Tuple

import torch.nn as nn

Params = Dict[str, Any]


def _first(v: Any) -> Any:
    # torch stores kernel_size / stride / padding as tuples; the graph stores the
    # scalar codeParser reads off the source when all dims agree.
    if isinstance(v, (tuple, list)):
        return v[0] if all(x == v[0] for x in v) else list(v)
    return v


def _conv(m: nn.Module) -> Params:
    p: Params = {
        "inChannels": m.in_channels,
        "outChannels": m.out_channels,
        "kernelSize": _first(m.kernel_size),
        "stride": _first(m.stride),
        "padding": _first(m.padding),
    }
    if _first(m.dilation) != 1:
        p["dilation"] = _first(m.dilation)
    if m.groups != 1:
        p["groups"] = m.groups
    if m.bias is None:
        p["bias"] = False
    return p


def _conv2d(m: nn.Module) -> Tuple[str, Params]:
    # codeParser's ViT heuristic: a wide conv whose stride equals its kernel on
    # a few input channels is a patch projection, not a feature extractor.
    k, s = _first(m.kernel_size), _first(m.stride)
    if isinstance(k, int) and k > 4 and k == s and m.in_channels <= 4:
        return "patchEmbed", {"patchSize": k, "embedDim": m.out_channels, "inChans": m.in_channels}
    return "conv2d", _conv(m)


def _conv_t2d(m: nn.Module) -> Params:
    p = _conv(m)
    if _first(m.output_padding) != 0:
        p["outputPadding"] = _first(m.output_padding)
    return p


def _pool(m: nn.Module) -> Params:
    return {"kernelSize": _first(m.kernel_size), "stride": _first(m.stride or m.kernel_size)}


def _adaptive_avg(m: nn.Module) -> Tuple[str, Params]:
    out = m.output_size if isinstance(m.output_size, (tuple, list)) else (m.output_size,)
    if all(o in (1, None) for o in out):
        return "globalAvgPool2d", {}
    return "adaptiveAvgPool2d", {"outputSize": _first(m.output_size)}


def _recurrent(m: nn.Module) -> Params:
    p: Params = {"inputSize": m.input_size, "hiddenSize": m.hidden_size, "numLayers": m.num_layers}
    if m.bidirectional:
        p["bidirectional"] = True
    return p


def _norm_shape(m: nn.Module) -> Params:
    return {"normalizedShape": _first(tuple(m.normalized_shape))}


# class name -> (component type, param extractor). An extractor may instead
# return (type, params) itself when the type depends on the module's config.
TORCH_CLASS_MAP: Dict[str, Tuple[Optional[str], Callable[[nn.Module], Any]]] = {
    "Linear": ("linear", lambda m: dict(
        {"inFeatures": m.in_features, "outFeatures": m.out_features}, **({} if m.bias is not None else {"bias": False}))),
    "Conv2d": (None, _conv2d),
    "ConvTranspose2d": ("transposeConv2d", _conv_t2d),
    "Conv1d": ("conv1d", _conv),
    "Conv3d": ("conv3d", _conv),
    "MaxPool2d": ("maxpool2d", _pool),
    "AvgPool2d": ("avgpool2d", _pool),
    "MaxPool1d": ("maxpool1d", _pool),
    "AvgPool1d": ("avgpool1d", _pool),
    "AdaptiveAvgPool2d": (None, _adaptive_avg),
    "AdaptiveMaxPool2d": ("adaptiveMaxPool2d", lambda m: {"outputSize": _first(m.output_size)}),
    "Upsample": ("upsample", lambda m: {"scaleFactor": _first(m.scale_factor)}),
    "Dropout": ("dropout", lambda m: {"p": m.p}),
    "Dropout2d": ("dropout", lambda m: {"p": m.p}),
    "BatchNorm1d": ("batchNorm", lambda m: {"numFeatures": m.num_features}),
    "BatchNorm2d": ("batchNorm", lambda m: {"numFeatures": m.num_features}),
    "BatchNorm3d": ("batchNorm", lambda m: {"numFeatures": m.num_features}),
    "LayerNorm": ("layerNorm", _norm_shape),
    "RMSNorm": ("rmsNorm", _norm_shape),
    "GroupNorm": ("groupNorm", lambda m: {"numGroups": m.num_groups, "numChannels": m.num_channels}),
    "InstanceNorm2d": ("instanceNorm", lambda m: {"numFeatures": m.num_features}),
    "Embedding": ("embedding", lambda m: {"vocabSize": m.num_embeddings, "embeddingDim": m.embedding_dim}),
    "EmbeddingBag": ("embeddingBag", lambda m: {"vocabSize": m.num_embeddings, "embeddingDim": m.embedding_dim}),
    "MultiheadAttention": ("multiHeadAttention", lambda m: {"hiddenDim": m.embed_dim, "numHeads": m.num_heads}),
    "TransformerEncoderLayer": ("transformerBlock", lambda m: {
        "embedDim": m.self_attn.embed_dim, "numHeads": m.self_attn.num_heads, "ffDim": m.linear1.out_features}),
    "LSTM": ("lstm", _recurrent),
    "GRU": ("gru", _recurrent),
    "RNN": ("rnn", _recurrent),
    "Flatten": ("flatten", lambda m: {}),
    "ReLU": ("relu", lambda m: {}),
    "ReLU6": ("relu6", lambda m: {}),
    "LeakyReLU": ("leakyRelu", lambda m: {"negativeSlope": m.negative_slope}),
    "ELU": ("elu", lambda m: {}),
    "PReLU": ("prelu", lambda m: {"numParameters": m.num_parameters}),
    "SELU": ("selu", lambda m: {}),
    "GELU": ("gelu", lambda m: {}),
    "SiLU": ("swish", lambda m: {}),
    "Mish": ("mish", lambda m: {}),
    "Hardswish": ("hardSwish", lambda m: {}),
    "Hardsigmoid": ("hardSigmoid", lambda m: {}),
    "Sigmoid": ("sigmoid", lambda m: {}),
    "Tanh": ("tanh", lambda m: {}),
    "Softmax": ("softmax", lambda m: {}),
    "LogSoftmax": ("logSoftmax", lambda m: {}),
    "Softplus": ("softplus", lambda m: {}),
    "GLU": ("glu", lambda m: {}),
}

# Modules that pass their input through untouched. They get no node, so the
# tensor keeps pointing at its real producer and no fake hop appears.
PASSTHROUGH_CLASSES = {"Identity"}

# Copied from codeParser.ts CUSTOM_CLASS_MAP (lowercased class name -> type).
# Only consulted when --depth stops descent at a container the torch table
# does not know, so a `LlamaAttention` cut off at depth 3 is still an
# attention node rather than an anonymous custom box.
CUSTOM_CLASS_MAP: Dict[str, str] = {
    "positionalencoding": "positionalEncoding", "posencoding": "positionalEncoding",
    "posemb": "positionalEncoding", "positionalembedding": "positionalEncoding",
    "rotaryembedding": "rope", "ropeembedding": "rope",
    "multiheadattention": "multiHeadAttention", "multiheadattn": "multiHeadAttention",
    "causalselfattention": "causalAttention", "selfattn": "selfAttention",
    "selfattention": "selfAttention", "attention": "attention", "crossattention": "crossModalAttention",
    "llamaattention": "groupedQueryAttention", "mistralattention": "groupedQueryAttention",
    "mixtralattention": "groupedQueryAttention", "qwenattention": "groupedQueryAttention",
    "gemmaattention": "groupedQueryAttention", "phi3attention": "groupedQueryAttention",
    "falcon7battention": "groupedQueryAttention", "groupedqueryattention": "groupedQueryAttention",
    "gqaattention": "groupedQueryAttention",
    "feedforward": "feedForward", "feedforwardnetwork": "feedForward", "ffn": "feedForward", "mlp": "feedForward",
    "llamamlp": "swiglu", "mistralmpl": "swiglu", "mixtralmlp": "swiglu", "qwenmlp": "swiglu",
    "gemmamlp": "swiglu", "phi3mlp": "swiglu", "swiglu": "swiglu", "gatedmlp": "swiglu",
    "mixtralsparsemoeblock": "moeLayer", "moelayer": "moeLayer", "expertlayer": "moeLayer", "sparsemlp": "moeLayer",
    "block": "transformerBlock", "gptblock": "transformerBlock", "bertlayer": "transformerBlock",
    "bertblock": "transformerBlock", "encoderlayer": "transformerBlock", "decoderlayer": "transformerBlock",
    "transformerblock": "transformerBlock", "transformerlayer": "transformerBlock",
    "transformer": "transformerBlock", "visionblock": "transformerBlock",
    "llamadecoderlayer": "transformerBlock", "llamadecoderblock": "transformerBlock",
    "mistraldecoderlayer": "transformerBlock", "mixtraldecoderlayer": "transformerBlock",
    "qwendecoderlayer": "transformerBlock", "gemmadecoderlayer": "transformerBlock",
    "phi3decoderlayer": "transformerBlock",
    "visiontransformer": "transformerBlock", "vitblock": "transformerBlock", "vitlayer": "transformerBlock",
    "patchembed": "patchEmbed", "patchembedding": "patchEmbed", "patchprojection": "patchEmbed",
    "seblock": "seBlock", "squeezeexcitation": "seBlock", "channelattention": "seBlock",
    "resblock": "residual", "residualblock": "residual", "resnetblock": "residual", "bottleneck": "residual",
}

# Attribute names real-world blocks use for the two numbers the estimators
# need most. Read off a container only when it is cut off by --depth.
_DIM_ATTRS = ("embed_dim", "d_model", "hidden_size", "dim", "n_embd")
_HEAD_ATTRS = ("num_heads", "n_head", "nhead", "num_attention_heads")


def _sniff(m: nn.Module, names: Tuple[str, ...]) -> Optional[int]:
    for n in names:
        v = getattr(m, n, None)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


def map_module(m: nn.Module) -> Tuple[str, Params]:
    """Return (component type, params) for one module treated as a graph node."""
    cls = type(m).__name__
    entry = TORCH_CLASS_MAP.get(cls)
    if entry is not None:
        typ, extract = entry
        result = extract(m)
        return (typ, result) if typ is not None else result

    n_params = sum(p.numel() for p in m.parameters())
    params: Params = {"className": cls, "paramCount": n_params}
    typ = CUSTOM_CLASS_MAP.get(cls.lower(), "customModule")
    if typ != "customModule":
        d, h = _sniff(m, _DIM_ATTRS), _sniff(m, _HEAD_ATTRS)
        if d is not None:
            params["embedDim"] = d
        if h is not None:
            params["numHeads"] = h
    return typ, params
