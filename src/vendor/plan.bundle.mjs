// src/utils/tiedHead.ts
var num = (v) => typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null;
function tiedHeadSource(model) {
  const emb = model.components.find((c) => c.type === "embedding" && c.params?.tiedHead === true);
  if (!emb) return null;
  const p = emb.params;
  const vocab = num(p.vocabSize);
  const dim = num(p.embeddingDim);
  if (vocab == null || dim == null) return null;
  const hasHead = model.components.some((c) => {
    if (!/^(linear|lmHead)$/.test(c.type)) return false;
    const q = c.params ?? {};
    return num(q.outFeatures) === vocab;
  });
  return hasHead ? null : { embedding: emb, vocab, dim };
}
function withTiedHead(model, shapes) {
  const src = tiedHeadSource(model);
  if (!src) return { model, shapes, added: false };
  const hasChildren = new Set((model.connections ?? []).map((e) => e.from));
  const output = model.components.find((c) => c.type === "output");
  const feedEdge = output ? (model.connections ?? []).find((e) => e.to === output.id) : void 0;
  const feeder = feedEdge ? model.components.find((c) => c.id === feedEdge.from) : [...model.components].reverse().find((c) => c.type !== "stickyNote" && !hasChildren.has(c.id));
  if (!feeder) return { model, shapes, added: false };
  const inShape = shapes.get(feeder.id)?.out ?? null;
  if (!inShape || !inShape.length || inShape[inShape.length - 1] !== src.dim) return { model, shapes, added: false };
  const head = {
    id: `${src.embedding.id}::tied-head`,
    type: "linear",
    name: "LM_Head (tied)",
    position: { x: 0, y: 0 },
    params: { inFeatures: src.dim, outFeatures: src.vocab, bias: false, tied: true },
    inputs: [],
    outputs: []
  };
  const outShape = [...inShape.slice(0, -1), src.vocab];
  const connections = (model.connections ?? []).map((e) => feedEdge && e === feedEdge ? { ...e, from: head.id } : e);
  connections.push({ id: `${head.id}::in`, from: feeder.id, to: head.id, fromPort: "bottom", toPort: "top" });
  const at = output ? model.components.indexOf(output) : model.components.length;
  const components = [...model.components.slice(0, at), head, ...model.components.slice(at)];
  const next = new Map(shapes);
  next.set(head.id, { in: inShape, out: outShape });
  if (output) next.set(output.id, { ...shapes.get(output.id) ?? {}, in: outShape, out: outShape });
  return { model: { ...model, components, connections }, shapes: next, added: true };
}

// src/components/MLComponents/componentRegistry.ts
var convDim = (v, i, fallback) => {
  const raw = Array.isArray(v) ? v[i] ?? v[0] : v;
  const n2 = Number(raw);
  return Number.isFinite(n2) ? n2 : fallback;
};
var convOutLen = (len, params, i) => {
  const k = convDim(params.kernelSize, i, 3);
  const s = convDim(params.stride, i, 1);
  const p = convDim(params.padding, i, 0);
  const d = convDim(params.dilation, i, 1);
  return Math.floor((len + 2 * p - d * (k - 1) - 1) / s + 1);
};
var conv2dOutputShape = (inputShape, params) => {
  if (inputShape.length >= 3) {
    const [, h, w] = inputShape;
    const { outChannels } = params;
    return [outChannels, convOutLen(h, params, 0), convOutLen(w, params, 1)];
  }
  return inputShape;
};
var conv1dOutputShape = (inputShape, params) => {
  if (inputShape.length >= 2) {
    const l = inputShape[inputShape.length - 1];
    const { outChannels } = params;
    return [...inputShape.slice(0, -2), outChannels, convOutLen(l, params, 0)];
  }
  return inputShape;
};
var pool2dOutputShape = (inputShape, params) => {
  if (inputShape.length >= 3) {
    const [c, h, w] = inputShape;
    const p = { ...params, stride: params.stride ?? params.kernelSize };
    return [c, convOutLen(h, p, 0), convOutLen(w, p, 1)];
  }
  return inputShape;
};
var recurrentOutputShape = (inputShape, params, multiplier) => {
  const hidden = (params.hiddenSize || 128) * multiplier;
  if (params.returnSequences === false) return [hidden];
  if (inputShape.length >= 3) return [inputShape[0], inputShape[1], hidden];
  return [inputShape[0] || 1, hidden];
};
var componentRegistry = {
  // ========== Basic ==========
  input: {
    type: "input",
    name: "Input",
    icon: "\u{1F4E5}",
    category: "basic",
    defaultParams: { shape: [1, 28, 28] },
    computeOutputShape: (_, params) => params.shape || [1]
  },
  output: {
    type: "output",
    name: "Output",
    icon: "\u{1F4E4}",
    category: "basic",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  linear: {
    type: "linear",
    name: "Linear",
    icon: "\u26A1",
    category: "basic",
    defaultParams: { outFeatures: 128 },
    // nn.Linear maps the LAST dim only; leading dims (seq, tokens, ...) pass
    // through. Collapsing to [outFeatures] used to fire false blocking
    // merge-shape mismatches on residual + projection patterns.
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [...inputShape.slice(0, -1), params.outFeatures];
      }
      return [params.outFeatures];
    }
  },
  flatten: {
    type: "flatten",
    name: "Flatten",
    icon: "\u{1F4C4}",
    category: "basic",
    defaultParams: {},
    computeOutputShape: (inputShape) => {
      const total = inputShape.reduce((a, b) => a * b, 1);
      return [total];
    }
  },
  // ========== CV - Computer Vision ==========
  conv2d: {
    type: "conv2d",
    name: "Conv2D",
    icon: "\u{1F537}",
    category: "cv",
    defaultParams: { outChannels: 32, kernelSize: 3, stride: 1, padding: 1 },
    computeOutputShape: conv2dOutputShape
  },
  conv3d: {
    type: "conv3d",
    name: "Conv3D",
    icon: "\u{1F9CA}",
    category: "cv",
    defaultParams: { outChannels: 32, kernelSize: 3, stride: 1, padding: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 4) {
        const [, d, h, w] = inputShape;
        const { outChannels } = params;
        return [outChannels, convOutLen(d, params, 0), convOutLen(h, params, 1), convOutLen(w, params, 2)];
      }
      return inputShape;
    }
  },
  depthwiseConv2d: {
    type: "depthwiseConv2d",
    name: "DepthwiseConv2D",
    icon: "\u{1F539}",
    category: "cv",
    defaultParams: { depthMultiplier: 1, kernelSize: 3, stride: 1, padding: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c, h, w] = inputShape;
        const { depthMultiplier } = params;
        const outChannels = c * depthMultiplier;
        return [outChannels, convOutLen(h, params, 0), convOutLen(w, params, 1)];
      }
      return inputShape;
    }
  },
  separableConv2d: {
    type: "separableConv2d",
    name: "SeparableConv2D",
    icon: "\u{1F536}",
    category: "cv",
    defaultParams: { outChannels: 64, kernelSize: 3, stride: 1, padding: 1 },
    computeOutputShape: conv2dOutputShape
  },
  transposeConv2d: {
    type: "transposeConv2d",
    name: "TransposeConv2D",
    icon: "\u2B06\uFE0F",
    category: "cv",
    defaultParams: { outChannels: 32, kernelSize: 3, stride: 2, padding: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [, h, w] = inputShape;
        const { outChannels } = params;
        const tOut = (len, i) => {
          const k = convDim(params.kernelSize, i, 3);
          const s = convDim(params.stride, i, 1);
          const p = convDim(params.padding, i, 0);
          const d = convDim(params.dilation, i, 1);
          const op = convDim(params.outputPadding, i, 0);
          return (len - 1) * s - 2 * p + d * (k - 1) + op + 1;
        };
        return [outChannels, tOut(h, 0), tOut(w, 1)];
      }
      return inputShape;
    }
  },
  maxpool2d: {
    type: "maxpool2d",
    name: "MaxPool2D",
    icon: "\u2B07\uFE0F",
    category: "cv",
    defaultParams: { kernelSize: 2, stride: 2, padding: 0 },
    computeOutputShape: pool2dOutputShape
  },
  avgpool2d: {
    type: "avgpool2d",
    name: "AvgPool2D",
    icon: "\u{1F4CA}",
    category: "cv",
    defaultParams: { kernelSize: 2, stride: 2, padding: 0 },
    computeOutputShape: pool2dOutputShape
  },
  adaptiveMaxPool2d: {
    type: "adaptiveMaxPool2d",
    name: "AdaptiveMaxPool2D",
    icon: "\u{1F4D0}",
    category: "cv",
    defaultParams: { outputSize: [1, 1] },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c] = inputShape;
        const raw = params.outputSize ?? [1, 1];
        const outputSize = Array.isArray(raw) ? raw : [raw, raw];
        return [c, ...outputSize];
      }
      return inputShape;
    }
  },
  adaptiveAvgPool2d: {
    type: "adaptiveAvgPool2d",
    name: "AdaptiveAvgPool2D",
    icon: "\u{1F4D0}",
    category: "cv",
    defaultParams: { outputSize: [1, 1] },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c] = inputShape;
        const raw = params.outputSize ?? [1, 1];
        const outputSize = Array.isArray(raw) ? raw : [raw, raw];
        return [c, ...outputSize];
      }
      return inputShape;
    }
  },
  globalAvgPool2d: {
    type: "globalAvgPool2d",
    name: "GlobalAvgPool2D",
    icon: "\u{1F310}",
    category: "cv",
    defaultParams: {},
    computeOutputShape: (inputShape) => {
      if (inputShape.length >= 3) {
        const [c] = inputShape;
        return [c];
      }
      return inputShape;
    }
  },
  upsample: {
    type: "upsample",
    name: "Upsample",
    icon: "\u{1F53A}",
    category: "cv",
    defaultParams: { scaleFactor: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c, h, w] = inputShape;
        const scale = params.scaleFactor || 2;
        return [c, h * scale, w * scale];
      }
      return inputShape;
    }
  },
  pixelShuffle: {
    type: "pixelShuffle",
    name: "PixelShuffle",
    icon: "\u{1F532}",
    category: "cv",
    defaultParams: { upscaleFactor: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c, h, w] = inputShape;
        const r = params.upscaleFactor || 2;
        return [Math.floor(c / (r * r)), h * r, w * r];
      }
      return inputShape;
    }
  },
  // ========== NLP - Natural Language Processing ==========
  conv1d: {
    type: "conv1d",
    name: "Conv1D",
    icon: "\u{1F538}",
    category: "nlp",
    defaultParams: { outChannels: 64, kernelSize: 3, stride: 1, padding: 1 },
    computeOutputShape: conv1dOutputShape
  },
  maxpool1d: {
    type: "maxpool1d",
    name: "MaxPool1D",
    icon: "\u2B07\uFE0F",
    category: "nlp",
    defaultParams: { kernelSize: 2, stride: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        const [c, l] = inputShape;
        return [c, convOutLen(l, { ...params, stride: params.stride ?? params.kernelSize }, 0)];
      }
      return inputShape;
    }
  },
  avgpool1d: {
    type: "avgpool1d",
    name: "AvgPool1D",
    icon: "\u{1F4CA}",
    category: "nlp",
    defaultParams: { kernelSize: 2, stride: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        const [c, l] = inputShape;
        return [c, convOutLen(l, { ...params, stride: params.stride ?? params.kernelSize }, 0)];
      }
      return inputShape;
    }
  },
  embedding: {
    type: "embedding",
    name: "Embedding",
    icon: "\u{1F524}",
    category: "nlp",
    defaultParams: { numEmbeddings: 3e4, embeddingDim: 128 },
    // Embedding consumes integer token ids and APPENDS an embed dim:
    // [seq] -> [seq, dim], [1, seq] -> [1, seq, dim]. The old rank>=2 branch
    // replaced the last dim, silently deleting the sequence axis.
    computeOutputShape: (inputShape, params) => {
      const dim = params.embeddingDim || params.embedDim || 128;
      return [...inputShape, dim];
    }
  },
  segmentEmbedding: {
    type: "segmentEmbedding",
    name: "Segment Embedding",
    icon: "\u{1FAAA}",
    category: "nlp",
    defaultParams: { numSegments: 2, embeddingDim: 768 },
    // Adds a segment lookup to existing hidden states: shape-preserving.
    // Overwriting the last dim with embeddingDim would mask real width
    // mismatches (same bug the plain embedding/positionalEncoding rules had).
    computeOutputShape: (inputShape) => [...inputShape]
  },
  lstm: {
    type: "lstm",
    name: "LSTM",
    icon: "\u{1F504}",
    category: "nlp",
    defaultParams: { hiddenSize: 128, numLayers: 1, bidirectional: false },
    computeOutputShape: (inputShape, params) => recurrentOutputShape(inputShape, params, params.bidirectional ? 2 : 1)
  },
  gru: {
    type: "gru",
    name: "GRU",
    icon: "\u{1F500}",
    category: "nlp",
    defaultParams: { hiddenSize: 128, numLayers: 1, bidirectional: false },
    computeOutputShape: (inputShape, params) => recurrentOutputShape(inputShape, params, params.bidirectional ? 2 : 1)
  },
  rnn: {
    type: "rnn",
    name: "RNN",
    icon: "\u21A9\uFE0F",
    category: "nlp",
    defaultParams: { hiddenSize: 128, numLayers: 1 },
    computeOutputShape: (inputShape, params) => recurrentOutputShape(inputShape, params, 1)
  },
  bidirectionalLSTM: {
    type: "bidirectionalLSTM",
    name: "BiLSTM",
    icon: "\u2194\uFE0F",
    category: "nlp",
    defaultParams: { hiddenSize: 128, numLayers: 1 },
    computeOutputShape: (inputShape, params) => recurrentOutputShape(inputShape, params, 2)
  },
  bidirectionalGRU: {
    type: "bidirectionalGRU",
    name: "BiGRU",
    icon: "\u2194\uFE0F",
    category: "nlp",
    defaultParams: { hiddenSize: 128, numLayers: 1 },
    computeOutputShape: (inputShape, params) => {
      const hiddenSize = params.hiddenSize || 128;
      if (inputShape.length >= 3) {
        return [inputShape[0], inputShape[1], hiddenSize * 2];
      }
      return [inputShape[0] || 1, hiddenSize * 2];
    }
  },
  attention: {
    type: "attention",
    name: "Attention",
    icon: "\u{1F441}\uFE0F",
    category: "nlp",
    defaultParams: { embedDim: 128, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  selfAttention: {
    type: "selfAttention",
    name: "Self-Attention",
    icon: "\u{1F50D}",
    category: "nlp",
    defaultParams: { embedDim: 128, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  crossAttention: {
    type: "crossAttention",
    name: "Cross-Attention",
    icon: "\u{1F500}",
    category: "nlp",
    defaultParams: { embedDim: 512, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  globalAvgPool1d: {
    type: "globalAvgPool1d",
    name: "GlobalAvgPool1D",
    icon: "\u{1F310}",
    category: "nlp",
    defaultParams: {},
    computeOutputShape: (inputShape) => {
      if (inputShape.length >= 2) return [inputShape[0]];
      return inputShape;
    }
  },
  depthwiseConv1d: {
    type: "depthwiseConv1d",
    name: "DepthwiseConv1D",
    icon: "\u{1F539}",
    category: "nlp",
    defaultParams: { channels: 64, kernelSize: 31, padding: 15, stride: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        const [c, l] = inputShape;
        const { kernelSize, stride = 1, padding = 0 } = params;
        return [c, Math.floor((l + 2 * padding - kernelSize) / stride + 1)];
      }
      return inputShape;
    }
  },
  // ========== LLM - Large Language Models ==========
  groupedQueryAttention: {
    type: "groupedQueryAttention",
    name: "Grouped Query Attn",
    icon: "\u{1F465}",
    // flashAttention keeps attention memory O(seq) at prefill (no N×N score
    // matrix materialised); it changes runtime memory, not the math/shapes.
    category: "llm",
    defaultParams: { embedDim: 4096, numHeads: 32, numKVHeads: 8, flashAttention: true },
    computeOutputShape: (inputShape) => inputShape
  },
  causalAttention: {
    type: "causalAttention",
    name: "Causal Attention",
    icon: "\u{1F512}",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  multiHeadAttention: {
    type: "multiHeadAttention",
    name: "Multi-Head Attention",
    icon: "\u{1F3AF}",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, flashAttention: false },
    computeOutputShape: (inputShape) => inputShape
  },
  localAttention: {
    type: "localAttention",
    name: "Local Attention",
    icon: "\u{1FA9F}",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, windowSize: 256 },
    computeOutputShape: (inputShape) => inputShape
  },
  linearAttention: {
    type: "linearAttention",
    name: "Linear Attention",
    icon: "\u{1F4C8}",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, featureMap: "elu" },
    computeOutputShape: (inputShape) => inputShape
  },
  transformerBlock: {
    type: "transformerBlock",
    name: "Transformer Block",
    icon: "\u2699\uFE0F",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, ffDim: 2048 },
    computeOutputShape: (inputShape) => inputShape
  },
  positionalEncoding: {
    type: "positionalEncoding",
    name: "Positional Encoding",
    icon: "\u{1F4CD}",
    category: "llm",
    defaultParams: { maxLen: 512, embedDim: 512 },
    // Adding a positional encoding never changes the tensor shape. The old
    // behaviour overwrote the last dim with the layer's own embedDim, which
    // MASKED real width mismatches downstream instead of surfacing them
    // (the shape gate compares upstream width against attention/linear dims,
    // so passing the true input through lets that check fire).
    computeOutputShape: (inputShape) => inputShape
  },
  learnedPositionalEmbedding: {
    type: "learnedPositionalEmbedding",
    name: "Learned Pos Embed",
    icon: "\u{1F4CD}",
    category: "llm",
    defaultParams: { maxLen: 512, embedDim: 768 },
    // Added to existing hidden states, like positionalEncoding: shape-preserving.
    computeOutputShape: (inputShape) => [...inputShape]
  },
  feedForward: {
    type: "feedForward",
    name: "Feed Forward",
    icon: "\u27A1\uFE0F",
    category: "nlp",
    defaultParams: { embedDim: 512, ffDim: 2048 },
    computeOutputShape: (inputShape, params) => {
      const last = inputShape.length ? inputShape[inputShape.length - 1] : void 0;
      const out = params.embedDim ?? params.hiddenDim ?? params.dModel ?? last ?? 512;
      return inputShape.length >= 2 ? [...inputShape.slice(0, -1), out] : [out];
    }
  },
  rope: {
    type: "rope",
    name: "RoPE",
    icon: "\u{1F300}",
    category: "llm",
    // scalingType extends usable context past the training length:
    //   none        — vanilla RoPE, no extrapolation
    //   linear      — Positional Interpolation (PI): divide positions by factor
    //   dynamic-ntk — NTK-aware dynamic scaling (adjusts base by factor)
    //   yarn        — YaRN (NTK-by-parts), best long-context retention
    // originalMaxPos is the training context; effective context ≈ originalMaxPos × scalingFactor.
    defaultParams: { dim: 128, scalingType: "none", scalingFactor: 1, originalMaxPos: 2048 },
    computeOutputShape: (inputShape) => inputShape
  },
  rmsNorm: {
    type: "rmsNorm",
    name: "RMSNorm",
    icon: "\u{1F4D0}",
    category: "normalization",
    defaultParams: { normalizedShape: 512 },
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== Audio ==========
  melSpectrogram: {
    type: "melSpectrogram",
    name: "Mel Spectrogram",
    icon: "\u{1F3B5}",
    category: "audio",
    defaultParams: { nMelBands: 80, hopLength: 512 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [inputShape[0], params.nMelBands || 80, Math.floor(inputShape[1] / (params.hopLength || 512))];
      }
      return [params.nMelBands || 80, 128];
    }
  },
  mfcc: {
    type: "mfcc",
    name: "MFCC",
    icon: "\u{1F3BC}",
    category: "audio",
    defaultParams: { nMFCC: 13 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [inputShape[0], params.nMFCC || 13, Math.floor(inputShape[1] / 512)];
      }
      return [params.nMFCC || 13, 128];
    }
  },
  stft: {
    type: "stft",
    name: "STFT",
    icon: "\u{1F4FB}",
    category: "audio",
    defaultParams: { nFFT: 2048, hopLength: 512 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        const nFFT = params.nFFT || 2048;
        const timeFrames = Math.floor(inputShape[1] / (params.hopLength || 512));
        return [inputShape[0], Math.floor(nFFT / 2) + 1, timeFrames];
      }
      return [1025, 128];
    }
  },
  audioConv: {
    type: "audioConv",
    name: "Audio Conv",
    icon: "\u{1F50A}",
    category: "audio",
    defaultParams: { outChannels: 64, kernelSize: 3, stride: 1 },
    computeOutputShape: conv1dOutputShape
  },
  conformerBlock: {
    type: "conformerBlock",
    name: "Conformer Block",
    icon: "\u{1F399}\uFE0F",
    category: "audio",
    defaultParams: { dModel: 256, numHeads: 4, ffMult: 4, kernelSize: 31 },
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== Tabular ==========
  featureInteraction: {
    type: "featureInteraction",
    name: "Feature Interaction",
    icon: "\u{1F517}",
    category: "tabular",
    defaultParams: { interactionDim: 64 },
    computeOutputShape: (inputShape, params, allInputShapes) => {
      const dense = allInputShapes?.[0] ?? inputShape;
      const table = allInputShapes?.[1];
      const width = dense[dense.length - 1];
      if (table && table.length >= 2 && typeof width === "number" && width > 0) {
        const n2 = table[0] + 1;
        const pairs = n2 * (n2 - 1) / 2;
        return [width + pairs];
      }
      return [inputShape[0] || 1, params.interactionDim || 64];
    }
  },
  embeddingBag: {
    type: "embeddingBag",
    name: "EmbeddingBag",
    icon: "\u{1F392}",
    category: "tabular",
    defaultParams: { vocabSize: 1e3, embedDim: 32 },
    computeOutputShape: (inputShape, params) => {
      const dim = params.embedDim ?? params.embeddingDim ?? 32;
      return [inputShape[0] || 1, dim];
    }
  },
  tabnet: {
    type: "tabnet",
    name: "TabNet",
    icon: "\u{1F4CB}",
    category: "tabular",
    defaultParams: { featureDim: 64, decisionDim: 64 },
    computeOutputShape: (inputShape, params) => {
      return [inputShape[0] || 1, params.decisionDim || 64];
    }
  },
  // ========== Recommendation: sequential / generative (2018-2025) ==========
  // The nine shipped recommendation templates all predate 2020 and all model a
  // user as ONE vector produced by a permutation-invariant or fixed-window
  // encoder. The four nodes below are the structures the field actually moved
  // to, and each one changes a shape the downstream layers have to agree with,
  // which is the only reason a node earns a place here rather than a doc page.
  targetAttention: {
    type: "targetAttention",
    name: "Target Attention (DIN)",
    icon: "\u{1F3AF}",
    category: "tabular",
    // DIN's local activation unit: the CANDIDATE item is the query and the
    // behaviour sequence is both key and value, so the sequence axis is
    // consumed and the layer hands on one interest vector per candidate.
    // That is what makes it different from selfAttention (shape-preserving)
    // and from crossAttention (keeps the query's own length).
    defaultParams: { embedDim: 64, hiddenDim: 36, activation: "dice" },
    computeOutputShape: (inputShape, params, allInputShapes) => {
      const ranked = (allInputShapes ?? []).filter((s) => s.length >= 2);
      const seq = ranked.sort((a, b) => b[b.length - 2] - a[a.length - 2])[0] ?? inputShape;
      const dim = seq.length >= 1 ? seq[seq.length - 1] : params.embedDim ?? 64;
      return seq.length >= 2 ? [...seq.slice(0, -2), dim] : [dim];
    }
  },
  behaviorRetrieval: {
    type: "behaviorRetrieval",
    name: "Long-Sequence Retrieval (SIM/ETA GSU)",
    icon: "\u{1F50E}",
    category: "tabular",
    // The general search unit of SIM / ETA: a lifelong behaviour sequence
    // (10^4 events) is cut to topK before anything quadratic touches it. The
    // whole point is that the ESU downstream sees topK, not L, so the cost
    // arithmetic on this graph is wrong by orders of magnitude if the shape
    // does not say so. Hard mode (category match) has no parameters at all.
    defaultParams: { topK: 50, embedDim: 64, mode: "soft" },
    computeOutputShape: (inputShape, params) => {
      const k = Math.max(1, Number(params.topK) || 50);
      if (inputShape.length >= 2) {
        const out = [...inputShape];
        out[out.length - 2] = k;
        return out;
      }
      return inputShape;
    }
  },
  multiInterest: {
    type: "multiInterest",
    name: "Multi-Interest Extractor (MIND/ComiRec)",
    icon: "\u{1F9ED}",
    category: "tabular",
    // One user becomes K vectors, not one. Every downstream layer that assumed
    // a single user embedding (a dot product against an item, a Linear on the
    // last dim) is now operating on a rank it did not expect, which is exactly
    // the class of break the shape gate exists to catch.
    defaultParams: { embedDim: 64, numInterests: 4, numIterations: 3 },
    computeOutputShape: (inputShape, params) => {
      const k = Math.max(1, Number(params.numInterests) || 4);
      const dim = inputShape.length >= 1 ? inputShape[inputShape.length - 1] : params.embedDim ?? 64;
      return inputShape.length >= 2 ? [...inputShape.slice(0, -2), k, dim] : [k, dim];
    }
  },
  // ========== Representation learning / retrieval objectives ==========
  // The two-tower template shipped with an encoder on each side and nothing at
  // all where the objective goes, which is the half of a retrieval model that
  // decides what the embeddings mean. These three are the ends that exist in
  // practice, and each one changes something a downstream layer has to agree
  // with rather than being a loss function in disguise.
  contrastiveHead: {
    type: "contrastiveHead",
    name: "Contrastive Head (InfoNCE / CLIP)",
    icon: "\u{1F39A}\uFE0F",
    category: "multimodal",
    // Projects to the shared space, L2-normalises, and owns the temperature.
    // The normalisation is the part that matters downstream: after it, a dot
    // product is a cosine and is bounded, which is what makes the temperature
    // meaningful at all.
    defaultParams: { projDim: 512, temperature: 0.07, learnableTemp: true, normalize: true },
    computeOutputShape: (inputShape, params) => {
      const d = Number(params.projDim) || 512;
      return inputShape.length >= 1 ? [...inputShape.slice(0, -1), d] : [d];
    }
  },
  matryoshkaHead: {
    type: "matryoshkaHead",
    name: "Matryoshka Head (MRL)",
    icon: "\u{1FA86}",
    category: "multimodal",
    // Adds NO parameters. It is a promise about the embedding above it: every
    // listed prefix length is independently usable, so a caller can truncate
    // instead of running a second model. The promise is only well formed if
    // every nested dim actually fits inside the embedding, which is what the
    // matryoshka-dims-exceed-width rule checks.
    defaultParams: { embedDim: 768, nestedDims: [768, 512, 256, 128, 64] },
    computeOutputShape: (inputShape, params) => {
      const d = Number(params.embedDim) || (inputShape.length >= 1 ? inputShape[inputShape.length - 1] : 768);
      return inputShape.length >= 1 ? [...inputShape.slice(0, -1), d] : [d];
    }
  },
  lateInteraction: {
    type: "lateInteraction",
    name: "Late Interaction (ColBERT MaxSim)",
    icon: "\u{1F517}",
    category: "multimodal",
    // Scores a query sequence against a document sequence by max-similarity per
    // query token, summed. BOTH sequence axes are consumed and one number comes
    // out, which is the whole difference from a single-vector retriever: the
    // document keeps its tokens until scoring time.
    defaultParams: { embedDim: 128, similarity: "cosine", docLen: 180 },
    computeOutputShape: () => [1]
  },
  hstuBlock: {
    type: "hstuBlock",
    name: "HSTU Block",
    icon: "\u{1F4C8}",
    category: "tabular",
    // Hierarchical Sequential Transduction Unit (Zhai et al. 2024). Shape
    // preserving like a transformer block, but the pointwise aggregated
    // attention has no softmax and the projections are split u/v/q/k, so the
    // parameter and FLOP arithmetic is NOT a transformer block's and must not
    // be approximated with one.
    defaultParams: { embedDim: 512, numHeads: 4, linearDim: 128, attnDim: 128 },
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== Reinforcement Learning ==========
  dqnHead: {
    type: "dqnHead",
    name: "DQN Head",
    icon: "\u{1F3AE}",
    category: "rl",
    defaultParams: { numActions: 4 },
    computeOutputShape: (inputShape, params) => {
      return [inputShape[0] || 1, params.numActions || 4];
    }
  },
  actorHead: {
    type: "actorHead",
    name: "Actor Head",
    icon: "\u{1F3AD}",
    category: "rl",
    defaultParams: { numActions: 4 },
    computeOutputShape: (inputShape, params) => {
      return [inputShape[0] || 1, params.numActions || 4];
    }
  },
  criticHead: {
    type: "criticHead",
    name: "Critic Head",
    icon: "\u2B50",
    category: "rl",
    defaultParams: {},
    computeOutputShape: (inputShape) => {
      return [inputShape[0] || 1, 1];
    }
  },
  policyNetwork: {
    type: "policyNetwork",
    name: "Policy Network",
    icon: "\u{1F4DC}",
    category: "rl",
    defaultParams: { numActions: 4 },
    computeOutputShape: (inputShape, params) => {
      return [inputShape[0] || 1, params.numActions || 4];
    }
  },
  valueNetwork: {
    type: "valueNetwork",
    name: "Value Network",
    icon: "\u{1F4B0}",
    category: "rl",
    defaultParams: {},
    computeOutputShape: (inputShape) => {
      return [inputShape[0] || 1, 1];
    }
  },
  // ========== Graph ML ==========
  graphConv: {
    type: "graphConv",
    name: "GraphConv",
    icon: "\u{1F578}\uFE0F",
    category: "graph",
    defaultParams: { outFeatures: 64 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [inputShape[0], params.outFeatures || 64];
      }
      return [inputShape[0] || 1, params.outFeatures || 64];
    }
  },
  graphAttention: {
    type: "graphAttention",
    name: "Graph Attention",
    icon: "\u{1F577}\uFE0F",
    category: "graph",
    defaultParams: { outFeatures: 64, numHeads: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [inputShape[0], params.outFeatures || 64];
      }
      return [inputShape[0] || 1, params.outFeatures || 64];
    }
  },
  graphSAGE: {
    type: "graphSAGE",
    name: "GraphSAGE",
    icon: "\u{1F4CA}",
    category: "graph",
    defaultParams: { outFeatures: 64 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [inputShape[0], params.outFeatures || 64];
      }
      return [inputShape[0] || 1, params.outFeatures || 64];
    }
  },
  gcn: {
    type: "gcn",
    name: "GCN",
    icon: "\u{1F517}",
    category: "graph",
    defaultParams: { outFeatures: 64 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [inputShape[0], params.outFeatures || 64];
      }
      return [inputShape[0] || 1, params.outFeatures || 64];
    }
  },
  gin: {
    type: "gin",
    name: "GIN",
    icon: "\u{1F52E}",
    category: "graph",
    defaultParams: { outFeatures: 64, eps: 0 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) return [inputShape[0], params.outFeatures || 64];
      return inputShape;
    }
  },
  gat: {
    type: "gat",
    name: "GAT",
    icon: "\u{1F441}\uFE0F\u200D\u{1F5E8}\uFE0F",
    category: "graph",
    defaultParams: { outFeatures: 64, numHeads: 8 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        const outDim = params.outFeatures || 64;
        const heads = params.numHeads || 8;
        return [inputShape[0], outDim * heads];
      }
      return [inputShape[0] || 1, (params.outFeatures || 64) * (params.numHeads || 8)];
    }
  },
  edgeConv: {
    type: "edgeConv",
    name: "EdgeConv",
    icon: "\u2702\uFE0F",
    category: "graph",
    defaultParams: { outFeatures: 64, k: 20 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) return [inputShape[0], params.outFeatures || 64];
      return [inputShape[0] || 1, params.outFeatures || 64];
    }
  },
  // ========== Multimodal ==========
  patchMerger: {
    type: "patchMerger",
    name: "Patch Merger (Qwen-VL / InternVL)",
    icon: "\u{1F9F1}",
    category: "multimodal",
    // The token-count reducer every any-resolution VLM uses: a k x k spatial
    // group of visual tokens becomes ONE token k^2 times as wide, then a small
    // MLP maps that to the language model's width. Both axes move at once and
    // in opposite directions, which is why this is the most shape-error-prone
    // node in a VLM and why it is not `pixelShuffle` (that one is [C,H,W]
    // upsampling, a different op on a different rank).
    defaultParams: { mergeSize: 2, inDim: 1280, outDim: 3584 },
    computeOutputShape: (inputShape, params) => {
      const k = Math.max(1, Number(params.mergeSize) || 2);
      const out = Number(params.outDim) || 3584;
      if (inputShape.length >= 2) {
        const n2 = Math.max(1, Math.floor(inputShape[inputShape.length - 2] / (k * k)));
        return [...inputShape.slice(0, -2), n2, out];
      }
      return [out];
    }
  },
  crossModalAttention: {
    type: "crossModalAttention",
    name: "Cross-Modal Attention",
    icon: "\u{1F500}",
    category: "multimodal",
    defaultParams: { embedDim: 256, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  fusion: {
    type: "fusion",
    name: "Fusion",
    icon: "\u{1F501}",
    category: "multimodal",
    defaultParams: { fusionDim: 256, method: "concat" },
    computeOutputShape: (inputShape, params) => {
      if (params.method === "concat") {
        return [inputShape[0] || 1, params.fusionDim || 256];
      }
      return inputShape;
    }
  },
  projection: {
    type: "projection",
    name: "Projection",
    icon: "\u27A1\uFE0F",
    category: "multimodal",
    defaultParams: { outDim: 256 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [...inputShape.slice(0, -1), params.outDim || 256];
      }
      return [params.outDim || 256];
    }
  },
  coAttention: {
    type: "coAttention",
    name: "Co-Attention",
    icon: "\u{1F501}",
    category: "multimodal",
    defaultParams: { embedDim: 512, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== Activation ==========
  relu: {
    type: "relu",
    name: "ReLU",
    icon: "\u26A1",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  relu6: {
    type: "relu6",
    name: "ReLU6",
    icon: "\u26A1",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  softplus: {
    type: "softplus",
    name: "Softplus",
    icon: "\u301C",
    category: "activation",
    defaultParams: { beta: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  leakyRelu: {
    type: "leakyRelu",
    name: "LeakyReLU",
    icon: "\u26A1",
    category: "activation",
    defaultParams: { negativeSlope: 0.01 },
    computeOutputShape: (inputShape) => inputShape
  },
  elu: {
    type: "elu",
    name: "ELU",
    icon: "\u26A1",
    category: "activation",
    defaultParams: { alpha: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  prelu: {
    type: "prelu",
    name: "PReLU",
    icon: "\u26A1",
    category: "activation",
    defaultParams: { numParameters: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  gelu: {
    type: "gelu",
    name: "GELU",
    icon: "\u2728",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  swish: {
    type: "swish",
    name: "Swish",
    icon: "\u{1F4AB}",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  sigmoid: {
    type: "sigmoid",
    name: "Sigmoid",
    icon: "\u{1F4C8}",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  tanh: {
    type: "tanh",
    name: "Tanh",
    icon: "\u{1F30A}",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  softmax: {
    type: "softmax",
    name: "Softmax",
    icon: "\u{1F4CA}",
    category: "activation",
    defaultParams: { dim: -1 },
    computeOutputShape: (inputShape) => inputShape
  },
  gumbelSoftmax: {
    type: "gumbelSoftmax",
    name: "Gumbel-Softmax",
    icon: "\u{1F3B2}",
    category: "activation",
    defaultParams: { tau: 1, hard: false, dim: -1 },
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== Activation extras ==========
  selu: {
    type: "selu",
    name: "SELU",
    icon: "\u26A1",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  mish: {
    type: "mish",
    name: "Mish",
    icon: "\u301C",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  hardSwish: {
    type: "hardSwish",
    name: "HardSwish",
    icon: "\u2312",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  hardSigmoid: {
    type: "hardSigmoid",
    name: "HardSigmoid",
    icon: "\u{1F4C8}",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  logSoftmax: {
    type: "logSoftmax",
    name: "LogSoftmax",
    icon: "\u{1F4C9}",
    category: "activation",
    defaultParams: { dim: -1 },
    computeOutputShape: (inputShape) => inputShape
  },
  glu: {
    type: "glu",
    name: "GLU",
    icon: "\u{1F6AA}",
    category: "activation",
    defaultParams: { dim: -1 },
    computeOutputShape: (inputShape, params) => {
      const dim = params.dim ?? -1;
      const idx = dim < 0 ? inputShape.length + dim : dim;
      const out = [...inputShape];
      out[idx] = Math.floor(inputShape[idx] / 2);
      return out;
    }
  },
  // ========== Normalization ==========
  batchNorm: {
    type: "batchNorm",
    name: "BatchNorm",
    icon: "\u{1F4CF}",
    category: "normalization",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  layerNorm: {
    type: "layerNorm",
    name: "LayerNorm",
    icon: "\u{1F4D0}",
    category: "normalization",
    defaultParams: { normalizedShape: [512] },
    computeOutputShape: (inputShape) => inputShape
  },
  instanceNorm: {
    type: "instanceNorm",
    name: "InstanceNorm",
    icon: "\u{1F4D0}",
    category: "normalization",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  groupNorm: {
    type: "groupNorm",
    name: "GroupNorm",
    icon: "\u{1F4D1}",
    category: "normalization",
    defaultParams: { numGroups: 32 },
    computeOutputShape: (inputShape) => inputShape
  },
  adaIN: {
    type: "adaIN",
    name: "AdaIN",
    icon: "\u{1F3A8}",
    category: "normalization",
    defaultParams: { numFeatures: 512 },
    computeOutputShape: (inputShape) => inputShape
  },
  spectralNorm: {
    type: "spectralNorm",
    name: "SpectralNorm",
    icon: "\u{1F4E1}",
    category: "normalization",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  weightNorm: {
    type: "weightNorm",
    name: "WeightNorm",
    icon: "\u2696\uFE0F",
    category: "normalization",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  localResponseNorm: {
    type: "localResponseNorm",
    name: "LocalResponseNorm",
    icon: "\u{1F4CA}",
    category: "normalization",
    defaultParams: { size: 5, alpha: 1e-4, beta: 0.75, k: 2 },
    computeOutputShape: (inputShape) => inputShape
  },
  pixelNorm: {
    type: "pixelNorm",
    name: "PixelNorm",
    icon: "\u{1F3A8}",
    category: "normalization",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== LLM extras ==========
  swiglu: {
    type: "swiglu",
    name: "SwiGLU",
    icon: "\u26A1",
    category: "llm",
    defaultParams: { embedDim: 4096, intermediateSize: 11008 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 1) {
        return [...inputShape.slice(0, -1), params.embedDim ?? inputShape[inputShape.length - 1]];
      }
      return inputShape;
    }
  },
  moeLayer: {
    type: "moeLayer",
    name: "MoE Layer",
    icon: "\u{1F500}",
    category: "llm",
    defaultParams: { embedDim: 4096, numExperts: 8, topK: 2, expertDim: 14336 },
    // Shape passes through, but a router asked for more experts than exist is
    // a guaranteed config bug worth blocking at design time.
    computeOutputShape: (inputShape, params) => {
      const k = Number(params.topK ?? 2);
      const e = Number(params.numExperts ?? 8);
      if (Number.isFinite(k) && Number.isFinite(e) && k > e) {
        throw new Error(`moeLayer: topK (${k}) exceeds numExperts (${e})`);
      }
      return inputShape;
    }
  },
  alibi: {
    type: "alibi",
    name: "ALiBi",
    icon: "\u{1F4D0}",
    category: "llm",
    defaultParams: { numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  lmHead: {
    type: "lmHead",
    name: "LM Head",
    icon: "\u{1F5E3}\uFE0F",
    category: "llm",
    defaultParams: { vocabSize: 50257 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [...inputShape.slice(0, -1), params.vocabSize || 50257];
      }
      return [params.vocabSize || 50257];
    }
  },
  timeEmbedding: {
    type: "timeEmbedding",
    name: "Time Embedding",
    icon: "\u23F1\uFE0F",
    category: "llm",
    defaultParams: { dim: 256 },
    computeOutputShape: (_inputShape, params) => [params.dim || 256]
  },
  mamba: {
    type: "mamba",
    name: "Mamba (SSM)",
    icon: "\u{1F40D}",
    category: "llm",
    defaultParams: { dModel: 256, dState: 16, dConv: 4, expand: 2 },
    computeOutputShape: (inputShape) => inputShape
  },
  relativePositionBias: {
    type: "relativePositionBias",
    name: "Relative Position Bias",
    icon: "\u{1F4CD}",
    category: "llm",
    defaultParams: { numHeads: 8, numBuckets: 32, maxDistance: 128 },
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== CV extras ==========
  fpn: {
    type: "fpn",
    name: "FPN",
    icon: "\u{1F53A}",
    category: "cv",
    defaultParams: { inChannels: [256, 512, 1024, 2048], outChannels: 256 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        return [params.outChannels || 256, inputShape[1], inputShape[2]];
      }
      return [params.outChannels || 256];
    }
  },
  invResidualBlock: {
    type: "invResidualBlock",
    name: "Inv. Residual (MBConv)",
    icon: "\u{1F4F1}",
    category: "cv",
    defaultParams: { inChannels: 32, outChannels: 32, expandRatio: 6, stride: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [, h, w] = inputShape;
        const stride = params.stride || 1;
        return [params.outChannels || inputShape[0], Math.floor(h / stride), Math.floor(w / stride)];
      }
      return inputShape;
    }
  },
  deformableConv2d: {
    type: "deformableConv2d",
    name: "DeformableConv2D",
    icon: "\u{1F300}",
    category: "cv",
    defaultParams: { outChannels: 64, kernelSize: 3, stride: 1, padding: 1 },
    computeOutputShape: conv2dOutputShape
  },
  interpolate: {
    type: "interpolate",
    name: "Interpolate",
    icon: "\u{1F53A}",
    category: "cv",
    defaultParams: { scaleFactor: 2, mode: "bilinear" },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c, h, w] = inputShape;
        if (params.size) {
          const size = Array.isArray(params.size) ? params.size : [params.size, params.size];
          return [c, size[0], size[1]];
        }
        const scale = params.scaleFactor || 2;
        return [c, Math.floor(h * scale), Math.floor(w * scale)];
      }
      return inputShape;
    }
  },
  channelShuffle: {
    type: "channelShuffle",
    name: "Channel Shuffle",
    icon: "\u{1F500}",
    category: "cv",
    defaultParams: { groups: 2 },
    computeOutputShape: (inputShape) => inputShape
  },
  gridSample: {
    type: "gridSample",
    name: "Grid Sample",
    icon: "\u{1F5FA}\uFE0F",
    category: "cv",
    defaultParams: { mode: "bilinear", paddingMode: "zeros", alignCorners: false },
    computeOutputShape: (inputShape) => inputShape
  },
  spatialPyramidPool: {
    type: "spatialPyramidPool",
    name: "Spatial Pyramid Pool",
    icon: "\u{1F53A}",
    category: "cv",
    defaultParams: { levels: [1, 2, 4] },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const c = inputShape[0];
        const levels = params.levels || [1, 2, 4];
        const total = levels.reduce((acc, l) => acc + l * l, 0);
        return [c * total];
      }
      return inputShape;
    }
  },
  dilatedConv2d: {
    type: "dilatedConv2d",
    name: "DilatedConv2D",
    icon: "\u{1F537}",
    category: "cv",
    defaultParams: { outChannels: 32, kernelSize: 3, stride: 1, padding: 2, dilation: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [, h, w] = inputShape;
        const { outChannels } = params;
        return [outChannels, convOutLen(h, params, 0), convOutLen(w, params, 1)];
      }
      return inputShape;
    }
  },
  globalMaxPool2d: {
    type: "globalMaxPool2d",
    name: "GlobalMaxPool2D",
    icon: "\u{1F310}",
    category: "cv",
    defaultParams: {},
    computeOutputShape: (inputShape) => {
      if (inputShape.length >= 3) {
        return [inputShape[0]];
      }
      return inputShape;
    }
  },
  roiAlign: {
    type: "roiAlign",
    name: "RoIAlign",
    icon: "\u{1F3AF}",
    category: "cv",
    defaultParams: { outputSize: [7, 7], spatialScale: 0.25 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c] = inputShape;
        const size = Array.isArray(params.outputSize) ? params.outputSize : [params.outputSize, params.outputSize];
        return [c, size[0] ?? 7, size[1] ?? 7];
      }
      return inputShape;
    }
  },
  maxpool3d: {
    type: "maxpool3d",
    name: "MaxPool3D",
    icon: "\u2B07\uFE0F",
    category: "cv",
    defaultParams: { kernelSize: 2, stride: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 4) {
        const [c, d, h, w] = inputShape;
        const { kernelSize, stride = kernelSize } = params;
        return [c, Math.floor((d - kernelSize) / stride + 1), Math.floor((h - kernelSize) / stride + 1), Math.floor((w - kernelSize) / stride + 1)];
      }
      return inputShape;
    }
  },
  avgpool3d: {
    type: "avgpool3d",
    name: "AvgPool3D",
    icon: "\u{1F4CA}",
    category: "cv",
    defaultParams: { kernelSize: 2, stride: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 4) {
        const [c, d, h, w] = inputShape;
        const { kernelSize, stride = kernelSize } = params;
        return [c, Math.floor((d - kernelSize) / stride + 1), Math.floor((h - kernelSize) / stride + 1), Math.floor((w - kernelSize) / stride + 1)];
      }
      return inputShape;
    }
  },
  windowAttention: {
    type: "windowAttention",
    name: "Window Attention",
    icon: "\u{1FA9F}",
    category: "cv",
    defaultParams: { embedDim: 96, numHeads: 3, windowSize: 7 },
    computeOutputShape: (inputShape) => inputShape
  },
  seBlock: {
    type: "seBlock",
    name: "SE Block",
    icon: "\u{1F3AF}",
    category: "cv",
    defaultParams: { channels: 64, reductionRatio: 16 },
    computeOutputShape: (inputShape) => inputShape
  },
  patchEmbed: {
    type: "patchEmbed",
    name: "Patch Embed",
    icon: "\u{1F532}",
    category: "cv",
    defaultParams: { imgSize: 224, patchSize: 16, embedDim: 768 },
    // Patch count comes from the ACTUAL upstream feature map when there is one,
    // and only falls back to the declared imgSize when this is the first layer
    // (rank < 3 input, or no parent). Reading imgSize unconditionally produced
    // NaN for every patchEmbed that omits it (the seeded HYB family and the
    // patch-tst template both do), and NaN silently poisoned every downstream
    // shape. Non-square inputs are handled per-axis rather than squared.
    // `clsToken: true` prepends ViT's learned class token, one more sequence
    // position (197 for ViT-B/16 rather than 196). `registerTokens: R` adds R
    // learned register tokens after it (DINOv2 with registers).
    computeOutputShape: (inputShape, params) => {
      const patch = params.patchSize || 16;
      const embedDim = params.embedDim ?? 768;
      const cls = (params.clsToken ? 1 : 0) + Math.max(0, Math.floor(Number(params.registerTokens) || 0));
      if (inputShape.length >= 3) {
        const [, h, w] = inputShape;
        return [Math.floor(h / patch) * Math.floor(w / patch) + cls, embedDim];
      }
      const side = Math.floor((params.imgSize ?? 224) / patch);
      return [side * side + cls, embedDim];
    }
  },
  // ========== Utility ==========
  reshape: {
    type: "reshape",
    name: "Reshape",
    icon: "\u{1F504}",
    category: "utility",
    defaultParams: { shape: [512] },
    computeOutputShape: (_inputShape, params) => {
      return Array.isArray(params.shape) ? params.shape : [params.shape || 512];
    }
  },
  dropout: {
    type: "dropout",
    name: "Dropout",
    icon: "\u{1F3B2}",
    category: "utility",
    defaultParams: { p: 0.5 },
    computeOutputShape: (inputShape) => inputShape
  },
  residual: {
    type: "residual",
    name: "Residual",
    icon: "\u2795",
    category: "utility",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  skipConnection: {
    type: "skipConnection",
    name: "Skip Connection",
    icon: "\u23ED\uFE0F",
    category: "utility",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  concatenate: {
    type: "concatenate",
    name: "Concatenate",
    icon: "\u{1F517}",
    category: "utility",
    defaultParams: { dim: -1 },
    computeOutputShape: (inputShape, params, allInputShapes) => {
      const shapes = allInputShapes && allInputShapes.length > 1 ? allInputShapes : [inputShape, inputShape];
      const rank = shapes[0].length;
      const dim = params.dim === -1 ? rank - 1 : params.dim ?? rank - 1;
      const concatDimTotal = shapes.reduce((sum, s) => sum + (s[dim] ?? 0), 0);
      const out = [...shapes[0]];
      out[dim] = concatDimTotal;
      return out;
    }
  },
  add: {
    type: "add",
    name: "Add",
    icon: "\u2795",
    category: "utility",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  multiply: {
    type: "multiply",
    name: "Multiply",
    icon: "\u2716\uFE0F",
    category: "utility",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  dropPath: {
    type: "dropPath",
    name: "DropPath",
    icon: "\u{1FA82}",
    category: "utility",
    defaultParams: { dropRate: 0.1 },
    computeOutputShape: (inputShape) => inputShape
  },
  layerScale: {
    type: "layerScale",
    name: "LayerScale",
    icon: "\u2696\uFE0F",
    category: "utility",
    defaultParams: { dim: 512, initValues: 1e-5 },
    computeOutputShape: (inputShape) => inputShape
  },
  split: {
    type: "split",
    name: "Split",
    icon: "\u2702\uFE0F",
    category: "utility",
    defaultParams: { sections: 2, dim: -1 },
    computeOutputShape: (inputShape, params) => {
      const dim = params.dim ?? -1;
      const sections = params.sections ?? 2;
      const idx = dim < 0 ? inputShape.length + dim : dim;
      if (idx >= 0 && idx < inputShape.length) {
        const out = [...inputShape];
        out[idx] = Math.floor(inputShape[idx] / sections);
        return out;
      }
      return inputShape;
    }
  },
  permute: {
    type: "permute",
    name: "Permute",
    icon: "\u2194\uFE0F",
    category: "utility",
    defaultParams: { dims: [0, 2, 1] },
    computeOutputShape: (inputShape, params) => {
      const dims = params.dims ?? [0, 2, 1];
      if (dims.length === inputShape.length) return dims.map((d) => inputShape[d]);
      if (dims.length === inputShape.length + 1 && dims[0] === 0) {
        return dims.slice(1).map((d) => inputShape[d - 1]);
      }
      return inputShape;
    }
  },
  customModule: {
    type: "customModule",
    name: "Custom Module",
    icon: "\u{1F9E9}",
    category: "utility",
    defaultParams: { _customLayerId: "" },
    computeOutputShape: (inputShape) => inputShape
  },
  stickyNote: {
    type: "stickyNote",
    name: "Sticky Note",
    icon: "\u{1F4DD}",
    category: "utility",
    defaultParams: { _noteText: "", _noteColor: "#fef08a" },
    computeOutputShape: (inputShape) => inputShape
  },
  squeeze: {
    type: "squeeze",
    name: "Squeeze",
    icon: "\u2195\uFE0F",
    category: "utility",
    defaultParams: { dim: null },
    computeOutputShape: (inputShape, params) => {
      if (params.dim !== null && params.dim !== void 0) {
        const idx = params.dim < 0 ? inputShape.length + params.dim : params.dim;
        if (inputShape[idx] === 1) {
          return [...inputShape.slice(0, idx), ...inputShape.slice(idx + 1)];
        }
        return inputShape;
      }
      return inputShape.filter((d) => d !== 1);
    }
  },
  unsqueeze: {
    type: "unsqueeze",
    name: "Unsqueeze",
    icon: "\u2194\uFE0F",
    category: "utility",
    defaultParams: { dim: 0 },
    computeOutputShape: (inputShape, params) => {
      const idx = (params.dim ?? 0) < 0 ? inputShape.length + 1 + params.dim : params.dim ?? 0;
      const out = [...inputShape];
      out.splice(idx, 0, 1);
      return out;
    }
  },
  pad: {
    type: "pad",
    name: "Pad",
    icon: "\u{1F532}",
    category: "utility",
    defaultParams: { padding: [0, 0, 0, 0] },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c, h, w] = inputShape;
        const p = params.padding || [0, 0, 0, 0];
        return [c, h + (p[2] ?? 0) + (p[3] ?? 0), w + (p[0] ?? 0) + (p[1] ?? 0)];
      }
      return inputShape;
    }
  },
  mean: {
    type: "mean",
    name: "Mean",
    icon: "\u2797",
    category: "utility",
    defaultParams: { dim: -1, keepdim: false },
    computeOutputShape: (inputShape, params) => {
      const dim = params.dim ?? -1;
      const idx = dim < 0 ? inputShape.length + dim : dim;
      if (params.keepdim) {
        const out = [...inputShape];
        out[idx] = 1;
        return out;
      }
      return [...inputShape.slice(0, idx), ...inputShape.slice(idx + 1)];
    }
  },
  matmul: {
    type: "matmul",
    name: "MatMul",
    icon: "\u2716\uFE0F",
    category: "utility",
    defaultParams: {},
    computeOutputShape: (inputShape, _params, allInputShapes) => {
      const shapes = allInputShapes && allInputShapes.length >= 2 ? allInputShapes : [inputShape, inputShape];
      const a = shapes[0];
      const b = shapes[1];
      if (a.length >= 2 && b.length >= 2) {
        return [...a.slice(0, -1), b[b.length - 1]];
      }
      return inputShape;
    }
  },
  clamp: {
    type: "clamp",
    name: "Clamp",
    icon: "\u{1F4CC}",
    category: "utility",
    defaultParams: { min: 0, max: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  norm: {
    type: "norm",
    name: "L2 Norm",
    icon: "\u{1F4CF}",
    category: "utility",
    defaultParams: { dim: -1, p: 2 },
    computeOutputShape: (inputShape) => inputShape
  },
  vaeBottleneck: {
    type: "vaeBottleneck",
    name: "VAE Bottleneck",
    icon: "\u{1F9EC}",
    category: "utility",
    defaultParams: { latentDim: 128 },
    computeOutputShape: (_inputShape, params) => [params.latentDim || 128]
  },
  miniBatchStdDev: {
    type: "miniBatchStdDev",
    name: "MiniBatch StdDev",
    icon: "\u{1F4CA}",
    category: "utility",
    defaultParams: { groupSize: 4 },
    computeOutputShape: (inputShape) => {
      if (inputShape.length >= 3) {
        return [inputShape[0] + 1, inputShape[1], inputShape[2]];
      }
      return inputShape;
    }
  },
  topK: {
    type: "topK",
    name: "TopK",
    icon: "\u{1F51D}",
    category: "utility",
    defaultParams: { k: 2, dim: -1, sorted: true },
    computeOutputShape: (inputShape, params) => {
      const dim = params.dim ?? -1;
      const idx = dim < 0 ? inputShape.length + dim : dim;
      const out = [...inputShape];
      out[idx] = params.k ?? 2;
      return out;
    }
  },
  gather: {
    type: "gather",
    name: "Gather",
    icon: "\u{1FA9D}",
    category: "utility",
    // indexSize 0 = "same as input" (index tensor length unknown at design
    // time); a positive value resizes the gathered dim to the index length.
    defaultParams: { dim: 0, indexSize: 0 },
    computeOutputShape: (inputShape, params) => {
      const dim = Number(params.dim ?? 0);
      const idx = dim < 0 ? inputShape.length + dim : dim;
      if (idx < 0 || idx >= inputShape.length) {
        throw new Error(`gather: dim ${params.dim} out of range for rank ${inputShape.length}`);
      }
      const out = [...inputShape];
      const n2 = Number(params.indexSize);
      if (Number.isFinite(n2) && n2 > 0) out[idx] = n2;
      return out;
    }
  },
  scatter: {
    type: "scatter",
    name: "Scatter",
    icon: "\u{1F4A7}",
    category: "utility",
    defaultParams: { dim: 0 },
    // Output shape IS the self tensor's shape (passthrough is correct); the
    // only design-time checkable bug is an out-of-range dim.
    computeOutputShape: (inputShape, params) => {
      const dim = Number(params.dim ?? 0);
      const idx = dim < 0 ? inputShape.length + dim : dim;
      if (idx < 0 || idx >= inputShape.length) {
        throw new Error(`scatter: dim ${params.dim} out of range for rank ${inputShape.length}`);
      }
      return inputShape;
    }
  },
  stack: {
    type: "stack",
    name: "Stack",
    icon: "\u{1F4DA}",
    category: "utility",
    defaultParams: { dim: 0 },
    computeOutputShape: (inputShape, params) => {
      const dim = params.dim ?? 0;
      const idx = dim < 0 ? inputShape.length + dim + 1 : dim;
      const out = [...inputShape];
      out.splice(idx, 0, 1);
      return out;
    }
  },
  einsum: {
    type: "einsum",
    name: "Einsum",
    icon: "\u2211",
    category: "utility",
    defaultParams: { equation: "ij,jk->ik" },
    // Real equation-driven shape: the first operand's subscripts bind to the
    // incoming shape (rank must agree — that mismatch is THE einsum bug worth
    // catching before runtime). Output dims resolve from those bindings;
    // letters bound only by other operands are unknowable from one input, so
    // the shape stays passthrough in that case rather than guessing.
    computeOutputShape: (inputShape, params) => {
      const eq = String(params.equation ?? "").replace(/\s+/g, "");
      const m = eq.match(/^([a-zA-Z]+(?:,[a-zA-Z]+)*)->([a-zA-Z]*)$/);
      if (!m) return inputShape;
      const first = m[1].split(",")[0];
      const rhs = m[2];
      if (first.length !== inputShape.length) {
        throw new Error(`einsum: first operand '${first}' expects rank ${first.length}, input has rank ${inputShape.length} (equation "${eq}")`);
      }
      const bound = /* @__PURE__ */ new Map();
      [...first].forEach((ch, i) => bound.set(ch, inputShape[i]));
      if (![...rhs].every((ch) => bound.has(ch))) return inputShape;
      return [...rhs].map((ch) => bound.get(ch));
    }
  },
  // ========== Frontier architectures (2024-2025) ==========
  mla: {
    type: "mla",
    name: "Multi-Head Latent Attention",
    icon: "\u{1F9EC}",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, kvLatentDim: 128, ropeHeadDim: 64 },
    computeOutputShape: (inputShape) => inputShape
  },
  mamba2: {
    type: "mamba2",
    name: "Mamba-2 (SSD)",
    icon: "\u{1F40D}",
    category: "llm",
    defaultParams: { dModel: 512, dState: 128, expand: 2, headDim: 64, chunkSize: 256 },
    computeOutputShape: (inputShape) => inputShape
  },
  qkNorm: {
    type: "qkNorm",
    name: "QK-Norm",
    icon: "\u{1F4D0}",
    category: "normalization",
    defaultParams: { dim: 64, eps: 1e-6 },
    computeOutputShape: (inputShape) => inputShape
  },
  multiTokenPrediction: {
    type: "multiTokenPrediction",
    name: "Multi-Token Prediction Head",
    icon: "\u{1F3B0}",
    category: "llm",
    defaultParams: { vocabSize: 32e3, numFutureTokens: 2, dModel: 512 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 1) {
        return [...inputShape.slice(0, -1), params.vocabSize || 32e3];
      }
      return [params.vocabSize || 32e3];
    }
  },
  xlstm: {
    type: "xlstm",
    name: "xLSTM",
    icon: "\u{1F501}",
    category: "llm",
    defaultParams: { dModel: 512, numHeads: 4, variant: "mLSTM" },
    computeOutputShape: (inputShape) => inputShape
  },
  differentialAttention: {
    type: "differentialAttention",
    name: "Differential Attention",
    icon: "\u2796",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, lambdaInit: 0.8 },
    computeOutputShape: (inputShape) => inputShape
  },
  rgLru: {
    type: "rgLru",
    name: "RG-LRU (Griffin)",
    icon: "\u{1F985}",
    category: "llm",
    defaultParams: { dModel: 512, expand: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  retention: {
    type: "retention",
    name: "Retention (RetNet)",
    icon: "\u{1F9F2}",
    category: "llm",
    defaultParams: { dModel: 512, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  hyena: {
    type: "hyena",
    name: "Hyena",
    icon: "\u{1F300}",
    category: "llm",
    defaultParams: { dModel: 512, order: 2, filterOrder: 64 },
    computeOutputShape: (inputShape) => inputShape
  },
  rwkv: {
    type: "rwkv",
    name: "RWKV (Time-Mix)",
    icon: "\u23F3",
    category: "llm",
    defaultParams: { dModel: 512 },
    computeOutputShape: (inputShape) => inputShape
  },
  kan: {
    type: "kan",
    name: "KAN Layer",
    icon: "\u{1FAA2}",
    category: "basic",
    defaultParams: { inFeatures: 128, outFeatures: 128, gridSize: 5, splineOrder: 3 },
    computeOutputShape: (inputShape, params) => {
      return [...inputShape.slice(0, -1), params.outFeatures || 128];
    }
  },
  mixtureOfDepths: {
    type: "mixtureOfDepths",
    name: "Mixture-of-Depths",
    icon: "\u{1FA9C}",
    category: "llm",
    defaultParams: { dModel: 512, capacityFactor: 0.5 },
    computeOutputShape: (inputShape) => inputShape
  },
  tttLayer: {
    type: "tttLayer",
    name: "Test-Time Training Layer",
    icon: "\u{1F9EA}",
    category: "llm",
    defaultParams: { dModel: 512, innerSteps: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  geglu: {
    type: "geglu",
    name: "GeGLU",
    icon: "\u26A1",
    category: "activation",
    defaultParams: { dim: 512, hiddenDim: 2048 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 1) {
        return [...inputShape.slice(0, -1), params.dim ?? inputShape[inputShape.length - 1]];
      }
      return inputShape;
    }
  },
  grn: {
    type: "grn",
    name: "Global Response Norm",
    icon: "\u{1F310}",
    category: "normalization",
    defaultParams: { channels: 256, eps: 1e-6 },
    computeOutputShape: (inputShape) => inputShape
  },
  titansMemory: {
    type: "titansMemory",
    name: "Titans Neural Memory",
    icon: "\u{1F5FF}",
    category: "llm",
    defaultParams: { dModel: 512, memoryDepth: 2 },
    computeOutputShape: (inputShape) => inputShape
  },
  deltaNet: {
    type: "deltaNet",
    name: "DeltaNet",
    icon: "\u{1F53A}",
    category: "llm",
    defaultParams: { dModel: 512, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  gatedDeltaNet: {
    type: "gatedDeltaNet",
    name: "Gated DeltaNet",
    icon: "\u{1F53B}",
    category: "llm",
    defaultParams: { dModel: 512, numHeads: 8, headDim: 64 },
    computeOutputShape: (inputShape) => inputShape
  },
  sharedExpertMoE: {
    type: "sharedExpertMoE",
    name: "Shared-Expert MoE",
    icon: "\u{1F9E9}",
    category: "llm",
    defaultParams: { embedDim: 4096, numExperts: 64, numSharedExperts: 2, topK: 6, expertDim: 1408 },
    computeOutputShape: (inputShape) => inputShape
  },
  ditBlock: {
    type: "ditBlock",
    name: "DiT Block (AdaLN-Zero)",
    icon: "\u{1F3A8}",
    category: "cv",
    defaultParams: { hiddenDim: 1152, numHeads: 16, condDim: 1152 },
    computeOutputShape: (inputShape) => inputShape
  },
  vectorQuantizer: {
    type: "vectorQuantizer",
    name: "Vector Quantizer (VQ)",
    icon: "\u{1F48E}",
    category: "utility",
    defaultParams: { codebookSize: 8192, embedDim: 256 },
    computeOutputShape: (inputShape) => inputShape
  },
  perceiverLatent: {
    type: "perceiverLatent",
    name: "Perceiver Latent / Q-Former",
    icon: "\u{1F52E}",
    category: "multimodal",
    defaultParams: { numLatents: 64, latentDim: 768, numHeads: 8 },
    computeOutputShape: (inputShape, params) => {
      const numLatents = params.numLatents ?? 64;
      const latentDim = params.latentDim ?? 768;
      if (inputShape.length >= 2) return [...inputShape.slice(0, -2), numLatents, latentDim];
      return [numLatents, latentDim];
    }
  },
  convNeXtBlock: {
    type: "convNeXtBlock",
    name: "ConvNeXt Block",
    icon: "\u{1F9F1}",
    category: "cv",
    defaultParams: { dim: 96, kernelSize: 7, expandRatio: 4 },
    computeOutputShape: (inputShape) => inputShape
  },
  gatedLinearAttention: {
    type: "gatedLinearAttention",
    name: "Gated Linear Attention (GLA)",
    icon: "\u{1F32C}\uFE0F",
    category: "llm",
    defaultParams: { dModel: 512, numHeads: 8, expandK: 0.5, expandV: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  s4Layer: {
    type: "s4Layer",
    name: "S4 / S5 (Structured SSM)",
    icon: "\u3030\uFE0F",
    category: "llm",
    defaultParams: { dModel: 512, dState: 64 },
    computeOutputShape: (inputShape) => inputShape
  },
  dyt: {
    type: "dyt",
    name: "Dynamic Tanh (DyT)",
    icon: "\u{1F4C9}",
    category: "normalization",
    defaultParams: { dim: 512, alphaInit: 0.5 },
    computeOutputShape: (inputShape) => inputShape
  },
  nativeSparseAttention: {
    type: "nativeSparseAttention",
    name: "Native Sparse Attention",
    icon: "\u{1F578}\uFE0F",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, blockSize: 64, topBlocks: 16 },
    computeOutputShape: (inputShape) => inputShape
  },
  film: {
    type: "film",
    name: "FiLM",
    icon: "\u{1F39A}\uFE0F",
    category: "utility",
    defaultParams: { numFeatures: 256 },
    computeOutputShape: (inputShape) => inputShape
  },
  residualVQ: {
    type: "residualVQ",
    name: "Residual VQ (RVQ)",
    icon: "\u{1F4A0}",
    category: "audio",
    defaultParams: { numQuantizers: 8, codebookSize: 1024, embedDim: 256 },
    computeOutputShape: (inputShape) => inputShape
  },
  crossNetworkDCN: {
    type: "crossNetworkDCN",
    name: "DCN Cross Network",
    icon: "\u{1F517}",
    category: "tabular",
    defaultParams: { numLayers: 3, inputDim: 256 },
    computeOutputShape: (inputShape) => inputShape
  },
  ftTransformerBlock: {
    type: "ftTransformerBlock",
    name: "FT-Transformer",
    icon: "\u{1F4CB}",
    category: "tabular",
    defaultParams: { dModel: 192, numHeads: 8, ffMult: 4 },
    computeOutputShape: (inputShape) => inputShape
  },
  deformableAttention: {
    type: "deformableAttention",
    name: "Deformable Attention",
    icon: "\u{1F9ED}",
    category: "cv",
    defaultParams: { embedDim: 256, numHeads: 8, numPoints: 4, numLevels: 4 },
    computeOutputShape: (inputShape) => inputShape
  },
  attentionPool: {
    type: "attentionPool",
    name: "Attention Pooling (PMA)",
    icon: "\u{1F3AF}",
    category: "basic",
    defaultParams: { dim: 512, numHeads: 8, numSeeds: 1 },
    computeOutputShape: (inputShape, params) => {
      const numSeeds = params.numSeeds ?? 1;
      if (inputShape.length >= 2) {
        const lastDim = inputShape[inputShape.length - 1];
        if (numSeeds === 1) return [...inputShape.slice(0, -2), lastDim];
        return [...inputShape.slice(0, -2), numSeeds, lastDim];
      }
      return inputShape;
    }
  },
  // ========== Time-series / 3D / Video (2024-2025) ==========
  revIN: {
    type: "revIN",
    name: "Reversible Instance Norm",
    icon: "\u{1F501}",
    category: "normalization",
    defaultParams: { numFeatures: 7, eps: 1e-5, affine: true },
    computeOutputShape: (inputShape) => inputShape
  },
  seriesDecomp: {
    type: "seriesDecomp",
    name: "Series Decomposition",
    icon: "\u303D\uFE0F",
    category: "nlp",
    defaultParams: { kernelSize: 25 },
    // Autoformer decomposition emits seasonal + trend, BOTH input-shaped, so
    // passthrough is the true per-branch shape. The moving average needs an
    // odd kernel for symmetric padding to preserve length.
    computeOutputShape: (inputShape, params) => {
      const k = Number(params.kernelSize ?? 25);
      if (!Number.isFinite(k) || k < 1) {
        throw new Error(`seriesDecomp: kernelSize must be a positive integer, got ${params.kernelSize}`);
      }
      if (k % 2 === 0) {
        throw new Error(`seriesDecomp: kernelSize ${k} is even; the moving-average padding needs an odd kernel to keep sequence length`);
      }
      return inputShape;
    }
  },
  setAbstraction: {
    type: "setAbstraction",
    name: "PointNet++ Set Abstraction",
    icon: "\u{1F7E2}",
    category: "cv",
    defaultParams: { numPoints: 512, radius: 0.2, numSamples: 32, mlp: [64, 64, 128] },
    computeOutputShape: (inputShape, params) => {
      const numPoints = params.numPoints ?? 512;
      const mlp = Array.isArray(params.mlp) ? params.mlp : [64, 64, 128];
      const lastDim = mlp.length > 0 ? mlp[mlp.length - 1] : 128;
      if (inputShape.length < 2) return [numPoints, 128];
      return [numPoints, lastDim];
    }
  },
  sparseConv3d: {
    type: "sparseConv3d",
    name: "Submanifold Sparse Conv3D",
    icon: "\u{1F537}",
    category: "cv",
    defaultParams: { inChannels: 32, outChannels: 64, kernelSize: 3 },
    computeOutputShape: (inputShape, params) => {
      const outChannels = params.outChannels ?? 64;
      if (inputShape.length >= 4) {
        const [, d, h, w] = inputShape;
        return [outChannels, d, h, w];
      }
      return inputShape;
    }
  },
  nerfPositionalEncoding: {
    type: "nerfPositionalEncoding",
    name: "Fourier Feature Encoding",
    icon: "\u{1F308}",
    category: "utility",
    defaultParams: { numFrequencies: 10, includeInput: true },
    computeOutputShape: (inputShape, params) => {
      const numFrequencies = params.numFrequencies ?? 10;
      const includeInput = params.includeInput ?? true;
      if (inputShape.length === 0) return inputShape;
      const out = [...inputShape];
      const last = out[out.length - 1];
      out[out.length - 1] = last * (2 * numFrequencies + (includeInput ? 1 : 0));
      return out;
    }
  },
  dividedSpaceTimeAttention: {
    type: "dividedSpaceTimeAttention",
    name: "Divided Space-Time Attention",
    icon: "\u{1F39E}\uFE0F",
    category: "cv",
    defaultParams: { embedDim: 768, numHeads: 12 },
    computeOutputShape: (inputShape) => inputShape
  },
  tubeletEmbed: {
    type: "tubeletEmbed",
    name: "Tubelet Embedding (3D Patch)",
    icon: "\u{1F4F9}",
    category: "cv",
    defaultParams: { tubeletSize: [2, 16, 16], embedDim: 768 },
    computeOutputShape: (inputShape, params) => {
      const embedDim = params.embedDim ?? 768;
      const tubelet = Array.isArray(params.tubeletSize) ? params.tubeletSize : [2, 16, 16];
      if (inputShape.length >= 4) {
        const [, t, h, w] = inputShape;
        const [tt, th, tw] = tubelet;
        const numTubelets = Math.max(1, Math.floor(t / (tt || 1))) * Math.max(1, Math.floor(h / (th || 1))) * Math.max(1, Math.floor(w / (tw || 1)));
        return [numTubelets, embedDim];
      }
      return [196, embedDim];
    }
  },
  // ========== Video latents / world models (2023-2025) ==========
  causalConv3d: {
    type: "causalConv3d",
    name: "Causal Conv3D",
    icon: "\u{1F39E}\uFE0F",
    category: "cv",
    // The building block of every modern video VAE: padding is applied to the
    // LEFT of the time axis only, so frame t never sees frame t+1 and the
    // encoder can be run on a growing clip. With temporalStride 1 the frame
    // count is preserved, which is the property the rest of the pipeline
    // depends on and which an ordinary conv3d silently breaks.
    defaultParams: { outChannels: 128, kernelSize: 3, stride: 1, temporalStride: 1, padding: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length < 4) return inputShape;
      const [, t, h, w] = inputShape;
      const outC = Number(params.outChannels) || 128;
      const k = Number(params.kernelSize) || 3;
      const sT = Math.max(1, Number(params.temporalStride) || 1);
      const s = Math.max(1, Number(params.stride) || 1);
      const p = Number(params.padding ?? Math.floor(k / 2));
      const outT = Math.max(1, Math.floor((t - 1) / sT) + 1);
      const spat = (len) => Math.max(1, Math.floor((len + 2 * p - k) / s + 1));
      return [outC, outT, spat(h), spat(w)];
    }
  },
  rssm: {
    type: "rssm",
    name: "RSSM (Dreamer)",
    icon: "\u{1F30D}",
    category: "rl",
    // The recurrent state-space model: a DETERMINISTIC recurrent path and a
    // STOCHASTIC categorical latent, concatenated. Everything downstream (the
    // reward head, the decoder, the actor) reads that concatenation, so the
    // width is deterDim + stochDim * stochClasses and not deterDim, which is
    // the mistake a plain GRU node would encode.
    defaultParams: { deterDim: 512, stochDim: 32, stochClasses: 32, hiddenDim: 512 },
    computeOutputShape: (_inputShape, params) => {
      const det = Number(params.deterDim) || 512;
      const st = Number(params.stochDim) || 32;
      const cls = Math.max(1, Number(params.stochClasses) || 1);
      return [det + st * cls];
    }
  },
  latentActionModel: {
    type: "latentActionModel",
    name: "Latent Action Model (Genie)",
    icon: "\u{1F579}\uFE0F",
    category: "rl",
    // Genie's LAM infers a discrete action from a PAIR of consecutive frames,
    // so a T-frame clip yields T-1 actions. Getting that off-by-one wrong is
    // the single most common way this component is mis-wired.
    defaultParams: { latentDim: 32, codebookSize: 8, hiddenDim: 512 },
    computeOutputShape: (inputShape, params) => {
      const dim = Number(params.latentDim) || 32;
      if (inputShape.length >= 2) {
        const t = inputShape[inputShape.length - 2];
        return [...inputShape.slice(0, -2), Math.max(1, t - 1), dim];
      }
      return [dim];
    }
  },
  jepaPredictor: {
    type: "jepaPredictor",
    name: "JEPA Predictor",
    icon: "\u{1F9E9}",
    category: "rl",
    // Predicts TARGET representations from CONTEXT representations, in
    // representation space rather than pixel space. Shape preserving in the
    // embedding width; the number of tokens is set by the mask, not by this
    // layer, so the token axis is passed through unchanged.
    defaultParams: { embedDim: 768, predictorDim: 384, depth: 6, numHeads: 12 },
    computeOutputShape: (inputShape) => inputShape
  },
  emaTarget: {
    type: "emaTarget",
    name: "EMA Target / Stop-Gradient",
    icon: "\u{1F9CA}",
    category: "utility",
    // Not a computation: a declaration that this branch is an
    // exponential-moving-average copy of its upstream and carries no gradient.
    // It exists so the graph can SAY the thing that keeps BYOL / SimSiam /
    // I-JEPA from collapsing, which makes the absence of it checkable.
    defaultParams: { momentum: 0.996, stopGradient: true },
    computeOutputShape: (inputShape) => inputShape
  }
};

// src/utils/lintThresholds.ts
var HIGH_DROPOUT_P = 0.65;
var DEEP_NO_RESIDUAL_MIN_LAYERS = 8;
var DEEP_NO_NORM_MIN_LAYERS = 7;
var VANISHING_MIN_LAYERS = 5;
var LARGE_ACTIVATION_MAX_ELEMENTS = 5e7;
var SWIGLU_RATIO_MIN = 2;
var SWIGLU_RATIO_MAX = 6;
var LARGE_LINEAR_MAX_PARAMS = 1e9;
var KV_BUDGET_CONTEXT_TOKENS = 8192;
var KV_BUDGET_MAX_GB = 4;
var SCALED_INIT_MIN_ATTENTION_LAYERS = 8;

// src/utils/weightInitAdvisor.ts
var ADVICE = {
  // ── Linear / Dense ────────────────────────────────────────────────────────
  linear: {
    method: "Kaiming Uniform",
    formula: "U(\u2212\u221A(6/fan_in), \u221A(6/fan_in))",
    reason: "Default for layers followed by ReLU; preserves activation variance.",
    pyTorch: 'nn.init.kaiming_uniform_(w, mode="fan_in", nonlinearity="relu")'
  },
  // ── Convolutions ─────────────────────────────────────────────────────────
  conv2d: {
    method: "Kaiming Normal",
    formula: "N(0, \u221A(2 / fan_in))",
    reason: "He init keeps variance stable through ReLU activations.",
    pyTorch: 'nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")'
  },
  conv1d: {
    method: "Kaiming Normal",
    formula: "N(0, \u221A(2 / fan_in))",
    reason: "Same as Conv2d: He init for ReLU-based feature extraction.",
    pyTorch: 'nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")'
  },
  depthwiseConv2d: {
    method: "Kaiming Normal",
    formula: "N(0, \u221A(2 / fan_in))",
    reason: "Depthwise conv has fan_in = kernel_h \xD7 kernel_w; apply He init.",
    pyTorch: 'nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")'
  },
  // ── Attention / Transformer ───────────────────────────────────────────────
  multiHeadAttention: {
    method: "Xavier Normal",
    formula: "N(0, \u221A(2 / (fan_in + fan_out)))",
    reason: "Balanced input/output variance; standard for attention projections.",
    pyTorch: "nn.init.xavier_normal_(w)"
  },
  causalAttention: {
    method: "GPT-2 style (scaled Normal)",
    formula: "N(0, 0.02 / \u221A(2L))",
    reason: "Scales down residual projections by depth to prevent explosion.",
    pyTorch: "nn.init.normal_(w, std=0.02 / math.sqrt(2 * n_layers))"
  },
  groupedQueryAttention: {
    method: "GPT-2 style (scaled Normal)",
    formula: "N(0, 0.02 / \u221A(2L))",
    reason: "Matches LLaMA/Mistral init: residual outputs scaled by depth.",
    pyTorch: "nn.init.normal_(w, std=0.02 / math.sqrt(2 * n_layers))"
  },
  transformerBlock: {
    method: "Scaled Normal (residual)",
    formula: "N(0, 0.02 / \u221A(2L))",
    reason: "Residual path init must scale with depth to keep output stable.",
    pyTorch: "nn.init.normal_(w, std=0.02 / math.sqrt(2 * n_layers))"
  },
  feedForward: {
    method: "Kaiming Normal + small output",
    formula: "N(0, 0.02 / \u221A(2L)) for out proj",
    reason: "Input proj: He init; output proj scaled down as residual stream.",
    pyTorch: "nn.init.kaiming_normal_(fc1.weight); nn.init.normal_(fc2.weight, std=0.02/\u221A(2L))"
  },
  swiglu: {
    method: "Scaled Normal",
    formula: "N(0, 0.02 / \u221A(2L))",
    reason: "LLaMA-style: down_proj uses scaled init; gate/up use standard.",
    pyTorch: "nn.init.normal_(down_proj.weight, std=0.02/math.sqrt(2*n_layers))"
  },
  // ── Embeddings ───────────────────────────────────────────────────────────
  embedding: {
    method: "Normal (small)",
    formula: "N(0, 1/\u221AD)",
    reason: "Small init prevents embedding norms from dominating early training.",
    pyTorch: "nn.init.normal_(embed.weight, std=1/math.sqrt(d_model))"
  },
  // ── Normalization (biases, gains) ────────────────────────────────────────
  layerNorm: {
    method: "Scale=1, Bias=0",
    formula: "\u03B3=1, \u03B2=0",
    reason: "Identity init so early gradients flow unimpeded.",
    pyTorch: "nn.init.ones_(ln.weight); nn.init.zeros_(ln.bias)"
  },
  rmsNorm: {
    method: "Scale=1",
    formula: "\u03B3=1",
    reason: "RMSNorm has no bias; init scale to 1 for identity pass-through.",
    pyTorch: "nn.init.ones_(rms.weight)"
  },
  batchNorm: {
    method: "Scale=1, Bias=0",
    formula: "\u03B3=1, \u03B2=0",
    reason: "Same as LayerNorm: identity at init lets batch statistics stabilize.",
    pyTorch: "nn.init.ones_(bn.weight); nn.init.zeros_(bn.bias)"
  },
  groupNorm: {
    method: "Scale=1, Bias=0",
    formula: "\u03B3=1, \u03B2=0",
    reason: "Standard normalisation layer init.",
    pyTorch: "nn.init.ones_(gn.weight); nn.init.zeros_(gn.bias)"
  },
  // ── RNN ──────────────────────────────────────────────────────────────────
  lstm: {
    method: "Orthogonal (hidden) + Xavier (input)",
    formula: "W_h = orthonormal; W_x = Xavier",
    reason: "Orthogonal hidden weights preserve gradient norms over long sequences.",
    pyTorch: "nn.init.orthogonal_(lstm.weight_hh_l0); nn.init.xavier_uniform_(lstm.weight_ih_l0)"
  },
  gru: {
    method: "Orthogonal (hidden) + Xavier (input)",
    formula: "W_h = orthonormal; W_x = Xavier",
    reason: "Same as LSTM: orthogonal init prevents vanishing/exploding in GRU.",
    pyTorch: "nn.init.orthogonal_(gru.weight_hh_l0); nn.init.xavier_uniform_(gru.weight_ih_l0)"
  },
  // ── Output heads ─────────────────────────────────────────────────────────
  patchEmbed: {
    method: "Kaiming Normal",
    formula: "N(0, \u221A(2 / fan_in))",
    reason: "PatchEmbed is a Conv2d projection; He init standard.",
    pyTorch: "nn.init.kaiming_normal_(patch_embed.proj.weight)"
  },
  // ── Activations (no weights, but biases) ─────────────────────────────────
  prelu: {
    method: "PReLU slope init",
    formula: "a = 0.25",
    reason: "PyTorch default: close to leaky ReLU for stable start.",
    pyTorch: "nn.init.constant_(prelu.weight, 0.25)"
  }
};
function getInitAdvice(type) {
  return ADVICE[type] ?? null;
}

// src/utils/architectureAdvisor.ts
var largestDivisorAtMost = (n2, max) => {
  for (let d = Math.min(max, n2); d >= 1; d--) if (n2 % d === 0) return d;
  return 1;
};
var ACTIVATION_TYPES = /* @__PURE__ */ new Set([
  "relu",
  "gelu",
  "swish",
  "sigmoid",
  "tanh",
  "leakyRelu",
  "softmax",
  "silu"
]);
var NORM_TYPES = /* @__PURE__ */ new Set([
  "batchNorm",
  "layerNorm",
  "instanceNorm",
  "groupNorm",
  "rmsNorm"
]);
var DEEP_LAYER_TYPES = /* @__PURE__ */ new Set([
  "conv2d",
  "conv1d",
  "conv3d",
  "linear",
  "depthwiseConv2d",
  "separableConv2d",
  "transposeConv2d"
]);
var RESIDUAL_TYPES = /* @__PURE__ */ new Set([
  "residual",
  "skipConnection",
  "add"
]);
var ATTENTION_TYPES = /* @__PURE__ */ new Set([
  "attention",
  "selfAttention",
  "multiHeadAttention",
  "groupedQueryAttention",
  "causalAttention",
  "mla"
]);
var PE_TYPES = /* @__PURE__ */ new Set([
  // A learned relative bias is how T5 (and every model that copied it) tells
  // attention where tokens are. Leaving it out of this set made R14 warn that
  // "attention is permutation-invariant" on models whose whole positional
  // scheme was sitting on the canvas.
  "positionalEncoding",
  "learnedPositionalEmbedding",
  "rope",
  "alibi",
  "relativePositionBias"
]);
var VANISHING_ACTIVATIONS = /* @__PURE__ */ new Set(["sigmoid", "tanh"]);
var observedEdges = (model) => model.connections.filter((c) => !c.inferred);
var hasUnmodelledMerge = (model) => model.connections.some((c) => c.inferred === "unmodelled-merge");
var noInputNode = (model) => {
  if (model.components.length > 0 && !model.components.some((c) => c.type === "input")) {
    return [{
      id: "no-input-node",
      ruleId: "no-input-node",
      // Warning, not error: a missing Input is the normal state while building
      // (e.g. right after dropping a block) and is trivially fixed, so it should
      // not raise a red "error" on a fresh, in-progress graph.
      severity: "warning",
      category: "structure",
      title: "No Input node",
      message: "The model has no Input node. Without an Input layer, tensor shapes cannot be propagated and generated code will be incomplete.",
      affectedIds: [],
      suggestion: "Drag an Input layer from the I/O section of the component palette."
    }];
  }
  return [];
};
var noOutputNode = (model) => {
  if (model.components.length > 1 && !model.components.some((c) => c.type === "output")) {
    return [{
      id: "no-output-node",
      ruleId: "no-output-node",
      severity: "warning",
      category: "structure",
      title: "No Output node",
      message: "The model has no Output node. The code generator won't know where the forward pass terminates.",
      affectedIds: [],
      suggestion: "Connect the last layer to an Output node."
    }];
  }
  return [];
};
var isolatedComponents = (model) => {
  if (model.components.length < 2) return [];
  const connected = /* @__PURE__ */ new Set();
  model.connections.forEach((c) => {
    connected.add(c.from);
    connected.add(c.to);
  });
  const isolated = model.components.filter((c) => !connected.has(c.id));
  if (isolated.length === 0) return [];
  const names = isolated.map((c) => `"${c.name}"`).join(", ");
  return [{
    id: "isolated-components",
    ruleId: "isolated-components",
    severity: "warning",
    category: "structure",
    title: `${isolated.length} isolated layer${isolated.length !== 1 ? "s" : ""}`,
    message: `${names} ${isolated.length === 1 ? "has" : "have"} no connections and will be excluded from generated code.`,
    affectedIds: isolated.map((c) => c.id),
    suggestion: "Connect these layers to the graph or delete them."
  }];
};
var deadEnds = (model) => {
  const hasOutgoing = new Set(model.connections.map((c) => c.from));
  const issues = [];
  for (const comp of model.components) {
    if (comp.type === "output") continue;
    const hasIn = model.connections.some((c) => c.to === comp.id);
    if (hasIn && !hasOutgoing.has(comp.id)) {
      issues.push({
        id: `dead-end-${comp.id}`,
        ruleId: "dead-end",
        severity: "warning",
        category: "structure",
        title: `Dead-end: "${comp.name}"`,
        message: `"${comp.name}" receives input but its output is not connected. This layer will be unreachable in the forward pass.`,
        affectedIds: [comp.id],
        suggestion: "Connect the output forward, or add an Output node if this is the final layer."
      });
    }
  }
  return issues;
};
var bnAfterActivation = (model) => {
  const issues = [];
  for (const conn of observedEdges(model)) {
    const from = model.components.find((c) => c.id === conn.from);
    const to = model.components.find((c) => c.id === conn.to);
    if (!from || !to) continue;
    if (ACTIVATION_TYPES.has(from.type) && NORM_TYPES.has(to.type)) {
      issues.push({
        id: `bn-after-act-${conn.id}`,
        ruleId: "bn-after-activation",
        severity: "warning",
        category: "ordering",
        title: "Normalization after activation",
        message: `"${to.name}" (${to.type}) follows "${from.name}" (activation). The standard pre-activation order is Conv/Linear \u2192 Norm \u2192 Activation. Normalizing post-activation limits expressivity.`,
        affectedIds: [from.id, to.id],
        suggestion: "Move the normalization layer before the activation function."
      });
    }
  }
  return issues;
};
var dropoutBeforeBN = (model) => {
  const issues = [];
  for (const conn of observedEdges(model)) {
    const from = model.components.find((c) => c.id === conn.from);
    const to = model.components.find((c) => c.id === conn.to);
    if (!from || !to) continue;
    if (from.type === "dropout" && NORM_TYPES.has(to.type)) {
      issues.push({
        id: `dropout-bn-${conn.id}`,
        ruleId: "dropout-before-bn",
        severity: "info",
        category: "ordering",
        title: "Dropout before normalization",
        message: `"${from.name}" \u2192 "${to.name}": BatchNorm re-normalizes the random zeros introduced by Dropout, nullifying most of its regularization effect.`,
        affectedIds: [from.id, to.id],
        suggestion: "Reorder to Conv \u2192 BN \u2192 Activation \u2192 Dropout."
      });
    }
  }
  return issues;
};
var outputActivation = (model) => {
  const outputNode = model.components.find((c) => c.type === "output");
  if (!outputNode) return [];
  const issues = [];
  for (const conn of observedEdges(model).filter((c) => c.to === outputNode.id)) {
    const from = model.components.find((c) => c.id === conn.from);
    if (!from) continue;
    if (from.type === "softmax" || from.type === "sigmoid") {
      issues.push({
        id: `output-act-${conn.id}`,
        ruleId: "output-activation",
        severity: "info",
        category: "ordering",
        title: `Explicit ${from.type} before Output`,
        message: `"${from.name}" feeds directly into Output. PyTorch's nn.CrossEntropyLoss already applies log-softmax internally, an explicit Softmax causes double-application and degrades training stability.`,
        affectedIds: [from.id, outputNode.id],
        suggestion: "Remove Softmax/Sigmoid for training. Restore it in a separate inference wrapper or ONNX export.",
        fix: {
          kind: "delete-component",
          componentId: from.id,
          label: `Remove "${from.name}" and reconnect the graph`
        }
      });
    }
  }
  return issues;
};
var bnAtOutput = (model) => {
  const outputNode = model.components.find((c) => c.type === "output");
  if (!outputNode) return [];
  const issues = [];
  for (const conn of observedEdges(model).filter((c) => c.to === outputNode.id)) {
    const from = model.components.find((c) => c.id === conn.from);
    if (!from) continue;
    if (NORM_TYPES.has(from.type)) {
      issues.push({
        id: `bn-output-${conn.id}`,
        ruleId: "bn-at-output",
        severity: "warning",
        category: "ordering",
        title: "Normalization immediately before Output",
        message: `"${from.name}" (${from.type}) is the last layer before Output. Normalizing the raw logits constrains the output range and breaks standard loss functions.`,
        affectedIds: [from.id, outputNode.id],
        suggestion: "Move normalization before the final Linear/Conv layer."
      });
    }
  }
  return issues;
};
var deepNoResidual = (model) => {
  const deepLayers = model.components.filter((c) => DEEP_LAYER_TYPES.has(c.type));
  if (deepLayers.length < DEEP_NO_RESIDUAL_MIN_LAYERS) return [];
  if (model.components.some((c) => RESIDUAL_TYPES.has(c.type))) return [];
  if (hasUnmodelledMerge(model)) return [];
  return [{
    id: "deep-no-residual",
    ruleId: "deep-no-residual",
    severity: "warning",
    category: "pattern",
    title: `${deepLayers.length}-layer network without skip connections`,
    message: `${deepLayers.length} conv/linear layers detected but no residual (Add/Skip) layers. Networks deeper than ${DEEP_NO_RESIDUAL_MIN_LAYERS} layers are highly prone to vanishing gradients without skip connections.`,
    affectedIds: [],
    suggestion: "Add Residual or Add layers every 2-4 layers (ResNet-style). For transformers, use the built-in TransformerBlock which includes residuals."
  }];
};
var jepaTargetNeedsStopGrad = (model) => {
  const predictors = model.components.filter((c) => c.type === "jepaPredictor");
  if (predictors.length === 0) return [];
  const stopped = model.components.some(
    (c) => c.type === "emaTarget" || c.params?.stopGradient === true
  );
  if (stopped) return [];
  return [{
    id: "jepa-target-no-stop-grad",
    ruleId: "jepa-target-no-stop-grad",
    severity: "warning",
    category: "pattern",
    title: "Predictive branch with no stop-gradient on the target",
    message: `A JEPA predictor is present but no branch is marked as an EMA / stop-gradient target. Joint-embedding predictive objectives have a trivial solution (predict a constant), and the momentum target with no gradient is the only thing that rules it out.`,
    affectedIds: predictors.map((c) => c.id),
    suggestion: "Add an EMA Target / Stop-Gradient node at the end of the target encoder branch (or set stopGradient on it). BYOL and SimSiam both collapse without it."
  }];
};
var matryoshkaDimsExceedWidth = (model) => {
  const issues = [];
  for (const c of model.components) {
    if (c.type !== "matryoshkaHead") continue;
    const width = Number(c.params?.embedDim);
    const nested = Array.isArray(c.params?.nestedDims) ? c.params.nestedDims : [];
    if (!Number.isFinite(width) || width <= 0 || nested.length === 0) continue;
    const tooBig = nested.map(Number).filter((d) => Number.isFinite(d) && d > width);
    if (tooBig.length === 0) continue;
    issues.push({
      id: `matryoshka-dims-exceed-width-${c.id}`,
      ruleId: "matryoshka-dims-exceed-width",
      severity: "error",
      category: "structure",
      title: "Matryoshka prefix wider than the embedding",
      message: `"${c.name}" publishes nested dimension(s) ${tooBig.join(", ")} against a ${width}-wide embedding. A prefix longer than the whole vector does not exist, so a caller who truncates to it silently gets ${width} instead.`,
      affectedIds: [c.id],
      suggestion: `Drop the dimensions above ${width}, or widen embedDim to at least ${Math.max(...tooBig)}.`
    });
  }
  return issues;
};
var POSITION_TABLE_NAME = /(^|[._\s-])(wpe|pos|position|positions|positional)([._\s-]|_?emb|_?embed|_?embedding|_?embeddings|$)/i;
var MIN_POSITION_ROWS = 16;
var hasLearnedPositionTable = (model) => {
  const embeddings = model.components.filter((c) => c.type === "embedding");
  if (embeddings.length < 2) return false;
  const embById = new Map(embeddings.map((c) => [c.id, c]));
  const typeById = new Map(model.components.map((c) => [c.id, c.type]));
  const embInputsByAdd = /* @__PURE__ */ new Map();
  for (const cn of model.connections ?? []) {
    if (!embById.has(cn.from) || typeById.get(cn.to) !== "add") continue;
    const list2 = embInputsByAdd.get(cn.to) ?? [];
    if (!list2.includes(cn.from)) list2.push(cn.from);
    embInputsByAdd.set(cn.to, list2);
  }
  for (const ids of embInputsByAdd.values()) {
    if (ids.length < 2) continue;
    for (const id of ids) {
      const e = embById.get(id);
      if (POSITION_TABLE_NAME.test(e.name ?? "") || POSITION_TABLE_NAME.test(e.id)) return true;
      const rows = Number(e.params?.vocabSize ?? e.params?.numEmbeddings);
      if (Number.isFinite(rows) && rows >= MIN_POSITION_ROWS) {
        const others = ids.filter((o) => o !== id).map((o) => Number(embById.get(o).params?.vocabSize ?? embById.get(o).params?.numEmbeddings));
        if (others.some((n2) => Number.isFinite(n2) && n2 > rows)) return true;
      }
    }
  }
  return false;
};
var attentionNoPE = (model) => {
  const attnNodes = model.components.filter((c) => ATTENTION_TYPES.has(c.type));
  if (attnNodes.length === 0) return [];
  if (model.components.some((c) => PE_TYPES.has(c.type))) return [];
  if (hasLearnedPositionTable(model)) return [];
  return [{
    id: "attention-no-pe",
    ruleId: "attention-no-pe",
    severity: "warning",
    category: "pattern",
    title: "Attention without positional encoding",
    message: `${attnNodes.length} attention layer(s) present but no positional encoding found. Attention is permutation-invariant, without position information the model cannot distinguish token order.`,
    affectedIds: attnNodes.map((c) => c.id),
    suggestion: "Add a PositionalEncoding (sinusoidal) or RoPE layer before the first attention layer."
  }];
};
var vanishingGradientRisk = (model) => {
  const nonIO = model.components.filter((c) => c.type !== "input" && c.type !== "output");
  if (nonIO.length < VANISHING_MIN_LAYERS) return [];
  return model.components.filter((c) => VANISHING_ACTIVATIONS.has(c.type)).map((c) => ({
    id: `vanishing-${c.id}`,
    ruleId: "vanishing-gradient",
    severity: "info",
    category: "pattern",
    title: `Vanishing gradient risk: "${c.name}"`,
    message: `${c.type === "sigmoid" ? "Sigmoid" : "Tanh"} saturates to [0,1] / [-1,1], and its gradient approaches zero for large inputs. In networks deeper than ${VANISHING_MIN_LAYERS} layers, this halts learning in early layers.`,
    affectedIds: [c.id],
    suggestion: "Use ReLU, GELU, or SiLU for hidden layers. Keep Sigmoid only at binary classification outputs; Tanh in specific contexts (GAN generators, LSTM gates)."
  }));
};
var longestUnnormalizedConvRun = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const next = /* @__PURE__ */ new Map();
  for (const cn of observedEdges(model)) next.set(cn.from, [...next.get(cn.from) ?? [], cn.to]);
  const input = model.components.find((c) => c.type === "input");
  const seen = /* @__PURE__ */ new Set();
  let frontier = input ? [input.id] : [];
  let run = [];
  let best = [];
  while (frontier.length > 0) {
    const id = frontier.shift();
    if (seen.has(id)) continue;
    seen.add(id);
    const c = byId.get(id);
    if (c) {
      if (c.type === "conv2d" || c.type === "conv1d" || c.type === "conv3d") {
        run = [...run, c.id];
        if (run.length > best.length) best = run;
      } else if (NORM_TYPES.has(c.type)) {
        run = [];
      }
    }
    frontier = [...frontier, ...next.get(id) ?? []];
  }
  return { run: best.length, ids: best };
};
var DEEP_CONV_RUN_WARN = 5;
var deepNoNorm = (model) => {
  const issues = [];
  const { run, ids } = longestUnnormalizedConvRun(model);
  if (run >= DEEP_CONV_RUN_WARN) {
    issues.push({
      id: "deep-no-norm-conv-run",
      ruleId: "deep-no-norm",
      severity: "warning",
      category: "pattern",
      title: `${run} consecutive conv layers with no normalization`,
      message: `A conv stack this deep with no normalization between the layers trains poorly on realistic budgets. Measured in our production training corpus: designs with a run of ${run >= 6 ? run : 6} reached 15-27% of the reference accuracy on the same budget; shallower or normalized stacks reached 93-100%.`,
      affectedIds: ids,
      suggestion: "Insert BatchNorm every 2-3 conv layers (or GroupNorm for small batches), and downsample with pooling or strides as depth grows."
    });
  }
  const nonIO = model.components.filter((c) => c.type !== "input" && c.type !== "output");
  if (nonIO.length >= DEEP_NO_NORM_MIN_LAYERS && !model.components.some((c) => NORM_TYPES.has(c.type)) && run < DEEP_CONV_RUN_WARN) {
    issues.push({
      id: "deep-no-norm",
      ruleId: "deep-no-norm",
      severity: "info",
      category: "pattern",
      title: "No normalization in deep network",
      message: `${nonIO.length} layers with no BatchNorm, LayerNorm, or GroupNorm. Without normalization, activations can explode or vanish across layers, causing slow or unstable training.`,
      affectedIds: [],
      suggestion: "Add BatchNorm after Conv2d (CV tasks), LayerNorm after attention/FFN (NLP/LLM), or GroupNorm for small batch sizes."
    });
  }
  return issues;
};
var highDropout = (model) => {
  return model.components.filter((c) => c.type === "dropout" && (c.params.p ?? 0.5) > HIGH_DROPOUT_P).map((c) => ({
    id: `high-dropout-${c.id}`,
    ruleId: "high-dropout",
    severity: "warning",
    category: "performance",
    title: `Excessive dropout: p=${c.params.p}`,
    message: `"${c.name}" drops ${Math.round((c.params.p ?? 0.5) * 100)}% of activations per forward pass. Rates above ${HIGH_DROPOUT_P} introduce so much noise that the model cannot learn stable representations.`,
    affectedIds: [c.id],
    suggestion: "Use p \u2208 [0.1, 0.5] for most hidden layers. p=0.1-0.2 for conv layers; p=0.3-0.5 for fully-connected layers."
  }));
};
var largeActivation = (model) => {
  const offenders = [];
  for (const comp of model.components) {
    if (!comp.inputShape || !Array.isArray(comp.inputShape)) continue;
    const def = componentRegistry[comp.type];
    if (!def) continue;
    try {
      const outShape = def.computeOutputShape(comp.inputShape, comp.params);
      if (!Array.isArray(outShape)) continue;
      const elements = outShape.reduce((a, b) => a * b, 1);
      if (elements > LARGE_ACTIVATION_MAX_ELEMENTS) offenders.push({ comp, elements });
    } catch {
    }
  }
  if (offenders.length === 0) return [];
  offenders.sort((a, b) => b.elements - a.elements);
  const top = offenders[0];
  const mb = Math.round(top.elements * 4 / 1048576);
  const others = offenders.length - 1;
  return [{
    id: `large-act-${top.comp.id}`,
    ruleId: "large-activation",
    severity: "warning",
    category: "performance",
    title: others > 0 ? `Large activations: ~${mb} MB/sample across ${offenders.length} layers` : `Large activation: ~${mb} MB/sample`,
    message: others > 0 ? `"${top.comp.name}" outputs ${(top.elements / 1e6).toFixed(1)}M elements (~${mb} MB float32/sample), and ${others} more layer${others !== 1 ? "s" : ""} carry similarly large tensors downstream of the same oversized dimension. At batch_size=32 the largest layer alone requires ~${mb * 32} MB of activation memory.` : `"${top.comp.name}" outputs ${(top.elements / 1e6).toFixed(1)}M elements (~${mb} MB float32/sample). At batch_size=32 this single layer requires ~${mb * 32} MB of activation memory.`,
    affectedIds: offenders.map((o) => o.comp.id),
    suggestion: "Shrink the oversized input dimension (sequence length / spatial size), or add pooling / strided convolutions before the first large layer. Use AMP (float16) to halve activation memory."
  }];
};
var moeNoAuxLoss = (model) => {
  const moes = model.components.filter((c) => c.type === "moeLayer");
  if (moes.length === 0) return [];
  const many = moes.length > 1;
  return [{
    id: `moe-aux-${moes[0].id}`,
    ruleId: "moe-no-aux-loss",
    severity: "info",
    category: "pattern",
    title: many ? `${moes.length} MoE layers: add auxiliary load-balancing loss` : `MoE "${moes[0].name}": add auxiliary load-balancing loss`,
    message: `MoE layers require an auxiliary router z-loss + load-balance loss during training to prevent expert collapse. This is not visible in the architecture diagram but must be in the training loop.${many ? ` Applies to all ${moes.length}: ${moes.map((c) => c.name).join(", ")}.` : ""}`,
    affectedIds: moes.map((c) => c.id),
    suggestion: `Add a note on ${many ? "these layers" : "this layer"}. Typical aux_loss coefficient: 1e-2 (Mixtral/Switch Transformer).`
  }];
};
var gqaHeadMismatch = (model) => {
  return model.components.filter((c) => c.type === "groupedQueryAttention").filter((c) => {
    const H = Number(c.params.numHeads) || 32;
    const Hkv = Number(c.params.numKVHeads) || 8;
    return H % Hkv !== 0;
  }).map((c) => {
    const H = Number(c.params.numHeads) || 32;
    const Hkv = Number(c.params.numKVHeads) || 8;
    const snapped = largestDivisorAtMost(H, Hkv);
    return {
      id: `gqa-mismatch-${c.id}`,
      ruleId: "gqa-head-mismatch",
      severity: "error",
      category: "structure",
      title: `GQA "${c.name}": numHeads not divisible by numKVHeads`,
      message: `numHeads (${H}) must be divisible by numKVHeads (${Hkv}) for grouped-query attention to work correctly.`,
      affectedIds: [c.id],
      suggestion: `Set numKVHeads to a divisor of ${H}, e.g. ${snapped}.`,
      fix: {
        kind: "update-params",
        componentId: c.id,
        params: { numKVHeads: snapped },
        label: `Set numKVHeads to ${snapped}`
      }
    };
  });
};
var swigluDimConvention = (model) => {
  return model.components.filter((c) => c.type === "swiglu").filter((c) => {
    const D = Number(c.params.embedDim ?? c.params.inFeatures) || 4096;
    const I = Number(c.params.intermediateSize ?? c.params.hiddenFeatures ?? c.params.ffDim);
    if (!I) return false;
    const ratio = I / D;
    return ratio < SWIGLU_RATIO_MIN || ratio > SWIGLU_RATIO_MAX;
  }).map((c) => {
    const D = Number(c.params.embedDim ?? c.params.inFeatures) || 4096;
    const I = Number(c.params.intermediateSize ?? c.params.hiddenFeatures ?? c.params.ffDim) || 0;
    const recommended = Math.round(Math.round(D * 8 / 3 / 256) * 256);
    return {
      id: `swiglu-dim-${c.id}`,
      ruleId: "swiglu-dim-convention",
      severity: "info",
      category: "performance",
      title: `SwiGLU "${c.name}": intermediateSize looks non-standard`,
      message: `LLaMA uses intermediateSize \u2248 \u230A(8/3 \xD7 D) / 256\u230B \xD7 256. Current: ${I} (${(I / D).toFixed(2)}\xD7 embedDim). Expected: ~${recommended}.`,
      affectedIds: [c.id],
      suggestion: `Set intermediateSize to ${recommended} for embedDim=${D}.`
    };
  });
};
var CONV_TYPES = /* @__PURE__ */ new Set([
  "conv1d",
  "conv2d",
  "conv3d",
  "depthwiseConv2d",
  "separableConv2d",
  "transposeConv2d"
]);
var linearAfterConvNoFlatten = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of observedEdges(model)) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (CONV_TYPES.has(from.type) && to.type === "linear") {
      issues.push({
        id: `conv-linear-${conn.id}`,
        ruleId: "linear-after-conv-no-flatten",
        severity: "error",
        category: "structure",
        title: `Conv feeds Linear without flattening: "${from.name}" \u2192 "${to.name}"`,
        message: `"${from.name}" (${from.type}) outputs a multi-dimensional feature map but connects straight into "${to.name}" (Linear), which expects a flat [batch, features] tensor. At runtime this raises a shape error (or silently mis-multiplies the spatial dims).`,
        affectedIds: [from.id, to.id],
        suggestion: "Insert a Flatten (keep spatial info) or a Global Average Pool (collapse spatial dims) between the convolution and the Linear layer.",
        fix: {
          kind: "insert-on-connection",
          connectionId: conn.id,
          componentType: "flatten",
          label: `Insert Flatten between "${from.name}" and "${to.name}"`
        }
      });
    }
  }
  return issues;
};
var redundantActivation = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of observedEdges(model)) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (ACTIVATION_TYPES.has(from.type) && ACTIVATION_TYPES.has(to.type)) {
      const same = from.type === to.type;
      issues.push({
        id: `redundant-act-${conn.id}`,
        ruleId: "redundant-activation",
        severity: "warning",
        category: "ordering",
        title: `Back-to-back activations: "${from.name}" \u2192 "${to.name}"`,
        message: same ? `"${from.name}" and "${to.name}" are both ${from.type}. Applying the same activation twice adds compute but no expressivity, almost always a duplicated line.` : `"${from.name}" (${from.type}) feeds straight into "${to.name}" (${to.type}). Stacking two activations with no Linear/Conv between them is rarely intended, e.g. a ReLU before a Softmax clips logits to \u2265 0 and distorts the output distribution.`,
        affectedIds: [from.id, to.id],
        suggestion: "Remove one activation, or insert the Linear/Conv/Norm layer that belongs between them."
      });
    }
  }
  return issues;
};
var consecutiveLinearNoActivation = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of observedEdges(model)) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (from.type === "linear" && to.type === "linear") {
      issues.push({
        id: `consecutive-linear-${conn.id}`,
        ruleId: "consecutive-linear-no-activation",
        severity: "info",
        category: "pattern",
        title: `Linear \u2192 Linear with no activation: "${from.name}" \u2192 "${to.name}"`,
        message: `"${from.name}" feeds directly into "${to.name}" with no activation between them. Two stacked linear maps collapse into one (W\u2082\xB7W\u2081), so the extra layer costs parameters but adds no representational power.`,
        affectedIds: [from.id, to.id],
        suggestion: "Add a non-linearity (ReLU/GELU) between them. If this is a deliberate low-rank / factorized projection (down-proj \u2192 up-proj), this hint is safe to ignore."
      });
    }
  }
  return issues;
};
var dropoutAtOutput = (model) => {
  const outputNode = model.components.find((c) => c.type === "output");
  if (!outputNode) return [];
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of observedEdges(model).filter((c) => c.to === outputNode.id)) {
    const from = byId.get(conn.from);
    if (!from || from.type !== "dropout") continue;
    issues.push({
      id: `dropout-output-${conn.id}`,
      ruleId: "dropout-at-output",
      severity: "warning",
      category: "ordering",
      title: `Dropout immediately before Output: "${from.name}"`,
      message: `"${from.name}" (Dropout) is the last layer before Output. In training it randomly zeroes the final logits themselves, injecting noise straight into the loss; at eval it is a no-op, so train and eval behaviour diverge. Dropout belongs before the final projection, not after it.`,
      affectedIds: [from.id, outputNode.id],
      suggestion: "Move the Dropout before the final Linear/Conv that produces the logits."
    });
  }
  return issues;
};
var STRIDED_CONV_TYPES = /* @__PURE__ */ new Set([
  "conv1d",
  "conv2d",
  "conv3d",
  "depthwiseConv1d",
  "depthwiseConv2d",
  "separableConv2d"
]);
var RESIDUAL_PASSTHROUGH = /* @__PURE__ */ new Set(["batchNorm", "layerNorm", "groupNorm", "instanceNorm"]);
var ELEMENTWISE_MERGE_TYPES = /* @__PURE__ */ new Set(["add", "multiply", "mean"]);
function isProjectionShortcut(model, convId, kernel) {
  if (kernel !== 1) return false;
  const byId = new Map(model.components.map((c) => [c.id, c]));
  let frontier = [convId];
  for (let hop = 0; hop < 3 && frontier.length; hop++) {
    const next = [];
    for (const id of frontier) {
      for (const e of model.connections.filter((x) => x.from === id)) {
        const to = byId.get(e.to);
        if (!to) continue;
        if (ELEMENTWISE_MERGE_TYPES.has(to.type)) return true;
        if (RESIDUAL_PASSTHROUGH.has(to.type)) next.push(to.id);
      }
    }
    frontier = next;
  }
  return false;
}
var convStrideGtKernel = (model) => {
  return model.components.filter((c) => STRIDED_CONV_TYPES.has(c.type)).map((c) => ({ c, k: Number(c.params.kernelSize), s: Number(c.params.stride) })).filter(({ k, s }) => Number.isFinite(k) && Number.isFinite(s) && k > 0 && s > k).filter(({ c, k }) => !isProjectionShortcut(model, c.id, k)).map(({ c, k, s }) => ({
    id: `conv-stride-gt-kernel-${c.id}`,
    ruleId: "conv-stride-gt-kernel",
    severity: "warning",
    category: "structure",
    title: `Conv stride exceeds kernel: "${c.name}"`,
    message: `"${c.name}" has stride ${s} > kernelSize ${k}. Each step the kernel jumps ${s - k} pixel(s) past its own footprint, so a band of the input is never read, a silent loss of information. Non-overlapping patches use stride == kernel (e.g. ViT 16/16); stride > kernel is almost always a typo.`,
    affectedIds: [c.id],
    suggestion: `Set stride \u2264 kernelSize (${k}); use stride == kernelSize for non-overlapping patches.`
  }));
};
var SPATIAL_CONV_TYPES = /* @__PURE__ */ new Set([
  "conv1d",
  "conv2d",
  "conv3d",
  "depthwiseConv1d",
  "depthwiseConv2d",
  "separableConv2d"
]);
var nonSpatialIntoConv = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of observedEdges(model)) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if ((from.type === "flatten" || from.type === "linear") && SPATIAL_CONV_TYPES.has(to.type)) {
      issues.push({
        id: `non-spatial-into-conv-${conn.id}`,
        ruleId: "non-spatial-into-conv",
        severity: "warning",
        category: "structure",
        title: `Non-spatial tensor into Conv: "${from.name}" \u2192 "${to.name}"`,
        message: `"${from.name}" (${from.type}) emits a flat [batch, features] vector, but "${to.name}" (${to.type}) expects a [channels, \u2026spatial] feature map. The forward pass raises a shape error unless the dimensions are restored first.`,
        affectedIds: [from.id, to.id],
        suggestion: "Insert a Reshape / Unflatten to rebuild the spatial dims before the convolution."
      });
    }
  }
  return issues;
};
var doubleNorm = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of observedEdges(model)) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (NORM_TYPES.has(from.type) && NORM_TYPES.has(to.type)) {
      issues.push({
        id: `double-norm-${conn.id}`,
        ruleId: "double-norm",
        severity: "info",
        category: "pattern",
        title: `Back-to-back normalization: "${from.name}" \u2192 "${to.name}"`,
        message: `"${from.name}" (${from.type}) feeds directly into "${to.name}" (${to.type}). Normalizing an already-normalized tensor is redundant, the second layer mostly re-centres/re-scales what the first produced and just burns its own learnable parameters.`,
        affectedIds: [from.id, to.id],
        suggestion: "Keep a single normalization layer here, or move one of them next to the layer whose activations it should stabilize."
      });
    }
  }
  return issues;
};
var duplicatePositionalEncoding = (model) => {
  const pes = model.components.filter((c) => PE_TYPES.has(c.type));
  if (pes.length < 2) return [];
  const outgoing = /* @__PURE__ */ new Map();
  for (const conn of observedEdges(model)) {
    const list2 = outgoing.get(conn.from);
    if (list2) list2.push(conn.to);
    else outgoing.set(conn.from, [conn.to]);
  }
  const peById = new Map(pes.map((c) => [c.id, c]));
  for (const pe of pes) {
    const seen = /* @__PURE__ */ new Set([pe.id]);
    const stack = [...outgoing.get(pe.id) ?? []];
    while (stack.length > 0) {
      const id = stack.pop();
      if (seen.has(id)) continue;
      seen.add(id);
      const downstream = peById.get(id);
      if (downstream) {
        return [{
          id: "duplicate-positional-encoding",
          ruleId: "duplicate-positional-encoding",
          severity: "info",
          category: "pattern",
          title: "Position injected twice on one stream",
          message: `"${pe.name}" (${pe.type}) and "${downstream.name}" (${downstream.type}) both add position to the same tensor. Stacking absolute + rotary, or two of the same, double-counts position and tends to hurt more than help. Separate towers with one each (a vision and a text branch, an encoder and a decoder) are not this.`,
          affectedIds: [pe.id, downstream.id],
          suggestion: "Keep a single positional scheme on this path: sinusoidal OR learned OR RoPE / ALiBi."
        }];
      }
      stack.push(...outgoing.get(id) ?? []);
    }
  }
  return [];
};
var SPATIAL_POOL_TYPES = /* @__PURE__ */ new Set([
  "maxpool2d",
  "avgpool2d",
  "maxpool1d",
  "avgpool1d",
  "adaptiveMaxPool2d",
  "adaptiveAvgPool2d"
]);
var poolIntoLinearNoFlatten = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of observedEdges(model)) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (SPATIAL_POOL_TYPES.has(from.type) && to.type === "linear") {
      issues.push({
        id: `pool-into-linear-${conn.id}`,
        ruleId: "pool-into-linear-no-flatten",
        severity: "warning",
        category: "structure",
        title: `Pooling feeds Linear without flattening: "${from.name}" \u2192 "${to.name}"`,
        message: `"${from.name}" (${from.type}) still carries [channels, \u2026spatial] dims, but "${to.name}" (Linear) expects a flat [batch, features] tensor. At runtime this raises a shape error.`,
        affectedIds: [from.id, to.id],
        suggestion: "Insert a Flatten (keep spatial info) or a Global Average Pool (collapse spatial dims) between the pooling and the Linear layer.",
        fix: {
          kind: "insert-on-connection",
          connectionId: conn.id,
          componentType: "flatten",
          label: `Insert Flatten between "${from.name}" and "${to.name}"`
        }
      });
    }
  }
  return issues;
};
var flattenIntoAttention = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of observedEdges(model)) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (from.type === "flatten" && ATTENTION_TYPES.has(to.type)) {
      issues.push({
        id: `flatten-into-attention-${conn.id}`,
        ruleId: "flatten-into-attention",
        severity: "warning",
        category: "structure",
        title: `Flatten before attention: "${from.name}" \u2192 "${to.name}"`,
        message: `"${from.name}" (Flatten) collapses the sequence dimension into one long vector, but "${to.name}" (${to.type}) attends over a [sequence, dim] tensor. Flattening first leaves a length-1 sequence, so attention has nothing to relate.`,
        affectedIds: [from.id, to.id],
        suggestion: "Remove the Flatten before attention; keep the [sequence, dim] layout and flatten (or pool) only after the attention stack."
      });
    }
  }
  return issues;
};
var transposeConvCheckerboard = (model) => {
  return model.components.filter((c) => c.type === "transposeConv2d").map((c) => ({ c, k: Number(c.params.kernelSize), s: Number(c.params.stride) })).filter(({ k, s }) => Number.isFinite(k) && Number.isFinite(s) && s > 1 && k % s !== 0).map(({ c, k, s }) => ({
    id: `transposeconv-checkerboard-${c.id}`,
    ruleId: "transposeconv-checkerboard",
    severity: "info",
    category: "pattern",
    title: `Checkerboard risk: "${c.name}"`,
    message: `ConvTranspose "${c.name}" has kernelSize ${k} not divisible by stride ${s}. The uneven kernel overlap during upsampling deposits more weight on some output pixels than others, producing checkerboard artifacts.`,
    affectedIds: [c.id],
    suggestion: `Make kernelSize a multiple of stride (e.g. ${s * Math.max(2, Math.round(k / s))}), or upsample with Upsample + Conv (resize-convolution) instead.`
  }));
};
var groupNormDivisibility = (model) => {
  const issues = [];
  for (const c of model.components) {
    if (c.type !== "groupNorm") continue;
    const groups = Number(c.params.numGroups);
    const channels = Number(
      c.params.numChannels ?? (Array.isArray(c.inputShape) ? c.inputShape[0] : void 0)
    );
    if (!Number.isFinite(groups) || !Number.isFinite(channels) || groups <= 0 || channels <= 0) continue;
    if (channels % groups !== 0) {
      const divisor = [1, 2, 4, 8, 16, 32].filter((d) => channels % d === 0).pop() ?? 1;
      issues.push({
        id: `groupnorm-divisibility-${c.id}`,
        ruleId: "groupnorm-divisibility",
        severity: "error",
        category: "structure",
        title: `GroupNorm "${c.name}": channels not divisible by numGroups`,
        message: `GroupNorm "${c.name}" has numGroups=${groups} but ${channels} channels. ${channels} % ${groups} = ${channels % groups}; PyTorch requires the channel count to be an exact multiple of numGroups.`,
        affectedIds: [c.id],
        suggestion: `Set numGroups to a divisor of ${channels}, e.g. ${divisor}.`,
        fix: {
          kind: "update-params",
          componentId: c.id,
          params: { numGroups: divisor },
          label: `Set numGroups to ${divisor}`
        }
      });
    }
  }
  return issues;
};
var hugeLinearParams = (model) => {
  const issues = [];
  for (const c of model.components) {
    if (c.type !== "linear") continue;
    const outF = Number(c.params.outFeatures);
    const inF = Number(
      c.params.inFeatures ?? (Array.isArray(c.inputShape) ? c.inputShape[c.inputShape.length - 1] : void 0)
    );
    if (!Number.isFinite(inF) || !Number.isFinite(outF) || inF <= 0 || outF <= 0) continue;
    const count = inF * outF;
    if (count > LARGE_LINEAR_MAX_PARAMS) {
      const gb2 = (count * 4 / 1073741824).toFixed(1);
      issues.push({
        id: `huge-linear-${c.id}`,
        ruleId: "huge-linear-params",
        severity: "warning",
        category: "performance",
        title: `Very large Linear: "${c.name}" (~${Math.round(count / 1e6)}M params)`,
        message: `Linear "${c.name}" is ${inF} \xD7 ${outF} = ${Math.round(count / 1e6)}M parameters (~${gb2} GB float32). A single dense layer this large usually means a feature map was flattened without pooling first; embedding / vocab-projection heads are the expected exception.`,
        affectedIds: [c.id],
        suggestion: "Add a Global Average Pool or more downsampling before the Linear, or factorize it (low-rank / bottleneck projection)."
      });
    }
  }
  return issues;
};
var MHA_SERVING_MIN_LAYERS = 6;
var MHA_SERVING_MIN_DIM = 2048;
var MHA_FAMILY = /* @__PURE__ */ new Set(["multiHeadAttention", "selfAttention", "causalAttention"]);
var fullMhaServingCost = (model) => {
  if (model.components.some((c) => c.type === "mla")) return [];
  const reducedGqa = model.components.some(
    (c) => c.type === "groupedQueryAttention" && Number(c.params.numKVHeads) > 0 && Number(c.params.numKVHeads) < Number(c.params.numHeads ?? c.params.numKVHeads)
  );
  if (reducedGqa) return [];
  const mhaLayers = model.components.filter((c) => {
    if (MHA_FAMILY.has(c.type)) return true;
    if (c.type === "groupedQueryAttention") {
      const q = Number(c.params.numHeads);
      const kv = Number(c.params.numKVHeads);
      return !(kv > 0 && q > 0 && kv < q);
    }
    return false;
  });
  if (mhaLayers.length < MHA_SERVING_MIN_LAYERS) return [];
  const dims = mhaLayers.map((c) => Number(c.params.embedDim ?? c.params.hiddenDim ?? c.params.dModel)).filter((d) => Number.isFinite(d) && d > 0);
  if (dims.length === 0) return [];
  const embedDim = Math.round(dims.reduce((a, b) => a + b, 0) / dims.length);
  if (embedDim < MHA_SERVING_MIN_DIM) return [];
  const kvPerTokenKB = mhaLayers.length * 2 * embedDim * 2 / 1024;
  return [{
    id: "full-mha-serving",
    ruleId: "full-mha-serving-cost",
    severity: "info",
    category: "performance",
    title: `Full multi-head attention at LLM scale (~${kvPerTokenKB.toFixed(0)} KB/token KV cache)`,
    message: `${mhaLayers.length} attention layers at embedDim ${embedDim} cache full per-head K/V: about ${kvPerTokenKB.toFixed(0)} KB per token at fp16, which dominates memory at long context. Grouped-query attention (e.g. 8:1) would cut this ~8\xD7; multi-head latent attention (MLA) shrinks it ~10\xD7 or more. This is the move production LLMs make; it does not change the parameter count.`,
    affectedIds: mhaLayers.map((c) => c.id),
    suggestion: "Switch attention to groupedQueryAttention (set numKVHeads below numHeads, e.g. numHeads/4) or mla (a low-rank cached latent)."
  }];
};
var SATURATING_ACTIVATIONS = /* @__PURE__ */ new Set(["sigmoid", "tanh"]);
var WEIGHTED_INIT_TYPES = /* @__PURE__ */ new Set([
  "linear",
  "conv1d",
  "conv2d",
  "conv3d",
  "depthwiseConv2d",
  "separableConv2d",
  "transposeConv2d"
]);
var initActivationMismatch = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const affected = [];
  const actNames = /* @__PURE__ */ new Set();
  for (const conn of observedEdges(model)) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (WEIGHTED_INIT_TYPES.has(from.type) && SATURATING_ACTIVATIONS.has(to.type)) {
      affected.push(from.id, to.id);
      actNames.add(to.type);
    }
  }
  if (affected.length === 0) return [];
  return [{
    id: "init-activation-mismatch",
    ruleId: "init-activation-mismatch",
    severity: "info",
    category: "pattern",
    title: `Default init assumes ReLU, but ${[...actNames].join("/")} follows`,
    message: "PyTorch initializes Linear/Conv with Kaiming (He) init, which is derived for ReLU-family activations. Feeding a saturating activation (sigmoid/tanh) from a He-initialized layer starts training in the saturated tails, shrinking early gradients.",
    affectedIds: [...new Set(affected)],
    suggestion: 'Initialize these layers with Xavier instead: nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("sigmoid"|"tanh")), or switch the activation to a ReLU-family one.'
  }];
};
var deepAttentionDefaultInit = (model) => {
  const attn = model.components.filter((c) => ATTENTION_TYPES.has(c.type));
  if (attn.length < SCALED_INIT_MIN_ATTENTION_LAYERS) return [];
  const advice = getInitAdvice("causalAttention");
  return [{
    id: "deep-attention-default-init",
    ruleId: "deep-attention-default-init",
    severity: "info",
    category: "pattern",
    title: `${attn.length} attention layers: use depth-scaled init`,
    message: `At ${attn.length} stacked attention layers, residual-branch outputs add up; unscaled init lets activation variance grow with depth. GPT-2/LLaMA-family models scale the residual projections by depth${advice ? ` (${advice.formula})` : ""}.`,
    affectedIds: attn.map((c) => c.id),
    suggestion: advice ? `Scale residual output projections by depth: ${advice.pyTorch}` : "Scale residual output projections by 1/sqrt(2 \xB7 numLayers)."
  }];
};
var lmHeadVocabMismatch = (model) => {
  const vocabs = model.components.filter((c) => c.type === "embedding").map((c) => Number(c.params.vocabSize)).filter((v) => Number.isFinite(v) && v > 0);
  if (vocabs.length === 0) return [];
  if (!model.components.some((c) => ATTENTION_TYPES.has(c.type))) return [];
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const outputIds = new Set(model.components.filter((c) => c.type === "output").map((c) => c.id));
  if (outputIds.size === 0) return [];
  const issues = [];
  for (const conn of observedEdges(model)) {
    if (!outputIds.has(conn.to)) continue;
    const head = byId.get(conn.from);
    if (!head || head.type !== "linear") continue;
    const out = Number(head.params.outFeatures);
    if (!Number.isFinite(out) || out <= 0) continue;
    if (vocabs.includes(out)) continue;
    issues.push({
      id: `lm-head-vocab-${head.id}`,
      ruleId: "lm-head-vocab-mismatch",
      severity: "info",
      category: "structure",
      title: `Head projects to ${out}, embedding vocab is ${vocabs[0]}`,
      message: `"${head.name}" feeds the output with outFeatures ${out}, but the embedding vocabulary is ${vocabs.join("/")}. For a language model the head must project back to vocab size (and is usually weight-tied to the embedding). If this is a classifier head over ${out} classes, ignore.`,
      affectedIds: [head.id],
      suggestion: `If this model predicts tokens, set outFeatures to ${vocabs[0]} and consider tying the head to the embedding weights.`
    });
  }
  return issues;
};
var kvCacheContextBudget = (model) => {
  let perTokenBytes = 0;
  const affected = [];
  for (const c of model.components) {
    if (!ATTENTION_TYPES.has(c.type) || c.type === "mla") continue;
    const embedDim = Number(c.params.embedDim ?? c.params.hiddenDim ?? c.params.dModel);
    const numHeads = Number(c.params.numHeads);
    if (!Number.isFinite(embedDim) || embedDim <= 0) continue;
    const kvHeads = c.type === "groupedQueryAttention" && Number(c.params.numKVHeads) > 0 && Number(c.params.numKVHeads) < numHeads ? Number(c.params.numKVHeads) : numHeads;
    const headDim = Number.isFinite(numHeads) && numHeads > 0 ? embedDim / numHeads : embedDim;
    const kvWidth = Number.isFinite(kvHeads) && kvHeads > 0 ? kvHeads * headDim : embedDim;
    perTokenBytes += 2 * kvWidth * 2;
    affected.push(c.id);
  }
  if (perTokenBytes === 0) return [];
  const totalGB = perTokenBytes * KV_BUDGET_CONTEXT_TOKENS / 1e9;
  if (totalGB <= KV_BUDGET_MAX_GB) return [];
  return [{
    id: "kv-cache-context-budget",
    ruleId: "kv-cache-context-budget",
    severity: "warning",
    category: "performance",
    title: `KV cache \u2248 ${totalGB.toFixed(1)} GB at ${KV_BUDGET_CONTEXT_TOKENS.toLocaleString()} tokens (fp16, 1 sequence)`,
    message: `Across ${affected.length} attention layers this design caches ${(perTokenBytes / 1024).toFixed(0)} KB per token, so a single ${KV_BUDGET_CONTEXT_TOKENS.toLocaleString()}-token sequence needs ~${totalGB.toFixed(1)} GB of KV cache before weights or activations. That exceeds the ${KV_BUDGET_MAX_GB} GB budget this rule assumes for serving headroom.`,
    affectedIds: affected,
    suggestion: "Cut KV width: raise the GQA ratio (fewer numKVHeads), switch to MLA, reduce depth or embedDim, or accept a shorter serving context."
  }];
};
var ALL_RULES = [
  noInputNode,
  // R01 structure  error
  noOutputNode,
  // R02 structure  warning
  isolatedComponents,
  // R03 structure  warning
  deadEnds,
  // R04 structure  warning
  bnAfterActivation,
  // R05 ordering   warning
  dropoutBeforeBN,
  // R06 ordering   info
  outputActivation,
  // R07 ordering   info
  bnAtOutput,
  // R08 ordering   warning
  deepNoResidual,
  // R09 pattern    warning
  attentionNoPE,
  // R10 pattern    warning
  vanishingGradientRisk,
  // R11 pattern info
  deepNoNorm,
  // R12 pattern    info
  highDropout,
  // R13 performance warning
  largeActivation,
  // R14 performance warning
  moeNoAuxLoss,
  // R15 pattern    info
  gqaHeadMismatch,
  // R16 structure  error
  swigluDimConvention,
  // R17 performance info
  linearAfterConvNoFlatten,
  // R18 structure error
  redundantActivation,
  // R19 ordering   warning
  consecutiveLinearNoActivation,
  // R20 pattern info
  dropoutAtOutput,
  // R21 ordering   warning
  convStrideGtKernel,
  // R22 structure  warning
  nonSpatialIntoConv,
  // R23 structure  warning
  doubleNorm,
  // R24 pattern    info
  duplicatePositionalEncoding,
  // R25 pattern info
  poolIntoLinearNoFlatten,
  // R26 structure  warning
  flattenIntoAttention,
  // R27 structure  warning
  transposeConvCheckerboard,
  // R28 pattern   info
  groupNormDivisibility,
  // R29 structure  error
  hugeLinearParams,
  // R30 performance warning
  fullMhaServingCost,
  // R31 performance info
  initActivationMismatch,
  // R32 pattern    info
  deepAttentionDefaultInit,
  // R33 pattern    info
  lmHeadVocabMismatch,
  // R34 structure  info
  kvCacheContextBudget,
  // R35 performance warning
  jepaTargetNeedsStopGrad,
  // R36 pattern    warning
  matryoshkaDimsExceedWidth
  // R37 structure  error
];
function runAdvisorRules(model, opts = {}) {
  if (model.components.length === 0) return [];
  const issues = [];
  for (const rule of ALL_RULES) {
    try {
      issues.push(...rule(model));
    } catch (e) {
      console.warn("[ArchitectureAdvisor] Rule error:", e);
    }
  }
  if (!opts.excludeIssueIds?.length) return issues;
  const drop = new Set(opts.excludeIssueIds);
  return issues.filter((i) => !drop.has(i.id));
}

// packages/graph-core/src/schema.ts
var COMPONENT_TYPES = [
  // Basic
  "input",
  "output",
  "linear",
  "flatten",
  // CV - Computer Vision
  "conv2d",
  "conv3d",
  "depthwiseConv2d",
  "separableConv2d",
  "transposeConv2d",
  "maxpool2d",
  "avgpool2d",
  "adaptiveAvgPool2d",
  "adaptiveMaxPool2d",
  "globalAvgPool2d",
  "globalMaxPool2d",
  "dilatedConv2d",
  "roiAlign",
  "maxpool3d",
  "avgpool3d",
  "upsample",
  "pixelShuffle",
  // NLP - Natural Language Processing
  "conv1d",
  "maxpool1d",
  "avgpool1d",
  "embedding",
  "segmentEmbedding",
  "lstm",
  "gru",
  "rnn",
  "bidirectionalLSTM",
  "bidirectionalGRU",
  "attention",
  "selfAttention",
  "crossAttention",
  "globalAvgPool1d",
  // LLM - Large Language Models
  "multiHeadAttention",
  "groupedQueryAttention",
  "causalAttention",
  "transformerBlock",
  "positionalEncoding",
  "feedForward",
  "rope",
  "lmHead",
  "timeEmbedding",
  "mamba",
  "relativePositionBias",
  "learnedPositionalEmbedding",
  "localAttention",
  "linearAttention",
  // Audio
  "melSpectrogram",
  "mfcc",
  "stft",
  "audioConv",
  "conformerBlock",
  "depthwiseConv1d",
  // Tabular
  "featureInteraction",
  "embeddingBag",
  "tabnet",
  // Recommendation (sequential / generative, 2018-2025)
  "targetAttention",
  "behaviorRetrieval",
  "multiInterest",
  "hstuBlock",
  // Representation learning / retrieval objectives
  "contrastiveHead",
  "matryoshkaHead",
  "lateInteraction",
  // Reinforcement Learning
  "dqnHead",
  "actorHead",
  "criticHead",
  "policyNetwork",
  "valueNetwork",
  // Graph ML
  "graphConv",
  "graphAttention",
  "graphSAGE",
  "gcn",
  "gat",
  "gin",
  "edgeConv",
  // Multimodal
  "patchMerger",
  "crossModalAttention",
  "fusion",
  "projection",
  "coAttention",
  // Activation
  "relu",
  "relu6",
  "leakyRelu",
  "elu",
  "prelu",
  "gelu",
  "swish",
  "selu",
  "mish",
  "hardSwish",
  "hardSigmoid",
  "logSoftmax",
  "glu",
  "softplus",
  "sigmoid",
  "tanh",
  "softmax",
  "gumbelSoftmax",
  // Normalization
  "batchNorm",
  "layerNorm",
  "instanceNorm",
  "groupNorm",
  "rmsNorm",
  "adaIN",
  "spectralNorm",
  "pixelNorm",
  "weightNorm",
  "localResponseNorm",
  // LLM extras
  "swiglu",
  "moeLayer",
  "alibi",
  // CV extras
  "seBlock",
  "patchEmbed",
  "windowAttention",
  "fpn",
  "invResidualBlock",
  "deformableConv2d",
  "interpolate",
  "channelShuffle",
  "gridSample",
  "spatialPyramidPool",
  // Frontier architectures (2024-2025)
  "mla",
  "mamba2",
  "qkNorm",
  "multiTokenPrediction",
  "xlstm",
  "differentialAttention",
  "rgLru",
  "retention",
  "hyena",
  "rwkv",
  "kan",
  "mixtureOfDepths",
  "tttLayer",
  "geglu",
  "grn",
  "titansMemory",
  "deltaNet",
  "gatedDeltaNet",
  "sharedExpertMoE",
  "ditBlock",
  "vectorQuantizer",
  "perceiverLatent",
  "convNeXtBlock",
  "gatedLinearAttention",
  "s4Layer",
  "dyt",
  "nativeSparseAttention",
  "film",
  "residualVQ",
  "crossNetworkDCN",
  "ftTransformerBlock",
  "deformableAttention",
  "attentionPool",
  // Time-series / 3D / Video (2024-2025)
  "revIN",
  "seriesDecomp",
  "setAbstraction",
  "sparseConv3d",
  "nerfPositionalEncoding",
  "dividedSpaceTimeAttention",
  "tubeletEmbed",
  "causalConv3d",
  // World models / model-based RL (2018-2025)
  "rssm",
  "latentActionModel",
  "jepaPredictor",
  "emaTarget",
  // Utility
  "dropout",
  "reshape",
  "residual",
  "skipConnection",
  "concatenate",
  "add",
  "multiply",
  "dropPath",
  "layerScale",
  "split",
  "permute",
  "customModule",
  "stickyNote",
  "squeeze",
  "unsqueeze",
  "pad",
  "mean",
  "matmul",
  "clamp",
  "norm",
  "vaeBottleneck",
  "miniBatchStdDev",
  "topK",
  "gather",
  "scatter",
  "stack",
  "einsum"
];
var KNOWN_TYPES = new Set(COMPONENT_TYPES);
var stripped = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
var STRIPPED_TYPES = COMPONENT_TYPES.map((t) => [t, stripped(t)]);
function resolveComponentType(raw) {
  if (KNOWN_TYPES.has(raw)) return raw;
  const lcc = raw.charAt(0).toLowerCase() + raw.slice(1);
  if (KNOWN_TYPES.has(lcc)) return lcc;
  const camel = raw.replace(/[-_](.)/g, (_, c) => c.toUpperCase());
  const camelLc = camel.charAt(0).toLowerCase() + camel.slice(1);
  if (KNOWN_TYPES.has(camelLc)) return camelLc;
  const rawStripped = stripped(raw);
  if (rawStripped.length === 0) return null;
  const exact = STRIPPED_TYPES.find(([, t]) => t === rawStripped);
  if (exact) return exact[0];
  const sub = STRIPPED_TYPES.filter(([, t]) => t.includes(rawStripped) || rawStripped.includes(t)).sort((a, b) => Math.abs(a[1].length - rawStripped.length) - Math.abs(b[1].length - rawStripped.length))[0];
  return sub ? sub[0] : null;
}

// src/utils/executorResolve.ts
function normalizeComponentType(raw) {
  const resolved = resolveComponentType(raw);
  if (resolved && resolved in componentRegistry) return resolved;
  if (raw in componentRegistry) return raw;
  return raw;
}

// src/utils/shapeInference.ts
var ELEMENTWISE_MERGE = /* @__PURE__ */ new Set(["add", "multiply", "mean"]);
var CONCAT_MERGE = /* @__PURE__ */ new Set(["concatenate"]);
var MATMUL_LAYERS = /* @__PURE__ */ new Set(["matmul"]);
var EMBED_MATCH_LAYERS = /* @__PURE__ */ new Set(["crossAttention"]);
var HEAD_DIM_LAYERS = /* @__PURE__ */ new Set([
  "multiHeadAttention",
  "selfAttention",
  "crossAttention",
  "attention",
  "causalAttention",
  "localAttention",
  "linearAttention",
  "groupedQueryAttention",
  // Further attention variants that carry the same embedDim(/dModel) % numHeads
  // contract (verified they expose both params in the registry).
  "crossModalAttention",
  "coAttention",
  "windowAttention",
  "differentialAttention",
  "nativeSparseAttention",
  "deformableAttention",
  "dividedSpaceTimeAttention",
  "gatedLinearAttention",
  // Block-level types carry the same embedDim/numHeads contract as their
  // inner attention; a non-divisible pair crashes identically at runtime.
  "transformerBlock",
  "conformerBlock",
  "ditBlock",
  "mla"
]);
function isFiniteDim(n2) {
  return typeof n2 === "number" && Number.isFinite(n2);
}
function shapeIsInvalid(s) {
  if (!s || s.length === 0) return false;
  return s.some((d) => !isFiniteDim(d) || d <= 0);
}
function shapesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
function topoSort(model) {
  const indeg = /* @__PURE__ */ new Map();
  const adj = /* @__PURE__ */ new Map();
  for (const c of model.components) {
    indeg.set(c.id, 0);
    adj.set(c.id, []);
  }
  for (const conn of model.connections) {
    if (!indeg.has(conn.from) || !indeg.has(conn.to)) continue;
    adj.get(conn.from).push(conn.to);
    indeg.set(conn.to, (indeg.get(conn.to) ?? 0) + 1);
  }
  const queue = [];
  for (const [id, d] of indeg) if (d === 0) queue.push(id);
  const order = [];
  while (queue.length) {
    const id = queue.shift();
    order.push(id);
    for (const nb of adj.get(id) ?? []) {
      const d = (indeg.get(nb) ?? 0) - 1;
      indeg.set(nb, d);
      if (d === 0) queue.push(nb);
    }
  }
  return order;
}
function checkAttentionDivisibility(c) {
  const p = c.params ?? {};
  const embed = p.embedDim ?? p.hiddenDim ?? p.dModel;
  const heads = p.numHeads;
  if (typeof embed !== "number" || typeof heads !== "number") return null;
  if (heads <= 0) return null;
  if (embed % heads !== 0) {
    return {
      componentId: c.id,
      componentName: c.name,
      componentType: c.type,
      kind: "head-dim-divisibility",
      message: `${c.type} '${c.name}': embed dim (${embed}) must be divisible by numHeads (${heads}); head_dim would be ${(embed / heads).toFixed(2)}`,
      expected: 0,
      actual: embed % heads
    };
  }
  if (c.type === "groupedQueryAttention") {
    const kvHeads = p.numKVHeads;
    if (typeof kvHeads === "number" && kvHeads > 0 && heads % kvHeads !== 0) {
      return {
        componentId: c.id,
        componentName: c.name,
        componentType: c.type,
        kind: "gqa-head-divisibility",
        message: `groupedQueryAttention '${c.name}': numHeads (${heads}) must be divisible by numKVHeads (${kvHeads})`,
        expected: 0,
        actual: heads % kvHeads
      };
    }
  }
  return null;
}
function checkAttentionInDim(c, inShape) {
  const p = c.params ?? {};
  const embed = p.embedDim ?? p.hiddenDim ?? p.dModel;
  if (typeof embed !== "number" || inShape.length === 0) return null;
  const lastDim = inShape[inShape.length - 1];
  if (!isFiniteDim(lastDim) || lastDim === embed) return null;
  return {
    componentId: c.id,
    componentName: c.name,
    componentType: c.type,
    kind: "attention-in-mismatch",
    message: `${c.type} '${c.name}': embedDim=${embed} but upstream last dim is ${lastDim}; attention expects them equal (project the input or fix embedDim)`,
    expected: embed,
    actual: lastDim
  };
}
function checkLinearInFeatures(c, inShape) {
  if (c.type !== "linear") return null;
  const inFeatures = (c.params ?? {}).inFeatures;
  if (typeof inFeatures !== "number" || !inShape || inShape.length === 0) return null;
  const lastDim = inShape[inShape.length - 1];
  if (lastDim !== inFeatures) {
    return {
      componentId: c.id,
      componentName: c.name,
      componentType: c.type,
      kind: "linear-in-mismatch",
      message: `linear '${c.name}': inFeatures=${inFeatures} but upstream last dim is ${lastDim}`,
      expected: inFeatures,
      actual: lastDim
    };
  }
  return null;
}
function checkMergeShapes(c, parentShapes) {
  if (parentShapes.length < 2) return null;
  if (ELEMENTWISE_MERGE.has(c.type)) {
    const first = parentShapes[0];
    for (let i = 1; i < parentShapes.length; i++) {
      if (!shapesEqual(first, parentShapes[i])) {
        return {
          componentId: c.id,
          componentName: c.name,
          componentType: c.type,
          kind: "merge-shape-mismatch",
          message: `${c.type} '${c.name}': elementwise inputs must match, got [${first.join(",")}] vs [${parentShapes[i].join(",")}]`,
          expected: first,
          actual: parentShapes[i]
        };
      }
    }
    return null;
  }
  if (CONCAT_MERGE.has(c.type)) {
    const p = c.params ?? {};
    const rank = parentShapes[0].length;
    const dimRaw = p.dim ?? -1;
    const dim = dimRaw < 0 ? rank + dimRaw : dimRaw;
    for (let i = 1; i < parentShapes.length; i++) {
      const a = parentShapes[0];
      const b = parentShapes[i];
      if (a.length !== b.length) {
        return {
          componentId: c.id,
          componentName: c.name,
          componentType: c.type,
          kind: "merge-shape-mismatch",
          message: `concat '${c.name}': inputs must have same rank, got rank ${a.length} vs ${b.length}`
        };
      }
      for (let k = 0; k < a.length; k++) {
        if (k === dim) continue;
        if (a[k] !== b[k]) {
          return {
            componentId: c.id,
            componentName: c.name,
            componentType: c.type,
            kind: "merge-shape-mismatch",
            message: `concat '${c.name}' on dim ${dim}: non-concat dim ${k} differs, ${a[k]} vs ${b[k]}`,
            expected: a,
            actual: b
          };
        }
      }
    }
    return null;
  }
  return null;
}
function checkMatmulShapes(c, parentShapes) {
  if (!MATMUL_LAYERS.has(c.type)) return null;
  if (parentShapes.length < 2) return null;
  const a = parentShapes[0];
  const b = parentShapes[1];
  if (a.length < 2 || b.length < 2) return null;
  const inner = a[a.length - 1];
  const other = b[b.length - 2];
  if (inner !== other) {
    return {
      componentId: c.id,
      componentName: c.name,
      componentType: c.type,
      kind: "matmul-shape-mismatch",
      message: `matmul '${c.name}': inner dims must match: first operand's last dim ${inner} vs second operand's dim ${other} ([${a.join(",")}] @ [${b.join(",")}])`,
      expected: a,
      actual: b
    };
  }
  return null;
}
function checkEmbedMatchShapes(c, parentShapes) {
  if (!EMBED_MATCH_LAYERS.has(c.type)) return null;
  if (parentShapes.length < 2) return null;
  const widths = parentShapes.map((s) => s[s.length - 1]);
  const first = widths[0];
  const mismatch = widths.find((w) => w !== first);
  if (mismatch !== void 0) {
    return {
      componentId: c.id,
      componentName: c.name,
      componentType: c.type,
      kind: "cross-attention-kv-mismatch",
      message: `${c.type} '${c.name}': query and key/value streams must share feature width, got ${widths.join(" vs ")}`,
      expected: parentShapes[0],
      actual: parentShapes[1]
    };
  }
  return null;
}
function propagateShapes(model) {
  const order = topoSort(model);
  const shapes = /* @__PURE__ */ new Map();
  const issues = [];
  const compById = new Map(model.components.map((c) => [c.id, c]));
  const parents = /* @__PURE__ */ new Map();
  for (const c of model.components) parents.set(c.id, []);
  for (const conn of model.connections) parents.get(conn.to)?.push(conn.from);
  for (const id of order) {
    const c = compById.get(id);
    if (!c) continue;
    const parentIds = parents.get(id) ?? [];
    const parentOuts = parentIds.map((p) => shapes.get(p)?.out ?? null).filter((s) => s !== null);
    if (HEAD_DIM_LAYERS.has(c.type)) {
      const att = checkAttentionDivisibility(c);
      if (att) issues.push(att);
    }
    const merge = checkMergeShapes(c, parentOuts);
    if (merge) issues.push(merge);
    const mm = checkMatmulShapes(c, parentOuts);
    if (mm) issues.push(mm);
    const emb = checkEmbedMatchShapes(c, parentOuts);
    if (emb) issues.push(emb);
    let inShape = null;
    if (c.type === "input") {
      const s = (c.params ?? {}).shape;
      if (Array.isArray(s) && s.length > 0 && s.every(isFiniteDim)) {
        inShape = s;
      } else {
        issues.push({
          componentId: c.id,
          componentName: c.name,
          componentType: c.type,
          kind: "no-input-shape",
          message: `input '${c.name}': params.shape is not set; downstream shapes cannot be inferred`
        });
      }
    } else if (parentOuts.length >= 1) {
      inShape = parentOuts[0];
    }
    if (c.type === "linear" && inShape) {
      const lin = checkLinearInFeatures(c, inShape);
      if (lin) issues.push(lin);
    }
    if (HEAD_DIM_LAYERS.has(c.type) && inShape) {
      const dim = checkAttentionInDim(c, inShape);
      if (dim) issues.push(dim);
    }
    let outShape = null;
    if (inShape) {
      try {
        const def = componentRegistry[c.type];
        if (def) {
          outShape = def.computeOutputShape(
            inShape,
            c.params ?? {},
            parentOuts.length > 1 ? parentOuts : void 0
          );
        } else {
          const alias = normalizeComponentType(c.type);
          const resolvable = alias !== c.type && alias in componentRegistry;
          issues.push({
            componentId: c.id,
            componentName: c.name,
            componentType: c.type,
            kind: "unknown-layer-type",
            message: resolvable ? `'${c.name}': "${c.type}" is not a known layer type. Did you mean "${alias}"? As submitted, nothing downstream of it can be verified and the trainer would drop it.` : `'${c.name}': "${c.type}" is not a known layer type; nothing downstream of it can be verified and the trainer would drop it`
          });
        }
      } catch (err) {
        issues.push({
          componentId: c.id,
          componentName: c.name,
          componentType: c.type,
          kind: "compute-error",
          message: `${c.type} '${c.name}': computeOutputShape threw ${err instanceof Error ? err.message : String(err)}`
        });
      }
    } else if (c.type === "input") {
      const s = (c.params ?? {}).shape;
      if (Array.isArray(s) && s.length > 0 && s.every(isFiniteDim)) {
        outShape = s;
      }
    }
    if (outShape && shapeIsInvalid(outShape) && inShape && !shapeIsInvalid(inShape)) {
      issues.push({
        componentId: c.id,
        componentName: c.name,
        componentType: c.type,
        kind: "invalid-output-shape",
        message: `${c.type} '${c.name}': computed output shape [${outShape.join(",")}] contains a non-positive or non-finite dim`,
        actual: outShape
      });
    }
    shapes.set(id, { in: inShape, out: outShape });
  }
  return { shapes, issues };
}
var FLOW_DEPENDENT_SHAPE_KINDS = /* @__PURE__ */ new Set([
  "invalid-output-shape",
  "merge-shape-mismatch",
  "attention-in-mismatch",
  "compute-error",
  "linear-in-mismatch"
]);
function downstreamOfInferredEdges(model) {
  const roots = model.connections.filter((c) => c.inferred).map((c) => c.to);
  if (roots.length === 0) return /* @__PURE__ */ new Set();
  const next = /* @__PURE__ */ new Map();
  for (const c of model.connections) next.set(c.from, [...next.get(c.from) ?? [], c.to]);
  const seen = new Set(roots);
  const queue = [...roots];
  while (queue.length) {
    for (const to of next.get(queue.shift()) ?? []) {
      if (!seen.has(to)) {
        seen.add(to);
        queue.push(to);
      }
    }
  }
  const names = /* @__PURE__ */ new Set();
  for (const c of model.components) if (seen.has(c.id) && c.name) names.add(c.name);
  return names;
}

// src/utils/paramEstimator.ts
function num2(v, fallback = 0) {
  const n2 = typeof v === "number" ? v : Number(v);
  return isFinite(n2) && n2 >= 0 ? n2 : fallback;
}
function gqaParams(p) {
  const D = num2(p.embedDim, 4096);
  const H = num2(p.numHeads, 32);
  const Hkv = num2(p.numKVHeads, H);
  const headDim = num2(p.headDim, H > 0 ? Math.floor(D / H) : 128);
  const qDim = H * headDim;
  const kvDim = Hkv * headDim;
  return D > 0 && headDim > 0 ? D * qDim + qDim * D + 2 * (D * kvDim) : 0;
}
function estimateLayerParams(type, params, inputShape) {
  const p = params ?? {};
  if (p.tied === true) return 0;
  const inp = inputShape ?? [];
  const lastDim = inp.length > 0 ? inp[inp.length - 1] : 0;
  const ch = inp.length >= 2 ? inp[0] : 1;
  switch (type) {
    // ── Basic ────────────────────────────────────────────────────────────────
    case "linear": {
      const inF = num2(p.inFeatures ?? lastDim);
      const outF = num2(p.outFeatures);
      return inF > 0 && outF > 0 ? inF * outF + outF : 0;
    }
    case "flatten":
      return 0;
    // ── Convolution ──────────────────────────────────────────────────────────
    case "conv2d": {
      const inC = num2(p.inChannels ?? ch, 1);
      const outC = num2(p.outChannels);
      const k = num2(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k + outC : 0;
    }
    case "conv1d": {
      const inC = num2(p.inChannels ?? ch, 1);
      const outC = num2(p.outChannels);
      const k = num2(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k + outC : 0;
    }
    case "conv3d": {
      const inC = num2(p.inChannels ?? ch, 1);
      const outC = num2(p.outChannels);
      const k = num2(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k * k + outC : 0;
    }
    case "depthwiseConv2d": {
      const inC = num2(p.inChannels ?? ch, 1);
      const k = num2(p.kernelSize, 3);
      const dm = num2(p.depthMultiplier, 1);
      return inC * dm * k * k + inC * dm;
    }
    case "separableConv2d": {
      const inC = num2(p.inChannels ?? ch, 1);
      const outC = num2(p.outChannels, inC);
      const k = num2(p.kernelSize, 3);
      return inC * k * k + inC * outC + outC;
    }
    case "transposeConv2d": {
      const inC = num2(p.inChannels ?? ch, 1);
      const outC = num2(p.outChannels);
      const k = num2(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k + outC : 0;
    }
    case "deformableConv2d":
    case "dilatedConv2d": {
      const inC = num2(p.inChannels ?? ch, 1);
      const outC = num2(p.outChannels);
      const k = num2(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k + outC : 0;
    }
    case "invResidualBlock": {
      const inC = num2(p.inChannels ?? ch, 32);
      const outC = num2(p.outChannels, inC);
      const e = num2(p.expandRatio, 6);
      const k = num2(p.kernelSize, 3);
      const mid = inC * e;
      return mid > 0 ? inC * mid + mid + mid * k * k + mid + mid * outC + outC : 0;
    }
    // ── Pooling (no learnable params) ────────────────────────────────────────
    case "maxpool2d":
    case "avgpool2d":
    case "adaptiveAvgPool2d":
    case "globalAvgPool2d":
    case "upsample":
      return 0;
    // ── NLP ─────────────────────────────────────────────────────────────────
    case "embedding":
    case "embeddingBag": {
      const V = num2(p.vocabSize ?? p.numEmbeddings);
      const D = num2(p.embeddingDim ?? p.embedDim);
      return V > 0 && D > 0 ? V * D : 0;
    }
    case "learnedPositionalEmbedding": {
      const L = num2(p.maxLen ?? p.maxPositions ?? p.numPositions);
      const D = num2(p.embedDim ?? p.embeddingDim);
      return L > 0 && D > 0 ? L * D : 0;
    }
    case "segmentEmbedding": {
      const S = num2(p.numSegments ?? p.numTypes, 2);
      const D = num2(p.embeddingDim ?? p.embedDim);
      return S > 0 && D > 0 ? S * D : 0;
    }
    case "lstm": {
      const H = num2(p.hiddenSize, 128);
      const I = num2(p.inputSize ?? lastDim, H);
      const L = num2(p.numLayers, 1);
      const dirs = p.bidirectional === true ? 2 : 1;
      const layer0 = dirs * 4 * (I * H + H * H + 2 * H);
      const layerRest = L > 1 ? (L - 1) * dirs * 4 * (dirs * H * H + H * H + 2 * H) : 0;
      return layer0 + layerRest;
    }
    case "gru": {
      const H = num2(p.hiddenSize, 128);
      const I = num2(p.inputSize ?? lastDim, H);
      const L = num2(p.numLayers, 1);
      const dirs = p.bidirectional === true ? 2 : 1;
      const layer0 = dirs * 3 * (I * H + H * H + 2 * H);
      const layerRest = L > 1 ? (L - 1) * dirs * 3 * (dirs * H * H + H * H + 2 * H) : 0;
      return layer0 + layerRest;
    }
    case "rnn": {
      const H = num2(p.hiddenSize, 128);
      const I = num2(p.inputSize ?? lastDim, H);
      return I * H + H * H + 2 * H;
    }
    case "bidirectionalLSTM": {
      const H = num2(p.hiddenSize, 128);
      const I = num2(p.inputSize ?? lastDim, H);
      const L = num2(p.numLayers, 1);
      const layer0 = 2 * 4 * (I * H + H * H + 2 * H);
      const layerRest = L > 1 ? (L - 1) * 2 * 4 * (2 * H * H + H * H + 2 * H) : 0;
      return layer0 + layerRest;
    }
    case "attention":
    case "selfAttention":
    case "multiHeadAttention": {
      const d = num2(p.hiddenDim ?? p.embedDim);
      return d > 0 ? 4 * d * d + 4 * d : 0;
    }
    case "crossModalAttention":
    case "crossAttention":
    case "coAttention":
    case "linearAttention":
    case "localAttention": {
      if (type === "localAttention" && (p.numKVHeads != null || p.headDim != null)) {
        return gqaParams(p);
      }
      const d = num2(p.embedDim ?? p.hiddenDim);
      return d > 0 ? 4 * d * d + 4 * d : 0;
    }
    // ── LLM ─────────────────────────────────────────────────────────────────
    case "feedForward": {
      const d = num2(p.hiddenDim ?? p.embedDim);
      const ff = num2(p.ffDim, d > 0 ? d * 4 : 0);
      const b = p.bias === false ? 0 : ff + d;
      return d > 0 && ff > 0 ? d * ff + ff * d + b : 0;
    }
    case "transformerBlock": {
      const d = num2(p.embedDim ?? p.hiddenDim ?? p.dModel);
      const ff = num2(p.ffDim ?? p.dFf, d > 0 ? d * 4 : 0);
      if (d <= 0) return 0;
      return 4 * d * d + 4 * d + d * ff + ff + ff * d + d + 4 * d;
    }
    case "positionalEncoding":
    case "rope":
      return 0;
    // learned or fixed, no gradient params
    // ── Normalization ────────────────────────────────────────────────────────
    case "layerNorm":
    case "batchNorm":
    case "instanceNorm": {
      const feat = p.normalizedShape ?? p.numFeatures ?? lastDim;
      const f = num2(Array.isArray(feat) ? feat[0] : feat);
      return f > 0 ? 2 * f : 0;
    }
    case "groupNorm": {
      const c = num2(p.numChannels ?? ch);
      return c > 0 ? 2 * c : 0;
    }
    // ── RL ───────────────────────────────────────────────────────────────────
    case "policyNetwork":
    case "valueNetwork": {
      const H = num2(p.hiddenSize, 256);
      const inF = num2(lastDim, H);
      return inF * H + H + H * H + H;
    }
    case "dqnHead": {
      const H = num2(p.hiddenSize ?? lastDim, 512);
      const A = num2(p.numActions, 18);
      return H * A + A;
    }
    case "actorHead": {
      const H = num2(lastDim, 256);
      const A = num2(p.numActions, 6);
      return H * A + A;
    }
    case "criticHead": {
      const H = num2(lastDim, 256);
      const O = num2(p.outputDim, 1);
      return H * O + O;
    }
    // ── Graph ────────────────────────────────────────────────────────────────
    case "graphConv":
    case "gcn": {
      const inC = num2(p.inChannels ?? p.inFeatures);
      const outC = num2(p.outChannels ?? p.outFeatures);
      return inC > 0 && outC > 0 ? inC * outC + outC : 0;
    }
    case "graphAttention":
    case "gat": {
      const inC = num2(p.inChannels ?? p.inFeatures ?? lastDim);
      const outC = num2(p.outChannels ?? p.outFeatures);
      const heads = num2(p.numHeads ?? p.heads, 1);
      return inC > 0 && outC > 0 ? heads * (inC * outC + 2 * outC) : 0;
    }
    case "graphSAGE": {
      const inC = num2(p.inChannels ?? p.inFeatures);
      const outC = num2(p.outChannels ?? p.outFeatures);
      return inC > 0 && outC > 0 ? inC * 2 * outC + outC : 0;
    }
    // ── Multimodal ───────────────────────────────────────────────────────────
    case "fusion": {
      const inD = num2(lastDim);
      const d = num2(p.fusionDim, 256);
      return d > 0 && inD > 0 ? inD * d + d : 0;
    }
    case "projection": {
      const inD = num2(p.inDim ?? lastDim);
      const outD = num2(p.outDim);
      return inD > 0 && outD > 0 ? inD * outD + outD : 0;
    }
    // ── Tabular ──────────────────────────────────────────────────────────────
    case "tabnet": {
      const fd = num2(p.featureDim, 64);
      const dd = num2(p.decisionDim, 64);
      const inF = num2(p.inputDim ?? lastDim, fd);
      const w = fd + dd;
      return w > 0 ? inF * w + w + 4 * w * w : 0;
    }
    case "featureInteraction":
      return 0;
    // ── Audio ────────────────────────────────────────────────────────────────
    case "audioConv": {
      const inC = num2(p.inChannels ?? 1, 1);
      const outC = num2(p.outChannels);
      const k = num2(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k + outC : 0;
    }
    case "melSpectrogram":
    case "mfcc":
    case "stft":
      return 0;
    // ── New attention types ──────────────────────────────────────────────────
    case "windowAttention": {
      const D = num2(p.embedDim, 96);
      return D > 0 ? 4 * D * D : 0;
    }
    case "groupedQueryAttention":
      return gqaParams(p);
    case "causalAttention": {
      const D = num2(p.embedDim, 512);
      return D > 0 ? 4 * D * D : 0;
    }
    case "adaptiveMaxPool2d":
      return 0;
    // no learnable params
    // ── New activations ──────────────────────────────────────────────────────
    case "prelu": {
      return num2(p.numParameters, 1);
    }
    case "rmsNorm": {
      const ns = num2(p.normalizedShape ?? lastDim, 512);
      return ns;
    }
    // ── Transformer extras ───────────────────────────────────────────────────
    case "swiglu": {
      const D = num2(p.embedDim ?? p.inFeatures, 4096);
      const I = num2(p.intermediateSize ?? p.hiddenFeatures ?? p.ffDim, Math.round(D * 8 / 3));
      return D > 0 && I > 0 ? 3 * D * I : 0;
    }
    case "moeLayer": {
      const D = num2(p.embedDim, 512);
      const E = num2(p.numExperts, 8);
      const I = num2(p.expertDim ?? p.ffDim, Math.round(D * 8 / 3));
      const m = p.gatedExperts === false ? 2 : 3;
      return D > 0 && I > 0 ? D * E + E * m * D * I : 0;
    }
    case "patchEmbed": {
      const inC = num2(p.inChans ?? p.inChannels, 3);
      const D = num2(p.embedDim, 768);
      const P = num2(p.patchSize, 16);
      const R = Math.max(0, Math.floor(num2(p.registerTokens, 0)));
      return inC * D * P * P + D + (p.clsToken ? D : 0) + R * D;
    }
    case "seBlock": {
      const C = num2(p.channels, 64);
      const r = num2(p.reductionRatio ?? p.reduction, 16);
      const mid = Math.max(1, Math.floor(C / r));
      return C * mid + mid + mid * C + C;
    }
    case "layerScale":
      return num2(p.dim, 512);
    case "alibi":
    case "dropPath":
      return 0;
    // ── Frontier architectures (2024-2025) ────────────────────────────────────
    case "mla": {
      const D = num2(p.embedDim, 512);
      const kv = num2(p.kvLatentDim, 128);
      const H = num2(p.numHeads);
      const nope = num2(p.qkNopeHeadDim);
      const vd = num2(p.vHeadDim);
      if (D > 0 && H > 0 && nope > 0 && vd > 0) {
        const rope = num2(p.ropeHeadDim);
        const qk = nope + rope;
        const qLora = num2(p.qLoraRank);
        const q = qLora > 0 ? D * qLora + qLora + qLora * H * qk : D * H * qk;
        return q + D * (kv + rope) + kv + kv * H * (nope + vd) + H * vd * D;
      }
      return D > 0 ? 2 * D * D + 2 * D * kv : 0;
    }
    case "mamba2": {
      const D = num2(p.dModel, 512);
      const E = num2(p.expand, 2);
      const S = num2(p.dState, 128);
      const H = num2(p.numHeads);
      if (D > 0 && H > 0) {
        const di = H * num2(p.headDim, 64);
        const G = num2(p.nGroups, 1);
        const K = num2(p.dConv, 4);
        const convDim2 = di + 2 * G * S;
        const inOut = 2 * di + 2 * G * S + H;
        const pb = p.projBias === true;
        return D * inOut + (pb ? inOut : 0) + convDim2 * K + (p.convBias === false ? 0 : convDim2) + 3 * H + di + di * D + (pb ? D : 0);
      }
      const inner = D * E;
      return D > 0 ? 2 * D * inner + inner * D + inner * S : 0;
    }
    case "differentialAttention":
    case "retention": {
      const D = num2(p.embedDim ?? p.dModel, 512);
      return D > 0 ? 4 * D * D : 0;
    }
    case "rgLru": {
      const D = num2(p.dModel, 512);
      const E = num2(p.expand, 1);
      const inner = D * E;
      return D > 0 ? 2 * D * inner + 2 * inner : 0;
    }
    case "hyena": {
      const D = num2(p.dModel, 512);
      const fo = num2(p.filterOrder, 64);
      const order = num2(p.order, 2);
      return D > 0 ? 2 * D * D + order * (D * fo + fo) : 0;
    }
    case "rwkv":
    case "xlstm": {
      const D = num2(p.dModel, 512);
      return D > 0 ? 4 * D * D : 0;
    }
    case "mixtureOfDepths": {
      const D = num2(p.dModel, 512);
      return D > 0 ? D : 0;
    }
    case "tttLayer":
    case "titansMemory": {
      const D = num2(p.dModel, 512);
      const depth = num2(p.memoryDepth, 2);
      return D > 0 ? depth * D * D : 0;
    }
    case "multiTokenPrediction": {
      const D = num2(p.dModel, 512);
      const V = num2(p.vocabSize, 32e3);
      const k = num2(p.numFutureTokens, 2);
      return D > 0 && V > 0 ? k * D * V : 0;
    }
    case "kan": {
      const inF = num2(p.inFeatures, 128);
      const outF = num2(p.outFeatures, 128);
      const grid = num2(p.gridSize, 5);
      const order = num2(p.splineOrder, 3);
      return inF > 0 && outF > 0 ? inF * outF * (grid + order + 1) : 0;
    }
    case "geglu": {
      const D = num2(p.dim, 512);
      const I = num2(p.hiddenDim, D > 0 ? D * 4 : 0);
      return D > 0 && I > 0 ? 3 * D * I : 0;
    }
    case "grn": {
      const C = num2(p.channels, 256);
      return C > 0 ? 2 * C : 0;
    }
    case "qkNorm": {
      const d = num2(p.dim, 64);
      return d > 0 ? 2 * d : 0;
    }
    // ── Frontier architectures (new batch) ────────────────────────────────────
    case "deltaNet":
    case "gatedDeltaNet":
    case "gatedLinearAttention": {
      const d = num2(p.dModel, 512);
      const Hk = num2(p.numKeyHeads);
      if (type === "gatedDeltaNet" && d > 0 && Hk > 0) {
        const Hv = num2(p.numHeads, Hk);
        const dv = num2(p.headDim, 128);
        const dk = num2(p.keyHeadDim, dv);
        const K = num2(p.convKernel, 4);
        const keyDim = Hk * dk;
        const valDim = Hv * dv;
        return d * (2 * keyDim + 2 * valDim) + d * 2 * Hv + (2 * keyDim + valDim) * K + 2 * Hv + dv + valDim * d;
      }
      return d > 0 ? 4 * d * d : 0;
    }
    case "nativeSparseAttention": {
      const d = num2(p.embedDim, 512);
      return d > 0 ? 4 * d * d : 0;
    }
    case "sharedExpertMoE": {
      const d = num2(p.embedDim, 4096);
      const e = num2(p.numExperts, 64);
      const s = num2(p.numSharedExperts, 2);
      const I = num2(p.expertDim, 1408);
      const Is = num2(p.sharedExpertDim, I);
      const m = p.gatedExperts === false ? 2 : 3;
      const lat = num2(p.latentDim);
      if (d > 0 && I > 0 && lat > 0) return d * e + 2 * d * lat + e * m * lat * I + s * m * d * Is;
      return d > 0 && I > 0 ? d * e + e * m * d * I + s * m * d * Is : 0;
    }
    case "ditBlock": {
      const d = num2(p.hiddenDim, 1152);
      const cond = num2(p.condDim, 1152);
      return d > 0 ? 4 * d * d + 8 * d * d + 6 * cond * d : 0;
    }
    case "vectorQuantizer": {
      const cb = num2(p.codebookSize, 8192);
      const d = num2(p.embedDim, 256);
      return cb > 0 && d > 0 ? cb * d : 0;
    }
    case "residualVQ": {
      const q = num2(p.numQuantizers, 8);
      const cb = num2(p.codebookSize, 1024);
      const d = num2(p.embedDim, 256);
      return q > 0 && cb > 0 && d > 0 ? q * cb * d : 0;
    }
    case "perceiverLatent": {
      const nL = num2(p.numLatents, 64);
      const lD = num2(p.latentDim, 768);
      return lD > 0 ? nL * lD + 4 * lD * lD : 0;
    }
    case "convNeXtBlock": {
      const dim = num2(p.dim, 96);
      const k = num2(p.kernelSize, 7);
      const ex = num2(p.expandRatio, 4);
      return dim > 0 ? dim * k * k + dim + 2 * dim * (dim * ex) : 0;
    }
    case "s4Layer": {
      const d = num2(p.dModel, 512);
      const s = num2(p.dState, 64);
      return d > 0 ? d * s * 2 + d * 2 : 0;
    }
    case "dyt": {
      const dim = num2(p.dim, 512);
      return dim > 0 ? 2 * dim + 1 : 0;
    }
    case "film": {
      const f = num2(p.numFeatures, 256);
      return f > 0 ? 2 * f : 0;
    }
    case "crossNetworkDCN": {
      const nL = num2(p.numLayers, 3);
      const d = num2(p.inputDim, 256);
      return d > 0 ? nL * (d * d + d) : 0;
    }
    case "ftTransformerBlock": {
      const d = num2(p.dModel, 192);
      const ff = num2(p.ffMult, 4);
      return d > 0 ? 4 * d * d + 2 * d * (d * ff) : 0;
    }
    case "deformableAttention": {
      const d = num2(p.embedDim, 256);
      const h = num2(p.numHeads, 8);
      const pts = num2(p.numPoints, 4);
      const lv = num2(p.numLevels, 4);
      return d > 0 ? 2 * d * d + d * (h * lv * pts * 2) + d * (h * lv * pts) : 0;
    }
    case "attentionPool": {
      const dim = num2(p.dim, 512);
      const seeds = num2(p.numSeeds, 1);
      return dim > 0 ? seeds * dim + 4 * dim * dim : 0;
    }
    // ── Recommendation: sequential / generative ──────────────────────────────
    case "targetAttention": {
      const d = num2(p.embedDim ?? lastDim, 64);
      const h = num2(p.hiddenDim, 36);
      return d > 0 ? 4 * d * h + h + h + 1 : 0;
    }
    case "behaviorRetrieval": {
      if (String(p.mode ?? "soft") === "hard") return 0;
      const d = num2(p.embedDim ?? lastDim, 64);
      return d > 0 ? d * d : 0;
    }
    case "multiInterest": {
      const d = num2(p.embedDim ?? lastDim, 64);
      return d > 0 ? d * d : 0;
    }
    case "hstuBlock": {
      const d = num2(p.embedDim, 512);
      const h = num2(p.numHeads, 4);
      const dv = num2(p.linearDim, 128);
      const dqk = num2(p.attnDim, 128);
      if (d <= 0) return 0;
      const inProj = d * h * (2 * dv + 2 * dqk);
      const outProj = h * dv * d;
      return inProj + outProj + 2 * d;
    }
    // ── Video latents / world models ─────────────────────────────────────────
    case "causalConv3d": {
      const inC = num2(p.inChannels ?? ch, 1);
      const outC = num2(p.outChannels);
      const k = num2(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k * k + outC : 0;
    }
    case "rssm": {
      const det = num2(p.deterDim, 512);
      const st = num2(p.stochDim, 32);
      const cls = Math.max(1, num2(p.stochClasses, 1));
      const hid = num2(p.hiddenDim, 512);
      const stochFlat = st * cls;
      const embed = num2(lastDim, 0);
      if (det <= 0) return 0;
      const gru = 3 * (det + stochFlat) * det + 3 * det;
      const prior = det * hid + hid + hid * stochFlat + stochFlat;
      const post = (det + embed) * hid + hid + hid * stochFlat + stochFlat;
      return gru + prior + post;
    }
    case "latentActionModel": {
      const inD = num2(lastDim, 512);
      const hid = num2(p.hiddenDim, 512);
      const lat = num2(p.latentDim, 32);
      const code = num2(p.codebookSize, 8);
      return 2 * inD * hid + hid + hid * lat + lat + code * lat;
    }
    case "jepaPredictor": {
      const d = num2(p.embedDim, 768);
      const pd = num2(p.predictorDim, 384);
      const depth = num2(p.depth, 6);
      if (d <= 0 || pd <= 0) return 0;
      const blocks = depth * (4 * pd * pd + 2 * pd * (4 * pd));
      return d * pd + pd + pd * d + d + blocks;
    }
    case "emaTarget":
      return 0;
    // ── Multimodal connectors / retrieval objectives ─────────────────────────
    case "patchMerger": {
      const k = Math.max(1, num2(p.mergeSize, 2));
      const inD = num2(p.inDim ?? lastDim, 1280);
      const outD = num2(p.outDim, 3584);
      const merged = inD * k * k;
      if (merged <= 0) return 0;
      return 2 * merged + merged * merged + merged + merged * outD + outD;
    }
    case "contrastiveHead": {
      const inD = num2(lastDim, 0);
      const outD = num2(p.projDim, 512);
      if (inD <= 0 || outD <= 0) return 0;
      return inD * outD + (p.learnableTemp === false ? 0 : 1);
    }
    case "matryoshkaHead":
      return 0;
    case "lateInteraction":
      return 0;
    case "revIN": {
      if (p.affine === false) return 0;
      const f = num2(p.numFeatures, 7);
      return f > 0 ? 2 * f : 0;
    }
    case "seriesDecomp":
      return 0;
    case "setAbstraction": {
      const mlp = Array.isArray(p.mlp) ? p.mlp.map((x) => num2(x, 0)) : [64, 64, 128];
      let inC = 3 + 3;
      let total = 0;
      for (const outC of mlp) {
        total += inC * outC + outC;
        inC = outC;
      }
      return total;
    }
    case "sparseConv3d": {
      const inC = num2(p.inChannels, 32);
      const outC = num2(p.outChannels, 64);
      const k = num2(p.kernelSize, 3);
      return inC > 0 && outC > 0 ? inC * outC * k * k * k + outC : 0;
    }
    case "nerfPositionalEncoding":
      return 0;
    case "dividedSpaceTimeAttention": {
      const d = num2(p.embedDim, 768);
      return d > 0 ? 2 * (4 * d * d) : 0;
    }
    case "tubeletEmbed": {
      const D = num2(p.embedDim, 768);
      const t = Array.isArray(p.tubeletSize) ? p.tubeletSize.map((x) => num2(x, 1)) : [2, 16, 16];
      const inC = num2(p.inChans ?? p.inChannels, 3);
      const kvol = (t[0] || 1) * (t[1] || 1) * (t[2] || 1);
      return D > 0 ? inC * D * kvol + D : 0;
    }
    // ── Non-LLM coverage (audio / SSM / vision / diffusion) ──────────────────
    case "conformerBlock": {
      const d = num2(p.dModel ?? p.embedDim ?? p.hiddenDim, 256);
      const k = num2(p.kernelSize, 31);
      if (d <= 0) return 0;
      const ff = p.ffDim != null ? num2(p.ffDim, d * 4) : d * num2(p.ffMult, 4);
      const ffn = 2 * (2 * d * ff);
      const mha = 4 * d * d;
      const conv = 2 * d * d + d * k + d * d;
      return ffn + mha + conv + 6 * d;
    }
    case "mamba": {
      const d = num2(p.dModel ?? p.embedDim ?? p.hiddenDim, 256);
      const e = num2(p.expand, 2);
      const dState = num2(p.dState, 16);
      const dConv = num2(p.dConv, 4);
      if (d <= 0) return 0;
      const di = d * e;
      const dtRank = num2(p.dtRank);
      if (dtRank > 0) {
        const pb = p.projBias === true;
        return d * 2 * di + (pb ? 2 * di : 0) + di * dConv + (p.convBias === false ? 0 : di) + di * (dtRank + 2 * dState) + dtRank * di + di + di * dState + di + di * d + (pb ? d : 0);
      }
      return d * (2 * di) + di * dConv + di * (2 * dState) + di * dState + di * d;
    }
    case "relativePositionBias": {
      const h = num2(p.numHeads, 8);
      const b = num2(p.numBuckets, 32);
      return h * b;
    }
    case "fpn": {
      const outC = num2(p.outChannels, 256);
      const ins = Array.isArray(p.inChannels) ? p.inChannels.map((x) => num2(x)) : [num2(p.inChannels, outC)];
      if (outC <= 0) return 0;
      let total = 0;
      for (const inC of ins) total += inC * outC + outC;
      total += ins.length * (outC * outC * 9 + outC);
      return total;
    }
    case "timeEmbedding": {
      const d = num2(p.dim ?? p.embedDim ?? p.hiddenDim, 256);
      return d > 0 ? 2 * d * d + 2 * d : 0;
    }
    case "lmHead": {
      const V = num2(p.vocabSize ?? p.numEmbeddings);
      const d = num2(p.inFeatures ?? p.hiddenSize ?? p.embedDim ?? lastDim);
      const bias = p.bias === true ? V : 0;
      return V > 0 && d > 0 ? d * V + bias : 0;
    }
    case "adaIN": {
      const f = num2(p.numFeatures ?? p.embedDim ?? lastDim);
      return f > 0 ? 2 * f : 0;
    }
    // ── No learnable params ──────────────────────────────────────────────────
    default:
      return 0;
  }
}

// src/utils/flopsEstimator.ts
function n(v, fallback = 0) {
  const x = typeof v === "number" ? v : Number(v);
  return isFinite(x) && x > 0 ? x : fallback;
}
function spatialElements(shape) {
  if (shape.length < 3) return 1;
  return shape.slice(2).reduce((a, b) => a * b, 1);
}
function estimateLayerFlops(type, params, inputShape, outputShape) {
  const p = params;
  {
    const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
    const exact = type === "mla" && n(p.qkNopeHeadDim) > 0 && n(p.vHeadDim) > 0 && n(p.numHeads) > 0 || type === "mamba" && n(p.dtRank) > 0 || type === "mamba2" && n(p.numHeads) > 0 || type === "gatedDeltaNet" && n(p.numKeyHeads) > 0 || type === "localAttention" && (p.numKVHeads != null || p.headDim != null);
    if (exact) {
      const proj = T * estimateLayerParams(type, p, inputShape);
      if (type === "mla") return proj + T * T * n(p.numHeads) * (n(p.qkNopeHeadDim) + n(p.ropeHeadDim));
      if (type === "localAttention") {
        const H = n(p.numHeads, 8);
        const hd = n(p.headDim, H > 0 ? Math.floor(n(p.embedDim, 512) / H) : 64);
        return proj + T * Math.min(T, n(p.windowSize, 256)) * H * hd;
      }
      return proj;
    }
  }
  switch (type) {
    // ── Linear / Dense ──────────────────────────────────────────────────────
    case "linear": {
      const inF = n(p.inFeatures ?? inputShape[inputShape.length - 1]);
      const outF = n(p.outFeatures ?? outputShape[outputShape.length - 1]);
      const batch = outputShape.slice(0, -1).reduce((a, b) => a * b, 1) || 1;
      return inF * outF * batch;
    }
    // ── Convolutions ─────────────────────────────────────────────────────────
    case "conv2d": {
      const Cin = n(p.inChannels ?? inputShape[0], 1);
      const Cout = n(p.outChannels ?? outputShape[0], 1);
      const kH = n(p.kernelSize, 3);
      const kW = typeof p.kernelSize === "object" ? n(p.kernelSize[1], kH) : kH;
      const groups = n(p.groups, 1);
      const Hout = outputShape[1] ?? 1;
      const Wout = outputShape[2] ?? 1;
      return Cin / groups * Cout * kH * kW * Hout * Wout;
    }
    case "conv1d": {
      const Cin = n(p.inChannels ?? inputShape[0], 1);
      const Cout = n(p.outChannels ?? outputShape[0], 1);
      const kL = n(p.kernelSize, 3);
      const Lout = outputShape[1] ?? 1;
      return Cin * Cout * kL * Lout;
    }
    case "conv3d": {
      const Cin = n(p.inChannels ?? inputShape[0], 1);
      const Cout = n(p.outChannels ?? outputShape[0], 1);
      const k = n(p.kernelSize, 3);
      const spat = spatialElements(outputShape);
      return Cin * Cout * k * k * k * spat;
    }
    case "depthwiseConv2d": {
      const Cin = n(p.inChannels ?? inputShape[0], 1);
      const dm = n(p.depthMultiplier, 1);
      const k = n(p.kernelSize, 3);
      const spat = spatialElements(outputShape);
      return Cin * dm * k * k * spat;
    }
    case "separableConv2d": {
      const Cin = n(p.inChannels ?? inputShape[0], 1);
      const Cout = n(p.outChannels ?? outputShape[0], Cin);
      const k = n(p.kernelSize, 3);
      const spat = spatialElements(outputShape);
      const depthwise = Cin * k * k * spat;
      const pointwise = Cin * Cout * spat;
      return depthwise + pointwise;
    }
    case "transposeConv2d": {
      const Cin = n(p.inChannels ?? inputShape[0], 1);
      const Cout = n(p.outChannels ?? outputShape[0], 1);
      const k = n(p.kernelSize, 3);
      const spat = spatialElements(outputShape);
      return Cin * Cout * k * k * spat;
    }
    // ── Attention ─────────────────────────────────────────────────────────────
    case "multiHeadAttention":
    case "attention":
    case "selfAttention":
    case "crossModalAttention":
    case "causalAttention": {
      const D = n(p.embedDim ?? p.hiddenDim ?? inputShape[inputShape.length - 1]);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      if (D === 0) return 0;
      const qkv = 3 * T * D * D;
      const attn = T * T * D;
      const out = T * D * D;
      return qkv + attn + out;
    }
    case "windowAttention": {
      const D = n(p.embedDim, 96);
      const W = n(p.windowSize, 7);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : W * W;
      const numWindows = Math.max(1, Math.round(T / (W * W)));
      const wsq = W * W;
      return numWindows * (3 * wsq * D * D + wsq * wsq * D + wsq * D * D);
    }
    case "groupedQueryAttention": {
      const D = n(p.embedDim, 4096);
      const H = n(p.numHeads, 32);
      const Hkv = n(p.numKVHeads, H);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      if (D === 0) return 0;
      const headDim = n(p.headDim, H > 0 ? Math.floor(D / H) : 128);
      const qDim = H * headDim;
      const kvDim = Hkv * headDim;
      const q_proj = T * D * qDim;
      const kv_proj = 2 * T * D * kvDim;
      const attn_w = T * T * qDim;
      const o_proj = T * qDim * D;
      return q_proj + kv_proj + attn_w + o_proj;
    }
    // ── Recurrent ─────────────────────────────────────────────────────────────
    case "lstm": {
      const H = n(p.hiddenSize, 128);
      const I = n(p.inputSize ?? inputShape[inputShape.length - 1], H);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      const L = n(p.numLayers, 1);
      const layer0 = 4 * T * (I * H + H * H);
      const layerRest = L > 1 ? (L - 1) * 4 * T * 2 * H * H : 0;
      return layer0 + layerRest;
    }
    case "gru": {
      const H = n(p.hiddenSize, 128);
      const I = n(p.inputSize ?? inputShape[inputShape.length - 1], H);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      const L = n(p.numLayers, 1);
      const layer0 = 3 * T * (I * H + H * H);
      const layerRest = L > 1 ? (L - 1) * 3 * T * 2 * H * H : 0;
      return layer0 + layerRest;
    }
    case "rnn": {
      const H = n(p.hiddenSize, 128);
      const I = n(p.inputSize ?? inputShape[inputShape.length - 1], H);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return T * (I * H + H * H);
    }
    case "bidirectionalLSTM": {
      const H = n(p.hiddenSize, 128);
      const I = n(p.inputSize ?? inputShape[inputShape.length - 1], H);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return 2 * 4 * T * (I * H + H * H);
    }
    // ── Transformer block ─────────────────────────────────────────────────────
    case "transformerBlock": {
      const D = n(p.embedDim ?? p.hiddenDim ?? p.dModel);
      const ff = n(p.ffDim ?? p.dFf, D > 0 ? D * 4 : 0);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      if (D === 0) return 0;
      const mha = 4 * T * D * D + T * T * D;
      const ffn = 2 * T * D * ff;
      return mha + ffn;
    }
    // ── Feed-forward (MLP block) ──────────────────────────────────────────────
    case "feedForward": {
      const D = n(p.embedDim ?? p.hiddenDim);
      const ff = n(p.ffDim, D > 0 ? D * 4 : 0);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 && ff > 0 ? 2 * T * D * ff : 0;
    }
    // ── Normalization (lightweight but non-zero) ──────────────────────────────
    case "batchNorm":
    case "layerNorm":
    case "rmsNorm":
    case "instanceNorm":
    case "groupNorm": {
      const elems = outputShape.reduce((a, b) => a * b, 1);
      return Math.round(elems * 2);
    }
    // ── Pooling ───────────────────────────────────────────────────────────────
    case "maxpool2d":
    case "avgpool2d": {
      const k = n(p.kernelSize, 2);
      const spat = spatialElements(outputShape);
      const C = outputShape[0] ?? 1;
      return C * k * k * spat;
    }
    case "adaptiveAvgPool2d":
    case "globalAvgPool2d": {
      const Cin = inputShape[0] ?? 1;
      const inSpat = spatialElements(inputShape);
      return Cin * inSpat;
    }
    // ── Activations (element-wise, ~1 MAC each) ───────────────────────────────
    case "relu":
    case "leakyRelu":
    case "sigmoid":
    case "tanh":
    case "softmax": {
      return outputShape.reduce((a, b) => a * b, 1);
    }
    case "gelu":
    case "swish":
    case "silu": {
      return outputShape.reduce((a, b) => a * b, 1) * 4;
    }
    // ── Embedding lookup (0 MACs — just a gather) ────────────────────────────
    case "embedding":
    case "embeddingBag":
    case "positionalEncoding":
    case "rope":
      return 0;
    // ── Transformer extras ────────────────────────────────────────────────────
    case "swiglu": {
      const D = n(p.embedDim ?? p.inFeatures, 4096);
      const I = n(p.intermediateSize ?? p.hiddenFeatures ?? p.ffDim, Math.round(D * 8 / 3));
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return T > 0 && D > 0 && I > 0 ? T * (2 * D * I + I + I * D) : 0;
    }
    case "moeLayer": {
      const D = n(p.embedDim, 512);
      const E = n(p.numExperts, 8);
      const K = n(p.topK, 2);
      const I = n(p.expertDim ?? p.ffDim, Math.round(D * 8 / 3));
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return T > 0 && D > 0 && I > 0 ? T * D * E + K * T * (2 * D * I + I + I * D) : 0;
    }
    case "patchEmbed": {
      const P = n(p.patchSize, 16);
      const D = n(p.embedDim, 768);
      const inC = n(p.inChans ?? p.inChannels, 3);
      const H = inputShape[1] ?? 224;
      const W = inputShape[2] ?? 224;
      const nPat = Math.floor(H / P) * Math.floor(W / P);
      return inC * D * P * P * nPat;
    }
    case "seBlock": {
      const C = n(p.channels, 64);
      const r = n(p.reductionRatio ?? p.reduction, 16);
      const mid = Math.max(1, Math.floor(C / r));
      const spat = spatialElements(inputShape);
      return C * spat + C * mid + mid * C;
    }
    case "alibi":
    case "dropPath":
    case "layerScale":
      return 0;
    // ── Frontier architectures (2024-2025) ────────────────────────────────────
    case "mla":
    case "differentialAttention":
    case "retention": {
      const D = n(p.embedDim ?? p.dModel, 512);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      if (D === 0) return 0;
      return 4 * T * D * D + T * T * D;
    }
    case "mamba2":
    case "rwkv":
    case "xlstm":
    case "rgLru":
    case "hyena":
    case "tttLayer":
    case "titansMemory": {
      const D = n(p.dModel ?? p.embedDim, 512);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 ? 4 * T * D * D : 0;
    }
    case "mixtureOfDepths": {
      const D = n(p.dModel, 512);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 ? T * D : 0;
    }
    case "multiTokenPrediction": {
      const D = n(p.dModel, 512);
      const V = n(p.vocabSize, 32e3);
      const k = n(p.numFutureTokens, 2);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 && V > 0 ? k * T * D * V : 0;
    }
    case "kan": {
      const inF = n(p.inFeatures, 128);
      const outF = n(p.outFeatures, 128);
      const grid = n(p.gridSize, 5);
      const order = n(p.splineOrder, 3);
      const batch = outputShape.slice(0, -1).reduce((a, b) => a * b, 1) || 1;
      return inF * outF * (grid + order + 1) * batch;
    }
    case "geglu": {
      const D = n(p.dim, 512);
      const I = n(p.hiddenDim, D > 0 ? D * 4 : 0);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 && I > 0 ? T * (2 * D * I + I + I * D) : 0;
    }
    case "grn":
    case "qkNorm": {
      const elems = outputShape.reduce((a, b) => a * b, 1);
      return Math.round(elems * 2);
    }
    // ── Frontier architectures (new batch) ────────────────────────────────────
    case "deltaNet":
    case "gatedDeltaNet":
    case "gatedLinearAttention":
    case "s4Layer": {
      const D = n(p.dModel ?? p.embedDim, 512);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 ? 4 * T * D * D : 0;
    }
    case "nativeSparseAttention": {
      const D = n(p.embedDim, 512);
      const blk = n(p.blockSize, 64);
      const top = n(p.topBlocks, 16);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 ? 4 * T * D * D + T * (top * blk) * D : 0;
    }
    case "ditBlock": {
      const D = n(p.hiddenDim, 1152);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      if (D === 0) return 0;
      return 4 * T * D * D + T * T * D + 2 * T * D * (D * 4);
    }
    case "sharedExpertMoE": {
      const D = n(p.embedDim, 4096);
      const s = n(p.numSharedExperts, 2);
      const K = n(p.topK, 6);
      const I = n(p.expertDim, 1408);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 && I > 0 ? T * D * n(p.numExperts, 64) + (K + s) * T * (2 * D * I + I + I * D) : 0;
    }
    case "vectorQuantizer": {
      const cb = n(p.codebookSize, 8192);
      const D = n(p.embedDim, 256);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return cb > 0 && D > 0 ? T * cb * D : 0;
    }
    case "residualVQ": {
      const q = n(p.numQuantizers, 8);
      const cb = n(p.codebookSize, 1024);
      const D = n(p.embedDim, 256);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return q > 0 && cb > 0 && D > 0 ? T * q * cb * D : 0;
    }
    case "perceiverLatent": {
      const nL = n(p.numLatents, 64);
      const lD = n(p.latentDim, 768);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return lD > 0 ? nL * T * lD + 4 * nL * lD * lD : 0;
    }
    case "convNeXtBlock": {
      const dim = n(p.dim, 96);
      const k = n(p.kernelSize, 7);
      const ex = n(p.expandRatio, 4);
      const spat = spatialElements(outputShape);
      return dim > 0 ? dim * k * k * spat + 2 * dim * (dim * ex) * spat : 0;
    }
    case "dyt":
    case "film": {
      return outputShape.reduce((a, b) => a * b, 1);
    }
    case "crossNetworkDCN": {
      const nL = n(p.numLayers, 3);
      const D = n(p.inputDim, 256);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 ? T * nL * D * D : 0;
    }
    case "ftTransformerBlock": {
      const D = n(p.dModel, 192);
      const ff = n(p.ffMult, 4);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      if (D === 0) return 0;
      return 4 * T * D * D + T * T * D + 2 * T * D * (D * ff);
    }
    case "deformableAttention": {
      const D = n(p.embedDim, 256);
      const h = n(p.numHeads, 8);
      const pts = n(p.numPoints, 4);
      const lv = n(p.numLevels, 4);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 ? 2 * T * D * D + T * (h * lv * pts) * D : 0;
    }
    case "attentionPool": {
      const D = n(p.dim, 512);
      const seeds = n(p.numSeeds, 1);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 ? seeds * T * D + 4 * seeds * D * D : 0;
    }
    // ── Recommendation: sequential / generative ──────────────────────────────
    case "targetAttention": {
      const D = n(p.embedDim, 64);
      const H = n(p.hiddenDim, 36);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 ? T * (4 * D * H + H) + T * D : 0;
    }
    case "behaviorRetrieval": {
      if (String(p.mode ?? "soft") === "hard") return 0;
      const D = n(p.embedDim, 64);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 ? T * D * D : 0;
    }
    case "multiInterest": {
      const D = n(p.embedDim, 64);
      const K = n(p.numInterests, 4);
      const it = n(p.numIterations, 3);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 ? T * D * D + it * T * K * D : 0;
    }
    case "hstuBlock": {
      const D = n(p.embedDim, 512);
      const H = n(p.numHeads, 4);
      const dv = n(p.linearDim, 128);
      const dqk = n(p.attnDim, 128);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      if (D === 0) return 0;
      const proj = T * D * H * (2 * dv + 2 * dqk) + T * H * dv * D;
      const attn = T * T * H * (dqk + dv);
      return proj + attn;
    }
    // ── Video latents / world models ─────────────────────────────────────────
    case "causalConv3d": {
      const inC = inputShape.length >= 4 ? inputShape[0] : n(p.inChannels, 1);
      const outC = n(p.outChannels, 128);
      const k = n(p.kernelSize, 3);
      const vox = outputShape.length >= 4 ? outputShape[1] * outputShape[2] * outputShape[3] : 1;
      return inC * outC * k * k * k * vox;
    }
    case "rssm": {
      const det = n(p.deterDim, 512);
      const st = n(p.stochDim, 32);
      const cls = Math.max(1, n(p.stochClasses, 1));
      const hid = n(p.hiddenDim, 512);
      const flat = st * cls;
      return 3 * (det + flat) * det + 2 * (det * hid + hid * flat);
    }
    case "latentActionModel": {
      const inD = inputShape.length >= 1 ? inputShape[inputShape.length - 1] : 512;
      const hid = n(p.hiddenDim, 512);
      const lat = n(p.latentDim, 32);
      const T = inputShape.length >= 2 ? Math.max(1, inputShape[inputShape.length - 2] - 1) : 1;
      return T * (2 * inD * hid + hid * lat);
    }
    case "jepaPredictor": {
      const D = n(p.embedDim, 768);
      const pd = n(p.predictorDim, 384);
      const depth = n(p.depth, 6);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      if (D === 0 || pd === 0) return 0;
      return 2 * T * D * pd + depth * (4 * T * pd * pd + T * T * pd + 2 * T * pd * (4 * pd));
    }
    case "emaTarget":
      return 0;
    // ── Multimodal connectors / retrieval objectives ─────────────────────────
    case "patchMerger": {
      const k = Math.max(1, n(p.mergeSize, 2));
      const inD = n(p.inDim, inputShape.length >= 1 ? inputShape[inputShape.length - 1] : 1280);
      const outD = n(p.outDim, 3584);
      const merged = inD * k * k;
      const outTokens = outputShape.length >= 2 ? outputShape[outputShape.length - 2] : 1;
      return outTokens * (merged * merged + merged * outD);
    }
    case "contrastiveHead": {
      const inD = inputShape.length >= 1 ? inputShape[inputShape.length - 1] : 0;
      const outD = n(p.projDim, 512);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return T * inD * outD;
    }
    case "matryoshkaHead":
      return 0;
    case "lateInteraction": {
      const D = n(p.embedDim, inputShape.length >= 1 ? inputShape[inputShape.length - 1] : 128);
      const lq = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      const ld = n(p.docLen, lq);
      return lq * ld * D;
    }
    case "revIN":
      return inputShape.reduce((a, b) => a * b, 1) * 2;
    case "seriesDecomp":
      return inputShape.reduce((a, b) => a * b, 1) * n(p.kernelSize, 25);
    case "setAbstraction": {
      const mlp = Array.isArray(p.mlp) ? p.mlp.map((x) => n(x, 0)) : [64, 64, 128];
      const numPoints = n(p.numPoints, 512);
      const numSamples = n(p.numSamples, 32);
      let inC = 3 + 3;
      let total = 0;
      for (const outC of mlp) {
        total += inC * outC;
        inC = outC;
      }
      return numPoints * numSamples * total;
    }
    case "sparseConv3d": {
      const inC = n(p.inChannels, 32);
      const outC = n(p.outChannels, 64);
      const k = n(p.kernelSize, 3);
      const spat = spatialElements(outputShape);
      return inC > 0 && outC > 0 ? inC * outC * k * k * k * spat : 0;
    }
    case "nerfPositionalEncoding":
      return inputShape.reduce((a, b) => a * b, 1) * n(p.numFrequencies, 10) * 2;
    case "dividedSpaceTimeAttention": {
      const D = n(p.embedDim, 768);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 ? 2 * (4 * T * D * D + T * T * D) : 0;
    }
    case "tubeletEmbed": {
      const D = n(p.embedDim, 768);
      const t = Array.isArray(p.tubeletSize) ? p.tubeletSize.map((x) => n(x, 1)) : [2, 16, 16];
      const inC = n(p.inChans ?? p.inChannels, 3);
      const kvol = (t[0] || 1) * (t[1] || 1) * (t[2] || 1);
      const numTubelets = outputShape.length >= 1 ? outputShape[outputShape.length - 2] ?? 196 : 196;
      return D > 0 ? numTubelets * inC * D * kvol : 0;
    }
    // ── Dropout / Flatten / IO (0 MACs) ──────────────────────────────────────
    case "dropout":
    case "flatten":
    case "input":
    case "output":
      return 0;
    default:
      return 0;
  }
}

// src/utils/hardwareSpecs.ts
var GPU_HARDWARE = [
  { id: "b200", label: "B200 (192 GB)", tier: "datacenter", memoryGB: 192, bf16TFLOPS: 2250, fp32TFLOPS: 80, memBandwidthGBs: 8e3, nvlinkGBs: 900, nodeEgressGBs: 800 },
  { id: "h200", label: "H200 (141 GB)", tier: "datacenter", memoryGB: 141, bf16TFLOPS: 989, fp32TFLOPS: 67, memBandwidthGBs: 4800, nvlinkGBs: 450, nodeEgressGBs: 400 },
  { id: "h100-sxm", label: "H100 SXM (80 GB)", tier: "datacenter", memoryGB: 80, bf16TFLOPS: 989, fp32TFLOPS: 67, memBandwidthGBs: 3350, nvlinkGBs: 450, nodeEgressGBs: 400 },
  { id: "a100-80g", label: "A100 (80 GB)", tier: "datacenter", memoryGB: 80, bf16TFLOPS: 312, fp32TFLOPS: 19.5, memBandwidthGBs: 2039, nvlinkGBs: 300, nodeEgressGBs: 200 },
  { id: "a100-40g", label: "A100 (40 GB)", tier: "datacenter", memoryGB: 40, bf16TFLOPS: 312, fp32TFLOPS: 19.5, memBandwidthGBs: 1555, nvlinkGBs: 300, nodeEgressGBs: 200 },
  { id: "l40s", label: "L40S (48 GB)", tier: "datacenter", memoryGB: 48, bf16TFLOPS: 181, fp32TFLOPS: 91.6, memBandwidthGBs: 864, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "l4", label: "L4 (24 GB)", tier: "datacenter", memoryGB: 24, bf16TFLOPS: 121, fp32TFLOPS: 30.3, memBandwidthGBs: 300, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "a10g", label: "A10G (24 GB)", tier: "datacenter", memoryGB: 24, bf16TFLOPS: 70, fp32TFLOPS: 31.2, memBandwidthGBs: 600, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "t4", label: "T4 (16 GB)", tier: "datacenter", memoryGB: 16, bf16TFLOPS: 65, fp32TFLOPS: 8.1, memBandwidthGBs: 320, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "v100", label: "V100 (16 GB)", tier: "datacenter", memoryGB: 16, bf16TFLOPS: 125, fp32TFLOPS: 15.7, memBandwidthGBs: 900, nvlinkGBs: 150, nodeEgressGBs: 100 },
  // Consumer bf16 numbers use fp32 accumulate (the training-relevant rate).
  { id: "rtx-4090", label: "RTX 4090 (24 GB)", tier: "consumer", memoryGB: 24, bf16TFLOPS: 82.6, fp32TFLOPS: 82.6, memBandwidthGBs: 1008, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "rtx-3090", label: "RTX 3090 (24 GB)", tier: "consumer", memoryGB: 24, bf16TFLOPS: 71, fp32TFLOPS: 35.6, memBandwidthGBs: 936, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "m3-max", label: "M3 Max (128 GB)", tier: "apple", memoryGB: 128, bf16TFLOPS: 14.2, fp32TFLOPS: 14.2, memBandwidthGBs: 400, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "m3-pro", label: "M3 Pro (36 GB)", tier: "apple", memoryGB: 36, bf16TFLOPS: 6, fp32TFLOPS: 6, memBandwidthGBs: 150, nvlinkGBs: 0, nodeEgressGBs: 0 },
  // TPUs (scaling-book ch. 2 numbers). nvlinkGBs carries the per-chip ICI
  // egress; TPU pods route everything over ICI so nodeEgressGBs stays 0.
  // fp32TFLOPS mirrors bf16 because no consumer routes TPUs through an fp32
  // path (training panels filter the tier out); do not read it as MXU fp32.
  { id: "tpu-v5e", label: "TPU v5e (16 GB)", tier: "tpu", memoryGB: 16, bf16TFLOPS: 197, fp32TFLOPS: 197, memBandwidthGBs: 819, nvlinkGBs: 180, nodeEgressGBs: 0 },
  { id: "tpu-v5p", label: "TPU v5p (96 GB)", tier: "tpu", memoryGB: 96, bf16TFLOPS: 459, fp32TFLOPS: 459, memBandwidthGBs: 2765, nvlinkGBs: 540, nodeEgressGBs: 0 },
  { id: "tpu-v6e", label: "TPU v6e (32 GB)", tier: "tpu", memoryGB: 32, bf16TFLOPS: 918, fp32TFLOPS: 918, memBandwidthGBs: 1640, nvlinkGBs: 360, nodeEgressGBs: 0 }
];
var BY_ID = new Map(GPU_HARDWARE.map((g) => [g.id, g]));
function gpuById(id) {
  return BY_ID.get(id);
}
function peakFlopsPerSec(gpu, precision = "bf16") {
  return (precision === "fp32" ? gpu.fp32TFLOPS : gpu.bf16TFLOPS) * 1e12;
}
function rooflineTime(flops, bytes, gpu, opts = {}) {
  const c = peakFlopsPerSec(gpu, opts.precision) * (opts.computeUtilization ?? 1);
  const w = gpu.memBandwidthGBs * 1e9 * (opts.memoryUtilization ?? 1);
  const computeSeconds = c > 0 ? flops / c : 0;
  const memorySeconds = w > 0 ? bytes / w : 0;
  return {
    seconds: Math.max(computeSeconds, memorySeconds),
    computeSeconds,
    memorySeconds,
    boundBy: computeSeconds >= memorySeconds ? "compute" : "memory"
  };
}
function machineBalance(gpu, precision = "bf16") {
  const w = gpu.memBandwidthGBs * 1e9;
  return w > 0 ? peakFlopsPerSec(gpu, precision) / w : 0;
}
function criticalBatchSize(gpu, bytesPerParam = 2) {
  return Math.round(machineBalance(gpu) * bytesPerParam / 2);
}
function kvCachePerTokenBytes(spec) {
  const L = Math.max(0, spec.numLayers);
  if (spec.mlaLatentDim && spec.mlaLatentDim > 0) {
    return (spec.mlaLatentDim + (spec.mlaRopeDim ?? 0)) * spec.bytesPerValue * L;
  }
  const kvHeads = spec.kvHeads ?? 0;
  const headDim = spec.headDim ?? 0;
  return 2 * kvHeads * headDim * spec.bytesPerValue * L;
}
function decodeStepEstimate(inp) {
  const w = inp.gpu.memBandwidthGBs * 1e9 * (inp.memoryUtilization ?? 0.8);
  const c = peakFlopsPerSec(inp.gpu) * (inp.computeUtilization ?? 0.6);
  const B = Math.max(1, inp.batchSize);
  if (w <= 0 || c <= 0) {
    return { stepSeconds: 0, tokensPerSecond: 0, perRequestTokensPerSecond: 0, boundBy: "memory", criticalBatchSize: 0 };
  }
  const attnSeconds = B * inp.kvBytesPerRequest / w;
  const mlpCompute = B * inp.flopsPerToken / c;
  const mlpMemory = inp.weightBytes / w;
  const stepSeconds = attnSeconds + Math.max(mlpCompute, mlpMemory);
  return {
    stepSeconds,
    tokensPerSecond: stepSeconds > 0 ? B / stepSeconds : 0,
    perRequestTokensPerSecond: stepSeconds > 0 ? 1 / stepSeconds : 0,
    boundBy: mlpCompute >= mlpMemory ? "compute" : "memory",
    criticalBatchSize: criticalBatchSize(inp.gpu)
  };
}
function prefillSeconds(promptTokens, flopsPerToken, gpu, computeUtilization = 0.6) {
  const c = peakFlopsPerSec(gpu) * computeUtilization;
  return c > 0 ? promptTokens * flopsPerToken / c : 0;
}

// src/utils/requiredParams.ts
var PARAM_ALIASES = {
  numEmbeddings: ["vocabSize"],
  embeddingDim: ["embedDim"],
  embedDim: ["embeddingDim", "hiddenDim", "dModel"],
  maxLen: ["maxPositions", "numPositions", "maxSeqLen"],
  hiddenSize: ["hiddenDim"]
};
var REQUIRED_PARAMS = {
  linear: ["outFeatures"],
  conv2d: ["outChannels", "kernelSize"],
  conv1d: ["outChannels", "kernelSize"],
  conv3d: ["outChannels", "kernelSize"],
  depthwiseConv2d: ["kernelSize"],
  separableConv2d: ["outChannels", "kernelSize"],
  transposeConv2d: ["outChannels", "kernelSize"],
  lstm: ["hiddenSize"],
  gru: ["hiddenSize"],
  rnn: ["hiddenSize"],
  bidirectionalLSTM: ["hiddenSize"],
  groupNorm: ["numGroups"],
  embedding: ["numEmbeddings", "embeddingDim"],
  multiHeadAttention: ["numHeads"],
  groupedQueryAttention: ["numHeads", "numKVHeads"],
  causalAttention: ["numHeads"],
  transformerBlock: ["numHeads"],
  positionalEncoding: ["maxLen"],
  maxpool2d: ["kernelSize"],
  avgpool2d: ["kernelSize"],
  adaptiveAvgPool2d: ["outputSize"],
  upsample: ["scaleFactor"],
  attention: ["numHeads"],
  selfAttention: ["numHeads"],
  patchEmbed: ["patchSize", "embedDim"],
  seBlock: ["reductionRatio"]
};
function isMissingParamValue(val) {
  if (val === void 0 || val === null || val === "") return true;
  if (typeof val === "number" && val === 0) return true;
  if (Array.isArray(val) && (val.length === 0 || val.every((v) => v === 0))) return true;
  return false;
}
function findMissingParams(components) {
  const out = [];
  for (const c of components) {
    const required = REQUIRED_PARAMS[c.type];
    if (!required) continue;
    const missing = required.filter((k) => isMissingParamValue(c.params?.[k]) && (PARAM_ALIASES[k] ?? []).every((alt) => isMissingParamValue(c.params?.[alt])));
    if (missing.length) {
      out.push({
        componentId: c.id,
        componentName: c.name ?? c.type,
        componentType: c.type,
        params: missing
      });
    }
  }
  return out;
}

// src/utils/trainingMemory.ts
var DEFAULT_PRECISION = "bf16-mixed";
var DEFAULT_OPTIMIZER = "adamw";
var CUDA_CONTEXT_GB = 1;
var CUDA_CONTEXT_SOURCE = "CUDA context, not counted by any allocator: about 0.3 to 0.5 GB with lazy loading, up to about 1.1 GB on some PyTorch builds; 1 GB assumed";
var BYTES = { f32: 4, half: 2, i64: 8, bool: 1 };
function stateBytes(precision, optimizer) {
  const w = precision === "bf16" ? 2 : 4;
  const g = w;
  const moments = optimizer === "adam" || optimizer === "adamw" ? 2 : optimizer === "sgd-momentum" ? 1 : 0;
  const stepTemp = optimizer === "adam" || optimizer === "adamw" ? w : 0;
  return { weight: w, grad: g, optim: moments * w, stepTemp };
}
function zeroModelStatesGb(params, largestLayerParams, gpus) {
  const n2 = Math.max(1, Math.floor(gpus));
  return {
    zero2Gb: (2 * params + 18 * params / n2) / 1e9,
    zero3Gb: (4 * largestLayerParams + 18 * params / n2) / 1e9
  };
}
var SHARDING_SOURCE = "DeepSpeed estimator formulas (estimate_zero2/3_model_states_mem_needs), Adam under mixed precision, no offload";
function shardedEstimates(mem, gpuCounts = [2, 4, 8]) {
  return gpuCounts.map((n2) => ({
    gpus: n2,
    ...zeroModelStatesGb(mem.params, mem.largestLayerParams, n2),
    activationsGb: mem.activationsGb
  }));
}
function fewestShardedCards(mem, cardGb, headroom = 1.15, gpuCounts = [2, 4, 8, 16, 32, 64]) {
  for (const n2 of gpuCounts) {
    const z = zeroModelStatesGb(mem.params, mem.largestLayerParams, n2);
    const per = z.zero3Gb + mem.activationsGb + mem.overheadGb;
    if (per * headroom <= cardGb) return { gpus: n2, perCardGb: per };
  }
  return null;
}
var ATTENTION_MHA = /^(multiHeadAttention|selfAttention|attention|causalAttention|crossAttention|crossModalAttention|targetAttention|windowAttention|localAttention|linearAttention)$/;
var GQA_LIKE = /^(groupedQueryAttention|mla)$/;
var MATMUL_LIKE = /^(linear|lmHead|projection|contrastiveHead|matryoshkaHead|dense|fc)$/;
var CONV_LIKE = /conv|patchEmbed|tubeletEmbed|patchMerger/i;
var NORM_FP32 = /^(layerNorm|rmsNorm|groupNorm|instanceNorm|qkNorm)$/;
var SAVES_OUTPUT = /^(relu|sigmoid|tanh|hardtanh|relu6)$/;
var SAVES_INPUT = /^(gelu|silu|swish|mish|elu|leakyRelu|prelu|selu|softplus|hardswish|batchNorm)$/;
var NO_SAVE = /^(add|residual|skipConnection|concatenate|concat|reshape|flatten|permute|transpose|view|squeeze|unsqueeze|identity|output|positionalEncoding|rope|alibi|avgpool2d|avgpool1d|adaptiveAvgPool2d|globalAvgPool2d|globalAvgPool1d|mean|upsample|split|slice)$/;
var VIEW_LIKE = /^(reshape|flatten|permute|transpose|view|squeeze|unsqueeze|identity|output)$/;
var EMBED_LIKE = /^(embedding|embeddingBag|timeEmbedding|positionEmbedding)$/;
var RECURRENT = { lstm: 4, gru: 3, rnn: 1, bidirectionalLSTM: 4 };
var prod = (s) => s && s.length ? s.reduce((a, b) => a * b, 1) : 0;
var finiteShape = (s) => !!s && s.length > 0 && s.every((d) => Number.isFinite(d) && d > 0);
var num3 = (v, d) => {
  const x = typeof v === "number" ? v : Number(v);
  return Number.isFinite(x) && x > 0 ? x : d;
};
function castIn(x, ctx, nodeId) {
  if (!ctx.mixed || x.dtype !== "f32") return x;
  return { id: `cast:${x.id}:${nodeId}`, bytes: x.bytes / 4 * 2, dtype: "half" };
}
function make(id, elems, dtype) {
  return { id, bytes: elems * BYTES[dtype], dtype };
}
function nodeSave(c, ins, shapes, params, ctx) {
  const sh = shapes.get(c.id);
  const outShape = sh?.out ?? null;
  const inShape = sh?.in ?? null;
  const p = c.params ?? {};
  const O = finiteShape(outShape) ? prod(outShape) : 0;
  const x = ins[0] ?? make(`none:${c.id}`, 0, ctx.low);
  const lowB = BYTES[ctx.low];
  const promote = (ts) => ts.some((t2) => t2.dtype === "f32") ? "f32" : ts[0]?.dtype === "i64" ? "f32" : ctx.low;
  const weightCopy = ctx.mixed ? params * 2 : 0;
  const t = c.type;
  const id = (k) => `${k}:${c.id}`;
  const D = outShape && outShape.length ? outShape[outShape.length - 1] : 0;
  const tokens = D > 0 ? O / D : 0;
  if (VIEW_LIKE.test(t)) return { saved: [], out: { ...x, id: x.id }, backwardScratch: 0, weightCopyBytes: 0, fallback: false };
  if (t === "input") return { saved: [], out: x, backwardScratch: 0, weightCopyBytes: 0, fallback: false };
  if (EMBED_LIKE.test(t)) {
    return { saved: [], out: make(id("out"), O, ctx.mixed ? "f32" : ctx.low), backwardScratch: 0, weightCopyBytes: 0, fallback: false };
  }
  if (NO_SAVE.test(t)) {
    const dtype = promote(ins.length ? ins : [x]);
    return { saved: [], out: make(id("out"), O, dtype), backwardScratch: 0, weightCopyBytes: 0, fallback: false };
  }
  if (t === "multiply" || t === "matmul") {
    return { saved: ins.map((i) => castIn(i, ctx, c.id)), out: make(id("out"), O, ctx.mixed ? ctx.low : promote(ins)), backwardScratch: 0, weightCopyBytes: 0, fallback: false };
  }
  if (t === "dropout" || t === "dropPath") {
    return { saved: [make(id("mask"), O, "bool")], out: make(id("out"), O, x.dtype), backwardScratch: 0, weightCopyBytes: 0, fallback: false };
  }
  if (t === "softmax" || t === "logSoftmax") {
    const out2 = make(id("out"), O, ctx.mixed ? "f32" : x.dtype);
    return { saved: [out2], out: out2, backwardScratch: 0, weightCopyBytes: 0, fallback: false };
  }
  if (SAVES_OUTPUT.test(t)) {
    const out2 = make(id("out"), O, x.dtype === "i64" ? "f32" : x.dtype);
    return { saved: [out2], out: out2, backwardScratch: 0, weightCopyBytes: 0, fallback: false };
  }
  if (SAVES_INPUT.test(t)) {
    return { saved: [x], out: make(id("out"), O, x.dtype === "i64" ? "f32" : x.dtype), backwardScratch: 0, weightCopyBytes: 0, fallback: false };
  }
  if (NORM_FP32.test(t)) {
    const saved2 = ctx.mixed && x.dtype !== "f32" ? make(`up:${x.id}:${c.id}`, x.bytes / BYTES[x.dtype], "f32") : x;
    return { saved: [saved2], out: make(id("out"), O, ctx.mixed ? "f32" : x.dtype), backwardScratch: 0, weightCopyBytes: 0, fallback: false };
  }
  if (/^(maxpool1d|maxpool2d|maxpool3d)$/.test(t)) {
    return { saved: [x, make(id("idx"), O, "i64")], out: make(id("out"), O, x.dtype), backwardScratch: 0, weightCopyBytes: 0, fallback: false };
  }
  if (MATMUL_LIKE.test(t) || CONV_LIKE.test(t)) {
    return { saved: [castIn(x, ctx, c.id)], out: make(id("out"), O, ctx.low), backwardScratch: 0, weightCopyBytes: weightCopy, fallback: false };
  }
  if (ATTENTION_MHA.test(t) && finiteShape(inShape) && inShape.length >= 2) {
    const T = inShape[inShape.length - 2];
    const d = inShape[inShape.length - 1];
    const heads = num3(p.numHeads ?? p.num_heads ?? p.nhead, 8);
    const Tk = ins[1] && ins[1].bytes > 0 ? Math.round(ins[1].bytes / BYTES[ins[1].dtype] / d) || T : T;
    const xin = castIn(x, ctx, c.id);
    const kvIn = ins[1] ? castIn(ins[1], ctx, `${c.id}:kv`) : null;
    const q = make(id("q"), T * d, ctx.low);
    const kv = make(id("kv"), 2 * Tk * d, ctx.low);
    const context = make(id("ctx"), T * d, ctx.low);
    const saved2 = [xin, q, kv, context];
    if (kvIn && kvIn.id !== xin.id) saved2.push(kvIn);
    let scratch = 0;
    let fscratch = 0;
    if (ctx.storesScores(c)) {
      const window = t === "windowAttention" ? Math.min(Tk, num3(p.windowSize, 7) ** 2) : Tk;
      const scores = heads * T * window;
      saved2.push(make(id("probs32"), scores, ctx.mixed ? "f32" : ctx.low));
      if (ctx.mixed) saved2.push(make(id("probs16"), scores, ctx.low));
      if (Number(p.dropout ?? p.attnDropout ?? 0) > 0) saved2.push(make(id("pmask"), scores, "bool"), make(id("pdrop"), scores, ctx.low));
      fscratch = scores * (ctx.mixed ? 2 + 4 : lowB);
      scratch = scores * (ctx.mixed ? 2 + 4 + 4 : 2 * lowB);
    } else {
      saved2.push(make(id("lse"), heads * T, "f32"));
    }
    return { saved: saved2, out: make(id("out"), O || T * d, ctx.low), backwardScratch: scratch, forwardScratch: fscratch, weightCopyBytes: weightCopy, fallback: false };
  }
  if (GQA_LIKE.test(t) && finiteShape(inShape) && inShape.length >= 2 && t === "groupedQueryAttention") {
    const T = inShape[inShape.length - 2];
    const d = inShape[inShape.length - 1];
    const heads = num3(p.numHeads, 32);
    const kvHeads = num3(p.numKVHeads, 8);
    const hd = d / heads;
    const xin = castIn(x, ctx, c.id);
    const saved2 = [
      xin,
      make(id("q"), T * d, ctx.low),
      make(id("kv"), 2 * T * kvHeads * hd, ctx.low),
      make(id("ctx"), T * d, ctx.low),
      make(id("lse"), heads * T, "f32")
    ];
    return { saved: saved2, out: make(id("out"), O || T * d, ctx.low), backwardScratch: 0, weightCopyBytes: weightCopy, fallback: false };
  }
  if (t === "feedForward" && finiteShape(inShape)) {
    const d = inShape[inShape.length - 1];
    const T = prod(inShape) / d;
    const ff = num3(p.ffDim ?? p.dFf ?? p.hiddenFeatures, 4 * d);
    const saved2 = [castIn(x, ctx, c.id), make(id("h"), T * ff, ctx.low)];
    return { saved: saved2, out: make(id("out"), O, ctx.low), backwardScratch: 0, weightCopyBytes: weightCopy, fallback: false };
  }
  if ((t === "swiglu" || t === "geglu") && finiteShape(inShape)) {
    const d = inShape[inShape.length - 1];
    const T = prod(inShape) / d;
    const I = num3(p.intermediateSize ?? p.hiddenFeatures ?? p.ffDim, Math.round(d * 8 / 3));
    const saved2 = [castIn(x, ctx, c.id), make(id("gate"), T * I, ctx.low), make(id("up"), T * I, ctx.low), make(id("act"), T * I, ctx.low), make(id("prod"), T * I, ctx.low)];
    return { saved: saved2, out: make(id("out"), O, ctx.low), backwardScratch: 0, weightCopyBytes: weightCopy, fallback: false };
  }
  if (t === "transformerBlock" && finiteShape(inShape) && inShape.length >= 2) {
    const T = inShape[inShape.length - 2];
    const d = num3(p.embedDim ?? p.hiddenDim ?? p.dModel, inShape[inShape.length - 1]);
    const heads = num3(p.numHeads, 8);
    const ff = num3(p.ffDim ?? p.dFf, 4 * d);
    const decoder = ins.length >= 2;
    const f32 = ctx.mixed ? "f32" : ctx.low;
    const saved2 = [
      castIn(x, ctx, c.id),
      // in-projection input
      make(id("qkv"), 3 * T * d, ctx.low),
      // q, k, v
      make(id("ctx"), T * d, ctx.low),
      // attention output, out-projection input
      make(id("lse"), heads * T, "f32"),
      make(id("m1"), T * d, "bool"),
      // dropout after attention
      make(id("n1"), T * d, f32),
      // norm1 input (fp32 under autocast)
      make(id("l1"), T * d, ctx.low),
      // linear1 input
      make(id("h"), T * ff, ctx.low),
      // ReLU output
      make(id("m2"), T * ff, "bool"),
      // dropout mask in the feed-forward
      make(id("h2"), T * ff, ctx.low),
      // linear2 input
      make(id("m3"), T * d, "bool"),
      make(id("n2"), T * d, f32)
    ];
    if (decoder) {
      saved2.push(make(id("xq"), T * d, ctx.low), make(id("xctx"), T * d, ctx.low), make(id("xkv"), 2 * T * d, ctx.low), make(id("m4"), T * d, "bool"), make(id("n3"), T * d, f32));
    }
    return { saved: saved2, out: make(id("out"), O || T * d, f32), backwardScratch: 0, weightCopyBytes: weightCopy, fallback: false };
  }
  const gates = RECURRENT[t];
  if (gates && finiteShape(inShape)) {
    const steps = inShape.length >= 2 ? inShape[inShape.length - 2] : 1;
    const hidden = num3(p.hiddenSize ?? p.hidden_size, 128);
    const layers = num3(p.numLayers ?? p.num_layers, 1);
    const dirs = p.bidirectional || t === "bidirectionalLSTM" ? 2 : 1;
    const elems = steps * hidden * (gates + 2) * layers * dirs;
    return { saved: [castIn(x, ctx, c.id), make(id("rnn"), elems, ctx.low)], out: make(id("out"), O, ctx.low), backwardScratch: 0, weightCopyBytes: weightCopy, fallback: false };
  }
  const din = finiteShape(inShape) ? inShape[inShape.length - 1] : 0;
  let internal = 0;
  if (din >= 2 && tokens > 0 && !/embed|positional|^ema/i.test(t)) {
    let pp = params;
    if (/moe/i.test(t)) {
      const e = num3(p.numExperts, 8), k = num3(p.topK, 2), s = Number(p.numSharedExperts ?? 0);
      pp *= Math.min(1, (k + s) / (e + s));
    }
    internal = Math.max(0, tokens * pp / din - O);
  }
  const out = make(id("out"), O, ctx.low);
  const saved = [castIn(x, ctx, c.id), out];
  if (internal > 0) saved.push(make(id("int"), internal, ctx.low));
  return { saved, out, backwardScratch: 0, weightCopyBytes: weightCopy, fallback: true };
}
function topo(model) {
  const byId = new Map(model.components.map((c, i) => [c.id, i]));
  const indeg = new Map(model.components.map((c) => [c.id, 0]));
  const next = new Map(model.components.map((c) => [c.id, []]));
  for (const e of model.connections ?? []) {
    if (!byId.has(e.from) || !byId.has(e.to)) continue;
    next.get(e.from).push(e.to);
    indeg.set(e.to, (indeg.get(e.to) ?? 0) + 1);
  }
  const ready = model.components.filter((c) => indeg.get(c.id) === 0).map((c) => c.id);
  const seen = /* @__PURE__ */ new Set();
  const out = [];
  while (ready.length) {
    ready.sort((a, b) => byId.get(a) - byId.get(b));
    const cid = ready.shift();
    if (seen.has(cid)) continue;
    seen.add(cid);
    out.push(model.components[byId.get(cid)]);
    for (const n2 of next.get(cid) ?? []) {
      const d = indeg.get(n2) - 1;
      indeg.set(n2, d);
      if (d === 0) ready.push(n2);
    }
  }
  for (const c of model.components) if (!seen.has(c.id)) out.push(c);
  return out;
}
function trainingMemoryFor(modelIn, opts = {}, shapesIn) {
  const { model, shapes } = withTiedHead(modelIn, shapesIn ?? propagateShapes(modelIn).shapes);
  const batch = Math.max(1, Math.round(opts.batchSize ?? 32));
  const precision = opts.precision ?? DEFAULT_PRECISION;
  const optimizer = opts.optimizer ?? DEFAULT_OPTIMIZER;
  const mixed = precision === "bf16-mixed" || precision === "fp16-mixed";
  const ctx = { low: precision === "fp32" ? "f32" : "half", mixed, storesScores: opts.storesScores ?? (() => false) };
  const order = topo(model).filter((c) => c.type !== "stickyNote");
  const index = new Map(order.map((c, i) => [c.id, i]));
  const parents = new Map(order.map((c) => [c.id, []]));
  const children = new Map(order.map((c) => [c.id, []]));
  for (const e of model.connections ?? []) {
    if (!index.has(e.from) || !index.has(e.to)) continue;
    parents.get(e.to).push(e.from);
    children.get(e.from).push(e.to);
  }
  const n2 = order.length;
  const sb = stateBytes(precision, optimizer);
  let params = 0;
  let largestLayerParams = 0;
  const layerParams = [];
  const outOf = /* @__PURE__ */ new Map();
  const saves = [];
  let anyShape = false;
  let fallbackLayers = 0;
  for (const c of order) {
    const inShape = shapes.get(c.id)?.in ?? void 0;
    const pc = c.type === "input" || c.type === "output" ? 0 : estimateLayerParams(c.type, c.params, inShape) ?? 0;
    params += pc;
    largestLayerParams = Math.max(largestLayerParams, pc);
    layerParams.push(pc);
    let ins = (parents.get(c.id) ?? []).map((pid) => outOf.get(pid)).filter((x) => !!x);
    if (c.type === "input") {
      const s2 = shapes.get(c.id)?.out ?? c.params?.shape;
      const feedsLookup = (children.get(c.id) ?? []).some((k) => EMBED_LIKE.test(order[index.get(k)].type));
      const elems = finiteShape(s2) ? prod(s2) : 0;
      ins = [make(`in:${c.id}`, elems, feedsLookup ? "i64" : ctx.mixed ? "f32" : ctx.low)];
    }
    const castable = /embed|norm|positional|^ema/i.test(c.type) ? 0 : pc;
    const s = nodeSave(c, ins, shapes, castable, ctx);
    if (finiteShape(shapes.get(c.id)?.out) && c.type !== "input" && c.type !== "output") anyShape = true;
    if (s.fallback && c.type !== "input" && c.type !== "output") fallbackLayers++;
    saves.push(s);
    outOf.set(c.id, s.out);
  }
  const fwd = (i) => i;
  const bwd = (i) => 2 * n2 - i;
  const STEP = 2 * n2 + 1;
  const lives = /* @__PURE__ */ new Map();
  const extend = (t, from, to, perSample = true) => {
    if (!(t.bytes > 0)) return;
    const bytes = perSample ? t.bytes * batch : t.bytes;
    const l = lives.get(t.id);
    if (l) {
      l.from = Math.min(l.from, from);
      l.to = Math.max(l.to, to);
    } else lives.set(t.id, { bytes, from, to });
  };
  let savedSum = 0;
  const seenSaved = /* @__PURE__ */ new Set();
  order.forEach((c, i) => {
    const s = saves[i];
    const consumers = (children.get(c.id) ?? []).map((k) => index.get(k));
    const lastUse = consumers.length ? Math.max(...consumers) : n2;
    extend(s.out, fwd(i), fwd(lastUse));
    for (const t of s.saved) {
      extend(t, fwd(i), bwd(i));
      if (!seenSaved.has(t.id)) {
        seenSaved.add(t.id);
        savedSum += t.bytes * batch;
      }
    }
    if (s.weightCopyBytes > 0) extend({ id: `wcopy:${c.id}`, bytes: s.weightCopyBytes, dtype: "half" }, fwd(i), bwd(i), false);
    const isView = s.saved.length === 0 && saves[i].out.id === (parents.get(c.id) ?? []).map((pid) => outOf.get(pid)?.id)[0];
    if (!isView && c.type !== "input") {
      extend({ id: `grad:${s.out.id}:${c.id}`, bytes: s.out.bytes, dtype: s.out.dtype }, consumers.length ? bwd(Math.max(...consumers)) : n2, bwd(i));
    }
    if (s.backwardScratch > 0) extend({ id: `scratch:${c.id}`, bytes: s.backwardScratch, dtype: "f32" }, bwd(i), bwd(i));
    if (s.forwardScratch) extend({ id: `fscratch:${c.id}`, bytes: s.forwardScratch, dtype: "f32" }, fwd(i), fwd(i));
    if (c.type === "input") extend(s.out, 0, STEP);
  });
  const lossKind = opts.loss ?? "cross-entropy";
  const sinks = order.filter((c) => (children.get(c.id) ?? []).length === 0);
  for (const c of sinks) {
    const o = outOf.get(c.id);
    if (!o || !(o.bytes > 0)) continue;
    const elems = o.bytes / BYTES[o.dtype];
    extend(o, n2, n2 + 1);
    if (lossKind === "cross-entropy") {
      extend({ id: `lossup:${c.id}`, bytes: o.dtype === "f32" ? 0 : elems * 4, dtype: "f32" }, n2, n2);
      extend({ id: `logsm:${c.id}`, bytes: elems * 4, dtype: "f32" }, n2, n2 + 1);
      extend({ id: `lossgrad:${c.id}`, bytes: elems * 4, dtype: "f32" }, n2 + 1, n2 + 1);
      if (o.dtype !== "f32") extend({ id: `lossgrad16:${c.id}`, bytes: o.bytes, dtype: o.dtype }, n2 + 1, n2 + 1);
    } else {
      extend({ id: `loss32:${c.id}`, bytes: o.dtype === "f32" ? 0 : elems * 4, dtype: "f32" }, n2, n2);
      if (o.dtype !== "f32") extend({ id: `lossgrad16:${c.id}`, bytes: o.bytes, dtype: o.dtype }, n2 + 1, n2 + 1);
    }
  }
  const weightsB = params * sb.weight;
  const optimB = params * sb.optim;
  const gradFrom = new Array(2 * n2 + 3).fill(0);
  layerParams.forEach((pc, i) => {
    if (pc > 0) gradFrom[bwd(i)] += pc * sb.grad;
  });
  const statesAt = (t) => {
    let g = 0;
    for (let k = 0; k <= Math.min(t, STEP); k++) g += gradFrom[k] ?? 0;
    return weightsB + optimB + (t >= n2 ? g : 0) + (t === STEP ? params * sb.stepTemp : 0);
  };
  const actAt = new Array(STEP + 1).fill(0);
  for (const l of lives.values()) {
    for (let t = Math.max(0, l.from); t <= Math.min(STEP, l.to); t++) actAt[t] += l.bytes;
  }
  let peak = -1, peakT = 0;
  for (let t = 0; t <= STEP; t++) {
    const v = statesAt(t) + actAt[t];
    if (v > peak) {
      peak = v;
      peakT = t;
    }
  }
  const peakAt = peakT === STEP ? "optimizer-step" : peakT < n2 ? "forward" : peakT === n2 ? "loss" : "backward";
  const layerAt = peakT < n2 ? order[peakT] : peakT > n2 && peakT < STEP ? order[2 * n2 - peakT] : null;
  const gb2 = (b) => b / 1e9;
  const modelStates = {
    weightsGb: gb2(weightsB),
    gradsGb: gb2(params * sb.grad),
    optimizerGb: gb2(optimB),
    bytesPerParam: sb.weight + sb.grad + sb.optim
  };
  const peakGb = gb2(peak);
  const statesAtPeakGb = gb2(statesAt(peakT));
  return {
    params,
    batchSize: batch,
    precision,
    optimizer,
    modelStates,
    modelStatesGb: modelStates.weightsGb + modelStates.gradsGb + modelStates.optimizerGb,
    activationsGb: gb2(actAt[peakT]),
    statesAtPeakGb,
    peakGb,
    overheadGb: params > 0 ? CUDA_CONTEXT_GB : 0,
    totalGb: peakGb + (params > 0 ? CUDA_CONTEXT_GB : 0),
    peakAt,
    peakLayer: layerAt ? layerAt.name || layerAt.type : null,
    savedSumGb: gb2(savedSum),
    largestLayerParams,
    activationsKnown: anyShape,
    fallbackLayers
  };
}
var FLOP_CONVENTION = "FLOPs count a multiply-add as 2 (torch.utils.flop_counter convention); backward is 2x forward; elementwise ops and norms count 0; causal attention is counted as full, about 2x what a causal kernel does";
function flopsBreakdown(forwardMacs) {
  const f = Math.max(0, forwardMacs) * 2;
  return { forwardMacs: Math.max(0, forwardMacs), forwardFlops: f, trainFlopsPerSample: 3 * f, convention: FLOP_CONVENTION };
}

// src/utils/preflightReport.ts
function shapeFixHint(s) {
  const both = Array.isArray(s.actual) ? s.actual.join("\xD7") : s.actual;
  switch (s.kind) {
    case "no-input-shape":
      return "Click the input layer and set its shape, e.g. [3, 224, 224] for images or [features] for tabular.";
    case "merge-shape-mismatch":
      return `Both inputs to this ${s.componentType} must have the same shape (${both ?? "they differ"}). Insert a Linear/reshape on the mismatched branch so the dimensions line up, or fix the upstream layer that changed the size.`;
    case "linear-in-mismatch":
      return "Set this layer's inFeatures to match the upstream output size (or use the Apply fix).";
    case "head-dim-divisibility":
      return "embedDim must be divisible by numHeads. Pick a numHeads that divides embedDim (or use the Apply fix).";
    case "gqa-head-divisibility":
      return "numHeads must be divisible by numKVHeads. Adjust one so it divides evenly (or use the Apply fix).";
    case "invalid-output-shape":
      return "This layer produces a non-positive dimension. Check its kernel/stride/padding or out-features so the output stays positive.";
    case "compute-error":
      return "This layer's parameters are inconsistent and the shape cannot be computed. Open it and check the values.";
    default:
      return void 0;
  }
}
function dataSpecInputShapes(d) {
  if (d?.inputs && d.inputs.length) return d.inputs.map((i) => i.shape ?? []);
  if (d?.inputShape && d.inputShape.length) return [d.inputShape];
  return [];
}
var PREFLIGHT_FIT_HEADROOM = 1.15;
var PREFLIGHT_GPUS = [
  { name: "T4 (16GB)", gb: 16, hourly: 0.59, speed: 0.55, hwId: "t4" },
  { name: "L4 (24GB)", gb: 24, hourly: 0.8, speed: 0.8, hwId: "l4" },
  { name: "A10G (24GB)", gb: 24, hourly: 1.1, speed: 1, hwId: "a10g" },
  { name: "L40S (48GB)", gb: 48, hourly: 1.95, speed: 2.5, hwId: "l40s" },
  { name: "A100 (40GB)", gb: 40, hourly: 2.1, speed: 2.2, hwId: "a100-40g" },
  { name: "A100 (80GB)", gb: 80, hourly: 2.5, speed: 2.4, hwId: "a100-80g" },
  { name: "H100 (80GB)", gb: 80, hourly: 3.95, speed: 4.5, hwId: "h100-sxm" }
];
var ASSUMED_MFU = 0.35;
function categoryForAdvisor(cat) {
  if (cat === "structure") return "structure";
  if (cat === "performance") return "training";
  return "training";
}
function buildPreflightReport(model, opts) {
  const findings = [];
  const comps = model.components.filter((c) => c.type !== "stickyNote");
  const realLayers = comps.filter((c) => c.type !== "input" && c.type !== "output");
  const { shapes, issues: shapeIssues } = propagateShapes(model);
  const unreliableFlow = downstreamOfInferredEdges(model);
  for (const s of shapeIssues) {
    if (FLOW_DEPENDENT_SHAPE_KINDS.has(s.kind) && unreliableFlow.has(s.componentName ?? "")) continue;
    findings.push({
      id: `shape-${s.kind}-${s.componentId}`,
      // `kind` is the stable rule id the lint engine and lib/ruleProvenance.ts
      // both key on; the `id` above bakes in a component and cannot join.
      ruleId: s.kind,
      category: s.kind === "no-input-shape" ? "data" : "shapes",
      // A shape that can't compute means the model literally won't run.
      severity: "block",
      title: s.kind === "no-input-shape" ? "Input shape not set" : `Shape problem: ${s.componentType}`,
      detail: s.message,
      // Concrete guidance per issue kind, so even when there's no one-click fix
      // the user knows exactly what to change (not just "Fix with AI").
      fix: shapeFixHint(s),
      affectedIds: [s.componentId]
    });
  }
  const accepted = model.acceptedAdvisorRules ?? [];
  const advisorIssues = runAdvisorRules(model).filter((i) => !accepted.includes(i.ruleId));
  for (const a of advisorIssues) {
    findings.push({
      id: `adv-${a.id}`,
      ruleId: a.ruleId,
      category: categoryForAdvisor(a.category),
      severity: a.severity === "error" ? "block" : a.severity === "warning" ? "warn" : "info",
      title: a.title,
      detail: a.message,
      fix: a.suggestion,
      affectedIds: a.affectedIds
    });
  }
  const missingParams = findMissingParams(comps);
  if (missingParams.length) {
    const first = missingParams[0];
    const others = missingParams.length - 1;
    const alsoText = others > 0 ? ` ${others} other layer${others !== 1 ? "s" : ""} ${others !== 1 ? "are" : "is"} incomplete too.` : "";
    findings.push({
      id: "structure-missing-params",
      category: "structure",
      severity: "block",
      title: missingParams.length === 1 ? `"${first.componentName}" is missing ${first.params.join(", ")}` : `${missingParams.length} layers are missing required parameters`,
      detail: `"${first.componentName}" (${first.componentType}) has no value for ${first.params.join(", ")}.${alsoText} A layer without these cannot be built, so the run fails at construction rather than part way through.`,
      fix: "Select the layer and fill the empty fields in the inspector.",
      affectedIds: missingParams.map((m) => m.componentId)
    });
  }
  const inputComp = comps.find((c) => c.type === "input");
  const outputComp = comps.find((c) => c.type === "output");
  if (!inputComp) {
    findings.push({
      id: "data-no-input",
      category: "data",
      severity: "block",
      title: "No Input node",
      detail: "Add an Input layer so the model knows the shape of your data.",
      fix: "Drag an Input from the I/O section and set its shape to your data."
    });
  }
  if (inputComp && !outputComp) {
    findings.push({
      id: "data-no-output",
      category: "data",
      severity: "warn",
      title: "No Output node",
      detail: "The trainer will not know where the forward pass terminates.",
      fix: "Connect the last layer to an Output node."
    });
  }
  const dataSpec = opts?.dataSpec;
  if (dataSpec) {
    const inputComps = comps.filter((c) => c.type === "input");
    const specShapes = dataSpecInputShapes(dataSpec);
    const multi = inputComps.length > 1 || specShapes.length > 1;
    inputComps.forEach((ic, i) => {
      const modelInShape = ic.params?.shape;
      const dataShape = specShapes[i];
      if (!dataShape || !dataShape.length || !modelInShape || !modelInShape.length) return;
      const tag = multi ? ` (input ${i + 1}${ic.name ? ` "${ic.name}"` : ""})` : "";
      if (modelInShape.join("x") !== dataShape.join("x")) {
        findings.push({
          id: `data-input-mismatch-${i}`,
          category: "data",
          severity: "block",
          title: `Input shape does not match your data${tag}`,
          detail: `The model's Input is [${modelInShape.join(", ")}] but your data is [${dataShape.join(", ")}].`,
          fix: `Set the Input shape to [${dataShape.join(", ")}].`,
          affectedIds: [ic.id]
        });
      } else {
        findings.push({
          id: `data-input-ok-${i}`,
          category: "data",
          severity: "pass",
          title: `Input matches your data${tag}`,
          detail: `Input shape [${modelInShape.join(", ")}] lines up with your data.`
        });
      }
    });
    if (dataSpec.numClasses && dataSpec.numClasses > 0) {
      const outInShape = outputComp ? shapes.get(outputComp.id)?.in : null;
      const outDim = outInShape && outInShape.length ? outInShape[outInShape.length - 1] : void 0;
      if (outDim != null) {
        if (outDim !== dataSpec.numClasses) {
          findings.push({
            id: "data-output-mismatch",
            category: "data",
            severity: "block",
            title: "Output size does not match your target",
            detail: `The model outputs ${outDim} value${outDim !== 1 ? "s" : ""} but your data has ${dataSpec.numClasses} class${dataSpec.numClasses !== 1 ? "es" : ""}.`,
            fix: `Make the final layer output ${dataSpec.numClasses}.`,
            affectedIds: outputComp ? [outputComp.id] : void 0
          });
        } else {
          findings.push({
            id: "data-output-ok",
            category: "data",
            severity: "pass",
            title: "Output matches your target",
            detail: `Final output of ${outDim} matches ${dataSpec.numClasses} class${dataSpec.numClasses !== 1 ? "es" : ""}.`
          });
        }
      }
    }
    if (dataSpec.classCounts && dataSpec.classCounts.length > 1) {
      const positive = dataSpec.classCounts.filter((x) => x > 0);
      const mx = Math.max(...positive), mn = Math.min(...positive);
      if (mn > 0 && mx / mn >= 10) {
        findings.push({
          id: "data-imbalance",
          category: "data",
          severity: "warn",
          title: "Class imbalance",
          detail: `The most vs. least frequent class is about ${Math.round(mx / mn)}:1. Plain accuracy will be misleading.`,
          fix: "Use class weights, resampling, or a balanced loss (e.g. focal loss)."
        });
      }
    }
  }
  let params = 0, flops = 0;
  for (const c of realLayers) {
    const inShape = shapes.get(c.id)?.in;
    params += estimateLayerParams(c.type, c.params, inShape) ?? 0;
    const outShape = shapes.get(c.id)?.out;
    flops += estimateLayerFlops(c.type, c.params, inShape ?? [], outShape ?? []) ?? 0;
  }
  const weightBytes2 = params * 4;
  const trainFootprintBytes = params * 16;
  const gpu = PREFLIGHT_GPUS.find((g) => g.name === opts?.plan?.gpu) ?? PREFLIGHT_GPUS[2];
  const epochs = Math.max(1, opts?.plan?.epochs ?? 10);
  const batch = Math.max(1, opts?.plan?.batchSize ?? 32);
  const planAssumed = opts?.plan?.epochs == null || opts?.plan?.batchSize == null;
  const precision = opts?.plan?.precision ?? DEFAULT_PRECISION;
  const optimizer = opts?.plan?.optimizer ?? DEFAULT_OPTIMIZER;
  const tm = trainingMemoryFor(model, { batchSize: batch, precision, optimizer }, shapes);
  const needGb = Math.max(tm.peakGb * PREFLIGHT_FIT_HEADROOM, tm.totalGb);
  const fitsCards = params > 0 ? PREFLIGHT_GPUS.filter((g) => needGb <= g.gb).map((g) => g.name) : [];
  const smallestFit = params > 0 ? PREFLIGHT_GPUS.find((g) => needGb <= g.gb) ?? null : null;
  const gpuFits = params > 0 ? needGb <= gpu.gb : null;
  const largestCard = PREFLIGHT_GPUS.reduce((a, g) => g.gb > a.gb ? g : a, PREFLIGHT_GPUS[0]);
  const zero3 = params > 0 && !smallestFit ? fewestShardedCards(tm, largestCard.gb, PREFLIGHT_FIT_HEADROOM) : null;
  const precisionWord = precision === "bf16-mixed" ? "bf16 mixed precision" : precision === "fp16-mixed" ? "fp16 mixed precision" : precision === "bf16" ? "pure bf16" : "fp32";
  const optimizerWord = optimizer === "adamw" ? "AdamW" : optimizer === "adam" ? "Adam" : optimizer === "sgd-momentum" ? "SGD with momentum" : "SGD";
  const memAssumptions = `Batch ${batch}${planAssumed ? " (assumed)" : ""}, ${precisionWord}, ${optimizerWord} (${tm.modelStates.bytesPerParam} B per parameter), fused attention, cross-entropy loss. Activations are the tensors each layer keeps for backward, replayed over the step and taken at the peak, not summed. ${CUDA_CONTEXT_SOURCE}. Arithmetic on the graph, not a measurement.`;
  const memory = {
    batchSize: batch,
    precision,
    optimizer,
    modelStatesGb: tm.modelStatesGb,
    activationsGb: tm.activationsGb,
    overheadGb: tm.overheadGb,
    peakGb: tm.peakGb,
    needGb,
    peakAt: tm.peakAt,
    peakLayer: tm.peakLayer,
    activationsKnown: tm.activationsKnown,
    sharded: params > 0 ? shardedEstimates(tm) : [],
    shardedFit: zero3 ? { gpus: zero3.gpus, gpu: largestCard.name, perCardGb: zero3.perCardGb } : null,
    assumptions: memAssumptions
  };
  const g1 = (v) => v.toFixed(v < 1 ? 2 : 1);
  const threeNumbers = `model states ${g1(tm.modelStatesGb)} GB, activations ${tm.activationsKnown ? `${g1(tm.activationsGb)} GB at batch ${batch}` : "unknown (no shape propagated)"}, CUDA context ${g1(tm.overheadGb)} GB`;
  if (params > 0) {
    if (gpuFits) {
      findings.push({
        id: "hw-fit",
        category: "hardware",
        severity: "pass",
        title: `Fits a ${gpu.name}`,
        detail: `Peak about ${g1(tm.totalGb)} GB: ${threeNumbers}.`
      });
    } else {
      const statesAlone = tm.modelStatesGb * PREFLIGHT_FIT_HEADROOM > gpu.gb;
      const lever = statesAlone ? "Model states alone exceed the card, so no batch size helps." : `Activations at batch ${batch} are what push it over; a smaller batch or activation checkpointing lowers that term.`;
      const shard = zero3 ? ` Sharded with ZeRO-3 / FSDP it needs about ${g1(zero3.perCardGb)} GB per card on ${zero3.gpus} x ${largestCard.name} (${SHARDING_SOURCE}, plus activations and context per card).` : "";
      findings.push({
        id: "hw-no-fit",
        category: "hardware",
        severity: "warn",
        title: `Will not fit a ${gpu.name}`,
        detail: `Peak about ${g1(tm.totalGb)} GB: ${threeNumbers}. ${lever}${smallestFit ? ` The smallest card that holds it is a ${smallestFit.name}.` : " No single offered card holds it."}${shard}`
      });
    }
  }
  const fromFile = dataSpec?.samples && dataSpec.samples > 0 ? dataSpec.samples : null;
  const fromSentence = opts?.workload && opts.workload.samples > 0 ? opts.workload.samples : null;
  const samples = fromFile ?? fromSentence ?? 5e4;
  const samplesAssumed = fromFile === null && fromSentence === null;
  const samplesFrom = fromFile !== null ? "" : fromSentence !== null ? ` (from "${opts.workload.source}")` : " (assumed)";
  const hw = gpuById(gpu.hwId) ?? gpuById("a10g");
  const flopsPerStep = 2 * Math.max(flops, params) * 3 * batch;
  const roofline = rooflineTime(flopsPerStep, params * 32, hw, {
    computeUtilization: ASSUMED_MFU,
    memoryUtilization: 0.8
  });
  const perStepSec = 5e-3 + roofline.seconds;
  const steps = Math.max(1, Math.ceil(samples / batch));
  const estTrainSec = params > 0 ? 60 + epochs * steps * perStepSec : 0;
  const estCostUsd = estTrainSec / 3600 * gpu.hourly;
  const trainFlops = flopsPerStep * steps * epochs;
  const peakTflops = hw.bf16TFLOPS;
  const computeSec = peakTflops > 0 ? trainFlops / (peakTflops * 1e12 * ASSUMED_MFU) : 0;
  if (params > 0) {
    const hrs = estTrainSec / 3600;
    const timeStr = hrs >= 1 ? `${hrs.toFixed(1)}h` : `${Math.ceil(estTrainSec / 60)}m`;
    const costStr = estCostUsd < 0.01 ? "<$0.01" : estCostUsd < 1 ? `~$${estCostUsd.toFixed(2)}` : estCostUsd < 1e3 ? `~$${estCostUsd.toFixed(1)}` : estCostUsd < 1e6 ? `~$${Math.round(estCostUsd).toLocaleString("en-US")}` : `~$${(estCostUsd / 1e6).toFixed(1)}M`;
    findings.push({
      id: "cost-estimate",
      category: "size",
      severity: "info",
      title: `Rough training cost: ${costStr}, ${timeStr}`,
      detail: `${epochs} epochs, batch ${batch}, on a ${gpu.name}, over ${samples.toLocaleString()} samples${samplesFrom}.`,
      fix: samplesAssumed ? 'Drop your CSV in "Your data" for an accurate sample count.' : void 0
    });
  }
  if (params >= 1e9) {
    findings.push({
      id: "size-large",
      category: "size",
      severity: "info",
      title: "Large model",
      detail: `About ${(params / 1e9).toFixed(1)}B parameters. Expect serious GPU time and cost; consider parameter-efficient fine-tuning.`
    });
  }
  const counts = { block: 0, warn: 0, info: 0, pass: 0 };
  for (const f of findings) counts[f.severity]++;
  let score = 100 - counts.warn * 8 - counts.info * 2;
  if (counts.block > 0) score = Math.min(score, 35) - counts.block * 5;
  score = Math.max(0, Math.min(100, Math.round(score)));
  const verdict = counts.block > 0 ? "not-ready" : counts.warn > 0 ? "caution" : "ready";
  return {
    score,
    verdict,
    findings,
    metrics: {
      layers: realLayers.length,
      params,
      flops,
      flopsDetail: flopsBreakdown(flops),
      memory,
      fitsCards,
      compute: params > 0 ? { trainFlops, peakTflops, assumedMfu: ASSUMED_MFU, computeSec } : void 0,
      weightBytes: weightBytes2,
      trainFootprintBytes,
      fitsGpu: gpu.name,
      gpuFits,
      smallestFitGpu: smallestFit?.name ?? null,
      estTrainSec,
      estCostUsd,
      samplesAssumed,
      samples,
      epochs,
      batchSize: batch,
      planAssumed
    },
    counts
  };
}

// src/utils/modelFormat.ts
function paramDiffKeys(a, b) {
  const keys = /* @__PURE__ */ new Set([...Object.keys(a ?? {}), ...Object.keys(b ?? {})]);
  const changed = [];
  for (const k of keys) {
    if (JSON.stringify(a?.[k]) !== JSON.stringify(b?.[k])) changed.push(k);
  }
  return changed.sort();
}
function diffArchitectures(prev, next) {
  const SEP = "\0";
  const displayName = (key) => key.split(SEP)[0];
  const byName3 = (m) => {
    const map = /* @__PURE__ */ new Map();
    const seen = /* @__PURE__ */ new Map();
    for (const c of m.components) {
      const n2 = seen.get(c.name) ?? 0;
      seen.set(c.name, n2 + 1);
      map.set(n2 === 0 ? c.name : `${c.name}${SEP}${n2}`, c);
    }
    return map;
  };
  const prevMap = byName3(prev);
  const nextMap = byName3(next);
  const layersAdded = [];
  const layersRemoved = [];
  const layersChanged = [];
  for (const [key, nc] of nextMap) {
    const pc = prevMap.get(key);
    if (!pc) {
      layersAdded.push(displayName(key));
      continue;
    }
    const typeChanged = pc.type !== nc.type ? { from: pc.type, to: nc.type } : void 0;
    const paramsChanged = paramDiffKeys(pc.params, nc.params);
    if (typeChanged || paramsChanged.length > 0) {
      layersChanged.push({
        name: displayName(key),
        ...typeChanged ? { typeChanged } : {},
        ...paramsChanged.length > 0 ? { paramsChanged } : {}
      });
    }
  }
  for (const key of prevMap.keys()) {
    if (!nextMap.has(key)) layersRemoved.push(displayName(key));
  }
  const edgeKeys = (m) => {
    const idToName = new Map(m.components.map((c) => [c.id, c.name]));
    const set = /* @__PURE__ */ new Set();
    for (const e of m.connections) {
      const f = idToName.get(e.from) ?? e.from;
      const t = idToName.get(e.to) ?? e.to;
      set.add(`${f} -> ${t}`);
    }
    return set;
  };
  const prevEdges = edgeKeys(prev);
  const nextEdges = edgeKeys(next);
  const connectionsAdded = [...nextEdges].filter((e) => !prevEdges.has(e)).sort();
  const connectionsRemoved = [...prevEdges].filter((e) => !nextEdges.has(e)).sort();
  const identical = layersAdded.length === 0 && layersRemoved.length === 0 && layersChanged.length === 0 && connectionsAdded.length === 0 && connectionsRemoved.length === 0;
  return {
    layersAdded: layersAdded.sort(),
    layersRemoved: layersRemoved.sort(),
    layersChanged,
    connectionsAdded,
    connectionsRemoved,
    identical
  };
}

// src/utils/modelImpact.ts
var SHAPE_SENSITIVE_TYPES = /* @__PURE__ */ new Set([
  "linear",
  "conv2d",
  "conv3d",
  "conv1d",
  "depthwiseConv2d",
  "separableConv2d",
  "transposeConv2d",
  "audioConv",
  "dilatedConv2d",
  "depthwiseConv1d",
  "embedding",
  "embeddingBag",
  "segmentEmbedding",
  "positionalEncoding",
  "learnedPositionalEmbedding",
  "lmHead",
  "multiHeadAttention",
  "groupedQueryAttention",
  "causalAttention",
  "selfAttention",
  "crossAttention",
  "attention",
  "transformerBlock",
  "lstm",
  "gru",
  "rnn",
  "bidirectionalLSTM",
  "bidirectionalGRU",
  "batchNorm",
  "layerNorm",
  "instanceNorm",
  "groupNorm",
  "rmsNorm",
  "mamba",
  "conformerBlock"
]);
var PARAM_CARRYING_TYPES = /* @__PURE__ */ new Set([
  "linear",
  "conv2d",
  "conv3d",
  "conv1d",
  "depthwiseConv2d",
  "separableConv2d",
  "transposeConv2d",
  "audioConv",
  "dilatedConv2d",
  "depthwiseConv1d",
  "embedding",
  "embeddingBag",
  "segmentEmbedding",
  "lstm",
  "gru",
  "rnn",
  "bidirectionalLSTM",
  "bidirectionalGRU",
  "multiHeadAttention",
  "groupedQueryAttention",
  "causalAttention",
  "selfAttention",
  "crossAttention",
  "attention",
  "transformerBlock",
  "feedForward",
  "mamba",
  "conformerBlock",
  "lmHead",
  "batchNorm",
  "layerNorm",
  "instanceNorm",
  "groupNorm",
  "rmsNorm",
  "tabnet",
  "graphConv",
  "graphAttention",
  "graphSAGE",
  "gcn"
]);
function buildAdjacency(model) {
  const downstream = /* @__PURE__ */ new Map();
  const upstream = /* @__PURE__ */ new Map();
  for (const c of model.components) {
    downstream.set(c.id, []);
    upstream.set(c.id, []);
  }
  for (const conn of model.connections) {
    downstream.get(conn.from)?.push(conn.to);
    upstream.get(conn.to)?.push(conn.from);
  }
  return { downstream, upstream };
}
function bfs(startIds, adj, direction, compById, excluded) {
  const visited = new Set(startIds);
  const out = [];
  let frontier = startIds.map((id) => ({ id, distance: 0 }));
  while (frontier.length) {
    const next = [];
    for (const { id, distance } of frontier) {
      for (const neighbor of adj.get(id) ?? []) {
        if (visited.has(neighbor) || excluded.has(neighbor)) continue;
        visited.add(neighbor);
        const comp = compById.get(neighbor);
        if (!comp) continue;
        out.push({
          id: neighbor,
          name: comp.name,
          type: comp.type,
          distance: distance + 1,
          direction,
          shapeSensitive: SHAPE_SENSITIVE_TYPES.has(comp.type)
        });
        next.push({ id: neighbor, distance: distance + 1 });
      }
    }
    frontier = next;
  }
  return out;
}
function analyzeImpact(model, targets, opts = {}) {
  const { includeUpstream = true, maxDistance } = opts;
  const compById = new Map(model.components.map((c) => [c.id, c]));
  const { downstream: dAdj, upstream: uAdj } = buildAdjacency(model);
  const targetIds = new Set(targets.map((t) => t.id));
  let downstreamNodes = bfs(targets.map((t) => t.id), dAdj, "downstream", compById, /* @__PURE__ */ new Set());
  let upstreamNodes = includeUpstream ? bfs(targets.map((t) => t.id), uAdj, "upstream", compById, /* @__PURE__ */ new Set()) : [];
  if (typeof maxDistance === "number") {
    downstreamNodes = downstreamNodes.filter((n2) => n2.distance <= maxDistance);
    upstreamNodes = upstreamNodes.filter((n2) => n2.distance <= maxDistance);
  }
  const dIds = new Set(downstreamNodes.map((n2) => n2.id));
  upstreamNodes = upstreamNodes.filter((n2) => !dIds.has(n2.id) && !targetIds.has(n2.id));
  downstreamNodes.sort((a, b) => a.distance - b.distance);
  upstreamNodes.sort((a, b) => a.distance - b.distance);
  const shapeChanging = downstreamNodes.filter((n2) => n2.shapeSensitive).length;
  const paramCarrying = downstreamNodes.filter((n2) => PARAM_CARRYING_TYPES.has(n2.type)).length;
  return {
    targets,
    unresolved: [],
    upstream: upstreamNodes,
    downstream: downstreamNodes,
    totals: {
      targets: targets.length,
      upstream: upstreamNodes.length,
      downstream: downstreamNodes.length,
      shapeChanging,
      paramCarrying,
      totalAffected: targets.length + upstreamNodes.length + downstreamNodes.length
    },
    summary: formatImpactSummary(targets, downstreamNodes, upstreamNodes, shapeChanging, paramCarrying)
  };
}
function formatImpactSummary(targets, downstream, upstream, shapeChanging, paramCarrying) {
  if (!targets.length) return "No matching components.";
  const lines = [];
  const tNames = targets.map((t) => t.name).slice(0, 4);
  const tSuffix = targets.length > 4 ? `, +${targets.length - 4} more` : "";
  lines.push(`**Target:** ${tNames.join(", ")}${tSuffix} (${targets.length})`);
  if (downstream.length) {
    lines.push(`**Downstream affected:** ${downstream.length} layer${downstream.length === 1 ? "" : "s"}`);
    if (shapeChanging) lines.push(`  - ${shapeChanging} may need shape recompute`);
    if (paramCarrying) lines.push(`  - ${paramCarrying} carry weights (rebuild needed)`);
  }
  if (upstream.length) {
    lines.push(`**Upstream context:** ${upstream.length} layer${upstream.length === 1 ? "" : "s"}`);
  }
  if (!downstream.length && !upstream.length) {
    lines.push("No connected neighbors. Safe to apply.");
  }
  return lines.join("\n");
}

// lib/archFingerprint.ts
function fnv1a(str2) {
  let h = 2166136261;
  for (let i = 0; i < str2.length; i++) {
    h ^= str2.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}
function architectureHashFromJson(architecture) {
  if (!architecture || typeof architecture !== "object") return null;
  const arch = architecture;
  if (!Array.isArray(arch.components) || arch.components.length === 0) return null;
  const counts = /* @__PURE__ */ new Map();
  for (const c of arch.components) {
    const type = c && typeof c === "object" && typeof c.type === "string" ? c.type : "unknown";
    counts.set(type, (counts.get(type) ?? 0) + 1);
  }
  const histogram = [...counts.entries()].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([t, n2]) => `${t}:${n2}`).join("|");
  const edgeCount = Array.isArray(arch.connections) ? arch.connections.length : 0;
  return fnv1a(`${histogram}#e${edgeCount}`);
}

// src/utils/costEstimator.ts
var GPU_SPECS = GPU_HARDWARE.filter((g) => g.tier !== "apple" && g.tier !== "tpu").map((g) => ({
  id: g.id,
  label: g.label,
  memoryGB: g.memoryGB,
  peakTFLOPS: g.bf16TFLOPS,
  peakFP32TFLOPS: g.fp32TFLOPS,
  memBandwidthGBs: g.memBandwidthGBs
}));
function kvCachePerTokenFor(c, bytesPerValue, seqLen) {
  const p = c.params || {};
  const n2 = (v, fb = 0) => {
    const x = typeof v === "number" ? v : Number(v);
    return isFinite(x) && x > 0 ? x : fb;
  };
  const fullAttention = (dim, heads) => kvCachePerTokenBytes({
    numLayers: 1,
    bytesPerValue,
    kvHeads: heads,
    headDim: heads > 0 ? dim / heads : dim
  });
  if (p.mtp === true) return 0;
  switch (c.type) {
    case "multiHeadAttention":
    case "selfAttention":
    case "causalAttention":
    case "transformerBlock":
    case "attention":
    case "crossAttention":
    case "differentialAttention":
    case "nativeSparseAttention": {
      const dim = n2(p.embedDim ?? p.hiddenDim, 768);
      const heads = n2(p.numHeads, 1);
      return fullAttention(dim, heads);
    }
    case "localAttention":
    case "windowAttention": {
      const dim = n2(p.embedDim ?? p.hiddenDim, 768);
      const heads = n2(p.numHeads, 1);
      const window = n2(p.windowSize, c.type === "localAttention" ? 256 : 7);
      const gqa = c.type === "localAttention" && (p.numKVHeads != null || p.headDim != null);
      const full = gqa ? kvCachePerTokenBytes({
        numLayers: 1,
        bytesPerValue,
        kvHeads: n2(p.numKVHeads, heads),
        headDim: n2(p.headDim, heads > 0 ? dim / heads : 64)
      }) : fullAttention(dim, heads);
      if (seqLen && seqLen > window) return full * (window / seqLen);
      return full;
    }
    case "linearAttention":
      return 0;
    case "groupedQueryAttention": {
      const dim = n2(p.embedDim, 768);
      const heads = n2(p.numHeads, 8);
      const kvHeads = n2(p.numKVHeads, heads);
      const headDim = n2(p.headDim, heads > 0 ? dim / heads : 64);
      return kvCachePerTokenBytes({ numLayers: 1, bytesPerValue, kvHeads, headDim });
    }
    case "mla": {
      const latent = n2(p.kvLatentDim, 128);
      const ropeDim = n2(p.ropeHeadDim, 0);
      return kvCachePerTokenBytes({ numLayers: 1, bytesPerValue, mlaLatentDim: latent, mlaRopeDim: ropeDim });
    }
    default:
      return 0;
  }
}
function kvBytesPerTokenForModel(model, bytesPerValue = 2, seqLen) {
  return model.components.reduce((s, c) => s + kvCachePerTokenFor(c, bytesPerValue, seqLen), 0);
}
function recurrentStatePerSequenceFor(c, bytesPerValue) {
  const p = c.params || {};
  const n2 = (v, fb = 0) => {
    const x = typeof v === "number" ? v : Number(v);
    return isFinite(x) && x > 0 ? x : fb;
  };
  switch (c.type) {
    case "mamba": {
      const d = n2(p.dModel ?? p.embedDim ?? p.hiddenDim, 256);
      const di = d * n2(p.expand, 2);
      const taps = Math.max(0, n2(p.dConv, 4) - 1);
      return (di * n2(p.dState, 16) + taps * di) * bytesPerValue;
    }
    case "mamba2": {
      const H = n2(p.numHeads);
      if (!H) return 0;
      const di = H * n2(p.headDim, 64);
      const S = n2(p.dState, 128);
      const taps = Math.max(0, n2(p.dConv, 4) - 1);
      const convDim2 = di + 2 * n2(p.nGroups, 1) * S;
      return (di * S + taps * convDim2) * bytesPerValue;
    }
    case "gatedDeltaNet": {
      const Hk = n2(p.numKeyHeads);
      if (!Hk) return 0;
      const Hv = n2(p.numHeads, Hk);
      const dv = n2(p.headDim, 128);
      const dk = n2(p.keyHeadDim, dv);
      const taps = Math.max(0, n2(p.convKernel, 4) - 1);
      return (Hv * dk * dv + taps * (2 * Hk * dk + Hv * dv)) * bytesPerValue;
    }
    default:
      return 0;
  }
}
function recurrentStateBytesForModel(model, bytesPerValue = 2) {
  return model.components.reduce((s, c) => s + recurrentStatePerSequenceFor(c, bytesPerValue), 0);
}

// src/utils/quantizationPreview.ts
var BYTES_PER_PARAM = { fp32: 4, fp16: 2, int8: 1, int4: 0.5 };
var SENSITIVE_TYPES = /* @__PURE__ */ new Set([
  "batchNorm",
  "layerNorm",
  "rmsNorm",
  "instanceNorm",
  "groupNorm",
  "softmax",
  "sigmoid"
]);
var IO_TYPES = /* @__PURE__ */ new Set(["input", "output"]);
function computeQuantPreview(model) {
  const layers = [];
  let totalParams = 0;
  let int8SafeParams = 0;
  const components = model.components;
  const isFirst = (idx) => idx === 0 || components[idx - 1]?.type === "input";
  const isLast = (idx) => idx === components.length - 1 || components[idx + 1]?.type === "output";
  components.forEach((comp, idx) => {
    const params = estimateLayerParams(comp.type, comp.params, comp.inputShape);
    totalParams += params;
    const isSensitive = SENSITIVE_TYPES.has(comp.type);
    const isIO = IO_TYPES.has(comp.type);
    const isEdge = isFirst(idx) || isLast(idx);
    let sensitivity = 0;
    let sensitivityNote;
    let quantizable = true;
    if (isIO) {
      sensitivity = 1;
      sensitivityNote = "I/O layers, keep FP32 for numerical stability";
      quantizable = false;
    } else if (isSensitive) {
      sensitivity = 2;
      sensitivityNote = "Normalization/softmax layers are numerically sensitive";
      quantizable = false;
    } else if (isEdge) {
      sensitivity = 1;
      sensitivityNote = "First/last layers, FP16 recommended, INT8 may reduce accuracy";
    }
    if (quantizable && sensitivity === 0) int8SafeParams += params;
    const bytes = {
      fp32: params * BYTES_PER_PARAM.fp32,
      fp16: params * BYTES_PER_PARAM.fp16,
      int8: params * BYTES_PER_PARAM.int8,
      int4: params * BYTES_PER_PARAM.int4
    };
    layers.push({ id: comp.id, name: comp.name || comp.type, type: comp.type, params, bytes, quantizable, sensitivityNote, sensitivity });
  });
  const totalBytes = { fp32: 0, fp16: 0, int8: 0, int4: 0 };
  for (const p of ["fp32", "fp16", "int8", "int4"]) {
    totalBytes[p] = layers.reduce((s, l) => s + l.bytes[p], 0);
  }
  return {
    layers,
    totalParams,
    totalBytes,
    int8SafeFraction: totalParams > 0 ? int8SafeParams / totalParams : 0
  };
}

// src/utils/servingPlanner.ts
function planServing(inp) {
  const batch = Math.max(1, inp.batchSize);
  const avgContext = Math.max(1, inp.promptTokens + inp.genTokens / 2);
  const kvPerRequest = inp.kvBytesPerToken * avgContext + Math.max(0, inp.stateBytesPerRequest ?? 0);
  const prefillSec = prefillSeconds(inp.promptTokens, inp.flopsPerToken, inp.gpu);
  const est = decodeStepEstimate({
    gpu: inp.gpu,
    batchSize: batch,
    weightBytes: inp.weightBytes,
    kvBytesPerRequest: kvPerRequest,
    flopsPerToken: inp.flopsPerToken
  });
  const decodePerReq = inp.genTokens * est.stepSeconds;
  const decodeQps = decodePerReq > 0 ? batch / decodePerReq : 0;
  const prefillQps = prefillSec > 0 ? 1 / prefillSec : 0;
  const ratio = prefillQps > 0 ? decodeQps / prefillQps : 0;
  const usable = inp.gpu.memoryGB * 1e9 * 0.9;
  const fits = inp.weightBytes + kvPerRequest * batch <= usable;
  let fleet;
  if (inp.targetQps && inp.targetQps > 0 && decodeQps > 0 && prefillQps > 0) {
    const decodeGpus = Math.ceil(inp.targetQps / decodeQps);
    const prefillGpus = Math.ceil(inp.targetQps / prefillQps);
    fleet = { prefillGpus, decodeGpus, totalGpus: prefillGpus + decodeGpus };
  }
  let costPerMTokens;
  if (inp.hourlyUsd && inp.hourlyUsd > 0 && est.tokensPerSecond > 0) {
    const gpusPerUnit = 1 + ratio;
    const dollarsPerSec = inp.hourlyUsd * gpusPerUnit / 3600;
    costPerMTokens = dollarsPerSec / est.tokensPerSecond * 1e6;
  }
  return {
    prefillSecondsPerRequest: prefillSec,
    decodeStepSeconds: est.stepSeconds,
    decodeSecondsPerRequest: decodePerReq,
    decodeQpsPerReplica: decodeQps,
    prefillQpsPerGpu: prefillQps,
    prefillToDecodeRatio: ratio,
    fits,
    boundBy: est.boundBy,
    fleet,
    costPerMTokens
  };
}

// src/utils/editChecks.ts
var EDIT_RULE_IDS = [
  "vocab-resize-incomplete",
  "weight-tying-lost",
  "attention-config-drift",
  "rope-context-drift",
  "lora-target-no-match",
  "lora-target-partial",
  "lora-target-wrapper"
];
var EDIT_RULE_SEEN_IN = {
  "vocab-resize-incomplete": [
    "https://github.com/huggingface/transformers/issues/45276",
    "https://github.com/huggingface/transformers/issues/39921"
  ],
  "weight-tying-lost": [
    "https://github.com/huggingface/transformers/issues/38160",
    "https://github.com/huggingface/peft/issues/2944"
  ],
  "attention-config-drift": [],
  "rope-context-drift": [
    "https://huggingface.co/Qwen/Qwen3-30B-A3B/discussions/7"
  ],
  "lora-target-no-match": ["https://github.com/huggingface/peft/issues/3533"],
  "lora-target-partial": ["https://github.com/huggingface/peft/issues/3533"],
  "lora-target-wrapper": ["https://github.com/huggingface/peft/issues/3129"]
};
var num4 = (v) => typeof v === "number" && Number.isFinite(v) ? v : null;
var plural = (n2, w) => `${n2} ${w}${n2 === 1 ? "" : "s"}`;
var list = (xs, max = 6) => xs.slice(0, max).join(", ") + (xs.length > max ? `, and ${xs.length - max} more` : "");
var VOCAB_ROW_TYPES = /* @__PURE__ */ new Set(["embedding", "embeddingBag", "lmHead"]);
function vocabField(c) {
  const p = c.params ?? {};
  if (!VOCAB_ROW_TYPES.has(c.type)) return null;
  for (const k of ["vocabSize", "numEmbeddings"]) {
    const v = num4(p[k]);
    if (v != null) return [k, v];
  }
  return null;
}
function linearOut(c) {
  return c.type === "linear" ? num4((c.params ?? {}).outFeatures) : null;
}
function byName(m) {
  const out = /* @__PURE__ */ new Map();
  for (const c of m.components ?? []) if (c && typeof c.name === "string") out.set(c.name, c);
  return out;
}
function roleOf(c) {
  const n2 = c.name.toLowerCase();
  if (/per_?layer/.test(n2)) return "per-layer embedding";
  if (c.type === "lmHead" || c.type === "linear" || /head|logit/.test(n2)) return "output head";
  return "embedding";
}
function inputEmbedding(m) {
  return (m.components ?? []).find((c) => (c.type === "embedding" || c.type === "embeddingBag") && vocabField(c) != null) ?? null;
}
function embedDimOf(c) {
  const p = c.params ?? {};
  return num4(p.embeddingDim) ?? num4(p.embedDim);
}
function checkGraphEdit(base, head) {
  const out = [];
  const b = byName(base);
  const h = byName(head);
  const resized = [];
  for (const [name, hc] of h) {
    const bc = b.get(name);
    if (!bc) continue;
    const bf = vocabField(bc);
    const hf = vocabField(hc);
    if (bf && hf && bf[1] !== hf[1]) resized.push({ name, from: bf[1], to: hf[1] });
  }
  const baseVocabs = new Set((base.components ?? []).map(vocabField).filter((x) => !!x).map((x) => x[1]));
  for (const [name, hc] of h) {
    const bc = b.get(name);
    const bo = bc ? linearOut(bc) : null;
    const ho = linearOut(hc);
    if (bo != null && ho != null && bo !== ho && baseVocabs.has(bo)) resized.push({ name, from: bo, to: ho });
  }
  const resizedNames = new Set(resized.map((r) => r.name));
  const byFrom = /* @__PURE__ */ new Map();
  for (const r of resized) if (!byFrom.has(r.from)) byFrom.set(r.from, r.to);
  for (const [from, to] of byFrom) {
    const stale = [];
    for (const [name, hc] of h) {
      if (resizedNames.has(name)) continue;
      const bc = b.get(name);
      if (!bc) continue;
      const bv = vocabField(bc)?.[1] ?? linearOut(bc);
      const hv = vocabField(hc)?.[1] ?? linearOut(hc);
      if (bv === from && hv === from) stale.push(hc);
    }
    if (!stale.length) continue;
    const names = stale.map((c) => `${c.name} (${roleOf(c)})`);
    out.push({
      rule: "vocab-resize-incomplete",
      severity: "block",
      title: `vocabulary resized ${from} -> ${to}, but ${plural(stale.length, "token-indexed tensor")} kept ${from} rows: ${list(names)}`,
      detail: `${list(resized.filter((r) => r.from === from).map((r) => r.name))} moved to ${to}. Token ids ${from} to ${to - 1} index past the end of every tensor listed, so the new tokens either fault on lookup or are never produced by the head. Nothing raises at resize or save time. Resize these with the input embedding, or resize through a path that knows about them.`,
      subjects: stale.map((c) => c.name)
    });
  }
  const untied = [];
  for (const [name, bc] of b) {
    if ((bc.params ?? {}).tied !== true) continue;
    const hc = h.get(name);
    if (hc && (hc.params ?? {}).tied !== true) untied.push(name);
  }
  if (untied.length) {
    out.push({
      rule: "weight-tying-lost",
      severity: "warn",
      title: `weight tying removed from ${list(untied)}`,
      detail: "The base shares the output head with the input embedding; the change gives the head its own matrix. That adds vocab x hidden trainable parameters and the two copies drift apart from the first step. If this was a save/reload round trip rather than a decision, the tie flag was lost on save.",
      subjects: untied
    });
  }
  const tiedApart = (m) => {
    const emb = inputEmbedding(m);
    if (!emb) return [];
    const V = vocabField(emb)?.[1];
    const D = embedDimOf(emb);
    const bad = [];
    for (const c of m.components ?? []) {
      if ((c.params ?? {}).tied !== true || c === emb) continue;
      const p = c.params ?? {};
      const out2 = num4(p.outFeatures) ?? num4(p.vocabSize);
      const inn = num4(p.inFeatures) ?? num4(p.hiddenSize) ?? num4(p.embedDim);
      if (out2 != null && V != null && out2 !== V || inn != null && D != null && inn !== D) bad.push(c.name);
    }
    return bad;
  };
  const baseApart = new Set(tiedApart(base));
  const apart = tiedApart(head).filter((n2) => !baseApart.has(n2));
  if (apart.length) {
    const emb = inputEmbedding(head);
    out.push({
      rule: "weight-tying-lost",
      severity: "warn",
      title: `${list(apart)} is declared tied to ${emb.name} but no longer has its shape`,
      detail: `${emb.name} is ${vocabField(emb)?.[1]} x ${embedDimOf(emb) ?? "?"}. A tie between two matrices of different shapes cannot hold: depending on the loader the tie is skipped, or the head is re-pointed at the embedding and the edit to the head is discarded.`,
      subjects: apart
    });
  }
  const baseTied = (base.components ?? []).filter((c) => (c.params ?? {}).tied === true && !h.has(c.name));
  if (baseTied.length && !untied.length) {
    const V = inputEmbedding(head) ? vocabField(inputEmbedding(head))?.[1] : null;
    const added = (head.components ?? []).filter((c) => !b.has(c.name) && (c.params ?? {}).tied !== true && (c.type === "linear" && num4((c.params ?? {}).outFeatures) === V || c.type === "lmHead"));
    if (V != null && added.length) {
      out.push({
        rule: "weight-tying-lost",
        severity: "warn",
        title: `tied head ${list(baseTied.map((c) => c.name))} replaced by untied ${list(added.map((c) => c.name))}`,
        detail: "The new head projects to the vocabulary with its own weights; the base shared them with the input embedding.",
        subjects: added.map((c) => c.name)
      });
    }
  }
  return out;
}
var NESTED = ["text_config", "llm_config", "language_config", "language_model_config", "decoder_config"];
function isObj(v) {
  return !!v && typeof v === "object" && !Array.isArray(v);
}
var VOCAB_KEYS = ["vocab_size", "vocab_size_per_layer_input", "padded_vocab_size", "n_vocab"];
var VOCAB_TENSOR = {
  vocab_size: "embed_tokens / lm_head",
  vocab_size_per_layer_input: "embed_tokens_per_layer",
  padded_vocab_size: "embed_tokens / lm_head (padded)",
  n_vocab: "embed_tokens / lm_head"
};
function vocabPaths(c) {
  const out = /* @__PURE__ */ new Map();
  for (const k of VOCAB_KEYS) {
    const v = num4(c[k]);
    if (v != null) out.set(k, v);
  }
  for (const s of NESTED) {
    const sub = c[s];
    if (!isObj(sub)) continue;
    for (const k of VOCAB_KEYS) {
      const v = num4(sub[k]);
      if (v != null) out.set(`${s}.${k}`, v);
    }
  }
  return out;
}
function textKey(c, k) {
  for (const s of NESTED) {
    const sub = c[s];
    if (isObj(sub) && sub[k] !== void 0 && sub[k] !== null) return sub[k];
  }
  return c[k];
}
function ropeFactor(c) {
  const found = [];
  const scan = (v, depth) => {
    if (!isObj(v) || depth > 2) return;
    const f = num4(v.factor);
    if (f != null) found.push(f);
    for (const x of Object.values(v)) scan(x, depth + 1);
  };
  scan(textKey(c, "rope_scaling"), 0);
  scan(textKey(c, "rope_parameters"), 0);
  return found.length ? Math.max(...found) : null;
}
function checkConfigEdit(baseIn, headIn) {
  if (!isObj(baseIn) || !isObj(headIn)) return [];
  const base = baseIn;
  const head = headIn;
  const out = [];
  const bv = vocabPaths(base);
  const hv = vocabPaths(head);
  const moved = [...hv].filter(([k, v]) => bv.has(k) && bv.get(k) !== v).map(([k, v]) => ({ k, from: bv.get(k), to: v }));
  const fromTo = /* @__PURE__ */ new Map();
  for (const m of moved) if (!fromTo.has(m.from)) fromTo.set(m.from, m.to);
  for (const [from, to] of fromTo) {
    const stale = [...hv].filter(([k, v]) => bv.get(k) === from && v === from).map(([k]) => k);
    if (!stale.length) continue;
    const leaf = (k) => k.split(".").pop();
    out.push({
      rule: "vocab-resize-incomplete",
      severity: "block",
      title: `${list(moved.filter((m) => m.from === from).map((m) => m.k))} moved ${from} -> ${to}, but ${list(stale)} stayed ${from}`,
      detail: `${list(stale.map((k) => `${k} sizes ${VOCAB_TENSOR[leaf(k)] ?? "a token-indexed tensor"}`))}. A token id at or above ${from} indexes past the end of it. resize_token_embeddings does not touch every vocabulary-sized tensor on every architecture (Gemma 3n and Gemma 4 per-layer embeddings are the known case), and the save writes whatever it was left with, so the config disagrees with itself and nothing raises.`,
      subjects: stale
    });
  }
  const tiePaths = ["tie_word_embeddings", ...NESTED.map((s) => `${s}.tie_word_embeddings`)];
  const at = (c, p) => {
    const [a, k] = p.includes(".") ? p.split(".") : [null, p];
    return a ? isObj(c[a]) ? c[a][k] : void 0 : c[k];
  };
  const wasTied = tiePaths.filter((p) => at(base, p) === true);
  if (wasTied.length) {
    const flipped = wasTied.filter((p) => at(head, p) === false);
    const dropped = wasTied.filter((p) => at(head, p) === void 0);
    const stillTied = tiePaths.some((p) => at(head, p) === true);
    if (flipped.length) {
      out.push({
        rule: "weight-tying-lost",
        severity: "warn",
        title: `${list(flipped)} changed true -> false: the output head is no longer the input embedding`,
        detail: "On reload the head gets its own vocab x hidden matrix, initialised from whatever the checkpoint stored for it (or randomly when it stored none), and training moves the two apart. If this came from save_pretrained on a model that was tied, the flag was lost on save.",
        subjects: flipped
      });
    } else if (dropped.length && !stillTied) {
      out.push({
        rule: "weight-tying-lost",
        severity: "warn",
        title: `${list(dropped)} was true and is absent from the changed config`,
        detail: "Absent means the config class default decides, and that default is not the value the base declared on every class. A save that drops the flag (transformers#38160) reloads untied with no error.",
        subjects: dropped
      });
    }
  }
  const g = (c, k) => num4(textKey(c, k));
  const H0 = g(base, "num_attention_heads");
  const H1 = g(head, "num_attention_heads");
  const K0 = g(base, "num_key_value_heads");
  const K1 = g(head, "num_key_value_heads");
  const D0 = g(base, "hidden_size");
  const D1 = g(head, "hidden_size");
  const d0 = g(base, "head_dim");
  const d1 = g(head, "head_dim");
  const changedAttn = H0 !== H1 || K0 !== K1 || D0 !== D1 || d0 !== d1;
  if (changedAttn && H1 != null) {
    const baseGqaOk = H0 == null || K0 == null || K0 > 0 && H0 % K0 === 0;
    if (K1 != null && (K1 <= 0 || H1 % K1 !== 0) && baseGqaOk) {
      out.push({
        rule: "gqa-head-divisibility",
        severity: "block",
        title: `num_attention_heads ${H1} is not a multiple of num_key_value_heads ${K1}`,
        detail: `Each key/value head serves num_attention_heads / num_key_value_heads query heads; ${H1} / ${K1} is not a whole number, so the attention module cannot be built. The base had ${H0} / ${K0}.`,
        subjects: ["num_attention_heads", "num_key_value_heads"]
      });
    }
    const baseDivOk = d0 != null || H0 == null || D0 == null || D0 % H0 === 0;
    if (d1 == null && D1 != null && H1 > 0 && D1 % H1 !== 0 && baseDivOk) {
      out.push({
        rule: "head-dim-divisibility",
        severity: "block",
        title: `hidden_size ${D1} is not divisible by num_attention_heads ${H1}, and no head_dim is declared`,
        detail: `With no head_dim the head width is hidden_size / num_attention_heads = ${(D1 / H1).toFixed(2)}, which is not an integer.`,
        subjects: ["hidden_size", "num_attention_heads"]
      });
    }
    if (d0 != null && d1 === d0 && H0 != null && D0 != null && H0 * d0 === D0 && D1 != null && H1 * d1 !== D1) {
      const moved2 = [D0 !== D1 ? `hidden_size ${D0} -> ${D1}` : null, H0 !== H1 ? `num_attention_heads ${H0} -> ${H1}` : null].filter(Boolean);
      out.push({
        rule: "attention-config-drift",
        severity: "warn",
        title: `head_dim stayed ${d1} while ${moved2.join(" and ")}: attention is now ${H1} x ${d1} = ${H1 * d1} wide against hidden_size ${D1}`,
        detail: `In the base, head_dim was exactly hidden_size / num_attention_heads (${D0} / ${H0}). An explicit head_dim is honoured as written, so q_proj becomes ${D1} -> ${H1 * d1} and o_proj ${H1 * d1} -> ${D1}. That builds and runs; it is a different model from the one the edit described. Set head_dim to ${Number.isInteger(D1 / H1) ? D1 / H1 : "hidden_size / num_attention_heads"} or remove it.`,
        subjects: ["head_dim", ...D0 !== D1 ? ["hidden_size"] : [], ...H0 !== H1 ? ["num_attention_heads"] : []]
      });
    }
    if (H0 != null && H0 !== H1 && K0 != null && K1 === K0 && K1 > 0 && H1 % K1 === 0) {
      const from = K0 === H0 ? "multi-head attention" : `groups of ${H0 / K0}`;
      const to = K1 === H1 ? "multi-head attention" : `groups of ${H1 / K1}`;
      out.push({
        rule: "attention-config-drift",
        severity: "warn",
        title: `num_attention_heads moved ${H0} -> ${H1} and num_key_value_heads stayed ${K1}: ${from} became ${to}`,
        detail: `k_proj and v_proj keep ${K1} heads, so each key/value head now serves ${H1 / K1} query heads instead of ${H0 / K0}. The KV cache and the pretrained k/v weights are sized for the old split. If the head count was meant to scale, scale num_key_value_heads with it.`,
        subjects: ["num_attention_heads", "num_key_value_heads"]
      });
    }
  }
  const M0 = g(base, "max_position_embeddings");
  const M1 = g(head, "max_position_embeddings");
  const f0 = ropeFactor(base);
  const f1 = ropeFactor(head);
  if (M0 != null && M1 != null) {
    if (M1 > M0 && f0 === f1) {
      out.push({
        rule: "rope-context-drift",
        severity: "warn",
        title: `max_position_embeddings raised ${M0} -> ${M1} with ${f1 == null ? "no RoPE scaling" : `the RoPE scaling factor unchanged at ${f1}`}`,
        detail: `The config now admits ${M1}-token inputs, and nothing changed how positions past ${M0} are encoded. The model runs at those lengths and its positions beyond the trained range are extrapolated rather than interpolated. Raise the scaling factor by ${(M1 / M0).toFixed(2)}x with the length, or leave the length where it was.`,
        subjects: ["max_position_embeddings", "rope_scaling"]
      });
    } else if (f0 !== f1 && M0 === M1 && f1 != null) {
      out.push({
        rule: "rope-context-drift",
        severity: "warn",
        title: `RoPE scaling factor moved ${f0 ?? "none"} -> ${f1} and max_position_embeddings stayed ${M1}`,
        detail: "Every position is now encoded for the scaled range, and the declared context is still the old one. Runtimes size the window from max_position_embeddings (vLLM's default max_model_len), so the extension is never used while short inputs pay for it.",
        subjects: ["rope_scaling", "max_position_embeddings"]
      });
    } else if (f0 != null && f1 != null && f0 !== f1 && M0 !== M1) {
      const rm = M1 / M0;
      const rf = f1 / f0;
      if (Math.abs(rm / rf - 1) > 0.01) {
        out.push({
          rule: "rope-context-drift",
          severity: "warn",
          title: `max_position_embeddings moved x${rm.toFixed(2)} (${M0} -> ${M1}) and the RoPE factor moved x${rf.toFixed(2)} (${f0} -> ${f1})`,
          detail: "The two were in step in the base and are not now: one of them describes a context the other does not.",
          subjects: ["max_position_embeddings", "rope_scaling"]
        });
      }
    }
  }
  return out;
}
function looksLikeHFConfig(v) {
  if (!isObj(v)) return false;
  if (typeof v.model_type === "string" || Array.isArray(v.architectures)) return true;
  return NESTED.some((s) => isObj(v[s]) && typeof v[s].model_type === "string");
}
var PEFT_SUPPORTED = /* @__PURE__ */ new Set(["linear", "lmHead", "embedding", "conv1d", "conv2d", "conv3d"]);
var ATTN_ROLES = /* @__PURE__ */ new Set(["q_proj", "k_proj", "v_proj", "o_proj"]);
var SKIP_TYPES = /* @__PURE__ */ new Set(["input", "output", "stickyNote"]);
var CHECKPOINT_ALIASES = {
  query: ["q_proj"],
  wq: ["q_proj"],
  q: ["q_proj"],
  key: ["k_proj"],
  wk: ["k_proj"],
  k: ["k_proj"],
  value: ["v_proj"],
  wv: ["v_proj"],
  v: ["v_proj"],
  out_proj: ["o_proj"],
  wo: ["o_proj", "fc2", "down_proj"],
  o: ["o_proj"],
  dense: ["o_proj", "fc2"],
  c_proj: ["o_proj", "fc2"],
  c_attn: ["q_proj", "k_proj", "v_proj"],
  query_key_value: ["q_proj", "k_proj", "v_proj"],
  qkv: ["q_proj", "k_proj", "v_proj"],
  qkv_proj: ["q_proj", "k_proj", "v_proj"],
  wqkv: ["q_proj", "k_proj", "v_proj"],
  w_pack: ["q_proj", "k_proj", "v_proj"],
  in_proj: ["q_proj", "k_proj", "v_proj"],
  w1: ["gate_proj", "fc1"],
  w3: ["up_proj"],
  w2: ["down_proj", "fc2"],
  gate_up_proj: ["gate_proj", "up_proj"],
  c_fc: ["fc1"],
  wi: ["fc1", "up_proj"],
  dense_h_to_4h: ["fc1"],
  dense_4h_to_h: ["fc2"],
  fc_in: ["fc1"],
  fc_out: ["fc2"]
};
var leafOf = (s) => s.split(".").pop() ?? s;
function checkLoraTargets(model, targetsIn, inventory) {
  const comps = (model.components ?? []).filter((c) => c && !SKIP_TYPES.has(c.type));
  const byNameMap = new Map(comps.map((c) => [c.name, c]));
  const path = (c, fallback) => c?.scope && c.scope.length ? c.scope : c?.name ?? fallback;
  const candidates = [];
  const opened = /* @__PURE__ */ new Set();
  for (const m of inventory.modules) {
    const c = byNameMap.get(m.layer);
    const own = m.name === m.layer;
    const key = own ? path(c, m.name) : `${path(c, m.layer)}.${m.module}`;
    candidates.push({ key, leaf: m.module, type: own ? m.type : "linear", supported: true, module: m, invented: !own });
    if (own) opened.add(m.layer);
  }
  for (const c of comps) {
    if (opened.has(c.name)) continue;
    if ((c.type === "linear" || c.type === "lmHead") && (c.params ?? {}).tied === true) continue;
    const key = path(c, c.name);
    candidates.push({ key, leaf: leafOf(key), type: c.type, supported: PEFT_SUPPORTED.has(c.type), invented: false });
  }
  const findings = [];
  const perTarget = [];
  let hits = [];
  const shown = Array.isArray(targetsIn) ? JSON.stringify(targetsIn) : JSON.stringify(targetsIn);
  const isAllLinear = (t) => t.toLowerCase() === "all-linear";
  const entryMatches = (t, c) => {
    if (c.key === t || c.key.endsWith(`.${t}`)) return true;
    if (!c.invented) return false;
    const lt = leafOf(t);
    return c.leaf === lt || (CHECKPOINT_ALIASES[lt.toLowerCase()] ?? []).includes(c.leaf);
  };
  const allLinear = () => candidates.filter((c) => c.module && c.module.role !== "head");
  if (typeof targetsIn === "string") {
    if (isAllLinear(targetsIn)) hits = allLinear();
    else {
      let re = null;
      try {
        re = new RegExp(`^(?:${targetsIn})$`);
      } catch {
        re = null;
      }
      if (!re) {
        findings.push({
          rule: "lora-target-no-match",
          severity: "block",
          title: `target_modules ${shown} is not a valid regular expression`,
          detail: "PEFT reads a string target_modules as a regular expression matched against each full module name.",
          subjects: [targetsIn]
        });
      } else {
        const r = re;
        hits = candidates.filter((c) => r.test(c.key) || c.invented && r.test(c.leaf));
      }
    }
  } else {
    const seen = /* @__PURE__ */ new Set();
    for (const t of targetsIn) {
      const got = isAllLinear(t) ? allLinear() : candidates.filter((c) => entryMatches(t, c));
      perTarget.push({ target: t, count: got.length });
      for (const c of got) seen.add(c);
    }
    hits = [...seen];
  }
  const matched = hits.filter((c) => c.supported);
  const wrappers = hits.filter((c) => !c.supported);
  const linearLeaves = [...new Set(candidates.filter((c) => c.supported && c.module).map((c) => c.leaf))];
  if (!findings.length && matched.length === 0) {
    findings.push({
      rule: "lora-target-no-match",
      severity: "block",
      title: `LoRA target_modules ${shown} match${Array.isArray(targetsIn) && targetsIn.length !== 1 ? "" : "es"} no Linear module in the graph${wrappers.length ? ` (only ${plural(wrappers.length, "non-Linear module")})` : ""}`,
      detail: "An adapter with nothing to attach to trains zero parameters. A first adapter like this raises in PEFT; a second one on the same model is accepted silently (peft#3533). " + (linearLeaves.length ? `Linear module names in this graph: ${list(linearLeaves, 12)}.` : "This graph has no Linear module to target."),
      subjects: Array.isArray(targetsIn) ? targetsIn : [targetsIn]
    });
  }
  if (matched.length) {
    const dead = perTarget.filter((p) => p.count === 0).map((p) => p.target);
    if (dead.length) {
      findings.push({
        rule: "lora-target-partial",
        severity: "warn",
        title: `${plural(dead.length, "target")} matched nothing: ${list(dead)}`,
        detail: `The adapter attaches to the other entries and says nothing about these. Linear module names here: ${list(linearLeaves, 12)}.`,
        subjects: dead
      });
    }
    const partialRoles = [];
    for (const role of ATTN_ROLES) {
      const total = candidates.filter((c) => c.module?.role === role).length;
      const got = matched.filter((c) => c.module?.role === role).length;
      if (got > 0 && got < total) partialRoles.push(`${role} in ${got} of ${total} layers`);
    }
    if (partialRoles.length) {
      findings.push({
        rule: "lora-target-partial",
        severity: "warn",
        title: `the targets reach only part of the attention stack: ${partialRoles.join("; ")}`,
        detail: "The pattern names a projection in some layers and misses it in the rest, so the adapter changes some layers and leaves the others exactly as pretrained. If the layer range was deliberate, ignore this.",
        subjects: partialRoles
      });
    }
  }
  if (wrappers.length) {
    const w = wrappers.map((c) => `${c.key} (${c.type})`);
    findings.push({
      rule: "lora-target-wrapper",
      severity: "warn",
      title: `target_modules hit ${plural(wrappers.length, "module")} that ${wrappers.length === 1 ? "is" : "are"} not a Linear: ${list(w)}`,
      detail: "PEFT injects LoRA into nn.Linear, nn.Embedding, nn.Conv1d/2d/3d and transformers Conv1D. A class that holds a Linear without being one (Gemma4ClippableLinear, peft#3129), a whole attention block, or a norm is refused or skipped. Target the Linear inside it by its own name.",
      subjects: wrappers.map((c) => c.key)
    });
  }
  return {
    targets: targetsIn,
    matched: matched.map((c) => ({ name: c.key, type: c.type })),
    perTarget,
    wrappers: wrappers.map((c) => ({ name: c.key, type: c.type })),
    byConvention: !(typeof targetsIn === "string" && isAllLinear(targetsIn)) && hits.some((c) => c.invented),
    findings
  };
}
function renderEditFindingsMarkdown(findings, heading = "**Edits that run and are wrong**") {
  if (!findings.length) return "";
  const L = heading ? [heading] : [];
  for (const f of findings) L.push(`- ${f.severity === "block" ? "**block**" : "warn"} \`${f.rule}\`: ${f.title}. ${f.detail}`);
  return L.join("\n");
}

// src/utils/changeSpec.ts
var PEFT_KINDS = ["lora", "qlora", "dora"];
var QUANT_PRECISIONS = ["fp32", "fp16", "int8", "int4"];
function fnv1a2(str2) {
  let h = 2166136261;
  for (let i = 0; i < str2.length; i++) {
    h ^= str2.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}
var MAX_BASE_LEN = 200;
var MAX_TARGETS = 32;
function readChangeSpec(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) return null;
  const raw = input;
  const problems = [];
  const spec = { base: null, adapter: null, quant: null, serving: null, constraints: {} };
  if (raw.base != null) {
    if (typeof raw.base === "string") spec.base = raw.base.slice(0, MAX_BASE_LEN);
    else problems.push("base: expected a string");
  }
  if (raw.adapter != null) {
    const a = raw.adapter;
    if (typeof a !== "object" || Array.isArray(a)) problems.push("adapter: expected an object");
    else {
      const peft = typeof a.peft === "string" ? a.peft.toLowerCase() : "lora";
      const rankRaw = a.rank == null ? 16 : typeof a.rank === "string" ? Number(a.rank) : a.rank;
      const pattern = typeof a.targets === "string" && a.targets.length > 0 && a.targets.toLowerCase() !== "all-linear" ? a.targets.slice(0, MAX_BASE_LEN) : void 0;
      const targets = Array.isArray(a.targets) ? a.targets.filter((t) => typeof t === "string" && t.length > 0).slice(0, MAX_TARGETS) : typeof a.targets === "string" && a.targets.toLowerCase() === "all-linear" ? ["all-linear"] : [];
      if (!PEFT_KINDS.includes(peft)) problems.push(`adapter.peft: expected one of ${PEFT_KINDS.join(", ")}, got ${JSON.stringify(a.peft)}`);
      else if (typeof rankRaw !== "number" || !Number.isInteger(rankRaw) || rankRaw <= 0) problems.push(`adapter.rank: expected a positive integer, got ${JSON.stringify(a.rank)}`);
      else {
        if (a.targets != null && !Array.isArray(a.targets) && typeof a.targets !== "string") problems.push("adapter.targets: expected a list of module names or one pattern");
        spec.adapter = { peft, rank: rankRaw, targets, ...pattern ? { pattern } : {} };
      }
    }
  }
  if (raw.quant != null) {
    const q = raw.quant;
    const precision = typeof q === "object" && q && !Array.isArray(q) ? q.precision : typeof q === "string" ? q : void 0;
    const p = typeof precision === "string" ? precision.toLowerCase() : precision;
    if (typeof p === "string" && QUANT_PRECISIONS.includes(p)) spec.quant = { precision: p };
    else problems.push(`quant.precision: expected one of ${QUANT_PRECISIONS.join(", ")}, got ${JSON.stringify(precision)}`);
  }
  if (raw.serving != null) {
    const s = raw.serving;
    if (typeof s !== "object" || Array.isArray(s)) problems.push("serving: expected an object");
    else if (typeof s.gpu !== "string" || !s.gpu) problems.push("serving.gpu: expected a GPU id like a100-40g");
    else {
      const int = (k, dflt) => {
        const v = s[k];
        if (v == null) return dflt;
        const n2 = typeof v === "string" ? Number(v) : v;
        if (typeof n2 !== "number" || !Number.isFinite(n2) || n2 <= 0) {
          problems.push(`serving.${k}: expected a positive number, got ${JSON.stringify(v)}`);
          return dflt;
        }
        return Math.round(n2);
      };
      spec.serving = { gpu: s.gpu, promptTokens: int("promptTokens", 1024), genTokens: int("genTokens", 256), batchSize: int("batchSize", 8) };
    }
  }
  if (raw.constraints != null) {
    const c = raw.constraints;
    if (typeof c !== "object" || Array.isArray(c)) problems.push("constraints: expected an object");
    else {
      if (c.must_fit != null) {
        if (typeof c.must_fit === "string") spec.constraints.must_fit = c.must_fit;
        else problems.push("constraints.must_fit: expected a GPU name like A100-40GB");
      }
      for (const k of ["max_memory_gb", "max_cost_usd"]) {
        const v = c[k];
        if (v == null) continue;
        const n2 = typeof v === "string" ? Number(v) : v;
        if (typeof n2 !== "number" || !Number.isFinite(n2) || n2 < 0) problems.push(`constraints.${k}: expected a number, got ${JSON.stringify(v)}`);
        else spec.constraints[k] = n2;
      }
      if (c.allow_gpu != null) {
        if (typeof c.allow_gpu === "boolean") spec.constraints.allow_gpu = c.allow_gpu;
        else problems.push("constraints.allow_gpu: expected true or false");
      }
    }
  }
  return { spec, problems };
}
function canonical(v) {
  if (Array.isArray(v)) return `[${v.map(canonical).join(",")}]`;
  if (v && typeof v === "object") {
    const o = v;
    return `{${Object.keys(o).sort().filter((k) => o[k] !== void 0).map((k) => `${JSON.stringify(k)}:${canonical(o[k])}`).join(",")}}`;
  }
  return JSON.stringify(v);
}
function changeSpecId(spec, baseFingerprint) {
  return fnv1a2(`${baseFingerprint}#${canonical(spec)}`);
}
function resolveGpu(name) {
  const n2 = name.trim().toLowerCase();
  if (!n2) return null;
  const direct = gpuById(n2);
  if (direct) return direct;
  const squash = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const key = squash(n2);
  const byLabel = GPU_HARDWARE.find((g) => squash(g.label).startsWith(key) || squash(g.id) === key);
  if (byLabel) return byLabel;
  if (/^a100(40)?(gb)?$/.test(key)) return gpuById("a100-40g") ?? null;
  if (/^a10080(gb)?$/.test(key)) return gpuById("a100-80g") ?? null;
  if (/^h100/.test(key)) return gpuById("h100-sxm") ?? null;
  return GPU_HARDWARE.find((g) => squash(g.id).startsWith(key)) ?? null;
}
function workloadPhrase(batchSize, seqLen) {
  return `at batch ${batchSize}${seqLen != null ? `, ${seqLen} tokens` : ""}`;
}
var CHANGE_HEADROOM = 1.15;
var HEADROOM = CHANGE_HEADROOM;
function evaluateChangeSpec(model, spec, problems, ctx) {
  const blockers = [];
  const warnings = [];
  for (const p of problems) warnings.push({ title: `change: unreadable line (${p})`, stage: "change" });
  let trainableParams = ctx.params;
  let trainMemGb = null;
  if (ctx.params > 0) {
    const peak = ctx.trainingPeak(spec.adapter);
    if (peak) {
      trainableParams = peak.trainableParams;
      trainMemGb = peak.peakGb;
    }
  }
  const precision = spec.quant?.precision ?? "fp16";
  const weightBytes2 = ctx.params > 0 ? computeQuantPreview(model).totalBytes[precision] : null;
  let adapterMatch = null;
  if (spec.adapter && ctx.lora && (spec.adapter.pattern || spec.adapter.targets.length)) {
    adapterMatch = checkLoraTargets(model, spec.adapter.pattern ?? spec.adapter.targets, ctx.lora);
    for (const f of adapterMatch.findings) {
      (f.severity === "block" ? blockers : warnings).push({ title: `change: ${f.title}`, detail: f.detail, stage: "change", rule: f.rule });
    }
  }
  if (ctx.baseLegal === false) {
    blockers.push({ title: "change: the base does not run, so nothing built on it can be approved", stage: "change" });
  }
  const c = spec.constraints;
  if (c.allow_gpu === false) {
    blockers.push({ title: "change: GPU dispatch is not allowed for this change (allow_gpu: false)", detail: "allow_gpu: false", stage: "change" });
  }
  if (c.must_fit != null) {
    const gpu = resolveGpu(c.must_fit);
    if (!gpu) warnings.push({ title: `change: unknown GPU in must_fit (${c.must_fit})`, detail: `Known: ${GPU_HARDWARE.map((g) => g.id).join(", ")}.`, stage: "change" });
    else if (trainMemGb != null && trainMemGb * HEADROOM > gpu.memoryGB) {
      blockers.push({
        title: `change: must_fit ${gpu.label} violated (needs ${(trainMemGb * HEADROOM).toFixed(1)} GB to train${spec.adapter ? ` with ${spec.adapter.peft} r=${spec.adapter.rank}` : ""} ${workloadPhrase(ctx.batchSize, ctx.seqLen)})`,
        detail: `must_fit: ${c.must_fit}`,
        stage: "change"
      });
    }
  }
  if (c.max_memory_gb != null && trainMemGb != null && trainMemGb * HEADROOM > c.max_memory_gb) {
    warnings.push({ title: `change: max_memory_gb ${c.max_memory_gb} exceeded (${(trainMemGb * HEADROOM).toFixed(1)} GB)`, detail: `max_memory_gb: ${c.max_memory_gb}`, stage: "change" });
  }
  if (c.max_cost_usd != null && ctx.estCostUsd != null && ctx.estCostUsd > c.max_cost_usd) {
    warnings.push({
      title: `change: max_cost_usd ${c.max_cost_usd} exceeded by the plan's quote ($${ctx.estCostUsd.toFixed(2)})`,
      detail: spec.adapter ? "The quote is the plan's full-run estimate; an adapter run is usually cheaper. Stated so the budget line is not silently skipped." : `max_cost_usd: ${c.max_cost_usd}`,
      stage: "change"
    });
  }
  let servingFits = null;
  if (spec.serving && weightBytes2 != null) {
    const gpu = resolveGpu(spec.serving.gpu);
    if (!gpu) warnings.push({ title: `change: unknown serving GPU (${spec.serving.gpu})`, detail: `Known: ${GPU_HARDWARE.map((g) => g.id).join(", ")}.`, stage: "change" });
    else {
      const kvBytesPerToken = kvBytesPerTokenForModel(model, 2, spec.serving.promptTokens + spec.serving.genTokens);
      const sp = planServing({
        gpu,
        weightBytes: weightBytes2,
        kvBytesPerToken,
        stateBytesPerRequest: recurrentStateBytesForModel(model, 2),
        flopsPerToken: 2 * ctx.params,
        promptTokens: spec.serving.promptTokens,
        genTokens: spec.serving.genTokens,
        batchSize: spec.serving.batchSize
      });
      servingFits = sp.fits;
      if (!sp.fits) {
        blockers.push({
          title: `change: serving target ${gpu.label} does not fit (${precision} weights ${(weightBytes2 / 1e9).toFixed(1)} GB plus KV for batch ${spec.serving.batchSize} at ${spec.serving.promptTokens + spec.serving.genTokens} tokens)`,
          detail: `serving: ${spec.serving.gpu}`,
          stage: "change"
        });
      }
    }
  }
  return {
    specId: changeSpecId(spec, ctx.baseFingerprint),
    baseFingerprint: ctx.baseFingerprint,
    spec,
    estimates: { trainableParams, trainMemGb, weightBytes: weightBytes2, servingFits, batchSize: ctx.batchSize, seqLen: ctx.seqLen ?? null },
    gate: { ok: blockers.length === 0, blockers, warnings },
    adapterMatch,
    problems
  };
}
function describeChangeSpec(spec) {
  const bits = [];
  if (spec.adapter) bits.push(`${spec.adapter.peft.toUpperCase()} r=${spec.adapter.rank}${spec.adapter.pattern ? ` on /${spec.adapter.pattern}/` : spec.adapter.targets.length ? ` on ${spec.adapter.targets.join(",")}` : ""}`);
  else bits.push("full weights");
  if (spec.quant) bits.push(spec.quant.precision);
  if (spec.serving) bits.push(`serve ${spec.serving.gpu} ${spec.serving.promptTokens + spec.serving.genTokens} tok \xD7 ${spec.serving.batchSize}`);
  const c = spec.constraints;
  const cons = [];
  if (c.must_fit) cons.push(`fit ${c.must_fit}`);
  if (c.max_memory_gb != null) cons.push(`\u2264 ${c.max_memory_gb} GB`);
  if (c.max_cost_usd != null) cons.push(`\u2264 $${c.max_cost_usd}`);
  if (c.allow_gpu === false) cons.push("no GPU");
  if (cons.length) bits.push(cons.join(", "));
  return bits.join(" \xB7 ");
}

// src/utils/comfyWorkflow.ts
var LATENT_PRODUCERS = {
  EmptyLatentImage: { channels: 4 },
  EmptySD3LatentImage: { channels: 16 },
  EmptyHunyuanLatentVideo: { channels: 16 },
  EmptyMochiLatentVideo: { channels: 12 },
  EmptyLTXVLatentVideo: { channels: 128 },
  EmptyCosmosLatentVideo: { channels: 16 },
  WanImageToVideo: { channels: 16, slot: 2 }
};
var LATENT_PASSTHROUGH = {
  LatentUpscale: "samples",
  LatentUpscaleBy: "samples",
  SetLatentNoiseMask: "samples",
  RepeatLatentBatch: "samples",
  LatentFromBatch: "samples",
  LatentFlip: "samples",
  LatentRotate: "samples",
  LatentCrop: "samples",
  KSampler: "latent_image",
  KSamplerAdvanced: "latent_image",
  SamplerCustom: "latent_image",
  SamplerCustomAdvanced: "latent_image"
};
var SAMPLERS = /* @__PURE__ */ new Set(["KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"]);
var MODEL_LOADERS = {
  CheckpointLoaderSimple: "ckpt_name",
  UNETLoader: "unet_name"
};
var FAMILIES = [
  { family: "SD3", channels: 16, re: /sd3|stable[-_ ]?diffusion[-_ ]?3/i },
  { family: "Flux", channels: 16, re: /flux/i },
  { family: "SDXL", channels: 4, re: /sdxl|sd_xl|xl[-_]base|xl[-_]refiner|pony/i },
  { family: "SD1.5", channels: 4, re: /sd[-_ ]?1[-_.]?5|v1[-_]5|sd15/i }
];
var CLASS_CHANNELS = /(\d+)[-_ ]?ch(?:annel)?s?(?![a-z])/i;
function str(v) {
  return typeof v === "string" ? v : typeof v === "number" ? String(v) : null;
}
function readWorkflow(json) {
  if (!json || typeof json !== "object" || Array.isArray(json)) return null;
  const o = json;
  const nodes = /* @__PURE__ */ new Map();
  if (Array.isArray(o.nodes)) {
    const linkSrc = /* @__PURE__ */ new Map();
    for (const l of Array.isArray(o.links) ? o.links : []) {
      if (Array.isArray(l) && l.length >= 3) {
        const id = str(l[0]);
        const from = str(l[1]);
        if (id && from) linkSrc.set(id, [from, Number(l[2]) || 0]);
      } else if (l && typeof l === "object") {
        const r = l;
        const id = str(r.id);
        const from = str(r.origin_id);
        if (id && from) linkSrc.set(id, [from, Number(r.origin_slot) || 0]);
      }
    }
    for (const raw of o.nodes) {
      if (!raw || typeof raw !== "object") continue;
      const n2 = raw;
      const id = str(n2.id);
      const classType = str(n2.type);
      if (!id || !classType) continue;
      const links = /* @__PURE__ */ new Map();
      for (const inp of Array.isArray(n2.inputs) ? n2.inputs : []) {
        if (!inp || typeof inp !== "object") continue;
        const i = inp;
        const name = str(i.name);
        const link = str(i.link);
        const src = link ? linkSrc.get(link) : void 0;
        if (name != null && src) links.set(name, src);
      }
      const widgetList = Array.isArray(n2.widgets_values) ? n2.widgets_values : [];
      nodes.set(id, { id, classType, title: str(n2.title) ?? classType, links, widgets: /* @__PURE__ */ new Map(), widgetList });
    }
    return { format: "ui", nodes };
  }
  const entries = Object.entries(o).filter(([, v]) => v && typeof v === "object" && typeof v.class_type === "string");
  if (!entries.length) return null;
  for (const [id, v] of entries) {
    const n2 = v;
    const classType = n2.class_type;
    const links = /* @__PURE__ */ new Map();
    const widgets = /* @__PURE__ */ new Map();
    const inputs = n2.inputs && typeof n2.inputs === "object" ? n2.inputs : {};
    for (const [name, val] of Object.entries(inputs)) {
      if (Array.isArray(val) && val.length === 2 && str(val[0]) != null && typeof val[1] === "number") links.set(name, [str(val[0]), val[1]]);
      else widgets.set(name, val);
    }
    const meta = n2._meta && typeof n2._meta === "object" ? n2._meta : {};
    nodes.set(id, { id, classType, title: str(meta.title) ?? classType, links, widgets, widgetList: [] });
  }
  return { format: "api", nodes };
}
function weightsName(n2) {
  const key = MODEL_LOADERS[n2.classType];
  if (!key) return null;
  return str(n2.widgets.get(key)) ?? str(n2.widgetList[0]);
}
function familyOf(file) {
  if (!file) return null;
  const hit = FAMILIES.find((f) => f.re.test(file));
  return hit ? { family: hit.family, channels: hit.channels } : null;
}
function latentSource(nodes, start) {
  let cur = start;
  for (let hops = 0; cur && hops < 64; hops++) {
    const n2 = nodes.get(cur[0]);
    if (!n2) return null;
    if (n2.classType === "Reroute") {
      cur = [...n2.links.values()][0];
      continue;
    }
    const prod2 = LATENT_PRODUCERS[n2.classType];
    if (prod2 && (prod2.slot == null || prod2.slot === cur[1])) return { node: n2, channels: prod2.channels, why: `${n2.classType} makes ${prod2.channels}-channel latents` };
    const named = CLASS_CHANNELS.exec(n2.classType);
    if (named) return { node: n2, channels: Number(named[1]), why: `${n2.classType} names ${named[1]} channels` };
    const through = LATENT_PASSTHROUGH[n2.classType];
    if (!through) return null;
    cur = n2.links.get(through);
  }
  return null;
}
function modelSource(nodes, sampler) {
  let cur = sampler.links.get("model") ?? sampler.links.get("guider");
  for (let hops = 0; cur && hops < 64; hops++) {
    const n2 = nodes.get(cur[0]);
    if (!n2) return null;
    if (n2.classType === "Reroute") {
      cur = [...n2.links.values()][0];
      continue;
    }
    if (MODEL_LOADERS[n2.classType]) {
      const file = weightsName(n2);
      const fam = familyOf(file);
      return fam ? { node: n2, channels: fam.channels, why: `${n2.classType} loads ${file} (${fam.family}, ${fam.channels}-channel latents)` } : null;
    }
    const named = CLASS_CHANNELS.exec(n2.classType);
    if (named) return { node: n2, channels: Number(named[1]), why: `${n2.classType} names ${named[1]} channels` };
    cur = n2.links.get("model");
  }
  return null;
}
function checkComfyWorkflow(json) {
  const wf = readWorkflow(json);
  if (!wf) return { format: null, checked: 0, skipped: 0, findings: [] };
  const findings = [];
  let checked = 0;
  let skipped = 0;
  for (const n2 of wf.nodes.values()) {
    if (!SAMPLERS.has(n2.classType)) continue;
    const latent = latentSource(wf.nodes, n2.links.get("latent_image"));
    const model = modelSource(wf.nodes, n2);
    if (!latent || !model) {
      skipped++;
      continue;
    }
    checked++;
    if (latent.channels !== model.channels) {
      findings.push({
        nodeId: n2.id,
        title: `${n2.title}: a ${latent.channels}-channel latent fed to a model that expects ${model.channels}`,
        detail: `model: ${model.why} (node ${model.node.id}); latent_image: ${latent.why} (node ${latent.node.id}). The first conv of the model will refuse the tensor. Use a latent node made for this model family.`
      });
    }
  }
  return { format: wf.format, checked, skipped, findings };
}

// src/utils/modelPlan.ts
var DEFAULT_SEQ_LEN = 2048;
function tokenInput(model) {
  const inp = model.components.find((c) => c.type === "input");
  const shape = inp?.params?.shape;
  const hasEmbedding = model.components.some((c) => /embed/i.test(c.type));
  if (!inp || !Array.isArray(shape) || shape.length < 1 || shape.length > 2 || !hasEmbedding) return null;
  return { inp, shape };
}
function tokenSeqLenOf(model) {
  const t = tokenInput(model);
  if (!t) return null;
  const L = Number(t.shape[t.shape.length - 1]);
  return Number.isInteger(L) && L > 0 ? L : null;
}
function readSeqLen(v) {
  if (v == null || v === "") return void 0;
  const n2 = typeof v === "string" ? Number(v.trim()) : typeof v === "number" ? v : NaN;
  return Number.isInteger(n2) && n2 > 0 ? n2 : null;
}
function withSeqLen(model, seqLen) {
  const t = tokenInput(model);
  if (!t) return { model, seqLen: null };
  const current = Number(t.shape[t.shape.length - 1]);
  const L = seqLen ?? Math.min(current, DEFAULT_SEQ_LEN);
  if (L === current) return { model, seqLen: L };
  const next = [...t.shape.slice(0, -1), L];
  const components = model.components.map((c) => c === t.inp ? { ...c, params: { ...c.params, shape: next }, inputShape: next } : c);
  return { model: { ...model, components }, seqLen: L };
}
var DEFAULT_ATTENTION = "sdpa";
function readAttention(v) {
  if (v == null || v === "") return DEFAULT_ATTENTION;
  if (typeof v !== "string") return null;
  const s = v.trim().toLowerCase();
  if (s === "") return DEFAULT_ATTENTION;
  if (s === "sdpa" || s === "eager" || s === "flash") return s;
  if (/^flash[_-]?attention(_?[23])?$/.test(s)) return "flash";
  return null;
}
function parseHumanSize(v) {
  if (typeof v === "number") return Number.isFinite(v) && v >= 0 ? v : null;
  if (typeof v !== "string") return null;
  const m = v.trim().replace(/,/g, "").match(/^(\d+(?:\.\d+)?)\s*([kmbt]?)$/i);
  if (!m) return null;
  const mult = { "": 1, k: 1e3, m: 1e6, b: 1e9, t: 1e12 };
  return Number(m[1]) * mult[m[2].toLowerCase()];
}
var GPU_ALIASES = [
  [/^t4/i, "T4", "T4 16GB"],
  [/^a100[-_ ]?40/i, "A100_40", "A100 40GB"],
  [/^a100$/i, "A100_40", "A100 40GB"],
  [/^h100/i, "H100_80", "H100 80GB"]
];
function readPolicy(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) return null;
  const raw = input;
  const policy = {};
  const problems = [];
  const num7 = (k) => {
    const v = raw[k];
    if (v == null) return void 0;
    const n2 = typeof v === "string" ? Number(v) : v;
    if (typeof n2 !== "number" || !Number.isFinite(n2) || n2 < 0) {
      problems.push(`${k}: expected a number, got ${JSON.stringify(v)}`);
      return void 0;
    }
    return n2;
  };
  const list2 = (k) => {
    const v = raw[k];
    if (v == null) return void 0;
    if (!Array.isArray(v) || !v.every((x) => typeof x === "string")) {
      problems.push(`${k}: expected a list of layer types`);
      return void 0;
    }
    return v;
  };
  if (raw.max_params != null) {
    const n2 = parseHumanSize(raw.max_params);
    if (n2 == null) problems.push(`max_params: expected a size like 200M, got ${JSON.stringify(raw.max_params)}`);
    else policy.max_params = raw.max_params;
  }
  if (raw.must_fit != null) {
    if (typeof raw.must_fit !== "string") problems.push("must_fit: expected a GPU name like A100-40GB");
    else policy.must_fit = raw.must_fit;
  }
  const mm = num7("max_memory_gb");
  if (mm != null) policy.max_memory_gb = mm;
  const ml = num7("max_layers");
  if (ml != null) policy.max_layers = ml;
  const mc = num7("max_cost_usd");
  if (mc != null) policy.max_cost_usd = mc;
  const ft = list2("forbid_types");
  if (ft) policy.forbid_types = ft;
  const rt = list2("require_types");
  if (rt) policy.require_types = rt;
  return { policy, problems };
}
function policyFindings(policy, problems, model, f, gpuFits) {
  const blockers = [];
  const warnings = [];
  for (const p of problems) warnings.push({ title: `policy: unreadable line (${p})`, stage: "policy" });
  const maxParams = policy.max_params != null ? parseHumanSize(policy.max_params) : null;
  if (maxParams != null && f.params > maxParams) {
    blockers.push({
      // The limit is quoted as the org wrote it ("200M"), not reformatted, so
      // the title and the `.neurarch.yml` line read as the same sentence.
      title: `policy: max_params ${policy.max_params} exceeded (${fmtParamCount(f.params)})`,
      detail: `max_params: ${policy.max_params}`,
      fix: `Reduce the model by ${fmtParamCount(f.params - maxParams)} parameters or raise max_params.`,
      stage: "policy"
    });
  }
  if (policy.must_fit) {
    const alias = GPU_ALIASES.find(([re]) => re.test(policy.must_fit.trim()));
    if (!alias) {
      warnings.push({ title: `policy: unknown GPU in must_fit (${policy.must_fit})`, detail: "Known: T4, A100-40GB, H100-80GB.", stage: "policy" });
    } else if (gpuFits[alias[1]] === false) {
      blockers.push({
        title: `policy: must_fit ${alias[2]} violated (needs ${(f.footprintGb * 1.3).toFixed(1)} GB)`,
        detail: `must_fit: ${policy.must_fit}`,
        stage: "policy"
      });
    }
  }
  const types = new Set(model.components.map((c) => c.type));
  for (const t of policy.forbid_types ?? []) {
    if (types.has(t)) blockers.push({ title: `policy: forbidden layer type ${t} present`, detail: `forbid_types: [${(policy.forbid_types ?? []).join(", ")}]`, stage: "policy" });
  }
  for (const t of policy.require_types ?? []) {
    if (!types.has(t)) blockers.push({ title: `policy: required layer type ${t} missing`, detail: `require_types: [${(policy.require_types ?? []).join(", ")}]`, stage: "policy" });
  }
  if (policy.max_layers != null && f.layers > policy.max_layers) {
    blockers.push({ title: `policy: max_layers ${policy.max_layers} exceeded (${f.layers})`, detail: `max_layers: ${policy.max_layers}`, stage: "policy" });
  }
  if (policy.max_memory_gb != null && f.params > 0 && f.footprintGb * 1.3 > policy.max_memory_gb) {
    warnings.push({ title: `policy: max_memory_gb ${policy.max_memory_gb} exceeded (${(f.footprintGb * 1.3).toFixed(1)} GB)`, detail: `max_memory_gb: ${policy.max_memory_gb}`, stage: "policy" });
  }
  if (policy.max_cost_usd != null && f.estCostUsd != null && f.estCostUsd > policy.max_cost_usd) {
    warnings.push({ title: `policy: max_cost_usd ${policy.max_cost_usd} exceeded (${fmtUsd(f.estCostUsd)})`, detail: `max_cost_usd: ${policy.max_cost_usd}`, stage: "policy" });
  }
  return { blockers, warnings };
}
function fmtParamCount(n2) {
  if (!Number.isFinite(n2)) return "?";
  if (n2 >= 1e9) return `${(n2 / 1e9).toFixed(2)}B`;
  if (n2 >= 1e6) return `${(n2 / 1e6).toFixed(1)}M`;
  if (n2 >= 1e3) return `${(n2 / 1e3).toFixed(1)}K`;
  return String(Math.round(n2));
}
function fmtUsd(v) {
  if (v == null) return "n/a";
  if (v < 0.01) return "<$0.01";
  return v < 1 ? `$${v.toFixed(2)}` : `$${v.toFixed(1)}`;
}
function fmtSignedUsd(v) {
  const sign = v < 0 ? "-" : "+";
  const abs = Math.abs(v);
  return `${sign}${abs < 0.01 ? "<$0.01" : abs < 1 ? `$${abs.toFixed(2)}` : `$${abs.toFixed(1)}`}`;
}
function fmtDuration(sec) {
  if (sec == null || sec <= 0) return "n/a";
  const h = sec / 3600;
  return h >= 1 ? `${h.toFixed(1)}h` : `${Math.max(1, Math.ceil(sec / 60))}m`;
}
function fmtShape(s) {
  return s && s.length ? `[${s.join(",")}]` : "?";
}
function fmtSigned(n2, unit = "") {
  return `${n2 >= 0 ? "+" : "-"}${fmtParamCount(Math.abs(n2))}${unit}`;
}
function shapeOf(s) {
  return Array.isArray(s) && s.length > 0 && s.every((x) => typeof x === "number") ? [...s] : null;
}
function findingOf(f) {
  const out = { title: f.title, stage: "preflight" };
  if (f.detail) out.detail = f.detail;
  if (f.fix) out.fix = f.fix;
  return out;
}
var OPTIMIZER_K = 1.2;
var OUTPUT_K = 0.88;
var ATTENTION_K = 3.64;
var INTERNAL_K = 1.34;
var RECURRENT_K = 5.2;
var TRANSIENT_K = 2.02;
var ACTIVATION_K = OUTPUT_K;
var MEMORY_CALIBRATION = {
  /** Measured runs the factors were fitted on (out-of-memory runs included). */
  fittedRuns: 178,
  architectures: 41,
  /** Held out one architecture at a time: median error on runs of 1 GB or more, percent. */
  heldOutMedianPct: 7,
  /** Held out: did-not-fit card checks flagged, of how many. */
  flagged: 113,
  noFit: 131,
  /** Runs measured after the factors were fixed, never in any fit, and the median miss on the ones that fit, percent. */
  afterFitRuns: 25,
  afterFitMedianPct: 14,
  /**
   * The upper multiplier U: held out one architecture at a time, measured peak
   * / predicted peak over the fitted runs of 1 GB or more, at the 90th and
   * 95th percentile (nearest rank), rounded up to two decimals. Predicted x U
   * is at or above the measured peak on that share of held-out runs.
   */
  upperK90: 1.17,
  upperK95: 1.43,
  /** Held-out runs (1 GB or more, measured peak) the multipliers were read from. */
  upperRuns: 91
};
var UPPER_K = MEMORY_CALIBRATION.upperK95;
var UPPER_COVERAGE = 0.95;
var UPPER_FLOOR_GB = 1;
function upperPeakGb(peakGb) {
  if (!(peakGb > 0)) return 0;
  return Math.max(peakGb * UPPER_K, UPPER_FLOOR_GB);
}
function fitVerdict(peakGb, params, gb2) {
  if (params <= 0) return null;
  if (upperPeakGb(peakGb) * FIT_HEADROOM <= gb2) return "fits";
  return peakGb <= gb2 ? "borderline" : "does_not_fit";
}
var MIG_PROFILES = {
  A100_40: [{ name: "1g.5gb", gb: 5 }, { name: "1g.10gb", gb: 10 }, { name: "2g.10gb", gb: 10 }, { name: "3g.20gb", gb: 20 }, { name: "4g.20gb", gb: 20 }, { name: "7g.40gb", gb: 40 }],
  A100_80: [{ name: "1g.10gb", gb: 10 }, { name: "1g.20gb", gb: 20 }, { name: "2g.20gb", gb: 20 }, { name: "3g.40gb", gb: 40 }, { name: "4g.40gb", gb: 40 }, { name: "7g.80gb", gb: 80 }],
  H100_80: [{ name: "1g.10gb", gb: 10 }, { name: "1g.20gb", gb: 20 }, { name: "2g.20gb", gb: 20 }, { name: "3g.40gb", gb: 40 }, { name: "4g.40gb", gb: 40 }, { name: "7g.80gb", gb: 80 }]
};
var MIG_SOURCE = "NVIDIA Multi-Instance GPU User Guide, Supported MIG Profiles";
var SHAREABLE_NOTE = "Memory only: copies of the conservative peak (upper bound x 1.15 headroom each) that fit the card's memory. Compute, memory bandwidth and the per-process CUDA context are not considered.";
function shareableFor(peakGbUpper) {
  if (!(peakGbUpper > 0)) return null;
  const per = peakGbUpper * FIT_HEADROOM;
  const mig = (k) => MIG_PROFILES[k].find((p) => per <= p.gb)?.name ?? null;
  return {
    perCopyGb: per,
    cards: PREFLIGHT_GPUS.map((g) => ({ gpu: g.name, gb: g.gb, copies: Math.floor(g.gb / per) })),
    mig: { A100_40: mig("A100_40"), A100_80: mig("A100_80"), H100_80: mig("H100_80") },
    migSource: MIG_SOURCE,
    memoryOnly: true,
    note: SHAREABLE_NOTE
  };
}
var ACTIVATION_BYTES_PER_ELEMENT = 2;
var ATTENTION_TYPES2 = /attention|transformerEncoder|transformerDecoder|^mla$|ditBlock|swin/i;
var ALWAYS_STORES_SCORES = /^hstuBlock$/;
var NOT_PER_TOKEN = /embed|positional|conv|^ema/i;
var RECURRENT_GATES = { lstm: 4, gru: 3, rnn: 1 };
function layerAttentionElems(c, shapes) {
  if (!ATTENTION_TYPES2.test(c.type) && !ALWAYS_STORES_SCORES.test(c.type)) return 0;
  const sh = shapes.get(c.id);
  const i = sh?.in ?? sh?.out;
  if (!i || i.length < 1) return 0;
  const p = c.params ?? {};
  const heads = p.numHeads ?? p.num_heads ?? p.nhead ?? 8;
  const L = i.length >= 2 ? i[i.length - 2] : i[0];
  const layers = p.numLayers ?? p.num_layers ?? 1;
  if (!(Number.isFinite(L) && L > 0)) return 0;
  if (c.type === "windowAttention") {
    const w = Number(p.windowSize ?? 7);
    return heads * L * Math.min(L, w > 0 ? w * w : L) * layers;
  }
  return heads * L * L * layers;
}
function storesScoresUnder(attention) {
  return attention === "eager" ? (c) => ATTENTION_TYPES2.test(c.type) || ALWAYS_STORES_SCORES.test(c.type) : (c) => ALWAYS_STORES_SCORES.test(c.type);
}
function layerMemoryTerms(c, shapes, storesScores) {
  if (c.type === "input" || c.type === "output" || c.type === "stickyNote") return null;
  const sh = shapes.get(c.id);
  const out = sh?.out ?? c.outputShape;
  if (!out || !out.length || out.some((d) => !Number.isFinite(d) || d <= 0)) return null;
  const outputs = out.reduce((a, b) => a * b, 1);
  const scores = storesScores(c) ? layerAttentionElems(c, shapes) : 0;
  const inShape = sh?.in ?? null;
  const p = c.params ?? {};
  let internal = 0;
  const din = inShape?.length ? inShape[inShape.length - 1] : 0;
  if (!NOT_PER_TOKEN.test(c.type) && din >= 2 && inShape.every((d) => Number.isFinite(d) && d > 0)) {
    let params = estimateLayerParams(c.type, c.params, inShape ?? void 0) ?? 0;
    if (/moe/i.test(c.type)) {
      const e = Number(p.numExperts ?? 8), k = Number(p.topK ?? 2), s = Number(p.numSharedExperts ?? 0);
      if (e + s > 0) params *= Math.min(1, (k + s) / (e + s));
    }
    const tokens = outputs / out[out.length - 1];
    internal = Math.max(0, tokens * params / din - outputs);
  }
  let recurrent = 0;
  const gates = RECURRENT_GATES[c.type.toLowerCase()];
  if (gates) {
    const steps = inShape && inShape.length >= 2 ? inShape[inShape.length - 2] : 1;
    const hidden = Number(p.hiddenSize ?? p.hidden_size ?? 128);
    const layers = Number(p.numLayers ?? p.num_layers ?? 1);
    recurrent = steps * hidden * gates * layers * (p.bidirectional ? 2 : 1);
  }
  return { outputs, scores, internal, recurrent };
}
function memoryTermsFor(modelIn, shapesIn, storesScores) {
  const { model, shapes } = withTiedHead(modelIn, shapesIn);
  const t = { outputs: 0, scores: 0, internal: 0, recurrent: 0, largest: 0 };
  let seen = false;
  for (const c of model.components) {
    const l = layerMemoryTerms(c, shapes, storesScores);
    if (!l) continue;
    seen = true;
    t.outputs += l.outputs;
    t.scores += l.scores;
    t.internal += l.internal;
    t.recurrent += l.recurrent;
    t.largest = Math.max(t.largest, l.outputs, l.scores);
  }
  return seen ? t : null;
}
function weighted(t) {
  return t.outputs * OUTPUT_K + t.scores * ATTENTION_K + t.internal * INTERNAL_K + t.recurrent * RECURRENT_K + (t.largest ?? 0) * TRANSIENT_K;
}
function activationElemsFor(model, shapes, attention) {
  const t = memoryTermsFor(model, shapes, storesScoresUnder(attention));
  return t == null ? null : weighted(t);
}
function activationGbAt(elems, batchSize) {
  return elems * batchSize * ACTIVATION_BYTES_PER_ELEMENT / 1e9;
}
function attentionElemsFor(model, shapes) {
  const eager = activationElemsFor(model, shapes, "eager");
  const sdpa = activationElemsFor(model, shapes, "sdpa");
  return eager == null || sdpa == null ? 0 : Math.max(0, eager - sdpa);
}
function attentionScoresGbFor(model, batchSize) {
  const shapes = propagateShapes(model).shapes;
  return activationGbAt(attentionElemsFor(model, shapes), Math.max(1, batchSize));
}
function activationGbFor(model, shapes, batchSize, attention = DEFAULT_ATTENTION) {
  const elems = activationElemsFor(model, shapes, attention);
  return elems == null ? null : activationGbAt(elems, batchSize);
}
var FIT_HEADROOM = 1.15;
function maxBatchFor(f, gb2) {
  const per = f.activationPerSampleGb;
  if (per == null || !(per > 0) || !(gb2 > 0)) return null;
  const fits = (b2) => (f.footprintGb + (f.activationGbAt ? f.activationGbAt(b2) : per * b2)) * FIT_HEADROOM <= gb2;
  if (!fits(1)) return null;
  let b = Math.max(1, Math.floor((gb2 / FIT_HEADROOM - f.footprintGb) / per));
  while (b > 1 && !fits(b)) b--;
  while (fits(b + 1)) b++;
  return b;
}
function topoOrder(model) {
  const byId = new Map(model.components.map((c, i) => [c.id, i]));
  const indeg = new Map(model.components.map((c) => [c.id, 0]));
  const next = new Map(model.components.map((c) => [c.id, []]));
  for (const e of model.connections ?? []) {
    if (!byId.has(e.from) || !byId.has(e.to)) continue;
    next.get(e.from).push(e.to);
    indeg.set(e.to, (indeg.get(e.to) ?? 0) + 1);
  }
  const ready = model.components.filter((c) => indeg.get(c.id) === 0).map((c) => c.id);
  const seen = /* @__PURE__ */ new Set();
  const out = [];
  while (ready.length) {
    ready.sort((a, b) => byId.get(a) - byId.get(b));
    const id = ready.shift();
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(model.components[byId.get(id)]);
    for (const n2 of next.get(id) ?? []) {
      const d = indeg.get(n2) - 1;
      indeg.set(n2, d);
      if (d === 0) ready.push(n2);
    }
  }
  for (const c of model.components) if (!seen.has(c.id)) out.push(c);
  return out;
}
function deviceMapFor(model, gpus, opts = {}) {
  const batch = Math.max(1, opts.batchSize ?? 32);
  const attention = opts.attention ?? DEFAULT_ATTENTION;
  const shapes = propagateShapes(model).shapes;
  const layers = topoOrder(model).filter((c) => c.type !== "input" && c.type !== "output" && c.type !== "stickyNote");
  const splits = [];
  if (!gpus.length || !layers.length) return { splits, fits: false };
  let fits = true;
  let cur = null;
  const stores = storesScoresUnder(attention);
  let largestId = null;
  let largest = 0;
  for (const c of layers) {
    const l = layerMemoryTerms(c, shapes, stores);
    if (l && Math.max(l.outputs, l.scores) > largest) {
      largest = Math.max(l.outputs, l.scores);
      largestId = c.id;
    }
  }
  for (const c of layers) {
    const inShape = shapes.get(c.id)?.in ?? void 0;
    const params = estimateLayerParams(c.type, c.params, inShape ?? void 0) ?? 0;
    const l = layerMemoryTerms(c, shapes, stores);
    const elems = l == null ? null : weighted({ ...l, largest: c.id === largestId ? largest : 0 });
    const w = params * 16 * OPTIMIZER_K / 1e9 + (elems == null ? 0 : activationGbAt(elems, batch));
    const cap = cur ? gpus[cur.card] / FIT_HEADROOM : 0;
    if (!cur || cur.gb + w > cap && cur.layers > 0 && cur.card + 1 < gpus.length) {
      cur = { card: cur ? cur.card + 1 : 0, gb: 0, firstLayer: c.name, lastLayer: c.name, layers: 0 };
      splits.push(cur);
    }
    cur.gb += w;
    cur.lastLayer = c.name;
    cur.layers++;
  }
  for (const sp of splits) if (sp.gb * FIT_HEADROOM > gpus[sp.card]) fits = false;
  return { splits, fits };
}
function twoCardSplitFor(model, peakGb, params, batchSize, attention) {
  if (params <= 0) return null;
  const sizes = [...new Set(PREFLIGHT_GPUS.map((g) => g.gb))].sort((a, b) => a - b);
  for (const gb2 of sizes) {
    if (peakGb * FIT_HEADROOM <= gb2) return null;
    if (peakGb * FIT_HEADROOM > 2 * gb2) continue;
    const map = deviceMapFor(model, [gb2, gb2], { batchSize, attention });
    if (map.fits) {
      const gpu = PREFLIGHT_GPUS.find((g) => g.gb === gb2).name;
      return { gpu, gbPerCard: gb2, ...map };
    }
  }
  return null;
}
function factsFor(model, workload, run, attention = DEFAULT_ATTENTION) {
  const report = buildPreflightReport(model, { workload, plan: run });
  const m = report.metrics;
  const prop = propagateShapes(model);
  const first = model.components.find((c) => c.type === "input");
  const last = [...model.components].reverse().find((c) => c.type === "output");
  const tail = model.components[model.components.length - 1];
  const outputShape = last ? shapeOf(prop.shapes.get(last.id)?.in) ?? shapeOf(prop.shapes.get(last.id)?.out) : tail ? shapeOf(prop.shapes.get(tail.id)?.out) : null;
  const batchSize = Math.max(1, run?.batchSize ?? 32);
  const footprintGb = m.trainFootprintBytes * OPTIMIZER_K / 1e9;
  const attentionGb = m.params > 0 ? activationGbAt(attentionElemsFor(model, prop.shapes), batchSize) : 0;
  const activationElems = m.params > 0 ? activationElemsFor(model, prop.shapes, attention) : null;
  const activationGb = activationElems == null ? null : activationGbAt(activationElems, batchSize);
  const peakGb = footprintGb + (activationGb ?? 0);
  const smallest = m.params > 0 ? PREFLIGHT_GPUS.find((g) => fitsCard(peakGb, m.params, g.gb)) : void 0;
  return {
    name: model.name || "model",
    layers: m.layers,
    params: m.params,
    footprintGb,
    activationGb,
    activationElems,
    attention,
    attentionGb,
    peakGb,
    estCostUsd: m.params > 0 && m.estTrainSec > 0 ? m.estCostUsd : null,
    estTrainSec: m.params > 0 && m.estTrainSec > 0 ? m.estTrainSec : null,
    samples: workload?.samples ?? 5e4,
    samplesSource: m.samplesAssumed ? null : workload?.source ?? null,
    epochs: Math.max(1, run?.epochs ?? 10),
    batchSize,
    fitsGpu: m.params > 0 ? m.fitsGpu : null,
    smallestFitGpu: smallest?.name ?? null,
    blockers: report.findings.filter((f) => f.severity === "block").map(findingOf),
    warnings: report.findings.filter((f) => f.severity === "warn").map(findingOf),
    inputShape: first ? shapeOf(prop.shapes.get(first.id)?.out) ?? shapeOf(first.params?.shape) : null,
    outputShape,
    seqLen: tokenSeqLenOf(model)
  };
}
function fitsCard(peakGb, params, gb2) {
  if (params <= 0) return null;
  return peakGb * FIT_HEADROOM <= gb2;
}
function smallestCardFor(peakGb, params) {
  return PREFLIGHT_GPUS.find((g) => fitsCard(peakGb, params, g.gb))?.name ?? null;
}
function smallestSafeCardFor(peakGb, params) {
  return PREFLIGHT_GPUS.find((g) => fitVerdict(peakGb, params, g.gb) === "fits")?.name ?? null;
}
function verdictWord(v) {
  return v == null ? "n/a" : v === "fits" ? "fits" : v === "borderline" ? "borderline" : "does not fit";
}
var UPPER_BOUND = { multiplier: UPPER_K, coverage: UPPER_COVERAGE, runs: MEMORY_CALIBRATION.upperRuns, multiplier90: MEMORY_CALIBRATION.upperK90 };
function upperBoundPhrase(plan) {
  const u = plan.cost.upperBound;
  return `upper bound ~${plan.cost.peakGbUpper.toFixed(1)} GB (the estimate x${u.multiplier}: held out by architecture, ${Math.round(u.coverage * 100)}% of ${u.runs} measured runs of 1 GB+ peaked at or under their own estimate x${u.multiplier})`;
}
var ATTENTION_LABEL = { sdpa: "SDPA", flash: "flash attention", eager: "eager" };
function attentionPhrase(attention, scoresGb) {
  if (scoresGb == null || !(scoresGb > 0.05)) return "";
  return attention === "eager" ? `attention: eager, scores stored (${scoresGb.toFixed(1)} GB of the peak)` : `attention: ${ATTENTION_LABEL[attention]}, scores not stored; eager would add ${scoresGb.toFixed(1)} GB`;
}
function calibrationSentence() {
  const c = MEMORY_CALIBRATION;
  return `Six factors fitted on ${c.fittedRuns} measured runs across ${c.architectures} architectures, out-of-memory runs included, with the attention-score term only where the run stored scores; held out one architecture at a time, the median error is ${c.heldOutMedianPct}% above 1 GB and it flagged ${c.flagged} of the ${c.noFit} times a run did not fit a card; on ${c.afterFitRuns} runs at the edge of a 24 GB card measured after the fit, the median miss is ${c.afterFitMedianPct}%.`;
}
function attentionSentence(attention, scoresGb) {
  const p = attentionPhrase(attention, scoresGb);
  return p ? `${p[0].toUpperCase()}${p.slice(1)}. ` : "";
}
function tokensAt(seqLen) {
  return seqLen != null ? `, ${seqLen} tokens` : "";
}
function batchOf(plan) {
  const m = /at batch (\d+)/.exec(plan.cost.assumptions);
  return m ? m[1] : "32";
}
function gpuGb(prefix, fallback) {
  const hit = PREFLIGHT_GPUS.find((g) => g.name.startsWith(prefix));
  return hit ? hit.gb : fallback;
}
function paramSummary(c) {
  const p = c.params ?? {};
  const keys = ["outChannels", "outFeatures", "embedDim", "numHeads", "hiddenSize", "kernelSize", "shape"];
  const bits = [];
  for (const k of keys) {
    if (p[k] != null) bits.push(`${k} ${JSON.stringify(p[k])}`);
    if (bits.length === 2) break;
  }
  return bits.length ? ` (${bits.join(", ")})` : "";
}
function byName2(model) {
  const map = /* @__PURE__ */ new Map();
  for (const c of model.components) map.set(c.name, c);
  return map;
}
function diffFor(head, headFacts, base) {
  const baseFacts = factsFor(base);
  const d = diffArchitectures(base, head);
  const baseBy = byName2(base);
  const headBy = byName2(head);
  const lines = [];
  for (const name of d.layersAdded) {
    const c = headBy.get(name);
    lines.push(`+ ${c?.type ?? "layer"} ${name}${c ? paramSummary(c) : ""}`);
  }
  for (const name of d.layersRemoved) {
    const c = baseBy.get(name);
    lines.push(`- ${c?.type ?? "layer"} ${name}`);
  }
  for (const ch of d.layersChanged) {
    if (ch.typeChanged) lines.push(`~ ${ch.name} type ${ch.typeChanged.from} -> ${ch.typeChanged.to}`);
    const before = baseBy.get(ch.name)?.params ?? {};
    const after = headBy.get(ch.name)?.params ?? {};
    for (const key of ch.paramsChanged ?? []) {
      lines.push(`~ ${ch.name}.${key} ${JSON.stringify(before[key])} -> ${JSON.stringify(after[key])}`);
    }
  }
  for (const e of d.connectionsAdded) lines.push(`+ edge ${e}`);
  for (const e of d.connectionsRemoved) lines.push(`- edge ${e}`);
  const touched = [...d.layersAdded, ...d.layersChanged.map((c) => c.name)].map((n2) => headBy.get(n2)).filter((c) => !!c);
  const blastRadius = touched.length ? analyzeImpact(head, touched, { includeUpstream: false }).totals.downstream : 0;
  const paramsDelta = headFacts.params - baseFacts.params;
  const baseLegal = baseFacts.blockers.length === 0;
  const headLegal = headFacts.blockers.length === 0;
  return {
    base: { name: baseFacts.name, layers: baseFacts.layers, params: baseFacts.params },
    added: d.layersAdded.length,
    removed: d.layersRemoved.length,
    changed: d.layersChanged.length,
    paramsDelta,
    paramsDeltaPct: baseFacts.params > 0 ? paramsDelta / baseFacts.params * 100 : null,
    // Same 16 bytes per parameter the pre-flight footprint uses (fp32 weights,
    // grads, two Adam moments). Activations are not in it, on either side.
    memoryDeltaGb: paramsDelta * 16 / 1e9,
    costDeltaUsd: headFacts.estCostUsd != null && baseFacts.estCostUsd != null ? headFacts.estCostUsd - baseFacts.estCostUsd : null,
    blastRadius,
    lines,
    legalityChanged: baseLegal && !headLegal ? "became_illegal" : !baseLegal && headLegal ? "became_legal" : null
  };
}
function summaryFor(f, plan, diff, withheld) {
  const parts = [`${f.name}: ${f.layers} layer${f.layers === 1 ? "" : "s"}, ${fmtParamCount(f.params)} params`];
  if (withheld) {
    parts.push("cannot say whether it runs: the read was incomplete");
    if (f.estCostUsd != null) parts.push(`about ${fmtUsd(f.estCostUsd)} to train if it does`);
  } else if (f.blockers.length) {
    parts.push(`will not run (${f.blockers.length} blocker${f.blockers.length === 1 ? "" : "s"}: ${f.blockers[0].title})`);
    if (f.estCostUsd != null) parts.push(`the run it refuses was quoted ${fmtUsd(f.estCostUsd)}`);
  } else {
    parts.push(f.warnings.length ? `will run, ${f.warnings.length} warning${f.warnings.length === 1 ? "" : "s"}` : "will run");
    const fits = plan.cost.gpuFits;
    const smallest = plan.cost.smallestFitGpu ? `${/^[AEIOU]/i.test(plan.cost.smallestFitGpu) ? "an" : "a"} ${plan.cost.smallestFitGpu}` : fits.T4 ? "a T4" : fits.A100_40 ? "an A100 40GB" : fits.H100_80 ? "an H100 80GB" : null;
    const article = (g) => `${/^[AEIOU]/i.test(g) ? "an" : "a"} ${g}`;
    const safe = plan.cost.smallestSafeGpu;
    if (smallest && plan.cost.smallestFitGpu && safe !== plan.cost.smallestFitGpu) {
      parts.push(`fits ${smallest} at batch ${batchOf(plan)}, borderline there (measure once; ${safe ? `${article(safe)} holds the upper bound` : "no listed card holds the upper bound"})`);
    } else if (smallest) parts.push(`fits ${smallest} at batch ${batchOf(plan)}`);
    else if (fits.T4 === false) parts.push("does not fit a single 80GB card");
    if (f.estCostUsd != null) parts.push(`about ${fmtUsd(f.estCostUsd)} to train`);
  }
  if (diff) {
    const delta = diff.paramsDelta === 0 ? "no param change" : `params ${fmtSigned(diff.paramsDelta)}`;
    parts.push(`vs ${diff.base.name}: ${delta}${diff.legalityChanged === "became_illegal" ? ", BECAME ILLEGAL" : diff.legalityChanged === "became_legal" ? ", became legal" : ""}`);
  }
  return parts.join(", ");
}
function readGraph(input) {
  const bad = (title, detail) => ({ blockers: [{ title, detail, stage: "preflight" }] });
  if (!input || typeof input !== "object" || Array.isArray(input)) return bad("Unreadable graph", "Expected an object with a components array.");
  const g = input;
  if (!Array.isArray(g.components)) return bad("Unreadable graph", "components must be an array.");
  const ids = /* @__PURE__ */ new Set();
  for (const c of g.components) {
    if (!c || typeof c !== "object" || Array.isArray(c)) return bad("Unreadable graph", "Every component must be an object.");
    const comp = c;
    if (typeof comp.type !== "string" || !comp.type) return bad("Unreadable graph", `Component ${String(comp.id ?? comp.name ?? "?")} has no type.`);
    if (comp.params != null && (typeof comp.params !== "object" || Array.isArray(comp.params))) return bad("Unreadable graph", `Component ${String(comp.id ?? comp.name ?? "?")} has params that are not an object.`);
    if (typeof comp.id === "string") ids.add(comp.id);
    if (comp.type === "input") {
      const shape = comp.params?.shape;
      if (Array.isArray(shape) && !shape.every((x) => typeof x === "number" && Number.isInteger(x) && x > 0)) {
        return bad("Input shape is not a list of positive integers", `Got ${JSON.stringify(shape)}.`);
      }
    }
  }
  const conns = Array.isArray(g.connections) ? g.connections : [];
  for (const e of conns) {
    if (!e || typeof e !== "object") return bad("Unreadable graph", "Every connection must be an object.");
    const edge = e;
    if (typeof edge.from !== "string" || typeof edge.to !== "string") return bad("Unreadable graph", "A connection is missing from/to.");
    if (edge.from === edge.to) return bad("Self-loop", `Connection ${edge.from} -> ${edge.to} points at itself.`);
    if (!ids.has(edge.from) || !ids.has(edge.to)) return bad("Connection to a missing layer", `${edge.from} -> ${edge.to} references a component that does not exist.`);
  }
  const real = g.components.filter((c) => c.type !== "input" && c.type !== "output" && c.type !== "stickyNote");
  if (real.length === 0) return bad("Empty graph", "No layers between input and output; nothing to run.");
  return {
    model: {
      id: typeof g.id === "string" ? g.id : "model",
      name: typeof g.name === "string" ? g.name : "model",
      components: g.components,
      connections: conns,
      ...g.hyperparams && typeof g.hyperparams === "object" ? { hyperparams: g.hyperparams } : {}
    }
  };
}
function unreadablePlan(blockers) {
  return {
    version: 1,
    summary: `unreadable: will not run (${blockers[0]?.title ?? "unreadable graph"})`,
    model: { name: "unreadable", layers: 0, params: 0, inputShape: null, outputShape: null, fingerprint: "" },
    run: { legal: false, blockers, warnings: [] },
    cost: {
      estCostUsd: null,
      estTrainSec: null,
      fitsGpu: null,
      peakGb: 0,
      activationGb: null,
      smallestFitGpu: null,
      peakGbUpper: 0,
      upperBound: UPPER_BOUND,
      gpuVerdict: { T4: null, A100_40: null, H100_80: null },
      smallestSafeGpu: null,
      maxBatchSafe: { T4: null, A100_40: null, H100_80: null },
      shareable: null,
      maxBatch: { T4: null, A100_40: null, H100_80: null },
      twoCardSplit: null,
      attention: DEFAULT_ATTENTION,
      attentionScoresGb: null,
      seqLen: null,
      // Nothing was refused here that we can price: an unreadable payload never
      // became a run with a number on it.
      refusedRunUsd: null,
      gpuFits: { T4: null, A100_40: null, H100_80: null },
      assumptions: "No estimate: the graph could not be read."
    },
    diff: null,
    policy: null,
    history: null,
    change: null,
    lora: null,
    edit: null
  };
}
function buildPlan(input, opts = {}) {
  const read = readGraph(input);
  if ("blockers" in read) return unreadablePlan(read.blockers);
  const want = opts.seqLen === "default" ? null : typeof opts.seqLen === "number" && Number.isInteger(opts.seqLen) && opts.seqLen > 0 ? opts.seqLen : void 0;
  const sized = (m) => want === void 0 ? m : withSeqLen(m, want).model;
  const model = sized(read.model);
  let base;
  if (opts.base != null) {
    const b = readGraph(opts.base);
    base = "model" in b ? sized(b.model) : void 0;
  }
  try {
    return buildPlanUnchecked(model, base, opts);
  } catch (e) {
    return unreadablePlan([{
      title: "Could not analyse the graph",
      detail: e instanceof Error ? e.message : String(e),
      stage: "preflight"
    }]);
  }
}
function buildPlanUnchecked(model, base, opts) {
  const attention = readAttention(opts.attention) ?? DEFAULT_ATTENTION;
  const f = factsFor(model, opts.workload, opts.run, attention);
  const gpuFits = {
    T4: fitsCard(f.peakGb, f.params, gpuGb("T4", 16)),
    A100_40: fitsCard(f.peakGb, f.params, gpuGb("A100 (40GB)", 40)),
    H100_80: fitsCard(f.peakGb, f.params, gpuGb("H100", 80))
  };
  const elems = f.activationElems;
  const batchFacts = {
    footprintGb: f.footprintGb,
    activationPerSampleGb: elems == null ? null : activationGbAt(elems, 1),
    activationGbAt: elems == null ? void 0 : (b) => activationGbAt(elems, b)
  };
  const maxBatch = f.params > 0 ? {
    T4: maxBatchFor(batchFacts, gpuGb("T4", 16)),
    A100_40: maxBatchFor(batchFacts, gpuGb("A100 (40GB)", 40)),
    H100_80: maxBatchFor(batchFacts, gpuGb("H100", 80))
  } : { T4: null, A100_40: null, H100_80: null };
  const peakGbUpper = f.params > 0 ? upperPeakGb(f.peakGb) : 0;
  const gpuVerdict = {
    T4: fitVerdict(f.peakGb, f.params, gpuGb("T4", 16)),
    A100_40: fitVerdict(f.peakGb, f.params, gpuGb("A100 (40GB)", 40)),
    H100_80: fitVerdict(f.peakGb, f.params, gpuGb("H100", 80))
  };
  const safeFacts = {
    footprintGb: f.footprintGb * UPPER_K,
    activationPerSampleGb: elems == null ? null : activationGbAt(elems, 1) * UPPER_K,
    activationGbAt: elems == null ? void 0 : (b) => activationGbAt(elems, b) * UPPER_K
  };
  const maxBatchSafe = f.params > 0 ? {
    T4: maxBatchFor(safeFacts, gpuGb("T4", 16)),
    A100_40: maxBatchFor(safeFacts, gpuGb("A100 (40GB)", 40)),
    H100_80: maxBatchFor(safeFacts, gpuGb("H100", 80))
  } : { T4: null, A100_40: null, H100_80: null };
  const cost = {
    estCostUsd: f.estCostUsd,
    estTrainSec: f.estTrainSec,
    // Filled in below, once policy findings have had their say: a graph that
    // passes pre-flight and fails a house rule is refused just the same.
    refusedRunUsd: null,
    fitsGpu: f.fitsGpu,
    gpuFits,
    peakGb: f.peakGb,
    peakGbUpper,
    upperBound: UPPER_BOUND,
    gpuVerdict,
    smallestSafeGpu: f.params > 0 ? smallestSafeCardFor(f.peakGb, f.params) : null,
    maxBatchSafe,
    shareable: f.params > 0 ? shareableFor(peakGbUpper) : null,
    activationGb: f.activationGb,
    smallestFitGpu: f.smallestFitGpu,
    maxBatch,
    twoCardSplit: twoCardSplitFor(model, f.peakGb, f.params, f.batchSize, attention),
    attention,
    attentionScoresGb: f.activationGb == null ? null : f.attentionGb,
    seqLen: f.seqLen,
    // Written from what was actually priced, never typed alongside it: the
    // sentence used to say "50,000 samples (assumed)" unconditionally, so a
    // plan priced over 40 million rows would have carried a footnote about a
    // workload it did not use.
    assumptions: `Cost is a roofline estimate on the priced GPU for ${f.epochs} epochs at batch ${f.batchSize}${tokensAt(f.seqLen)} over ${f.samples.toLocaleString()} samples` + (f.samplesSource ? ` (from "${f.samplesSource}")` : " (assumed; no dataset attached)") + (f.activationGb != null ? `. GPU fit is the training peak at batch ${f.batchSize}${tokensAt(f.seqLen)}, AdamW with bf16 autocast: weights, gradients and optimizer state (${f.footprintGb.toFixed(2)} GB, ${OPTIMIZER_K} x 16 bytes per parameter) plus activations (${f.activationGb.toFixed(1)} GB: every layer's output, the intermediates inside each layer, recurrent gate states and the largest tensor's backward copy${attention === "eager" && f.attentionGb > 0.05 ? `, and ${f.attentionGb.toFixed(1)} GB of attention scores eager attention stores for backward` : ""}), with 1.15x headroom. ${attentionSentence(attention, f.attentionGb)}${calibrationSentence()}` : ". GPU fit is fp32 weights plus gradients plus two Adam moments (16 bytes per parameter) with 1.15x headroom; no shape propagated, so activations are not included.")
  };
  const diff = base ? diffFor(model, f, base) : null;
  const read = opts.policy != null ? readPolicy(opts.policy) : null;
  let policy = null;
  if (read) {
    const pf = policyFindings(read.policy, read.problems, model, f, gpuFits);
    f.blockers = [...f.blockers, ...pf.blockers];
    f.warnings = [...f.warnings, ...pf.warnings];
    policy = { applied: true, violations: pf.blockers.length + pf.warnings.length };
  } else if (opts.policy != null) {
    policy = { applied: false, violations: 0 };
  }
  let change = null;
  let lora = f.params > 0 ? loraTargetsFor(model) : null;
  if (opts.spec != null) {
    const readSpec = readChangeSpec(opts.spec);
    if (!readSpec) {
      f.warnings = [...f.warnings, { title: "change: unreadable spec (expected an object)", stage: "change" }];
    } else {
      const withheld = opts.incompleteRead && typeof opts.incompleteRead.reason === "string";
      lora ??= loraTargetsFor(model);
      change = evaluateChangeSpec(model, readSpec.spec, readSpec.problems, {
        lora,
        baseFingerprint: architectureHashFromJson(model) ?? "",
        baseLegal: withheld ? null : f.blockers.length === 0,
        trainingPeak: (adapter) => trainingPeakFor({ params: f.params, peakGb: f.peakGb, activationGb: f.activationGb, lora }, adapter),
        params: f.params,
        estCostUsd: f.estCostUsd,
        batchSize: f.batchSize,
        seqLen: f.seqLen
      });
      f.blockers = [...f.blockers, ...change.gate.blockers];
      f.warnings = [...f.warnings, ...change.gate.warnings];
    }
  }
  let edit = null;
  {
    const checked = [];
    const found = [];
    if (base) {
      checked.push("graph");
      found.push(...checkGraphEdit(base, model));
    }
    if (opts.config != null && opts.baseConfig != null && typeof opts.config === "object" && typeof opts.baseConfig === "object") {
      checked.push("config");
      found.push(...checkConfigEdit(opts.baseConfig, opts.config));
    }
    if (checked.length) {
      edit = { checked, findings: found };
      for (const x of found) {
        const pf = { title: `edit: ${x.title}`, detail: x.detail, stage: "edit", rule: x.rule };
        if (x.severity === "block") f.blockers = [...f.blockers, pf];
        else f.warnings = [...f.warnings, pf];
      }
    }
  }
  cost.refusedRunUsd = f.blockers.length > 0 ? f.estCostUsd : null;
  const history = Array.isArray(opts.history) && opts.history.length ? { rows: opts.history.slice(0, 20) } : null;
  const incomplete = opts.incompleteRead && typeof opts.incompleteRead.reason === "string" ? { reason: opts.incompleteRead.reason } : null;
  const plan = {
    version: 1,
    summary: "",
    model: {
      name: f.name,
      layers: f.layers,
      params: f.params,
      inputShape: f.inputShape,
      outputShape: f.outputShape,
      fingerprint: architectureHashFromJson(model) ?? ""
    },
    run: incomplete ? { legal: null, blockers: f.blockers, warnings: f.warnings, withheld: incomplete.reason } : { legal: f.blockers.length === 0, blockers: f.blockers, warnings: f.warnings },
    cost,
    diff,
    policy,
    history,
    change,
    lora: f.params > 0 ? lora : null,
    edit
  };
  plan.summary = summaryFor(f, plan, diff, Boolean(incomplete));
  return plan;
}
function historySentence(r) {
  const value = String(Number(r.value.toFixed(4)));
  const bits = [`${r.metric} ${value}`];
  if (r.epochs != null) bits.push(`${r.epochs} epochs`);
  if (r.estCostUsd != null) bits.push(fmtUsd(r.estCostUsd));
  if (r.wallSec != null) bits.push(fmtDuration(r.wallSec));
  bits.push(r.at.slice(0, 10));
  return `Last time this structure trained here: ${bits.join(", ")}`;
}
function changeGateSentence(ch) {
  const e = ch.estimates;
  const bits = [ch.gate.ok ? "ok" : `REFUSED (${ch.gate.blockers.length} blocker${ch.gate.blockers.length === 1 ? "" : "s"})`];
  if (e.trainMemGb != null) bits.push(`est train mem ${(e.trainMemGb * CHANGE_HEADROOM).toFixed(1)} GB ${workloadPhrase(e.batchSize, e.seqLen)}${ch.spec.adapter ? ` (${fmtParamCount(e.trainableParams)} trainable)` : ""}`);
  if (e.weightBytes != null) bits.push(`${ch.spec.quant?.precision ?? "fp16"} weights ${(e.weightBytes / 1e9).toFixed(2)} GB`);
  if (e.servingFits != null) bits.push(e.servingFits ? "serving fits" : "serving does NOT fit");
  if (ch.gate.warnings.length) bits.push(`${ch.gate.warnings.length} warning${ch.gate.warnings.length === 1 ? "" : "s"}`);
  return bits.join(" \xB7 ");
}
function changeLines(ch) {
  return [
    `  change    ${describeChangeSpec(ch.spec)}   spec ${ch.specId}`,
    `  gate      ${changeGateSentence(ch)}`
  ];
}
var MHA_TYPES = /* @__PURE__ */ new Set(["attention", "selfAttention", "multiHeadAttention", "crossModalAttention", "crossAttention", "coAttention", "linearAttention", "localAttention", "causalAttention", "windowAttention"]);
var MHA_BIASED = /* @__PURE__ */ new Set(["attention", "selfAttention", "multiHeadAttention", "crossModalAttention", "crossAttention", "coAttention", "linearAttention", "localAttention"]);
function roleFromName(leaf) {
  const n2 = leaf.toLowerCase();
  if (/(^|_)(q_proj|query|wq|q)$/.test(n2)) return "q_proj";
  if (/(^|_)(k_proj|key|wk|k)$/.test(n2)) return "k_proj";
  if (/(^|_)(v_proj|value|wv|v)$/.test(n2)) return "v_proj";
  if (/(^|_)(o_proj|out_proj|wo|attn_out)$/.test(n2)) return "o_proj";
  if (/gate/.test(n2)) return "gate";
  if (/(^|_)up(_proj)?$/.test(n2)) return "up";
  if (/(^|_)down(_proj)?$/.test(n2)) return "down";
  if (/(lm_)?head|classifier|logits/.test(n2)) return "head";
  if (/fc|mlp|ffn|dense|proj|linear/.test(n2)) return "fc";
  return "other";
}
function loraTargetsFor(model, shapesIn) {
  const shapes = shapesIn ?? propagateShapes(model).shapes;
  const modules = [];
  const uncovered = [];
  const feedsOutput = new Set(
    (model.connections ?? []).filter((e) => model.components.find((c) => c.id === e.to)?.type === "output").map((e) => e.from)
  );
  let baseParams = 0;
  const add = (c, module, role, inF, outF, bias, own = false) => {
    if (!(inF > 0 && outF > 0)) return;
    modules.push({
      name: own ? c.name : `${c.name}.${module}`,
      module,
      layer: c.name,
      type: c.type,
      role,
      inFeatures: inF,
      outFeatures: outF,
      params: inF * outF + (bias ? outF : 0),
      lora: { r8: 8 * (inF + outF), r16: 16 * (inF + outF) }
    });
  };
  const n2 = (v, d = 0) => typeof v === "number" && Number.isFinite(v) && v > 0 ? v : d;
  for (const c of topoOrder(model)) {
    if (c.type === "input" || c.type === "output" || c.type === "stickyNote") continue;
    const p = c.params ?? {};
    const inShape = shapes.get(c.id)?.in ?? void 0;
    const lastDim = inShape && inShape.length ? inShape[inShape.length - 1] : 0;
    const total = estimateLayerParams(c.type, c.params, inShape ?? void 0) ?? 0;
    baseParams += total;
    const before = modules.length;
    const t = c.type;
    if (p.tied === true) continue;
    if (t === "linear") {
      const leaf = (c.name || c.id).split(".").pop() || c.id;
      let role = roleFromName(leaf);
      if ((role === "fc" || role === "other") && feedsOutput.has(c.id)) role = "head";
      add(c, leaf, role, n2(p.inFeatures, lastDim), n2(p.outFeatures), p.bias !== false, true);
    } else if (t === "lmHead") {
      add(c, (c.name || "lm_head").split(".").pop() || "lm_head", "head", n2(p.inFeatures ?? p.hiddenSize ?? p.embedDim, lastDim), n2(p.vocabSize ?? p.numEmbeddings), p.bias === true, true);
    } else if (MHA_TYPES.has(t)) {
      const d = n2(p.embedDim ?? p.hiddenDim, t === "causalAttention" ? 512 : t === "windowAttention" ? 96 : 0);
      const b = MHA_BIASED.has(t);
      for (const r of ["q_proj", "k_proj", "v_proj", "o_proj"]) add(c, r, r, d, d, b);
    } else if (t === "groupedQueryAttention") {
      const D = n2(p.embedDim, 4096);
      const H = n2(p.numHeads, 32);
      const Hkv = n2(p.numKVHeads, H);
      const hd = n2(p.headDim, H > 0 ? Math.floor(D / H) : 128);
      add(c, "q_proj", "q_proj", D, H * hd, false);
      add(c, "k_proj", "k_proj", D, Hkv * hd, false);
      add(c, "v_proj", "v_proj", D, Hkv * hd, false);
      add(c, "o_proj", "o_proj", H * hd, D, false);
    } else if (t === "feedForward") {
      const d = n2(p.hiddenDim ?? p.embedDim);
      const ff = n2(p.ffDim, d * 4);
      add(c, "fc1", "fc", d, ff, true);
      add(c, "fc2", "fc", ff, d, true);
    } else if (t === "swiglu" || t === "geglu") {
      const D = t === "swiglu" ? n2(p.embedDim ?? p.inFeatures, 4096) : n2(p.dim, 512);
      const I = t === "swiglu" ? n2(p.intermediateSize ?? p.hiddenFeatures ?? p.ffDim, Math.round(D * 8 / 3)) : n2(p.hiddenDim, D * 4);
      add(c, "gate_proj", "gate", D, I, false);
      add(c, "up_proj", "up", D, I, false);
      add(c, "down_proj", "down", I, D, false);
    } else if (t === "transformerBlock") {
      const d = n2(p.embedDim ?? p.hiddenDim ?? p.dModel);
      const ff = n2(p.ffDim ?? p.dFf, d * 4);
      for (const r of ["q_proj", "k_proj", "v_proj", "o_proj"]) add(c, r, r, d, d, true);
      add(c, "fc1", "fc", d, ff, true);
      add(c, "fc2", "fc", ff, d, true);
    }
    if (modules.length === before && total > 0 && !/norm|embed|positional|rope|alibi|layerScale|prelu|dyt/i.test(t)) {
      uncovered.push({ name: c.name, type: t, params: total });
    }
  }
  const byRole = {};
  for (const m of modules) {
    const r = byRole[m.role] ??= { count: 0, params: 0 };
    r.count++;
    r.params += m.params;
  }
  const preset = (pick) => {
    const ms = modules.filter(pick);
    const r8 = ms.reduce((a, m) => a + m.lora.r8, 0);
    const r16 = ms.reduce((a, m) => a + m.lora.r16, 0);
    const pct2 = (x) => baseParams > 0 ? x / baseParams * 100 : 0;
    return {
      targetModules: [...new Set(ms.map((m) => m.module))],
      modules: ms.length,
      loraParams: { r8, r16 },
      pctOfBase: { r8: pct2(r8), r16: pct2(r16) }
    };
  };
  const ATTN3 = /* @__PURE__ */ new Set(["q_proj", "k_proj", "v_proj", "o_proj"]);
  return {
    baseParams,
    modules,
    byRole,
    presets: {
      attention: preset((m) => ATTN3.has(m.role)),
      allLinear: preset((m) => m.role !== "head")
    },
    uncovered
  };
}
function peakInputsOf(plan) {
  return { params: plan.model.params, peakGb: plan.cost.peakGb, activationGb: plan.cost.activationGb, lora: plan.lora };
}
function adapterParamsFor(lora, rank, targets = []) {
  const want = targets.map((t) => t.toLowerCase()).filter((t) => t && t !== "all-linear" && t !== "all_linear");
  const pick = (m) => want.some((t) => m.module.toLowerCase() === t || m.role === t || m.name.toLowerCase().endsWith(`.${t}`));
  const named = want.length ? lora.modules.filter(pick) : [];
  const modules = named.length ? named : lora.modules.filter((m) => m.role !== "head");
  return modules.reduce((a, m) => a + m.lora.r8 / 8 * rank, 0);
}
function trainingPeakFor(p, adapter) {
  if (!(p.params > 0)) return null;
  if (!adapter) return { trainableParams: p.params, peakGb: p.peakGb };
  if (!p.lora) return null;
  const adapterParams = adapterParamsFor(p.lora, adapter.rank, adapter.targets ?? []);
  if (!(adapterParams > 0)) return null;
  const baseBytesPerParam = adapter.peft === "qlora" ? 0.5 : 2;
  const peakGb = (p.activationGb ?? 0) + p.params * baseBytesPerParam / 1e9 + adapterParams * 16 * OPTIMIZER_K / 1e9;
  return { trainableParams: Math.round(adapterParams), peakGb };
}
function renderLoraText(lora) {
  const pct2 = (x) => x < 0.01 ? "<0.01%" : `${x.toFixed(2)}%`;
  const L = [];
  L.push(`LoRA targets   base ${fmtParamCount(lora.baseParams)} params, ${lora.modules.length} linear module${lora.modules.length === 1 ? "" : "s"}`);
  const row = (label, pr) => {
    L.push(`  ${label.padEnd(11)}  r=8 +${fmtParamCount(pr.loraParams.r8)} (${pct2(pr.pctOfBase.r8)})  r=16 +${fmtParamCount(pr.loraParams.r16)} (${pct2(pr.pctOfBase.r16)})  over ${pr.modules} module${pr.modules === 1 ? "" : "s"}`);
    L.push(`  ${"".padEnd(11)}  target_modules=${JSON.stringify(pr.targetModules)}`);
  };
  row("attention", lora.presets.attention);
  row("all-linear", lora.presets.allLinear);
  const roles = Object.entries(lora.byRole);
  if (roles.length) L.push(`  by role      ${roles.map(([r, v]) => `${r} ${v.count}`).join(" \xB7 ")}`);
  if (lora.uncovered.length) {
    const sum = lora.uncovered.reduce((a, u) => a + u.params, 0);
    L.push(`  not opened   ${lora.uncovered.length} layer${lora.uncovered.length === 1 ? "" : "s"} (${fmtParamCount(sum)} params): ${[...new Set(lora.uncovered.map((u) => u.type))].join(", ")}`);
  }
  L.push("  r x (in + out) per module; the head is excluded from all-linear, as PEFT does.");
  return L.join("\n");
}
var MAX_LIST = 3;
function yesNo(v) {
  return v == null ? "n/a" : v ? "yes" : "no";
}
function clip(items, render) {
  const out = items.slice(0, MAX_LIST).map(render);
  if (items.length > MAX_LIST) out.push(`  ... ${items.length - MAX_LIST} more`);
  return out;
}
function maxBatchLine(plan) {
  const mb = plan.cost.maxBatch;
  if (!mb || mb.T4 == null && mb.A100_40 == null && mb.H100_80 == null) return null;
  const v = (x) => x == null ? "n/a" : x.toLocaleString("en-US");
  return `T4 ${v(mb.T4)} \xB7 A100 40GB ${v(mb.A100_40)} \xB7 H100 80GB ${v(mb.H100_80)} (est.)`;
}
function safeBatchLine(plan) {
  const mb = plan.cost.maxBatchSafe;
  if (!mb || mb.T4 == null && mb.A100_40 == null && mb.H100_80 == null) return null;
  const v = (x) => x == null ? "n/a" : x.toLocaleString("en-US");
  return `T4 ${v(mb.T4)} \xB7 A100 40GB ${v(mb.A100_40)} \xB7 H100 80GB ${v(mb.H100_80)} (upper bound fits)`;
}
function fitWord(plan, k) {
  const v = plan.cost.gpuVerdict?.[k];
  if (v === void 0) return yesNo(plan.cost.gpuFits[k]);
  return v == null ? "n/a" : v === "fits" ? "yes" : v === "borderline" ? "BORDERLINE" : "no";
}
function fitsRow(plan) {
  return `T4 16GB ${fitWord(plan, "T4")} \xB7 A100 40GB ${fitWord(plan, "A100_40")} \xB7 H100 80GB ${fitWord(plan, "H100_80")}`;
}
function borderlineSentence(plan) {
  const v = plan.cost.gpuVerdict;
  if (!v) return null;
  const names = { T4: "T4 16GB", A100_40: "A100 40GB", H100_80: "H100 80GB" };
  const cards = Object.keys(names).filter((k) => v[k] === "borderline").map((k) => names[k]);
  if (!cards.length) return null;
  return `borderline on ${cards.join(", ")}: the point estimate fits, the upper bound with 1.15x headroom does not. Measure once (one step at this batch) before reserving.`;
}
function shareLine(plan) {
  const s = plan.cost.shareable;
  if (!s) return null;
  const pick = ["T4 (16GB)", "A100 (40GB)", "H100 (80GB)"];
  const cards = s.cards.filter((c) => pick.includes(c.gpu)).map((c) => `${c.copies} on ${c.gpu.replace(/ \((\d+GB)\)/, " $1")}`);
  const mig = [s.mig.A100_40 ? `A100-40 ${s.mig.A100_40}` : null, s.mig.A100_80 ? `A100-80 ${s.mig.A100_80}` : null, s.mig.H100_80 ? `H100 ${s.mig.H100_80}` : null].filter(Boolean);
  return `${s.perCopyGb.toFixed(1)} GB a copy: ${cards.join(" \xB7 ")}${mig.length ? `; smallest MIG slice ${mig.join(", ")}` : "; no MIG slice holds it"} (memory only)`;
}
function splitLine(plan) {
  const s = plan.cost.twoCardSplit;
  if (!s) return null;
  const part = (x) => `card ${x.card + 1} ${x.firstLayer}..${x.lastLayer} (${x.layers} layer${x.layers === 1 ? "" : "s"}, ${x.gb.toFixed(1)} GB)`;
  return `2 x ${s.gpu} at batch ${batchOf(plan)}: ${s.splits.map(part).join(" \xB7 ")} (est.)`;
}
function renderPlanText(plan) {
  const L = [];
  const m = plan.model;
  L.push(`Plan: ${m.name}   ${m.layers} layers \xB7 ${fmtParamCount(m.params)} params \xB7 fingerprint ${m.fingerprint || "n/a"}`);
  L.push(`  shape     ${fmtShape(m.inputShape)} -> ${fmtShape(m.outputShape)}`);
  L.push(
    plan.run.legal == null ? `  will run  CANNOT SAY (${plan.run.withheld ?? "the read was incomplete"})` : plan.run.legal ? `  will run  yes (${plan.run.blockers.length} blockers, ${plan.run.warnings.length} warnings)` : `  will run  NO  (${plan.run.blockers.length} blocker${plan.run.blockers.length === 1 ? "" : "s"})`
  );
  L.push(`  fits      ${fitsRow(plan)}   (peak ~${plan.cost.peakGb.toFixed(1)} GB at batch ${batchOf(plan)}${tokensAt(plan.cost.seqLen)}${plan.cost.activationGb == null ? ", activations unknown" : ", est."})`);
  if (plan.cost.peakGbUpper > 0) L.push(`  upper     ${upperBoundPhrase(plan).slice("upper bound ".length)}`);
  const bl = borderlineSentence(plan);
  if (bl) L.push(`  measure   ${bl}`);
  const share = shareLine(plan);
  if (share) L.push(`  share     ${share}`);
  const att = attentionPhrase(plan.cost.attention, plan.cost.attentionScoresGb);
  if (att) L.push(`  attention ${att.slice("attention: ".length)}`);
  const mbLine = maxBatchLine(plan);
  if (mbLine) L.push(`  max batch ${mbLine}`);
  const sbLine = safeBatchLine(plan);
  if (sbLine) L.push(`  safe batch ${sbLine}`);
  const split = splitLine(plan);
  if (split) L.push(`  split     ${split}`);
  L.push(
    plan.cost.estCostUsd != null ? `  train     ~${fmtUsd(plan.cost.estCostUsd)} / ${fmtDuration(plan.cost.estTrainSec)} on ${plan.cost.fitsGpu ?? "the priced GPU"}   (10 epochs, batch 32, 50k samples assumed)` : "  train     cost not estimable"
  );
  if (plan.run.blockers.length) {
    L.push("  blockers");
    L.push(...clip(plan.run.blockers, (b) => `    ! ${b.title}`));
  }
  if (plan.run.warnings.length) {
    L.push("  warnings");
    L.push(...clip(plan.run.warnings, (w) => `    - ${w.title}`));
  }
  if (plan.policy?.applied) {
    L.push(`  policy    ${plan.policy.violations === 0 ? "all lines pass" : `${plan.policy.violations} line${plan.policy.violations === 1 ? "" : "s"} violated (listed above)`}`);
  }
  if (plan.history?.rows.length) {
    const more = plan.history.rows.length - 1;
    L.push(`  history   ${historySentence(plan.history.rows[0])}${more > 0 ? ` (+${more} earlier)` : ""}`);
  }
  if (plan.change) {
    L.push(...changeLines(plan.change));
  }
  const d = plan.diff;
  if (d) {
    L.push(`Changes vs ${d.base.name}:`);
    const pct2 = d.paramsDeltaPct == null ? "" : ` (${d.paramsDeltaPct >= 0 ? "+" : ""}${d.paramsDeltaPct.toFixed(0)}%)`;
    const mem = d.memoryDeltaGb == null ? "" : `   memory ${d.memoryDeltaGb >= 0 ? "+" : "-"}${Math.abs(d.memoryDeltaGb).toFixed(2)} GB`;
    const cost = d.costDeltaUsd == null ? "" : `   cost ${fmtSignedUsd(d.costDeltaUsd)}`;
    L.push(`  +${d.added} -${d.removed} ~${d.changed}   params ${fmtSigned(d.paramsDelta)}${pct2}${mem}${cost}   blast radius ${d.blastRadius} layer${d.blastRadius === 1 ? "" : "s"}`);
    L.push(...clip(d.lines, (l) => `  ${l}`));
    if (d.legalityChanged === "became_illegal") L.push("  LEGALITY  base ran, this does not");
    if (d.legalityChanged === "became_legal") L.push("  LEGALITY  base did not run, this does");
  }
  if (plan.edit?.findings.length) {
    L.push(`Edits that run and are wrong (${plan.edit.checked.join(" + ")}):`);
    L.push(...clip(plan.edit.findings, (x) => `  ${x.severity === "block" ? "!" : "-"} ${x.rule}: ${x.title}`));
  }
  return L.join("\n");
}
function renderPlanMarkdown(plan) {
  const m = plan.model;
  const d = plan.diff;
  const L = [];
  L.push(`**${plan.summary}**`, "");
  L.push("| | |", "|---|---|");
  L.push(`| Layers / params | ${m.layers} / ${fmtParamCount(m.params)}${d ? ` (${fmtSigned(d.paramsDelta)}${d.paramsDeltaPct == null ? "" : `, ${d.paramsDeltaPct >= 0 ? "+" : ""}${d.paramsDeltaPct.toFixed(0)}%`})` : ""} |`);
  L.push(`| Shape | \`${fmtShape(m.inputShape)}\` -> \`${fmtShape(m.outputShape)}\` |`);
  L.push(`| Will it run | ${plan.run.legal == null ? `**cannot say**: ${plan.run.withheld ?? "the read was incomplete"}` : plan.run.legal ? `yes, ${plan.run.warnings.length} warning${plan.run.warnings.length === 1 ? "" : "s"}` : `**no**, ${plan.run.blockers.length} blocker${plan.run.blockers.length === 1 ? "" : "s"}`}${plan.run.legal == null ? "" : d?.legalityChanged === "became_illegal" ? " (**base ran, this does not**)" : d?.legalityChanged === "became_legal" ? " (base did not run, this does)" : ""} |`);
  L.push(`| GPU fit | T4 ${fitWord(plan, "T4")} \xB7 A100 40GB ${fitWord(plan, "A100_40")} \xB7 H100 80GB ${fitWord(plan, "H100_80")}${d?.memoryDeltaGb != null ? ` (memory ${d.memoryDeltaGb >= 0 ? "+" : "-"}${Math.abs(d.memoryDeltaGb).toFixed(2)} GB)` : ""}, peak ~${plan.cost.peakGb.toFixed(1)} GB at batch ${batchOf(plan)}${tokensAt(plan.cost.seqLen)}${plan.cost.activationGb == null ? " (activations unknown)" : " (est.)"} |`);
  if (plan.cost.peakGbUpper > 0) L.push(`| Upper bound | ${upperBoundPhrase(plan).slice("upper bound ".length)} |`);
  const blm = borderlineSentence(plan);
  if (blm) L.push(`| **Borderline** | ${blm.slice(blm.indexOf(": ") + 2)} (${blm.slice("borderline on ".length, blm.indexOf(": "))}) |`);
  const shm = shareLine(plan);
  if (shm) L.push(`| Shareable | ${shm} |`);
  const att = attentionPhrase(plan.cost.attention, plan.cost.attentionScoresGb);
  if (att) L.push(`| Attention | ${att.slice("attention: ".length)} |`);
  const mb = maxBatchLine(plan);
  if (mb) L.push(`| Max batch | ${mb} |`);
  const sb = safeBatchLine(plan);
  if (sb) L.push(`| Safe batch | ${sb} |`);
  const sp = splitLine(plan);
  if (sp) L.push(`| Two-card split | ${sp} |`);
  L.push(`| Est. training | ${plan.cost.estCostUsd != null ? `~${fmtUsd(plan.cost.estCostUsd)} / ${fmtDuration(plan.cost.estTrainSec)} on ${plan.cost.fitsGpu ?? "the priced GPU"}` : "not estimable"}${d?.costDeltaUsd != null ? ` (${fmtSignedUsd(d.costDeltaUsd)})` : ""} |`);
  if (d) L.push(`| Blast radius | ${d.blastRadius} downstream layer${d.blastRadius === 1 ? "" : "s"} from ${d.added + d.changed} touched |`);
  L.push("");
  if (plan.run.blockers.length) {
    L.push(plan.run.legal == null ? "**Findings from an incomplete read** (unverified: a layer this read could not open may explain, or cause, any of them)" : "**Blockers**");
    for (const b of plan.run.blockers.slice(0, 6)) L.push(`- ${b.title}${b.detail && b.stage !== "edit" ? `: ${b.detail}` : ""}${b.fix ? ` Fix: ${b.fix}` : ""}`);
    if (plan.run.blockers.length > 6) L.push(`- and ${plan.run.blockers.length - 6} more`);
    L.push("");
  }
  if (plan.run.warnings.length) {
    L.push("**Warnings**");
    for (const w of plan.run.warnings.slice(0, 6)) L.push(`- ${w.title}${w.detail && w.stage !== "edit" ? `: ${w.detail}` : ""}`);
    if (plan.run.warnings.length > 6) L.push(`- and ${plan.run.warnings.length - 6} more`);
    L.push("");
  }
  if (plan.policy?.applied) {
    L.push("**Policy**");
    const lines = [...plan.run.blockers, ...plan.run.warnings].filter((x) => x.stage === "policy");
    if (lines.length === 0) L.push("- every policy line passes");
    for (const x of lines.slice(0, 8)) L.push(`- ${x.title}${x.detail ? ` (\`${x.detail}\`)` : ""}`);
    L.push("");
  }
  if (plan.history?.rows.length) {
    const more = plan.history.rows.length - 1;
    L.push(`${historySentence(plan.history.rows[0])}${more > 0 ? ` (+${more} earlier run${more === 1 ? "" : "s"} on this structure)` : ""}`);
    L.push("");
  }
  if (plan.change) {
    const ch = plan.change;
    L.push(`**Change** (spec \`${ch.specId}\` on base \`${ch.baseFingerprint || "n/a"}\`)`);
    L.push(`- ${describeChangeSpec(ch.spec)}`);
    L.push(`- gate: ${changeGateSentence(ch)}`);
    for (const x of [...ch.gate.blockers, ...ch.gate.warnings].slice(0, 8)) L.push(`- ${x.title}${x.detail ? ` (\`${x.detail}\`)` : ""}`);
    L.push("");
  }
  if (plan.edit?.findings.length) {
    L.push(renderEditFindingsMarkdown(plan.edit.findings, `**Edits that run and are wrong** (${plan.edit.checked.join(" + ")} compared with the base; nothing here raises an error)`));
    L.push("");
  }
  if (d && d.lines.length) {
    L.push(`**Changes vs ${d.base.name}** (+${d.added} -${d.removed} ~${d.changed})`);
    L.push("```");
    L.push(...d.lines.slice(0, 12));
    if (d.lines.length > 12) L.push(`... ${d.lines.length - 12} more`);
    L.push("```", "");
  }
  L.push(`<sub>${plan.cost.assumptions}</sub>`);
  return L.join("\n");
}

// src/utils/oomDiagnosis.ts
var UNIT = { b: 1e-9, kib: 1024 / 1e9, kb: 1e-6, mib: 1024 ** 2 / 1e9, mb: 1e-3, gib: 1024 ** 3 / 1e9, gb: 1, tib: 1024 ** 4 / 1e9 };
var num5 = (v, unit) => Number(v.replace(/,/g, "")) * (UNIT[unit.toLowerCase()] ?? NaN);
var SIZE = String.raw`([\d.,]+)\s*(B|KiB|KB|MiB|MB|GiB|GB|TiB)`;
function grab(text, re) {
  const m = re.exec(text);
  if (!m) return null;
  const v = num5(m[1], m[2]);
  return Number.isFinite(v) ? v : null;
}
function parseOomMessage(text) {
  if (!/out of memory/i.test(text)) return null;
  const t = text.replace(/\s+/g, " ");
  const triedGb = grab(t, new RegExp(String.raw`Tried to allocate ${SIZE}`, "i"));
  const total2 = grab(t, new RegExp(String.raw`total capacity of ${SIZE}`, "i"));
  if (total2 != null) {
    const allocatedGb = grab(t, new RegExp(String.raw`${SIZE} is allocated by PyTorch`, "i"));
    const reservedUnallocatedGb = grab(t, new RegExp(String.raw`${SIZE} is reserved by PyTorch but unallocated`, "i"));
    let other = 0;
    for (const m of t.matchAll(new RegExp(String.raw`Process \d+ has ${SIZE} memory in use`, "gi"))) {
      const v = num5(m[1], m[2]);
      if (Number.isFinite(v)) other += v;
    }
    const processInUseGb = grab(t, new RegExp(String.raw`this process has ${SIZE} memory in use`, "i"));
    return {
      triedGb,
      totalGb: total2,
      freeGb: grab(t, new RegExp(String.raw`of which ${SIZE} is free`, "i")),
      processInUseGb,
      allocatedGb,
      reservedUnallocatedGb,
      reservedGb: allocatedGb != null && reservedUnallocatedGb != null ? allocatedGb + reservedUnallocatedGb : null,
      otherProcessesGb: processInUseGb != null ? other : 0,
      format: "torch2"
    };
  }
  const total1 = grab(t, new RegExp(String.raw`${SIZE} total capacity`, "i"));
  if (total1 != null) {
    const allocatedGb = grab(t, new RegExp(String.raw`${SIZE} already allocated`, "i"));
    const reservedGb = grab(t, new RegExp(String.raw`${SIZE} reserved in total by PyTorch`, "i"));
    return {
      triedGb,
      totalGb: total1,
      freeGb: grab(t, new RegExp(String.raw`${SIZE} free`, "i")),
      processInUseGb: null,
      allocatedGb,
      reservedUnallocatedGb: allocatedGb != null && reservedGb != null ? Math.max(0, reservedGb - allocatedGb) : null,
      reservedGb,
      otherProcessesGb: 0,
      format: "torch1"
    };
  }
  return { triedGb, totalGb: null, freeGb: null, processInUseGb: null, allocatedGb: null, reservedUnallocatedGb: null, reservedGb: null, otherProcessesGb: 0, format: "unknown" };
}
var FRAGMENTATION_SHARE = 0.1;
var OTHER_PROCESS_GB = 1;
var FITS_WITH_ROOM = 0.85;
var GIANT_REQUEST_SHARE = 0.2;
var HEADROOM2 = 1.15;
var LATE_STEP = 5;
var f2 = (x) => x >= 10 ? x.toFixed(1) : x.toFixed(2);
function batchThatFits(facts, cardGb, k = 1) {
  const { footprintGb, activationPerSampleGb } = facts;
  if (footprintGb == null || activationPerSampleGb == null || !(activationPerSampleGb > 0)) return null;
  const b = Math.floor((cardGb / HEADROOM2 - footprintGb * k) / (activationPerSampleGb * k));
  return b >= 1 ? b : null;
}
function inferenceSizeFixes(facts, cardGb) {
  const out = [];
  const b = batchThatFits(facts, cardGb);
  if (b != null && facts.batch != null && b < facts.batch) {
    out.push({
      action: `generate at batch ${b} and run the rest as further batches`,
      because: `${b} is the largest batch whose weights plus KV cache fit ${f2(cardGb)} GB with ${Math.round((HEADROOM2 - 1) * 100)}% headroom`,
      changesTraining: false
    });
  }
  if (facts.activationPerSampleGb != null && facts.batch != null && facts.footprintGb != null) {
    const kv = facts.activationPerSampleGb * facts.batch;
    out.push({
      action: "shorten the prompt or max_new_tokens, or quantize the KV cache to 8 bits",
      because: `the KV cache is ${f2(kv)} GB of the predicted peak and grows with every token kept`,
      changesTraining: false
    });
  }
  if (facts.footprintGb != null && facts.footprintGb * HEADROOM2 > cardGb * 0.5) {
    out.push({
      action: 'split the weights across cards (tensor parallel, or device_map="auto")',
      because: `the weights alone are ${f2(facts.footprintGb)} GB, more than half the card`,
      changesTraining: false
    });
    out.push({
      action: "load the weights in 8 or 4 bits",
      because: "halving or quartering the weight bytes is the largest single reduction",
      changesTraining: true
    });
  }
  return out;
}
function sizeFixes(facts, cardGb) {
  if (facts.mode === "inference") return inferenceSizeFixes(facts, cardGb);
  const out = [];
  const b = batchThatFits(facts, cardGb);
  if (b != null && facts.batch != null && b < facts.batch) {
    let micro = b;
    while (micro > 1 && facts.batch % micro !== 0) micro--;
    const accum = facts.batch / micro;
    out.push({
      action: `batch ${micro} with gradient accumulation of ${accum} steps`,
      because: `${b} is the largest batch whose predicted peak fits ${f2(cardGb)} GB with ${Math.round((HEADROOM2 - 1) * 100)}% headroom; ${micro} x ${accum} keeps the effective batch at ${facts.batch}`,
      changesTraining: false
    });
  }
  if (facts.activationPerSampleGb != null && facts.batch != null && facts.footprintGb != null && facts.footprintGb * HEADROOM2 <= cardGb) {
    const act = facts.activationPerSampleGb * facts.batch;
    out.push({
      action: "turn on activation (gradient) checkpointing",
      because: `activations are ${f2(act)} GB of the predicted peak at this batch; checkpointing recomputes most of them in the backward pass, at roughly a third more compute`,
      changesTraining: false
    });
  }
  if (facts.footprintGb != null && facts.footprintGb > cardGb * 0.5) {
    out.push({
      action: "shard the optimizer state and gradients across cards (FSDP or ZeRO), or use an 8-bit optimizer",
      because: `weights, gradients and optimizer state alone are ${f2(facts.footprintGb)} GB, more than half the card`,
      changesTraining: false
    });
    out.push({
      action: "train adapters (LoRA) on a frozen base instead of every weight",
      because: "the optimizer state then covers the adapters only",
      changesTraining: true
    });
  }
  return out;
}
function diagnoseOom(text, facts = {}) {
  const m = parseOomMessage(text);
  const evidence = [];
  const card = m?.totalGb ?? null;
  if (m?.triedGb != null) evidence.push(`the failing allocation asked for ${f2(m.triedGb)} GB`);
  if (card != null) evidence.push(`the card has ${f2(card)} GB`);
  if (m?.allocatedGb != null) evidence.push(`${f2(m.allocatedGb)} GB was held by live tensors`);
  if (m?.reservedUnallocatedGb != null) evidence.push(`${f2(m.reservedUnallocatedGb)} GB was reserved by PyTorch but held no tensor`);
  const inference = facts.mode === "inference";
  if (facts.predictedPeakGb != null) {
    evidence.push(inference ? `the plan predicts a ${f2(facts.predictedPeakGb)} GB inference peak at this batch (weights ${f2(facts.footprintGb ?? 0)} GB plus KV cache)` : `the plan predicts a ${f2(facts.predictedPeakGb)} GB training peak at this batch`);
  }
  const result = (cause, summary, fixes) => ({ cause, summary, evidence, fixes, message: m, mode: facts.mode === "inference" ? "inference" : "train" });
  if (!m) {
    return result("unknown", "This is not a CUDA out-of-memory message, or it has none of the numbers PyTorch prints with one.", []);
  }
  if (card != null && facts.predictedPeakGb != null && facts.predictedPeakGb * HEADROOM2 > card) {
    return result(
      "too-big-for-card",
      `The job is bigger than this card: the predicted peak is ${f2(facts.predictedPeakGb)} GB (${f2(facts.predictedPeakGb * HEADROOM2)} GB with headroom) on a ${f2(card)} GB card.`,
      sizeFixes(facts, card)
    );
  }
  if (m.otherProcessesGb >= OTHER_PROCESS_GB) {
    evidence.push(`other processes on the card held ${f2(m.otherProcessesGb)} GB`);
    return result(
      "other-process",
      `Another process holds ${f2(m.otherProcessesGb)} GB of this card; the job itself may fit.`,
      [{ action: "run the job on a card nothing else is using (check nvidia-smi), or stop the other process", because: `${f2(m.otherProcessesGb)} GB is held outside this process`, changesTraining: false }]
    );
  }
  if (card != null && m.reservedUnallocatedGb != null && m.triedGb != null && (m.reservedUnallocatedGb >= m.triedGb || m.reservedUnallocatedGb >= card * FRAGMENTATION_SHARE)) {
    return result(
      "fragmentation",
      `Memory is fragmented: ${f2(m.reservedUnallocatedGb)} GB is reserved and empty, ${m.reservedUnallocatedGb >= m.triedGb ? "more than" : "a large share next to"} the ${f2(m.triedGb)} GB request that failed.`,
      [
        { action: "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before the process starts", because: "it lets the allocator grow segments instead of needing one free block of the requested size", changesTraining: false },
        { action: "keep tensor shapes steady between steps (pad or bucket sequence lengths)", because: "varying shapes leave differently sized holes in the cache", changesTraining: false }
      ]
    );
  }
  if (card != null && m.triedGb != null && m.triedGb >= card * GIANT_REQUEST_SHARE) {
    if (facts.vocabSize && facts.seqLen && facts.batch) {
      const logitsGb = facts.batch * facts.seqLen * facts.vocabSize * 4 / 1e9;
      evidence.push(`fp32 logits for batch ${facts.batch} x ${facts.seqLen} tokens x ${facts.vocabSize} vocabulary are ${f2(logitsGb)} GB`);
      if (logitsGb >= m.triedGb * 0.5) {
        return result(
          "large-logits",
          `The failing ${f2(m.triedGb)} GB request matches the output logits: ${f2(logitsGb)} GB in fp32 over a ${facts.vocabSize.toLocaleString("en-US")}-token vocabulary.`,
          [
            { action: "compute the loss in chunks over the sequence (chunked or fused cross-entropy, e.g. Liger kernels or Cut Cross-Entropy)", because: "the full logits tensor is never materialised", changesTraining: false },
            { action: "compute logits only for the positions the loss uses", because: "prompt tokens with no label do not need logits", changesTraining: false }
          ]
        );
      }
    }
    return result(
      "undeclared-memory",
      `A single ${f2(m.triedGb)} GB request, ${Math.round(m.triedGb / card * 100)}% of the card, failed. That is one very large tensor: typically attention scores under eager attention, logits over a large vocabulary, or an unusually long sample.`,
      [
        { action: "use SDPA or flash attention instead of eager attention", because: "they do not store the full attention-score matrix for backward", changesTraining: false },
        { action: "cap or bucket the sequence length, and check the longest sample in the dataset", because: "one long sample sets the size of that tensor", changesTraining: false }
      ]
    );
  }
  if (!inference && card != null && facts.predictedPeakGb != null && facts.predictedUpperGb != null && facts.predictedUpperGb * HEADROOM2 > card && !(facts.step != null && facts.step > LATE_STEP)) {
    const k = facts.upperK ?? facts.predictedUpperGb / facts.predictedPeakGb;
    const cov = facts.upperCoverage != null ? `${Math.round(facts.upperCoverage * 100)}% of held-out measured runs peaked at or under it` : "the conservative estimate";
    evidence.push(`the conservative peak is ${f2(facts.predictedUpperGb)} GB (${f2(facts.predictedUpperGb * HEADROOM2)} GB with headroom): ${cov}`);
    const fixes = [];
    const b = batchThatFits(facts, card, k);
    if (b != null && facts.batch != null && b < facts.batch) {
      let micro = b;
      while (micro > 1 && facts.batch % micro !== 0) micro--;
      const accum = facts.batch / micro;
      fixes.push({
        action: `batch ${micro} with gradient accumulation of ${accum} steps`,
        because: `${b} is the largest batch whose conservative peak fits ${f2(card)} GB with ${Math.round((HEADROOM2 - 1) * 100)}% headroom; ${micro} x ${accum} keeps the effective batch at ${facts.batch}`,
        changesTraining: false
      });
    }
    fixes.push({
      action: "measure one step before the full run: log torch.cuda.max_memory_allocated() after the first optimizer step",
      because: "at the edge of a card the estimate is a fit check, not a precise number; one measured step settles it",
      changesTraining: false
    });
    return result(
      "borderline",
      `The job was borderline on this card: the predicted peak of ${f2(facts.predictedPeakGb)} GB fits ${f2(card)} GB with headroom, but the conservative peak of ${f2(facts.predictedUpperGb)} GB does not. This out-of-memory is inside the estimate's own error, not evidence of memory the model does not declare.`,
      fixes
    );
  }
  if (card != null && facts.predictedPeakGb != null && facts.predictedPeakGb <= card * FITS_WITH_ROOM) {
    const late = facts.step != null && facts.step > LATE_STEP;
    return result(
      late ? "grows-during-run" : "undeclared-memory",
      late ? `The plan says ${f2(facts.predictedPeakGb)} GB fits this ${f2(card)} GB card, and the job ran ${facts.step} steps first: memory grew during the run, from a leak or from a longer batch.` : `The plan says ${f2(facts.predictedPeakGb)} GB fits this ${f2(card)} GB card, so the job holds memory the model does not declare: an evaluation pass, a second model, eager attention, or a longer sequence than planned.`,
      late ? [
        { action: "log torch.cuda.max_memory_allocated() each step and look for a steady climb", because: "a climb is a leak (tensors kept across steps, e.g. a loss appended without .item()); a jump is one long batch", changesTraining: false },
        { action: "check the longest sample in the dataset and cap or bucket it", because: "a data-dependent spike appears at the step that draws it", changesTraining: false }
      ] : inference ? [
        { action: "run generation under torch.inference_mode() (or torch.no_grad())", because: "without it every forward keeps activations for a backward pass that never comes", changesTraining: false },
        { action: "confirm the prompt length, max_new_tokens and batch match what was planned", because: "the KV cache grows with all three", changesTraining: false }
      ] : [
        { action: "run the eval pass under torch.no_grad() and with its own smaller batch", because: "evaluation at the training batch can need more than training does", changesTraining: false },
        { action: "confirm the sequence length and attention implementation match what was planned", because: "both multiply activation memory", changesTraining: false }
      ]
    );
  }
  return result(
    "unknown",
    facts.predictedPeakGb == null ? "The message alone does not say why. Give the model, batch and sequence length so the predicted peak can be compared with the card." : "The predicted peak is close to the card; small differences in what the run holds decide it.",
    card != null ? sizeFixes(facts, card) : []
  );
}
function renderOomText(d) {
  const L = [`cause    ${d.cause}`, "", d.summary, ""];
  if (d.evidence.length) L.push("read from", ...d.evidence.map((e) => `  - ${e}`), "");
  if (d.fixes.length) {
    L.push("fixes, least invasive first");
    const changes = d.mode === "inference" ? "  (changes the model's outputs)" : "  (changes what is trained)";
    d.fixes.forEach((f, i) => L.push(`  ${i + 1}. ${f.action}${f.changesTraining ? changes : ""}`, `     ${f.because}`));
  }
  return L.join("\n") + "\n";
}

// src/utils/inferenceMemory.ts
var DEFAULT_INFERENCE_PRECISION = "bf16";
var DEFAULT_GEN_TOKENS = 256;
var DEFAULT_INFERENCE_BATCH = 1;
var INFERENCE_NOT_COUNTED = [
  "the prefill working set (one layer's activations and attention at the prompt length) and the logits over the prompt",
  "the CUDA context and any framework reserve (vLLM, for one, preallocates a fixed share of the card for its cache)",
  "a second model, vision tower or tokenizer-side buffers loaded beside this one"
];
function inferencePeakFor(model, opts) {
  const precision = opts.precision ?? DEFAULT_INFERENCE_PRECISION;
  const batch = Math.max(1, Math.floor(opts.batch ?? DEFAULT_INFERENCE_BATCH));
  const promptTokens = Math.max(1, Math.floor(opts.promptTokens));
  const genTokens = Math.max(0, Math.floor(opts.genTokens ?? DEFAULT_GEN_TOKENS));
  const bytes = computeQuantPreview(model).totalBytes[precision === "bf16" ? "fp16" : precision];
  if (!(bytes > 0)) return null;
  const kvValueBytes = precision === "fp32" ? 4 : 2;
  const tokens = promptTokens + genTokens;
  const kvBytesPerToken = kvBytesPerTokenForModel(model, kvValueBytes, tokens);
  const kvPerSequenceGb = kvBytesPerToken * tokens / 1e9;
  const weightsGb = bytes / 1e9;
  const kvGb = kvPerSequenceGb * batch;
  const stateBytesPerSequence = recurrentStateBytesForModel(model, kvValueBytes);
  const stateGb = stateBytesPerSequence * batch / 1e9;
  const peakGb = weightsGb + kvGb + stateGb;
  const fits = PREFLIGHT_GPUS.map((g) => ({ gpu: g.name, gb: g.gb, fits: peakGb * FIT_HEADROOM <= g.gb }));
  const smallest = [...fits].filter((f) => f.fits).sort((a, b) => a.gb - b.gb)[0];
  return {
    precision,
    batch,
    promptTokens,
    genTokens,
    weightsGb,
    kvBytesPerToken,
    kvPerSequenceGb,
    kvGb,
    stateBytesPerSequence,
    stateGb,
    peakGb,
    headroom: FIT_HEADROOM,
    fits,
    smallestFitGpu: smallest?.gpu ?? null,
    notCounted: INFERENCE_NOT_COUNTED
  };
}

// src/utils/oomFacts.ts
var DEFAULT_OOM_TRAIN_BATCH = 32;
function vocabOf(model) {
  for (const c of model.components) {
    const v = Number(c.params?.vocabSize);
    if (Number.isFinite(v) && v > 0) return v;
  }
  return null;
}
function factsFor2(model, batch, seqLen, attention, step) {
  const sized = withSeqLen(model, seqLen);
  const plan = buildPlan(sized.model, { run: { batchSize: batch }, attention });
  const act = plan.cost.activationGb;
  return {
    predictedPeakGb: plan.cost.peakGb,
    predictedUpperGb: plan.cost.peakGbUpper,
    upperK: plan.cost.upperBound.multiplier,
    upperCoverage: plan.cost.upperBound.coverage,
    footprintGb: act != null ? plan.cost.peakGb - act : null,
    activationPerSampleGb: act != null && batch > 0 ? act / batch : null,
    batch,
    step,
    vocabSize: vocabOf(model),
    seqLen: sized.seqLen
  };
}
function inferenceFactsFor(model, batch, promptTokens, inference, step) {
  const prompt = promptTokens ?? withSeqLen(model, null).seqLen ?? DEFAULT_SEQ_LEN;
  const ip = inferencePeakFor(model, { batch, promptTokens: prompt, genTokens: inference.genTokens, precision: inference.precision });
  if (!ip) return { mode: "inference", batch, step };
  return {
    mode: "inference",
    predictedPeakGb: ip.peakGb,
    footprintGb: ip.weightsGb,
    activationPerSampleGb: ip.kvPerSequenceGb,
    batch,
    step,
    vocabSize: vocabOf(model),
    seqLen: ip.promptTokens
  };
}
function oomFactsForModel(model, opts = {}) {
  return opts.inference ? inferenceFactsFor(model, opts.batch ?? DEFAULT_INFERENCE_BATCH, opts.seqLen ?? null, opts.inference, opts.step ?? null) : factsFor2(model, opts.batch ?? DEFAULT_OOM_TRAIN_BATCH, opts.seqLen ?? null, opts.attention ?? "sdpa", opts.step ?? null);
}

// src/utils/configDataFit.ts
var CLASSIFIER_ARCH = /For(Sequence|Token|Image|Audio|MultipleChoice)Classification$/;
var posInt = (v) => {
  const n2 = typeof v === "string" && v.trim() !== "" ? Number(v) : v;
  return typeof n2 === "number" && Number.isInteger(n2) && n2 > 0 ? n2 : null;
};
function inputContractOf(model, config) {
  const comps = model.components ?? [];
  const first = (t) => comps.find((c) => c.type === t);
  const source = {};
  const emb = first("embedding");
  const vocabSize = posInt(emb?.params?.vocabSize) ?? posInt(emb?.params?.numEmbeddings);
  if (vocabSize != null) source.vocabSize = "the token embedding";
  const patch = first("patchEmbed");
  const learned = first("learnedPositionalEmbedding");
  const sinusoidal = first("positionalEncoding");
  let maxPositions = null;
  let positionKind = null;
  if (!patch) {
    if (learned && posInt(learned.params?.maxLen) != null) {
      maxPositions = posInt(learned.params?.maxLen);
      positionKind = "learned";
      source.maxPositions = "the learned position table";
    } else if (sinusoidal && posInt(sinusoidal.params?.maxLen) != null) {
      maxPositions = posInt(sinusoidal.params?.maxLen);
      positionKind = "buffer";
      source.maxPositions = "the positional encoding buffer";
    } else if (posInt(emb?.params?.maxSeqLen) != null) {
      maxPositions = posInt(emb?.params?.maxSeqLen);
      positionKind = comps.some((c) => c.type === "rope") ? "rotary" : "declared";
      source.maxPositions = positionKind === "rotary" ? "max_position_embeddings (rotary)" : "the declared maximum length";
    }
  }
  let imageSize = null;
  let channels = null;
  if (patch) {
    imageSize = posInt(patch.params?.imgSize);
    channels = posInt(patch.params?.inChans);
    if (imageSize != null) source.imageSize = "the patch embedding";
    if (channels != null) source.channels = "the patch embedding";
  } else if (!emb) {
    const shape = first("input")?.params?.shape ?? null;
    if (Array.isArray(shape) && shape.length === 3 && shape.every((x) => posInt(x) != null)) {
      channels = shape[0];
      source.channels = "the input shape";
    }
  }
  let numLabels = null;
  const archs = Array.isArray(config?.architectures) ? config.architectures.map(String) : [];
  if (archs.some((a) => CLASSIFIER_ARCH.test(a))) {
    const id2label = config?.id2label;
    numLabels = id2label && typeof id2label === "object" ? Object.keys(id2label).length || null : posInt(config?.num_labels);
    if (numLabels != null) source.numLabels = `the ${archs.find((a) => CLASSIFIER_ARCH.test(a))} head`;
  } else if (!emb) {
    const tail = [...comps].reverse().find((c) => !["output", "softmax", "dropout", "flatten", "activation", "relu"].includes(c.type));
    if (tail?.type === "linear" && posInt(tail.params?.outFeatures) != null) {
      numLabels = posInt(tail.params?.outFeatures);
      source.numLabels = "the final Linear";
    }
  }
  return { vocabSize, maxPositions, positionKind, numLabels, imageSize, channels, source };
}
function readDataSpec(raw) {
  if (raw == null || typeof raw !== "object") return { error: "`data` must be an object." };
  const r = raw;
  const fields = [
    ["seqLen", ["seq_len", "seqLen"]],
    ["tokenizerSize", ["tokenizer_size", "tokenizerSize", "vocab_size"]],
    ["numClasses", ["num_classes", "numClasses", "num_labels"]],
    ["imageSize", ["image_size", "imageSize"]],
    ["channels", ["channels", "num_channels"]]
  ];
  const out = {};
  for (const [key, names] of fields) {
    const name = names.find((n2) => r[n2] != null);
    if (!name) continue;
    const v = posInt(r[name]);
    if (v == null) return { error: `\`data.${name}\` must be a positive integer.` };
    out[key] = v;
  }
  return out;
}
function checkDataFit(contract, data) {
  const checked = [];
  const skipped = [];
  const findings = [];
  const c = contract;
  if (data.seqLen != null) {
    if (c.maxPositions == null) {
      skipped.push({ check: "sequence length", reason: c.imageSize != null ? "a vision model: its position table is checked against the image size" : "the model declares no maximum length" });
    } else {
      checked.push("sequence length");
      if (data.seqLen > c.maxPositions) {
        if (c.positionKind === "learned" || c.positionKind === "buffer") {
          findings.push({
            rule: "seq-over-positions",
            severity: "block",
            title: `Sequences of ${data.seqLen} tokens are longer than the ${c.maxPositions}-row position ${c.positionKind === "learned" ? "table" : "buffer"}`,
            detail: c.positionKind === "learned" ? `Position ids past ${c.maxPositions - 1} index outside a learned table, so the first batch that long raises an IndexError (a device-side assert on GPU) after the reservation is taken.` : `The encoding is a precomputed ${c.maxPositions}-row buffer, so adding it to a longer sequence fails on shape at the first batch that long, after the reservation is taken.`,
            fix: `Truncate or chunk to ${c.maxPositions} tokens, or pick a model whose context covers ${data.seqLen}.`
          });
        } else {
          findings.push({
            rule: "seq-over-context",
            severity: "warn",
            title: `Sequences of ${data.seqLen} tokens are past the model's ${c.maxPositions}-token context`,
            detail: c.positionKind === "rotary" ? "Rotary positions are computed, so the job runs; quality past the trained length is not something the config can vouch for, and attention memory grows with the length." : "The config declares this as the longest input; past it, behaviour depends on the implementation.",
            fix: c.positionKind === "rotary" ? "Set rope_scaling for the longer context, or truncate to the trained length." : `Truncate to ${c.maxPositions} tokens.`
          });
        }
      }
    }
  }
  if (data.tokenizerSize != null) {
    if (c.vocabSize == null) skipped.push({ check: "tokenizer size", reason: "the model has no token embedding" });
    else {
      checked.push("tokenizer size");
      if (data.tokenizerSize > c.vocabSize) {
        findings.push({
          rule: "tokenizer-over-vocab",
          severity: "block",
          title: `The tokenizer has ${data.tokenizerSize} ids and the embedding has ${c.vocabSize} rows`,
          detail: `Any of the ${data.tokenizerSize - c.vocabSize} ids past ${c.vocabSize - 1} indexes outside the embedding. The usual cause is special tokens added to the tokenizer without resizing the model.`,
          fix: "Call model.resize_token_embeddings(len(tokenizer)) before training, and budget for the new rows."
        });
      }
    }
  }
  if (data.numClasses != null) {
    if (c.numLabels == null) skipped.push({ check: "label count", reason: "no classification head is visible in the config or the graph" });
    else {
      checked.push("label count");
      if (data.numClasses > c.numLabels) {
        findings.push({
          rule: "labels-over-head",
          severity: "block",
          title: `The data has ${data.numClasses} labels and ${c.source.numLabels ?? "the head"} emits ${c.numLabels}`,
          detail: `A label id of ${c.numLabels} or more is out of range for the loss, which on GPU is a device-side assert with no line number.`,
          fix: `Load with num_labels=${data.numClasses} (and ignore_mismatched_sizes=True for a pretrained head).`
        });
      } else if (data.numClasses < c.numLabels) {
        findings.push({
          rule: "labels-under-head",
          severity: "warn",
          title: `The data has ${data.numClasses} labels and the head emits ${c.numLabels}`,
          detail: "It runs; the extra outputs are never a target, which is usually a head left over from a different task.",
          fix: `Load with num_labels=${data.numClasses} if the head is meant for this data.`
        });
      }
    }
  }
  if (data.imageSize != null) {
    if (c.imageSize == null) skipped.push({ check: "image size", reason: "the model has no fixed image size (no patch position table)" });
    else {
      checked.push("image size");
      if (data.imageSize !== c.imageSize) {
        findings.push({
          rule: "image-size-mismatch",
          severity: "block",
          title: `Images are ${data.imageSize}px and the position table was built for ${c.imageSize}px`,
          detail: "The patch count no longer matches the learned position table, so the forward pass raises on the first batch.",
          fix: `Resize to ${c.imageSize}px, or pass interpolate_pos_encoding=True where the model supports it.`
        });
      }
    }
  }
  if (data.channels != null) {
    if (c.channels == null) skipped.push({ check: "channels", reason: "the model states no input channels" });
    else {
      checked.push("channels");
      if (data.channels !== c.channels) {
        findings.push({
          rule: "channels-mismatch",
          severity: "block",
          title: `Images have ${data.channels} channel${data.channels === 1 ? "" : "s"} and the first layer expects ${c.channels}`,
          detail: "The first convolution or patch projection has a fixed input-channel count; a mismatch raises on the first batch.",
          fix: data.channels === 1 && c.channels === 3 ? "Repeat the grey channel to three in the transform." : `Convert the images to ${c.channels} channels.`
        });
      }
    }
  }
  return { checked, skipped, findings, contract };
}

// src/utils/nonGemmCost.ts
var NONGEMM_BW_EFFICIENCY = 0.9;
var KERNEL_GAP_US = 0.9;
var NONGEMM_RATES = {
  eager: {
    /** transformers' RMSNorm: fwd to(fp32), pow, mean, add+rsqrt, mul, to(bf16), mul weight = 36 B, 7 kernels; bwd 64 B, 10 kernels. */
    rmsNorm: { bytes: 100, kernels: 17 },
    /** torch.nn.LayerNorm: one fused kernel forward, two backward (input grad, weight grad). */
    layerNorm: { bytes: 16, kernels: 3 },
    /** Residual add forward; the gradient summed where the stream forks backward. */
    add: { bytes: 12, kernels: 2 },
    /** Per intermediate element: fwd silu, mul; bwd mul (two grads), silu backward. */
    swiglu: { bytes: 28, kernels: 5 },
    /** Per intermediate element: one GELU kernel each way. */
    gelu: { bytes: 10, kernels: 2 },
    /** Per q / k element: rotate_half (neg, cat), two muls and an add, for q and for k, each way. */
    rope: { bytes: 52, kernels: 22 },
    /** Qwen3 / OLMo-2 style RMSNorm on every q and k head, per q / k element: the RMSNorm count, twice the kernels (q and k). */
    qkNorm: { bytes: 100, kernels: 34 },
    /**
     * Backward only: the q, k, v gradients come out of attention in [B, H, T, hd]
     * and the projections' GEMMs need [B, T, H x hd], so each is copied (read 2 +
     * write 2) per q / k / v element.
     */
    attnLayout: { bytes: 4, kernels: 3 },
    /** Dropout with p > 0: fwd read, write, mask; bwd read grad and mask, write. */
    dropout: { bytes: 10, kernels: 2 }
  },
  fused: {
    /** One kernel forward (read x, write y, bf16), two backward (dx, partial dweight). */
    rmsNorm: { bytes: 10, kernels: 3 },
    layerNorm: { bytes: 12, kernels: 3 },
    /** Same bytes (the residual stream is still written and read); the kernel is folded into the neighbouring norm or bias-dropout-add. */
    add: { bytes: 12, kernels: 0 },
    /** Fused SwiGLU: fwd read gate, up, write 6 B; bwd read grad, gate, up, write two grads 10 B. */
    swiglu: { bytes: 16, kernels: 2 },
    /** Fused bias-GELU: fwd 4 B, bwd 6 B. */
    gelu: { bytes: 10, kernels: 2 },
    /** Fused RoPE: read and write each element once each way. */
    rope: { bytes: 8, kernels: 4 },
    qkNorm: { bytes: 10, kernels: 6 },
    attnLayout: { bytes: 0, kernels: 0 },
    /** Folded into the bias-dropout-add kernel: the mask still moves. */
    dropout: { bytes: 4, kernels: 0 }
  }
};
var GQA_BWD_BYTES_PER_Q_ELEM = 8;
var OPTIMIZER_BYTES_PER_PARAM = { eager: 40, fused: 34 };
var GRAD_ACCUM_BYTES_PER_PARAM = { eager: 6, fused: 0 };
var GEMM_KERNELS = 3;
var ATTN_KERNELS = 4;
var ATTN = /attention/i;
function nonGemmOf(c, out, stack, qkNorm) {
  if (!out.length) return { bytes: 0, kernels: 0 };
  const R = NONGEMM_RATES[stack];
  const elems = out.reduce((a, b) => a * b, 1);
  const width = out[out.length - 1] || 1;
  const tokens = elems / width;
  const p = c.params ?? {};
  const num7 = (v) => typeof v === "number" && v > 0 ? v : null;
  const op = (r, n2) => ({ bytes: r.bytes * n2, kernels: r.kernels });
  switch (c.type) {
    case "rmsNorm":
      return op(R.rmsNorm, elems);
    case "layerNorm":
      return op(R.layerNorm, elems);
    case "add":
      return op(R.add, elems);
    case "dropout": {
      const pr = num7(p.p) ?? num7(p.rate) ?? 0;
      return pr > 0 ? op(R.dropout, elems) : { bytes: 0, kernels: 0 };
    }
    case "swiglu":
    case "geglu": {
      const I = num7(p.intermediateSize) ?? num7(p.ffDim) ?? num7(p.hiddenFeatures);
      const g = I ? op(R.swiglu, tokens * I) : { bytes: 0, kernels: 0 };
      return { bytes: g.bytes, kernels: g.kernels + 3 * GEMM_KERNELS };
    }
    case "feedForward": {
      const I = num7(p.intermediateSize) ?? num7(p.ffDim) ?? num7(p.hiddenFeatures);
      const g = I ? op(R.gelu, tokens * I) : { bytes: 0, kernels: 0 };
      return { bytes: g.bytes, kernels: g.kernels + 2 * GEMM_KERNELS };
    }
    case "linear":
      return { bytes: 0, kernels: GEMM_KERNELS };
    case "embedding": {
      const V = num7(p.vocabSize), D = num7(p.embeddingDim) ?? width;
      return { bytes: (V ? V * D * 2 : 0) + tokens * D * 10, kernels: 3 };
    }
    default: {
      if (!ATTN.test(c.type)) return { bytes: 0, kernels: 0 };
      const E = num7(p.embedDim) ?? num7(p.hiddenDim) ?? width;
      const H = num7(p.numHeads);
      if (!H) return { bytes: 0, kernels: 4 * GEMM_KERNELS + ATTN_KERNELS };
      const hd = num7(p.headDim) ?? E / H;
      const kv = num7(p.numKVHeads) ?? H;
      const qk = tokens * (H + kv) * hd;
      let bytes = R.attnLayout.bytes * tokens * (H + 2 * kv) * hd;
      let kernels = 4 * GEMM_KERNELS + ATTN_KERNELS + R.attnLayout.kernels;
      if (qkNorm) {
        bytes += R.qkNorm.bytes * qk;
        kernels += R.qkNorm.kernels;
      }
      if (kv < H) bytes += GQA_BWD_BYTES_PER_Q_ELEM * tokens * H * hd;
      return { bytes, kernels };
    }
  }
}
function ropeOf(c, out, stack) {
  if (!out.length || !ATTN.test(c.type)) return { bytes: 0, kernels: 0 };
  const p = c.params ?? {};
  const num7 = (v) => typeof v === "number" && v > 0 ? v : null;
  const width = out[out.length - 1] || 1;
  const tokens = out.reduce((a, b) => a * b, 1) / width;
  const H = num7(p.numHeads);
  if (!H) return { bytes: 0, kernels: 0 };
  const hd = num7(p.headDim) ?? (num7(p.embedDim) ?? width) / H;
  const kv = num7(p.numKVHeads) ?? H;
  const r = NONGEMM_RATES[stack].rope;
  return { bytes: r.bytes * tokens * (H + kv) * hd, kernels: r.kernels };
}
function nonGemmMs(bytes, kernels, bwGBs) {
  if (!(bwGBs > 0)) return 0;
  return bytes / (bwGBs * 1e9 * NONGEMM_BW_EFFICIENCY) * 1e3 + kernels * KERNEL_GAP_US / 1e3;
}

// src/utils/gemmEfficiency.table.ts
var GEMM_EFFICIENCY_TABLE = {
  "sources": {
    "neusight": {
      "repo": "https://github.com/scai-tech/NeuSight",
      "commit": "6945927d9afcca2b9daf021f8395e53edc5b4eef",
      "licence": "MIT",
      "copyright": "Copyright (c) 2025 Georgia Institute of Technology and Microsoft Corporation",
      "files": [
        "scripts/asplos/data/dataset/train/linear.csv",
        "scripts/asplos/data/dataset/test/linear.csv",
        "scripts/asplos/fp16mm.csv"
      ]
    },
    "pytorch": {
      "repo": "https://github.com/pytorch/pytorch",
      "commit": "6a6db44e15f9678d573ad0af5e1f3e2b0713414a",
      "licence": "BSD-3-Clause",
      "files": [
        "torch/_inductor/analysis/device_info.py"
      ]
    }
  },
  "fit": {
    "script": "scripts/gemm-efficiency/fit_gemm_efficiency.py",
    "plateauMinUs": 2e3,
    "bucket": "half octave of time at the plateau, median, then isotonic (non-decreasing), capped at 1",
    "minRowsPerBucket": 8
  },
  "cards": {
    "h100-sxm": {
      "neusightDevice": "NVIDIA H100 80GB HBM3",
      "rows": 1040,
      "plateauFp32TFLOPS": 50.2,
      "fp32PeakTFLOPS": 67,
      "plateauOfFp32Peak": 0.749,
      "curve": [
        [5.25, 0.4625],
        [5.75, 0.492],
        [6.25, 0.5737],
        [6.75, 0.5775],
        [7.25, 0.6399],
        [7.75, 0.6527],
        [8.25, 0.7348],
        [8.75, 0.7348],
        [9.25, 0.8118],
        [9.75, 0.8118],
        [10.25, 0.8977],
        [10.75, 0.8977],
        [11.25, 0.9606],
        [11.75, 0.9606],
        [12.25, 0.9864],
        [12.75, 0.9864],
        [13.25, 0.9986],
        [13.75, 0.9986],
        [14.25, 1]
      ]
    },
    "l4": {
      "neusightDevice": "NVIDIA L4",
      "rows": 1040,
      "plateauFp32TFLOPS": 8.9,
      "fp32PeakTFLOPS": 30.3,
      "plateauOfFp32Peak": 0.294,
      "curve": [
        [7.75, 0.6977],
        [8.25, 0.8307],
        [8.75, 0.8465],
        [9.25, 0.8868],
        [9.75, 0.8868],
        [10.25, 0.8868],
        [10.75, 0.9848],
        [11.25, 0.9867],
        [11.75, 0.9867],
        [12.25, 0.9867],
        [12.75, 0.9887]
      ]
    },
    "a100-40g": {
      "neusightDevice": "NVIDIA A100-PCIE-40GB",
      "rows": 7500,
      "plateauFp32TFLOPS": 14.88,
      "fp32PeakTFLOPS": 19.5,
      "plateauOfFp32Peak": 0.763,
      "curve": [
        [-3.75, 13e-4],
        [-2.25, 41e-4],
        [-1.75, 52e-4],
        [-1.25, 7e-3],
        [-0.75, 0.0107],
        [-0.25, 0.0148],
        [0.25, 0.0207],
        [0.75, 0.0304],
        [1.25, 0.0417],
        [1.75, 0.0606],
        [2.25, 0.0858],
        [2.75, 0.1142],
        [3.25, 0.1608],
        [3.75, 0.2152],
        [4.25, 0.2762],
        [4.75, 0.337],
        [5.25, 0.4209],
        [5.75, 0.4993],
        [6.25, 0.5871],
        [6.75, 0.6628],
        [7.25, 0.7263],
        [7.75, 0.7778],
        [8.25, 0.8345],
        [8.75, 0.8615],
        [9.25, 0.8717],
        [9.75, 0.9284],
        [10.25, 0.9312],
        [10.75, 0.9483],
        [11.25, 0.9553],
        [11.75, 0.9814],
        [12.25, 0.9882],
        [12.75, 0.9995]
      ]
    },
    "a100-80g": {
      "neusightDevice": "NVIDIA A100 80GB PCIe",
      "rows": 1040,
      "plateauFp32TFLOPS": 17.51,
      "fp32PeakTFLOPS": 19.5,
      "plateauOfFp32Peak": 0.898,
      "curve": [
        [6.75, 0.6136],
        [7.25, 0.6876],
        [7.75, 0.7516],
        [8.25, 0.7685],
        [8.75, 0.819],
        [9.25, 0.8328],
        [9.75, 0.893],
        [10.25, 0.9067],
        [10.75, 0.9401],
        [11.25, 0.9704],
        [11.75, 0.987],
        [12.25, 0.992]
      ]
    },
    "v100": {
      "neusightDevice": "Tesla V100-PCIE-32GB",
      "rows": 7500,
      "plateauFp32TFLOPS": 12.01,
      "fp32PeakTFLOPS": 14,
      "plateauOfFp32Peak": 0.858,
      "curve": [
        [-3.75, 12e-4],
        [-1.75, 46e-4],
        [-1.25, 71e-4],
        [-0.75, 9e-3],
        [-0.25, 0.0123],
        [0.25, 0.018],
        [0.75, 0.0244],
        [1.25, 0.034],
        [1.75, 0.05],
        [2.25, 0.0681],
        [2.75, 0.0956],
        [3.25, 0.1398],
        [3.75, 0.1818],
        [4.25, 0.2316],
        [4.75, 0.2963],
        [5.25, 0.3427],
        [5.75, 0.4064],
        [6.25, 0.4777],
        [6.75, 0.5333],
        [7.25, 0.6233],
        [7.75, 0.6822],
        [8.25, 0.7467],
        [8.75, 0.7942],
        [9.25, 0.8572],
        [9.75, 0.8877],
        [10.25, 0.9254],
        [10.75, 0.9398],
        [11.25, 0.9497],
        [11.75, 0.9653],
        [12.25, 0.9897],
        [12.75, 0.997]
      ]
    },
    "t4": {
      "neusightDevice": "Tesla T4",
      "rows": 7498,
      "plateauFp32TFLOPS": 3.14,
      "fp32PeakTFLOPS": 8.1,
      "plateauOfFp32Peak": 0.387,
      "curve": [
        [-1.75, 57e-4],
        [0.25, 0.0208],
        [0.75, 0.0315],
        [1.25, 0.0413],
        [1.75, 0.0587],
        [2.25, 0.0857],
        [2.75, 0.1067],
        [3.25, 0.1559],
        [3.75, 0.2183],
        [4.25, 0.2903],
        [4.75, 0.3519],
        [5.25, 0.4754],
        [5.75, 0.5216],
        [6.25, 0.6392],
        [6.75, 0.7926],
        [7.25, 0.8978],
        [7.75, 0.9814],
        [8.25, 0.9832],
        [8.75, 0.9832],
        [9.25, 0.9832],
        [9.75, 0.9832],
        [10.25, 0.9832],
        [10.75, 0.9832],
        [11.25, 0.9832],
        [11.75, 0.9832],
        [12.25, 0.9832],
        [12.75, 0.9832],
        [13.25, 0.9832],
        [13.75, 0.9832],
        [14.25, 0.9832],
        [14.75, 0.9845],
        [15.25, 1]
      ]
    }
  },
  "neusightFp16H100OfDensePeak": [
    0.119,
    0.154,
    0.178,
    0.309,
    0.342,
    0.357,
    0.418,
    0.385,
    0.37
  ],
  "deviceInfo": {
    "h100-sxm": {
      "deviceInfoKey": "NVIDIA H100",
      "bf16TFLOPSListed": 1979,
      "sparsityFactor": 2,
      "bf16TFLOPSDense": 989.5,
      "fp32TFLOPS": 67,
      "dramGBs": 3350,
      "dramGB": 80
    },
    "a100-80g": {
      "deviceInfoKey": "NVIDIA A100",
      "bf16TFLOPSListed": 312.5,
      "sparsityFactor": 1,
      "bf16TFLOPSDense": 312.5,
      "fp32TFLOPS": 19.5,
      "dramGBs": 2039,
      "dramGB": 80
    },
    "l4": {
      "deviceInfoKey": "NVIDIA L4",
      "bf16TFLOPSListed": 242,
      "sparsityFactor": 1,
      "bf16TFLOPSDense": 242,
      "fp32TFLOPS": 30.3,
      "dramGBs": 3350,
      "dramGB": 24
    },
    "b200": {
      "deviceInfoKey": "NVIDIA B200",
      "bf16TFLOPSListed": 2250,
      "sparsityFactor": 1,
      "bf16TFLOPSDense": 2250,
      "fp32TFLOPS": 75,
      "dramGBs": 7700,
      "dramGB": 180
    }
  }
};

// src/utils/gemmEfficiency.ts
var CARDS = GEMM_EFFICIENCY_TABLE.cards;
var GEMM_CURVE_CARDS = Object.keys(CARDS);
function gemmShapeFactor(gpuId, plateauUs) {
  const c = CARDS[gpuId];
  if (!c || !(plateauUs > 0)) return 1;
  const pts = c.curve;
  const x = Math.log2(plateauUs);
  if (x <= pts[0][0]) return pts[0][1];
  const last = pts[pts.length - 1];
  if (x >= last[0]) return last[1];
  for (let i = 1; i < pts.length; i++) {
    if (x <= pts[i][0]) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }
  }
  return last[1];
}
function gemmMs(gpuId, flops, peakFlopsPerSec2, anchor, axis = "time") {
  if (!(flops > 0) || !(peakFlopsPerSec2 > 0)) return 0;
  const plateau = peakFlopsPerSec2 * anchor;
  const fp32 = CARDS[gpuId]?.plateauFp32TFLOPS;
  const us = axis === "flops" && fp32 ? flops / (fp32 * 1e12) * 1e6 : flops / plateau * 1e6;
  const s = gemmShapeFactor(gpuId, us);
  return flops / (plateau * s) * 1e3;
}
var GEMM_SHAPE_CARDS = {
  // attnOfPeak on h100-sxm and l4 is our own measurement, not a published one:
  // scripts/layout-calibration/eager_profile.py timed torch 2.8 sdpa (the
  // FlashAttention-2 kernels) inside one eager transformers block, forward and
  // backward, and this is the model's attention FLOPs (3 x a causal forward)
  // over the measured kernel time, pooled over Qwen3-0.6B and the
  // Llama-3.2-1B geometry. FA2's backward recomputes, so the kernels' own
  // utilisation is higher (H100: forward 28% to 32%, backward 24% to 27%);
  // the number here is what makes this model's FLOP count come out at the
  // measured time. In-sample for those two one-GPU rows.
  "h100-sxm": { sms: 132, sustainedOfPeak: 786 / 989, attnOfPeak: 0.229, source: "MSMF 786 of 989 TFLOPS (ml-engineering); attention 22.9% measured (torch 2.8 sdpa FA2, eager block fwd+bwd at 4096 x 4, data/eager_profile.json; the FlashAttention-3 paper puts FA2 near 35% on H100 forward)" },
  l4: { sms: 58, sustainedOfPeak: 76.6 / 121, attnOfPeak: 0.33, source: "MSMF 76.6 of 121 TFLOPS, our mamf-finder run on a Modal L4 (data/mamf_l4_2026-10-18.txt; MAMF 84, power-capped at 72 W); attention 33% measured (torch 2.8 sdpa FA2, eager block fwd+bwd at 2048 x 1, data/eager_profile.json)" },
  h200: { sms: 132, sustainedOfPeak: 755 / 989, attnOfPeak: 0.35, source: "MSMF 755 of 989 TFLOPS, H200 SXM (ml-engineering); FA2 utilisation taken from H100 (same SM)" },
  b200: { sms: 148, sustainedOfPeak: 1439 / 2250, attnOfPeak: null, source: "MSMF 1439 of 2250 TFLOPS, B200 SXM (ml-engineering)" },
  "a100-80g": { sms: 108, sustainedOfPeak: 258 / 312, attnOfPeak: 0.5, source: "MSMF 258 of 312 TFLOPS, A100 PCIe (ml-engineering; no SXM sustained row); FA2 50% to 73% on A100 (FlashAttention-2 paper, arXiv 2307.08691), low end" },
  "a100-40g": { sms: 108, sustainedOfPeak: 258 / 312, attnOfPeak: 0.5, source: "as a100-80g" }
};
var GEMM_SHAPE_MODEL_CARDS = Object.keys(GEMM_SHAPE_CARDS);
var SHAPE_TILES = [
  [256, 128, 1],
  [128, 256, 1],
  [128, 128, 0.9],
  [128, 64, 0.75],
  [64, 128, 0.75],
  [64, 64, 0.6]
];
var SHAPE_K0 = 128;
var SHAPE_BW_EFFICIENCY = 0.8;
function gemmShapeEfficiency(gpuId, M, N, K, batch = 1) {
  const c = GEMM_SHAPE_CARDS[gpuId];
  if (!c || !(M > 0 && N > 0 && K > 0)) return null;
  let best = 0;
  for (const [tm, tn, te] of SHAPE_TILES) {
    const gm = Math.ceil(M / tm), gn = Math.ceil(N / tn);
    const tiles = gm * gn * Math.max(1, batch);
    const wave = tiles / (Math.ceil(tiles / c.sms) * c.sms);
    const pad = M * N / (gm * tm * gn * tn);
    best = Math.max(best, te * wave * pad);
  }
  return c.sustainedOfPeak * best * (K / (K + SHAPE_K0));
}
function gemmShapeMs(gpuId, M, N, K, batch, peakFlopsPerSec2, bwBytesPerSec, bytesPerElem = 2) {
  const e = gemmShapeEfficiency(gpuId, M, N, K, batch);
  if (e == null || !(peakFlopsPerSec2 > 0)) return null;
  const flops = 2 * M * N * K * Math.max(1, batch);
  const compute = flops / (peakFlopsPerSec2 * e);
  const bytes = bytesPerElem * (M * K + K * N + M * N) * Math.max(1, batch);
  const memory = bwBytesPerSec > 0 ? bytes / (bwBytesPerSec * SHAPE_BW_EFFICIENCY) : 0;
  return Math.max(compute, memory) * 1e3;
}
function attentionMs(gpuId, flops, peakFlopsPerSec2) {
  const a = GEMM_SHAPE_CARDS[gpuId]?.attnOfPeak;
  if (!a || !(flops > 0) || !(peakFlopsPerSec2 > 0)) return null;
  return flops / (peakFlopsPerSec2 * a) * 1e3;
}

// src/utils/pipelineSchedule.ts
function scheduleTable(pp, v, m) {
  const table = Array.from({ length: pp }, () => []);
  if (v <= 1) {
    for (let r = 0; r < pp; r++) {
      const warm = Math.min(pp - r - 1, m);
      for (let i = 0; i < warm; i++) table[r].push({ kind: "F", mb: i, chunk: 0 });
      for (let i = warm; i < m; i++) {
        table[r].push({ kind: "F", mb: i, chunk: 0 });
        table[r].push({ kind: "B", mb: i - warm, chunk: 0 });
      }
      for (let i = m - warm; i < m; i++) table[r].push({ kind: "B", mb: i, chunk: 0 });
    }
    return table;
  }
  if (m % pp !== 0) throw new Error(`interleaved schedule needs microbatches (${m}) divisible by pp (${pp})`);
  const span = pp * v;
  const total = m * v;
  const fwdAt = (i) => ({ mb: i % pp + Math.floor(i / span) * pp, chunk: Math.floor(i % span / pp) });
  const bwdAt = (i) => ({ mb: i % pp + Math.floor(i / span) * pp, chunk: v - 1 - Math.floor(i % span / pp) });
  for (let r = 0; r < pp; r++) {
    const warm = Math.min((pp - 1 - r) * 2 + (v - 1) * pp, total);
    for (let i = 0; i < warm; i++) table[r].push({ kind: "F", ...fwdAt(i) });
    for (let i = warm; i < total; i++) {
      table[r].push({ kind: "F", ...fwdAt(i) });
      table[r].push({ kind: "B", ...bwdAt(i - warm) });
    }
    for (let i = total - warm; i < total; i++) table[r].push({ kind: "B", ...bwdAt(i) });
  }
  return table;
}
function simulatePipeline(fwdMs, bwdMs, m, v = 1, p2pMs = 0) {
  const pp = fwdMs.length;
  const table = scheduleTable(pp, v, m);
  const nv = pp * v;
  const doneF = new Float64Array(nv * m).fill(-1);
  const doneB = new Float64Array(nv * m).fill(-1);
  const clock = new Array(pp).fill(0);
  const idx = new Array(pp).fill(0);
  let progress = true;
  while (progress) {
    progress = false;
    for (let r = 0; r < pp; r++) {
      const ops = table[r];
      while (idx[r] < ops.length) {
        const op = ops[idx[r]];
        const vs = op.chunk * pp + r;
        let ready = 0;
        if (op.kind === "F") {
          if (vs > 0) {
            const t = doneF[(vs - 1) * m + op.mb];
            if (t < 0) break;
            ready = t + ((vs - 1) % pp !== r ? p2pMs : 0);
          }
        } else if (vs < nv - 1) {
          const t = doneB[(vs + 1) * m + op.mb];
          if (t < 0) break;
          ready = t + ((vs + 1) % pp !== r ? p2pMs : 0);
        } else {
          const t = doneF[vs * m + op.mb];
          if (t < 0) break;
          ready = t;
        }
        const start = clock[r] > ready ? clock[r] : ready;
        const end = start + (op.kind === "F" ? fwdMs[r][op.chunk] : bwdMs[r][op.chunk]);
        (op.kind === "F" ? doneF : doneB)[vs * m + op.mb] = end;
        clock[r] = end;
        idx[r]++;
        progress = true;
      }
    }
  }
  if (idx.some((i, r) => i < table[r].length)) throw new Error("pipeline schedule deadlocked");
  return { stepMs: Math.max(...clock), rankEndMs: clock };
}
function simulatedStepTime(stageMs, m, v = 1, fwdShare = 1 / 3, p2pMs = 0) {
  const f = stageMs.map((t) => new Array(v).fill(t * fwdShare / v));
  const b = stageMs.map((t) => new Array(v).fill(t * (1 - fwdShare) / v));
  return simulatePipeline(f, b, Math.max(1, m), v, p2pMs).stepMs;
}

// src/utils/parallelismPlanner.ts
var NODE_SIZE = 8;
var COMPUTE_BOUND_MFU = 0.4;
function tpCeilingFor(gpu, ffDim) {
  const c = gpu.bf16TFLOPS * 1e12;
  const linkGBs = gpu.nvlinkGBs > 0 ? gpu.nvlinkGBs : 32;
  const cw = c / (linkGBs * 1e9);
  return Math.max(1, Math.min(NODE_SIZE, Math.floor((ffDim > 0 ? ffDim : 4096) / cw)));
}

// src/utils/parallelLayout.ts
var BUBBLE_WARN = 0.15;
var STRAGGLE_THRESHOLD = 0.1;
var SEQ_IMBALANCE_WARN = 0.05;
var LOSS_FLOPS_PER_LOGIT = 10;
var W_BYTES = 4;
var G_BYTES = 4;
var O_BYTES = 8;
var ACT_BYTES = 2;
var MEM_BW_EFFICIENCY = 0.8;
var MEM_BYTES_PER_ELEM = {
  /**
   * transformers' LlamaRMSNorm / Qwen3RMSNorm: fwd to(fp32) 6, pow 8, mean 4,
   * x*rsqrt 8, to(bf16) 6, *weight 4 = 36; bwd mirrors it with two gradient
   * paths summed (weight grad 8, cast 6, mul 16, mean 4, pow 12, sum 12, cast 6) = 64.
   */
  rmsNorm: 100,
  /** torch.nn.LayerNorm is one fused kernel each way: fwd read + write 4, bwd grad + input + write + weight-grad read 12. */
  layerNorm: 16,
  /** Residual add: fwd two reads + one write 6; bwd the gradient is summed where the residual stream forks, 6. */
  add: 12,
  /** SwiGLU per intermediate element: fwd silu 4 + mul 6; bwd mul (two grads) 12 + silu 6. */
  swiglu: 28,
  /** GELU MLP per intermediate element: fwd 4, bwd 6. */
  gelu: 10,
  /** RoPE per q / k element (transformers' rotate_half): fwd mul, neg, cat, mul, add = 26; bwd the same. */
  rope: 52,
  /**
   * Cross-entropy per logit on the stock path, `cross_entropy(logits.float())`:
   * fwd fp32 cast 6, log_softmax 12; bwd nll grad 4, log_softmax backward 12,
   * cast back to bf16 6 = 40.
   */
  crossEntropy: 40,
  /** Fused / chunked linear + cross-entropy: bf16 logits written and read once each way, never upcast whole. An assumption, not a count. */
  crossEntropyFused: 8
};
var LAYOUT_BASIS = "Derived from the graph, not measured. A stage's time is the sum of its layers: each matmul at 40% of the card's peak when long and less when short (a per-card curve fitted to NeuSight's measured kernels on H100, L4, A100, V100 and T4; flat 40% on other cards), plus memory-bound bytes (norms, elementwise ops, RoPE, the fp32 loss) at 80% of its memory bandwidth. Step time comes from simulating the 1F1B or interleaved schedule at this layout's microbatch count. Predicted speedups of a split landed within 5.4 points of measured on 2 and 4 L4 and within 2.3 on 2 H100; absolute step time runs over, by up to 7% on L4 and 27% on H100. Memory uses the same per-layer terms and calibrated upper bound as the training-peak estimate, plus the fp32 logits on the stage with the loss. Nothing here was timed on a GPU.";
var DEFAULT_GPU = "h100-sxm";
var posInt2 = (v) => {
  const n2 = typeof v === "string" && v.trim() !== "" ? Number(v) : typeof v === "number" ? v : NaN;
  return Number.isInteger(n2) && n2 > 0 ? n2 : null;
};
function readLayout(raw) {
  if (raw == null || typeof raw !== "object" || Array.isArray(raw)) return { error: "`layout` must be an object." };
  const r = raw;
  const pick = (...names) => {
    const n2 = names.find((k) => r[k] != null);
    return n2 == null ? null : { name: n2, v: r[n2] };
  };
  const int = (required, ...names) => {
    const p = pick(...names);
    if (!p) return required ? { error: `\`layout.${names[0]}\` is required.` } : void 0;
    const v = posInt2(p.v);
    return v == null ? { error: `\`layout.${p.name}\` must be a positive integer.` } : v;
  };
  const out = {};
  const spec = [
    ["tp", false, ["tp", "tensor_parallel"]],
    ["pp", false, ["pp", "pipeline_parallel"]],
    ["dp", false, ["dp", "data_parallel"]],
    ["cp", false, ["cp", "context_parallel"]],
    ["ep", false, ["ep", "expert_parallel"]],
    ["numGpus", false, ["num_gpus", "numGpus", "gpus"]],
    ["microBatch", true, ["micro_batch", "microBatch", "micro_batch_size"]],
    ["globalBatch", false, ["global_batch", "globalBatch", "global_batch_size"]],
    ["gradAccum", false, ["grad_accum", "gradAccum", "gradient_accumulation_steps"]],
    ["seqLen", true, ["seq_len", "seqLen"]],
    ["virtualStages", false, ["virtual_stages", "virtualStages", "virtual_pipeline_stages"]]
  ];
  for (const [key, req, names] of spec) {
    const v = int(req, ...names);
    if (v && typeof v === "object") return v;
    if (v != null) out[key] = v;
  }
  out.tp = out.tp ?? 1;
  out.pp = out.pp ?? 1;
  out.dp = out.dp ?? 1;
  const gpu = pick("gpu");
  if (gpu) {
    if (typeof gpu.v !== "string") return { error: "`layout.gpu` must be a string." };
    out.gpu = gpu.v;
  }
  const sch = pick("schedule");
  if (sch) {
    const s = String(sch.v).toLowerCase();
    if (s !== "1f1b" && s !== "interleaved") return { error: '`layout.schedule` must be "1f1b" or "interleaved".' };
    out.schedule = s;
  }
  const lps = pick("layers_per_stage", "layersPerStage");
  if (lps) {
    const arr = typeof lps.v === "string" ? lps.v.split(",") : lps.v;
    if (!Array.isArray(arr) || !arr.length) return { error: `\`layout.${lps.name}\` must be a list of positive integers.` };
    const nums = arr.map(posInt2);
    if (nums.some((n2) => n2 == null)) return { error: `\`layout.${lps.name}\` must be a list of positive integers.` };
    out.layersPerStage = nums;
  }
  const ks = pick("kernel_stack", "kernelStack");
  if (ks) {
    if (ks.v !== "eager" && ks.v !== "fused") return { error: `\`layout.${ks.name}\` must be "eager" or "fused".` };
    out.kernelStack = ks.v;
  }
  for (const [key, names] of [["embeddingOnFirst", ["embedding_on_first", "embeddingOnFirst"]], ["headOnLast", ["head_on_last", "headOnLast"]], ["fsdp", ["fsdp"]], ["activationCheckpointing", ["activation_checkpointing", "activationCheckpointing", "recompute"]], ["fusedCrossEntropy", ["fused_cross_entropy", "fusedCrossEntropy", "fused_ce"]], ["qkNorm", ["qk_norm", "qkNorm"]]]) {
    const p = pick(...names);
    if (!p) continue;
    if (typeof p.v !== "boolean") return { error: `\`layout.${p.name}\` must be true or false.` };
    out[key] = p.v;
  }
  const z = pick("zero_stage", "zeroStage", "zero");
  if (z) {
    const v = Number(z.v);
    if (![0, 1, 2, 3].includes(v)) return { error: `\`layout.${z.name}\` must be 0, 1, 2 or 3.` };
    out.zeroStage = v;
  }
  const h = pick("seq_len_histogram", "seqLenHistogram");
  if (h) {
    const hist = readHistogram(h.v);
    if ("error" in hist) return { error: `\`layout.${h.name}\`: ${hist.error}` };
    out.seqLenHistogram = hist.histogram;
  }
  return out;
}
function readHistogram(raw) {
  let rows = [];
  if (typeof raw === "string") {
    rows = raw.split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !l.startsWith("#")).map((l) => l.split(/[,\t ]+/)).filter(([a]) => !/^[a-z_]/i.test(String(a)));
  } else if (Array.isArray(raw)) {
    rows = raw.map((e) => Array.isArray(e) ? [e[0], e[1]] : e && typeof e === "object" ? [e.len ?? e.length, e.count] : [null, null]);
  } else return { error: "must be a list of {len, count} or two-column CSV text." };
  const out = [];
  for (const [a, b] of rows) {
    const len = posInt2(a), count = Number(b);
    if (len == null || !(Number.isFinite(count) && count >= 0)) return { error: `unreadable row ${JSON.stringify([a, b])}; each row is a positive length and a non-negative count.` };
    if (count > 0) out.push({ len, count });
  }
  if (!out.length) return { error: "no rows with a positive count." };
  return { histogram: out };
}
function resolveLayoutGpu(id) {
  const want = String(id ?? DEFAULT_GPU).toLowerCase().replace(/[^a-z0-9]/g, "");
  const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const exact = GPU_HARDWARE.find((g) => norm(g.id) === want) ?? gpuById(String(id ?? DEFAULT_GPU));
  if (exact) return exact;
  const hits = GPU_HARDWARE.filter((g) => norm(g.id).startsWith(want));
  return hits.length === 1 ? hits[0] : null;
}
function topoOrder2(model) {
  const idx = new Map(model.components.map((c, i) => [c.id, i]));
  const indeg = new Map(model.components.map((c) => [c.id, 0]));
  const next = new Map(model.components.map((c) => [c.id, []]));
  for (const e of model.connections ?? []) {
    if (!idx.has(e.from) || !idx.has(e.to)) continue;
    next.get(e.from).push(e.to);
    indeg.set(e.to, (indeg.get(e.to) ?? 0) + 1);
  }
  const ready = model.components.filter((c) => indeg.get(c.id) === 0).map((c) => c.id);
  const seen = /* @__PURE__ */ new Set();
  const out = [];
  while (ready.length) {
    ready.sort((a, b) => idx.get(a) - idx.get(b));
    const id = ready.shift();
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(model.components[idx.get(id)]);
    for (const n2 of next.get(id) ?? []) {
      const d = indeg.get(n2) - 1;
      indeg.set(n2, d);
      if (d === 0) ready.push(n2);
    }
  }
  for (const c of model.components) if (!seen.has(c.id)) out.push(c);
  return out;
}
var SKIP_TYPES2 = /* @__PURE__ */ new Set(["input", "output", "stickyNote"]);
var BLOCK_TYPES = /^(transformerBlock|transformerEncoder|transformerDecoder|transformerEncoderLayer|transformerDecoderLayer|ditBlock|llamaBlock|decoderBlock|encoderBlock)$/;
function blocksOf(model) {
  const order = topoOrder2(model).filter((c) => !SKIP_TYPES2.has(c.type));
  const numOf = (c) => {
    if (/^(Vision|Audio)_/.test(c.name ?? "")) return null;
    const m = /_(\d+)(?=_|$)/.exec(c.name ?? "");
    return m ? Number(m[1]) : null;
  };
  const numbered = order.map(numOf);
  const firstIdx = numbered.findIndex((n2) => n2 != null);
  let lastIdx = -1;
  for (let i = numbered.length - 1; i >= 0; i--) if (numbered[i] != null) {
    lastIdx = i;
    break;
  }
  if (firstIdx >= 0) {
    const blocks = [];
    let cur = null;
    let monotone = true;
    for (let i = firstIdx; i <= lastIdx; i++) {
      const n2 = numbered[i];
      if (n2 != null && n2 !== cur) {
        if (cur != null && n2 < cur) monotone = false;
        cur = n2;
        blocks.push([]);
      }
      blocks[blocks.length - 1].push(order[i]);
    }
    if (monotone && blocks.length >= 2) {
      return { pre: order.slice(0, firstIdx), blocks, post: order.slice(lastIdx + 1), method: "numbered-names" };
    }
  }
  const bIdx = order.map((c, i) => BLOCK_TYPES.test(c.type) ? i : -1).filter((i) => i >= 0);
  if (bIdx.length >= 2) {
    const blocks = [];
    for (let k = 0; k < bIdx.length; k++) {
      const end = k + 1 < bIdx.length ? bIdx[k + 1] : bIdx[k] + 1;
      blocks.push(order.slice(bIdx[k], end));
    }
    return { pre: order.slice(0, bIdx[0]), blocks, post: order.slice(bIdx[bIdx.length - 1] + 1), method: "block-components" };
  }
  return null;
}
var ATTN2 = /attention|^mla$/i;
function geometryOf(model) {
  const g = { numHeads: null, numKVHeads: null, hidden: null, ffn: null, numExperts: null, vocab: null };
  const num7 = (v) => typeof v === "number" && v > 0 ? v : null;
  for (const c of model.components) {
    const p = c.params ?? {};
    if (g.numHeads == null && ATTN2.test(c.type)) {
      g.numHeads = num7(p.numHeads);
      g.numKVHeads = num7(p.numKVHeads) ?? g.numHeads;
      g.hidden = g.hidden ?? num7(p.embedDim);
    }
    if (g.ffn == null && /^(swiglu|feedForward|geglu|mlp)$/i.test(c.type)) g.ffn = num7(p.intermediateSize) ?? num7(p.ffDim) ?? num7(p.hiddenFeatures);
    if (/moe/i.test(c.type)) {
      g.numExperts = g.numExperts ?? num7(p.numExperts);
      g.ffn = g.ffn ?? num7(p.expertDim) ?? num7(p.ffDim);
    }
    if (c.type === "embedding" && g.vocab == null) {
      g.vocab = num7(p.vocabSize);
      g.hidden = g.hidden ?? num7(p.embeddingDim);
    }
  }
  return g;
}
function weightedNoTransient(l) {
  return l.outputs * OUTPUT_K + l.scores * ATTENTION_K + l.internal * INTERNAL_K + l.recurrent * RECURRENT_K;
}
function memBoundBytes(c, shapes, hasRope) {
  const sh = shapes.get(c.id);
  const outS = sh?.out ?? [];
  if (!outS.length) return 0;
  const elems = outS.reduce((a, b) => a * b, 1);
  const width = outS[outS.length - 1] || 1;
  const tokens = elems / width;
  const p = c.params ?? {};
  const num7 = (v) => typeof v === "number" && v > 0 ? v : null;
  const B = MEM_BYTES_PER_ELEM;
  switch (c.type) {
    case "rmsNorm":
      return elems * B.rmsNorm;
    case "layerNorm":
      return elems * B.layerNorm;
    case "add":
      return elems * B.add;
    case "swiglu":
    case "geglu": {
      const I = num7(p.intermediateSize) ?? num7(p.ffDim) ?? num7(p.hiddenFeatures);
      return I ? tokens * I * B.swiglu : 0;
    }
    case "feedForward": {
      const I = num7(p.intermediateSize) ?? num7(p.ffDim) ?? num7(p.hiddenFeatures);
      return I ? tokens * I * B.gelu : 0;
    }
    case "embedding": {
      const V = num7(p.vocabSize), D = num7(p.embeddingDim) ?? width;
      return (V ? V * D * ACT_BYTES : 0) + tokens * D * 10;
    }
    default:
      if (hasRope && ATTN2.test(c.type)) {
        const H = num7(p.numHeads), E = num7(p.embedDim) ?? width;
        if (!H) return 0;
        const hd = num7(p.headDim) ?? E / H;
        const kv = num7(p.numKVHeads) ?? H;
        return tokens * (H + kv) * hd * B.rope;
      }
      return 0;
  }
}
function gemmsOf(c, shapes, macs) {
  if (c.type === "feedForward") return macs > 0 ? [macs] : [];
  return gemmShapesOf(c, shapes, macs).map((g) => g.macs);
}
function gemmShapesOf(c, shapes, macs) {
  if (!(macs > 0)) return [];
  const sh = shapes.get(c.id);
  const inS = sh?.in ?? [];
  const outS = sh?.out ?? [];
  const T = inS.length >= 2 ? inS[inS.length - 2] : 1;
  const p = c.params ?? {};
  const num7 = (v) => typeof v === "number" && v > 0 ? v : null;
  const parts = [];
  const mm = (m, n2, k) => ({ macs: m * n2 * k, m, n: n2, k });
  if (c.type === "groupedQueryAttention" || c.type === "multiHeadAttention" || c.type === "causalAttention" || c.type === "selfAttention" || c.type === "attention") {
    const D = num7(p.embedDim) ?? num7(p.hiddenDim) ?? (inS[inS.length - 1] || null);
    if (D) {
      if (c.type === "groupedQueryAttention") {
        const H = num7(p.numHeads) ?? 32, kv = num7(p.numKVHeads) ?? H;
        const hd = num7(p.headDim) ?? Math.floor(D / H);
        parts.push(
          mm(T, H * hd, D),
          mm(T, kv * hd, D),
          mm(T, kv * hd, D),
          { macs: T * T * H * hd, m: T, n: T, k: hd, batch: H, attn: true },
          mm(T, D, H * hd)
        );
      } else {
        const H = num7(p.numHeads) ?? 1;
        parts.push(mm(T, D, D), mm(T, D, D), mm(T, D, D), { macs: T * T * D, m: T, n: T, k: Math.max(1, Math.floor(D / H)), batch: H, attn: true }, mm(T, D, D));
      }
    }
  } else if (c.type === "swiglu" || c.type === "geglu") {
    const D = num7(p.embedDim) ?? num7(p.inFeatures), I = num7(p.intermediateSize) ?? num7(p.hiddenFeatures) ?? num7(p.ffDim);
    if (D && I) parts.push(mm(T, I, D), mm(T, I, D), mm(T, D, I));
  } else if (c.type === "feedForward") {
    const D = num7(p.embedDim) ?? num7(p.hiddenDim), I = num7(p.ffDim) ?? (D ? 4 * D : null);
    if (D && I && Math.abs(2 * T * D * I - macs) <= macs * 1e-6) parts.push(mm(T, I, D), mm(T, D, I));
  } else if (c.type === "linear") {
    const K = inS[inS.length - 1], N = outS[outS.length - 1];
    const tokens = K && N ? macs / (K * N) : 0;
    parts.push(tokens >= 1 && Number.isInteger(Math.round(tokens)) && Math.abs(tokens - Math.round(tokens)) < 1e-6 ? mm(Math.round(tokens), N, K) : { macs });
  }
  const known = parts.reduce((a, b) => a + b.macs, 0);
  if (known > macs * (1 + 1e-9)) return [{ macs }];
  const rest = macs - known;
  if (rest > macs * 1e-6) parts.push({ macs: rest });
  return parts;
}
function unitCost(comps, shapes, isPost, hasRope, stack = "eager", qkNorm = false) {
  const stores = storesScoresUnder("sdpa");
  const u = { params: 0, expertParams: 0, macs: 0, gemms: [], gemmShapes: [], actElems: 0, largest: 0, boundaryElems: 0, logits: 0, memBytes: 0, ngBytes: 0, ngKernels: 0 };
  for (const c of comps) {
    const sh = shapes.get(c.id);
    const inS = sh?.in ?? [];
    const outS = sh?.out ?? [];
    const p = estimateLayerParams(c.type, c.params, inS.length ? inS : void 0) ?? 0;
    u.params += p;
    if (/moe/i.test(c.type)) u.expertParams += p;
    const macs = estimateLayerFlops(c.type, c.params ?? {}, inS, outS) || 0;
    u.macs += macs;
    u.gemms.push(...gemmsOf(c, shapes, macs));
    u.gemmShapes.push(...gemmShapesOf(c, shapes, macs));
    u.memBytes += memBoundBytes(c, shapes, hasRope);
    const ng = nonGemmOf(c, outS, stack, qkNorm);
    const rp = hasRope ? ropeOf(c, outS, stack) : { bytes: 0, kernels: 0 };
    u.ngBytes += ng.bytes + rp.bytes;
    u.ngKernels += ng.kernels + rp.kernels;
    const l = layerMemoryTerms(c, shapes, stores);
    if (l) {
      u.actElems += weightedNoTransient(l);
      u.largest = Math.max(u.largest, l.outputs, l.scores);
    }
    if (outS.length) u.boundaryElems = outS.reduce((a, b) => a * b, 1);
  }
  if (isPost) {
    const head = [...comps].reverse().find((c) => c.type === "linear");
    const o = head ? shapes.get(head.id)?.out ?? [] : [];
    if (o.length) u.logits = o.reduce((a, b) => a * b, 1);
  }
  return u;
}
function optimalPartition(w, k, head0 = 0, tailK = 0) {
  const n2 = w.length;
  if (k < 1 || k > n2) return null;
  const pre = [0];
  for (const x of w) pre.push(pre[pre.length - 1] + x);
  const sum = (i, j2) => pre[j2] - pre[i];
  const f = Array.from({ length: k + 1 }, () => new Array(n2 + 1).fill(Infinity));
  f[0][0] = 0;
  for (let s = 1; s <= k; s++) {
    for (let j2 = s; j2 <= n2 - (k - s); j2++) {
      for (let i = s - 1; i < j2; i++) {
        if (f[s - 1][i] === Infinity) continue;
        const part = sum(i, j2) + (s === 1 ? head0 : 0) + (s === k ? tailK : 0);
        const v = Math.max(f[s - 1][i], part);
        if (v < f[s][j2]) f[s][j2] = v;
      }
    }
  }
  if (f[k][n2] === Infinity) return null;
  const cap = f[k][n2] * (1 + 1e-9);
  const g = Array.from({ length: k + 1 }, () => new Array(n2 + 1).fill(Infinity));
  const arg2 = Array.from({ length: k + 1 }, () => new Array(n2 + 1).fill(-1));
  g[0][0] = 0;
  for (let s = 1; s <= k; s++) {
    for (let j2 = s; j2 <= n2 - (k - s); j2++) {
      for (let i = s - 1; i < j2; i++) {
        if (g[s - 1][i] === Infinity) continue;
        const part = sum(i, j2) + (s === 1 ? head0 : 0) + (s === k ? tailK : 0);
        if (part > cap) continue;
        const val = g[s - 1][i] + part * part;
        if (val < g[s][j2]) {
          g[s][j2] = val;
          arg2[s][j2] = i;
        }
      }
    }
  }
  const sizes = [];
  let j = n2;
  for (let s = k; s >= 1; s--) {
    const i = arg2[s][j];
    sizes.unshift(j - i);
    j = i;
  }
  return sizes;
}
function splitBySchedule(w, k, head0, tailK, step, seed, microbatches = 1) {
  const mOf = microbatches;
  const n2 = w.length;
  const costs = (sizes) => {
    const out = [];
    let s = 0;
    sizes.forEach((c, i) => {
      let f = 0;
      for (let j = s; j < s + c; j++) f += w[j];
      if (i === 0) f += head0;
      if (i === sizes.length - 1) f += tailK;
      out.push(f);
      s += c;
    });
    return out;
  };
  let best = seed.slice(), bestT = step(costs(best));
  const better = (t) => t < bestT * (1 - 1e-9);
  let count = 1;
  for (let i = 0; i < k - 1; i++) count = count * (n2 - 1 - i) / (i + 1);
  if (count <= 2e4) {
    const pre = [0];
    for (const x of w) pre.push(pre[pre.length - 1] + x);
    const sizes = new Array(k).fill(1);
    const rec = (i, at) => {
      const left = n2 - at;
      const bound = bestT / Math.max(1, mOf);
      if (i === k - 1) {
        const last = pre[n2] - pre[at] + tailK + (k === 1 ? head0 : 0);
        if (last >= bound) return;
        sizes[i] = left;
        const t = step(costs(sizes));
        if (better(t)) {
          best = sizes.slice();
          bestT = t;
        }
        return;
      }
      for (let c = 1; c <= left - (k - 1 - i); c++) {
        const part = pre[at + c] - pre[at] + (i === 0 ? head0 : 0);
        if (part >= bound) break;
        sizes[i] = c;
        rec(i + 1, at + c);
      }
    };
    rec(0, 0);
    return best;
  }
  for (let moved = true; moved; ) {
    moved = false;
    for (let b = 0; b < k - 1; b++) {
      for (const d of [1, -1]) {
        const s = best.slice();
        s[b] += d;
        s[b + 1] -= d;
        if (s[b] < 1 || s[b + 1] < 1) continue;
        const t = step(costs(s));
        if (better(t)) {
          best = s;
          bestT = t;
          moved = true;
        }
      }
    }
  }
  return best;
}
function evenSplit(blocks, pp) {
  const base = Math.floor(blocks / pp), rem = blocks % pp;
  return Array.from({ length: pp }, (_, i) => base + (i < rem ? 1 : 0));
}
function bubbleFraction(pp, microbatches, virtualStages = 1) {
  if (pp <= 1 || microbatches <= 0) return 0;
  const x = (pp - 1) / Math.max(1, virtualStages);
  return x / (microbatches + x);
}
function pipelineStepTime(stageTimes, microbatches, virtualStages = 1) {
  if (!stageTimes.length) return 0;
  const max = Math.max(...stageTimes);
  const sum = stageTimes.reduce((a, b) => a + b, 0);
  return Math.max(1, microbatches) * max + (sum - max) / Math.max(1, virtualStages);
}
function microbatchesForBubble(pp, target, virtualStages = 1) {
  const x = (pp - 1) / Math.max(1, virtualStages);
  return Math.max(1, Math.ceil(x * (1 - target) / target - 1e-9));
}
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = a + 1831565813 >>> 0;
    let t = a;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function invNorm(p) {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) {
    const q2 = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q2 + c[1]) * q2 + c[2]) * q2 + c[3]) * q2 + c[4]) * q2 + c[5]) / ((((d[0] * q2 + d[1]) * q2 + d[2]) * q2 + d[3]) * q2 + 1);
  }
  if (p > 1 - pl) {
    const q2 = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q2 + c[1]) * q2 + c[2]) * q2 + c[3]) * q2 + c[4]) * q2 + c[5]) / ((((d[0] * q2 + d[1]) * q2 + d[2]) * q2 + d[3]) * q2 + 1);
  }
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
function seqImbalance(hist, ranks, perRank, a, b, seed = 7) {
  const total = hist.reduce((s, h) => s + h.count, 0);
  const lens = hist.map((h) => h.len);
  const cdf = [];
  let acc = 0;
  for (const h of hist) {
    acc += h.count / total;
    cdf.push(acc);
  }
  const work = (L) => a * L + b * L * L;
  const meanLen = hist.reduce((s, h) => s + h.len * h.count, 0) / total;
  const meanW = hist.reduce((s, h) => s + work(h.len) * h.count, 0) / total;
  const varW = hist.reduce((s, h) => s + (work(h.len) - meanW) ** 2 * h.count, 0) / total;
  const budget = perRank * meanLen;
  if (ranks <= 1) return { expectedMaxOverMean: 1, packedEstimate: 1, method: "monte-carlo" };
  const draws = ranks * perRank;
  if (draws <= 4e4) {
    const rnd = mulberry32(seed);
    const sample = () => {
      const u = rnd();
      let i = 0;
      while (i < cdf.length - 1 && u > cdf[i]) i++;
      return lens[i];
    };
    const trials = Math.max(50, Math.min(2e3, Math.floor(2e6 / draws)));
    let sumRatio = 0, sumPacked = 0;
    for (let t = 0; t < trials; t++) {
      let mx = 0, sm = 0, pmx = 0, psm = 0;
      for (let r = 0; r < ranks; r++) {
        let w = 0;
        for (let k = 0; k < perRank; k++) w += work(sample());
        mx = Math.max(mx, w);
        sm += w;
        let left = budget, q = 0;
        while (left > 0) {
          const L = Math.min(sample(), left);
          q += L * L;
          left -= L;
        }
        const pw = a * budget + b * q;
        pmx = Math.max(pmx, pw);
        psm += pw;
      }
      sumRatio += mx / (sm / ranks);
      sumPacked += pmx / (psm / ranks);
    }
    return { expectedMaxOverMean: sumRatio / trials, packedEstimate: sumPacked / trials, method: "monte-carlo" };
  }
  const z = invNorm((ranks - 0.375) / (ranks + 0.25));
  const mean = perRank * meanW;
  const ratio = 1 + z * Math.sqrt(perRank * varW) / mean;
  const meanL2 = hist.reduce((s, h) => s + h.len * h.len * h.count, 0) / total;
  const varL2 = hist.reduce((s, h) => s + (h.len * h.len - meanL2) ** 2 * h.count, 0) / total;
  const docs = budget / meanLen;
  const pMean = a * budget + b * docs * meanL2;
  const packed = 1 + z * b * Math.sqrt(docs * varL2) / pMean;
  return { expectedMaxOverMean: ratio, packedEstimate: packed, method: "normal-approximation" };
}
var pct = (x) => `${(x * 100).toFixed(1)}%`;
var gb = (x) => `${x.toFixed(1)} GB`;
function checkParallelLayout(model, layout, opts = {}) {
  const useCurve = opts.gemmCurve !== false;
  const checked = [];
  const skipped = [];
  const findings = [];
  const recommendations = [];
  const add = (f) => findings.push(f);
  const tp = Math.max(1, Math.floor(layout.tp || 1));
  const pp = Math.max(1, Math.floor(layout.pp || 1));
  const dp = Math.max(1, Math.floor(layout.dp || 1));
  const cp = Math.max(1, Math.floor(layout.cp ?? 1));
  const ep = Math.max(1, Math.floor(layout.ep ?? 1));
  const micro = Math.max(1, Math.floor(layout.microBatch || 1));
  const seqLen = Math.max(1, Math.floor(layout.seqLen || 1));
  const schedule = layout.schedule ?? "1f1b";
  const v = schedule === "interleaved" ? Math.max(1, Math.floor(layout.virtualStages ?? 2)) : 1;
  const zero = layout.fsdp ? 3 : layout.zeroStage ?? 0;
  const embFirst = layout.embeddingOnFirst !== false;
  const headLast = layout.headOnLast !== false;
  const world = tp * pp * dp * cp;
  const gpu = resolveLayoutGpu(layout.gpu);
  const geo = geometryOf(model);
  if (layout.numGpus != null) {
    checked.push("world-size");
    if (world !== layout.numGpus) {
      add({
        rule: "world-size",
        severity: "block",
        title: `tp x pp x dp x cp = ${world}, but the job has ${layout.numGpus} GPUs`,
        detail: `${tp} x ${pp} x ${dp} x ${cp} = ${world}. The launcher refuses a world size that does not match the process count${world < layout.numGpus ? `, or ${layout.numGpus - world} GPUs sit idle` : ""}.`,
        fix: layout.numGpus % (tp * pp * cp) === 0 ? `Set dp = ${layout.numGpus / (tp * pp * cp)}.` : `Choose tp x pp x cp dividing ${layout.numGpus}.`
      });
    }
  } else skipped.push({ check: "world-size", reason: "numGpus not given; the product tp x pp x dp x cp is taken as the world size." });
  const legalTp = [];
  for (let d = 1; d <= NODE_SIZE; d++) {
    if (geo.numHeads != null && geo.numHeads % d !== 0) continue;
    if (geo.numKVHeads != null && geo.numKVHeads % d !== 0 && d % geo.numKVHeads !== 0) continue;
    if (geo.hidden != null && geo.hidden % d !== 0) continue;
    if (geo.ffn != null && geo.ffn % d !== 0) continue;
    legalTp.push(d);
  }
  const legalTpText = legalTp.length ? legalTp.join(", ") : `none up to ${NODE_SIZE}`;
  if (geo.numHeads != null) {
    checked.push("tp-heads");
    if (geo.numHeads % tp !== 0) {
      add({
        rule: "tp-heads",
        severity: "block",
        title: `TP ${tp} does not divide ${geo.numHeads} attention heads`,
        detail: "Megatron-style TP gives each rank whole heads; a remainder fails at model construction.",
        fix: `Use a TP degree that passes every TP rule for this model: ${legalTpText}.`
      });
    }
    if (geo.numKVHeads != null && geo.numKVHeads !== geo.numHeads) {
      checked.push("tp-kv-heads");
      if (geo.numKVHeads % tp !== 0) {
        if (tp % geo.numKVHeads === 0) {
          add({
            rule: "tp-kv-replicated",
            severity: "warn",
            title: `TP ${tp} exceeds ${geo.numKVHeads} KV heads: each KV head is replicated ${tp / geo.numKVHeads}x`,
            detail: "It runs, but every replica stores and computes the same K and V projections, so the KV weights and KV activations do not shrink past TP = KV heads.",
            fix: `Keep TP at or under ${geo.numKVHeads}, or accept the replication knowingly.`
          });
        } else {
          add({
            rule: "tp-kv-heads",
            severity: "block",
            title: `TP ${tp} neither divides nor is a multiple of ${geo.numKVHeads} KV heads`,
            detail: "Grouped-query attention needs whole KV heads per rank or an exact replication factor.",
            fix: `Use a TP degree that passes every TP rule for this model: ${legalTpText}.`
          });
        }
      }
    }
  } else skipped.push({ check: "tp-heads", reason: "no attention layer with a head count in the graph." });
  if (geo.hidden != null) {
    checked.push("tp-hidden");
    if (geo.hidden % tp !== 0) add({ rule: "tp-hidden", severity: "block", title: `TP ${tp} does not divide hidden size ${geo.hidden}`, detail: "Column- and row-parallel linears split the hidden dimension evenly across ranks.", fix: `Use a TP degree that passes every TP rule for this model: ${legalTpText}.` });
  }
  if (geo.ffn != null) {
    checked.push("tp-ffn");
    if (geo.ffn % tp !== 0) add({ rule: "tp-ffn", severity: "block", title: `TP ${tp} does not divide FFN width ${geo.ffn}`, detail: "The MLP up and down projections are split along the intermediate dimension.", fix: `Use a TP degree that passes every TP rule for this model: ${legalTpText}.` });
  }
  checked.push("tp-node");
  if (tp > NODE_SIZE) {
    add({
      rule: "tp-crosses-node",
      severity: "warn",
      title: `TP ${tp} spans more than one ${NODE_SIZE}-GPU node`,
      detail: "Every layer does TP all-reduces; past the NVLink domain they run on the node fabric, an order of magnitude slower (the planner caps TP at one node for this reason).",
      fix: `Keep TP at ${NODE_SIZE} or under and move the rest of the split to PP or DP.`
    });
  } else if (gpu && tp > 1 && geo.ffn != null) {
    const ceiling = tpCeilingFor(gpu, geo.ffn);
    if (tp > ceiling) {
      add({
        rule: "tp-past-link-ceiling",
        severity: "info",
        title: `TP ${tp} is past the ~${ceiling}-way ceiling for FFN width ${geo.ffn} on ${gpu.label}`,
        detail: "Roofline estimate (scaling-book): past about ffDim x link bandwidth / peak FLOPs ways, the TP collectives cost more than the matmuls they split.",
        fix: `Consider TP ${ceiling} with more PP or DP.`
      });
    }
  }
  if (geo.numExperts != null) {
    checked.push("ep-experts");
    if (geo.numExperts % ep !== 0) add({ rule: "ep-experts", severity: "block", title: `EP ${ep} does not divide ${geo.numExperts} experts`, detail: "Expert parallelism places whole experts on each rank.", fix: `Use an EP degree dividing ${geo.numExperts}.` });
    if (dp * cp % ep !== 0) add({ rule: "ep-ranks", severity: "block", title: `EP ${ep} does not divide dp x cp = ${dp * cp}`, detail: "Expert-parallel groups are carved out of the data- and context-parallel ranks (Megatron-Core).", fix: `Use an EP degree dividing ${dp * cp}.` });
  } else if (ep > 1) {
    add({ rule: "ep-experts", severity: "block", title: `EP ${ep} on a graph with no MoE layer`, detail: "There are no experts to distribute.", fix: "Set ep = 1." });
  }
  checked.push("cp-seq");
  if (cp > 1) {
    if (seqLen % cp !== 0) add({ rule: "cp-seq", severity: "block", title: `CP ${cp} does not divide sequence length ${seqLen}`, detail: "Context parallelism gives each rank an equal slice of every sequence.", fix: `Use a sequence length divisible by ${2 * cp}, or a CP degree dividing ${seqLen}.` });
    else if (seqLen % (2 * cp) !== 0) add({ rule: "cp-causal-balance", severity: "warn", title: `Sequence length ${seqLen} is not divisible by 2 x CP = ${2 * cp}`, detail: "Load-balanced causal CP splits each sequence into 2 x CP chunks so every rank gets an early and a late chunk; without it the last rank does most of the attention.", fix: `Pad or choose a sequence length divisible by ${2 * cp}.` });
  }
  let m = null;
  let globalBatch = null;
  checked.push("batch-arithmetic");
  if (layout.gradAccum != null) {
    m = layout.gradAccum;
    globalBatch = micro * m * dp;
    if (layout.globalBatch != null && layout.globalBatch !== globalBatch) {
      add({
        rule: "batch-arithmetic",
        severity: "block",
        title: `global batch ${layout.globalBatch} != micro ${micro} x grad-accum ${m} x dp ${dp} = ${globalBatch}`,
        detail: "Megatron and DeepSpeed both assert this identity at startup.",
        fix: `Set grad-accum to ${layout.globalBatch / (micro * dp)} if that is an integer, or change the global batch to ${globalBatch}.`
      });
    }
  } else if (layout.globalBatch != null) {
    globalBatch = layout.globalBatch;
    const q = layout.globalBatch / (micro * dp);
    if (!Number.isInteger(q)) {
      add({
        rule: "batch-arithmetic",
        severity: "block",
        title: `global batch ${layout.globalBatch} is not a multiple of micro ${micro} x dp ${dp} = ${micro * dp}`,
        detail: "The microbatches per step must be a whole number on every DP rank.",
        fix: `Use a global batch that is a multiple of ${micro * dp} (e.g. ${Math.max(1, Math.round(q)) * micro * dp}).`
      });
    } else m = q;
  } else skipped.push({ check: "batch-arithmetic", reason: "neither globalBatch nor gradAccum given." });
  const split = blocksOf(model);
  const resolvedBase = {
    gpu: gpu?.id ?? null,
    gpuMemoryGb: gpu?.memoryGB ?? null,
    worldSize: world,
    microbatchesPerStep: m,
    globalBatch,
    blocks: split?.blocks.length ?? 0,
    layersPerStage: null,
    schedule,
    virtualStages: v,
    zeroStage: zero
  };
  if (!gpu) skipped.push({ check: "stage-memory", reason: `unknown or ambiguous GPU "${layout.gpu}"; one of ${GPU_HARDWARE.map((g) => g.id).join(", ")}.` });
  if (!split) {
    skipped.push({ check: "stage-balance", reason: "no repeated blocks found in the graph (numbered layer names or block components), so there is nothing to cut into stages." });
    if (gpu) skipped.push({ check: "stage-memory", reason: "needs the stage split." });
    return finish(null, [], null);
  }
  const nBlocks = split.blocks.length;
  const units = nBlocks + (embFirst ? 0 : 1) + (headLast ? 0 : 1);
  checked.push("pp-over-blocks");
  if (pp > units) {
    add({ rule: "pp-over-blocks", severity: "block", title: `PP ${pp} is more than the ${nBlocks} blocks`, detail: "Every stage needs at least one block.", fix: `Use PP ${Math.min(units, 8)} or fewer.` });
  }
  let lps;
  if (layout.layersPerStage) {
    checked.push("layers-per-stage");
    lps = layout.layersPerStage;
    const sum = lps.reduce((a, b) => a + b, 0);
    if (lps.length !== pp || sum !== units) {
      add({
        rule: "layers-per-stage",
        severity: "block",
        title: `layersPerStage [${lps.join(", ")}] does not describe ${pp} stages of ${units} layers`,
        detail: `It has ${lps.length} entries summing to ${sum}; the graph has ${nBlocks} blocks${units !== nBlocks ? ` plus ${units - nBlocks} counted end layer${units - nBlocks > 1 ? "s" : ""}` : ""}, cut into PP ${pp}.`,
        fix: `Give ${pp} positive counts summing to ${units}.`
      });
    }
  } else {
    lps = evenSplit(units, Math.min(pp, units));
    if (units % pp !== 0 && pp <= units) {
      add({
        rule: "uneven-default-split",
        severity: "warn",
        title: `${units} layers do not divide evenly into PP ${pp}`,
        detail: `Megatron-LM's default split asserts num_layers % pp == 0. The report assumes [${lps.join(", ")}].`,
        fix: "Set the first and last stage layer counts explicitly (decoder_first/last_pipeline_num_layers) or choose a PP degree dividing the layer count."
      });
    }
  }
  if (schedule === "interleaved") {
    checked.push("interleaved-chunks");
    if (nBlocks % (pp * v) !== 0) {
      add({ rule: "interleaved-chunks", severity: "block", title: `${nBlocks} blocks do not split into PP ${pp} x ${v} virtual stages`, detail: "The interleaved schedule gives each rank v equal model chunks.", fix: `Choose v with ${nBlocks} divisible by ${pp} x v.` });
    }
    if (m != null && pp > 1 && m % pp !== 0) {
      add({ rule: "interleaved-microbatches", severity: "warn", title: `${m} microbatches per step is not a multiple of PP ${pp}`, detail: "Megatron-LM's interleaved schedule has required the microbatch count to be divisible by the pipeline size; check your version.", fix: `Use a microbatch count that is a multiple of ${pp}.` });
    }
  }
  if (findings.some((f) => f.severity === "block" && (f.rule === "pp-over-blocks" || f.rule === "layers-per-stage"))) {
    return finish(null, [], null);
  }
  const sized = withSeqLen(model, seqLen).model;
  const shapes = propagateShapes(sized).shapes;
  const hasRope = model.components.some((c) => c.type === "rope");
  const stack = layout.kernelStack ?? "eager";
  const qkNorm = layout.qkNorm ?? model.components.some((c) => c.type === "qkNorm");
  const ngRate = NONGEMM_RATES[stack];
  const preC = unitCost(split.pre, shapes, false, hasRope, stack, qkNorm);
  const postC = unitCost(split.post, shapes, true, hasRope, stack, qkNorm);
  if (postC.logits === 0 && geo.vocab != null && geo.hidden != null) {
    postC.macs += seqLen * geo.hidden * geo.vocab;
    postC.gemms.push(seqLen * geo.hidden * geo.vocab);
    postC.gemmShapes.push({ macs: seqLen * geo.hidden * geo.vocab, m: seqLen, n: geo.vocab, k: geo.hidden });
    postC.logits = seqLen * geo.vocab;
    postC.params += geo.hidden * geo.vocab;
    postC.boundaryElems = Math.max(postC.boundaryElems, seqLen * geo.vocab);
    postC.largest = Math.max(postC.largest, seqLen * geo.vocab);
    postC.ngKernels += GEMM_KERNELS;
    if (!split.post.some((c) => /norm/i.test(c.type))) {
      const normType = split.blocks[0].find((c) => /norm/i.test(c.type))?.type;
      postC.memBytes += seqLen * geo.hidden * (normType === "layerNorm" ? MEM_BYTES_PER_ELEM.layerNorm : normType ? MEM_BYTES_PER_ELEM.rmsNorm : 0);
      const nr = normType === "layerNorm" ? ngRate.layerNorm : normType ? ngRate.rmsNorm : null;
      if (nr) {
        postC.ngBytes += seqLen * geo.hidden * nr.bytes;
        postC.ngKernels += nr.kernels;
      }
    }
    if (headLast && pp > 1) add({
      rule: "tied-head",
      severity: "info",
      title: "Tied LM head costed on the last stage",
      detail: `The checkpoint ties the output head to the embedding (${geo.vocab} x ${geo.hidden}). The weight is shared; the matmul and the loss are not, and under PP the last stage keeps its own copy of the matrix.`,
      fix: "Nothing to change; this says why the last stage is heavier than its block count suggests."
    });
  }
  const fusedCe = layout.fusedCrossEntropy === true;
  postC.memBytes += postC.logits * (fusedCe ? MEM_BYTES_PER_ELEM.crossEntropyFused : MEM_BYTES_PER_ELEM.crossEntropy);
  postC.ngBytes += postC.logits * (fusedCe ? MEM_BYTES_PER_ELEM.crossEntropyFused : MEM_BYTES_PER_ELEM.crossEntropy);
  postC.ngKernels += postC.logits > 0 ? fusedCe ? 2 : 6 : 0;
  const blockC = split.blocks.map((b) => unitCost(b, shapes, false, hasRope, stack, qkNorm));
  const recompute = layout.activationCheckpointing ? 4 / 3 : 1;
  const flopsPerTok = (u, isBlock) => 6 * u.macs / seqLen * (isBlock ? recompute : 1);
  const lossPerTok = postC.logits / seqLen * LOSS_FLOPS_PER_LOGIT;
  const perGpu = micro / (tp * cp);
  const useShape = opts.gemmModel === "shape" && !!gpu && GEMM_SHAPE_MODEL_CARDS.includes(gpu.id);
  const mmMs = (u, isBlock) => {
    if (!gpu) return 0;
    let t = 0;
    if (useShape) {
      const peak = gpu.bf16TFLOPS * 1e12, bw = gpu.memBandwidthGBs * 1e9, sus = GEMM_SHAPE_CARDS[gpu.id].sustainedOfPeak;
      const tokens = micro / cp;
      for (const g of u.gemmShapes) {
        if (g.m && g.n && g.k) {
          if (g.attn) {
            const a = attentionMs(gpu.id, 3 * 2 * g.macs * perGpu, peak);
            if (a != null) {
              t += a;
              continue;
            }
          }
          const b = (g.batch ?? 1) * (g.attn ? micro / tp : 1);
          const M = g.attn ? g.m / cp : g.m * tokens, N = g.attn ? g.n : g.n / tp, K = g.k;
          const share = g.macs / (g.m * g.n * g.k * (g.batch ?? 1));
          t += share * ((gemmShapeMs(gpu.id, M, N, K, b, peak, bw) ?? 0) + (gemmShapeMs(gpu.id, M, K, N, b, peak, bw) ?? 0) + (gemmShapeMs(gpu.id, K, N, M, b, peak, bw) ?? 0));
        } else {
          t += 3 * 2 * g.macs * perGpu / (peak * sus) * 1e3;
        }
      }
    } else {
      for (const g of u.gemms) t += 3 * gemmMs(useCurve ? gpu.id : "", 2 * g * perGpu, gpu.bf16TFLOPS * 1e12, COMPUTE_BOUND_MFU, opts.gemmAxis);
    }
    return t * (isBlock ? recompute : 1);
  };
  const useNonGemm = opts.nonGemm === true && useShape;
  const unitParamsPerGpu = (u) => (u.params - u.expertParams) / tp + u.expertParams / (tp * ep);
  const memMs = (u, isBlock) => {
    if (!gpu) return 0;
    const rc = isBlock ? recompute : 1;
    if (!useNonGemm) return u.memBytes * rc * perGpu / (gpu.memBandwidthGBs * 1e9 * MEM_BW_EFFICIENCY) * 1e3;
    const mb = m ?? 1;
    const accum = mb > 1 ? GRAD_ACCUM_BYTES_PER_PARAM[stack] * unitParamsPerGpu(u) * (mb - 1) / mb : 0;
    return nonGemmMs(u.ngBytes * rc * perGpu + accum, u.ngKernels * rc, gpu.memBandwidthGBs);
  };
  const costBasis = gpu ? "time" : "flops";
  const unitCostOf = (u, kind) => {
    const isBlock = kind === "block";
    if (costBasis === "time") return mmMs(u, isBlock) + memMs(u, isBlock);
    return flopsPerTok(u, isBlock) + (kind === "post" ? lossPerTok : 0);
  };
  const preF = flopsPerTok(preC, false);
  const postF = flopsPerTok(postC, false) + lossPerTok;
  const unitF = [];
  const unitW = [];
  const unitList = [];
  if (!embFirst) {
    unitF.push(preF);
    unitW.push(unitCostOf(preC, "pre"));
    unitList.push({ kind: "pre", c: preC });
  }
  blockC.forEach((c, i) => {
    unitF.push(flopsPerTok(c, true));
    unitW.push(unitCostOf(c, "block"));
    unitList.push({ kind: "block", c, i });
  });
  if (!headLast) {
    unitF.push(postF);
    unitW.push(unitCostOf(postC, "post"));
    unitList.push({ kind: "post", c: postC });
  }
  const head0F = embFirst ? preF : 0, tailF = headLast ? postF : 0;
  const head0W = embFirst ? unitCostOf(preC, "pre") : 0, tailW = headLast ? unitCostOf(postC, "post") : 0;
  const stageRanges = [];
  {
    let s = 0;
    for (const n2 of lps) {
      stageRanges.push([s, s + n2]);
      s += n2;
    }
  }
  const stageSum = (sizes, per, h0, tk) => {
    const out = [];
    let s = 0;
    sizes.forEach((n2, k) => {
      let f = 0;
      for (let j = s; j < s + n2; j++) f += per[j];
      if (k === 0) f += h0;
      if (k === sizes.length - 1) f += tk;
      out.push(f);
      s += n2;
    });
    return out;
  };
  const sf = stageSum(lps, unitF, head0F, tailF);
  const sw = stageSum(lps, unitW, head0W, tailW);
  const mean = sw.reduce((a, b) => a + b, 0) / sw.length;
  const maxW = Math.max(...sw);
  const slowest = sw.indexOf(maxW);
  const simulated = opts.simulateSchedule !== false && m != null && pp > 1 && (v === 1 || m % pp === 0);
  const optMsOf = (u) => useNonGemm && gpu ? nonGemmMs(OPTIMIZER_BYTES_PER_PARAM[stack] * unitParamsPerGpu(u) / (zero >= 1 ? dp * cp : 1), 0, gpu.memBandwidthGBs) : 0;
  const unitO = unitList.map((x) => optMsOf(x.c));
  const optOf = (sizes) => useNonGemm ? Math.max(...stageSum(sizes, unitO, embFirst ? optMsOf(preC) : 0, headLast ? optMsOf(postC) : 0)) : 0;
  const optNow = optOf(lps);
  const pipeOf = (stages2) => m == null ? null : simulated ? simulatedStepTime(stages2, m, v) : pipelineStepTime(stages2, m, v);
  const stepOf = (stages2, opt = optNow) => {
    const t = pipeOf(stages2);
    return t == null ? null : t + opt;
  };
  const minMax = pp > 1 ? optimalPartition(unitW, pp, head0W, tailW) : null;
  const best = minMax && simulated ? splitBySchedule(unitW, pp, head0W, tailW, (st) => stepOf(st), minMax, m) : minMax;
  const bestW = best ? stageSum(best, unitW, head0W, tailW) : sw;
  const bestMax = Math.max(...bestW);
  const balanced = sw.map(() => mean);
  const curStep = stepOf(sw), balStep = stepOf(balanced), bestStep = best ? stepOf(bestW, optOf(best)) : stepOf(bestW);
  const improves = best != null && (simulated && curStep != null && bestStep != null ? bestStep < curStep * (1 - 1e-9) : bestMax < maxW * (1 - 1e-9));
  const asymptoticSlowdown = maxW / mean - 1;
  const imbalance = {
    costBasis,
    maxOverMean: maxW / mean,
    slowestStage: slowest,
    estStepSlowdown: curStep != null && balStep != null ? curStep / balStep - 1 : asymptoticSlowdown,
    asymptoticSlowdown,
    step: {
      microbatches: m,
      formula: m == null ? "max / mean (microbatches per step unknown, so fill and drain are left out and this overstates the gap)" : simulated ? `${v > 1 ? `interleaved schedule with v = ${v}` : "1F1B schedule"} simulated op by op, a port of Primus's simulator, m = ${m}, forward a third of each stage` : v > 1 ? `t = m x max + (sum - max) / v, m = ${m}, v = ${v} (interleaved)` : `t = m x max + (sum - max), m = ${m} (1F1B)`,
      currentMs: costBasis === "time" ? curStep : null,
      balancedMs: costBasis === "time" ? balStep : null
    },
    rebalanced: best && improves ? {
      layersPerStage: best,
      maxOverMean: bestMax / mean,
      estStepSpeedup: curStep != null && bestStep != null ? curStep / bestStep - 1 : maxW / bestMax - 1,
      asymptoticSpeedup: maxW / bestMax - 1,
      stepMs: costBasis === "time" ? bestStep : null
    } : null
  };
  const costWord = costBasis === "time" ? "predicted time" : "FLOPs";
  if (pp > 1) {
    checked.push("stage-balance");
    const what = slowest === pp - 1 && headLast ? " (it also carries the LM head and the loss)" : slowest === 0 && embFirst ? " (it also carries the embedding)" : "";
    if (imbalance.estStepSlowdown > 0.03) {
      const reb = imbalance.rebalanced;
      const per = costBasis === "time" ? `Per-stage predicted ms per microbatch: ${sw.map((x) => x.toFixed(1)).join(" | ")}.` : `Per-stage FLOPs/token: ${sf.map((x) => fmtFlops2(x)).join(" | ")} (unknown card, so FLOPs stand in for time and the memory-bound loss is understated).`;
      add({
        rule: "stage-imbalance",
        severity: imbalance.estStepSlowdown >= STRAGGLE_THRESHOLD && reb && reb.estStepSpeedup > 0.03 ? "warn" : "info",
        title: `Stage ${slowest} has ${pct(asymptoticSlowdown)} more ${costWord} than the mean stage${what}`,
        detail: `${per} Every microbatch waits on the slowest stage, so ${m != null ? `at ${m} microbatches per step the predicted step is ${pct(imbalance.estStepSlowdown)} longer than a perfectly balanced split (${imbalance.step.formula})` : `the step is up to ${pct(imbalance.estStepSlowdown)} longer than a perfectly balanced split (microbatches per step unknown; the fill and drain that shrink this are left out)`}. Predicted from the graph, not measured.`,
        fix: reb ? `layersPerStage ${reb.layersPerStage.join(",")}: predicted ${pct(reb.estStepSpeedup)} faster steps${m != null ? ` at ${m} microbatches` : ""} (${pct(reb.asymptoticSpeedup)} as microbatches grow).` : "No contiguous split of whole blocks does better; consider moving the loss onto its own layer (account_for_loss_in_pipeline_split)."
      });
    }
    if (imbalance.rebalanced) {
      recommendations.push(`Rebalance: layersPerStage ${imbalance.rebalanced.layersPerStage.join(",")} (now ${lps.join(",")}); slowest stage drops from ${imbalance.maxOverMean.toFixed(2)}x to ${imbalance.rebalanced.maxOverMean.toFixed(2)}x the mean, predicted ${pct(imbalance.rebalanced.estStepSpeedup)} faster per step${m != null ? ` at ${m} microbatches` : ""}.`);
    }
  } else skipped.push({ check: "stage-balance", reason: "PP 1: one stage, nothing to balance." });
  let bubble = null;
  if (pp > 1 && m == null) {
    skipped.push({ check: "pipeline-bubble", reason: "the microbatches per step are unknown (no readable globalBatch or gradAccum); per-stage memory assumes at least PP microbatches in flight." });
  } else if (pp > 1 && m != null) {
    const mb = m;
    const frac = bubbleFraction(pp, mb, v);
    const need = microbatchesForBubble(pp, BUBBLE_WARN, v);
    bubble = { fraction: frac, microbatches: mb, microbatchesForTarget: frac > BUBBLE_WARN ? need : null };
    checked.push("pipeline-bubble");
    if (frac > BUBBLE_WARN) {
      add({
        rule: "pipeline-bubble",
        severity: "warn",
        title: `Pipeline bubble is ${pct(frac)} of each step`,
        detail: `${schedule === "interleaved" ? `Interleaved (v=${v})` : "1F1B"} with PP ${pp} and ${mb} microbatch${mb === 1 ? "" : "es"} per step: (p-1)${v > 1 ? "/v" : ""} / (m + (p-1)${v > 1 ? "/v" : ""}). Every GPU idles this share of the step.`,
        fix: `Use at least ${need} microbatches per step (raise grad-accum, lower micro-batch, or lower DP) to bring it under ${pct(BUBBLE_WARN)}${schedule === "1f1b" && pp > 2 ? "; the interleaved schedule also divides it by its virtual stages" : ""}.`
      });
      recommendations.push(`Bubble: ${need}+ microbatches per step brings it from ${pct(frac)} to under ${pct(BUBBLE_WARN)}.`);
    }
  }
  const shardDp = dp * cp;
  const stateBytesPerParam = W_BYTES / (zero >= 3 ? shardDp : 1) + G_BYTES / (zero >= 2 ? shardDp : 1) + O_BYTES / (zero >= 1 ? shardDp : 1);
  const actFactor = micro * ACT_BYTES / (tp * cp) / 1e9;
  const stages = [];
  for (let k = 0; k < lps.length; k++) {
    const [s, e] = stageRanges[k];
    const parts = unitList.slice(s, e);
    if (k === 0 && embFirst) parts.unshift({ kind: "pre", c: preC });
    if (k === lps.length - 1 && headLast) parts.push({ kind: "post", c: postC });
    const blocksHere = parts.filter((p) => p.kind === "block").map((p) => p.i + 1);
    const extras = parts.filter((p) => p.kind !== "block").map((p) => p.kind === "pre" ? "embedding" : "head+loss");
    const params = parts.reduce((a, p) => a + p.c.params, 0);
    const expert = parts.reduce((a, p) => a + p.c.expertParams, 0);
    const paramsPerGpu = (params - expert) / tp + expert / (tp * ep);
    const stateGb = paramsPerGpu * stateBytesPerParam * OPTIMIZER_K / 1e9;
    let actElems = 0, largest = 0, maxBlock = 0;
    for (const p of parts) {
      largest = Math.max(largest, p.c.largest);
      if (layout.activationCheckpointing && p.kind === "block") {
        actElems += p.c.boundaryElems * OUTPUT_K;
        maxBlock = Math.max(maxBlock, p.c.actElems);
      } else actElems += p.c.actElems;
    }
    actElems += maxBlock;
    const actPerMb = actElems * actFactor;
    const transientGb = largest * TRANSIENT_K * actFactor;
    const mb = m ?? pp;
    const inflightBase = schedule === "interleaved" ? (pp - k) * (1 + (pp - 1) / (pp * v)) : pp - k;
    const inflight = Math.min(mb * v, Math.max(1, inflightBase));
    const hasLoss = parts.some((p) => p.kind === "post");
    const lossLogitsGb = hasLoss && !fusedCe ? 2 * postC.logits * 4 * micro / (tp * cp) / 1e9 : 0;
    const peak = stateGb + actPerMb * inflight + transientGb + lossLogitsGb;
    const flopsHere = sf[k];
    const mm = gpu ? parts.reduce((a, p) => a + mmMs(p.c, p.kind === "block"), 0) : null;
    const mem = gpu ? parts.reduce((a, p) => a + memMs(p.c, p.kind === "block"), 0) : null;
    const memBytes = parts.reduce((a, p) => a + p.c.memBytes * (p.kind === "block" ? recompute : 1), 0) * perGpu;
    stages.push({
      index: k,
      layers: { first: blocksHere.length ? blocksHere[0] : null, last: blocksHere.length ? blocksHere[blocksHere.length - 1] : null, count: blocksHere.length, extras },
      params,
      flopsPerToken: flopsHere,
      estMsPerMicrobatch: mm != null && mem != null ? mm + mem : null,
      msBreakdown: mm != null && mem != null ? { matmul: mm, memoryBound: mem } : null,
      memBoundGbPerMicrobatch: memBytes / 1e9,
      activationGbPerMicrobatch: actPerMb,
      lossLogitsGb,
      inFlightMicrobatches: Math.round(inflight * 100) / 100,
      stateGbPerGpu: stateGb,
      peakGbPerGpu: peak,
      peakGbUpperPerGpu: upperPeakGb(peak),
      fitsGpu: gpu ? fitVerdict(peak, params, gpu.memoryGB) : null
    });
  }
  if (gpu) {
    checked.push("stage-memory");
    const over = stages.filter((s) => s.fitsGpu === "does_not_fit");
    const border = stages.filter((s) => s.fitsGpu === "borderline");
    if (over.length) {
      const w = over.reduce((a, b) => b.peakGbPerGpu > a.peakGbPerGpu ? b : a);
      add({
        rule: "stage-memory",
        severity: "block",
        title: `Stage ${over.map((s) => s.index).join(", ")} does not fit ${gpu.label}: ${gb(w.peakGbPerGpu)} per GPU on stage ${w.index}`,
        detail: `Weights, gradients and Adam state ${gb(w.stateGbPerGpu)} (ZeRO ${zero}, TP ${tp}${ep > 1 ? `, EP ${ep}` : ""}) plus activations for ${w.inFlightMicrobatches} in-flight microbatch${w.inFlightMicrobatches === 1 ? "" : "es"} at ${gb(w.activationGbPerMicrobatch)} each. Derived, with the same per-layer terms as the training-peak estimate.`,
        fix: memoryFix(zero, layout.activationCheckpointing, micro, tp, pp)
      });
    } else if (border.length) {
      add({
        rule: "stage-memory-borderline",
        severity: "warn",
        title: `Stage ${border.map((s) => s.index).join(", ")} is borderline on ${gpu.label}`,
        detail: "The point estimate fits, the calibrated upper bound (x1.43, 95% of held-out runs) with 1.15x headroom does not. Measure once before reserving the full job.",
        fix: memoryFix(zero, layout.activationCheckpointing, micro, tp, pp)
      });
    }
  }
  let dpSeq;
  if (!layout.seqLenHistogram?.length) {
    skipped.push({ check: "dp-seq-imbalance", reason: "no sequence-length histogram given (seqLenHistogram / --seq-hist); padded or packed fixed-length batches are assumed." });
  } else if (dp <= 1) {
    skipped.push({ check: "dp-seq-imbalance", reason: "DP 1: no ranks to wait on each other." });
  } else {
    checked.push("dp-seq-imbalance");
    const fAt = (L) => {
      const sh = propagateShapes(withSeqLen(model, L).model).shapes;
      let s = 0;
      for (const c of model.components) {
        const x = sh.get(c.id);
        s += estimateLayerFlops(c.type, c.params ?? {}, x?.in ?? [], x?.out ?? []) || 0;
      }
      return s;
    };
    const T = Math.max(2, seqLen - seqLen % 2);
    const f1 = fAt(T), f22 = fAt(T / 2);
    const b = Math.max(0, 2 * (f1 - 2 * f22) / (T * T));
    const a = Math.max(0, (f1 - b * T * T) / T);
    const r = seqImbalance(layout.seqLenHistogram, dp, micro * (m ?? 1), a, b);
    dpSeq = { expectedMaxOverMean: r.expectedMaxOverMean, estSlowdown: r.expectedMaxOverMean - 1, packedEstimate: r.packedEstimate, method: r.method };
    if (dpSeq.estSlowdown > SEQ_IMBALANCE_WARN) {
      add({
        rule: "dp-seq-imbalance",
        severity: "warn",
        title: `Uneven sequence lengths make the slowest of ${dp} DP ranks ~${pct(dpSeq.estSlowdown)} slower than the mean`,
        detail: `Each rank draws ${micro * (m ?? 1)} sequences per step from your histogram; work per sequence is modelled as a*len + b*len^2 from this graph's FLOPs (attention is the len^2 term). Every rank waits for the slowest at the gradient all-reduce. ${r.method === "monte-carlo" ? "Monte Carlo, fixed seed" : "Normal approximation"}; derived, not measured.`,
        fix: `Pack sequences to a fixed token budget per rank (with per-document attention masking), predicted ${pct(dpSeq.packedEstimate - 1)}; or bucket by length so ranks in a step draw similar lengths.`
      });
      recommendations.push(`Sequence packing: slowest-rank overhead ${pct(dpSeq.estSlowdown)} -> ${pct(dpSeq.packedEstimate - 1)} (predicted).`);
    }
  }
  return finish(imbalance, stages, bubble, dpSeq);
  function finish(imb, st, bub, seq) {
    const order = { block: 0, warn: 1, info: 2 };
    findings.sort((x, y) => order[x.severity] - order[y.severity]);
    return {
      checked,
      skipped,
      findings,
      resolved: { ...resolvedBase, layersPerStage: st.length ? lps : null },
      stages: st,
      imbalance: imb,
      bubble: bub,
      ...seq ? { dpSeqImbalance: seq } : {},
      recommendations,
      basis: LAYOUT_BASIS
    };
  }
}
function memoryFix(zero, ac, micro, tp, pp) {
  const opts = [];
  if (zero < 1) opts.push("ZeRO-1 / distributed optimizer (shards the 8 bytes/param of Adam state across DP)");
  if (!ac) opts.push("activation checkpointing");
  if (micro > 1) opts.push("a smaller micro-batch");
  if (tp < NODE_SIZE) opts.push(`TP ${tp * 2}`);
  opts.push(`PP ${pp * 2}`);
  return `Options, cheapest first: ${opts.join("; ")}.`;
}
function fmtFlops2(f) {
  if (f >= 1e12) return `${(f / 1e12).toFixed(2)} TFLOP`;
  if (f >= 1e9) return `${(f / 1e9).toFixed(2)} GFLOP`;
  if (f >= 1e6) return `${(f / 1e6).toFixed(1)} MFLOP`;
  return `${Math.round(f)} FLOP`;
}
function renderLayoutText(r, name = "model") {
  const L = [];
  const res = r.resolved;
  L.push(`Parallel layout check: ${name}`);
  L.push(`  ${res.worldSize} GPUs${res.gpu ? ` (${res.gpu}, ${res.gpuMemoryGb} GB)` : ""}, ${res.blocks} blocks, schedule ${res.schedule}${res.virtualStages > 1 ? ` v=${res.virtualStages}` : ""}, ZeRO ${res.zeroStage}${res.microbatchesPerStep != null ? `, ${res.microbatchesPerStep} microbatches/step` : ""}${res.globalBatch != null ? `, global batch ${res.globalBatch}` : ""}`);
  if (r.stages.length) {
    L.push("");
    const head = ["stage", "blocks", "extra", "params", "FLOPs/tok", "ms/mb*", "mem-bound", "vs mean", "act/mb", "in-flight", "peak/GPU", "upper", "fits"];
    const byTime = r.stages.every((s) => s.estMsPerMicrobatch != null);
    const cost = (s) => byTime ? s.estMsPerMicrobatch : s.flopsPerToken;
    const mean = r.stages.reduce((a, s) => a + cost(s), 0) / r.stages.length;
    const rows = r.stages.map((s) => [
      String(s.index),
      s.layers.count ? `${s.layers.first}-${s.layers.last} (${s.layers.count})` : "-",
      s.layers.extras.join("+") || "-",
      s.params >= 1e9 ? `${(s.params / 1e9).toFixed(2)}B` : `${(s.params / 1e6).toFixed(0)}M`,
      fmtFlops2(s.flopsPerToken),
      s.estMsPerMicrobatch != null ? s.estMsPerMicrobatch.toFixed(0) : "-",
      s.msBreakdown && s.estMsPerMicrobatch ? `${Math.round(s.msBreakdown.memoryBound / s.estMsPerMicrobatch * 100)}%` : "-",
      `${(cost(s) / mean).toFixed(2)}x`,
      gb(s.activationGbPerMicrobatch),
      String(s.inFlightMicrobatches),
      gb(s.peakGbPerGpu),
      gb(s.peakGbUpperPerGpu),
      s.fitsGpu == null ? "n/a" : s.fitsGpu === "does_not_fit" ? "NO" : s.fitsGpu
    ]);
    const w = head.map((h, i) => Math.max(h.length, ...rows.map((row) => row[i].length)));
    const line = (cells) => "  " + cells.map((c, i) => i <= 2 ? c.padEnd(w[i]) : c.padStart(w[i])).join("  ");
    L.push(line(head));
    for (const row of rows) L.push(line(row));
    L.push(`  * predicted fwd+bwd ms per microbatch per GPU: matmuls at up to ${Math.round(COMPUTE_BOUND_MFU * 100)}% of peak FLOPs (less for short GEMMs${r.resolved.gpu && GEMM_CURVE_CARDS.includes(r.resolved.gpu) ? `, from a measured curve for this card` : ", no measured curve for this card so flat"}) plus memory-bound bytes (the mem-bound share) at ${Math.round(MEM_BW_EFFICIENCY * 100)}% of memory bandwidth. Not measured.`);
    const loss = r.stages.find((s) => s.lossLogitsGb > 0);
    if (loss) L.push(`  Stage ${loss.index}'s peak includes ${gb(loss.lossLogitsGb)} of fp32 logits and their gradient for the loss (--fused-ce if the loss is fused or chunked).`);
  }
  if (r.imbalance) {
    const im = r.imbalance;
    const unit = im.costBasis === "time" ? "predicted time" : "FLOPs (unknown card)";
    L.push("");
    L.push(`  Balance: slowest stage ${im.slowestStage} at ${im.maxOverMean.toFixed(2)}x the mean stage's ${unit}.`);
    if (im.step.microbatches != null) {
      const ms = im.step.currentMs != null && im.step.balancedMs != null ? `: ${im.step.currentMs.toFixed(0)} ms vs ${im.step.balancedMs.toFixed(0)} ms balanced` : "";
      L.push(`  Predicted step at ${im.step.microbatches} microbatches (${im.step.formula.replace(/, m = \d+.*$/, "")})${ms}, ${pct(im.estStepSlowdown)} slower than a perfect split (${pct(im.asymptoticSlowdown)} as microbatches grow).`);
    } else {
      L.push(`  Microbatches per step unknown: up to ${pct(im.estStepSlowdown)} slower than a perfect split, the asymptotic ratio, which overstates the gap at small microbatch counts.`);
    }
    if (im.rebalanced) {
      const rb = im.rebalanced;
      L.push(`  Rebalanced split ${rb.layersPerStage.join(",")}: ${rb.maxOverMean.toFixed(2)}x, predicted ${pct(rb.estStepSpeedup)} faster per step${im.step.microbatches != null ? ` at ${im.step.microbatches} microbatches${rb.stepMs != null ? ` (${rb.stepMs.toFixed(0)} ms)` : ""}; ${pct(rb.asymptoticSpeedup)} as microbatches grow` : ""}.`);
    }
  }
  if (r.bubble) L.push(`  Bubble: ${pct(r.bubble.fraction)} at ${r.bubble.microbatches} microbatch${r.bubble.microbatches === 1 ? "" : "es"}/step${r.bubble.microbatchesForTarget ? ` (${r.bubble.microbatchesForTarget}+ for under ${pct(BUBBLE_WARN)})` : ""}.`);
  if (r.dpSeqImbalance) L.push(`  DP sequence imbalance: slowest rank ${r.dpSeqImbalance.expectedMaxOverMean.toFixed(3)}x the mean; packed ${r.dpSeqImbalance.packedEstimate.toFixed(3)}x (${r.dpSeqImbalance.method}).`);
  L.push("");
  if (!r.findings.length) L.push("  No findings on the checks that ran.");
  for (const f of r.findings) {
    L.push(`  ${f.severity.padEnd(5)} ${f.title}`);
    L.push(`        ${f.detail}`);
    L.push(`        fix: ${f.fix}`);
  }
  L.push("");
  L.push(`  checked: ${r.checked.join(", ") || "none"}`);
  for (const s of r.skipped) L.push(`  skipped ${s.check}: ${s.reason}`);
  L.push("");
  L.push(`  ${r.basis}`);
  return L.join("\n") + "\n";
}

// node_modules/@huggingface/jinja/dist/index.js
var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => {
  __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};
var TOKEN_TYPES = Object.freeze({
  Text: "Text",
  // The text between Jinja statements or expressions
  NumericLiteral: "NumericLiteral",
  // e.g., 123, 1.0
  StringLiteral: "StringLiteral",
  // 'string'
  Identifier: "Identifier",
  // Variables, functions, statements, booleans, etc.
  Equals: "Equals",
  // =
  OpenParen: "OpenParen",
  // (
  CloseParen: "CloseParen",
  // )
  OpenStatement: "OpenStatement",
  // {%
  CloseStatement: "CloseStatement",
  // %}
  OpenExpression: "OpenExpression",
  // {{
  CloseExpression: "CloseExpression",
  // }}
  OpenSquareBracket: "OpenSquareBracket",
  // [
  CloseSquareBracket: "CloseSquareBracket",
  // ]
  OpenCurlyBracket: "OpenCurlyBracket",
  // {
  CloseCurlyBracket: "CloseCurlyBracket",
  // }
  Comma: "Comma",
  // ,
  Dot: "Dot",
  // .
  Colon: "Colon",
  // :
  Pipe: "Pipe",
  // |
  CallOperator: "CallOperator",
  // ()
  AdditiveBinaryOperator: "AdditiveBinaryOperator",
  // + - ~
  MultiplicativeBinaryOperator: "MultiplicativeBinaryOperator",
  // * / // %
  ExponentiationBinaryOperator: "ExponentiationBinaryOperator",
  // **
  ComparisonBinaryOperator: "ComparisonBinaryOperator",
  // < > <= >= == !=
  UnaryOperator: "UnaryOperator",
  // ! - +
  Comment: "Comment"
  // {# ... #}
});
var Token = class {
  /**
   * Constructs a new Token.
   * @param {string} value The raw value as seen inside the source code.
   * @param {TokenType} type The type of token.
   */
  constructor(value, type) {
    this.value = value;
    this.type = type;
  }
};
function isWord(char) {
  return /\w/.test(char);
}
function isInteger(char) {
  return /[0-9]/.test(char);
}
function isWhitespace(char) {
  return /\s/.test(char);
}
var ORDERED_MAPPING_TABLE = [
  // Control sequences
  ["{%", TOKEN_TYPES.OpenStatement],
  ["%}", TOKEN_TYPES.CloseStatement],
  ["{{", TOKEN_TYPES.OpenExpression],
  ["}}", TOKEN_TYPES.CloseExpression],
  // Single character tokens
  ["(", TOKEN_TYPES.OpenParen],
  [")", TOKEN_TYPES.CloseParen],
  ["{", TOKEN_TYPES.OpenCurlyBracket],
  ["}", TOKEN_TYPES.CloseCurlyBracket],
  ["[", TOKEN_TYPES.OpenSquareBracket],
  ["]", TOKEN_TYPES.CloseSquareBracket],
  [",", TOKEN_TYPES.Comma],
  [".", TOKEN_TYPES.Dot],
  [":", TOKEN_TYPES.Colon],
  ["|", TOKEN_TYPES.Pipe],
  // Comparison operators
  ["<=", TOKEN_TYPES.ComparisonBinaryOperator],
  [">=", TOKEN_TYPES.ComparisonBinaryOperator],
  ["==", TOKEN_TYPES.ComparisonBinaryOperator],
  ["!=", TOKEN_TYPES.ComparisonBinaryOperator],
  ["<", TOKEN_TYPES.ComparisonBinaryOperator],
  [">", TOKEN_TYPES.ComparisonBinaryOperator],
  // Arithmetic operators
  ["+", TOKEN_TYPES.AdditiveBinaryOperator],
  ["-", TOKEN_TYPES.AdditiveBinaryOperator],
  ["~", TOKEN_TYPES.AdditiveBinaryOperator],
  ["**", TOKEN_TYPES.ExponentiationBinaryOperator],
  // NOTE: must come before "*" so that "**" is matched first
  ["*", TOKEN_TYPES.MultiplicativeBinaryOperator],
  ["//", TOKEN_TYPES.MultiplicativeBinaryOperator],
  // NOTE: must come before "/" so that "//" is matched first
  ["/", TOKEN_TYPES.MultiplicativeBinaryOperator],
  ["%", TOKEN_TYPES.MultiplicativeBinaryOperator],
  // Assignment operator
  ["=", TOKEN_TYPES.Equals]
];
var ESCAPE_CHARACTERS = /* @__PURE__ */ new Map([
  ["n", "\n"],
  // New line
  ["t", "	"],
  // Horizontal tab
  ["r", "\r"],
  // Carriage return
  ["b", "\b"],
  // Backspace
  ["f", "\f"],
  // Form feed
  ["v", "\v"],
  // Vertical tab
  ["'", "'"],
  // Single quote
  ['"', '"'],
  // Double quote
  ["\\", "\\"]
  // Backslash
]);
function preprocess(template, options = {}) {
  if (template.endsWith("\n")) {
    template = template.slice(0, -1);
  }
  if (options.lstrip_blocks) {
    template = template.replace(/^[ \t]*({[#%-])/gm, "$1");
  }
  if (options.trim_blocks) {
    template = template.replace(/([#%-]})\n/g, "$1");
  }
  return template.replace(
    /(\s*){%(-?)\s*(?:end)?generation\s*(-?)%}(\s*)/gs,
    (_, before, lstrip, rstrip, after) => (lstrip ? "" : before) + (rstrip ? "" : after)
  );
}
function tokenize(source, options = {}) {
  const tokens = [];
  const src = preprocess(source, options);
  let cursorPosition = 0;
  let curlyBracketDepth = 0;
  const consumeWhile = (predicate) => {
    let str2 = "";
    while (predicate(src[cursorPosition])) {
      if (src[cursorPosition] === "\\") {
        ++cursorPosition;
        if (cursorPosition >= src.length) {
          throw new SyntaxError("Unexpected end of input");
        }
        const escaped = src[cursorPosition++];
        const unescaped = ESCAPE_CHARACTERS.get(escaped);
        if (unescaped === void 0) {
          throw new SyntaxError(`Unexpected escaped character: ${escaped}`);
        }
        str2 += unescaped;
        continue;
      }
      str2 += src[cursorPosition++];
      if (cursorPosition >= src.length) {
        throw new SyntaxError("Unexpected end of input");
      }
    }
    return str2;
  };
  const stripTrailingWhitespace = () => {
    const lastToken = tokens.at(-1);
    if (lastToken && lastToken.type === TOKEN_TYPES.Text) {
      lastToken.value = lastToken.value.trimEnd();
      if (lastToken.value === "") {
        tokens.pop();
      }
    }
  };
  const skipLeadingWhitespace = () => {
    while (cursorPosition < src.length && isWhitespace(src[cursorPosition])) {
      ++cursorPosition;
    }
  };
  main:
    while (cursorPosition < src.length) {
      const lastTokenType = tokens.at(-1)?.type;
      if (lastTokenType === void 0 || lastTokenType === TOKEN_TYPES.CloseStatement || lastTokenType === TOKEN_TYPES.CloseExpression || lastTokenType === TOKEN_TYPES.Comment) {
        let text = "";
        while (cursorPosition < src.length && // Keep going until we hit the next Jinja statement or expression
        !(src[cursorPosition] === "{" && (src[cursorPosition + 1] === "%" || src[cursorPosition + 1] === "{" || src[cursorPosition + 1] === "#"))) {
          text += src[cursorPosition++];
        }
        if (text.length > 0) {
          tokens.push(new Token(text, TOKEN_TYPES.Text));
          continue;
        }
      }
      if (src[cursorPosition] === "{" && src[cursorPosition + 1] === "#") {
        cursorPosition += 2;
        const stripBefore = src[cursorPosition] === "-";
        if (stripBefore) {
          ++cursorPosition;
        }
        let comment = "";
        while (src[cursorPosition] !== "#" || src[cursorPosition + 1] !== "}") {
          if (cursorPosition + 2 >= src.length) {
            throw new SyntaxError("Missing end of comment tag");
          }
          comment += src[cursorPosition++];
        }
        const stripAfter = comment.endsWith("-");
        if (stripAfter) {
          comment = comment.slice(0, -1);
        }
        if (stripBefore) {
          stripTrailingWhitespace();
        }
        tokens.push(new Token(comment, TOKEN_TYPES.Comment));
        cursorPosition += 2;
        if (stripAfter) {
          skipLeadingWhitespace();
        }
        continue;
      }
      if (src.slice(cursorPosition, cursorPosition + 3) === "{%-") {
        stripTrailingWhitespace();
        tokens.push(new Token("{%", TOKEN_TYPES.OpenStatement));
        cursorPosition += 3;
        continue;
      }
      if (src.slice(cursorPosition, cursorPosition + 3) === "{{-") {
        stripTrailingWhitespace();
        tokens.push(new Token("{{", TOKEN_TYPES.OpenExpression));
        curlyBracketDepth = 0;
        cursorPosition += 3;
        continue;
      }
      consumeWhile(isWhitespace);
      if (src.slice(cursorPosition, cursorPosition + 3) === "-%}") {
        tokens.push(new Token("%}", TOKEN_TYPES.CloseStatement));
        cursorPosition += 3;
        skipLeadingWhitespace();
        continue;
      }
      if (src.slice(cursorPosition, cursorPosition + 3) === "-}}") {
        tokens.push(new Token("}}", TOKEN_TYPES.CloseExpression));
        cursorPosition += 3;
        skipLeadingWhitespace();
        continue;
      }
      const char = src[cursorPosition];
      if (char === "-" || char === "+") {
        const lastTokenType2 = tokens.at(-1)?.type;
        if (lastTokenType2 === TOKEN_TYPES.Text || lastTokenType2 === void 0) {
          throw new SyntaxError(`Unexpected character: ${char}`);
        }
        switch (lastTokenType2) {
          case TOKEN_TYPES.Identifier:
          case TOKEN_TYPES.NumericLiteral:
          case TOKEN_TYPES.StringLiteral:
          case TOKEN_TYPES.CloseParen:
          case TOKEN_TYPES.CloseSquareBracket:
            break;
          default: {
            ++cursorPosition;
            let num7 = consumeWhile(isInteger);
            if (num7.length > 0 && src[cursorPosition] === "." && isInteger(src[cursorPosition + 1])) {
              ++cursorPosition;
              const frac = consumeWhile(isInteger);
              num7 = `${num7}.${frac}`;
            }
            tokens.push(
              new Token(`${char}${num7}`, num7.length > 0 ? TOKEN_TYPES.NumericLiteral : TOKEN_TYPES.UnaryOperator)
            );
            continue;
          }
        }
      }
      for (const [seq, type] of ORDERED_MAPPING_TABLE) {
        if (seq === "}}" && curlyBracketDepth > 0) {
          continue;
        }
        const slice2 = src.slice(cursorPosition, cursorPosition + seq.length);
        if (slice2 === seq) {
          tokens.push(new Token(seq, type));
          if (type === TOKEN_TYPES.OpenExpression) {
            curlyBracketDepth = 0;
          } else if (type === TOKEN_TYPES.OpenCurlyBracket) {
            ++curlyBracketDepth;
          } else if (type === TOKEN_TYPES.CloseCurlyBracket) {
            --curlyBracketDepth;
          }
          cursorPosition += seq.length;
          continue main;
        }
      }
      if (char === "'" || char === '"') {
        ++cursorPosition;
        const str2 = consumeWhile((c) => c !== char);
        tokens.push(new Token(str2, TOKEN_TYPES.StringLiteral));
        ++cursorPosition;
        continue;
      }
      if (isInteger(char)) {
        let num7 = consumeWhile(isInteger);
        if (tokens.at(-1)?.type !== TOKEN_TYPES.Dot && src[cursorPosition] === "." && isInteger(src[cursorPosition + 1])) {
          ++cursorPosition;
          const frac = consumeWhile(isInteger);
          num7 = `${num7}.${frac}`;
        }
        tokens.push(new Token(num7, TOKEN_TYPES.NumericLiteral));
        continue;
      }
      if (isWord(char)) {
        const word = consumeWhile(isWord);
        tokens.push(new Token(word, TOKEN_TYPES.Identifier));
        continue;
      }
      throw new SyntaxError(`Unexpected character: ${char}`);
    }
  return tokens;
}
var Statement = class {
  type = "Statement";
};
var Program = class extends Statement {
  constructor(body) {
    super();
    this.body = body;
  }
  type = "Program";
};
var If = class extends Statement {
  constructor(test, body, alternate) {
    super();
    this.test = test;
    this.body = body;
    this.alternate = alternate;
  }
  type = "If";
};
var For = class extends Statement {
  constructor(loopvar, iterable, body, defaultBlock) {
    super();
    this.loopvar = loopvar;
    this.iterable = iterable;
    this.body = body;
    this.defaultBlock = defaultBlock;
  }
  type = "For";
};
var Break = class extends Statement {
  type = "Break";
};
var Continue = class extends Statement {
  type = "Continue";
};
var SetStatement = class extends Statement {
  constructor(assignee, value, body) {
    super();
    this.assignee = assignee;
    this.value = value;
    this.body = body;
  }
  type = "Set";
};
var Macro = class extends Statement {
  constructor(name, args, body) {
    super();
    this.name = name;
    this.args = args;
    this.body = body;
  }
  type = "Macro";
};
var Comment = class extends Statement {
  constructor(value) {
    super();
    this.value = value;
  }
  type = "Comment";
};
var Expression = class extends Statement {
  type = "Expression";
};
var MemberExpression = class extends Expression {
  constructor(object, property, computed) {
    super();
    this.object = object;
    this.property = property;
    this.computed = computed;
  }
  type = "MemberExpression";
};
var CallExpression = class extends Expression {
  constructor(callee, args) {
    super();
    this.callee = callee;
    this.args = args;
  }
  type = "CallExpression";
};
var Identifier = class extends Expression {
  /**
   * @param {string} value The name of the identifier
   */
  constructor(value) {
    super();
    this.value = value;
  }
  type = "Identifier";
};
var Literal = class extends Expression {
  constructor(value) {
    super();
    this.value = value;
  }
  type = "Literal";
};
var IntegerLiteral = class extends Literal {
  type = "IntegerLiteral";
};
var FloatLiteral = class extends Literal {
  type = "FloatLiteral";
};
var StringLiteral = class extends Literal {
  type = "StringLiteral";
};
var ArrayLiteral = class extends Literal {
  type = "ArrayLiteral";
};
var TupleLiteral = class extends Literal {
  type = "TupleLiteral";
};
var ObjectLiteral = class extends Literal {
  type = "ObjectLiteral";
};
var BinaryExpression = class extends Expression {
  constructor(operator, left, right) {
    super();
    this.operator = operator;
    this.left = left;
    this.right = right;
  }
  type = "BinaryExpression";
};
var FilterExpression = class extends Expression {
  constructor(operand, filter) {
    super();
    this.operand = operand;
    this.filter = filter;
  }
  type = "FilterExpression";
};
var FilterStatement = class extends Statement {
  constructor(filter, body) {
    super();
    this.filter = filter;
    this.body = body;
  }
  type = "FilterStatement";
};
var SelectExpression = class extends Expression {
  constructor(lhs, test) {
    super();
    this.lhs = lhs;
    this.test = test;
  }
  type = "SelectExpression";
};
var TestExpression = class extends Expression {
  constructor(operand, negate, test) {
    super();
    this.operand = operand;
    this.negate = negate;
    this.test = test;
  }
  type = "TestExpression";
};
var UnaryExpression = class extends Expression {
  constructor(operator, argument) {
    super();
    this.operator = operator;
    this.argument = argument;
  }
  type = "UnaryExpression";
};
var SliceExpression = class extends Expression {
  constructor(start = void 0, stop = void 0, step = void 0) {
    super();
    this.start = start;
    this.stop = stop;
    this.step = step;
  }
  type = "SliceExpression";
};
var KeywordArgumentExpression = class extends Expression {
  constructor(key, value) {
    super();
    this.key = key;
    this.value = value;
  }
  type = "KeywordArgumentExpression";
};
var SpreadExpression = class extends Expression {
  constructor(argument) {
    super();
    this.argument = argument;
  }
  type = "SpreadExpression";
};
var KeywordSpreadExpression = class extends Expression {
  constructor(argument) {
    super();
    this.argument = argument;
  }
  type = "KeywordSpreadExpression";
};
var CallStatement = class extends Statement {
  constructor(call, callerArgs, body) {
    super();
    this.call = call;
    this.callerArgs = callerArgs;
    this.body = body;
  }
  type = "CallStatement";
};
var Ternary = class extends Expression {
  constructor(condition, trueExpr, falseExpr) {
    super();
    this.condition = condition;
    this.trueExpr = trueExpr;
    this.falseExpr = falseExpr;
  }
  type = "Ternary";
};
function parse(tokens) {
  const program = new Program([]);
  let current = 0;
  function expect(type, error) {
    const prev = tokens[current++];
    if (!prev || prev.type !== type) {
      throw new Error(`Parser Error: ${error}. ${prev.type} !== ${type}.`);
    }
    return prev;
  }
  function expectIdentifier(name) {
    if (!isIdentifier(name)) {
      throw new SyntaxError(`Expected ${name}`);
    }
    ++current;
  }
  function parseAny() {
    switch (tokens[current].type) {
      case TOKEN_TYPES.Comment:
        return new Comment(tokens[current++].value);
      case TOKEN_TYPES.Text:
        return parseText();
      case TOKEN_TYPES.OpenStatement:
        return parseJinjaStatement();
      case TOKEN_TYPES.OpenExpression:
        return parseJinjaExpression();
      default:
        throw new SyntaxError(`Unexpected token type: ${tokens[current].type}`);
    }
  }
  function is(...types) {
    return current + types.length <= tokens.length && types.every((type, i) => type === tokens[current + i].type);
  }
  function isStatement(...names) {
    return tokens[current]?.type === TOKEN_TYPES.OpenStatement && tokens[current + 1]?.type === TOKEN_TYPES.Identifier && names.includes(tokens[current + 1]?.value);
  }
  function isIdentifier(...names) {
    return current + names.length <= tokens.length && names.every((name, i) => tokens[current + i].type === "Identifier" && name === tokens[current + i].value);
  }
  function parseText() {
    return new StringLiteral(expect(TOKEN_TYPES.Text, "Expected text token").value);
  }
  function parseJinjaStatement() {
    expect(TOKEN_TYPES.OpenStatement, "Expected opening statement token");
    if (tokens[current].type !== TOKEN_TYPES.Identifier) {
      throw new SyntaxError(`Unknown statement, got ${tokens[current].type}`);
    }
    const name = tokens[current].value;
    let result;
    switch (name) {
      case "set":
        ++current;
        result = parseSetStatement();
        break;
      case "if":
        ++current;
        result = parseIfStatement();
        expect(TOKEN_TYPES.OpenStatement, "Expected {% token");
        expectIdentifier("endif");
        expect(TOKEN_TYPES.CloseStatement, "Expected %} token");
        break;
      case "macro":
        ++current;
        result = parseMacroStatement();
        expect(TOKEN_TYPES.OpenStatement, "Expected {% token");
        expectIdentifier("endmacro");
        expect(TOKEN_TYPES.CloseStatement, "Expected %} token");
        break;
      case "for":
        ++current;
        result = parseForStatement();
        expect(TOKEN_TYPES.OpenStatement, "Expected {% token");
        expectIdentifier("endfor");
        expect(TOKEN_TYPES.CloseStatement, "Expected %} token");
        break;
      case "call": {
        ++current;
        let callerArgs = null;
        if (is(TOKEN_TYPES.OpenParen)) {
          callerArgs = parseArgs("parameters");
        }
        const callee = parsePrimaryExpression();
        if (callee.type !== "Identifier") {
          throw new SyntaxError(`Expected identifier following call statement`);
        }
        const callArgs = parseArgs("call");
        expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
        const body = [];
        while (!isStatement("endcall")) {
          body.push(parseAny());
        }
        expect(TOKEN_TYPES.OpenStatement, "Expected '{%'");
        expectIdentifier("endcall");
        expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
        const callExpr = new CallExpression(callee, callArgs);
        result = new CallStatement(callExpr, callerArgs, body);
        break;
      }
      case "break":
        ++current;
        expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
        result = new Break();
        break;
      case "continue":
        ++current;
        expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
        result = new Continue();
        break;
      case "filter": {
        ++current;
        let filterNode = parsePrimaryExpression();
        if (filterNode instanceof Identifier && is(TOKEN_TYPES.OpenParen)) {
          filterNode = parseCallExpression(filterNode);
        }
        expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
        const filterBody = [];
        while (!isStatement("endfilter")) {
          filterBody.push(parseAny());
        }
        expect(TOKEN_TYPES.OpenStatement, "Expected '{%'");
        expectIdentifier("endfilter");
        expect(TOKEN_TYPES.CloseStatement, "Expected '%}'");
        result = new FilterStatement(filterNode, filterBody);
        break;
      }
      default:
        throw new SyntaxError(`Unknown statement type: ${name}`);
    }
    return result;
  }
  function parseJinjaExpression() {
    expect(TOKEN_TYPES.OpenExpression, "Expected opening expression token");
    const result = parseExpression();
    expect(TOKEN_TYPES.CloseExpression, "Expected closing expression token");
    return result;
  }
  function parseSetStatement() {
    const left = parseExpressionSequence();
    let value = null;
    const body = [];
    if (is(TOKEN_TYPES.Equals)) {
      ++current;
      value = parseExpressionSequence();
    } else {
      expect(TOKEN_TYPES.CloseStatement, "Expected %} token");
      while (!isStatement("endset")) {
        body.push(parseAny());
      }
      expect(TOKEN_TYPES.OpenStatement, "Expected {% token");
      expectIdentifier("endset");
    }
    expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
    return new SetStatement(left, value, body);
  }
  function parseIfStatement() {
    const test = parseExpression();
    expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
    const body = [];
    const alternate = [];
    while (!isStatement("elif", "else", "endif")) {
      body.push(parseAny());
    }
    if (isStatement("elif")) {
      ++current;
      ++current;
      const result = parseIfStatement();
      alternate.push(result);
    } else if (isStatement("else")) {
      ++current;
      ++current;
      expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
      while (!isStatement("endif")) {
        alternate.push(parseAny());
      }
    }
    return new If(test, body, alternate);
  }
  function parseMacroStatement() {
    const name = parsePrimaryExpression();
    if (name.type !== "Identifier") {
      throw new SyntaxError(`Expected identifier following macro statement`);
    }
    const args = parseArgs("parameters");
    expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
    const body = [];
    while (!isStatement("endmacro")) {
      body.push(parseAny());
    }
    return new Macro(name, args, body);
  }
  function parseExpressionSequence(primary = false) {
    const fn = primary ? parsePrimaryExpression : parseExpression;
    const expressions = [fn()];
    const isTuple = is(TOKEN_TYPES.Comma);
    while (isTuple) {
      ++current;
      expressions.push(fn());
      if (!is(TOKEN_TYPES.Comma)) {
        break;
      }
    }
    return isTuple ? new TupleLiteral(expressions) : expressions[0];
  }
  function parseForStatement() {
    const loopVariable = parseExpressionSequence(true);
    if (!(loopVariable instanceof Identifier || loopVariable instanceof TupleLiteral)) {
      throw new SyntaxError(`Expected identifier/tuple for the loop variable, got ${loopVariable.type} instead`);
    }
    if (!isIdentifier("in")) {
      throw new SyntaxError("Expected `in` keyword following loop variable");
    }
    ++current;
    const iterable = parseExpression();
    expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
    const body = [];
    while (!isStatement("endfor", "else")) {
      body.push(parseAny());
    }
    const alternative = [];
    if (isStatement("else")) {
      ++current;
      ++current;
      expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
      while (!isStatement("endfor")) {
        alternative.push(parseAny());
      }
    }
    return new For(loopVariable, iterable, body, alternative);
  }
  function parseExpression() {
    return parseIfExpression();
  }
  function parseIfExpression() {
    const a = parseLogicalOrExpression();
    if (isIdentifier("if")) {
      ++current;
      const test = parseLogicalOrExpression();
      if (isIdentifier("else")) {
        ++current;
        const falseExpr = parseIfExpression();
        return new Ternary(test, a, falseExpr);
      } else {
        return new SelectExpression(a, test);
      }
    }
    return a;
  }
  function parseLogicalOrExpression() {
    let left = parseLogicalAndExpression();
    while (isIdentifier("or")) {
      const operator = tokens[current];
      ++current;
      const right = parseLogicalAndExpression();
      left = new BinaryExpression(operator, left, right);
    }
    return left;
  }
  function parseLogicalAndExpression() {
    let left = parseLogicalNegationExpression();
    while (isIdentifier("and")) {
      const operator = tokens[current];
      ++current;
      const right = parseLogicalNegationExpression();
      left = new BinaryExpression(operator, left, right);
    }
    return left;
  }
  function parseLogicalNegationExpression() {
    let right;
    while (isIdentifier("not")) {
      const operator = tokens[current];
      ++current;
      const arg = parseLogicalNegationExpression();
      right = new UnaryExpression(operator, arg);
    }
    return right ?? parseComparisonExpression();
  }
  function parseComparisonExpression() {
    let left = parseAdditiveExpression();
    while (true) {
      let operator;
      if (isIdentifier("not", "in")) {
        operator = new Token("not in", TOKEN_TYPES.Identifier);
        current += 2;
      } else if (isIdentifier("in")) {
        operator = tokens[current++];
      } else if (is(TOKEN_TYPES.ComparisonBinaryOperator)) {
        operator = tokens[current++];
      } else {
        break;
      }
      const right = parseAdditiveExpression();
      left = new BinaryExpression(operator, left, right);
    }
    return left;
  }
  function parseAdditiveExpression() {
    let left = parseMultiplicativeExpression();
    while (is(TOKEN_TYPES.AdditiveBinaryOperator)) {
      const operator = tokens[current];
      ++current;
      const right = parseMultiplicativeExpression();
      left = new BinaryExpression(operator, left, right);
    }
    return left;
  }
  function parseCallMemberExpression() {
    const member = parseMemberExpression(parsePrimaryExpression());
    if (is(TOKEN_TYPES.OpenParen)) {
      return parseCallExpression(member);
    }
    return member;
  }
  function parseCallExpression(callee) {
    let expression = new CallExpression(callee, parseArgs("call"));
    expression = parseMemberExpression(expression);
    if (is(TOKEN_TYPES.OpenParen)) {
      expression = parseCallExpression(expression);
    }
    return expression;
  }
  function parseArgs(mode) {
    expect(TOKEN_TYPES.OpenParen, "Expected opening parenthesis for arguments list");
    const args = parseArgumentsList(mode);
    expect(TOKEN_TYPES.CloseParen, "Expected closing parenthesis for arguments list");
    return args;
  }
  function parseArgumentsList(mode) {
    const args = [];
    const parameterNames = /* @__PURE__ */ new Set();
    let sawKeywordOrDefault = false;
    let sawSpreadArgument = false;
    while (!is(TOKEN_TYPES.CloseParen)) {
      const isKeywordSpread = is(TOKEN_TYPES.ExponentiationBinaryOperator);
      const isPositionalSpread = is(TOKEN_TYPES.MultiplicativeBinaryOperator) && tokens[current].value === "*";
      if (mode === "parameters" && (isKeywordSpread || isPositionalSpread)) {
        throw new SyntaxError("Argument unpacking is not allowed in parameter declarations");
      }
      if (isKeywordSpread) {
        ++current;
        args.push(new KeywordSpreadExpression(parseExpression()));
        if (is(TOKEN_TYPES.Comma)) {
          ++current;
        }
        if (!is(TOKEN_TYPES.CloseParen)) {
          throw new SyntaxError("Expected closing parenthesis: `**` must be applied to the final argument");
        }
        break;
      }
      let argument;
      if (isPositionalSpread) {
        if (sawSpreadArgument) {
          throw new SyntaxError("Only one `*` argument unpacking is allowed");
        }
        sawSpreadArgument = true;
        ++current;
        const expr = parseExpression();
        argument = new SpreadExpression(expr);
      } else {
        argument = parseExpression();
        if (is(TOKEN_TYPES.Equals)) {
          ++current;
          if (!(argument instanceof Identifier)) {
            throw new SyntaxError(
              mode === "parameters" ? "Expected identifier for parameter declaration" : "Expected identifier for keyword argument"
            );
          }
          const value = parseExpression();
          argument = new KeywordArgumentExpression(argument, value);
        }
        if (mode === "parameters") {
          if (!(argument instanceof Identifier || argument instanceof KeywordArgumentExpression)) {
            throw new SyntaxError("Expected identifier for parameter declaration");
          }
          const parameterName = argument instanceof Identifier ? argument.value : argument.key.value;
          if (parameterNames.has(parameterName)) {
            throw new SyntaxError(`Duplicate parameter name: ${parameterName}`);
          }
          parameterNames.add(parameterName);
        }
        if (argument instanceof KeywordArgumentExpression) {
          sawKeywordOrDefault = true;
        } else if (sawKeywordOrDefault) {
          throw new SyntaxError(
            mode === "call" ? "Positional arguments must come before keyword arguments" : "Non-default argument follows default argument"
          );
        } else if (sawSpreadArgument) {
          throw new SyntaxError("Positional arguments must not follow `*` argument unpacking");
        }
      }
      args.push(argument);
      if (is(TOKEN_TYPES.Comma)) {
        ++current;
      }
    }
    return args;
  }
  function parseMemberExpressionArgumentsList() {
    const slices = [];
    let isSlice = false;
    while (!is(TOKEN_TYPES.CloseSquareBracket)) {
      if (is(TOKEN_TYPES.Colon)) {
        slices.push(void 0);
        ++current;
        isSlice = true;
      } else {
        slices.push(parseExpression());
        if (is(TOKEN_TYPES.Colon)) {
          ++current;
          isSlice = true;
        }
      }
    }
    if (slices.length === 0) {
      throw new SyntaxError(`Expected at least one argument for member/slice expression`);
    }
    if (isSlice) {
      if (slices.length > 3) {
        throw new SyntaxError(`Expected 0-3 arguments for slice expression`);
      }
      return new SliceExpression(...slices);
    }
    return slices[0];
  }
  function parseMemberExpression(object) {
    while (is(TOKEN_TYPES.Dot) || is(TOKEN_TYPES.OpenSquareBracket)) {
      const operator = tokens[current];
      ++current;
      let property;
      const computed = operator.type === TOKEN_TYPES.OpenSquareBracket;
      if (computed) {
        property = parseMemberExpressionArgumentsList();
        expect(TOKEN_TYPES.CloseSquareBracket, "Expected closing square bracket");
      } else {
        property = parsePrimaryExpression();
        if (property.type !== "Identifier" && property.type !== "IntegerLiteral") {
          throw new SyntaxError(`Expected identifier or integer following dot operator`);
        }
      }
      object = new MemberExpression(object, property, computed);
    }
    return object;
  }
  function parseMultiplicativeExpression() {
    let left = parsePowerExpression();
    while (is(TOKEN_TYPES.MultiplicativeBinaryOperator)) {
      const operator = tokens[current++];
      const right = parsePowerExpression();
      left = new BinaryExpression(operator, left, right);
    }
    return left;
  }
  function parsePowerExpression() {
    let left = parseTestExpression();
    while (is(TOKEN_TYPES.ExponentiationBinaryOperator)) {
      const operator = tokens[current++];
      const right = parseTestExpression();
      left = new BinaryExpression(operator, left, right);
    }
    return left;
  }
  function parseTestExpression() {
    let operand = parseFilterExpression();
    while (isIdentifier("is")) {
      ++current;
      const negate = isIdentifier("not");
      if (negate) {
        ++current;
      }
      const filter = parsePrimaryExpression();
      if (!(filter instanceof Identifier)) {
        throw new SyntaxError(`Expected identifier for the test`);
      }
      operand = new TestExpression(operand, negate, filter);
    }
    return operand;
  }
  function parseFilterExpression() {
    let operand = parseUnarySignExpression();
    while (is(TOKEN_TYPES.Pipe)) {
      ++current;
      let filter = parsePrimaryExpression();
      if (!(filter instanceof Identifier)) {
        throw new SyntaxError(`Expected identifier for the filter`);
      }
      if (is(TOKEN_TYPES.OpenParen)) {
        filter = parseCallExpression(filter);
      }
      operand = new FilterExpression(operand, filter);
    }
    return operand;
  }
  function parseUnarySignExpression() {
    const token = tokens[current];
    if (token && (token.type === TOKEN_TYPES.UnaryOperator || token.type === TOKEN_TYPES.AdditiveBinaryOperator) && (token.value === "-" || token.value === "+")) {
      const operator = tokens[current++];
      const argument = parseUnarySignExpression();
      return new UnaryExpression(operator, argument);
    }
    return parseCallMemberExpression();
  }
  function parsePrimaryExpression() {
    const token = tokens[current++];
    switch (token.type) {
      case TOKEN_TYPES.NumericLiteral: {
        const num7 = token.value;
        return num7.includes(".") ? new FloatLiteral(Number(num7)) : new IntegerLiteral(Number(num7));
      }
      case TOKEN_TYPES.StringLiteral: {
        let value = token.value;
        while (is(TOKEN_TYPES.StringLiteral)) {
          value += tokens[current++].value;
        }
        return new StringLiteral(value);
      }
      case TOKEN_TYPES.Identifier:
        return new Identifier(token.value);
      case TOKEN_TYPES.OpenParen: {
        const expression = parseExpressionSequence();
        expect(TOKEN_TYPES.CloseParen, "Expected closing parenthesis, got ${tokens[current].type} instead.");
        return expression;
      }
      case TOKEN_TYPES.OpenSquareBracket: {
        const values = [];
        while (!is(TOKEN_TYPES.CloseSquareBracket)) {
          values.push(parseExpression());
          if (is(TOKEN_TYPES.Comma)) {
            ++current;
          }
        }
        ++current;
        return new ArrayLiteral(values);
      }
      case TOKEN_TYPES.OpenCurlyBracket: {
        const values = /* @__PURE__ */ new Map();
        while (!is(TOKEN_TYPES.CloseCurlyBracket)) {
          const key = parseExpression();
          expect(TOKEN_TYPES.Colon, "Expected colon between key and value in object literal");
          const value = parseExpression();
          values.set(key, value);
          if (is(TOKEN_TYPES.Comma)) {
            ++current;
          }
        }
        ++current;
        return new ObjectLiteral(values);
      }
      default:
        throw new SyntaxError(`Unexpected token: ${token.type}`);
    }
  }
  while (current < tokens.length) {
    program.body.push(parseAny());
  }
  return program;
}
function range(start, stop, step = 1) {
  if (stop === void 0) {
    stop = start;
    start = 0;
  }
  if (step === 0) {
    throw new Error("range() step must not be zero");
  }
  const result = [];
  if (step > 0) {
    for (let i = start; i < stop; i += step) {
      result.push(i);
    }
  } else {
    for (let i = start; i > stop; i += step) {
      result.push(i);
    }
  }
  return result;
}
function slice(array, start, stop, step = 1) {
  const direction = Math.sign(step);
  if (direction >= 0) {
    start = (start ??= 0) < 0 ? Math.max(array.length + start, 0) : Math.min(start, array.length);
    stop = (stop ??= array.length) < 0 ? Math.max(array.length + stop, 0) : Math.min(stop, array.length);
  } else {
    start = (start ??= array.length - 1) < 0 ? Math.max(array.length + start, -1) : Math.min(start, array.length - 1);
    stop = (stop ??= -1) < -1 ? Math.max(array.length + stop, -1) : Math.min(stop, array.length - 1);
  }
  const result = [];
  for (let i = start; direction * i < direction * stop; i += step) {
    result.push(array[i]);
  }
  return result;
}
function titleCase(value) {
  return value.replace(/\b\w/g, (c) => c.toUpperCase());
}
function strftime_now(format2) {
  return strftime(/* @__PURE__ */ new Date(), format2);
}
function strftime(date, format2) {
  const monthFormatterLong = new Intl.DateTimeFormat(void 0, { month: "long" });
  const monthFormatterShort = new Intl.DateTimeFormat(void 0, { month: "short" });
  const pad2 = (n2) => n2 < 10 ? "0" + n2 : n2.toString();
  return format2.replace(/%[YmdbBHM%]/g, (token) => {
    switch (token) {
      case "%Y":
        return date.getFullYear().toString();
      case "%m":
        return pad2(date.getMonth() + 1);
      case "%d":
        return pad2(date.getDate());
      case "%b":
        return monthFormatterShort.format(date);
      case "%B":
        return monthFormatterLong.format(date);
      case "%H":
        return pad2(date.getHours());
      case "%M":
        return pad2(date.getMinutes());
      case "%%":
        return "%";
      default:
        return token;
    }
  });
}
function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function replace(str2, oldvalue, newvalue, count) {
  if (count === 0) {
    return str2;
  }
  let remaining = count == null || count < 0 ? Infinity : count;
  const pattern = oldvalue.length === 0 ? new RegExp("(?=)", "gu") : new RegExp(escapeRegExp(oldvalue), "gu");
  return str2.replaceAll(pattern, (match) => {
    if (remaining > 0) {
      --remaining;
      return newvalue;
    }
    return match;
  });
}
var BreakControl = class extends Error {
};
var ContinueControl = class extends Error {
};
var EMPTY_BUILTINS = /* @__PURE__ */ new Map();
var RuntimeValue = class {
  type = "RuntimeValue";
  value;
  /**
   * A collection of built-in functions for this type.
   */
  get builtins() {
    return EMPTY_BUILTINS;
  }
  /**
   * Creates a new RuntimeValue.
   */
  constructor(value = void 0) {
    this.value = value;
  }
  /**
   * Determines truthiness or falsiness of the runtime value.
   * This function should be overridden by subclasses if it has custom truthiness criteria.
   * @returns {BooleanValue} BooleanValue(true) if the value is truthy, BooleanValue(false) otherwise.
   */
  __bool__() {
    return new BooleanValue(!!this.value);
  }
  toString() {
    return String(this.value);
  }
};
var IntegerValue = class extends RuntimeValue {
  type = "IntegerValue";
};
var FloatValue = class extends RuntimeValue {
  type = "FloatValue";
  toString() {
    if (Object.is(this.value, -0)) {
      return "-0.0";
    }
    return this.value % 1 === 0 ? this.value.toFixed(1) : this.value.toString();
  }
};
var StringValue = class extends RuntimeValue {
  type = "StringValue";
  _builtins;
  get builtins() {
    return this._builtins ??= /* @__PURE__ */ new Map([
      [
        "upper",
        new FunctionValue(() => {
          return new StringValue(this.value.toUpperCase());
        })
      ],
      [
        "lower",
        new FunctionValue(() => {
          return new StringValue(this.value.toLowerCase());
        })
      ],
      [
        "strip",
        new FunctionValue(() => {
          return new StringValue(this.value.trim());
        })
      ],
      [
        "title",
        new FunctionValue(() => {
          return new StringValue(titleCase(this.value));
        })
      ],
      [
        "capitalize",
        new FunctionValue(() => {
          return new StringValue(this.value.charAt(0).toUpperCase() + this.value.slice(1));
        })
      ],
      ["length", new IntegerValue(this.value.length)],
      [
        "rstrip",
        new FunctionValue(() => {
          return new StringValue(this.value.trimEnd());
        })
      ],
      [
        "lstrip",
        new FunctionValue(() => {
          return new StringValue(this.value.trimStart());
        })
      ],
      [
        "startswith",
        new FunctionValue((args) => {
          if (args.length === 0) {
            throw new Error("startswith() requires at least one argument");
          }
          const pattern = args[0];
          if (pattern instanceof StringValue) {
            return new BooleanValue(this.value.startsWith(pattern.value));
          } else if (pattern instanceof ArrayValue) {
            for (const item of pattern.value) {
              if (!(item instanceof StringValue)) {
                throw new Error("startswith() tuple elements must be strings");
              }
              if (this.value.startsWith(item.value)) {
                return new BooleanValue(true);
              }
            }
            return new BooleanValue(false);
          }
          throw new Error("startswith() argument must be a string or tuple of strings");
        })
      ],
      [
        "endswith",
        new FunctionValue((args) => {
          if (args.length === 0) {
            throw new Error("endswith() requires at least one argument");
          }
          const pattern = args[0];
          if (pattern instanceof StringValue) {
            return new BooleanValue(this.value.endsWith(pattern.value));
          } else if (pattern instanceof ArrayValue) {
            for (const item of pattern.value) {
              if (!(item instanceof StringValue)) {
                throw new Error("endswith() tuple elements must be strings");
              }
              if (this.value.endsWith(item.value)) {
                return new BooleanValue(true);
              }
            }
            return new BooleanValue(false);
          }
          throw new Error("endswith() argument must be a string or tuple of strings");
        })
      ],
      [
        "split",
        // follows Python's `str.split(sep=None, maxsplit=-1)` function behavior
        // https://docs.python.org/3.13/library/stdtypes.html#str.split
        new FunctionValue((args) => {
          const sep = args[0] ?? new NullValue();
          if (!(sep instanceof StringValue || sep instanceof NullValue)) {
            throw new Error("sep argument must be a string or null");
          }
          const maxsplit = args[1] ?? new IntegerValue(-1);
          if (!(maxsplit instanceof IntegerValue)) {
            throw new Error("maxsplit argument must be a number");
          }
          let result = [];
          if (sep instanceof NullValue) {
            const text = this.value.trimStart();
            for (const { 0: match, index } of text.matchAll(/\S+/g)) {
              if (maxsplit.value !== -1 && result.length >= maxsplit.value && index !== void 0) {
                result.push(match + text.slice(index + match.length));
                break;
              }
              result.push(match);
            }
          } else {
            if (sep.value === "") {
              throw new Error("empty separator");
            }
            result = this.value.split(sep.value);
            if (maxsplit.value !== -1 && result.length > maxsplit.value) {
              result.push(result.splice(maxsplit.value).join(sep.value));
            }
          }
          return new ArrayValue(result.map((part) => new StringValue(part)));
        })
      ],
      [
        "replace",
        new FunctionValue((args) => {
          if (args.length < 2) {
            throw new Error("replace() requires at least two arguments");
          }
          const oldValue = args[0];
          const newValue = args[1];
          if (!(oldValue instanceof StringValue && newValue instanceof StringValue)) {
            throw new Error("replace() arguments must be strings");
          }
          let count;
          if (args.length > 2) {
            if (args[2].type === "KeywordArgumentsValue") {
              count = args[2].value.get("count") ?? new NullValue();
            } else {
              count = args[2];
            }
          } else {
            count = new NullValue();
          }
          if (!(count instanceof IntegerValue || count instanceof NullValue)) {
            throw new Error("replace() count argument must be a number or null");
          }
          return new StringValue(replace(this.value, oldValue.value, newValue.value, count.value));
        })
      ]
    ]);
  }
};
var BooleanValue = class extends RuntimeValue {
  type = "BooleanValue";
};
var NON_ASCII_CHARS = /[\x7f-\uffff]/g;
function makeAsciiSafe(str2) {
  return str2.replace(NON_ASCII_CHARS, (char) => "\\u" + char.charCodeAt(0).toString(16).padStart(4, "0"));
}
function toJSON(input, options = {}, depth = 0, convertUndefinedToNull = true) {
  const { indent = null, ensureAscii = false, separators = null, sortKeys = false } = options;
  let itemSeparator;
  let keySeparator;
  if (separators) {
    [itemSeparator, keySeparator] = separators;
  } else if (indent) {
    itemSeparator = ",";
    keySeparator = ": ";
  } else {
    itemSeparator = ", ";
    keySeparator = ": ";
  }
  switch (input.type) {
    case "NullValue":
      return "null";
    case "UndefinedValue":
      return convertUndefinedToNull ? "null" : "undefined";
    case "IntegerValue":
    case "FloatValue":
    case "BooleanValue":
      return JSON.stringify(input.value);
    case "StringValue": {
      let result = JSON.stringify(input.value);
      if (ensureAscii) {
        result = makeAsciiSafe(result);
      }
      return result;
    }
    case "ArrayValue":
    case "NamespaceValue":
    case "ObjectValue": {
      const indentValue = indent ? " ".repeat(indent) : "";
      const basePadding = "\n" + indentValue.repeat(depth);
      const childrenPadding = basePadding + indentValue;
      if (input.type === "ArrayValue") {
        const core = input.value.map((x) => toJSON(x, options, depth + 1, convertUndefinedToNull));
        return indent ? `[${childrenPadding}${core.join(`${itemSeparator}${childrenPadding}`)}${basePadding}]` : `[${core.join(itemSeparator)}]`;
      } else {
        let entries = Array.from(input.value.entries());
        if (sortKeys) {
          entries = entries.sort(([a], [b]) => a.localeCompare(b));
        }
        const core = entries.map(([key, value]) => {
          let keyStr = JSON.stringify(key);
          if (ensureAscii) {
            keyStr = makeAsciiSafe(keyStr);
          }
          const v = `${keyStr}${keySeparator}${toJSON(value, options, depth + 1, convertUndefinedToNull)}`;
          return indent ? `${childrenPadding}${v}` : v;
        });
        return indent ? `{${core.join(itemSeparator)}${basePadding}}` : `{${core.join(itemSeparator)}}`;
      }
    }
    default:
      throw new Error(`Cannot convert to JSON: ${input.type}`);
  }
}
var ObjectValue = class extends RuntimeValue {
  type = "ObjectValue";
  _builtins;
  /**
   * NOTE: necessary to override since all JavaScript arrays are considered truthy,
   * while only non-empty Python arrays are consider truthy.
   *
   * e.g.,
   *  - JavaScript:  {} && 5 -> 5
   *  - Python:      {} and 5 -> {}
   */
  __bool__() {
    return new BooleanValue(this.value.size > 0);
  }
  get builtins() {
    return this._builtins ??= /* @__PURE__ */ new Map([
      [
        "get",
        new FunctionValue(([key, defaultValue]) => {
          if (!(key instanceof StringValue)) {
            throw new Error(`Object key must be a string: got ${key.type}`);
          }
          return this.value.get(key.value) ?? defaultValue ?? new NullValue();
        })
      ],
      ["items", new FunctionValue(() => this.items())],
      ["keys", new FunctionValue(() => this.keys())],
      ["values", new FunctionValue(() => this.values())],
      [
        "dictsort",
        new FunctionValue((args) => {
          let kwargs = /* @__PURE__ */ new Map();
          const positionalArgs = args.filter((arg) => {
            if (arg instanceof KeywordArgumentsValue) {
              kwargs = arg.value;
              return false;
            }
            return true;
          });
          const caseSensitive = positionalArgs.at(0) ?? kwargs.get("case_sensitive") ?? new BooleanValue(false);
          if (!(caseSensitive instanceof BooleanValue)) {
            throw new Error("case_sensitive must be a boolean");
          }
          const by = positionalArgs.at(1) ?? kwargs.get("by") ?? new StringValue("key");
          if (!(by instanceof StringValue)) {
            throw new Error("by must be a string");
          }
          if (!["key", "value"].includes(by.value)) {
            throw new Error("by must be either 'key' or 'value'");
          }
          const reverse = positionalArgs.at(2) ?? kwargs.get("reverse") ?? new BooleanValue(false);
          if (!(reverse instanceof BooleanValue)) {
            throw new Error("reverse must be a boolean");
          }
          const items = Array.from(this.value.entries()).map(([key, value]) => new ArrayValue([new StringValue(key), value])).sort((a, b) => {
            const index = by.value === "key" ? 0 : 1;
            const aVal = a.value[index];
            const bVal = b.value[index];
            const result = compareRuntimeValues(aVal, bVal, caseSensitive.value);
            return reverse.value ? -result : result;
          });
          return new ArrayValue(items);
        })
      ]
    ]);
  }
  items() {
    return new ArrayValue(
      Array.from(this.value.entries()).map(([key, value]) => new ArrayValue([new StringValue(key), value]))
    );
  }
  keys() {
    return new ArrayValue(Array.from(this.value.keys()).map((key) => new StringValue(key)));
  }
  values() {
    return new ArrayValue(Array.from(this.value.values()));
  }
  toString() {
    return toJSON(this, {}, 0, false);
  }
};
var KeywordArgumentsValue = class extends ObjectValue {
  type = "KeywordArgumentsValue";
};
var NamespaceValue = class extends RuntimeValue {
  type = "NamespaceValue";
  toString() {
    return toJSON(this, {}, 0, false);
  }
};
var ArrayValue = class extends RuntimeValue {
  type = "ArrayValue";
  _builtins;
  get builtins() {
    return this._builtins ??= /* @__PURE__ */ new Map([["length", new IntegerValue(this.value.length)]]);
  }
  /**
   * NOTE: necessary to override since all JavaScript arrays are considered truthy,
   * while only non-empty Python arrays are consider truthy.
   *
   * e.g.,
   *  - JavaScript:  [] && 5 -> 5
   *  - Python:      [] and 5 -> []
   */
  __bool__() {
    return new BooleanValue(this.value.length > 0);
  }
  toString() {
    return toJSON(this, {}, 0, false);
  }
};
var TupleValue = class extends ArrayValue {
  type = "TupleValue";
};
var FunctionValue = class extends RuntimeValue {
  type = "FunctionValue";
};
var NullValue = class extends RuntimeValue {
  type = "NullValue";
};
var UndefinedValue = class extends RuntimeValue {
  type = "UndefinedValue";
};
function normalizeNamespaceEntry(pair) {
  let values;
  if (pair instanceof ArrayValue) {
    values = pair.value;
  } else if (pair instanceof StringValue) {
    values = Array.from(pair.value, (char) => new StringValue(char));
  }
  if (!values || values.length !== 2) {
    throw new Error("namespace expected an object or an iterable of [key, value] pairs");
  }
  const [key, value] = values;
  if (!(key instanceof StringValue)) {
    throw new Error("namespace keys must be strings");
  }
  return [key, value];
}
var _Environment = class {
  constructor(parent) {
    this.parent = parent;
  }
  /**
   * The variables declared in this environment.
   */
  variables = /* @__PURE__ */ new Map([
    [
      "namespace",
      new FunctionValue((args) => {
        const positional = args.slice();
        let kwargs;
        if (positional.at(-1) instanceof KeywordArgumentsValue) {
          kwargs = positional.pop();
        }
        if (positional.length > 1) {
          throw new Error(`namespace expected at most 1 argument, got ${positional.length}`);
        }
        const entries = /* @__PURE__ */ new Map();
        if (positional.length === 1) {
          const source = positional[0];
          if (source instanceof ObjectValue) {
            for (const [key, value] of source.value) {
              entries.set(key, value);
            }
          } else if (source instanceof ArrayValue) {
            for (const pair of source.value) {
              const [key, value] = normalizeNamespaceEntry(pair);
              entries.set(key.value, value);
            }
          } else {
            throw new Error(`'${source.type}' object is not iterable`);
          }
        }
        if (kwargs) {
          for (const [key, value] of kwargs.value) {
            entries.set(key, value);
          }
        }
        return new NamespaceValue(entries);
      })
    ]
  ]);
  tests = _Environment.TESTS;
  /**
   * Set the value of a variable in the current environment.
   */
  set(name, value) {
    return this.declareVariable(name, convertToRuntimeValues(value));
  }
  declareVariable(name, value) {
    if (this.variables.has(name)) {
      throw new SyntaxError(`Variable already declared: ${name}`);
    }
    this.variables.set(name, value);
    return value;
  }
  // private assignVariable(name: string, value: AnyRuntimeValue): AnyRuntimeValue {
  // 	const env = this.resolve(name);
  // 	env.variables.set(name, value);
  // 	return value;
  // }
  /**
   * Set variable in the current scope.
   * See https://jinja.palletsprojects.com/en/3.0.x/templates/#assignments for more information.
   */
  setVariable(name, value) {
    this.variables.set(name, value);
    return value;
  }
  /**
   * Resolve the environment in which the variable is declared.
   * @param {string} name The name of the variable.
   * @returns {Environment} The environment in which the variable is declared.
   */
  resolve(name) {
    if (this.variables.has(name)) {
      return this;
    }
    if (this.parent) {
      return this.parent.resolve(name);
    }
    throw new Error(`Unknown variable: ${name}`);
  }
  lookupVariable(name) {
    try {
      return this.resolve(name).variables.get(name) ?? new UndefinedValue();
    } catch {
      return new UndefinedValue();
    }
  }
};
var Environment = _Environment;
__publicField(Environment, "TESTS", /* @__PURE__ */ new Map([
  ["boolean", (operand) => operand.type === "BooleanValue"],
  ["callable", (operand) => operand instanceof FunctionValue],
  [
    "odd",
    (operand) => {
      if (!(operand instanceof IntegerValue)) {
        throw new Error(`cannot odd on ${operand.type}`);
      }
      return operand.value % 2 !== 0;
    }
  ],
  [
    "even",
    (operand) => {
      if (!(operand instanceof IntegerValue)) {
        throw new Error(`cannot even on ${operand.type}`);
      }
      return operand.value % 2 === 0;
    }
  ],
  ["false", (operand) => operand.type === "BooleanValue" && !operand.value],
  ["true", (operand) => operand.type === "BooleanValue" && operand.value],
  ["none", (operand) => operand.type === "NullValue"],
  ["string", (operand) => operand.type === "StringValue"],
  ["number", (operand) => operand instanceof IntegerValue || operand instanceof FloatValue],
  ["integer", (operand) => operand instanceof IntegerValue],
  ["iterable", (operand) => operand.type === "ArrayValue" || operand.type === "StringValue"],
  ["mapping", (operand) => operand instanceof ObjectValue],
  [
    "sequence",
    (operand) => operand instanceof ArrayValue || operand instanceof ObjectValue || operand instanceof StringValue
  ],
  [
    "lower",
    (operand) => {
      const str2 = operand.value;
      return operand.type === "StringValue" && str2 === str2.toLowerCase();
    }
  ],
  [
    "upper",
    (operand) => {
      const str2 = operand.value;
      return operand.type === "StringValue" && str2 === str2.toUpperCase();
    }
  ],
  ["none", (operand) => operand.type === "NullValue"],
  ["defined", (operand) => operand.type !== "UndefinedValue"],
  ["undefined", (operand) => operand.type === "UndefinedValue"],
  ["equalto", (a, b) => a.value === b.value],
  ["eq", (a, b) => a.value === b.value]
]));
function setupGlobals(env) {
  env.set("false", false);
  env.set("true", true);
  env.set("none", null);
  env.set("raise_exception", (args) => {
    throw new Error(args);
  });
  env.set("range", range);
  env.set("strftime_now", strftime_now);
  env.set("True", true);
  env.set("False", false);
  env.set("None", null);
}
function isAttributeContainer(value) {
  return value instanceof ObjectValue || value instanceof NamespaceValue;
}
function isNumericLikeValue(value) {
  return value instanceof IntegerValue || value instanceof FloatValue || value instanceof BooleanValue;
}
function getNumericValue(value) {
  return value instanceof BooleanValue ? Number(value.value) : value.value;
}
function getAttributeValue(item, attributePath) {
  const parts = attributePath.split(".");
  let value = item;
  for (const part of parts) {
    if (isAttributeContainer(value)) {
      value = value.value.get(part) ?? new UndefinedValue();
    } else if (value instanceof ArrayValue) {
      const index = parseInt(part, 10);
      if (!isNaN(index) && index >= 0 && index < value.value.length) {
        value = value.value[index];
      } else {
        return new UndefinedValue();
      }
    } else {
      return new UndefinedValue();
    }
  }
  return value;
}
function compareRuntimeValues(a, b, caseSensitive = false) {
  if (a instanceof NullValue && b instanceof NullValue) {
    return 0;
  }
  if (a instanceof NullValue || b instanceof NullValue) {
    throw new Error(`Cannot compare ${a.type} with ${b.type}`);
  }
  if (a instanceof UndefinedValue && b instanceof UndefinedValue) {
    return 0;
  }
  if (a instanceof UndefinedValue || b instanceof UndefinedValue) {
    throw new Error(`Cannot compare ${a.type} with ${b.type}`);
  }
  if (isNumericLikeValue(a) && isNumericLikeValue(b)) {
    const aNum = getNumericValue(a);
    const bNum = getNumericValue(b);
    return aNum < bNum ? -1 : aNum > bNum ? 1 : 0;
  }
  if (a.type !== b.type) {
    throw new Error(`Cannot compare different types: ${a.type} and ${b.type}`);
  }
  switch (a.type) {
    case "StringValue": {
      let aStr = a.value;
      let bStr = b.value;
      if (!caseSensitive) {
        aStr = aStr.toLowerCase();
        bStr = bStr.toLowerCase();
      }
      return aStr < bStr ? -1 : aStr > bStr ? 1 : 0;
    }
    default:
      throw new Error(`Cannot compare type: ${a.type}`);
  }
}
function getParameterDescriptor(argument) {
  if (argument.type === "Identifier") {
    return { name: argument.value };
  }
  const keywordArgument = argument;
  return { name: keywordArgument.key.value, defaultValue: keywordArgument.value };
}
function isSpecialMacroArgument(name) {
  return name === "kwargs" || name === "varargs";
}
function findAccessedSpecialMacroArguments(parameters, body) {
  const declared = /* @__PURE__ */ new Set();
  const accessed = /* @__PURE__ */ new Set();
  const declareName = (name) => {
    if (isSpecialMacroArgument(name)) {
      declared.add(name);
    }
  };
  const declareTarget = (target) => {
    if (target.type === "Identifier") {
      declareName(target.value);
    } else if (target.type === "TupleLiteral") {
      for (const item of target.value) {
        declareTarget(item);
      }
    }
  };
  const declareArgument = (argument) => {
    declareName(getParameterDescriptor(argument).name);
  };
  const visitArgumentDefaults = (args) => {
    for (const argument of args) {
      const parameter = getParameterDescriptor(argument);
      if (parameter.defaultValue) {
        visit(parameter.defaultValue);
      }
    }
  };
  const visitFilter = (filter) => {
    if (filter.type === "CallExpression") {
      visit(filter.args);
    }
  };
  const visit = (node) => {
    if (Array.isArray(node)) {
      for (const item of node) {
        visit(item);
      }
      return;
    }
    if (node instanceof Map) {
      for (const [key, value] of node) {
        visit(key);
        visit(value);
      }
      return;
    }
    if (!(node instanceof Statement)) {
      return;
    }
    switch (node.type) {
      case "For": {
        const statement = node;
        declareTarget(statement.loopvar);
        if (statement.iterable.type === "SelectExpression") {
          const iterable = statement.iterable;
          visit(iterable.lhs);
          visit(statement.body);
          visit(statement.defaultBlock);
          visit(iterable.test);
          return;
        }
        visit(statement.iterable);
        visit(statement.body);
        visit(statement.defaultBlock);
        return;
      }
      case "Set": {
        const statement = node;
        declareTarget(statement.assignee);
        visit(statement.value);
        visit(statement.body);
        return;
      }
      case "Macro": {
        const statement = node;
        for (const argument of statement.args) {
          declareArgument(argument);
        }
        visitArgumentDefaults(statement.args);
        visit(statement.body);
        return;
      }
      case "CallStatement": {
        const statement = node;
        visit(statement.call);
        for (const argument of statement.callerArgs ?? []) {
          declareArgument(argument);
        }
        visitArgumentDefaults(statement.callerArgs ?? []);
        visit(statement.body);
        return;
      }
      case "FilterStatement": {
        const statement = node;
        visit(statement.body);
        visitFilter(statement.filter);
        return;
      }
      case "Identifier": {
        const name = node.value;
        if (isSpecialMacroArgument(name) && !declared.has(name)) {
          accessed.add(name);
        }
        return;
      }
      case "MemberExpression": {
        const expression = node;
        visit(expression.object);
        if (expression.computed) {
          visit(expression.property);
        }
        return;
      }
      case "FilterExpression": {
        const expression = node;
        visit(expression.operand);
        visitFilter(expression.filter);
        return;
      }
      case "TestExpression":
        visit(node.operand);
        return;
      case "SelectExpression": {
        const expression = node;
        visit(expression.test);
        visit(expression.lhs);
        return;
      }
      case "KeywordArgumentExpression":
        visit(node.value);
        return;
      default:
        for (const child of Object.values(node)) {
          visit(child);
        }
    }
  };
  for (const parameter of parameters) {
    declareArgument(parameter);
  }
  visit(body);
  return accessed;
}
var Interpreter = class {
  global;
  constructor(env) {
    this.global = env ?? new Environment();
  }
  /**
   * Run the program.
   */
  run(program) {
    return this.evaluate(program, this.global);
  }
  /**
   * Evaluates expressions following the binary operation type.
   */
  evaluateBinaryExpression(node, environment) {
    const left = this.evaluate(node.left, environment);
    switch (node.operator.value) {
      case "and":
        return left.__bool__().value ? this.evaluate(node.right, environment) : left;
      case "or":
        return left.__bool__().value ? left : this.evaluate(node.right, environment);
    }
    const right = this.evaluate(node.right, environment);
    switch (node.operator.value) {
      case "==":
        return new BooleanValue(left.value == right.value);
      case "!=":
        return new BooleanValue(left.value != right.value);
    }
    if (left instanceof UndefinedValue || right instanceof UndefinedValue) {
      if (right instanceof UndefinedValue && ["in", "not in"].includes(node.operator.value)) {
        return new BooleanValue(node.operator.value === "not in");
      }
      throw new Error(`Cannot perform operation ${node.operator.value} on undefined values`);
    } else if (left instanceof NullValue || right instanceof NullValue) {
      throw new Error("Cannot perform operation on null values");
    } else if (node.operator.value === "~") {
      return new StringValue(left.value.toString() + right.value.toString());
    } else if (node.operator.value === "**" && isNumericLikeValue(left) && isNumericLikeValue(right)) {
      const a = getNumericValue(left);
      const b = getNumericValue(right);
      if (a === 0 && b < 0) {
        throw new Error("0.0 cannot be raised to a negative power");
      }
      const result = a ** b;
      if (!Number.isFinite(result)) {
        throw new Error("Exponentiation result is not a finite real number");
      }
      const isFloat = left instanceof FloatValue || right instanceof FloatValue || b < 0;
      return isFloat ? new FloatValue(result) : new IntegerValue(result);
    } else if ((left instanceof IntegerValue || left instanceof FloatValue) && (right instanceof IntegerValue || right instanceof FloatValue)) {
      const a = left.value, b = right.value;
      switch (node.operator.value) {
        case "+":
        case "-":
        case "*": {
          const res = node.operator.value === "+" ? a + b : node.operator.value === "-" ? a - b : a * b;
          const isFloat = left instanceof FloatValue || right instanceof FloatValue;
          return isFloat ? new FloatValue(res) : new IntegerValue(res);
        }
        case "/":
          return new FloatValue(a / b);
        case "//": {
          const res = Math.floor(a / b);
          const isFloat = left instanceof FloatValue || right instanceof FloatValue;
          return isFloat ? new FloatValue(res) : new IntegerValue(res);
        }
        case "%": {
          const rem = a % b;
          const isFloat = left instanceof FloatValue || right instanceof FloatValue;
          return isFloat ? new FloatValue(rem) : new IntegerValue(rem);
        }
        case "<":
          return new BooleanValue(a < b);
        case ">":
          return new BooleanValue(a > b);
        case ">=":
          return new BooleanValue(a >= b);
        case "<=":
          return new BooleanValue(a <= b);
      }
    } else if (left instanceof ArrayValue && right instanceof ArrayValue) {
      switch (node.operator.value) {
        case "+":
          return new ArrayValue(left.value.concat(right.value));
      }
    } else if (right instanceof ArrayValue) {
      const member = right.value.find((x) => x.value === left.value) !== void 0;
      switch (node.operator.value) {
        case "in":
          return new BooleanValue(member);
        case "not in":
          return new BooleanValue(!member);
      }
    }
    if (left instanceof StringValue || right instanceof StringValue) {
      switch (node.operator.value) {
        case "+":
          return new StringValue(left.value.toString() + right.value.toString());
      }
    }
    if (left instanceof StringValue && right instanceof StringValue) {
      switch (node.operator.value) {
        case "in":
          return new BooleanValue(right.value.includes(left.value));
        case "not in":
          return new BooleanValue(!right.value.includes(left.value));
      }
    }
    if (left instanceof StringValue && right instanceof ObjectValue) {
      switch (node.operator.value) {
        case "in":
          return new BooleanValue(right.value.has(left.value));
        case "not in":
          return new BooleanValue(!right.value.has(left.value));
      }
    }
    throw new SyntaxError(`Unknown operator "${node.operator.value}" between ${left.type} and ${right.type}`);
  }
  evaluateArguments(args, environment) {
    const positionalArguments = [];
    const keywordArguments = /* @__PURE__ */ new Map();
    const addKeywordArgument = (key, value) => {
      if (keywordArguments.has(key)) {
        throw new Error(`Got multiple values for keyword argument '${key}'`);
      }
      keywordArguments.set(key, value);
    };
    for (const argument of args) {
      if (argument.type === "SpreadExpression") {
        const spreadNode = argument;
        const val = this.evaluate(spreadNode.argument, environment);
        if (!(val instanceof ArrayValue)) {
          throw new Error(`Cannot unpack non-iterable type: ${val.type}`);
        }
        for (const item of val.value) {
          positionalArguments.push(item);
        }
      } else if (argument.type !== "KeywordArgumentExpression" && argument.type !== "KeywordSpreadExpression") {
        positionalArguments.push(this.evaluate(argument, environment));
      }
    }
    for (const argument of args) {
      if (argument.type === "KeywordArgumentExpression") {
        const kwarg = argument;
        addKeywordArgument(kwarg.key.value, this.evaluate(kwarg.value, environment));
      } else if (argument.type === "KeywordSpreadExpression") {
        const spreadNode = argument;
        const val = this.evaluate(spreadNode.argument, environment);
        if (!(val instanceof ObjectValue)) {
          throw new Error(`Argument after ** must be a mapping, not ${val.type}`);
        }
        for (const [key, value] of val.value) {
          addKeywordArgument(key, value);
        }
      }
    }
    return [positionalArguments, keywordArguments];
  }
  applyFilter(operand, filterNode, environment) {
    if (filterNode.type === "Identifier") {
      const filter = filterNode;
      if (filter.value === "safe") {
        return operand;
      }
      if (filter.value === "tojson") {
        return new StringValue(toJSON(operand, {}));
      }
      if (operand instanceof ArrayValue) {
        switch (filter.value) {
          case "list":
            return operand;
          case "first":
            return operand.value[0];
          case "last":
            return operand.value[operand.value.length - 1];
          case "length":
            return new IntegerValue(operand.value.length);
          case "reverse":
            return new ArrayValue(operand.value.slice().reverse());
          case "sort": {
            return new ArrayValue(operand.value.slice().sort((a, b) => compareRuntimeValues(a, b, false)));
          }
          case "join":
            return new StringValue(operand.value.map((x) => x.value).join(""));
          case "string":
            return new StringValue(toJSON(operand, {}, 0, false));
          case "unique": {
            const seen = /* @__PURE__ */ new Set();
            const output = [];
            for (const item of operand.value) {
              if (!seen.has(item.value)) {
                seen.add(item.value);
                output.push(item);
              }
            }
            return new ArrayValue(output);
          }
          default:
            throw new Error(`Unknown ArrayValue filter: ${filter.value}`);
        }
      } else if (operand instanceof StringValue) {
        switch (filter.value) {
          case "length":
          case "upper":
          case "lower":
          case "title":
          case "capitalize": {
            const builtin = operand.builtins.get(filter.value);
            if (builtin instanceof FunctionValue) {
              return builtin.value(
                /* no arguments */
                [],
                environment
              );
            } else if (builtin instanceof IntegerValue) {
              return builtin;
            } else {
              throw new Error(`Unknown StringValue filter: ${filter.value}`);
            }
          }
          case "trim":
            return new StringValue(operand.value.trim());
          case "indent":
            return new StringValue(
              operand.value.split("\n").map(
                (x, i) => (
                  // By default, don't indent the first line or empty lines
                  i === 0 || x.length === 0 ? x : "    " + x
                )
              ).join("\n")
            );
          case "join":
          case "string":
            return operand;
          case "int": {
            const val = parseInt(operand.value, 10);
            return new IntegerValue(isNaN(val) ? 0 : val);
          }
          case "float": {
            const val = parseFloat(operand.value);
            return new FloatValue(isNaN(val) ? 0 : val);
          }
          default:
            throw new Error(`Unknown StringValue filter: ${filter.value}`);
        }
      } else if (operand instanceof IntegerValue || operand instanceof FloatValue) {
        switch (filter.value) {
          case "abs":
            return operand instanceof IntegerValue ? new IntegerValue(Math.abs(operand.value)) : new FloatValue(Math.abs(operand.value));
          case "int":
            return new IntegerValue(Math.floor(operand.value));
          case "float":
            return new FloatValue(operand.value);
          case "string":
            return new StringValue(operand.toString());
          default:
            throw new Error(`Unknown NumericValue filter: ${filter.value}`);
        }
      } else if (operand instanceof ObjectValue) {
        switch (filter.value) {
          case "items":
            return new ArrayValue(
              Array.from(operand.value.entries()).map(([key, value]) => new ArrayValue([new StringValue(key), value]))
            );
          case "length":
            return new IntegerValue(operand.value.size);
          default: {
            const builtin = operand.builtins.get(filter.value);
            if (builtin) {
              if (builtin instanceof FunctionValue) {
                return builtin.value([], environment);
              }
              return builtin;
            }
            throw new Error(`Unknown ObjectValue filter: ${filter.value}`);
          }
        }
      } else if (operand instanceof BooleanValue) {
        switch (filter.value) {
          case "bool":
            return new BooleanValue(operand.value);
          case "int":
            return new IntegerValue(operand.value ? 1 : 0);
          case "float":
            return new FloatValue(operand.value ? 1 : 0);
          case "string":
            return new StringValue(operand.value ? "true" : "false");
          default:
            throw new Error(`Unknown BooleanValue filter: ${filter.value}`);
        }
      }
      throw new Error(`Cannot apply filter "${filter.value}" to type: ${operand.type}`);
    } else if (filterNode.type === "CallExpression") {
      const filter = filterNode;
      if (filter.callee.type !== "Identifier") {
        throw new Error(`Unknown filter: ${filter.callee.type}`);
      }
      const filterName = filter.callee.value;
      if (filterName === "tojson") {
        const [, kwargs] = this.evaluateArguments(filter.args, environment);
        const indent = kwargs.get("indent") ?? new NullValue();
        if (!(indent instanceof IntegerValue || indent instanceof NullValue)) {
          throw new Error("If set, indent must be a number");
        }
        const ensureAscii = kwargs.get("ensure_ascii") ?? new BooleanValue(false);
        if (!(ensureAscii instanceof BooleanValue)) {
          throw new Error("If set, ensure_ascii must be a boolean");
        }
        const sortKeys = kwargs.get("sort_keys") ?? new BooleanValue(false);
        if (!(sortKeys instanceof BooleanValue)) {
          throw new Error("If set, sort_keys must be a boolean");
        }
        const separatorsArg = kwargs.get("separators") ?? new NullValue();
        let separators = null;
        if (separatorsArg instanceof ArrayValue || separatorsArg instanceof TupleValue) {
          if (separatorsArg.value.length !== 2) {
            throw new Error("separators must be a tuple of two strings");
          }
          const [itemSep, keySep] = separatorsArg.value;
          if (!(itemSep instanceof StringValue) || !(keySep instanceof StringValue)) {
            throw new Error("separators must be a tuple of two strings");
          }
          separators = [itemSep.value, keySep.value];
        } else if (!(separatorsArg instanceof NullValue)) {
          throw new Error("If set, separators must be a tuple of two strings");
        }
        return new StringValue(
          toJSON(operand, {
            indent: indent.value,
            ensureAscii: ensureAscii.value,
            sortKeys: sortKeys.value,
            separators
          })
        );
      } else if (filterName === "join") {
        let value;
        if (operand instanceof StringValue) {
          value = Array.from(operand.value);
        } else if (operand instanceof ArrayValue) {
          value = operand.value.map((x) => x.value);
        } else {
          throw new Error(`Cannot apply filter "${filterName}" to type: ${operand.type}`);
        }
        const [args, kwargs] = this.evaluateArguments(filter.args, environment);
        const separator = args.at(0) ?? kwargs.get("separator") ?? new StringValue("");
        if (!(separator instanceof StringValue)) {
          throw new Error("separator must be a string");
        }
        return new StringValue(value.join(separator.value));
      } else if (filterName === "int" || filterName === "float") {
        const [args, kwargs] = this.evaluateArguments(filter.args, environment);
        const defaultValue = args.at(0) ?? kwargs.get("default") ?? (filterName === "int" ? new IntegerValue(0) : new FloatValue(0));
        if (operand instanceof StringValue) {
          const val = filterName === "int" ? parseInt(operand.value, 10) : parseFloat(operand.value);
          return isNaN(val) ? defaultValue : filterName === "int" ? new IntegerValue(val) : new FloatValue(val);
        } else if (operand instanceof IntegerValue || operand instanceof FloatValue) {
          return operand;
        } else if (operand instanceof BooleanValue) {
          return filterName === "int" ? new IntegerValue(operand.value ? 1 : 0) : new FloatValue(operand.value ? 1 : 0);
        } else {
          throw new Error(`Cannot apply filter "${filterName}" to type: ${operand.type}`);
        }
      } else if (filterName === "default") {
        const [args, kwargs] = this.evaluateArguments(filter.args, environment);
        const defaultValue = args[0] ?? new StringValue("");
        const booleanValue = args[1] ?? kwargs.get("boolean") ?? new BooleanValue(false);
        if (!(booleanValue instanceof BooleanValue)) {
          throw new Error("`default` filter flag must be a boolean");
        }
        if (operand instanceof UndefinedValue || booleanValue.value && !operand.__bool__().value) {
          return defaultValue;
        }
        return operand;
      }
      if (operand instanceof ArrayValue) {
        switch (filterName) {
          case "sort": {
            const [args, kwargs] = this.evaluateArguments(filter.args, environment);
            const reverse = args.at(0) ?? kwargs.get("reverse") ?? new BooleanValue(false);
            if (!(reverse instanceof BooleanValue)) {
              throw new Error("reverse must be a boolean");
            }
            const caseSensitive = args.at(1) ?? kwargs.get("case_sensitive") ?? new BooleanValue(false);
            if (!(caseSensitive instanceof BooleanValue)) {
              throw new Error("case_sensitive must be a boolean");
            }
            const attribute = args.at(2) ?? kwargs.get("attribute") ?? new NullValue();
            if (!(attribute instanceof StringValue || attribute instanceof IntegerValue || attribute instanceof NullValue)) {
              throw new Error("attribute must be a string, integer, or null");
            }
            const getSortValue = (item) => {
              if (attribute instanceof NullValue) {
                return item;
              }
              const attrPath = attribute instanceof IntegerValue ? String(attribute.value) : attribute.value;
              return getAttributeValue(item, attrPath);
            };
            return new ArrayValue(
              operand.value.slice().sort((a, b) => {
                const aVal = getSortValue(a);
                const bVal = getSortValue(b);
                const result = compareRuntimeValues(aVal, bVal, caseSensitive.value);
                return reverse.value ? -result : result;
              })
            );
          }
          case "selectattr":
          case "rejectattr": {
            const select = filterName === "selectattr";
            if (operand.value.some((item) => !isAttributeContainer(item))) {
              throw new Error(`\`${filterName}\` can only be applied to array of objects`);
            }
            if (filter.args.some((x) => x.type !== "StringLiteral")) {
              throw new Error(`arguments of \`${filterName}\` must be strings`);
            }
            const [attr, testName, value] = filter.args.map((x) => this.evaluate(x, environment));
            let testFunction;
            if (testName) {
              const test = environment.tests.get(testName.value);
              if (!test) {
                throw new Error(`Unknown test: ${testName.value}`);
              }
              testFunction = test;
            } else {
              testFunction = (...x) => x[0].__bool__().value;
            }
            const filtered = operand.value.filter((item) => {
              const a = item.value.get(attr.value);
              const result = a ? testFunction(a, value) : false;
              return select ? result : !result;
            });
            return new ArrayValue(filtered);
          }
          case "map": {
            const [, kwargs] = this.evaluateArguments(filter.args, environment);
            if (kwargs.has("attribute")) {
              const attr = kwargs.get("attribute");
              if (!(attr instanceof StringValue)) {
                throw new Error("attribute must be a string");
              }
              const defaultValue = kwargs.get("default");
              const mapped = operand.value.map((item) => {
                if (!isAttributeContainer(item)) {
                  throw new Error("items in map must be an object");
                }
                const value = getAttributeValue(item, attr.value);
                return value instanceof UndefinedValue ? defaultValue ?? new UndefinedValue() : value;
              });
              return new ArrayValue(mapped);
            } else {
              throw new Error("`map` expressions without `attribute` set are not currently supported.");
            }
          }
        }
        throw new Error(`Unknown ArrayValue filter: ${filterName}`);
      } else if (operand instanceof StringValue) {
        switch (filterName) {
          case "indent": {
            const [args, kwargs] = this.evaluateArguments(filter.args, environment);
            const width = args.at(0) ?? kwargs.get("width") ?? new IntegerValue(4);
            if (!(width instanceof IntegerValue)) {
              throw new Error("width must be a number");
            }
            const first = args.at(1) ?? kwargs.get("first") ?? new BooleanValue(false);
            const blank = args.at(2) ?? kwargs.get("blank") ?? new BooleanValue(false);
            const lines = operand.value.split("\n");
            const indent = " ".repeat(width.value);
            const indented = lines.map(
              (x, i) => !first.value && i === 0 || !blank.value && x.length === 0 ? x : indent + x
            );
            return new StringValue(indented.join("\n"));
          }
          case "replace": {
            const replaceFn = operand.builtins.get("replace");
            if (!(replaceFn instanceof FunctionValue)) {
              throw new Error("replace filter not available");
            }
            const [args, kwargs] = this.evaluateArguments(filter.args, environment);
            return replaceFn.value([...args, new KeywordArgumentsValue(kwargs)], environment);
          }
        }
        throw new Error(`Unknown StringValue filter: ${filterName}`);
      } else if (operand instanceof ObjectValue) {
        const builtin = operand.builtins.get(filterName);
        if (builtin && builtin instanceof FunctionValue) {
          const [args, kwargs] = this.evaluateArguments(filter.args, environment);
          if (kwargs.size > 0) {
            args.push(new KeywordArgumentsValue(kwargs));
          }
          return builtin.value(args, environment);
        }
        throw new Error(`Unknown ObjectValue filter: ${filterName}`);
      } else {
        throw new Error(`Cannot apply filter "${filterName}" to type: ${operand.type}`);
      }
    }
    throw new Error(`Unknown filter: ${filterNode.type}`);
  }
  /**
   * Evaluates expressions following the filter operation type.
   */
  evaluateFilterExpression(node, environment) {
    const operand = this.evaluate(node.operand, environment);
    return this.applyFilter(operand, node.filter, environment);
  }
  /**
   * Evaluates expressions following the test operation type.
   */
  evaluateTestExpression(node, environment) {
    const operand = this.evaluate(node.operand, environment);
    const test = environment.tests.get(node.test.value);
    if (!test) {
      throw new Error(`Unknown test: ${node.test.value}`);
    }
    const result = test(operand);
    return new BooleanValue(node.negate ? !result : result);
  }
  /**
   * Evaluates expressions following the select operation type.
   */
  evaluateSelectExpression(node, environment) {
    const predicate = this.evaluate(node.test, environment);
    if (!predicate.__bool__().value) {
      return new UndefinedValue();
    }
    return this.evaluate(node.lhs, environment);
  }
  /**
   * Evaluates expressions following the unary operation type.
   */
  evaluateUnaryExpression(node, environment) {
    const argument = this.evaluate(node.argument, environment);
    switch (node.operator.value) {
      case "not":
        return new BooleanValue(!argument.value);
      case "+":
      case "-": {
        const sign = node.operator.value === "-" ? -1 : 1;
        if (argument instanceof IntegerValue || argument instanceof FloatValue || argument instanceof BooleanValue) {
          const value = argument instanceof BooleanValue ? argument.value ? 1 : 0 : argument.value;
          const result = sign * value;
          return argument instanceof FloatValue ? new FloatValue(result) : new IntegerValue(result);
        }
        throw new SyntaxError(`Unknown operator "${node.operator.value}" for ${argument.type}`);
      }
      default:
        throw new SyntaxError(`Unknown operator: ${node.operator.value}`);
    }
  }
  evaluateTernaryExpression(node, environment) {
    const cond = this.evaluate(node.condition, environment);
    return cond.__bool__().value ? this.evaluate(node.trueExpr, environment) : this.evaluate(node.falseExpr, environment);
  }
  evalProgram(program, environment) {
    return this.evaluateBlock(program.body, environment);
  }
  evaluateBlock(statements, environment) {
    let result = "";
    for (const statement of statements) {
      const lastEvaluated = this.evaluate(statement, environment);
      if (lastEvaluated.type !== "NullValue" && lastEvaluated.type !== "UndefinedValue") {
        result += lastEvaluated.toString();
      }
    }
    return new StringValue(result);
  }
  evaluateIdentifier(node, environment) {
    return environment.lookupVariable(node.value);
  }
  evaluateCallExpression(expr, environment) {
    const [args, kwargs] = this.evaluateArguments(expr.args, environment);
    if (kwargs.size > 0) {
      args.push(new KeywordArgumentsValue(kwargs));
    }
    const fn = this.evaluate(expr.callee, environment);
    if (fn.type !== "FunctionValue") {
      throw new Error(`Cannot call something that is not a function: got ${fn.type}`);
    }
    return fn.value(args, environment);
  }
  evaluateSliceExpression(object, expr, environment) {
    if (!(object instanceof ArrayValue || object instanceof StringValue)) {
      throw new Error("Slice object must be an array or string");
    }
    const start = this.evaluate(expr.start, environment);
    const stop = this.evaluate(expr.stop, environment);
    const step = this.evaluate(expr.step, environment);
    if (!(start instanceof IntegerValue || start instanceof UndefinedValue)) {
      throw new Error("Slice start must be numeric or undefined");
    }
    if (!(stop instanceof IntegerValue || stop instanceof UndefinedValue)) {
      throw new Error("Slice stop must be numeric or undefined");
    }
    if (!(step instanceof IntegerValue || step instanceof UndefinedValue)) {
      throw new Error("Slice step must be numeric or undefined");
    }
    if (object instanceof ArrayValue) {
      return new ArrayValue(slice(object.value, start.value, stop.value, step.value));
    } else {
      return new StringValue(slice(Array.from(object.value), start.value, stop.value, step.value).join(""));
    }
  }
  evaluateMemberExpression(expr, environment) {
    const object = this.evaluate(expr.object, environment);
    let property;
    if (expr.computed) {
      if (expr.property.type === "SliceExpression") {
        return this.evaluateSliceExpression(object, expr.property, environment);
      } else {
        property = this.evaluate(expr.property, environment);
      }
    } else if (expr.property.type === "IntegerLiteral") {
      property = new IntegerValue(expr.property.value);
    } else {
      property = new StringValue(expr.property.value);
    }
    let value;
    if (isAttributeContainer(object)) {
      if (!(property instanceof StringValue)) {
        throw new Error(`Cannot access property with non-string: got ${property.type}`);
      }
      value = object.value.get(property.value) ?? (object instanceof ObjectValue ? object.builtins.get(property.value) : void 0);
    } else if (object instanceof ArrayValue || object instanceof StringValue) {
      if (property instanceof IntegerValue) {
        value = object.value.at(property.value);
        if (object instanceof StringValue) {
          value = new StringValue(object.value.at(property.value));
        }
      } else if (property instanceof StringValue) {
        value = object.builtins.get(property.value);
      } else {
        throw new Error(`Cannot access property with non-string/non-number: got ${property.type}`);
      }
    } else {
      if (!(property instanceof StringValue)) {
        throw new Error(`Cannot access property with non-string: got ${property.type}`);
      }
      value = object.builtins.get(property.value);
    }
    return value instanceof RuntimeValue ? value : new UndefinedValue();
  }
  evaluateSet(node, environment) {
    const rhs = node.value ? this.evaluate(node.value, environment) : this.evaluateBlock(node.body, environment);
    if (node.assignee.type === "Identifier") {
      const variableName = node.assignee.value;
      environment.setVariable(variableName, rhs);
    } else if (node.assignee.type === "TupleLiteral") {
      const tuple = node.assignee;
      if (!(rhs instanceof ArrayValue)) {
        throw new Error(`Cannot unpack non-iterable type in set: ${rhs.type}`);
      }
      const arr = rhs.value;
      if (arr.length !== tuple.value.length) {
        throw new Error(`Too ${tuple.value.length > arr.length ? "few" : "many"} items to unpack in set`);
      }
      for (let i = 0; i < tuple.value.length; ++i) {
        const elem = tuple.value[i];
        if (elem.type !== "Identifier") {
          throw new Error(`Cannot unpack to non-identifier in set: ${elem.type}`);
        }
        environment.setVariable(elem.value, arr[i]);
      }
    } else if (node.assignee.type === "MemberExpression") {
      const member = node.assignee;
      const object = this.evaluate(member.object, environment);
      if (!(object instanceof NamespaceValue)) {
        throw new Error("cannot assign attribute on non-namespace object");
      }
      if (member.property.type !== "Identifier") {
        throw new Error("Cannot assign to member with non-identifier property");
      }
      object.value.set(member.property.value, rhs);
    } else {
      throw new Error(`Invalid LHS inside assignment expression: ${JSON.stringify(node.assignee)}`);
    }
    return new NullValue();
  }
  evaluateIf(node, environment) {
    const test = this.evaluate(node.test, environment);
    return this.evaluateBlock(test.__bool__().value ? node.body : node.alternate, environment);
  }
  evaluateFor(node, environment) {
    const scope = new Environment(environment);
    let test, iterable;
    if (node.iterable.type === "SelectExpression") {
      const select = node.iterable;
      iterable = this.evaluate(select.lhs, scope);
      test = select.test;
    } else {
      iterable = this.evaluate(node.iterable, scope);
    }
    if (!(iterable instanceof ArrayValue || iterable instanceof ObjectValue)) {
      throw new Error(`Expected iterable or object type in for loop: got ${iterable.type}`);
    }
    if (iterable instanceof ObjectValue) {
      iterable = iterable.keys();
    }
    const items = [];
    const scopeUpdateFunctions = [];
    for (let i = 0; i < iterable.value.length; ++i) {
      const loopScope = new Environment(scope);
      const current = iterable.value[i];
      let scopeUpdateFunction;
      if (node.loopvar.type === "Identifier") {
        scopeUpdateFunction = (scope2) => scope2.setVariable(node.loopvar.value, current);
      } else if (node.loopvar.type === "TupleLiteral") {
        const loopvar = node.loopvar;
        if (current.type !== "ArrayValue") {
          throw new Error(`Cannot unpack non-iterable type: ${current.type}`);
        }
        const c = current;
        if (loopvar.value.length !== c.value.length) {
          throw new Error(`Too ${loopvar.value.length > c.value.length ? "few" : "many"} items to unpack`);
        }
        scopeUpdateFunction = (scope2) => {
          for (let j = 0; j < loopvar.value.length; ++j) {
            if (loopvar.value[j].type !== "Identifier") {
              throw new Error(`Cannot unpack non-identifier type: ${loopvar.value[j].type}`);
            }
            scope2.setVariable(loopvar.value[j].value, c.value[j]);
          }
        };
      } else {
        throw new Error(`Invalid loop variable(s): ${node.loopvar.type}`);
      }
      if (test) {
        scopeUpdateFunction(loopScope);
        const testValue = this.evaluate(test, loopScope);
        if (!testValue.__bool__().value) {
          continue;
        }
      }
      items.push(current);
      scopeUpdateFunctions.push(scopeUpdateFunction);
    }
    let result = "";
    let noIteration = true;
    for (let i = 0; i < items.length; ++i) {
      const loop = /* @__PURE__ */ new Map([
        ["index", new IntegerValue(i + 1)],
        ["index0", new IntegerValue(i)],
        ["revindex", new IntegerValue(items.length - i)],
        ["revindex0", new IntegerValue(items.length - i - 1)],
        ["first", new BooleanValue(i === 0)],
        ["last", new BooleanValue(i === items.length - 1)],
        ["length", new IntegerValue(items.length)],
        ["previtem", i > 0 ? items[i - 1] : new UndefinedValue()],
        ["nextitem", i < items.length - 1 ? items[i + 1] : new UndefinedValue()]
      ]);
      scope.setVariable("loop", new ObjectValue(loop));
      scopeUpdateFunctions[i](scope);
      try {
        const evaluated = this.evaluateBlock(node.body, scope);
        result += evaluated.value;
      } catch (err) {
        if (err instanceof ContinueControl) {
          continue;
        }
        if (err instanceof BreakControl) {
          break;
        }
        throw err;
      }
      noIteration = false;
    }
    if (noIteration) {
      const defaultEvaluated = this.evaluateBlock(node.defaultBlock, scope);
      result += defaultEvaluated.value;
    }
    return new StringValue(result);
  }
  /**
   * Bind every parameter name and special argument before evaluating defaults,
   * preventing unbound parameters from resolving in an outer scope.
   */
  bindMacroArguments(displayName, parameters, specialArguments, args, scope) {
    const positionalArguments = args.slice();
    let keywordArguments = /* @__PURE__ */ new Map();
    if (positionalArguments.at(-1) instanceof KeywordArgumentsValue) {
      keywordArguments = new Map(positionalArguments.pop().value);
    }
    const pendingDefaults = [];
    for (let i = 0; i < parameters.length; ++i) {
      const { name, defaultValue } = getParameterDescriptor(parameters[i]);
      let value = positionalArguments[i];
      if (value === void 0 && keywordArguments.has(name)) {
        value = keywordArguments.get(name);
        keywordArguments.delete(name);
      }
      if (value === void 0 && defaultValue !== void 0) {
        pendingDefaults.push([name, defaultValue]);
      }
      scope.setVariable(name, value ?? new UndefinedValue());
    }
    if (specialArguments.has("kwargs")) {
      scope.setVariable("kwargs", new ObjectValue(keywordArguments));
    } else if (keywordArguments.size > 0) {
      throw new Error(`macro ${displayName} takes no keyword argument '${keywordArguments.keys().next().value}'`);
    }
    if (specialArguments.has("varargs")) {
      scope.setVariable("varargs", new ArrayValue(positionalArguments.slice(parameters.length)));
    } else if (positionalArguments.length > parameters.length) {
      throw new Error(`macro ${displayName} takes not more than ${parameters.length} argument(s)`);
    }
    for (const [name, defaultValue] of pendingDefaults) {
      scope.setVariable(name, this.evaluate(defaultValue, scope));
    }
  }
  /**
   * See https://jinja.palletsprojects.com/en/3.1.x/templates/#macros for more information.
   */
  evaluateMacro(node, environment) {
    const specialArguments = findAccessedSpecialMacroArguments(node.args, node.body);
    environment.setVariable(
      node.name.value,
      new FunctionValue((args, scope) => {
        const macroScope = new Environment(scope);
        this.bindMacroArguments(`'${node.name.value}'`, node.args, specialArguments, args, macroScope);
        return this.evaluateBlock(node.body, macroScope);
      })
    );
    return new NullValue();
  }
  evaluateCallStatement(node, environment) {
    const parameters = node.callerArgs ?? [];
    const specialArguments = findAccessedSpecialMacroArguments(parameters, node.body);
    const callerFn = new FunctionValue((callerArguments) => {
      const callBlockEnv = new Environment(environment);
      this.bindMacroArguments("None", parameters, specialArguments, callerArguments, callBlockEnv);
      return this.evaluateBlock(node.body, callBlockEnv);
    });
    const [macroArgs, macroKwargs] = this.evaluateArguments(node.call.args, environment);
    macroArgs.push(new KeywordArgumentsValue(macroKwargs));
    const fn = this.evaluate(node.call.callee, environment);
    if (fn.type !== "FunctionValue") {
      throw new Error(`Cannot call something that is not a function: got ${fn.type}`);
    }
    const newEnv = new Environment(environment);
    newEnv.setVariable("caller", callerFn);
    return fn.value(macroArgs, newEnv);
  }
  evaluateFilterStatement(node, environment) {
    const rendered = this.evaluateBlock(node.body, environment);
    return this.applyFilter(rendered, node.filter, environment);
  }
  evaluate(statement, environment) {
    if (!statement) {
      return new UndefinedValue();
    }
    switch (statement.type) {
      case "Program":
        return this.evalProgram(statement, environment);
      case "Set":
        return this.evaluateSet(statement, environment);
      case "If":
        return this.evaluateIf(statement, environment);
      case "For":
        return this.evaluateFor(statement, environment);
      case "Macro":
        return this.evaluateMacro(statement, environment);
      case "CallStatement":
        return this.evaluateCallStatement(statement, environment);
      case "Break":
        throw new BreakControl();
      case "Continue":
        throw new ContinueControl();
      case "IntegerLiteral":
        return new IntegerValue(statement.value);
      case "FloatLiteral":
        return new FloatValue(statement.value);
      case "StringLiteral":
        return new StringValue(statement.value);
      case "ArrayLiteral":
        return new ArrayValue(statement.value.map((x) => this.evaluate(x, environment)));
      case "TupleLiteral":
        return new TupleValue(statement.value.map((x) => this.evaluate(x, environment)));
      case "ObjectLiteral": {
        const mapping = /* @__PURE__ */ new Map();
        for (const [key, value] of statement.value) {
          const evaluatedKey = this.evaluate(key, environment);
          if (!(evaluatedKey instanceof StringValue)) {
            throw new Error(`Object keys must be strings: got ${evaluatedKey.type}`);
          }
          mapping.set(evaluatedKey.value, this.evaluate(value, environment));
        }
        return new ObjectValue(mapping);
      }
      case "Identifier":
        return this.evaluateIdentifier(statement, environment);
      case "CallExpression":
        return this.evaluateCallExpression(statement, environment);
      case "MemberExpression":
        return this.evaluateMemberExpression(statement, environment);
      case "UnaryExpression":
        return this.evaluateUnaryExpression(statement, environment);
      case "BinaryExpression":
        return this.evaluateBinaryExpression(statement, environment);
      case "FilterExpression":
        return this.evaluateFilterExpression(statement, environment);
      case "FilterStatement":
        return this.evaluateFilterStatement(statement, environment);
      case "TestExpression":
        return this.evaluateTestExpression(statement, environment);
      case "SelectExpression":
        return this.evaluateSelectExpression(statement, environment);
      case "Ternary":
        return this.evaluateTernaryExpression(statement, environment);
      case "Comment":
        return new NullValue();
      default:
        throw new SyntaxError(`Unknown node type: ${statement.type}`);
    }
  }
};
function convertToRuntimeValues(input) {
  switch (typeof input) {
    case "number":
      return Number.isInteger(input) ? new IntegerValue(input) : new FloatValue(input);
    case "string":
      return new StringValue(input);
    case "boolean":
      return new BooleanValue(input);
    case "undefined":
      return new UndefinedValue();
    case "object":
      if (input === null) {
        return new NullValue();
      } else if (Array.isArray(input)) {
        return new ArrayValue(input.map(convertToRuntimeValues));
      } else {
        return new ObjectValue(
          new Map(Object.entries(input).map(([key, value]) => [key, convertToRuntimeValues(value)]))
        );
      }
    case "function":
      return new FunctionValue((args, _scope) => {
        const result = input(...args.map((x) => x.value)) ?? null;
        return convertToRuntimeValues(result);
      });
    default:
      throw new Error(`Cannot convert to runtime value: ${input}`);
  }
}
var NEWLINE = "\n";
var OPEN_STATEMENT = "{%- ";
var CLOSE_STATEMENT = " -%}";
var PRECEDENCE = Object.freeze({
  CONDITIONAL: 0,
  // a if b else c
  LOGICAL_OR: 1,
  // or
  LOGICAL_AND: 2,
  // and
  LOGICAL_NOT: 3,
  // not a
  COMPARISON: 4,
  // == != < <= > >= in, not in
  ADDITIVE: 5,
  // + -
  MULTIPLICATIVE: 6,
  // * / // %
  EXPONENTIATION: 7,
  // **
  TEST: 8,
  // is
  FILTER: 9,
  // |
  UNARY_SIGN: 10,
  // -a +a
  ATOM: 11
  // literals, identifiers, calls, member accesses
});
function getBinaryOperatorPrecedence(expr) {
  switch (expr.operator.type) {
    case "ExponentiationBinaryOperator":
      return PRECEDENCE.EXPONENTIATION;
    case "MultiplicativeBinaryOperator":
      return PRECEDENCE.MULTIPLICATIVE;
    case "AdditiveBinaryOperator":
      return PRECEDENCE.ADDITIVE;
    case "ComparisonBinaryOperator":
      return PRECEDENCE.COMPARISON;
    case "Identifier":
      if (expr.operator.value === "and") {
        return PRECEDENCE.LOGICAL_AND;
      }
      if (expr.operator.value === "in" || expr.operator.value === "not in") {
        return PRECEDENCE.COMPARISON;
      }
      return PRECEDENCE.LOGICAL_OR;
  }
  return PRECEDENCE.LOGICAL_OR;
}
function getPrecedence(node) {
  switch (node.type) {
    case "SelectExpression":
    case "Ternary":
      return PRECEDENCE.CONDITIONAL;
    case "BinaryExpression":
      return getBinaryOperatorPrecedence(node);
    case "UnaryExpression":
      return node.operator.value === "not" ? PRECEDENCE.LOGICAL_NOT : PRECEDENCE.UNARY_SIGN;
    case "TestExpression":
      return PRECEDENCE.TEST;
    case "FilterExpression":
      return PRECEDENCE.FILTER;
    default:
      return PRECEDENCE.ATOM;
  }
}
function formatOperand(node, minPrecedence) {
  const expr = formatExpression(node);
  return getPrecedence(node) < minPrecedence ? `(${expr})` : expr;
}
function format(program, indent = "	") {
  const indentStr = typeof indent === "number" ? " ".repeat(indent) : indent;
  const body = formatStatements(program.body, 0, indentStr);
  return body.replace(/\n$/, "");
}
function createStatement(...text) {
  return OPEN_STATEMENT + text.join(" ") + CLOSE_STATEMENT;
}
function formatStatements(stmts, depth, indentStr) {
  return stmts.map((stmt) => formatStatement(stmt, depth, indentStr)).join(NEWLINE);
}
function formatExpressionList(expressions) {
  return expressions.map((expression) => formatExpression(expression)).join(", ");
}
function formatStatement(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  switch (node.type) {
    case "Program":
      return formatStatements(node.body, depth, indentStr);
    case "If":
      return formatIf(node, depth, indentStr);
    case "For":
      return formatFor(node, depth, indentStr);
    case "Set":
      return formatSet(node, depth, indentStr);
    case "Macro":
      return formatMacro(node, depth, indentStr);
    case "Break":
      return pad + createStatement("break");
    case "Continue":
      return pad + createStatement("continue");
    case "CallStatement":
      return formatCallStatement(node, depth, indentStr);
    case "FilterStatement":
      return formatFilterStatement(node, depth, indentStr);
    case "Comment":
      return pad + "{# " + node.value + " #}";
    default:
      return pad + "{{- " + formatExpression(node) + " -}}";
  }
}
function formatIf(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  const clauses = [];
  let current = node;
  while (current) {
    clauses.push({ test: current.test, body: current.body });
    if (current.alternate.length === 1 && current.alternate[0].type === "If") {
      current = current.alternate[0];
    } else {
      break;
    }
  }
  let out = pad + createStatement("if", formatExpression(clauses[0].test)) + NEWLINE + formatStatements(clauses[0].body, depth + 1, indentStr);
  for (let i = 1; i < clauses.length; ++i) {
    out += NEWLINE + pad + createStatement("elif", formatExpression(clauses[i].test)) + NEWLINE + formatStatements(clauses[i].body, depth + 1, indentStr);
  }
  if (current && current.alternate.length > 0) {
    out += NEWLINE + pad + createStatement("else") + NEWLINE + formatStatements(current.alternate, depth + 1, indentStr);
  }
  out += NEWLINE + pad + createStatement("endif");
  return out;
}
function formatFor(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  let formattedIterable = "";
  if (node.iterable.type === "SelectExpression") {
    const n2 = node.iterable;
    formattedIterable = `${formatExpression(n2.lhs)} if ${formatExpression(n2.test)}`;
  } else {
    formattedIterable = formatExpression(node.iterable);
  }
  let out = pad + createStatement("for", formatExpression(node.loopvar), "in", formattedIterable) + NEWLINE + formatStatements(node.body, depth + 1, indentStr);
  if (node.defaultBlock.length > 0) {
    out += NEWLINE + pad + createStatement("else") + NEWLINE + formatStatements(node.defaultBlock, depth + 1, indentStr);
  }
  out += NEWLINE + pad + createStatement("endfor");
  return out;
}
function formatSet(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  const left = formatExpression(node.assignee);
  const right = node.value ? formatExpression(node.value) : "";
  const value = pad + createStatement("set", `${left}${node.value ? " = " + right : ""}`);
  if (node.body.length === 0) {
    return value;
  }
  return value + NEWLINE + formatStatements(node.body, depth + 1, indentStr) + NEWLINE + pad + createStatement("endset");
}
function formatMacro(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  const args = formatExpressionList(node.args);
  return pad + createStatement("macro", `${node.name.value}(${args})`) + NEWLINE + formatStatements(node.body, depth + 1, indentStr) + NEWLINE + pad + createStatement("endmacro");
}
function formatCallStatement(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  const params = node.callerArgs && node.callerArgs.length > 0 ? `(${formatExpressionList(node.callerArgs)})` : "";
  const callExpr = formatExpression(node.call);
  let out = pad + createStatement(`call${params}`, callExpr) + NEWLINE;
  out += formatStatements(node.body, depth + 1, indentStr) + NEWLINE;
  out += pad + createStatement("endcall");
  return out;
}
function formatFilterStatement(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  const spec = node.filter.type === "Identifier" ? node.filter.value : formatExpression(node.filter);
  let out = pad + createStatement("filter", spec) + NEWLINE;
  out += formatStatements(node.body, depth + 1, indentStr) + NEWLINE;
  out += pad + createStatement("endfilter");
  return out;
}
function formatExpression(node) {
  switch (node.type) {
    case "SpreadExpression": {
      const n2 = node;
      return `*${formatExpression(n2.argument)}`;
    }
    case "KeywordSpreadExpression": {
      const n2 = node;
      return `**${formatExpression(n2.argument)}`;
    }
    case "Identifier":
      return node.value;
    case "IntegerLiteral":
      return `${node.value}`;
    case "FloatLiteral": {
      const value = node.value;
      if (Object.is(value, -0)) {
        return "-0.0";
      }
      return value % 1 === 0 ? value.toFixed(1) : value.toString();
    }
    case "StringLiteral":
      return JSON.stringify(node.value);
    case "BinaryExpression": {
      const n2 = node;
      const thisPrecedence = getBinaryOperatorPrecedence(n2);
      const left = formatOperand(n2.left, thisPrecedence);
      const right = formatOperand(n2.right, thisPrecedence + 1);
      return `${left} ${n2.operator.value} ${right}`;
    }
    case "UnaryExpression": {
      const n2 = node;
      const operandPrecedence = n2.argument.type === "UnaryExpression" ? getPrecedence(n2) : PRECEDENCE.ATOM;
      return n2.operator.value + (n2.operator.value === "not" ? " " : "") + formatOperand(n2.argument, operandPrecedence);
    }
    case "CallExpression": {
      const n2 = node;
      const args = formatExpressionList(n2.args);
      return `${formatExpression(n2.callee)}(${args})`;
    }
    case "MemberExpression": {
      const n2 = node;
      const obj = formatOperand(n2.object, PRECEDENCE.ATOM);
      let prop = formatExpression(n2.property);
      if (!n2.computed && n2.property.type !== "Identifier" && n2.property.type !== "IntegerLiteral") {
        prop = `(${prop})`;
      }
      return n2.computed ? `${obj}[${prop}]` : `${obj}.${prop}`;
    }
    case "FilterExpression": {
      const n2 = node;
      const operand = formatOperand(n2.operand, PRECEDENCE.FILTER);
      if (n2.filter.type === "CallExpression") {
        return `${operand} | ${formatExpression(n2.filter)}`;
      }
      return `${operand} | ${n2.filter.value}`;
    }
    case "SelectExpression": {
      const n2 = node;
      return `${formatOperand(n2.lhs, PRECEDENCE.LOGICAL_OR)} if ${formatOperand(n2.test, PRECEDENCE.LOGICAL_OR)}`;
    }
    case "TestExpression": {
      const n2 = node;
      return `${formatOperand(n2.operand, PRECEDENCE.TEST)} is${n2.negate ? " not" : ""} ${n2.test.value}`;
    }
    case "ArrayLiteral":
    case "TupleLiteral": {
      const elems = formatExpressionList(node.value);
      const brackets = node.type === "ArrayLiteral" ? "[]" : "()";
      return `${brackets[0]}${elems}${brackets[1]}`;
    }
    case "ObjectLiteral": {
      const entries = Array.from(node.value.entries()).map(
        ([k, v]) => `${formatExpression(k)}: ${formatExpression(v)}`
      );
      return `{${entries.join(", ")}}`;
    }
    case "SliceExpression": {
      const n2 = node;
      const s = n2.start ? formatExpression(n2.start) : "";
      const t = n2.stop ? formatExpression(n2.stop) : "";
      const st = n2.step ? `:${formatExpression(n2.step)}` : "";
      return `${s}:${t}${st}`;
    }
    case "KeywordArgumentExpression": {
      const n2 = node;
      return `${n2.key.value}=${formatExpression(n2.value)}`;
    }
    case "Ternary": {
      const n2 = node;
      return `${formatOperand(n2.trueExpr, PRECEDENCE.LOGICAL_OR)} if ${formatOperand(
        n2.condition,
        PRECEDENCE.LOGICAL_OR
      )} else ${formatOperand(n2.falseExpr, PRECEDENCE.CONDITIONAL)}`;
    }
    default:
      throw new Error(`Unknown expression type: ${node.type}`);
  }
}
var Template = class {
  parsed;
  /**
   * @param {string} template The template string
   */
  constructor(template) {
    const tokens = tokenize(template, {
      lstrip_blocks: true,
      trim_blocks: true
    });
    this.parsed = parse(tokens);
  }
  render(items) {
    const env = new Environment();
    setupGlobals(env);
    if (items) {
      for (const [key, value] of Object.entries(items)) {
        env.set(key, value);
      }
    }
    const interpreter = new Interpreter(env);
    const result = interpreter.run(this.parsed);
    return result.value;
  }
  format(options) {
    return format(this.parsed, options?.indent || "	");
  }
};

// src/utils/templatePortability.ts
var PORTABILITY_RENDERER = "@huggingface/jinja";
var KNOWN_CALLABLES = /* @__PURE__ */ new Set([
  "range",
  "dict",
  "namespace",
  "cycler",
  "joiner",
  "lipsum",
  "loop",
  "caller",
  "super",
  "self",
  "varargs",
  "kwargs",
  "raise_exception",
  "strftime_now"
]);
var NOT_CALLS = /* @__PURE__ */ new Set([
  "if",
  "elif",
  "else",
  "for",
  "in",
  "not",
  "and",
  "or",
  "is",
  "set",
  "macro",
  "call",
  "filter",
  "with",
  "endif",
  "endfor",
  "return",
  "block",
  "include",
  "import",
  "from",
  "as",
  "recursive",
  "none",
  "None",
  "true",
  "false",
  "True",
  "False"
]);
var TAG = /\{\{-?([\s\S]*?)-?\}\}|\{%-?([\s\S]*?)-?%\}/g;
function stripStrings(s) {
  return s.replace(/'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"/g, '""');
}
function undefinedCalls(template) {
  const defined = /* @__PURE__ */ new Set();
  const calls = /* @__PURE__ */ new Map();
  const tags = [];
  for (const m of template.matchAll(TAG)) tags.push(stripStrings(m[1] ?? m[2] ?? ""));
  for (const body of tags) {
    const macro = body.match(/^\s*macro\s+([A-Za-z_]\w*)\s*\(([^)]*)\)/);
    if (macro) {
      defined.add(macro[1]);
      for (const p of macro[2].split(",")) {
        const n2 = p.split("=")[0].trim();
        if (n2) defined.add(n2);
      }
    }
    const set = body.match(/^\s*set\s+([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)/);
    if (set) for (const n2 of set[1].split(",")) defined.add(n2.trim());
    const loop = body.match(/^\s*for\s+([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s+in\b/);
    if (loop) for (const n2 of loop[1].split(",")) defined.add(n2.trim());
    const imp = body.match(/^\s*from\s+\S+\s+import\s+(.+)$/);
    if (imp) for (const n2 of imp[1].split(",")) defined.add(n2.trim().split(/\s+as\s+/).pop().trim());
  }
  for (const body of tags) {
    if (/^\s*macro\s/.test(body)) continue;
    for (const m of body.matchAll(/([A-Za-z_]\w*)\s*\(/g)) {
      const name = m[1];
      const before = body.slice(0, m.index);
      if (/(\.|\|\s*|\bis\s+(not\s+)?)$/.test(before) || /\w$/.test(before)) continue;
      if (NOT_CALLS.has(name) || KNOWN_CALLABLES.has(name) || defined.has(name)) continue;
      calls.set(name, (calls.get(name) ?? 0) + 1);
    }
  }
  return [...calls].map(([name, count]) => ({ name, count }));
}
var TOOL = {
  type: "function",
  function: { name: "get_weather", description: "Current weather for a city.", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } }
};
var CASES = [
  { name: "user", messages: [{ role: "user", content: "Hello." }], tools: null },
  { name: "system+multi-turn", messages: [{ role: "system", content: "Be brief." }, { role: "user", content: "Name a prime." }, { role: "assistant", content: "Seven." }, { role: "user", content: "Another." }], tools: null },
  {
    name: "tools",
    messages: [
      { role: "user", content: "Weather in Paris?" },
      { role: "assistant", content: "", tool_calls: [{ id: "call00001", type: "function", function: { name: "get_weather", arguments: { city: "Paris" } } }] },
      { role: "tool", tool_call_id: "call00001", name: "get_weather", content: '{"temp_c": 18}' }
    ],
    tools: [TOOL]
  }
];
function checkTemplatePortability(template, specials = {}) {
  const out = { renderer: PORTABILITY_RENDERER, parsed: true, undefinedCalls: undefinedCalls(template), cases: [] };
  let t;
  try {
    t = new Template(template);
  } catch (e) {
    return { ...out, parsed: false, parseError: String(e.message ?? e).slice(0, 200) };
  }
  for (const c of CASES) {
    try {
      t.render({
        messages: structuredClone(c.messages),
        tools: c.tools ? structuredClone(c.tools) : null,
        documents: null,
        add_generation_prompt: true,
        bos_token: specials.bos_token ?? "",
        eos_token: specials.eos_token ?? ""
      });
      out.cases.push({ case: c.name, ok: true });
    } catch (e) {
      const message = String(e.message ?? e).slice(0, 200);
      const raised = e.name === "Error" && !/^(Unknown|Cannot|Expected|Invalid|Unexpected|Variable|Unsupported|Unable|Not )/.test(message);
      out.cases.push({ case: c.name, ok: false, raised, message });
    }
  }
  return out;
}
function specialTokenText(v) {
  if (typeof v === "string") return v;
  if (v && typeof v === "object" && typeof v.content === "string") return v.content;
  return null;
}

// src/utils/serveEngineTables.generated.ts
var ENGINE_QUANT_TABLES = [
  {
    "engine": "vllm",
    "tag": "v0.11.0",
    "commit": "b8b302cde434df8c9289a2b465406b47ebab1c2d",
    "date": "2025-10-10",
    "repo": "vllm-project/vllm",
    "methods": {
      "awq": {
        "cls": "AWQConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/awq.py#L60"
      },
      "deepspeedfp": {
        "cls": "DeepSpeedFPConfig",
        "minSm": 60,
        "at": "vllm/model_executor/layers/quantization/deepspeedfp.py#L65"
      },
      "tpu_int8": {
        "cls": "Int8TpuConfig",
        "minSm": null,
        "at": "vllm/model_executor/layers/quantization/tpu_int8.py#L19"
      },
      "fp8": {
        "cls": "Fp8Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/fp8.py#L114"
      },
      "fbgemm_fp8": {
        "cls": "FBGEMMFp8Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/fbgemm_fp8.py#L50"
      },
      "modelopt": {
        "cls": "ModelOptFp8Config",
        "minSm": 89,
        "at": "vllm/model_executor/layers/quantization/modelopt.py#L85"
      },
      "modelopt_fp4": {
        "cls": "ModelOptNvFp4Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/modelopt.py#L643"
      },
      "bitblas": {
        "cls": "BitBLASConfig",
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/bitblas.py#L117"
      },
      "gguf": {
        "cls": "GGUFConfig",
        "minSm": 60,
        "at": "vllm/model_executor/layers/quantization/gguf.py#L48"
      },
      "gptq_marlin_24": {
        "cls": "GPTQMarlin24Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/gptq_marlin_24.py#L99"
      },
      "gptq_marlin": {
        "cls": "GPTQMarlinConfig",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/gptq_marlin.py#L156"
      },
      "gptq_bitblas": {
        "cls": "GPTQBitBLASConfig",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/gptq_bitblas.py#L140"
      },
      "awq_marlin": {
        "cls": "AWQMarlinConfig",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/awq_marlin.py#L90"
      },
      "gptq": {
        "cls": "GPTQConfig",
        "minSm": 60,
        "at": "vllm/model_executor/layers/quantization/gptq.py#L107"
      },
      "compressed-tensors": {
        "cls": "CompressedTensorsConfig",
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py#L90"
      },
      "bitsandbytes": {
        "cls": "BitsAndBytesConfig",
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/bitsandbytes.py#L75"
      },
      "ptpc_fp8": {
        "cls": "PTPCFp8Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/fp8.py#L114"
      },
      "hqq": {
        "cls": "HQQMarlinConfig",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/hqq_marlin.py#L63"
      },
      "experts_int8": {
        "cls": "ExpertsInt8Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/experts_int8.py#L36"
      },
      "ipex": {
        "cls": "IPEXConfig",
        "minSm": null,
        "at": "vllm/model_executor/layers/quantization/ipex_quant.py#L31"
      },
      "quark": {
        "cls": "QuarkConfig",
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/quark/quark.py#L52"
      },
      "moe_wna16": {
        "cls": "MoeWNA16Config",
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/moe_wna16.py#L80"
      },
      "torchao": {
        "cls": "TorchAOConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/torchao.py#L72"
      },
      "auto-round": {
        "cls": "AutoRoundConfig",
        "minSm": 60,
        "at": "vllm/model_executor/layers/quantization/auto_round.py#L96"
      },
      "rtn": {
        "cls": "RTNConfig",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/rtn.py#L65"
      },
      "inc": {
        "cls": "INCConfig",
        "minSm": null,
        "at": "vllm/model_executor/layers/quantization/inc.py#L32"
      },
      "mxfp4": {
        "cls": "Mxfp4Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/mxfp4.py#L121"
      },
      "petit_nvfp4": {
        "cls": "PetitNvFp4Config",
        "minSm": 90,
        "at": "vllm/model_executor/layers/quantization/petit.py#L75"
      }
    },
    "compressedTensorsSchemes": {
      "24": {
        "minSm": 90,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_24.py#L60"
      },
      "w4a16sparse24": {
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_24.py#L49"
      },
      "w4a16fp4": {
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py#L26"
      },
      "w4a4fp4": {
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L52"
      },
      "w4a8fp8": {
        "minSm": 90,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L62"
      },
      "w4a8int": {
        "minSm": 1,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py#L48"
      },
      "w8a16fp8": {
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py#L33"
      },
      "w8a8fp8": {
        "minSm": 89,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L55"
      },
      "w8a8int8": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L32"
      },
      "wna16": {
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py#L69"
      }
    }
  },
  {
    "engine": "vllm",
    "tag": "v0.19.1",
    "commit": "b1388b1fbf5aaef47937fabe98931211684666a6",
    "date": "2026-04-17",
    "repo": "vllm-project/vllm",
    "methods": {
      "awq": {
        "cls": "AWQConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/awq.py#L73"
      },
      "fp8": {
        "cls": "Fp8Config",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/fp8.py#L145"
      },
      "fbgemm_fp8": {
        "cls": "FBGEMMFp8Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/fbgemm_fp8.py#L65"
      },
      "fp_quant": {
        "cls": "FPQuantConfig",
        "minSm": 100,
        "at": "vllm/model_executor/layers/quantization/fp_quant.py#L69"
      },
      "modelopt": {
        "cls": "ModelOptFp8Config",
        "minSm": 89,
        "at": "vllm/model_executor/layers/quantization/modelopt.py#L408"
      },
      "modelopt_fp4": {
        "cls": "ModelOptNvFp4Config",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/modelopt.py#L1013"
      },
      "modelopt_mxfp8": {
        "cls": "ModelOptMxFp8Config",
        "minSm": 100,
        "at": "vllm/model_executor/layers/quantization/modelopt.py#L1501"
      },
      "modelopt_mixed": {
        "cls": "ModelOptMixedPrecisionConfig",
        "minSm": 89,
        "at": "vllm/model_executor/layers/quantization/modelopt.py#L2084"
      },
      "gguf": {
        "cls": "GGUFConfig",
        "minSm": 60,
        "at": "vllm/model_executor/layers/quantization/gguf.py#L71"
      },
      "gptq_marlin": {
        "cls": "GPTQMarlinConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/gptq_marlin.py#L183"
      },
      "awq_marlin": {
        "cls": "AWQMarlinConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/awq_marlin.py#L206"
      },
      "gptq": {
        "cls": "GPTQConfig",
        "minSm": 60,
        "at": "vllm/model_executor/layers/quantization/gptq.py#L135"
      },
      "compressed-tensors": {
        "cls": "CompressedTensorsConfig",
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py#L116"
      },
      "bitsandbytes": {
        "cls": "BitsAndBytesConfig",
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/bitsandbytes.py#L104"
      },
      "experts_int8": {
        "cls": "ExpertsInt8Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/experts_int8.py#L42"
      },
      "quark": {
        "cls": "QuarkConfig",
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/quark/quark.py#L88"
      },
      "moe_wna16": {
        "cls": "MoeWNA16Config",
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/moe_wna16.py#L97"
      },
      "torchao": {
        "cls": "TorchAOConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/torchao.py#L129"
      },
      "auto-round": {
        "cls": "INCConfig",
        "minSm": 60,
        "at": "vllm/model_executor/layers/quantization/inc.py#L105"
      },
      "inc": {
        "cls": "INCConfig",
        "minSm": 60,
        "at": "vllm/model_executor/layers/quantization/inc.py#L105"
      },
      "mxfp4": {
        "cls": "Mxfp4Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/mxfp4.py#L50"
      },
      "mxfp8": {
        "cls": "Mxfp8Config",
        "minSm": 100,
        "at": "vllm/model_executor/layers/quantization/mxfp8.py#L78"
      },
      "petit_nvfp4": {
        "cls": "PetitNvFp4Config",
        "minSm": 90,
        "at": "vllm/model_executor/layers/quantization/petit.py#L84"
      },
      "cpu_awq": {
        "cls": "CPUAWQConfig",
        "minSm": null,
        "at": "vllm/model_executor/layers/quantization/cpu_wna16.py#L40"
      }
    },
    "compressedTensorsSchemes": {
      "w4a16mxfp4": {
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4.py#L40"
      },
      "w4a16fp4": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py#L29"
      },
      "w4a4fp4": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L31"
      },
      "w4a8fp8": {
        "minSm": 90,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L68"
      },
      "w4a8int": {
        "minSm": 1,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py#L56"
      },
      "w8a16fp8": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py#L49"
      },
      "w8a8fp8": {
        "minSm": 89,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L95"
      },
      "w8a8int8": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L35"
      },
      "wna16": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py#L79"
      }
    }
  },
  {
    "engine": "vllm",
    "tag": "v0.32.0",
    "commit": "5df63a7203cdb9328d824a27b682e77c7b9928be",
    "date": "2026-10-17",
    "repo": "vllm-project/vllm",
    "methods": {
      "awq": {
        "cls": "AutoAWQConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/auto_awq.py#L233"
      },
      "awq_marlin": {
        "cls": "AutoAWQConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/auto_awq.py#L233"
      },
      "auto_awq": {
        "cls": "AutoAWQConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/auto_awq.py#L233"
      },
      "fp8": {
        "cls": "Fp8Config",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/fp8.py#L146"
      },
      "fbgemm_fp8": {
        "cls": "FBGEMMFp8Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/fbgemm_fp8.py#L66"
      },
      "fp_quant": {
        "cls": "FPQuantConfig",
        "minSm": 100,
        "at": "vllm/model_executor/layers/quantization/fp_quant.py#L63"
      },
      "modelopt": {
        "cls": "ModelOptFp8Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/modelopt.py#L445"
      },
      "modelopt_fp4": {
        "cls": "ModelOptNvFp4Config",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/modelopt.py#L768"
      },
      "modelopt_mxfp8": {
        "cls": "ModelOptMxFp8Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/modelopt.py#L1200"
      },
      "modelopt_mixed": {
        "cls": "ModelOptMixedPrecisionConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/modelopt.py#L1616"
      },
      "auto_gptq": {
        "cls": "AutoGPTQConfig",
        "minSm": 60,
        "at": "vllm/model_executor/layers/quantization/auto_gptq.py#L182"
      },
      "gptq": {
        "cls": "AutoGPTQConfig",
        "minSm": 60,
        "at": "vllm/model_executor/layers/quantization/auto_gptq.py#L182"
      },
      "gptq_marlin": {
        "cls": "AutoGPTQConfig",
        "minSm": 60,
        "at": "vllm/model_executor/layers/quantization/auto_gptq.py#L182"
      },
      "compressed-tensors": {
        "cls": "CompressedTensorsConfig",
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py#L123"
      },
      "experts_int8": {
        "cls": "ExpertsInt8Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/experts_int8.py#L42"
      },
      "quark": {
        "cls": "QuarkConfig",
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/quark/quark.py#L127"
      },
      "moe_wna16": {
        "cls": "MoeWNA16Config",
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/moe_wna16.py#L110"
      },
      "torchao": {
        "cls": "TorchAOConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/torchao.py#L160"
      },
      "inc": {
        "cls": "INCConfig",
        "minSm": 60,
        "at": "vllm/model_executor/layers/quantization/inc/inc.py#L266"
      },
      "mxfp4": {
        "cls": "Mxfp4Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/mxfp4.py#L71"
      },
      "gpt_oss_mxfp4": {
        "cls": "GptOssMxfp4Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/mxfp4.py#L71"
      },
      "deepseek_v4_fp8": {
        "cls": "deepseek_config",
        "minSm": null,
        "at": null
      },
      "humming": {
        "cls": "HummingConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/humming.py#L167"
      },
      "online": {
        "cls": "OnlineQuantizationConfig",
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/online/base.py#L209"
      },
      "mxfp8": {
        "cls": "ModelOptMxFp8Config",
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/modelopt.py#L1200"
      }
    },
    "compressedTensorsSchemes": {
      "w4a4mxfp4": {
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py#L46"
      },
      "w4a4fp4": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L35"
      },
      "w4a8fp8": {
        "minSm": 90,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L65"
      },
      "w4a8int": {
        "minSm": 1,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py#L56"
      },
      "w4a8mxfp4": {
        "minSm": 80,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py#L46"
      },
      "w8a16fp8": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py#L53"
      },
      "w8a8fp8": {
        "minSm": 89,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L83"
      },
      "w8a8int8": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L35"
      },
      "w8a8mxfp8": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py#L37"
      },
      "wna16": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py#L102"
      },
      "wna4int": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa4.py#L111"
      },
      "wna8int": {
        "minSm": 75,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8.py#L111"
      },
      "wna8o8int": {
        "minSm": 70,
        "at": "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py#L83"
      }
    }
  },
  {
    "engine": "sglang",
    "tag": "v0.5.22",
    "commit": "baa9befa381511cd4d56f83980951e5a45fffc6d",
    "date": "2026-10-13",
    "repo": "sgl-project/sglang",
    "methods": {
      "fp8": {
        "cls": "Fp8Config",
        "minSm": 80,
        "conditional": [
          94,
          100,
          80
        ],
        "at": "python/sglang/srt/layers/quantization/fp8.py#L323"
      },
      "mxfp8": {
        "cls": "Fp8Config",
        "minSm": 80,
        "conditional": [
          94,
          100,
          80
        ],
        "at": "python/sglang/srt/layers/quantization/fp8.py#L323"
      },
      "blockwise_int8": {
        "cls": "BlockInt8Config",
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/blockwise_int8.py#L83"
      },
      "modelopt": {
        "cls": "ModelOptFp8Config",
        "minSm": 89,
        "at": "python/sglang/srt/layers/quantization/modelopt_quant.py#L461"
      },
      "modelopt_fp8": {
        "cls": "ModelOptFp8Config",
        "minSm": 89,
        "at": "python/sglang/srt/layers/quantization/modelopt_quant.py#L461"
      },
      "modelopt_fp4": {
        "cls": "ModelOptFp4Config",
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/modelopt_quant.py#L1528"
      },
      "nvfp4_online": {
        "cls": "NvFp4OnlineConfig",
        "minSm": 100,
        "at": "python/sglang/srt/layers/quantization/nvfp4_online.py#L108"
      },
      "modelopt_mixed": {
        "cls": "ModelOptMixedPrecisionConfig",
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/modelopt_quant.py#L1528"
      },
      "w8a8_int8": {
        "cls": "W8A8Int8Config",
        "minSm": 75,
        "at": "python/sglang/srt/layers/quantization/w8a8_int8.py#L87"
      },
      "w8a8_fp8": {
        "cls": "W8A8Fp8Config",
        "minSm": 89,
        "at": "python/sglang/srt/layers/quantization/w8a8_fp8.py#L66"
      },
      "awq": {
        "cls": "AWQConfig",
        "minSm": 75,
        "at": "python/sglang/srt/layers/quantization/awq/awq.py#L109"
      },
      "awq_marlin": {
        "cls": "AWQMarlinConfig",
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/awq/awq.py#L307"
      },
      "bitsandbytes": {
        "cls": "BitsAndBytesConfig",
        "minSm": 70,
        "at": "python/sglang/srt/layers/quantization/bitsandbytes.py#L100"
      },
      "gguf": {
        "cls": "GGUFConfig",
        "minSm": 60,
        "at": "python/sglang/srt/layers/quantization/gguf.py#L107"
      },
      "gptq_marlin": {
        "cls": "GPTQMarlinConfig",
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/gptq/gptq.py#L386"
      },
      "moe_wna16": {
        "cls": "MoeWNA16Config",
        "minSm": 70,
        "at": "python/sglang/srt/layers/quantization/moe_wna16.py#L126"
      },
      "compressed-tensors": {
        "cls": "CompressedTensorsConfig",
        "minSm": 70,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py#L159"
      },
      "w4afp8": {
        "cls": "W4AFp8Config",
        "minSm": 90,
        "at": "python/sglang/srt/layers/quantization/w4afp8.py#L70"
      },
      "petit_nvfp4": {
        "cls": "PetitNvFp4Config",
        "minSm": 90,
        "at": "python/sglang/srt/layers/quantization/petit.py#L62"
      },
      "quark": {
        "cls": "QuarkConfig",
        "minSm": 70,
        "at": "python/sglang/srt/layers/quantization/quark/quark.py#L371"
      },
      "quark_mxfp4": {
        "cls": "QuarkConfig",
        "minSm": 70,
        "at": "python/sglang/srt/layers/quantization/quark/quark.py#L371"
      },
      "auto-round": {
        "cls": "AutoRoundConfig",
        "minSm": 60,
        "at": "python/sglang/srt/layers/quantization/auto_round.py#L131"
      },
      "auto-round-int8": {
        "cls": "W8A8Int8Config",
        "minSm": 75,
        "at": "python/sglang/srt/layers/quantization/w8a8_int8.py#L87"
      },
      "modelslim": {
        "cls": "ModelSlimConfig",
        "minSm": 0,
        "at": "python/sglang/srt/layers/quantization/modelslim/modelslim.py#L242"
      },
      "quark_int4fp8_moe": {
        "cls": "QuarkInt4Fp8Config",
        "minSm": 70,
        "at": "python/sglang/srt/layers/quantization/quark_int4fp8_moe.py#L91"
      },
      "humming": {
        "cls": "HummingConfig",
        "minSm": 75,
        "at": "python/sglang/srt/layers/quantization/humming.py#L488"
      },
      "mxfp_w4a8": {
        "cls": "Mxfp4W4A8Config",
        "minSm": 0,
        "at": "python/sglang/srt/layers/quantization/npu_mxfp4.py#L62"
      },
      "mxfp4": {
        "cls": "Mxfp4Config",
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/mxfp4.py#L338"
      }
    },
    "compressedTensorsSchemes": {
      "w4a16fp4": {
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py#L135"
      },
      "w4a16nvfp4moe": {
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4_moe.py#L53"
      },
      "mxint4moe": {
        "minSm": 100,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py#L83"
      },
      "w4a4fp4": {
        "minSm": 100,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L37"
      },
      "w4a4nvfp4moe": {
        "minSm": 100,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py#L50"
      },
      "w4afp8moe": {
        "minSm": 90,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py#L102"
      },
      "w8a16fp8": {
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py#L35"
      },
      "w8a8fp8": {
        "minSm": 89,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L58"
      },
      "w8a8fp8moe": {
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py#L80"
      },
      "w8a8int8": {
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L97"
      },
      "wna16": {
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py#L94"
      },
      "wna16moe": {
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py#L96"
      },
      "wna16tritonmoe": {
        "minSm": 80,
        "at": "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py#L96"
      }
    }
  }
];
var VLLM_RECIPES = { "repo": "vllm-project/recipes", "commit": "2338e9edec1d5df5f823abf3c1f19e7ba0b2ed0e", "date": "2026-10-18", "count": 215, "recipes": { "ByteDance-Seed/Seed-OSS-36B-Instruct": { "file": "models/ByteDance-Seed/Seed-OSS-36B-Instruct.yaml", "minVllmVersion": "0.11.0", "contextLength": 524288, "toolCallParser": "seed_oss", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 86, "extraArgs": ["--tensor-parallel-size", "8"] } } }, "google/diffusiongemma-26B-A4B-it": { "file": "models/Google/diffusiongemma-26B-A4B-it.yaml", "minVllmVersion": "0.24.0", "contextLength": 262144, "baseArgs": ["--max-num-seqs", "4"], "toolCallParser": "gemma4", "reasoningParser": "gemma4", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 64 }, "fp8": { "modelId": "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 32 }, "nvfp4": { "modelId": "RedHatAI/diffusiongemma-26B-A4B-it-NVFP4", "precision": "nvfp4", "vramMinimumGb": 24 }, "nvidia_nvfp4": { "modelId": "nvidia/diffusiongemma-26B-A4B-it-NVFP4", "precision": "nvfp4", "vramMinimumGb": 24 } }, "verifiedGpus": ["h100", "h200", "dgx_spark_gb10", "xeon6", "arc_pro_b60", "arc_pro_b70"] }, "google/gemma-4-12B-it": { "file": "models/Google/gemma-4-12B-it.yaml", "minVllmVersion": "0.23.0", "contextLength": 131072, "toolCallParser": "gemma4", "reasoningParser": "gemma4", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 29 }, "nvfp4": { "modelId": "RedHatAI/gemma-4-12B-it-NVFP4", "precision": "nvfp4", "vramMinimumGb": 8, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "w4a16": { "modelId": "google/gemma-4-12B-it-qat-w4a16-ct", "precision": "int4", "vramMinimumGb": 9 }, "fp8": { "modelId": "RedHatAI/gemma-4-12B-it-FP8-Dynamic", "precision": "fp8", "vramMinimumGb": 15, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h100", "arc_pro_b60", "arc_pro_b70"] }, "google/gemma-4-26B-A4B-it": { "file": "models/Google/gemma-4-26B-A4B-it.yaml", "minVllmVersion": "0.25.0", "contextLength": 131072, "toolCallParser": "gemma4", "reasoningParser": "gemma4", "optInFeatures": ["text_only", "spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 64 }, "fp8": { "modelId": "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 32, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "nvfp4": { "modelId": "RedHatAI/gemma-4-26B-A4B-it-NVFP4", "precision": "nvfp4", "vramMinimumGb": 16, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "nvidia_nvfp4": { "modelId": "nvidia/Gemma-4-26B-A4B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 16 } }, "verifiedGpus": ["h100", "h200", "mi300x", "mi325x", "mi355x", "trillium", "ironwood", "xeon6", "dgx_spark_gb10", "dgx_station_gb300", "rtx_pro_6000", "rtx_5090", "arc_pro_b60", "arc_pro_b70"] }, "google/gemma-4-31B-it": { "file": "models/Google/gemma-4-31B-it.yaml", "minVllmVersion": "0.19.1", "contextLength": 262144, "toolCallParser": "gemma4", "reasoningParser": "gemma4", "optInFeatures": ["text_only", "spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 75 }, "fp8": { "modelId": "RedHatAI/gemma-4-31B-it-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 38, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "nvfp4": { "modelId": "nvidia/gemma-4-31B-it-NVFP4", "precision": "nvfp4", "vramMinimumGb": 19, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/gemma-4-31B-it-NVFP4", "precision": "nvfp4", "vramMinimumGb": 19, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "w4a16": { "modelId": "google/gemma-4-31B-it-qat-w4a16-ct", "precision": "int4", "vramMinimumGb": 20 }, "redhat_fp8": { "modelId": "RedHatAI/gemma-4-31B-it-FP8-block", "precision": "fp8", "vramMinimumGb": 38, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h100", "mi300x", "mi325x", "mi355x", "trillium", "ironwood", "arc_pro_b60", "arc_pro_b70"] }, "google/gemma-4-E2B-it": { "file": "models/Google/gemma-4-E2B-it.yaml", "minVllmVersion": "0.19.1", "contextLength": 131072, "toolCallParser": "gemma4", "reasoningParser": "gemma4", "optInFeatures": ["text_only", "spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 13 }, "fp8": { "modelId": "RedHatAI/gemma-4-E2B-it-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 6, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "w4a16": { "modelId": "google/gemma-4-E2B-it-qat-w4a16-ct", "precision": "int4", "vramMinimumGb": 8 }, "mobile": { "modelId": "google/gemma-4-E2B-it-qat-mobile-ct", "precision": "int2/4/8", "vramMinimumGb": 4 } }, "verifiedGpus": ["h100", "mi300x", "mi325x", "mi355x", "trillium", "ironwood", "xeon6"] }, "google/gemma-4-E4B-it": { "file": "models/Google/gemma-4-E4B-it.yaml", "minVllmVersion": "0.19.1", "contextLength": 131072, "toolCallParser": "gemma4", "reasoningParser": "gemma4", "optInFeatures": ["text_only", "spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 20 }, "fp8": { "modelId": "RedHatAI/gemma-4-E4B-it-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 10, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "w4a16": { "modelId": "google/gemma-4-E4B-it-qat-w4a16-ct", "precision": "int4", "vramMinimumGb": 10 }, "mobile": { "modelId": "google/gemma-4-E4B-it-qat-mobile-ct", "precision": "int2/4/8", "vramMinimumGb": 5 } }, "verifiedGpus": ["h100", "mi300x", "mi325x", "mi355x", "trillium", "ironwood", "xeon6"] }, "google/translategemma-27b-it": { "file": "models/Google/translategemma-27b-it.yaml", "minVllmVersion": "0.14.1", "contextLength": 131072, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 65 }, "vllm_optimized": { "modelId": "Infomaniak-AI/vllm-translategemma-27b-it", "precision": "bf16", "vramMinimumGb": 65 }, "small_4b": { "modelId": "google/translategemma-4b-it", "precision": "bf16", "vramMinimumGb": 10 } } }, "IFM/Jais-3-0.9B": { "file": "models/IFM/Jais-3-0.9B.yaml", "contextLength": 131072, "baseArgs": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16", "--max-model-len", "131072", "--gpu-memory-utilization", "0.85"], "toolCallParser": "k2_horizon", "reasoningParser": "k2_horizon", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 3 } }, "verifiedGpus": ["h200"] }, "IFM/Jais-3-3.7B": { "file": "models/IFM/Jais-3-3.7B.yaml", "contextLength": 524288, "baseArgs": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16"], "toolCallParser": "k2_horizon", "reasoningParser": "k2_horizon", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 13 } }, "verifiedGpus": ["h200"] }, "IFM/Jais-3-32B": { "file": "models/IFM/Jais-3-32B.yaml", "contextLength": 524288, "baseArgs": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16"], "toolCallParser": "k2_horizon", "reasoningParser": "k2_horizon", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 84 } }, "verifiedGpus": ["h200"] }, "IFM/Jais-3-375B-A23B": { "file": "models/IFM/Jais-3-375B-A23B.yaml", "contextLength": 524288, "baseArgs": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16"], "toolCallParser": "k2_horizon", "reasoningParser": "k2_horizon", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 911 } }, "verifiedGpus": ["h200", "mi355x"] }, "IFM/Jais-3-7B": { "file": "models/IFM/Jais-3-7B.yaml", "contextLength": 524288, "baseArgs": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16"], "toolCallParser": "k2_horizon", "reasoningParser": "k2_horizon", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 22 } }, "verifiedGpus": ["h200"] }, "IFM/Jais-3-MoVA-36B-A4B": { "file": "models/IFM/Jais-3-MoVA-36B-A4B.yaml", "contextLength": 524288, "baseArgs": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16", "--enable-expert-parallel"], "toolCallParser": "k2_horizon", "reasoningParser": "k2_horizon", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 90 } }, "verifiedGpus": ["h200"] }, "IFM/K2-Horizon-0.9B": { "file": "models/IFM/K2-Horizon-0.9B.yaml", "contextLength": 131072, "baseArgs": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16", "--max-model-len", "131072", "--gpu-memory-utilization", "0.85"], "toolCallParser": "k2_horizon", "reasoningParser": "k2_horizon", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 3 } }, "verifiedGpus": ["h200"] }, "IFM/K2-Horizon-3.7B": { "file": "models/IFM/K2-Horizon-3.7B.yaml", "contextLength": 524288, "baseArgs": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16"], "toolCallParser": "k2_horizon", "reasoningParser": "k2_horizon", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 13 } }, "verifiedGpus": ["h200"] }, "IFM/K2-Horizon-32B": { "file": "models/IFM/K2-Horizon-32B.yaml", "contextLength": 524288, "baseArgs": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16"], "toolCallParser": "k2_horizon", "reasoningParser": "k2_horizon", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 84 } }, "verifiedGpus": ["h200"] }, "IFM/K2-Horizon-375B-A23B": { "file": "models/IFM/K2-Horizon-375B-A23B.yaml", "contextLength": 524288, "baseArgs": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16"], "toolCallParser": "k2_horizon", "reasoningParser": "k2_horizon", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 911 } }, "verifiedGpus": ["h200", "mi355x"] }, "IFM/K2-Horizon-7B": { "file": "models/IFM/K2-Horizon-7B.yaml", "contextLength": 524288, "baseArgs": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16"], "toolCallParser": "k2_horizon", "reasoningParser": "k2_horizon", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 22 } }, "verifiedGpus": ["h200"] }, "IFM/K2-Horizon-MoVA-36B-A4B": { "file": "models/IFM/K2-Horizon-MoVA-36B-A4B.yaml", "contextLength": 524288, "baseArgs": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16", "--enable-expert-parallel"], "toolCallParser": "k2_horizon", "reasoningParser": "k2_horizon", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 90 } }, "verifiedGpus": ["h200"] }, "IndexTeam/IndexTTS-2.5": { "file": "models/IndexTeam/IndexTTS-2.5.yaml", "minVllmVersion": "0.27.0", "baseArgs": ["--trust-remote-code"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 96 } } }, "JetBrains/Mellum2-12B-A2.5B-Instruct": { "file": "models/JetBrains/Mellum2-12B-A2.5B-Instruct.yaml", "minVllmVersion": "0.23.0", "contextLength": 131072, "baseArgs": ["--max-model-len", "131072", "--trust-remote-code"], "toolCallParser": "hermes", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 29 } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "JetBrains/Mellum2-12B-A2.5B-Thinking": { "file": "models/JetBrains/Mellum2-12B-A2.5B-Thinking.yaml", "minVllmVersion": "0.23.0", "contextLength": 131072, "baseArgs": ["--max-model-len", "131072"], "toolCallParser": "hermes", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 29 }, "fp8": { "modelId": "RedHatAI/Mellum2-12B-A2.5B-Thinking-FP8-Dynamic", "precision": "fp8", "vramMinimumGb": 15, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "nvfp4": { "modelId": "RedHatAI/Mellum2-12B-A2.5B-Thinking-NVFP4", "precision": "nvfp4", "vramMinimumGb": 8 } }, "verifiedGpus": ["h200"] }, "Lightricks/LTX-2.5-Diffusers": { "file": "models/Lightricks/LTX-2.5-Diffusers.yaml", "minVllmVersion": "0.27.0", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 80 }, "fp8": { "precision": "fp8", "vramMinimumGb": 61, "extraArgs": ["--quantization", "fp8"] } }, "verifiedGpus": ["h200", "b300"] }, "LiquidAI/LFM2.5-1.2B-Base": { "file": "models/LiquidAI/LFM2.5-1.2B-Base.yaml", "minVllmVersion": "0.23.0", "contextLength": 32768, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 3 } }, "verifiedGpus": ["h100", "h200", "b200"] }, "LiquidAI/LFM2.5-1.2B-Instruct": { "file": "models/LiquidAI/LFM2.5-1.2B-Instruct.yaml", "minVllmVersion": "0.23.0", "contextLength": 32768, "toolCallParser": "lfm2", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 3 } }, "verifiedGpus": ["h100", "h200", "b200"] }, "LiquidAI/LFM2.5-1.2B-JP-202606": { "file": "models/LiquidAI/LFM2.5-1.2B-JP-202606.yaml", "minVllmVersion": "0.23.0", "contextLength": 32768, "toolCallParser": "lfm2", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 3 } }, "verifiedGpus": ["h100", "h200", "b200"] }, "LiquidAI/LFM2.5-1.2B-JP": { "file": "models/LiquidAI/LFM2.5-1.2B-JP.yaml", "minVllmVersion": "0.23.0", "contextLength": 32768, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 3 } }, "verifiedGpus": ["h100", "h200", "b200"] }, "LiquidAI/LFM2.5-1.2B-Thinking": { "file": "models/LiquidAI/LFM2.5-1.2B-Thinking.yaml", "minVllmVersion": "0.23.0", "contextLength": 32768, "toolCallParser": "lfm2", "reasoningParser": "qwen3", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 3 } }, "verifiedGpus": ["h100", "h200", "b200"] }, "LiquidAI/LFM2.5-230M": { "file": "models/LiquidAI/LFM2.5-230M.yaml", "minVllmVersion": "0.23.0", "contextLength": 32768, "toolCallParser": "lfm2", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 1 } }, "verifiedGpus": ["h100", "h200", "b200"] }, "LiquidAI/LFM2.5-350M": { "file": "models/LiquidAI/LFM2.5-350M.yaml", "minVllmVersion": "0.23.0", "contextLength": 32768, "toolCallParser": "lfm2", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 1 } }, "verifiedGpus": ["h100", "h200", "b200"] }, "LiquidAI/LFM2.5-8B-A1B": { "file": "models/LiquidAI/LFM2.5-8B-A1B.yaml", "minVllmVersion": "0.23.0", "contextLength": 128e3, "toolCallParser": "lfm2", "reasoningParser": "qwen3", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 21 } }, "verifiedGpus": ["h100", "h200", "b200"] }, "LiquidAI/LFM2.5-VL-1.6B": { "file": "models/LiquidAI/LFM2.5-VL-1.6B.yaml", "minVllmVersion": "0.23.0", "contextLength": 32768, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 4 } }, "verifiedGpus": ["h100", "h200", "b200"] }, "LiquidAI/LFM2.5-VL-450M": { "file": "models/LiquidAI/LFM2.5-VL-450M.yaml", "minVllmVersion": "0.23.0", "contextLength": 32768, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 2 } }, "verifiedGpus": ["h100", "h200", "b200"] }, "MiniMaxAI/MiniMax-H3": { "file": "models/MiniMaxAI/MiniMax-H3.yaml", "minVllmVersion": "0.26.0", "baseArgs": ["--trust-remote-code", "--host", "0.0.0.0", "--port", "8000", "--num-gpus", "4", "--usp", "4", "--ring", "1", "--vae-patch-parallel-size", "4", "--vae-parallel-mode", "tile", "--vae-use-tiling"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 202 } }, "verifiedGpus": ["gb200", "b300", "dgx_spark_gb10", "rtx_4090_2x", "rtx_pro_5000_4x", "rtx_5090_2x", "rtx_pro_6000_2x", "mi300x", "mi325x", "mi355x", "ascend_950pr"] }, "MiniMaxAI/MiniMax-M2.1": { "file": "models/MiniMaxAI/MiniMax-M2.1.yaml", "minVllmVersion": "0.11.1", "contextLength": 196608, "baseArgs": ["--trust-remote-code"], "toolCallParser": "minimax_m2", "reasoningParser": "minimax_m2", "variants": { "default": { "precision": "fp8", "vramMinimumGb": 276 } }, "verifiedGpus": ["h100", "mi300x", "mi325x", "mi355x"] }, "MiniMaxAI/MiniMax-M2.5": { "file": "models/MiniMaxAI/MiniMax-M2.5.yaml", "minVllmVersion": "0.20.2", "contextLength": 196608, "baseArgs": ["--trust-remote-code"], "toolCallParser": "minimax_m2", "reasoningParser": "minimax_m2", "variants": { "default": { "precision": "fp8", "vramMinimumGb": 276 }, "nvfp4": { "modelId": "nvidia/MiniMax-M2.5-NVFP4", "precision": "nvfp4", "vramMinimumGb": 138, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/MiniMax-M2.5-NVFP4", "precision": "nvfp4", "vramMinimumGb": 146 } }, "verifiedGpus": ["h100", "h200", "mi300x", "mi325x", "mi355x"] }, "MiniMaxAI/MiniMax-M2.7": { "file": "models/MiniMaxAI/MiniMax-M2.7.yaml", "minVllmVersion": "0.20.0", "contextLength": 196608, "baseArgs": ["--trust-remote-code"], "toolCallParser": "minimax_m2", "reasoningParser": "minimax_m2", "variants": { "default": { "precision": "fp8", "vramMinimumGb": 276 }, "nvfp4": { "modelId": "nvidia/MiniMax-M2.7-NVFP4", "precision": "nvfp4", "vramMinimumGb": 138, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h100", "h200", "mi300x", "mi325x", "mi355x", "dgx_station_gb300", "rtx_pro_6000_2x"] }, "MiniMaxAI/MiniMax-M2": { "file": "models/MiniMaxAI/MiniMax-M2.yaml", "minVllmVersion": "0.11.1", "contextLength": 196608, "baseArgs": ["--trust-remote-code"], "toolCallParser": "minimax_m2", "reasoningParser": "minimax_m2", "variants": { "default": { "precision": "fp8", "vramMinimumGb": 276 }, "nvfp4": { "modelId": "RedHatAI/MiniMax-M2-NVFP4", "precision": "nvfp4", "vramMinimumGb": 138, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h100", "h200", "mi300x", "mi325x", "mi355x"] }, "MiniMaxAI/MiniMax-M3": { "file": "models/MiniMaxAI/MiniMax-M3.yaml", "minVllmVersion": "0.24.0", "nightlyRequired": true, "contextLength": 1048576, "baseArgs": ["--block-size", "128"], "toolCallParser": "minimax_m3", "reasoningParser": "minimax_m3", "optInFeatures": ["spec_decoding", "text_only", "thinking_always_on", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 1025 }, "mxfp8": { "modelId": "MiniMaxAI/MiniMax-M3-MXFP8", "precision": "mxfp8", "vramMinimumGb": 513, "extraArgs": ["--attention_config.indexer_kv_dtype", "fp8"] }, "nvfp4": { "modelId": "nvidia/MiniMax-M3-NVFP4", "precision": "nvfp4", "vramMinimumGb": 257, "extraArgs": ["--trust-remote-code", "--kv-cache-dtype", "fp8", "--attention_config.backend", "FLASHINFER", "--attention_config.use_trtllm_attention", "true", "--attention_config.indexer_kv_dtype", "fp8", "--attention_config.minimax_m3_msa_decode_backend", "cutlass"] }, "mxfp4": { "modelId": "amd/MiniMax-M3-MXFP4", "precision": "mxfp4", "vramMinimumGb": 257 } }, "verifiedGpus": ["h200", "b200", "b300", "mi300x", "mi355x"] }, "OpenGVLab/InternVL3_5-8B": { "file": "models/OpenGVLab/InternVL3_5-8B.yaml", "minVllmVersion": "0.10.0", "contextLength": 40960, "baseArgs": ["--trust-remote-code"], "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 19 } }, "verifiedGpus": ["h100", "mi300x", "mi325x", "mi355x"] }, "OpenMOSS-Team/MOSS-Image-Base": { "file": "models/OpenMOSS-Team/MOSS-Image-Base.yaml", "minVllmVersion": "0.31.0", "baseArgs": ["--enforce-eager"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 33 }, "sft": { "modelId": "OpenMOSS-Team/MOSS-Image-SFT", "precision": "bf16", "vramMinimumGb": 33 } } }, "OpenMOSS-Team/MOSS-SoundEffect": { "file": "models/OpenMOSS-Team/MOSS-SoundEffect.yaml", "minVllmVersion": "0.22.0", "baseArgs": ["--omni", "--deploy-config", "vllm_omni/deploy/moss_sound_effect.yaml"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 32 } } }, "OpenMOSS-Team/MOSS-TTS-Realtime": { "file": "models/OpenMOSS-Team/MOSS-TTS-Realtime.yaml", "minVllmVersion": "0.22.0", "baseArgs": ["--omni"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 16 } } }, "OpenMOSS-Team/MOSS-TTS": { "file": "models/OpenMOSS-Team/MOSS-TTS.yaml", "minVllmVersion": "0.22.0", "baseArgs": ["--omni", "--deploy-config", "vllm_omni/deploy/moss_tts.yaml"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 32 } } }, "OpenMOSS-Team/MOSS-TTSD-v1.0": { "file": "models/OpenMOSS-Team/MOSS-TTSD-v1.0.yaml", "minVllmVersion": "0.22.0", "baseArgs": ["--omni", "--deploy-config", "vllm_omni/deploy/moss_ttsd.yaml"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 32 } } }, "OpenMOSS-Team/MOSS-Transcribe-Diarize": { "file": "models/OpenMOSS-Team/MOSS-Transcribe-Diarize.yaml", "minVllmVersion": "0.27.0", "baseArgs": ["--trust-remote-code"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 4 } } }, "OpenMOSS-Team/MOSS-VoiceGenerator": { "file": "models/OpenMOSS-Team/MOSS-VoiceGenerator.yaml", "minVllmVersion": "0.22.0", "baseArgs": ["--omni", "--deploy-config", "vllm_omni/deploy/moss_voice_generator.yaml"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 16 } } }, "PaddlePaddle/PaddleOCR-VL-1.5": { "file": "models/PaddlePaddle/PaddleOCR-VL-1.5.yaml", "minVllmVersion": "0.11.1", "contextLength": 131072, "baseArgs": ["--trust-remote-code", "--max-num-batched-tokens", "16384", "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"], "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 2 } } }, "PaddlePaddle/PaddleOCR-VL-1.6": { "file": "models/PaddlePaddle/PaddleOCR-VL-1.6.yaml", "minVllmVersion": "0.11.1", "contextLength": 131072, "baseArgs": ["--trust-remote-code", "--max-num-batched-tokens", "16384", "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"], "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 2 } } }, "PaddlePaddle/PaddleOCR-VL": { "file": "models/PaddlePaddle/PaddleOCR-VL.yaml", "minVllmVersion": "0.11.1", "contextLength": 131072, "baseArgs": ["--trust-remote-code", "--max-num-batched-tokens", "16384", "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"], "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 2 } } }, "Qwen/QwQ-32B": { "file": "models/Qwen/QwQ-32B.yaml", "minVllmVersion": "0.7.3", "contextLength": 40960, "toolCallParser": "hermes", "reasoningParser": "deepseek_r1", "optInFeatures": ["long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 77 }, "w8a8_int8": { "modelId": "RedHatAI/QwQ-32B-quantized.w8a8", "precision": "int8", "vramMinimumGb": 39 }, "awq": { "modelId": "Qwen/QwQ-32B-AWQ", "precision": "int4", "vramMinimumGb": 20, "extraArgs": ["--quantization", "awq"] } }, "verifiedGpus": ["xeon6"] }, "Qwen/Qwen-Image-2.1": { "file": "models/Qwen/Qwen-Image-2.1.yaml", "minVllmVersion": "0.29.0", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 40 }, "fp8": { "precision": "fp8", "vramMinimumGb": 40, "extraArgs": ["--diffusion-quantization-config", '{"transformer":{"method":"fp8","ignored_layers":["img_mlp"]}}'] } }, "verifiedGpus": ["gb300"] }, "Qwen/Qwen-Image": { "file": "models/Qwen/Qwen-Image.yaml", "minVllmVersion": "0.18.0", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 48 }, "fp8": { "precision": "fp8", "vramMinimumGb": 24, "extraArgs": ["--quantization", "fp8", "--ignored-layers", "img_mlp"] }, "int8": { "precision": "int8", "vramMinimumGb": 24, "extraArgs": ["--quantization", "int8"] } } }, "Qwen/Qwen2.5-32B": { "file": "models/Qwen/Qwen2.5-32B.yaml", "minVllmVersion": "0.6.2", "contextLength": 131072, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 77 }, "fp8": { "modelId": "RedHatAI/Qwen2.5-32B-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 39, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["trillium"] }, "Qwen/Qwen2.5-VL-72B-Instruct": { "file": "models/Qwen/Qwen2.5-VL-72B-Instruct.yaml", "minVllmVersion": "0.10.2", "contextLength": 128e3, "optInFeatures": ["text_only"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 173 }, "awq": { "modelId": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "precision": "int4", "vramMinimumGb": 43, "extraArgs": ["--quantization", "awq"] }, "fp8": { "modelId": "RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 87, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["mi300x", "mi325x", "mi355x"] }, "Qwen/Qwen2.5-VL-7B-Instruct": { "file": "models/Qwen/Qwen2.5-VL-7B-Instruct.yaml", "minVllmVersion": "0.10.2", "contextLength": 128e3, "optInFeatures": ["text_only"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 17 }, "awq": { "modelId": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", "precision": "int4", "vramMinimumGb": 5, "extraArgs": ["--quantization", "awq"] }, "fp8": { "modelId": "RedHatAI/Qwen2.5-VL-7B-Instruct-FP8-Dynamic", "precision": "fp8", "vramMinimumGb": 9, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["trillium", "xeon6"] }, "Qwen/Qwen3-1.7B": { "file": "models/Qwen/Qwen3-1.7B.yaml", "minVllmVersion": "0.8.5", "contextLength": 40960, "toolCallParser": "hermes", "reasoningParser": "qwen3", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 5 } }, "verifiedGpus": ["xeon6"] }, "Qwen/Qwen3-14B": { "file": "models/Qwen/Qwen3-14B.yaml", "minVllmVersion": "0.8.5", "contextLength": 40960, "toolCallParser": "hermes", "reasoningParser": "qwen3", "optInFeatures": ["reasoning", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 34 }, "fp8": { "modelId": "Qwen/Qwen3-14B-FP8", "precision": "fp8", "vramMinimumGb": 17 }, "awq": { "modelId": "Qwen/Qwen3-14B-AWQ", "precision": "int4", "vramMinimumGb": 10, "extraArgs": ["--quantization", "awq"] } }, "verifiedGpus": ["xeon6"] }, "Qwen/Qwen3-235B-A22B-Instruct-2507": { "file": "models/Qwen/Qwen3-235B-A22B-Instruct-2507.yaml", "minVllmVersion": "0.10.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "hermes", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 564 }, "fp8": { "modelId": "Qwen/Qwen3-235B-A22B-FP8", "precision": "fp8", "vramMinimumGb": 240 }, "nvfp4": { "modelId": "nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4", "precision": "nvfp4", "vramMinimumGb": 141, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/Qwen3-235B-A22B-Instruct-2507-NVFP4", "precision": "nvfp4", "vramMinimumGb": 152 } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "Qwen/Qwen3-30B-A3B": { "file": "models/Qwen/Qwen3-30B-A3B.yaml", "minVllmVersion": "0.8.5", "contextLength": 40960, "toolCallParser": "hermes", "reasoningParser": "qwen3", "optInFeatures": ["long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 74 }, "ascend_w8a8": { "modelId": "Eco-Tech/Qwen3-30B-A3B-w8a8", "precision": "int8", "vramMinimumGb": 37, "extraArgs": ["--trust-remote-code", "--max-num-seqs", "100", "--max-model-len", "40960", "--max-num-batched-tokens", "16384", "--enable-expert-parallel", "--quantization", "ascend", "--distributed_executor_backend", "mp", "--no-enable-prefix-caching", "--async-scheduling", "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}', "--additional-config", '{"enable_flashcomm1": true, "weight_nz_mode": 2}', "--gpu-memory-utilization", "0.95"] } }, "verifiedGpus": ["xeon6", "ascend_910c", "arc_pro_b70"] }, "Qwen/Qwen3-32B": { "file": "models/Qwen/Qwen3-32B.yaml", "minVllmVersion": "0.8.5", "contextLength": 40960, "toolCallParser": "hermes", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 77 }, "fp8": { "modelId": "Qwen/Qwen3-32B-FP8", "precision": "fp8", "vramMinimumGb": 39 }, "awq": { "modelId": "Qwen/Qwen3-32B-AWQ", "precision": "int4", "vramMinimumGb": 20, "extraArgs": ["--quantization", "awq"] }, "redhat_fp8": { "modelId": "RedHatAI/Qwen3-32B-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 39, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8_block": { "modelId": "RedHatAI/Qwen3-32B-FP8-block", "precision": "fp8", "vramMinimumGb": 39, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "nvfp4": { "modelId": "RedHatAI/Qwen3-32B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 22 }, "redhat_nvfp4": { "modelId": "RedHatAI/Qwen3-32B-NVFP4A16", "precision": "nvfp4", "vramMinimumGb": 22 } }, "verifiedGpus": ["trillium", "ironwood", "arc_pro_b60", "arc_pro_b70"] }, "Qwen/Qwen3-4B": { "file": "models/Qwen/Qwen3-4B.yaml", "minVllmVersion": "0.8.5", "contextLength": 40960, "toolCallParser": "hermes", "reasoningParser": "qwen3", "optInFeatures": ["long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 10 }, "fp8": { "modelId": "Qwen/Qwen3-4B-FP8", "precision": "fp8", "vramMinimumGb": 5 }, "redhat_fp8": { "modelId": "RedHatAI/Qwen3-4B-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 5, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["trillium", "xeon6"] }, "Qwen/Qwen3-8B": { "file": "models/Qwen/Qwen3-8B.yaml", "minVllmVersion": "0.8.5", "contextLength": 40960, "toolCallParser": "hermes", "reasoningParser": "qwen3", "optInFeatures": ["long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 20 } }, "verifiedGpus": ["xeon6"] }, "Qwen/Qwen3-ASR-1.7B": { "file": "models/Qwen/Qwen3-ASR-1.7B.yaml", "minVllmVersion": "0.16.0", "contextLength": 65536, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 4 } } }, "Qwen/Qwen3-Coder-480B-A35B-Instruct": { "file": "models/Qwen/Qwen3-Coder-480B-A35B-Instruct.yaml", "minVllmVersion": "0.10.0", "contextLength": 262144, "toolCallParser": "qwen3_coder", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 1152 }, "fp8": { "modelId": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", "precision": "fp8", "vramMinimumGb": 576 }, "nvfp4": { "modelId": "nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4", "precision": "nvfp4", "vramMinimumGb": 288, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8": { "modelId": "RedHatAI/Qwen3-Coder-480B-A35B-Instruct-FP8", "precision": "fp8", "vramMinimumGb": 576, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x", "ironwood"] }, "Qwen/Qwen3-Next-80B-A3B-Instruct": { "file": "models/Qwen/Qwen3-Next-80B-A3B-Instruct.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "toolCallParser": "hermes", "optInFeatures": ["spec_decoding", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 192 }, "fp8": { "modelId": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8", "precision": "fp8", "vramMinimumGb": 96 }, "nvfp4": { "modelId": "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4", "precision": "nvfp4", "vramMinimumGb": 48, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8": { "modelId": "RedHatAI/Qwen3-Next-80B-A3B-Instruct-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 96, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8_plain": { "modelId": "RedHatAI/Qwen3-Next-80B-A3B-Instruct-FP8", "precision": "fp8", "vramMinimumGb": 96, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8_block": { "modelId": "RedHatAI/Qwen3-Next-80B-A3B-Instruct-FP8-block", "precision": "fp8", "vramMinimumGb": 96, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/Qwen3-Next-80B-A3B-Instruct-NVFP4", "precision": "nvfp4", "vramMinimumGb": 48 } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": { "file": "models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice.yaml", "minVllmVersion": "0.20.0", "baseArgs": ["--omni"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 16 }, "voicedesign-1.7b": { "modelId": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "precision": "bf16", "vramMinimumGb": 16 }, "base-1.7b": { "modelId": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "precision": "bf16", "vramMinimumGb": 16 }, "customvoice-0.6b": { "modelId": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "precision": "bf16", "vramMinimumGb": 8 }, "base-0.6b": { "modelId": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "precision": "bf16", "vramMinimumGb": 8 } } }, "Qwen/Qwen3-VL-235B-A22B-Instruct": { "file": "models/Qwen/Qwen3-VL-235B-A22B-Instruct.yaml", "minVllmVersion": "0.11.0", "contextLength": 262144, "optInFeatures": ["text_only", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 564 }, "fp8": { "modelId": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8", "precision": "fp8", "vramMinimumGb": 282 }, "nvfp4": { "modelId": "nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4", "precision": "nvfp4", "vramMinimumGb": 141, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8": { "modelId": "RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 282, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8_block": { "modelId": "RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-block", "precision": "fp8", "vramMinimumGb": 282, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4", "precision": "nvfp4", "vramMinimumGb": 150 } }, "verifiedGpus": ["h100", "h200", "b200", "mi300x", "mi325x", "mi355x"] }, "Qwen/Qwen3-VL-30B-A3B-Instruct": { "file": "models/Qwen/Qwen3-VL-30B-A3B-Instruct.yaml", "minVllmVersion": "0.11.0", "contextLength": 262144, "optInFeatures": ["text_only"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 72 } }, "verifiedGpus": ["xeon6"] }, "Qwen/Qwen3.5-0.8B": { "file": "models/Qwen/Qwen3.5-0.8B.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["text_only"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 2 } }, "verifiedGpus": ["arc_pro_b60", "arc_pro_b70"] }, "Qwen/Qwen3.5-122B-A10B": { "file": "models/Qwen/Qwen3.5-122B-A10B.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "text_only", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 293 }, "fp8": { "modelId": "Qwen/Qwen3.5-122B-A10B-FP8", "precision": "fp8", "vramMinimumGb": 147 }, "nvfp4": { "modelId": "RedHatAI/Qwen3.5-122B-A10B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 74, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "gptq_int4": { "modelId": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4", "precision": "int4", "vramMinimumGb": 74 }, "redhat_fp8": { "modelId": "RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 147, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["mi325x", "mi355x"] }, "Qwen/Qwen3.5-27B": { "file": "models/Qwen/Qwen3.5-27B.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "text_only", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 65 }, "fp8": { "modelId": "Qwen/Qwen3.5-27B-FP8", "precision": "fp8", "vramMinimumGb": 33 }, "gptq_int4": { "modelId": "Qwen/Qwen3.5-27B-GPTQ-Int4", "precision": "int4", "vramMinimumGb": 17 } }, "verifiedGpus": ["arc_pro_b60", "arc_pro_b70"] }, "Qwen/Qwen3.5-2B": { "file": "models/Qwen/Qwen3.5-2B.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "text_only"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 5 } }, "verifiedGpus": ["arc_pro_b60", "arc_pro_b70"] }, "Qwen/Qwen3.5-35B-A3B": { "file": "models/Qwen/Qwen3.5-35B-A3B.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "text_only", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 84 }, "fp8": { "modelId": "Qwen/Qwen3.5-35B-A3B-FP8", "precision": "fp8", "vramMinimumGb": 42 }, "gptq_int4": { "modelId": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", "precision": "int4", "vramMinimumGb": 21 }, "redhat_fp8": { "modelId": "RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 42, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["xeon6", "arc_pro_b60", "arc_pro_b70"] }, "Qwen/Qwen3.5-397B-A17B": { "file": "models/Qwen/Qwen3.5-397B-A17B.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "text_only", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 953 }, "nvfp4": { "modelId": "nvidia/Qwen3.5-397B-A17B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 238, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "fp8": { "modelId": "RedHatAI/Qwen3.5-397B-A17B-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 477, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "gptq_int4": { "modelId": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4", "precision": "int4", "vramMinimumGb": 239 }, "amd_mxfp4": { "modelId": "amd/Qwen3.5-397B-A17B-MXFP4", "precision": "mxfp4", "vramMinimumGb": 287, "extraArgs": ["--attention-backend", "ROCM_AITER_FA", "--max-num-seqs", "64"] }, "ascend_w8a8_mtp": { "modelId": "Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp", "precision": "int8", "vramMinimumGb": 477, "extraArgs": ["--distributed-executor-backend", "mp", "--data-parallel-size", "1", "--quantization", "ascend", "--served-model-name", "qwen3.5", "--max-num-seqs", "128", "--max-model-len", "133000", "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.95", "--enable-prefix-caching", "--async-scheduling", "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}', "--additional-config", '{"enable_cpu_binding":true}'] }, "ascend_w4a4_mxfp4": { "modelId": "Eco-Tech/Qwen3.5-397B-A17B-w4a4-mxfp4", "precision": "mxfp4", "vramMinimumGb": 239, "extraArgs": ["--distributed-executor-backend", "mp", "--data-parallel-size", "1", "--quantization", "ascend", "--served-model-name", "qwen3.5", "--max-num-seqs", "128", "--max-model-len", "133000", "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.95", "--enable-prefix-caching", "--async-scheduling", "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}', "--additional-config", '{"enable_cpu_binding":true}'] } }, "verifiedGpus": ["h200", "gb200", "mi300x", "mi325x", "mi350x", "mi355x", "ascend_950dt"] }, "Qwen/Qwen3.5-4B": { "file": "models/Qwen/Qwen3.5-4B.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "text_only", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 10 }, "fp8": { "modelId": "RedHatAI/Qwen3.5-4B-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 5, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["xeon6", "arc_pro_b60", "arc_pro_b70"] }, "Qwen/Qwen3.5-9B": { "file": "models/Qwen/Qwen3.5-9B.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "text_only", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 22 }, "fp8": { "modelId": "RedHatAI/Qwen3.5-9B-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 11, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["arc_pro_b60", "arc_pro_b70"] }, "Qwen/Qwen3.6-27B": { "file": "models/Qwen/Qwen3.6-27B.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "text_only", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 65 }, "fp8": { "modelId": "Qwen/Qwen3.6-27B-FP8", "precision": "fp8", "vramMinimumGb": 33, "extraArgs": ["--max-num-seqs", "512"] }, "nvfp4": { "modelId": "nvidia/Qwen3.6-27B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 27 }, "redhat_fp8": { "modelId": "RedHatAI/Qwen3.6-27B-FP8", "precision": "fp8", "vramMinimumGb": 33 }, "ascend_w8a8_mxfp8": { "modelId": "Eco-Tech/Qwen3.6-27B-w8a8-mxfp8", "precision": "mxfp8", "vramMinimumGb": 34, "extraArgs": ["--distributed-executor-backend", "mp", "--data-parallel-size", "8", "--quantization", "ascend", "--served-model-name", "qwen3.6", "--max-num-seqs", "32", "--max-model-len", "262144", "--max-num-batched-tokens", "16384", "--gpu-memory-utilization", "0.90", "--no-enable-prefix-caching", "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}', "--additional-config", '{"enable_cpu_binding":true}'] }, "ascend_w8a8": { "modelId": "Eco-Tech/Qwen3.6-27B-w8a8", "precision": "int8", "vramMinimumGb": 33, "extraArgs": ["--data-parallel-size", "8", "--quantization", "ascend", "--served-model-name", "qwen3.6", "--max-num-seqs", "32", "--max-model-len", "133000", "--max-num-batched-tokens", "16384", "--gpu-memory-utilization", "0.90", "--no-enable-prefix-caching", "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}', "--additional-config", '{"enable_cpu_binding":true}', "--async-scheduling"] } }, "verifiedGpus": ["dgx_spark_gb10", "rtx_pro_6000", "dgx_station_gb300", "rtx_5090", "ascend_950dt", "ascend_910c", "arc_pro_b60", "arc_pro_b70", "mi300x"] }, "Qwen/Qwen3.6-35B-A3B": { "file": "models/Qwen/Qwen3.6-35B-A3B.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "text_only", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 84 }, "fp8": { "modelId": "Qwen/Qwen3.6-35B-A3B-FP8", "precision": "fp8", "vramMinimumGb": 42 }, "nvfp4": { "modelId": "nvidia/Qwen3.6-35B-A3B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 21 }, "redhat_nvfp4": { "modelId": "RedHatAI/Qwen3.6-35B-A3B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 21, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8": { "modelId": "RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 42, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8_plain": { "modelId": "RedHatAI/Qwen3.6-35B-A3B-FP8", "precision": "fp8", "vramMinimumGb": 42, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["dgx_spark_gb10", "h100", "h200", "mi300x", "mi325x", "mi355x", "rtx_pro_6000", "dgx_station_gb300", "rtx_5090", "arc_pro_b60", "arc_pro_b70"] }, "Qwen/Qwen3.8-2.4T-A95B": { "file": "models/Qwen/Qwen3.8-2.4T-A95B.yaml", "minVllmVersion": "nightly", "nightlyRequired": true, "contextLength": 262144, "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 5871 }, "fp8": { "modelId": "Qwen/Qwen3.8-2.4T-A95B-FP8", "precision": "fp8", "vramMinimumGb": 2996 }, "mxfp4": { "modelId": "Inferact/Qwen3.8-2.4T-A95B-MXFP4", "precision": "mxfp4", "vramMinimumGb": 1917 }, "nvfp4": { "modelId": "Inferact/Qwen3.8-2.4T-A95B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 1737, "extraArgs": ["--linear-backend", "flashinfer_cutedsl"] }, "nvfp4_reap25": { "modelId": "RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-25", "precision": "nvfp4", "vramMinimumGb": 1373 } }, "verifiedGpus": ["b300", "gb300"] }, "Qwen/Qwen3.8-27B": { "file": "models/Qwen/Qwen3.8-27B.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "text_only", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 67 }, "fp8": { "modelId": "Qwen/Qwen3.8-27B-FP8", "precision": "fp8", "vramMinimumGb": 38 }, "nvfp4": { "modelId": "Inferact/Qwen3.8-27B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 32 }, "nvfp4_nvidia": { "modelId": "nvidia/Qwen3.8-27B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 32 }, "int4": { "modelId": "RedHatAI/Qwen3.8-27B-INT4", "precision": "int4", "vramMinimumGb": 24 }, "mxfp4": { "modelId": "RedHatAI/Qwen3.8-27B-MXFP4", "precision": "mxfp4", "vramMinimumGb": 24 }, "ascend_w8a8": { "modelId": "Eco-Tech/Qwen3.8-27B-w8a8", "precision": "int8", "vramMinimumGb": 39, "extraArgs": ["--quantization", "ascend", "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}', "--additional-config", '{"enable_cpu_binding":true}', "--max-num-batched-tokens", "16384"] } }, "verifiedGpus": ["gb300", "b200", "rtx_5090", "rtx_5090_2x", "rtx_pro_6000", "dgx_spark_gb10", "ascend_950pr", "arc_pro_b60", "arc_pro_b70"] }, "Qwen/Qwen3.8-Flash-Next": { "file": "models/Qwen/Qwen3.8-Flash-Next.yaml", "minVllmVersion": "0.30.0", "nightlyRequired": true, "contextLength": 262144, "baseArgs": ["--max-num-seqs", "256", "--gpu-memory-utilization", "0.90", "--enable-prefix-caching"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["text_only", "long_context"], "variants": { "default": { "modelId": "Inferact/Qwen3.8-Flash-Next-NVFP4", "precision": "nvfp4", "vramMinimumGb": 130 }, "fp8": { "modelId": "Qwen/Qwen3.8-Flash-Next-FP8", "precision": "fp8", "vramMinimumGb": 250 }, "bf16": { "modelId": "Qwen/Qwen3.8-Flash-Next", "precision": "bf16", "vramMinimumGb": 423 } }, "verifiedGpus": ["h200", "h100", "gb200", "gb300", "mi355x", "rtx_pro_6000_4x", "dgx_station_gb300"] }, "Qwen/Qwen3Guard-Gen-8B": { "file": "models/Qwen/Qwen3Guard-Gen-8B.yaml", "minVllmVersion": "0.10.0", "contextLength": 32768, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 19 }, "small_4b": { "modelId": "Qwen/Qwen3Guard-Gen-4B", "precision": "bf16", "vramMinimumGb": 10 }, "tiny_0_6b": { "modelId": "Qwen/Qwen3Guard-Gen-0.6B", "precision": "bf16", "vramMinimumGb": 4 } }, "verifiedGpus": ["mi300x"] }, "Wan-AI/Wan2.2-T2V-A14B-Diffusers": { "file": "models/Wan-AI/Wan2.2-T2V-A14B-Diffusers.yaml", "minVllmVersion": "0.12.0", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 152 } } }, "XiaomiMiMo/MiMo-V2-Flash": { "file": "models/XiaomiMiMo/MiMo-V2-Flash.yaml", "minVllmVersion": "0.20.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code", "--generation-config", "vllm"], "toolCallParser": "mimo", "reasoningParser": "qwen3", "variants": { "default": { "precision": "fp8", "vramMinimumGb": 371, "extraArgs": ["--tensor-parallel-size", "4", "--gpu-memory-utilization", "0.9"] } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "XiaomiMiMo/MiMo-V2.5-Pro": { "file": "models/XiaomiMiMo/MiMo-V2.5-Pro.yaml", "minVllmVersion": "0.21.0", "contextLength": 1048576, "baseArgs": ["--trust-remote-code", "--max-model-len auto", "--generation-config", "vllm"], "toolCallParser": "mimo", "reasoningParser": "mimo", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 1224, "extraArgs": ["--tensor-parallel-size", "8", "--gpu-memory-utilization", "0.95", "--max-model-len", "auto"] } }, "verifiedGpus": ["h200"] }, "XiaomiMiMo/MiMo-V2.5": { "file": "models/XiaomiMiMo/MiMo-V2.5.yaml", "minVllmVersion": "0.21.0", "contextLength": 1048576, "baseArgs": ["--trust-remote-code", "--generation-config", "vllm"], "toolCallParser": "mimo", "reasoningParser": "mimo", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 374, "extraArgs": ["--gpu-memory-utilization", "0.95", "--max-model-len", "auto"] } }, "verifiedGpus": ["h200"] }, "XiaomiMiMo/MiMo-V2.6-Flash-RL": { "file": "models/XiaomiMiMo/MiMo-V2.6-Flash-RL.yaml", "minVllmVersion": "0.31.0", "contextLength": 1048576, "baseArgs": ["--trust-remote-code", "--max-model-len", "auto", "--generation-config", "vllm"], "toolCallParser": "mimo", "reasoningParser": "mimo", "optInFeatures": ["spec_decoding"], "variants": { "default": { "modelId": "XiaomiMiMo/MiMo-V2.6-Flash-MOPD", "precision": "fp8", "vramMinimumGb": 208, "extraArgs": ["--tensor-parallel-size", "4", "--gpu-memory-utilization", "0.95"] }, "rl": { "precision": "fp8", "vramMinimumGb": 208, "extraArgs": ["--tensor-parallel-size", "4", "--gpu-memory-utilization", "0.95"] } }, "verifiedGpus": ["h200", "gb300", "mi300x", "mi325x", "mi355x"] }, "XiaomiMiMo/MiMo-V2.6-Pro-RL": { "file": "models/XiaomiMiMo/MiMo-V2.6-Pro-RL.yaml", "minVllmVersion": "0.31.0", "contextLength": 1048576, "baseArgs": ["--trust-remote-code", "--max-model-len", "auto", "--generation-config", "vllm"], "toolCallParser": "mimo", "reasoningParser": "mimo", "optInFeatures": ["spec_decoding"], "variants": { "default": { "modelId": "XiaomiMiMo/MiMo-V2.6-Pro-MOPD", "precision": "fp8", "vramMinimumGb": 680, "extraArgs": ["--tensor-parallel-size", "8", "--gpu-memory-utilization", "0.95"] }, "rl": { "precision": "fp8", "vramMinimumGb": 680, "extraArgs": ["--tensor-parallel-size", "8", "--gpu-memory-utilization", "0.95"] } }, "verifiedGpus": ["h200", "gb300", "mi355x"] }, "arcee-ai/Trinity-Large-Thinking": { "file": "models/arcee-ai/Trinity-Large-Thinking.yaml", "minVllmVersion": "0.11.1", "contextLength": 262144, "baseArgs": ["--dtype", "bfloat16"], "toolCallParser": "qwen3_coder", "reasoningParser": "deepseek_r1", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 955 }, "nvfp4": { "modelId": "arcee-ai/Trinity-Large-Thinking-NVFP4", "precision": "nvfp4", "vramMinimumGb": 239, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/Trinity-Large-Thinking-NVFP4", "precision": "nvfp4", "vramMinimumGb": 254 } } }, "baidu/ERNIE-4.5-21B-A3B-PT": { "file": "models/baidu/ERNIE-4.5-21B-A3B-PT.yaml", "minVllmVersion": "0.10.1", "contextLength": 131072, "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 106 }, "300b": { "modelId": "baidu/ERNIE-4.5-300B-A47B-PT", "precision": "bf16", "vramMinimumGb": 640, "extraArgs": ["--tensor-parallel-size", "8", "--gpu-memory-utilization", "0.95", "--quantization", "fp8"] } } }, "baidu/ERNIE-4.5-VL-28B-A3B-PT": { "file": "models/baidu/ERNIE-4.5-VL-28B-A3B-PT.yaml", "minVllmVersion": "0.11.0", "contextLength": 131072, "baseArgs": ["--trust-remote-code"], "optInFeatures": ["text_only"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 67 }, "424b": { "modelId": "baidu/ERNIE-4.5-VL-424B-A47B-PT", "precision": "bf16", "vramMinimumGb": 1120, "extraArgs": ["--trust-remote-code", "--tensor-parallel-size", "8"] }, "424b_fp8": { "modelId": "baidu/ERNIE-4.5-VL-424B-A47B-PT", "precision": "fp8", "vramMinimumGb": 640, "extraArgs": ["--trust-remote-code", "--tensor-parallel-size", "8", "--quantization", "fp8", "--cpu-offload-gb", "50"] } }, "verifiedGpus": ["mi300x", "mi325x", "mi355x"] }, "baidu/Unlimited-OCR": { "file": "models/baidu/Unlimited-OCR.yaml", "minVllmVersion": "0.25.0", "contextLength": 32768, "baseArgs": ["--trust-remote-code", "--logits_processors", "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor", "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"], "optInFeatures": ["encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 8 } } }, "bharatgenai/Param2-17B-A2.4B-Thinking": { "file": "models/bharatgenai/Param2-17B-A2.4B-Thinking.yaml", "minVllmVersion": "0.20.0", "contextLength": 32768, "baseArgs": ["--trust-remote-code", "--async-scheduling"], "toolCallParser": "hermes", "reasoningParser": "qwen3", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 41 } }, "verifiedGpus": ["h100", "l40s"] }, "black-forest-labs/flux-3-image-klein": { "file": "models/black-forest-labs/flux-3-image-klein.yaml", "minVllmVersion": "0.32.0", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 24 }, "fp8r": { "precision": "fp8", "vramMinimumGb": 17, "extraArgs": ["--stage-overrides", '{"0":{"additional_config":{"variant":"fp8r"}}}'] }, "nvfp4": { "precision": "nvfp4", "vramMinimumGb": 13, "extraArgs": ["--stage-overrides", '{"0":{"additional_config":{"variant":"nvfp4"}}}'] } }, "verifiedGpus": ["gb200"] }, "bosonai/higgs-audio-v3-tts-4b": { "file": "models/bosonai/higgs-audio-v3-tts-4b.yaml", "minVllmVersion": "0.22.0", "baseArgs": ["--omni", "--trust-remote-code"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 24 } } }, "deepseek-ai/DeepSeek-OCR-2": { "file": "models/deepseek-ai/DeepSeek-OCR-2.yaml", "minVllmVersion": "0.16.0", "contextLength": 8192, "baseArgs": ["--trust-remote-code", "--logits_processors", "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor", "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"], "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 7 } } }, "deepseek-ai/DeepSeek-OCR": { "file": "models/deepseek-ai/DeepSeek-OCR.yaml", "minVllmVersion": "0.12.0", "contextLength": 8192, "baseArgs": ["--trust-remote-code", "--logits-processors", "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor", "--mm-processor-cache-gb", "0"], "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 7 } }, "verifiedGpus": ["mi300x", "mi325x", "mi355x", "arc_pro_b60", "arc_pro_b70"] }, "deepseek-ai/DeepSeek-R1": { "file": "models/deepseek-ai/DeepSeek-R1.yaml", "minVllmVersion": "0.12.0", "contextLength": 163840, "baseArgs": ["--trust-remote-code"], "toolCallParser": "deepseek_v3", "reasoningParser": "deepseek_r1", "variants": { "default": { "precision": "fp8", "vramMinimumGb": 805 }, "r1_0528": { "modelId": "deepseek-ai/DeepSeek-R1-0528", "precision": "fp8", "vramMinimumGb": 805 }, "nvfp4": { "modelId": "nvidia/DeepSeek-R1-0528-NVFP4-v2", "precision": "nvfp4", "vramMinimumGb": 403, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "mxfp4": { "modelId": "amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4", "precision": "mxfp4", "vramMinimumGb": 403, "extraArgs": ["--no-enable-expert-parallel", "--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/DeepSeek-R1-NVFP4-FP8-BLOCK", "precision": "nvfp4", "vramMinimumGb": 444 } }, "verifiedGpus": ["h200", "b200", "gb200", "mi300x", "mi325x", "mi350x", "mi355x"] }, "deepseek-ai/DeepSeek-V3.1": { "file": "models/deepseek-ai/DeepSeek-V3.1.yaml", "minVllmVersion": "0.12.0", "contextLength": 163840, "baseArgs": ["--trust-remote-code", "--enable-expert-parallel"], "toolCallParser": "deepseek_v31", "reasoningParser": "deepseek_v3", "variants": { "default": { "precision": "fp8", "vramMinimumGb": 805 }, "nvfp4": { "modelId": "nvidia/DeepSeek-V3.1-NVFP4", "precision": "nvfp4", "vramMinimumGb": 403, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "deepseek-ai/DeepSeek-V3.2-Exp": { "file": "models/deepseek-ai/DeepSeek-V3.2-Exp.yaml", "minVllmVersion": "0.12.0", "contextLength": 163840, "baseArgs": ["--trust-remote-code"], "toolCallParser": "deepseek_v32", "reasoningParser": "deepseek_v3", "variants": { "default": { "precision": "fp8", "vramMinimumGb": 805 } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "deepseek-ai/DeepSeek-V3.2": { "file": "models/deepseek-ai/DeepSeek-V3.2.yaml", "minVllmVersion": "0.18.0", "contextLength": 163840, "baseArgs": ["--trust-remote-code", "--kernel-config.enable_flashinfer_autotune=False"], "toolCallParser": "deepseek_v32", "reasoningParser": "deepseek_v3", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 805 }, "nvfp4": { "modelId": "nvidia/DeepSeek-V3.2-NVFP4", "precision": "nvfp4", "vramMinimumGb": 403, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/DeepSeek-V3.2-NVFP4-FP8-BLOCK", "precision": "nvfp4", "vramMinimumGb": 430 } }, "verifiedGpus": ["h100", "h200"] }, "deepseek-ai/DeepSeek-V3": { "file": "models/deepseek-ai/DeepSeek-V3.yaml", "minVllmVersion": "0.12.0", "contextLength": 163840, "baseArgs": ["--trust-remote-code", "--enable-expert-parallel"], "toolCallParser": "deepseek_v3", "reasoningParser": "deepseek_v3", "variants": { "default": { "precision": "fp8", "vramMinimumGb": 805 }, "v3_0324": { "modelId": "deepseek-ai/DeepSeek-V3-0324", "precision": "fp8", "vramMinimumGb": 805 }, "fp4": { "modelId": "nvidia/DeepSeek-V3-FP4", "precision": "fp4", "vramMinimumGb": 403 } }, "verifiedGpus": ["h200", "b200", "gb200", "mi300x", "mi325x", "mi350x", "mi355x"] }, "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp": { "file": "models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp.yaml", "minVllmVersion": "0.30.0", "nightlyRequired": true, "contextLength": 1048576, "baseArgs": ["--kv-cache-dtype", "fp8", "--block-size", "256"], "toolCallParser": "deepseek_v4", "reasoningParser": "deepseek_v4", "variants": { "default": { "precision": "fp8", "vramMinimumGb": 202 } }, "verifiedGpus": ["gb200", "mi350x", "ascend_950dt_8x", "ascend_950pr_8x"] }, "deepseek-ai/DeepSeek-V4-Flash": { "file": "models/deepseek-ai/DeepSeek-V4-Flash.yaml", "minVllmVersion": "0.20.0", "contextLength": 1048576, "baseArgs": ["--trust-remote-code", "--kv-cache-dtype", "fp8", "--block-size", "256"], "toolCallParser": "deepseek_v4", "reasoningParser": "deepseek_v4", "optInFeatures": ["spec_decoding"], "variants": { "default": { "modelId": "deepseek-ai/DeepSeek-V4-Flash-0731", "precision": "fp8", "vramMinimumGb": 200 }, "fp8": { "precision": "fp8", "vramMinimumGb": 170 }, "nvfp4": { "modelId": "nvidia/DeepSeek-V4-Flash-NVFP4", "precision": "nvfp4", "vramMinimumGb": 170 }, "redhat_nvfp4": { "modelId": "RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8", "precision": "nvfp4", "vramMinimumGb": 170 }, "dspark": { "modelId": "deepseek-ai/DeepSeek-V4-Flash-DSpark", "precision": "fp8", "vramMinimumGb": 200 }, "ascend_w8a8": { "modelId": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", "precision": "int8", "vramMinimumGb": 341, "extraArgs": ["--data-parallel-size", "2", "--quantization", "ascend", "--max-num-seqs", "32", "--max-num-batched-tokens", "10240", "--gpu-memory-utilization", "0.90", "--block-size", "128", "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}', "--async-scheduling", "--api-server-count", "1", "--safetensors-load-strategy", "prefetch", "--additional-config", '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'] } }, "verifiedGpus": ["h200", "b200", "gb200", "b300", "gb300", "dgx_station_gb300", "dgx_spark_gb10", "rtx_pro_6000_8x", "mi300x", "mi325x", "mi355x", "ascend_950dt_8x", "ascend_950pr_8x", "ascend_910c"] }, "deepseek-ai/DeepSeek-V4-Pro": { "file": "models/deepseek-ai/DeepSeek-V4-Pro.yaml", "minVllmVersion": "0.20.0", "contextLength": 1048576, "baseArgs": ["--trust-remote-code", "--kv-cache-dtype", "fp8", "--block-size", "256"], "toolCallParser": "deepseek_v4", "reasoningParser": "deepseek_v4", "optInFeatures": ["spec_decoding"], "variants": { "default": { "modelId": "deepseek-ai/DeepSeek-V4-Pro-0813", "precision": "fp8", "vramMinimumGb": 960 }, "fp8": { "precision": "fp8", "vramMinimumGb": 960 }, "nvfp4": { "modelId": "nvidia/DeepSeek-V4-Pro-NVFP4", "precision": "nvfp4", "vramMinimumGb": 960 }, "dspark": { "modelId": "deepseek-ai/DeepSeek-V4-Pro-DSpark", "precision": "fp8", "vramMinimumGb": 960 }, "redhat_nvfp4": { "modelId": "RedHatAI/DeepSeek-V4-Pro-NVFP4-FP8", "precision": "nvfp4", "vramMinimumGb": 1e3 } }, "unsupportedGpus": ["mi300x", "mi325x"], "verifiedGpus": ["h200", "b200", "gb200", "b300", "gb300", "mi355x"] }, "deepseek-ai/DeepSeek-V4.1-Flash": { "file": "models/deepseek-ai/DeepSeek-V4.1-Flash.yaml", "minVllmVersion": "0.31.0", "contextLength": 1048576, "baseArgs": ["--tokenizer-mode", "deepseek_v41"], "toolCallParser": "deepseek_v41", "reasoningParser": "deepseek_v41", "optInFeatures": ["spec_decoding", "text_only"], "variants": { "default": { "modelId": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "precision": "fp8", "vramMinimumGb": 614 }, "fp8": { "precision": "fp8", "vramMinimumGb": 614 } }, "verifiedGpus": ["h100", "h200", "b200", "gb200", "gb300", "b300", "mi300x", "mi325x", "mi350x", "mi355x", "dgx_station_gb300"] }, "dots-studio/dots3-note-prev": { "file": "models/dots-studio/dots3-note-prev.yaml", "minVllmVersion": "0.28.0", "contextLength": 524288, "baseArgs": ["--served-model-name", "dots3-note-prev", "--gpu-memory-utilization", "0.9", "--max-model-len", "262144", "--max-num-batched-tokens", "8192", "--block-size", "64", "--mm-processor-cache-gb", "32"], "toolCallParser": "dots", "reasoningParser": "qwen3", "optInFeatures": ["reasoning", "tool_calling", "text_only", "spec_decoding"], "variants": { "default": { "modelId": "dots-studio/dots3-note-prev-fp8", "precision": "fp8", "vramMinimumGb": 359 }, "bf16": { "modelId": "dots-studio/dots3-note-prev", "precision": "bf16", "vramMinimumGb": 692, "extraArgs": ["--gpu-memory-utilization", "0.95"] } }, "unsupportedGpus": ["b200", "gb200", "b300", "gb300", "mi300x", "mi325x", "mi355x"], "verifiedGpus": ["h100"] }, "fishaudio/drama-3": { "file": "models/fishaudio/drama-3.yaml", "minVllmVersion": "0.31.0", "baseArgs": ["--omni"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 48 } }, "verifiedGpus": ["h100"] }, "fishaudio/s2-pro": { "file": "models/fishaudio/s2-pro.yaml", "minVllmVersion": "0.19.0", "baseArgs": ["--omni"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 48 } } }, "ibm-granite/granite-3.2-2b-instruct": { "file": "models/ibm-granite/granite-3.2-2b-instruct.yaml", "minVllmVersion": "0.6.1", "contextLength": 131072, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 6 } }, "verifiedGpus": ["xeon6"] }, "inclusionAI/Ling-2.6-1T": { "file": "models/inclusionAI/Ling-2.6-1T.yaml", "minVllmVersion": "0.20.2", "contextLength": 262144, "baseArgs": ["--trust-remote-code", "--tensor-parallel-size", "8"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 1200 } }, "verifiedGpus": ["b300", "mi300x"] }, "inclusionAI/Ling-2.6-flash": { "file": "models/inclusionAI/Ling-2.6-flash.yaml", "minVllmVersion": "0.20.2", "contextLength": 131072, "baseArgs": ["--trust-remote-code", "--tensor-parallel-size", "4"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 250 } }, "verifiedGpus": ["h200", "mi300x"] }, "inclusionAI/Ling-3.0-flash-VL": { "file": "models/inclusionAI/Ling-3.0-flash-VL.yaml", "minVllmVersion": "nightly", "nightlyRequired": true, "contextLength": 131072, "baseArgs": ["--trust-remote-code", "--dtype", "bfloat16", "--gpu-memory-utilization", "0.9", "--enable-chunked-prefill", "--compilation-config", '{"cudagraph_mode":"FULL_AND_PIECEWISE"}', "--enable-prefix-caching"], "toolCallParser": "ling3", "reasoningParser": "ling3", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 300 }, "fp8": { "modelId": "inclusionAI/Ling-3.0-flash-VL-fp8", "precision": "fp8", "vramMinimumGb": 150 }, "fp4": { "modelId": "inclusionAI/Ling-3.0-flash-VL-fp4", "precision": "fp4", "vramMinimumGb": 85 }, "int4": { "modelId": "inclusionAI/Ling-3.0-flash-VL-int4", "precision": "int4", "vramMinimumGb": 95 } } }, "inclusionAI/Ling-3.0-flash": { "file": "models/inclusionAI/Ling-3.0-flash.yaml", "minVllmVersion": "0.28.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code", "--dtype", "bfloat16", "--gpu-memory-utilization", "0.9", "--enable-chunked-prefill", "--compilation-config", '{"cudagraph_mode":"FULL_AND_PIECEWISE"}', "--enable-prefix-caching"], "toolCallParser": "ling3", "reasoningParser": "ling3", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 300 }, "fp8": { "modelId": "inclusionAI/Ling-3.0-flash-fp8", "precision": "fp8", "vramMinimumGb": 150 }, "fp4": { "modelId": "inclusionAI/Ling-3.0-flash-fp4", "precision": "fp4", "vramMinimumGb": 85 }, "int4": { "modelId": "inclusionAI/Ling-3.0-flash-int4", "precision": "int4", "vramMinimumGb": 95 } }, "verifiedGpus": ["h200", "dgx_spark_gb10", "mi300x", "mi355x"] }, "inclusionAI/Ling-3.0-tiny": { "file": "models/inclusionAI/Ling-3.0-tiny.yaml", "minVllmVersion": "0.28.0", "contextLength": 131072, "baseArgs": ["--trust-remote-code", "--dtype", "bfloat16", "--gpu-memory-utilization", "0.85", "--enable-prefix-caching"], "toolCallParser": "ling3", "reasoningParser": "ling3", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 18 }, "fp8": { "modelId": "inclusionAI/Ling-3.0-tiny-fp8", "precision": "fp8", "vramMinimumGb": 11 }, "int4": { "modelId": "inclusionAI/Ling-3.0-tiny-int4", "precision": "int4", "vramMinimumGb": 8 } }, "verifiedGpus": ["h200"] }, "inclusionAI/Ling-3.1-flash": { "file": "models/inclusionAI/Ling-3.1-flash.yaml", "minVllmVersion": "nightly", "nightlyRequired": true, "contextLength": 262144, "baseArgs": ["--trust-remote-code", "--enable-expert-parallel", "--gpu-memory-utilization", "0.9", "--enable-chunked-prefill", "--compilation-config", '{"cudagraph_mode":"FULL_AND_PIECEWISE"}', "--enable-prefix-caching"], "toolCallParser": "ling3", "reasoningParser": "ling3", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 689 } }, "verifiedGpus": ["mi355x"] }, "inclusionAI/Ming-omni-tts-0.5B": { "file": "models/inclusionAI/Ming-omni-tts-0.5B.yaml", "minVllmVersion": "0.22.0", "baseArgs": ["--omni", "--enforce-eager"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 16 } } }, "inclusionAI/Ring-1T-FP8": { "file": "models/inclusionAI/Ring-1T-FP8.yaml", "minVllmVersion": "0.11.0", "contextLength": 65536, "baseArgs": ["--trust-remote-code", "--tensor-parallel-size", "8", "--max-num-seqs", "32", "--kv-cache-dtype", "fp8", "--served-model-name", "Ring-1T-FP8"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 1200, "extraArgs": ["--gpu-memory-utilization", "0.97", "--compilation-config", '{"use_inductor": false}'] } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "inclusionAI/Ring-2.6-1T": { "file": "models/inclusionAI/Ring-2.6-1T.yaml", "minVllmVersion": "0.20.2", "contextLength": 131072, "baseArgs": ["--trust-remote-code", "--tensor-parallel-size", "8"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 1200 } } }, "internlm/Intern-S1": { "file": "models/internlm/Intern-S1.yaml", "minVllmVersion": "0.10.2", "contextLength": 65536, "baseArgs": ["--trust-remote-code"], "toolCallParser": "internlm", "reasoningParser": "deepseek_r1", "optInFeatures": ["text_only"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 578 }, "fp8": { "modelId": "internlm/Intern-S1-FP8", "precision": "fp8", "vramMinimumGb": 289 } }, "verifiedGpus": ["mi300x", "mi325x", "mi355x"] }, "internlm/Intern-S2-397B": { "file": "models/internlm/Intern-S2-397B.yaml", "minVllmVersion": "0.22.1", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "text_only", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 969 }, "fp8": { "modelId": "internlm/Intern-S2-397B-FP8", "precision": "fp8", "vramMinimumGb": 488 } } }, "internlm/Intern-S2-Preview": { "file": "models/internlm/Intern-S2-Preview.yaml", "minVllmVersion": "0.22.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "qwen3", "optInFeatures": ["spec_decoding", "text_only"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 87 }, "fp8": { "modelId": "internlm/Intern-S2-Preview-FP8", "precision": "fp8", "vramMinimumGb": 44 } }, "verifiedGpus": ["h200", "gb200"] }, "jinaai/jina-embeddings-v5-text-small": { "file": "models/jinaai/jina-embeddings-v5-text-small.yaml", "minVllmVersion": "0.20.0", "contextLength": 32768, "baseArgs": ["--trust-remote-code", "--runner", "pooling"], "variants": { "default": { "modelId": "jinaai/jina-embeddings-v5-text-small-retrieval", "precision": "bf16", "vramMinimumGb": 2 }, "text_matching": { "modelId": "jinaai/jina-embeddings-v5-text-small-text-matching", "precision": "bf16", "vramMinimumGb": 2 }, "classification": { "modelId": "jinaai/jina-embeddings-v5-text-small-classification", "precision": "bf16", "vramMinimumGb": 2 }, "clustering": { "modelId": "jinaai/jina-embeddings-v5-text-small-clustering", "precision": "bf16", "vramMinimumGb": 2 } } }, "jinaai/jina-reranker-m0": { "file": "models/jinaai/jina-reranker-m0.yaml", "minVllmVersion": "0.10.0", "contextLength": 32768, "baseArgs": ["--gpu-memory-utilization", "0.75", "--max-num-seqs", "32"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 6 } } }, "meituan-longcat/LongCat-Image-Edit": { "file": "models/meituan-longcat/LongCat-Image-Edit.yaml", "minVllmVersion": "0.12.0", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 36 } } }, "meta-llama/Llama-3.1-8B-Instruct": { "file": "models/meta-llama/Llama-3.1-8B-Instruct.yaml", "minVllmVersion": "0.6.0", "contextLength": 131072, "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 20 }, "w8a8_int8": { "modelId": "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", "precision": "int8", "vramMinimumGb": 10 }, "awq_int4": { "modelId": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", "precision": "int4", "vramMinimumGb": 5, "extraArgs": ["--quantization", "awq"] }, "nvfp4": { "modelId": "nvidia/Llama-3.1-8B-Instruct-NVFP4", "precision": "nvfp4", "vramMinimumGb": 5, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "nvidia_fp8": { "modelId": "nvidia/Llama-3.1-8B-Instruct-FP8", "precision": "fp8", "vramMinimumGb": 10, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "fp8": { "modelId": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 10, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8": { "modelId": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8", "precision": "fp8", "vramMinimumGb": 10, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8_block": { "modelId": "RedHatAI/Llama-3.1-8B-Instruct-FP8-block", "precision": "fp8", "vramMinimumGb": 10, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/Llama-3.1-8B-Instruct-NVFP4", "precision": "nvfp4", "vramMinimumGb": 6 } }, "verifiedGpus": ["h100", "h200", "mi300x", "mi325x", "mi355x", "trillium", "xeon6", "arc_pro_b60", "arc_pro_b70"] }, "meta-llama/Llama-3.1-8B": { "file": "models/meta-llama/Llama-3.1-8B.yaml", "minVllmVersion": "0.6.0", "contextLength": 131072, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 20 }, "w8a8_int8": { "modelId": "RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8", "precision": "int8", "vramMinimumGb": 10 } }, "verifiedGpus": ["xeon6"] }, "meta-llama/Llama-3.2-1B-Instruct": { "file": "models/meta-llama/Llama-3.2-1B-Instruct.yaml", "minVllmVersion": "0.6.2", "contextLength": 131072, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 3 }, "w8a8_int8": { "modelId": "RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8", "precision": "int8", "vramMinimumGb": 2 }, "awq_int4": { "modelId": "AMead10/Llama-3.2-1B-Instruct-AWQ", "precision": "int4", "vramMinimumGb": 2, "extraArgs": ["--quantization", "awq"] } }, "verifiedGpus": ["xeon6"] }, "meta-llama/Llama-3.2-1B": { "file": "models/meta-llama/Llama-3.2-1B.yaml", "minVllmVersion": "0.6.2", "contextLength": 131072, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 3 } }, "verifiedGpus": ["xeon6"] }, "meta-llama/Llama-3.2-3B-Instruct": { "file": "models/meta-llama/Llama-3.2-3B-Instruct.yaml", "minVllmVersion": "0.6.2", "contextLength": 131072, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 8 }, "w8a8_int8": { "modelId": "RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8", "precision": "int8", "vramMinimumGb": 4 }, "awq_int4": { "modelId": "AMead10/Llama-3.2-3B-Instruct-AWQ", "precision": "int4", "vramMinimumGb": 3, "extraArgs": ["--quantization", "awq"] } }, "verifiedGpus": ["xeon6"] }, "meta-llama/Llama-3.3-70B-Instruct": { "file": "models/meta-llama/Llama-3.3-70B-Instruct.yaml", "minVllmVersion": "0.12.0", "contextLength": 131072, "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 170 }, "fp8": { "modelId": "nvidia/Llama-3.3-70B-Instruct-FP8", "precision": "fp8", "vramMinimumGb": 84, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8": { "modelId": "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 84, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "nvfp4": { "modelId": "nvidia/Llama-3.3-70B-Instruct-NVFP4", "precision": "nvfp4", "vramMinimumGb": 42, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8_block": { "modelId": "RedHatAI/Llama-3.3-70B-Instruct-FP8-block", "precision": "fp8", "vramMinimumGb": 84, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/Llama-3.3-70B-Instruct-NVFP4", "precision": "nvfp4", "vramMinimumGb": 46 } }, "verifiedGpus": ["h100", "h200", "b200", "gb200", "mi300x", "mi325x", "mi355x", "trillium", "xeon6", "arc_pro_b70"] }, "meta-llama/Llama-4-Scout-17B-16E-Instruct": { "file": "models/meta-llama/Llama-4-Scout-17B-16E-Instruct.yaml", "minVllmVersion": "0.12.0", "contextLength": 10485760, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 262 }, "fp8": { "modelId": "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8", "precision": "fp8", "vramMinimumGb": 131, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "nvfp4": { "modelId": "nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4", "precision": "nvfp4", "vramMinimumGb": 65, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8": { "modelId": "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 131, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4", "precision": "nvfp4", "vramMinimumGb": 65, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8_block": { "modelId": "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-block", "precision": "fp8", "vramMinimumGb": 131, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h100", "b200", "gb200", "mi300x", "mi325x", "mi355x", "xeon6"] }, "meta-models/Muse-Glimmer-30B": { "file": "models/meta-models/Muse-Glimmer-30B.yaml", "minVllmVersion": "0.28.0", "contextLength": 131072, "baseArgs": ["--generation-config", "auto"], "toolCallParser": "muse_glimmer", "reasoningParser": "muse_glimmer", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 72 }, "fp8_block": { "modelId": "RedHatAI/Muse-Glimmer-30B-FP8-block", "precision": "fp8", "vramMinimumGb": 40 }, "nvfp4": { "modelId": "Inferact/Muse-Glimmer-30B-NVFP4-W4A4", "precision": "nvfp4", "vramMinimumGb": 31 } }, "verifiedGpus": ["gb300", "dgx_spark_gb10", "mi300x", "mi325x", "mi355x", "arc_pro_b60", "arc_pro_b70", "rtx_5090", "rtx_5090_2x"] }, "meta-models/Muse-Spark-Open-816B-A42B": { "file": "models/meta-models/Muse-Spark-Open-816B-A42B.yaml", "minVllmVersion": "0.33.0", "nightlyRequired": true, "contextLength": 1048576, "toolCallParser": "muse", "reasoningParser": "muse", "optInFeatures": ["spec_decoding"], "variants": { "bf16": { "modelId": "meta-models/Muse-Spark-Open-816B-A42B", "precision": "bf16", "vramMinimumGb": 1959 }, "bf16_online_mxfp4": { "modelId": "meta-models/Muse-Spark-Open-816B-A42B", "precision": "mxfp4", "vramMinimumGb": 567 }, "mxfp4": { "modelId": "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4", "precision": "mxfp4", "vramMinimumGb": 611 }, "default": { "modelId": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 640 } }, "verifiedGpus": ["h200", "gb200", "gb300", "mi350x", "mi355x"] }, "microsoft/Phi-4-mini-instruct": { "file": "models/microsoft/Phi-4-mini-instruct.yaml", "minVllmVersion": "0.7.0", "contextLength": 131072, "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 10 }, "mini_reasoning": { "modelId": "microsoft/Phi-4-mini-reasoning", "precision": "bf16", "vramMinimumGb": 10 }, "reasoning": { "modelId": "microsoft/Phi-4-reasoning", "precision": "bf16", "vramMinimumGb": 30 }, "multimodal": { "modelId": "microsoft/Phi-4-multimodal-instruct", "precision": "bf16", "vramMinimumGb": 16, "extraArgs": ["--trust-remote-code"] }, "fp8": { "modelId": "RedHatAI/Phi-4-mini-instruct-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 5, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h100"] }, "microsoft/Phi-4-multimodal-instruct": { "file": "models/microsoft/Phi-4-multimodal-instruct.yaml", "minVllmVersion": "0.8.0", "contextLength": 131072, "baseArgs": ["--trust-remote-code"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 14 } }, "verifiedGpus": ["xeon6"] }, "microsoft/Phi-4-reasoning": { "file": "models/microsoft/Phi-4-reasoning.yaml", "minVllmVersion": "0.4.2", "contextLength": 32768, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 34 } }, "verifiedGpus": ["xeon6"] }, "mindlab-research/Macaron-V1-Coding-Venti": { "file": "models/mindlab-research/Macaron-V1-Coding-Venti.yaml", "minVllmVersion": "0.25.0", "contextLength": 1048576, "baseArgs": ["--kv-cache-dtype", "fp8_e4m3"], "toolCallParser": "glm47", "reasoningParser": "glm45", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 1786 }, "fp8": { "precision": "fp8", "vramMinimumGb": 893, "extraArgs": ["--quantization", "fp8"] } }, "verifiedGpus": ["b300"] }, "mistralai/Ministral-3-14B-Instruct-2512": { "file": "models/mistralai/Ministral-3-14B-Instruct-2512.yaml", "minVllmVersion": "0.11.0", "contextLength": 262144, "baseArgs": ["--tokenizer_mode", "mistral", "--config_format", "mistral", "--load_format", "mistral"], "toolCallParser": "mistral", "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 17 }, "8b": { "modelId": "mistralai/Ministral-3-8B-Instruct-2512", "precision": "fp8", "vramMinimumGb": 12 }, "3b": { "modelId": "mistralai/Ministral-3-3B-Instruct-2512", "precision": "fp8", "vramMinimumGb": 6 }, "fp8": { "modelId": "mistralai/Ministral-3-14B-Instruct-2512-FP8", "precision": "fp8", "vramMinimumGb": 17, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8": { "modelId": "RedHatAI/Ministral-3-14B-Instruct-2512-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 17, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "nvfp4": { "modelId": "RedHatAI/Ministral-3-14B-Instruct-2512-NVFP4", "precision": "nvfp4", "vramMinimumGb": 9 } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "mistralai/Ministral-3-8B-Reasoning-2512": { "file": "models/mistralai/Ministral-3-8B-Reasoning-2512.yaml", "minVllmVersion": "0.11.0", "contextLength": 262144, "baseArgs": ["--tokenizer_mode", "mistral", "--config_format", "mistral", "--load_format", "mistral"], "toolCallParser": "mistral", "reasoningParser": "mistral", "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 22 }, "3b": { "modelId": "mistralai/Ministral-3-3B-Reasoning-2512", "precision": "bf16", "vramMinimumGb": 8 }, "14b": { "modelId": "mistralai/Ministral-3-14B-Reasoning-2512", "precision": "bf16", "vramMinimumGb": 32, "extraArgs": ["--tensor-parallel-size", "2"] } }, "verifiedGpus": ["h200", "gb200", "mi300x"] }, "mistralai/Mistral-Large-3-675B-Instruct-2512": { "file": "models/mistralai/Mistral-Large-3-675B-Instruct-2512.yaml", "minVllmVersion": "0.11.0", "contextLength": 294912, "baseArgs": ["--tokenizer_mode", "mistral", "--config_format", "mistral", "--load_format", "mistral"], "toolCallParser": "mistral", "optInFeatures": ["text_only"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 810, "extraArgs": ["--tensor-parallel-size", "8"] }, "nvfp4": { "modelId": "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4", "precision": "nvfp4", "vramMinimumGb": 405, "extraArgs": ["--tensor-parallel-size", "4"] }, "fp8": { "modelId": "mistralai/Mistral-Large-3-675B-Instruct-2512-FP8", "precision": "fp8", "vramMinimumGb": 810, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/Mistral-Large-3-675B-Instruct-2512-NVFP4", "precision": "nvfp4", "vramMinimumGb": 405 } }, "verifiedGpus": ["h100", "b200"] }, "mistralai/Mistral-Medium-3.5-128B": { "file": "models/mistralai/Mistral-Medium-3.5-128B.yaml", "minVllmVersion": "0.18.0", "contextLength": 262144, "baseArgs": ["--tokenizer_mode", "mistral", "--config_format", "mistral", "--load_format", "mistral"], "toolCallParser": "mistral", "reasoningParser": "mistral", "optInFeatures": ["text_only", "encoder_parallel", "spec_decoding"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 154 } }, "verifiedGpus": ["mi300x"] }, "mistralai/Mistral-Small-4-119B-2603": { "file": "models/mistralai/Mistral-Small-4-119B-2603.yaml", "minVllmVersion": "0.20.0", "contextLength": 262144, "baseArgs": ["--max-model-len", "262144"], "toolCallParser": "mistral", "reasoningParser": "mistral", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 143 }, "nvfp4": { "modelId": "mistralai/Mistral-Small-4-119B-2603-NVFP4", "precision": "nvfp4", "vramMinimumGb": 72 }, "redhat_nvfp4": { "modelId": "RedHatAI/Mistral-Small-4-119B-2603-NVFP4", "precision": "nvfp4", "vramMinimumGb": 72 } } }, "mistralai/Voxtral-4B-TTS-2603": { "file": "models/mistralai/Voxtral-4B-TTS-2603.yaml", "minVllmVersion": "0.20.0", "baseArgs": ["--omni"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 24 } } }, "mistralai/Voxtral-Mini-4B-Realtime-2602": { "file": "models/mistralai/Voxtral-Mini-4B-Realtime-2602.yaml", "minVllmVersion": "0.20.0", "contextLength": 131072, "baseArgs": ["--tokenizer-mode", "mistral", "--compilation_config", '{"cudagraph_mode": "PIECEWISE"}'], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 16 } }, "verifiedGpus": ["dgx_spark_gb10"] }, "moonshotai/Kimi-K2-Instruct": { "file": "models/moonshotai/Kimi-K2-Instruct.yaml", "minVllmVersion": "0.12.0", "contextLength": 131072, "baseArgs": ["--trust-remote-code", "--tokenizer-mode", "auto"], "toolCallParser": "kimi_k2", "variants": { "default": { "precision": "fp8", "vramMinimumGb": 1200 } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "moonshotai/Kimi-K2-Thinking": { "file": "models/moonshotai/Kimi-K2-Thinking.yaml", "minVllmVersion": "0.12.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "kimi_k2", "reasoningParser": "kimi_k2", "variants": { "default": { "precision": "int4", "vramMinimumGb": 600 }, "nvfp4": { "modelId": "nvidia/Kimi-K2-Thinking-NVFP4", "precision": "nvfp4", "vramMinimumGb": 600, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "fp8": { "modelId": "RedHatAI/Kimi-K2-Thinking-FP8-Block", "precision": "fp8", "vramMinimumGb": 1200, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "moonshotai/Kimi-K2.5": { "file": "models/moonshotai/Kimi-K2.5.yaml", "minVllmVersion": "0.19.1", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "kimi_k2", "reasoningParser": "kimi_k2", "optInFeatures": ["text_only"], "variants": { "default": { "precision": "int4", "vramMinimumGb": 714 }, "nvfp4": { "modelId": "nvidia/Kimi-K2.5-NVFP4", "precision": "nvfp4", "vramMinimumGb": 600, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "mxfp4": { "modelId": "amd/Kimi-K2.5-MXFP4", "precision": "mxfp4", "vramMinimumGb": 669, "extraArgs": ["--kv-cache-dtype", "fp8", "--mm-encoder-tp-mode", "data"] } }, "verifiedGpus": ["h200", "gb200", "mi300x", "mi325x", "mi355x"] }, "moonshotai/Kimi-K2.6": { "file": "models/moonshotai/Kimi-K2.6.yaml", "minVllmVersion": "0.25.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "kimi_k2", "reasoningParser": "kimi_k2", "optInFeatures": ["text_only"], "variants": { "default": { "precision": "int4", "vramMinimumGb": 714 }, "nvfp4": { "modelId": "nvidia/Kimi-K2.6-NVFP4", "precision": "nvfp4", "vramMinimumGb": 600, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "fp8": { "modelId": "RedHatAI/Kimi-K2.6-FP8-BLOCK", "precision": "fp8", "vramMinimumGb": 1200, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/Kimi-K2.6-NVFP4", "precision": "nvfp4", "vramMinimumGb": 600 } }, "verifiedGpus": ["h200", "b300", "gb200", "mi300x", "mi325x", "mi355x"] }, "moonshotai/Kimi-K2.7-Code": { "file": "models/moonshotai/Kimi-K2.7-Code.yaml", "minVllmVersion": "0.19.1", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "kimi_k2", "reasoningParser": "kimi_k2", "optInFeatures": ["text_only"], "variants": { "default": { "precision": "int4", "vramMinimumGb": 714 } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "moonshotai/Kimi-K3": { "file": "models/moonshotai/Kimi-K3.yaml", "minVllmVersion": "0.29.0", "contextLength": 1048576, "baseArgs": ["--trust-remote-code", "--gpu-memory-utilization", "0.95"], "toolCallParser": "kimi_k3", "reasoningParser": "kimi_k3", "optInFeatures": ["spec_decoding", "text_only", "decode_context_parallelism"], "variants": { "default": { "precision": "mxfp4", "vramMinimumGb": 1680 }, "nvfp4": { "modelId": "RedHatAI/Kimi-K3-NVFP4", "precision": "nvfp4", "vramMinimumGb": 1650 }, "ascend_w4a8": { "modelId": "Eco-Tech/Kimi-K3-w4a8", "precision": "int4", "vramMinimumGb": 1711, "extraArgs": ["--quantization", "ascend", "--enable-prefix-caching", "--enable-expert-parallel", "--max-num-seqs", "16", "--max-model-len", "131072", "--max-num-batched-tokens", "24576", "--gpu-memory-utilization", "0.9", "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}', "--mm-processor-cache-gb", "0", "--additional-config", '{"enable_cpu_binding":true, "enable_flashcomm1":true}', "--mm-encoder-tp-mode", "data", "--limit-mm-per-prompt", '{"vision_chunk": 2}', "--allowed-local-media-path", "/"] }, "fp8": { "modelId": "RedHatAI/Kimi-K3-FP8-BLOCK", "precision": "fp8", "vramMinimumGb": 3360 } }, "verifiedGpus": ["h200", "b200", "b300", "gb200", "gb300", "mi355x", "ascend_910c"] }, "moonshotai/Kimi-Linear-48B-A3B-Instruct": { "file": "models/moonshotai/Kimi-Linear-48B-A3B-Instruct.yaml", "minVllmVersion": "0.11.2", "contextLength": 1048576, "baseArgs": ["--trust-remote-code"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 115 } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "nvidia/Cosmos3-Nano": { "file": "models/nvidia/Cosmos3-Nano.yaml", "minVllmVersion": "0.21.0", "contextLength": 262144, "baseArgs": ["--host", "0.0.0.0", "--port", "8000", "--init-timeout", "1800"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 39 } }, "verifiedGpus": ["h200", "h100"] }, "nvidia/Cosmos3-Super-Image2Video": { "file": "models/nvidia/Cosmos3-Super-Image2Video.yaml", "minVllmVersion": "0.21.0", "contextLength": 262144, "baseArgs": ["--host", "0.0.0.0", "--port", "8000", "--cfg-parallel-size", "2", "--ulysses-degree", "4", "--use-hsdp", "--hsdp-shard-size", "8", "--init-timeout", "1800"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 154 } }, "verifiedGpus": ["h200", "h100"] }, "nvidia/Cosmos3-Super-Text2Image": { "file": "models/nvidia/Cosmos3-Super-Text2Image.yaml", "minVllmVersion": "0.21.0", "contextLength": 262144, "baseArgs": ["--host", "0.0.0.0", "--port", "8000", "--cfg-parallel-size", "2", "--ulysses-degree", "4", "--tensor-parallel-size", "1", "--use-hsdp", "--hsdp-shard-size", "8", "--init-timeout", "1800"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 154 } }, "verifiedGpus": ["h200", "h100", "gb200"] }, "nvidia/Cosmos3-Super": { "file": "models/nvidia/Cosmos3-Super.yaml", "minVllmVersion": "0.21.0", "contextLength": 262144, "baseArgs": ["--host", "0.0.0.0", "--port", "8000", "--cfg-parallel-size", "2", "--ulysses-degree", "4", "--use-hsdp", "--hsdp-shard-size", "8", "--init-timeout", "1800"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 154 } }, "verifiedGpus": ["h200", "h100"] }, "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": { "file": "models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml", "minVllmVersion": "0.17.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code", "--async-scheduling"], "toolCallParser": "qwen3_coder", "reasoningParser": "nano_v3", "optInFeatures": ["long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 72, "extraArgs": ["--kv-cache-dtype", "auto"] }, "fp8": { "modelId": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8", "precision": "fp8", "vramMinimumGb": 35, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h100", "h200", "arc_pro_b60", "arc_pro_b70"] }, "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": { "file": "models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16.yaml", "minVllmVersion": "0.11.2", "contextLength": 262144, "baseArgs": ["--trust-remote-code", "--async-scheduling"], "toolCallParser": "qwen3_coder", "reasoningParser": "nano_v3", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 10, "extraArgs": ["--kv-cache-dtype", "auto"] }, "fp8": { "modelId": "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8", "precision": "fp8", "vramMinimumGb": 5, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h100", "h200", "b200", "arc_pro_b60", "arc_pro_b70"] }, "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": { "file": "models/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16.yaml", "minVllmVersion": "0.17.1", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "nemotron_v3", "optInFeatures": ["spec_decoding", "long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 298, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "fp8": { "modelId": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8", "precision": "fp8", "vramMinimumGb": 149, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "nvfp4": { "modelId": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 75 }, "base_bf16": { "modelId": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16", "precision": "bf16", "vramMinimumGb": 298 } }, "verifiedGpus": ["h100", "h200", "b200", "dgx_station_gb300", "dgx_spark_gb10", "rtx_pro_6000"] }, "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": { "file": "models/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.yaml", "minVllmVersion": "0.22.0", "contextLength": 262144, "baseArgs": ["--served-model-name", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B", "--trust-remote-code", "--kv-cache-dtype", "fp8", "--max-num-seqs", "16", "--max-model-len", "262144", "--gpu-memory-utilization", "0.90", "--max-num-batched-tokens", "32768", "--enable-flashinfer-autotune", "--async-scheduling", "--mamba-backend", "triton", "--mamba-ssm-cache-dtype", "float32"], "toolCallParser": "qwen3_coder", "reasoningParser": "nemotron_v3", "optInFeatures": ["long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 1320 }, "fp4": { "modelId": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 330 } }, "verifiedGpus": ["b200"] }, "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": { "file": "models/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.yaml", "minVllmVersion": "0.27.1", "contextLength": 262144, "baseArgs": ["--mamba-backend", "flashinfer", "--mamba-cache-mode", "align", "--enable-prefix-caching", "--max-num-batched-tokens", "16384"], "toolCallParser": "qwen3_coder", "reasoningParser": "nemotron_v3", "optInFeatures": ["spec_decoding", "long_context"], "variants": { "default": { "modelId": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 18, "extraArgs": ["--kv-cache-dtype", "fp8", "--moe-backend", "marlin"] }, "bf16": { "modelId": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "precision": "bf16", "vramMinimumGb": 72, "extraArgs": ["--moe-backend", "flashinfer_cutlass"] } }, "verifiedGpus": ["h100", "dgx_spark_gb10", "rtx_pro_6000", "dgx_station_gb300"] }, "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4": { "file": "models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.yaml", "minVllmVersion": "0.31.0", "contextLength": 1048576, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "reasoningParser": "nemotron_v3", "optInFeatures": ["local_media"], "variants": { "default": { "precision": "nvfp4", "vramMinimumGb": 91 }, "fp8": { "modelId": "RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8", "precision": "fp8", "vramMinimumGb": 155, "extraArgs": ["--gpu-memory-utilization", "0.89", "--prefix-match-unit", "64", "--block-size", "64", "--compilation-config", '{"max_cudagraph_capture_size":192}', "--enable-expert-parallel", "--max-num-seqs", "128", "--max-model-len", "114688", "--max-num-batched-tokens", "16384", "--enable-prefix-caching", "--async-scheduling", "--mamba-backend", "flashinfer", "--mamba-ssm-cache-dtype", "float16", "--enable-mamba-cache-stochastic-rounding", "--mamba-cache-philox-rounds", "5", "--chat-template-content-format", "string", "--mamba-cache-mode", "align", "--enable-mamba-fine-grained-prefix-cache"] }, "bf16": { "modelId": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "precision": "bf16", "vramMinimumGb": 299 }, "nvfp4_gb200_4x": { "modelId": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 91, "extraArgs": ["--enable-expert-parallel", "--max-num-seqs", "512", "--max-model-len", "28672", "--no-enable-prefix-caching", "--async-scheduling", "--mamba-backend", "flashinfer", "--mamba-ssm-cache-dtype", "float16", "--enable-mamba-cache-stochastic-rounding", "--mamba-cache-philox-rounds", "5", "--chat-template-content-format", "string", "--mamba-cache-mode", "align"] }, "bf16_gb200_4x": { "modelId": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "precision": "bf16", "vramMinimumGb": 299, "extraArgs": ["--enable-expert-parallel", "--max-num-seqs", "512", "--max-model-len", "28672", "--no-enable-prefix-caching", "--async-scheduling", "--mamba-backend", "flashinfer", "--mamba-ssm-cache-dtype", "float32", "--chat-template-content-format", "string", "--mamba-cache-mode", "align"] } }, "verifiedGpus": ["h100", "h200", "gb200", "dgx_spark_gb10", "rtx_pro_6000"] }, "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16": { "file": "models/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16.yaml", "minVllmVersion": "0.11.1", "contextLength": 131072, "baseArgs": ["--trust-remote-code"], "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 29 }, "fp8": { "modelId": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8", "precision": "fp8", "vramMinimumGb": 14 }, "nvfp4": { "modelId": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD", "precision": "nvfp4", "vramMinimumGb": 8 } }, "verifiedGpus": ["h100", "b200"] }, "nvidia/NVIDIA-Nemotron-Nano-9B-v2": { "file": "models/nvidia/NVIDIA-Nemotron-Nano-9B-v2.yaml", "minVllmVersion": "0.10.1", "contextLength": 131072, "baseArgs": ["--trust-remote-code"], "toolCallParser": "nemotron_json", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 22 }, "fp8": { "modelId": "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8", "precision": "fp8", "vramMinimumGb": 11 }, "nvfp4": { "modelId": "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4", "precision": "nvfp4", "vramMinimumGb": 6 }, "base": { "modelId": "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base", "precision": "bf16", "vramMinimumGb": 22 }, "japanese": { "modelId": "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese", "precision": "bf16", "vramMinimumGb": 22 }, "redhat_fp8": { "modelId": "RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 11, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h100", "h200", "b200"] }, "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16": { "file": "models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16.yaml", "minVllmVersion": "0.20.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code", "--max-model-len", "131072", "--media-io-kwargs", '{"video": {"num_frames": 512, "fps": 1}}', "--video-pruning-rate", "0.5"], "toolCallParser": "qwen3_coder", "reasoningParser": "nemotron_v3", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 75 }, "fp8": { "modelId": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8", "precision": "fp8", "vramMinimumGb": 38, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "nvfp4": { "modelId": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4", "precision": "nvfp4", "vramMinimumGb": 28, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h100", "h200", "b200"] }, "openai/gpt-oss-120b": { "file": "models/openai/gpt-oss-120b.yaml", "minVllmVersion": "0.10.1", "contextLength": 131072, "toolCallParser": "openai", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "mxfp4", "vramMinimumGb": 96 }, "amd_fp8": { "modelId": "amd/gpt-oss-120b-w-mxfp4-a-fp8", "precision": "mxfp4", "vramMinimumGb": 80 } }, "verifiedGpus": ["h100", "h200", "b200", "mi300x", "mi325x", "mi355x", "ironwood", "arc_pro_b70"] }, "openai/gpt-oss-20b": { "file": "models/openai/gpt-oss-20b.yaml", "minVllmVersion": "0.10.1", "contextLength": 131072, "toolCallParser": "openai", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "mxfp4", "vramMinimumGb": 16 } }, "verifiedGpus": ["h100", "h200", "b200", "mi300x", "mi325x", "mi355x", "arc_pro_b60", "arc_pro_b70", "xeon6"] }, "openai/whisper-large-v3": { "file": "models/openai/whisper-large-v3.yaml", "minVllmVersion": "0.7.0", "contextLength": 448, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 4 } }, "verifiedGpus": ["xeon6"] }, "openbmb/MiniCPM-V-4.6": { "file": "models/openbmb/MiniCPM-V-4.6.yaml", "minVllmVersion": "0.22.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "qwen3_coder", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 4 } } }, "openbmb/MiniCPM5-1B": { "file": "models/openbmb/MiniCPM5-1B.yaml", "minVllmVersion": "0.21.0", "contextLength": 131072, "optInFeatures": ["reasoning"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 3 } } }, "openbmb/MiniCPM5-2B": { "file": "models/openbmb/MiniCPM5-2B.yaml", "minVllmVersion": "0.21.0", "contextLength": 131072, "toolCallParser": "minicpm5", "optInFeatures": ["reasoning"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 5 } } }, "openbmb/VoxCPM2": { "file": "models/openbmb/VoxCPM2.yaml", "minVllmVersion": "0.21.0", "baseArgs": ["--omni"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 24 } } }, "pfnet/plamo-3-nict-31b-base": { "file": "models/pfnet/plamo-3-nict-31b-base.yaml", "minVllmVersion": "0.12.0", "contextLength": 4096, "baseArgs": ["--trust-remote-code", "--max-model-len", "4096"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 77 } } }, "poolside/Laguna-M.1": { "file": "models/poolside/Laguna-M.1.yaml", "minVllmVersion": "0.21.0", "contextLength": 262144, "toolCallParser": "poolside_v1", "reasoningParser": "poolside_v1", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 540 }, "fp8": { "modelId": "poolside/Laguna-M.1-FP8", "precision": "fp8", "vramMinimumGb": 270 }, "nvfp4": { "modelId": "poolside/Laguna-M.1-NVFP4", "precision": "nvfp4", "vramMinimumGb": 135 } } }, "poolside/Laguna-S-2.1": { "file": "models/poolside/Laguna-S-2.1.yaml", "minVllmVersion": "0.25.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "poolside_v1", "reasoningParser": "poolside_v1", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 282 }, "fp8": { "modelId": "poolside/Laguna-S-2.1-FP8", "precision": "fp8", "vramMinimumGb": 145 }, "nvfp4": { "modelId": "poolside/Laguna-S-2.1-NVFP4", "precision": "nvfp4", "vramMinimumGb": 86 }, "int4": { "modelId": "poolside/Laguna-S-2.1-INT4", "precision": "int4", "vramMinimumGb": 86 } }, "verifiedGpus": ["h200", "mi325x"] }, "poolside/Laguna-XS-2.1": { "file": "models/poolside/Laguna-XS-2.1.yaml", "minVllmVersion": "0.21.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "poolside_v1", "reasoningParser": "poolside_v1", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 80 }, "fp8": { "modelId": "poolside/Laguna-XS-2.1-FP8", "precision": "fp8", "vramMinimumGb": 40 }, "nvfp4": { "modelId": "poolside/Laguna-XS-2.1-NVFP4", "precision": "nvfp4", "vramMinimumGb": 26 }, "int4": { "modelId": "poolside/Laguna-XS-2.1-INT4", "precision": "int4", "vramMinimumGb": 29 } }, "verifiedGpus": ["h200"] }, "poolside/Laguna-XS.2": { "file": "models/poolside/Laguna-XS.2.yaml", "minVllmVersion": "0.21.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "poolside_v1", "reasoningParser": "poolside_v1", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 80 }, "fp8": { "modelId": "poolside/Laguna-XS.2-FP8", "precision": "fp8", "vramMinimumGb": 40 }, "nvfp4": { "modelId": "poolside/Laguna-XS.2-NVFP4", "precision": "nvfp4", "vramMinimumGb": 26 }, "int4": { "modelId": "poolside/Laguna-XS.2-INT4", "precision": "int4", "vramMinimumGb": 29 }, "redhat_fp8": { "modelId": "RedHatAI/Laguna-XS.2-FP8", "precision": "fp8", "vramMinimumGb": 40, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/Laguna-XS.2-NVFP4", "precision": "nvfp4", "vramMinimumGb": 22 } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "reflection/Beam-1.0-501B-A23B": { "file": "models/reflection/Beam-1.0-501B-A23B.yaml", "minVllmVersion": "0.33.0", "contextLength": 1048576, "baseArgs": ["--kv-cache-dtype", "fp8"], "toolCallParser": "beam", "reasoningParser": "beam", "optInFeatures": ["spec_decoding"], "variants": { "default": { "modelId": "reflection/Beam-1.0-501B-A23B-NVFP4", "precision": "nvfp4", "vramMinimumGb": 353 }, "fp8": { "modelId": "reflection/Beam-1.0-501B-A23B-FP8", "precision": "fp8", "vramMinimumGb": 604, "extraArgs": ["--moe-backend", "triton"] }, "bf16": { "modelId": "reflection/Beam-1.0-501B-A23B", "precision": "bf16", "vramMinimumGb": 1204, "extraArgs": ["--kv-cache-dtype", "auto"] } }, "unsupportedGpus": ["mi300x", "mi325x", "mi350x", "mi355x"], "verifiedGpus": ["b300", "h200"] }, "stabilityai/stable-audio-open-1.0": { "file": "models/stabilityai/stable-audio-open-1.0.yaml", "minVllmVersion": "0.22.0", "baseArgs": ["--trust-remote-code", "--enforce-eager", "--gpu-memory-utilization", "0.9"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 19 } } }, "stabilityai/stable-diffusion-3.5-medium": { "file": "models/stabilityai/stable-diffusion-3.5-medium.yaml", "minVllmVersion": "0.12.0", "variants": { "default": { "precision": "bf16", "vramMinimumGb": 44 }, "large": { "modelId": "stabilityai/stable-diffusion-3.5-large", "precision": "bf16", "vramMinimumGb": 24 }, "large_turbo": { "modelId": "stabilityai/stable-diffusion-3.5-large-turbo", "precision": "bf16", "vramMinimumGb": 24 } }, "verifiedGpus": ["mi300x", "mi325x", "mi355x"] }, "stepfun-ai/Step-3.5-Flash": { "file": "models/stepfun-ai/Step-3.5-Flash.yaml", "minVllmVersion": "0.15.1", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "step3p5", "reasoningParser": "step3p5", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 470 }, "fp8": { "modelId": "stepfun-ai/Step-3.5-Flash-FP8", "precision": "fp8", "vramMinimumGb": 235 }, "int4": { "modelId": "stepfun-ai/Step-3.5-Flash-INT4", "precision": "int4", "vramMinimumGb": 118 }, "int8": { "modelId": "stepfun-ai/Step-3.5-Flash-INT8", "precision": "int8", "vramMinimumGb": 235 } }, "verifiedGpus": ["h200", "b200"] }, "stepfun-ai/Step-3.7-Flash": { "file": "models/stepfun-ai/Step-3.7-Flash.yaml", "minVllmVersion": "0.23.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code", "--enable-expert-parallel", "--disable-cascade-attn"], "toolCallParser": "step3p5", "reasoningParser": "step3p5", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 475 }, "fp8": { "modelId": "stepfun-ai/Step-3.7-Flash-FP8", "precision": "fp8", "vramMinimumGb": 238 }, "nvfp4": { "modelId": "stepfun-ai/Step-3.7-Flash-NVFP4", "precision": "nvfp4", "vramMinimumGb": 119, "extraArgs": ["--quantization", "modelopt", "--kv-cache-dtype", "fp8", "--gpu-memory-utilization", "0.9", "--async-scheduling"] } }, "verifiedGpus": ["h200", "b200", "dgx_station_gb300"] }, "stepfun-ai/Step-5-Preview": { "file": "models/stepfun-ai/Step-5-Preview.yaml", "contextLength": 1048576, "baseArgs": ["--data-parallel-backend", "mp", "--distributed-executor-backend", "mp", "--disable-custom-all-reduce", "--max-model-len", "1048576", "--max-num-batched-tokens", "16384", "--max-num-seqs", "64", "--gpu-memory-utilization", "0.85", "--enable-prefix-caching"], "toolCallParser": "step5", "reasoningParser": "step5", "variants": { "default": { "precision": "fp8", "vramMinimumGb": 773 }, "nvfp4": { "modelId": "stepfun-ai/Step-5-Preview-NVFP4", "precision": "nvfp4", "vramMinimumGb": 473 } }, "verifiedGpus": ["h200", "b300", "gb300"] }, "tencent/Hunyuan-A13B-Instruct": { "file": "models/tencent/Hunyuan-A13B-Instruct.yaml", "minVllmVersion": "0.11.0", "contextLength": 32768, "baseArgs": ["--trust-remote-code"], "optInFeatures": ["long_context"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 580 }, "fp8": { "modelId": "tencent/Hunyuan-A13B-Instruct-FP8", "precision": "fp8", "vramMinimumGb": 96, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["mi300x", "mi325x", "mi355x"] }, "tencent/HunyuanOCR": { "file": "models/tencent/HunyuanOCR.yaml", "minVllmVersion": "0.12.0", "contextLength": 32768, "baseArgs": ["--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"], "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 2 } } }, "tencent/Hy3-preview": { "file": "models/tencent/Hy3-preview.yaml", "minVllmVersion": "0.20.0", "contextLength": 262144, "baseArgs": ["--trust-remote-code"], "toolCallParser": "hy_v3", "reasoningParser": "hy_v3", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 708 } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "tencent/Hy3": { "file": "models/tencent/Hy3.yaml", "minVllmVersion": "0.28.0", "contextLength": 262144, "toolCallParser": "hy_v3", "reasoningParser": "hy_v3", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 708 }, "fp8": { "modelId": "tencent/Hy3-FP8", "precision": "fp8", "vramMinimumGb": 354 }, "nvfp4": { "modelId": "RedHatAI/Hy3-NVFP4-FP8", "precision": "nvfp4", "vramMinimumGb": 214 } }, "verifiedGpus": ["gb300", "mi300x", "mi355x"] }, "tencent/Hy4-preview": { "file": "models/tencent/Hy4-preview.yaml", "minVllmVersion": "0.29.0", "nightlyRequired": true, "contextLength": 1048576, "toolCallParser": "hy_v4", "reasoningParser": "hy_v4", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 1848 }, "fp8": { "modelId": "tencent/Hy4-preview-FP8", "precision": "mxfp8", "vramMinimumGb": 924 } }, "verifiedGpus": ["b200", "b300", "mi325x", "mi355x"] }, "thinkingmachines/Inkling-Small-NVFP4": { "file": "models/thinkingmachines/Inkling-Small.yaml", "minVllmVersion": "0.26.0", "nightlyRequired": true, "contextLength": 1048576, "baseArgs": ["--trust-remote-code", "--tokenizer-mode", "inkling"], "toolCallParser": "inkling", "reasoningParser": "inkling", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "nvfp4", "vramMinimumGb": 180, "extraArgs": ["--kernel-config.enable_flashinfer_autotune=False"] }, "mxfp4": { "modelId": "EmbeddedLLM/Inkling-Small-MXFP4", "precision": "mxfp4", "vramMinimumGb": 160 }, "bf16": { "modelId": "thinkingmachines/Inkling-Small", "precision": "bf16", "vramMinimumGb": 600 }, "fp8": { "modelId": "RedHatAI/Inkling-Small-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 332, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["mi300x", "mi355x"] }, "thinkingmachines/Inkling-NVFP4": { "file": "models/thinkingmachines/Inkling.yaml", "minVllmVersion": "0.26.0", "nightlyRequired": true, "contextLength": 1048576, "baseArgs": ["--trust-remote-code", "--tokenizer-mode", "inkling"], "toolCallParser": "inkling", "reasoningParser": "inkling", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "nvfp4", "vramMinimumGb": 651, "extraArgs": ["--kernel-config.enable_flashinfer_autotune=False"] }, "mxfp4": { "modelId": "lightseekorg/Inkling-MXFP4", "precision": "mxfp4", "vramMinimumGb": 540 }, "bf16": { "modelId": "thinkingmachines/Inkling", "precision": "bf16", "vramMinimumGb": 2400, "extraArgs": ["--kernel-config.enable_flashinfer_autotune=False"] }, "fp8": { "modelId": "RedHatAI/Inkling-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 1170, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8": { "modelId": "RedHatAI/Inkling-FP8-block", "precision": "fp8", "vramMinimumGb": 1170, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_nvfp4": { "modelId": "RedHatAI/Inkling-NVFP4", "precision": "nvfp4", "vramMinimumGb": 618 } }, "verifiedGpus": ["gb200", "b200", "h200", "mi300x", "mi355x"] }, "vllm-sr/d3-edge": { "file": "models/vllm-sr/d3-edge.yaml", "minVllmVersion": "nightly", "nightlyRequired": true, "contextLength": 262144, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 2 } }, "verifiedGpus": ["mi300x"] }, "vllm-sr/d3-flash": { "file": "models/vllm-sr/d3-flash.yaml", "minVllmVersion": "nightly", "nightlyRequired": true, "contextLength": 262144, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 21 } }, "verifiedGpus": ["mi300x"] }, "vllm-sr/d3-lite": { "file": "models/vllm-sr/d3-lite.yaml", "minVllmVersion": "nightly", "nightlyRequired": true, "contextLength": 262144, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 3 } }, "verifiedGpus": ["mi300x"] }, "vllm-sr/d3-mini": { "file": "models/vllm-sr/d3-mini.yaml", "minVllmVersion": "nightly", "nightlyRequired": true, "contextLength": 262144, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 11 } }, "verifiedGpus": ["mi300x"] }, "vllm-sr/d3-nano": { "file": "models/vllm-sr/d3-nano.yaml", "minVllmVersion": "nightly", "nightlyRequired": true, "contextLength": 262144, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 6 } }, "verifiedGpus": ["mi300x"] }, "vllm-sr/d3": { "file": "models/vllm-sr/d3.yaml", "minVllmVersion": "nightly", "nightlyRequired": true, "contextLength": 262144, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 63 } }, "verifiedGpus": ["mi300x"] }, "zai-org/GLM-4.5": { "file": "models/zai-org/GLM-4.5.yaml", "minVllmVersion": "0.11.0", "contextLength": 131072, "baseArgs": ["--trust-remote-code"], "toolCallParser": "glm45", "reasoningParser": "glm45", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 859 }, "fp8": { "modelId": "zai-org/GLM-4.5-FP8", "precision": "fp8", "vramMinimumGb": 430 } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "zai-org/GLM-4.5V": { "file": "models/zai-org/GLM-4.5V.yaml", "minVllmVersion": "0.12.0", "contextLength": 65536, "baseArgs": ["--trust-remote-code", "--enable-expert-parallel", "--allowed-local-media-path", "/", "--mm-encoder-tp-mode", "data", "--mm-processor-cache-type", "shm"], "toolCallParser": "glm45", "reasoningParser": "glm45", "optInFeatures": ["text_only"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 257 }, "fp8": { "modelId": "zai-org/GLM-4.5V-FP8", "precision": "fp8", "vramMinimumGb": 128 } }, "verifiedGpus": ["h100", "h200", "mi300x", "mi325x", "mi355x"] }, "zai-org/GLM-4.6": { "file": "models/zai-org/GLM-4.6.yaml", "minVllmVersion": "0.11.0", "contextLength": 202752, "baseArgs": ["--trust-remote-code"], "toolCallParser": "glm45", "reasoningParser": "glm45", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 857 }, "fp8": { "modelId": "zai-org/GLM-4.6-FP8", "precision": "fp8", "vramMinimumGb": 428 }, "redhat_fp8": { "modelId": "RedHatAI/GLM-4.6-FP8-dynamic", "precision": "fp8", "vramMinimumGb": 429, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "redhat_fp8_plain": { "modelId": "RedHatAI/GLM-4.6-FP8", "precision": "fp8", "vramMinimumGb": 429, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "nvfp4": { "modelId": "RedHatAI/GLM-4.6-NVFP4", "precision": "nvfp4", "vramMinimumGb": 223 } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "zai-org/GLM-4.6V": { "file": "models/zai-org/GLM-4.6V.yaml", "minVllmVersion": "0.12.0", "contextLength": 131072, "baseArgs": ["--trust-remote-code", "--enable-expert-parallel", "--allowed-local-media-path", "/", "--mm-encoder-tp-mode", "data", "--mm-processor-cache-type", "shm"], "toolCallParser": "glm45", "reasoningParser": "glm45", "optInFeatures": ["text_only"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 257 }, "fp8": { "modelId": "zai-org/GLM-4.6V-FP8", "precision": "fp8", "vramMinimumGb": 128 } }, "verifiedGpus": ["h100", "h200", "mi300x", "mi325x", "mi355x"] }, "zai-org/GLM-4.7": { "file": "models/zai-org/GLM-4.7.yaml", "minVllmVersion": "0.24.0", "contextLength": 202752, "baseArgs": ["--trust-remote-code"], "toolCallParser": "glm47", "reasoningParser": "glm47", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 859 }, "fp8": { "modelId": "zai-org/GLM-4.7-FP8", "precision": "fp8", "vramMinimumGb": 430 }, "nvfp4": { "modelId": "nvidia/GLM-4.7-NVFP4", "precision": "nvfp4", "vramMinimumGb": 215, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h200", "mi300x", "mi325x", "mi355x"] }, "zai-org/GLM-5.1": { "file": "models/zai-org/GLM-5.1.yaml", "minVllmVersion": "0.24.0", "contextLength": 202752, "baseArgs": ["--trust-remote-code", "--chat-template-content-format=string"], "toolCallParser": "glm47", "reasoningParser": "glm47", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 1786 }, "fp8": { "modelId": "zai-org/GLM-5.1-FP8", "precision": "fp8", "vramMinimumGb": 893 }, "nvfp4": { "modelId": "nvidia/GLM-5.1-NVFP4", "precision": "nvfp4", "vramMinimumGb": 446, "extraArgs": ["--kv-cache-dtype", "fp8"] }, "mxfp4": { "modelId": "amd/GLM-5.1-MXFP4", "precision": "mxfp4", "vramMinimumGb": 447, "extraArgs": ["--kv-cache-dtype", "fp8_e4m3", "-cc.pass_config.fuse_allreduce_rms=False", "--block-size=1", "--enable-prefix-caching", "--max-num-seqs", "64"] } }, "verifiedGpus": ["h200", "mi325x", "mi350x", "mi355x"] }, "zai-org/GLM-5.2": { "file": "models/zai-org/GLM-5.2.yaml", "minVllmVersion": "0.24.0", "contextLength": 1048576, "baseArgs": ["--kv-cache-dtype", "fp8_e4m3"], "toolCallParser": "glm47", "reasoningParser": "glm47", "optInFeatures": ["spec_decoding"], "variants": { "default": { "modelId": "zai-org/GLM-5.2-FP8", "precision": "fp8", "vramMinimumGb": 893 }, "mxfp4": { "modelId": "amd/GLM-5.2-MXFP4", "precision": "mxfp4", "vramMinimumGb": 446 }, "nvfp4": { "modelId": "nvidia/GLM-5.2-NVFP4", "precision": "nvfp4", "vramMinimumGb": 558 }, "redhat_nvfp4": { "modelId": "RedHatAI/GLM-5.2-NVFP4-FP8", "precision": "nvfp4", "vramMinimumGb": 558 }, "bf16": { "precision": "bf16", "vramMinimumGb": 1786 }, "redhat_fp8": { "modelId": "RedHatAI/GLM-5.2-FP8", "precision": "fp8", "vramMinimumGb": 892 }, "redhat_nvfp4_plain": { "modelId": "RedHatAI/GLM-5.2-NVFP4", "precision": "nvfp4", "vramMinimumGb": 477 } }, "verifiedGpus": ["b300", "b200", "mi300x", "mi325x", "mi350x", "mi355x"] }, "zai-org/GLM-5.3-Flash": { "file": "models/zai-org/GLM-5.3-Flash.yaml", "minVllmVersion": "0.29.0", "nightlyRequired": true, "contextLength": 1048576, "toolCallParser": "glm47", "reasoningParser": "glm47", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 386 }, "nvfp4": { "modelId": "RedHatAI/GLM-5.3-Flash-NVFP4", "precision": "nvfp4", "vramMinimumGb": 229 }, "bf16": { "modelId": "zai-org/GLM-5.3-Flash-BF16", "precision": "bf16", "vramMinimumGb": 772 } }, "verifiedGpus": ["h100", "b200", "gb200", "mi355x", "ascend_950pr_8x"] }, "zai-org/GLM-5.3": { "file": "models/zai-org/GLM-5.3.yaml", "minVllmVersion": "0.29.0", "contextLength": 1048576, "baseArgs": ["--kv-cache-dtype", "fp8_e4m3"], "toolCallParser": "glm47", "reasoningParser": "glm47", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "fp8", "vramMinimumGb": 893 }, "nvfp4": { "modelId": "Inferact/GLM-5.3-NVFP4", "precision": "nvfp4", "vramMinimumGb": 558 }, "w4a8": { "modelId": "gpustack/GLM-5.3-W4A8", "precision": "int4", "vramMinimumGb": 447, "extraArgs": ["--enable-expert-parallel", "--trust-remote-code"] }, "bf16": { "modelId": "zai-org/GLM-5.3-BF16", "precision": "bf16", "vramMinimumGb": 1786 } }, "verifiedGpus": ["b300", "h200", "mi325x", "mi350x", "ascend_950pr_8x", "dgx_station_gb300"] }, "zai-org/GLM-5": { "file": "models/zai-org/GLM-5.yaml", "minVllmVersion": "0.24.0", "contextLength": 202752, "baseArgs": ["--trust-remote-code", "--chat-template-content-format=string"], "toolCallParser": "glm47", "reasoningParser": "glm47", "optInFeatures": ["spec_decoding"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 1786 }, "fp8": { "modelId": "zai-org/GLM-5-FP8", "precision": "fp8", "vramMinimumGb": 893 }, "nvfp4": { "modelId": "nvidia/GLM-5-NVFP4", "precision": "nvfp4", "vramMinimumGb": 446, "extraArgs": ["--kv-cache-dtype", "fp8"] } }, "verifiedGpus": ["h200", "mi325x", "mi350x"] }, "zai-org/GLM-ASR-Nano-2512": { "file": "models/zai-org/GLM-ASR-Nano-2512.yaml", "minVllmVersion": "0.14.1", "contextLength": 8192, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 11 } } }, "zai-org/GLM-GA": { "file": "models/zai-org/GLM-GA.yaml", "minVllmVersion": "0.21.0", "contextLength": 131072, "baseArgs": ["--trust-remote-code", "--allowed-local-media-path", "/", "--mm-processor-cache-type", "shm", "--reasoning-parser", "glm45"], "optInFeatures": ["text_only"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 29 } } }, "zai-org/GLM-Image": { "file": "models/zai-org/GLM-Image.yaml", "minVllmVersion": "0.11.0", "contextLength": 4096, "baseArgs": ["--trust-remote-code"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 38 } }, "verifiedGpus": ["h100"] }, "zai-org/GLM-OCR": { "file": "models/zai-org/GLM-OCR.yaml", "minVllmVersion": "0.16.0", "contextLength": 131072, "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 2 } } }, "zai-org/GLM-TTS": { "file": "models/zai-org/GLM-TTS.yaml", "minVllmVersion": "0.22.0", "baseArgs": ["--omni", "--trust-remote-code"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 24 } } }, "zai-org/Glyph": { "file": "models/zai-org/Glyph.yaml", "minVllmVersion": "0.11.0", "contextLength": 131072, "baseArgs": ["--no-enable-prefix-caching", "--mm-processor-cache-gb", "0", "--limit-mm-per-prompt.video", "0"], "reasoningParser": "glm45", "optInFeatures": ["text_only", "encoder_parallel"], "variants": { "default": { "precision": "bf16", "vramMinimumGb": 24 } }, "verifiedGpus": ["h100", "mi300x", "mi325x"] }, "zai-org/glm-4-9b-hf": { "file": "models/zai-org/glm-4-9b-hf.yaml", "minVllmVersion": "0.6.5", "contextLength": 8192, "variants": { "default": { "precision": "bf16", "vramMinimumGb": 22 } }, "verifiedGpus": ["xeon6"] } } };

// src/utils/serveConfigCheck.ts
var SERVE_VALIDATION = {
  llmdRuns: 16,
  llmdKvMae: 0.33,
  poolLogs: 21,
  poolTokensMae: 1.07,
  poolConcMae: 3.95,
  e2eLogs: 9,
  e2eKvMedian: 7.63,
  /** Table D, the CUDA graph term against vLLM's printed estimate, out-of-sample rows without #50780 (GiB). */
  graphLogs: 10,
  graphMaeGib: 0.38,
  graphConstantMaeGib: 0.63,
  /** Table E, vision-language launches end to end, KV GiB. */
  mmLogs: 10,
  mmKvMedian: 5.96
};
var SERVE_BASIS = `Predicted from config.json, the engine flags and the card, not measured. Memory follows each engine's own arithmetic, ported from its source (vLLM's KV cache groups and blocks, SGLang's mem_fraction_static formula, llama.cpp's sliding-window cells); the terms an engine measures at startup (vLLM's activation peak, non-torch memory and CUDA graph pool) are estimates and are marked as such. Checked against vLLM's own logged numbers: ${SERVE_VALIDATION.llmdRuns} llm-d-planner H100 runs, KV memory mean absolute error ${SERVE_VALIDATION.llmdKvMae}% (in-sample for one fitted factor); given the KV memory vLLM logged, ${SERVE_VALIDATION.poolLogs} public logs, token count ${SERVE_VALIDATION.poolTokensMae}% and concurrency ${SERVE_VALIDATION.poolConcMae}%; end to end on ${SERVE_VALIDATION.e2eLogs} public logs, median ${SERVE_VALIDATION.e2eKvMedian}% with outliers past 100%; vision-language launches end to end on ${SERVE_VALIDATION.mmLogs} public logs, median ${SERVE_VALIDATION.mmKvMedian}%; the CUDA graph term against the estimate vLLM printed, ${SERVE_VALIDATION.graphLogs} held-out public logs, mean absolute error ${SERVE_VALIDATION.graphMaeGib} GiB (scripts/serve-check-validation.ts). Template checks read files and render three fixed conversations with @huggingface/jinja to find calls the template never defines; they cannot prove two stacks produce the same prompt.`;
var SERVE_BASIS_UPPER_BOUND = "The config gave no layer geometry, so memory was sized from the graph alone: the engine's reserve (activation profiling, CUDA graphs, communication buffers) is not counted and the card's nominal GB is used, so every KV capacity and concurrency here is an upper bound.";
var CARD_SM = {
  "b200": 100,
  "h200": 90,
  "h100-sxm": 90,
  "a100-80g": 80,
  "a100-40g": 80,
  "l40s": 89,
  "l4": 89,
  "a10g": 86,
  "t4": 75,
  "v100": 70,
  "rtx-4090": 89,
  "rtx-3090": 86,
  // Serve-check cards below (SERVE_EXTRA_CARDS): CUDA compute capability from NVIDIA's public list (developer.nvidia.com/cuda-gpus).
  "a40": 86,
  "gh200-96g": 90,
  "gh200-144g": 90,
  "rtx-pro-6000": 120
};
var SERVE_EXTRA_CARDS = [
  { id: "a40", label: "A40 (48 GB)", tier: "datacenter", memoryGB: 48, bf16TFLOPS: 149.7, fp32TFLOPS: 37.4, memBandwidthGBs: 696, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "gh200-96g", label: "GH200 (96 GB HBM3)", tier: "datacenter", memoryGB: 96, bf16TFLOPS: 989, fp32TFLOPS: 67, memBandwidthGBs: 4e3, nvlinkGBs: 450, nodeEgressGBs: 400 },
  { id: "gh200-144g", label: "GH200 (144 GB HBM3e)", tier: "datacenter", memoryGB: 144, bf16TFLOPS: 989, fp32TFLOPS: 67, memBandwidthGBs: 4900, nvlinkGBs: 450, nodeEgressGBs: 400 },
  { id: "rtx-pro-6000", label: "RTX PRO 6000 Blackwell (96 GB)", tier: "datacenter", memoryGB: 96, bf16TFLOPS: 0, fp32TFLOPS: 0, memBandwidthGBs: 1792, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "ascend-910b", label: "Ascend 910B (64 GB)", tier: "datacenter", memoryGB: 64, bf16TFLOPS: 0, fp32TFLOPS: 0, memBandwidthGBs: 1600, nvlinkGBs: 0, nodeEgressGBs: 0 }
];
var baseCardId = (id) => id.split("@")[0];
var smOfCard = (id) => CARD_SM[baseCardId(id)] ?? null;
function resolveServeGpu(id, memGiB) {
  let base = null;
  if (id != null && id !== "" && id !== "custom") {
    const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
    const want = norm(id);
    base = SERVE_EXTRA_CARDS.find((g2) => norm(g2.id) === want) ?? null;
    if (!base) {
      const extraHits = SERVE_EXTRA_CARDS.filter((g2) => norm(g2.id).startsWith(want) || norm(g2.label).startsWith(want));
      const main = resolveLayoutGpu(id);
      if (main && !extraHits.length) base = main;
      else if (!main && extraHits.length === 1) base = extraHits[0];
      else if (main && extraHits.length) base = null;
    }
    if (!base) return null;
  }
  if (memGiB == null) return base;
  if (!(memGiB > 0)) return null;
  const g = Math.round(memGiB * 100) / 100;
  return {
    ...base ?? { id: "custom", label: "custom card", tier: "datacenter", bf16TFLOPS: 0, fp32TFLOPS: 0, memBandwidthGBs: 0, nvlinkGBs: 0, nodeEgressGBs: 0 },
    id: `${base?.id ?? "custom"}@${g}`,
    label: `${base ? base.label.replace(/\s*\(.*\)$/, "") : "custom card"} (${g} GiB, --gpu-mem)`,
    memoryGB: Math.round(g * GiB / 1e9 * 100) / 100
  };
}
var VLLM_QUANT_TABLE = 'vLLM docs, "Supported Hardware" quantization table (docs.vllm.ai/en/latest/features/quantization/)';
var QUANT_SUPPORT = [
  {
    method: "fp8",
    nativeSm: 89,
    fallbackSm: 75,
    fallbackNote: "runs as weight-only W8A16 through the FP8 Marlin kernel: weights are stored in fp8, the matmuls run in 16 bits, so the memory saving is there and the FP8 compute speedup is not",
    source: 'vLLM docs, FP8 W8A8: "FP8 computation is supported on NVIDIA GPUs with compute capability >= 8.9"; "FP8 models will run on compute capability >= 7.5 (Turing) as weight-only W8A16, utilizing FP8 Marlin" (docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/)'
  },
  {
    method: "mxfp4",
    nativeSm: 90,
    fallbackSm: 80,
    fallbackNote: "runs on Ampere and Ada through the Marlin MXFP4 MoE kernel (weight-only), not the Hopper / Blackwell MXFP4 path",
    source: 'vLLM gpt-oss recipe: "GPT-OSS works on Ampere devices by default, using the TRITON_ATTN attention backend and Marlin MXFP4 MoE"; the quantization table footnote: "Turing does not support Marlin MXFP4" (docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)'
  },
  {
    method: "nvfp4",
    nativeSm: 100,
    fallbackSm: 75,
    fallbackNote: "FP4 tensor cores are Blackwell (sm100) only; elsewhere the weights run weight-only through the Marlin FP4 kernel",
    source: `${VLLM_QUANT_TABLE}: Marlin (GPTQ/AWQ/FP8/FP4) supported from Turing`
  },
  { method: "awq", nativeSm: 75, fallbackSm: 75, fallbackNote: null, source: `${VLLM_QUANT_TABLE}: AWQ not on Volta (sm70)` },
  { method: "gptq", nativeSm: 70, fallbackSm: 70, fallbackNote: null, source: `${VLLM_QUANT_TABLE}: GPTQ Volta and later` },
  { method: "marlin", nativeSm: 75, fallbackSm: 75, fallbackNote: null, source: `${VLLM_QUANT_TABLE}: Marlin (GPTQ/AWQ/FP8/FP4) not on Volta` },
  { method: "int8", nativeSm: 75, fallbackSm: 75, fallbackNote: null, source: `${VLLM_QUANT_TABLE}: llm-compressor INT8 (W8A8) not on Volta` },
  { method: "bitsandbytes", nativeSm: 70, fallbackSm: 70, fallbackNote: null, source: `${VLLM_QUANT_TABLE}: bitsandbytes Volta and later` },
  { method: "gguf", nativeSm: 70, fallbackSm: 70, fallbackNote: null, source: `${VLLM_QUANT_TABLE}: GGUF Volta and later` }
];
var BF16_MIN_SM = 80;
var BF16_SOURCE = 'bf16 tensor cores start at Ampere (sm80); vLLM refuses an explicit --dtype bfloat16 below it ("Bfloat16 is only supported on GPUs with compute capability of at least 8.0")';
var GGUF_BPW = {
  IQ1_S: 2.0042,
  IQ1_M: 2.146,
  IQ2_XXS: 2.3824,
  IQ2_XS: 2.5882,
  IQ2_S: 2.7403,
  IQ2_M: 2.9294,
  IQ3_XXS: 3.2548,
  IQ3_XS: 3.4977,
  IQ3_S: 3.6606,
  IQ3_M: 3.7628,
  IQ4_XS: 4.4597,
  IQ4_NL: 4.6818,
  Q2_K_S: 2.9697,
  Q2_K: 3.1593,
  Q3_K_S: 3.6429,
  Q3_K_M: 3.996,
  Q3_K_L: 4.2979,
  Q4_K_S: 4.6672,
  Q4_K_M: 4.8944,
  Q5_K_S: 5.5704,
  Q5_K_M: 5.7036,
  Q6_K: 6.5633,
  Q8_0: 8.5008,
  F16: 16.0005,
  BF16: 16,
  F32: 32,
  MXFP4: 4.25
};
var GGUF_BPW_SOURCE = "llama.cpp tools/quantize/README.md, bits/weight measured on meta-llama/Llama-3.1-8B";
var GGUF_KV_BYTES = {
  f32: 4,
  f16: 2,
  bf16: 2,
  q8_0: 34 / 32,
  q4_0: 18 / 32,
  q4_1: 20 / 32,
  iq4_nl: 18 / 32,
  q5_0: 22 / 32,
  q5_1: 24 / 32
};
var ENGINE_SAMPLER_DEFAULTS = {
  // OpenAI-compatible defaults used when --generation-config vllm turns the model's file off.
  vllm: { temperature: 1, top_p: 1, top_k: 0, repetition_penalty: 1, source: "vLLM SamplingParams defaults (temperature 1.0, top_p 1.0, top_k off, repetition_penalty 1.0)" },
  sglang: { temperature: 1, top_p: 1, top_k: 0, repetition_penalty: 1, source: "SGLang sampling defaults when not taken from the model (temperature 1.0, top_p 1.0, top_k off)" },
  llamacpp: { temperature: 0.8, top_p: 0.95, top_k: 40, repetition_penalty: 1, source: "llama.cpp tools/server/README.md: --temp 0.80, --top-k 40, --top-p 0.95, --repeat-penalty 1.00 (and --min-p 0.05)" }
};
var KNOWN_SERVE_ISSUES = [
  {
    id: "gpt-oss-jinja-vs-harmony",
    match: /gpt[_-]?oss/i,
    engines: ["sglang", "llamacpp"],
    severity: "warn",
    title: "gpt-oss: the Jinja chat template in the repo was reported to diverge from the harmony training format",
    detail: "Same weights, same payload, same harness: 0.400 on vllm-mlx against 0.650 on Ollama on a 20-task agentic benchmark. The bundled template differs from harmony in developer-block order, tool-call header placement and tool-result wrapping. An engine that renders the Jinja template inherits that; vLLM renders gpt-oss through openai-harmony instead.",
    fix: "Render through openai-harmony, or diff a rendered 5-message conversation with tool calls against harmony's output before serving.",
    source: "github.com/waybarrios/vllm-mlx/issues/568; vLLM gpt-oss recipe (openai-harmony rendering)"
  },
  {
    id: "kimi-k2-tool-calls",
    match: /kimi[_-]?k2/i,
    engines: ["vllm", "sglang", "llamacpp"],
    severity: "warn",
    title: "Kimi K2: tool-call accuracy and decoding parameters vary by host",
    detail: "Moonshot's K2 Vendor Verifier measured tool-call schema accuracy from 100% (Moonshot) down to 87.22% (a vLLM deployment) for kimi-k2-thinking, and says a significant share of discrepancies came from decoding-parameter misuse.",
    fix: "Set the K2 tool-call parser (--tool-call-parser kimi_k2), keep the model card's temperature, and run the vendor verifier before taking traffic.",
    source: "github.com/MoonshotAI/K2-Vendor-Verifier; kimi.com/blog/kimi-vendor-verifier"
  },
  {
    id: "qwen3-greedy",
    match: /^qwen3/i,
    engines: ["vllm", "sglang", "llamacpp"],
    severity: "warn",
    title: "Qwen3 is served with temperature 0 (greedy)",
    detail: "The Qwen3 model card says not to use greedy decoding in thinking mode: it can cause performance degradation and endless repetition.",
    fix: "Use the model card's values (thinking: temperature 0.6, top_p 0.95, top_k 20).",
    source: 'Qwen3 model card, "Best Practices" (huggingface.co/Qwen/Qwen3-8B)',
    when: ({ sampler }) => sampler.temperature === 0
  }
];
function tokenizeCommand(cmd) {
  const out = [];
  let cur = "";
  let has = false;
  let q = null;
  const s = cmd.replace(/\\\r?\n/g, " ");
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (q) {
      if (ch === q) {
        q = null;
        continue;
      }
      if (ch === "\\" && q === '"' && i + 1 < s.length && /["\\$`]/.test(s[i + 1])) {
        cur += s[++i];
        continue;
      }
      cur += ch;
      continue;
    }
    if (ch === '"' || ch === "'") {
      q = ch;
      has = true;
      continue;
    }
    if (ch === "\\" && i + 1 < s.length) {
      cur += s[++i];
      has = true;
      continue;
    }
    if (/\s/.test(ch)) {
      if (has || cur) {
        out.push(cur);
        cur = "";
        has = false;
      }
      continue;
    }
    cur += ch;
    has = true;
  }
  if (has || cur) out.push(cur);
  return out;
}
function detectEngine(tokens) {
  const joined = tokens.join(" ");
  if (/\bsglang(\.launch_server|\s+serve)\b|sglang\.launch_server/.test(joined)) return "sglang";
  if (/(^|\s|\/)vllm\s+serve\b|vllm\.entrypoints/.test(joined)) return "vllm";
  if (/(^|\s|\/)(llama-server|llama-cli|llama-run|server)(\s|$)/.test(joined) && /(^|\s)(-m|--model|-hf|-hfr|--hf-repo)(\s|=)/.test(joined)) return "llamacpp";
  if (/(^|\s|\/)llama-(server|cli)(\s|$)/.test(joined)) return "llamacpp";
  return null;
}
function readTokenCount(v) {
  if (typeof v === "number") return Number.isInteger(v) && v > 0 ? v : null;
  const m = /^\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*$/.exec(String(v ?? ""));
  if (!m) return null;
  const mult = m[2] === "" ? 1 : /k/i.test(m[2]) ? 1024 : 1024 * 1024;
  const n2 = Math.round(Number(m[1]) * mult);
  return n2 > 0 ? n2 : null;
}
function readHumanBytes(v) {
  const m = /^\s*(\d+(?:\.\d+)?)\s*([kKmMgGtT]?)\s*$/.exec(String(v ?? ""));
  if (!m) return null;
  const pow = { "": 0, k: 1, m: 2, g: 3, t: 4 }[m[2].toLowerCase()];
  const base = m[2] && m[2] === m[2].toUpperCase() ? 1024 : 1e3;
  const n2 = Math.round(Number(m[1]) * base ** pow);
  return n2 > 0 ? n2 : null;
}
var num6 = (v) => {
  const n2 = typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
  return Number.isFinite(n2) ? n2 : null;
};
var posInt3 = (v) => {
  const n2 = num6(v);
  return n2 != null && Number.isInteger(n2) && n2 > 0 ? n2 : null;
};
function parseJsonObject(v) {
  if (v == null) return null;
  try {
    const o = JSON.parse(v);
    return o && typeof o === "object" && !Array.isArray(o) ? o : null;
  } catch {
    return null;
  }
}
var BOOL_FLAGS = /* @__PURE__ */ new Set([
  "--enable-expert-parallel",
  "--enable-auto-tool-choice",
  "--trust-remote-code",
  "--enforce-eager",
  "--enable-prefix-caching",
  "--no-enable-prefix-caching",
  "--enable-chunked-prefill",
  "--disable-log-requests",
  "--disable-log-stats",
  "--enable-reasoning",
  "--enable-ep-moe",
  "--enable-dp-attention",
  "--disable-radix-cache",
  "--disable-cuda-graph",
  "--enable-torch-compile",
  "--enable-lora",
  "--jinja",
  "--no-jinja",
  "-cb",
  "--cont-batching",
  "-nocb",
  "--no-cont-batching",
  "-kvu",
  "--kv-unified",
  "-no-kvu",
  "--no-kv-unified",
  "--swa-full",
  "--mlock",
  "--no-mmap",
  "--verbose",
  "-v",
  "--metrics",
  "--embedding",
  "--embeddings",
  "--async-scheduling",
  "--skip-mm-profiling"
]);
function parseServeCommand(cmd, engineHint) {
  const tokens = Array.isArray(cmd) ? cmd : tokenizeCommand(cmd);
  const engine = engineHint ?? detectEngine(tokens);
  const flags = /* @__PURE__ */ new Map();
  const positional = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t === "-m" && i > 0 && /(^|\/)python[\d.]*$/.test(tokens[i - 1])) {
      positional.push(tokens[++i] ?? "");
      continue;
    }
    if (/^--?[A-Za-z]/.test(t)) {
      const eq = t.indexOf("=");
      if (eq > 0) {
        flags.set(t.slice(0, eq), t.slice(eq + 1));
        continue;
      }
      const next = tokens[i + 1];
      if (BOOL_FLAGS.has(t) || next == null || /^--?[A-Za-z]/.test(next)) {
        flags.set(t, true);
        continue;
      }
      flags.set(t, next);
      i++;
      continue;
    }
    positional.push(t);
  }
  const used = /* @__PURE__ */ new Set();
  const get = (...names) => {
    for (const n2 of names) {
      if (flags.has(n2)) {
        used.add(n2);
        const v = flags.get(n2);
        return v === true ? "true" : v;
      }
    }
    return void 0;
  };
  const has = (...names) => names.some((n2) => {
    if (flags.has(n2)) {
      used.add(n2);
      return true;
    }
    return false;
  });
  const args = {};
  const setNum = (k, v, reader = posInt3) => {
    if (v == null) return;
    const n2 = reader(v);
    if (n2 != null) args[k] = n2;
  };
  if (engine === "vllm" || engine === "sglang") {
    if (engine === "vllm") {
      const si = positional.findIndex((p) => p === "serve");
      const pm = si >= 0 ? positional[si + 1] : void 0;
      args.model = get("--model") ?? pm;
      setNum("tp", get("--tensor-parallel-size", "-tp"));
      setNum("pp", get("--pipeline-parallel-size", "-pp"));
      setNum("dp", get("--data-parallel-size", "-dp"));
      if (has("--enable-expert-parallel")) args.expertParallel = true;
      setNum("maxModelLen", get("--max-model-len"), readTokenCount);
      setNum("maxNumSeqs", get("--max-num-seqs"));
      setNum("gpuMemoryUtilization", get("--gpu-memory-utilization"), num6);
      const rope = parseJsonObject(get("--rope-scaling"));
      const hfo = parseJsonObject(get("--hf-overrides"));
      if (rope) args.ropeScaling = rope;
      if (hfo?.rope_scaling && typeof hfo.rope_scaling === "object") args.ropeScaling = hfo.rope_scaling;
      if (hfo && posInt3(hfo.max_position_embeddings) != null) args.maxPositionOverride = posInt3(hfo.max_position_embeddings);
      args.generationConfigMode = get("--generation-config");
      const ogc = parseJsonObject(get("--override-generation-config"));
      if (ogc) args.samplerFlags = readSampler(ogc);
      if (has("--enable-auto-tool-choice")) args.enableAutoToolChoice = true;
      setNum("maxNumBatchedTokens", get("--max-num-batched-tokens"), readTokenCount);
      if (has("--enforce-eager")) args.enforceEager = true;
      setNum("blockSize", get("--block-size"));
      setNum("kvCacheMemoryBytes", get("--kv-cache-memory-bytes"), readHumanBytes);
      const lim = get("--limit-mm-per-prompt");
      if (lim != null) {
        const o = parseJsonObject(lim) ?? Object.fromEntries(lim.split(",").map((kv) => kv.split("=")).filter((x) => x.length === 2).map(([k, v]) => [k.trim(), Number(v)]));
        const out = {};
        for (const [k, v] of Object.entries(o)) {
          const n2 = typeof v === "object" && v ? Number(v.count ?? NaN) : Number(v);
          if (Number.isFinite(n2) && n2 >= 0) out[k] = n2;
        }
        if (Object.keys(out).length) args.limitMmPerPrompt = out;
      }
      const mpk = parseJsonObject(get("--mm-processor-kwargs"));
      if (mpk) args.mmProcessorKwargs = mpk;
      if (has("--skip-mm-profiling")) args.skipMmProfiling = true;
      const mmTp = get("--mm-encoder-tp-mode");
      if (mmTp) args.mmEncoderTpMode = mmTp.trim().toLowerCase();
      const cc = parseJsonObject(get("--compilation-config", "-cc", "-O"));
      if (cc) {
        if (Array.isArray(cc.cudagraph_capture_sizes)) args.cudagraphCaptureSizes = cc.cudagraph_capture_sizes.map(Number).filter((n2) => Number.isInteger(n2) && n2 > 0);
        if (posInt3(cc.max_cudagraph_capture_size) != null) args.maxCudagraphCaptureSize = posInt3(cc.max_cudagraph_capture_size);
        if (cc.cudagraph_mode != null) args.cudagraphMode = String(cc.cudagraph_mode).toUpperCase();
      }
      const spec = parseJsonObject(get("--speculative-config"));
      if (spec && posInt3(spec.num_speculative_tokens) != null) args.numSpeculativeTokens = posInt3(spec.num_speculative_tokens);
    } else {
      args.model = get("--model-path", "--model");
      setNum("tp", get("--tp-size", "--tp", "--tensor-parallel-size"));
      setNum("pp", get("--pp-size", "--pipeline-parallel-size"));
      setNum("dp", get("--dp-size", "--dp", "--data-parallel-size"));
      setNum("epSize", get("--ep-size", "--ep", "--expert-parallel-size"));
      if (has("--enable-ep-moe") || args.epSize != null && args.epSize > 1) args.expertParallel = true;
      setNum("maxModelLen", get("--context-length"), readTokenCount);
      setNum("maxNumSeqs", get("--max-running-requests"));
      setNum("memFractionStatic", get("--mem-fraction-static"), num6);
      const over = parseJsonObject(get("--json-model-override-args"));
      if (over?.rope_scaling && typeof over.rope_scaling === "object") args.ropeScaling = over.rope_scaling;
      if (over && posInt3(over.max_position_embeddings) != null) args.maxPositionOverride = posInt3(over.max_position_embeddings);
      args.generationConfigMode = get("--sampling-defaults");
      setNum("chunkedPrefillSize", get("--chunked-prefill-size"), num6);
      if (has("--disable-cuda-graph")) args.enforceEager = true;
      if (has("--enable-dp-attention")) args.dpAttention = true;
      if (/deepep/i.test(get("--moe-a2a-backend") ?? "")) args.deepep = true;
    }
    args.dtype = get("--dtype");
    args.quantization = get("--quantization", "-q");
    args.kvCacheDtype = get("--kv-cache-dtype");
    args.chatTemplate = get("--chat-template");
    args.toolCallParser = get("--tool-call-parser");
    args.reasoningParser = get("--reasoning-parser");
  } else if (engine === "llamacpp") {
    const m = get("-m", "--model");
    const hf = get("-hf", "-hfr", "--hf-repo");
    const hff = get("-hff", "--hf-file");
    args.model = m ?? hf;
    args.ggufFile = hff ?? m ?? (hf ? hf.includes(":") ? hf : `${hf}:Q4_K_M` : void 0);
    const c = get("-c", "--ctx-size");
    if (c != null) {
      const n2 = num6(c);
      if (n2 != null && n2 >= 0) args.ctxSize = n2;
    }
    const ngl = get("-ngl", "--gpu-layers", "--n-gpu-layers");
    if (ngl != null) args.nGpuLayers = ngl === "all" || ngl === "auto" ? ngl : posInt3(ngl) ?? (num6(ngl) === 0 ? 0 : "auto");
    const np = get("-np", "--parallel", "--n-parallel");
    if (np != null) {
      const n2 = num6(np);
      if (n2 != null && n2 > 0) args.nParallel = n2;
    }
    if (has("-kvu", "--kv-unified")) args.kvUnified = true;
    if (has("-no-kvu", "--no-kv-unified")) args.kvUnified = false;
    args.cacheTypeK = get("-ctk", "--cache-type-k");
    args.cacheTypeV = get("-ctv", "--cache-type-v");
    args.ropeScalingType = get("--rope-scaling");
    setNum("ropeScale", get("--rope-scale"), num6);
    setNum("yarnOrigCtx", get("--yarn-orig-ctx"));
    args.chatTemplate = get("--chat-template", "--chat-template-file");
    const s = {};
    const t = num6(get("--temp", "--temperature"));
    if (t != null) s.temperature = t;
    const tp = num6(get("--top-p"));
    if (tp != null) s.top_p = tp;
    const tk = num6(get("--top-k"));
    if (tk != null) s.top_k = tk;
    const rp = num6(get("--repeat-penalty"));
    if (rp != null) s.repetition_penalty = rp;
    if (Object.keys(s).length) args.samplerFlags = s;
    setNum("nUbatch", get("-ub", "--ubatch-size"));
    if (has("--swa-full")) args.swaFull = true;
  }
  has("--trust-remote-code");
  if (flags.size) args.flags = [...flags.keys()];
  for (const k of Object.keys(args)) if (args[k] == null) delete args[k];
  const ignored = [...flags.keys()].filter((f) => !used.has(f));
  return { engine, args, ignored };
}
function readSampler(raw) {
  const r = raw && typeof raw === "object" ? raw : {};
  const out = {};
  const pick = (k, ...names) => {
    for (const n2 of names) {
      const v = num6(r[n2]);
      if (v != null) {
        out[k] = v;
        return;
      }
    }
  };
  pick("temperature", "temperature", "temp");
  pick("top_p", "top_p", "topP");
  pick("top_k", "top_k", "topK");
  pick("repetition_penalty", "repetition_penalty", "repetitionPenalty", "repeat_penalty");
  return out;
}
function readServeSpec(raw) {
  if (raw == null || typeof raw !== "object") return { error: "`serve` must be an object." };
  const r = raw;
  const hint = r.engine == null ? void 0 : String(r.engine).toLowerCase().replace(/[^a-z]/g, "");
  const engineHint = hint === "llamacpp" || hint === "llama" ? "llamacpp" : hint === "vllm" || hint === "sglang" ? hint : hint == null ? void 0 : null;
  if (engineHint === null) return { error: "`serve.engine` must be one of vllm, sglang, llamacpp." };
  const parsed = typeof r.command === "string" ? parseServeCommand(r.command, engineHint) : { engine: engineHint ?? null, args: {}, ignored: [] };
  const engine = parsed.engine;
  if (!engine) return { error: "Could not tell the engine from `serve.command`; pass `serve.engine` (vllm, sglang or llamacpp)." };
  const args = { ...parsed.args, ...r.args && typeof r.args === "object" ? r.args : {} };
  const gpus = r.gpus == null ? void 0 : posInt3(r.gpus);
  if (r.gpus != null && gpus == null) return { error: "`serve.gpus` must be a positive integer." };
  const gpu = r.gpu == null ? void 0 : String(r.gpu);
  if (gpu != null && !resolveServeGpu(gpu)) return { error: `\`serve.gpu\` "${gpu}" is not a card id or an unambiguous prefix (h100, h200, b200, a100-80g, a100-40g, l40s, l4, a10g, t4, v100, ...).` };
  const obj = (v) => v && typeof v === "object" && !Array.isArray(v) ? v : null;
  return {
    engine,
    args,
    ignored: parsed.ignored,
    gpu,
    gpus: gpus ?? void 0,
    sampler: r.sampler != null ? readSampler(r.sampler) : void 0,
    tokenizerConfig: obj(r.tokenizer_config ?? r.tokenizerConfig),
    chatTemplateFile: typeof (r.chat_template ?? r.chatTemplate) === "string" ? String(r.chat_template ?? r.chatTemplate) : null,
    generationConfig: obj(r.generation_config ?? r.generationConfig),
    modelId: typeof (r.model_id ?? r.modelId) === "string" ? String(r.model_id ?? r.modelId) : void 0
  };
}
function normalizeQuant(q) {
  if (q == null) return null;
  const s = String(q).toLowerCase().replace(/-/g, "_");
  if (!s || s === "none" || s === "null") return null;
  if (/^fp8|fbgemm_fp8|^ptpc_fp8|^modelopt$/.test(s)) return "fp8";
  if (/mxfp4/.test(s)) return "mxfp4";
  if (/nvfp4|modelopt_fp4|fp4/.test(s)) return "nvfp4";
  if (/^awq_marlin|^gptq_marlin|^marlin|machete/.test(s)) return "marlin";
  if (/^awq/.test(s)) return "awq";
  if (/^gptq/.test(s)) return "gptq";
  if (/bitsandbytes|bnb/.test(s)) return "bitsandbytes";
  if (/gguf/.test(s)) return "gguf";
  if (/int8|w8a8|smoothquant/.test(s)) return "int8";
  return s;
}
var MODELOPT_METHOD = { FP8: "modelopt", FP8_PER_CHANNEL_PER_TOKEN: "modelopt", FP8_PB_WO: "modelopt", NVFP4: "modelopt_fp4" };
function modelOptKvDtype(qc, raw) {
  if (!raw || !raw.toLowerCase().startsWith("modelopt")) return null;
  const inner = qc.quantization && typeof qc.quantization === "object" ? qc.quantization : qc;
  let kv = inner.kv_cache_scheme ?? qc.kv_cache_scheme ?? inner.kv_cache_quant_algo ?? qc.kv_cache_quant_algo;
  if (kv && typeof kv === "object") {
    const k = kv;
    kv = k.dynamic === false && k.num_bits === 8 && k.type === "float" ? "fp8" : k.num_bits === 4 && k.type === "float" ? "nvfp4" : null;
  }
  if (typeof kv !== "string") return null;
  return { fp8: "fp8_e4m3", nvfp4: "nvfp4" }[kv.toLowerCase()] ?? null;
}
function checkpointQuant(config) {
  const qc = config?.quantization_config ?? config?.compression_config ?? (config ? textConfigOf(config).quantization_config : null) ?? null;
  if (!qc || typeof qc !== "object") return { method: null, raw: null, bits: null, blockN: null };
  let raw = qc.quant_method != null ? String(qc.quant_method) : null;
  const inner = qc.quantization && typeof qc.quantization === "object" ? qc.quantization : null;
  const producer = qc.producer && typeof qc.producer === "object" ? qc.producer.name : null;
  const algo = String((producer === "modelopt" || inner && "modelopt_quant_config" in inner ? inner?.quant_algo : raw?.toLowerCase().startsWith("modelopt") ? qc.quant_algo ?? inner?.quant_algo : null) ?? "").toUpperCase();
  if (algo && MODELOPT_METHOD[algo]) raw = MODELOPT_METHOD[algo];
  let bits = posInt3(qc.bits) ?? posInt3(qc.w_bit);
  if (raw === "compressed-tensors") {
    const groups = qc.config_groups ?? {};
    const w = Object.values(groups)[0]?.weights;
    if (w?.num_bits === 8 && w.type === "float") raw = "fp8";
    else if (w?.num_bits === 4 && w.type === "float") raw = "nvfp4";
    else if (w?.num_bits === 8) raw = "int8";
    else if (w?.num_bits === 4) {
      raw = "gptq";
      bits = 4;
    }
  }
  if (raw === "bitsandbytes") bits = qc.load_in_8bit === true ? 8 : 4;
  const wbs = Array.isArray(qc.weight_block_size) ? qc.weight_block_size.map(posInt3) : null;
  const store = qc.store_dtype != null ? normalizeQuant(String(qc.store_dtype)) : null;
  return { method: normalizeQuant(raw), raw, bits, blockN: wbs?.[0] ?? null, store, kvDtype: modelOptKvDtype(qc, raw) };
}
var isMoe = (c) => /moe/i.test(c.type);
var isLmHead = (c, comps) => c.type === "linear" && comps.filter((x) => x.type === "linear").pop() === c;
function paramSplit(model) {
  let total = 0, expert = 0, edge = 0;
  const comps = model.components ?? [];
  for (const c of comps) {
    const p = estimateLayerParams(c.type, c.params, c.inputShape);
    total += p;
    if (isMoe(c)) expert += p;
    if (c.type === "embedding" || isLmHead(c, comps)) edge += p;
  }
  return { total, expert, edge };
}
var DTYPE_BYTES = { bf16: 2, fp16: 2, fp32: 4 };
function normDtype(d) {
  if (d == null) return null;
  const s = String(d).toLowerCase();
  if (s === "auto") return "auto";
  if (/bfloat16|bf16/.test(s)) return "bf16";
  if (/float16|half|fp16/.test(s)) return "fp16";
  if (/float32|fp32|float$/.test(s)) return "fp32";
  return null;
}
function ggufQuantOf(file) {
  if (!file) return null;
  const s = file.toUpperCase();
  const keys = Object.keys(GGUF_BPW).sort((a, b) => b.length - a.length);
  for (const k of keys) {
    const re = new RegExp(`(^|[^A-Z0-9])${k.replace(/_/g, "[_-]")}([^A-Z0-9]|$)`);
    if (re.test(s)) return k;
  }
  return null;
}
function derivedMaxLenRaw(config) {
  if (!config) return null;
  const c = textConfigOf(config);
  return posInt3(c.max_position_embeddings) ?? posInt3(c.n_positions) ?? null;
}
var YARN_FAMILY = /* @__PURE__ */ new Set(["yarn", "deepseek_yarn", "deepseek_llama_scaling", "telechat3-yarn"]);
function maxLenRuleFor(engine, version) {
  if (engine === "sglang") return "sglang";
  if (engine === "vllm" && version && !vAtLeast(version, [0, 30, 0])) return "vllm-legacy";
  return "vllm";
}
function derivedMaxLen(config, ropeOverride, maxPosOverride, graphMax, rule = "vllm") {
  const c = config ?? {};
  const native = maxPosOverride ?? posInt3(c.max_position_embeddings) ?? posInt3(c.n_positions) ?? posInt3(c.max_seq_len) ?? posInt3(c.seq_length) ?? graphMax ?? null;
  const rope = ropeOverride ?? (c.rope_scaling && typeof c.rope_scaling === "object" ? c.rope_scaling : null);
  if (native == null) return { len: null, native: null, rope, scaledByFactor: false };
  if (!rope) return { len: native, native, rope: null, scaledByFactor: false };
  const unscaled = { len: native, native, rope, scaledByFactor: false };
  const type = String(rope.rope_type ?? rope.type ?? "").toLowerCase();
  const factor = num6(rope.factor);
  if (rule === "sglang") {
    if (!factor || rope.original_max_position_embeddings != null || type === "llama3") return unscaled;
    return { len: Math.floor(native * factor), native, rope, scaledByFactor: true };
  }
  if (!factor || ["llama3", "longrope", "su", "mrope", "default"].includes(type)) return unscaled;
  if (!ropeOverride && /gemma3/.test(String(c.model_type ?? ""))) return unscaled;
  if (rule === "vllm" && YARN_FAMILY.has(type)) return unscaled;
  const base = type === "yarn" ? posInt3(rope.original_max_position_embeddings) ?? native : native;
  return { len: Math.floor(base * factor), native, rope, scaledByFactor: true };
}
function chatTemplateOf(side) {
  if (typeof side.chatTemplateFile === "string" && side.chatTemplateFile.trim()) return side.chatTemplateFile;
  const t = side.tokenizerConfig?.chat_template;
  if (typeof t === "string") return t;
  if (Array.isArray(t)) {
    const d = t.find((x) => x.name === "default") ?? t[0];
    return typeof d?.template === "string" ? d.template : null;
  }
  return null;
}
function templateHash(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}
var VLLM_SRC = "vllm-project/vllm@83c9a547bf";
var SGLANG_SRC = "sgl-project/sglang@80914b7c9b";
var TRTLLM_SRC = "NVIDIA/TensorRT-LLM@8e4f13322e";
var LLAMACPP_SRC = "ggml-org/llama.cpp@547208dca";
var cdiv = (a, b) => Math.ceil(a / b);
var GiB = 1024 ** 3;
var CARD_NVSMI_MIB = {
  "b200": { mib: 183359, approx: true },
  "h200": { mib: 143771 },
  "h100-sxm": { mib: 81559 },
  "a100-80g": { mib: 81920 },
  "a100-40g": { mib: 40960 },
  "l40s": { mib: 46068 },
  "l4": { mib: 23034 },
  "a10g": { mib: 23028 },
  "t4": { mib: 15360 },
  "v100": { mib: 16384, approx: true },
  "rtx-4090": { mib: 24564 },
  "rtx-3090": { mib: 24576 },
  "a40": { mib: 46068, approx: true },
  "gh200-96g": { mib: 97871, approx: true },
  "gh200-144g": { mib: 146831, approx: true },
  "rtx-pro-6000": { mib: 97887, approx: true }
};
var CARD_TORCH_TOTAL_GIB = {
  "h100-sxm": { gib: 79.19, source: 'vLLM log, llm-d-planner accuracy report (H100-80GB "physical 79.19 GiB")' },
  "a100-80g": { gib: 79.15, source: 'vLLM log "total_gpu_memory (79.15GiB)" on A100-SXM4-80GB (github.com/vllm-project/vllm/issues/18446)' }
};
var TORCH_GAP_MIB = 81559 - 79.19 * 1024;
function cardTotalGiB(cardId, nominalGb) {
  const at = /@(\d+(?:\.\d+)?)$/.exec(cardId);
  if (at) return { gib: Number(at[1]), source: "given by --gpu-mem" };
  const t = CARD_TORCH_TOTAL_GIB[cardId];
  if (t) return t;
  const n2 = CARD_NVSMI_MIB[cardId];
  if (n2) return { gib: Math.round((n2.mib - TORCH_GAP_MIB) / 1024 * 100) / 100, source: `derived: nvidia-smi ${n2.mib} MiB less the ${Math.round(TORCH_GAP_MIB)} MiB gap measured on H100` };
  return { gib: Math.round(nominalGb * 0.9899 * 100) / 100, source: "derived: nominal GB x 0.9899 (the H100 ratio)" };
}
function readEngineVersion(v) {
  const m = /^\s*v?(\d+)\.(\d+)(?:\.(\d+))?/.exec(String(v ?? ""));
  return m ? [Number(m[1]), Number(m[2]), Number(m[3] ?? 0)] : null;
}
var vAtLeast = (v, w) => {
  for (let i = 0; i < 3; i++) {
    if ((v[i] ?? 0) !== w[i]) return (v[i] ?? 0) > w[i];
  }
  return true;
};
var ASSUMED_VLLM_VERSION = "0.32.0";
var ASSUMED_SGLANG_VERSION = "0.5.22";
function vllmVersionDefaults(version, card, pp) {
  const v20 = vAtLeast(version, [0, 20, 0]);
  const v26 = vAtLeast(version, [0, 26, 0]);
  const totalGib = cardTotalGiB(card.id, card.memoryGB).gib;
  const big = totalGib >= 70 && !/a100/.test(card.id);
  const huge = totalGib >= 160 && vAtLeast(version, [0, 26, 0]);
  return {
    gpuMemoryUtilization: v20 ? 0.92 : 0.9,
    cudaGraphsCounted: v20,
    maxNumBatchedTokens: huge ? 16384 : big ? 8192 : 2048,
    maxNumSeqs: big || huge ? 1024 : 256,
    inFlightBatches: v26 ? pp > 1 ? pp : 2 : 1,
    printsTokensAs: vAtLeast(version, [0, 21, 0]) ? "concurrency" : "blocks",
    linearRecurrentAlign: vAtLeast(version, [0, 15, 0])
  };
}
function textConfigOf(c) {
  if (posInt3(c.num_hidden_layers) != null || posInt3(c.n_layer) != null || posInt3(c.num_layers) != null || posInt3(c.decoder_layers) != null) return c;
  for (const k of ["text_config", "llm_config", "language_config", "text_model_config"]) {
    const t = c[k];
    if (t && typeof t === "object") return t;
  }
  const th = c.thinker_config;
  if (th?.text_config && typeof th.text_config === "object") return th.text_config;
  return c;
}
var DTYPE_BYTES_OF = (s, fallback) => {
  const t = String(s ?? "").toLowerCase();
  if (/float32|fp32/.test(t)) return 4;
  if (/bfloat16|float16|bf16|fp16|half/.test(t)) return 2;
  return fallback;
};
function kvLayoutFromConfig(config, opts) {
  const top = config;
  const c = textConfigOf(config);
  const notes = [];
  const mt = String(top.model_type ?? c.model_type ?? "").toLowerCase();
  const tp = Math.max(1, opts.tp);
  const n2 = posInt3(c.num_hidden_layers) ?? posInt3(c.n_layer) ?? posInt3(c.num_layers) ?? posInt3(c.decoder_layers);
  if (n2 == null) return null;
  const hidden = posInt3(c.hidden_size) ?? posInt3(c.d_model) ?? posInt3(c.n_embd);
  const heads = posInt3(c.num_attention_heads) ?? posInt3(c.n_head) ?? posInt3(c.decoder_attention_heads);
  const kvHeads = posInt3(c.num_key_value_heads) ?? (c.multi_query === true ? 1 : null) ?? posInt3(c.num_kv_heads) ?? heads;
  const headDim = posInt3(c.head_dim) ?? (hidden && heads ? hidden / heads : null);
  const kvRank = kvHeads != null ? Math.max(1, Math.floor(kvHeads / tp)) : null;
  const replicated = kvHeads != null && tp > kvHeads;
  const attnBytes = kvRank != null && headDim != null ? kvRank * headDim * 2 * opts.kvBytes : null;
  if (Array.isArray(c.compress_ratios) && posInt3(c.sliding_window) != null && headDim != null) {
    const ratios = c.compress_ratios.map((x) => Number(x) || 0);
    const swin = posInt3(c.sliding_window);
    const rec = headDim * opts.kvBytes;
    const sources = new Set(Array.isArray(c.kv_source_layer_ids) ? c.kv_source_layer_ids.map(Number) : []);
    const idxSources = new Set(Array.isArray(c.index_source_layer_ids) ? c.index_source_layer_ids.map(Number) : []);
    const idxDim = posInt3(c.index_head_dim);
    const idxRec = idxDim != null ? idxDim + Math.floor(idxDim / 128) * 4 : null;
    const out = [];
    for (let i = 0; i < n2; i++) {
      out.push({ kind: "sliding", bytesPerToken: rec, window: swin });
      const r = ratios[i] ?? 0;
      if (r > 0 && sources.has(i)) out.push({ kind: "mla", bytesPerToken: rec, tokensPerState: r });
      if (r > 0 && idxSources.has(i) && idxRec != null) out.push({ kind: "mla", bytesPerToken: idxRec, tokensPerState: r });
    }
    const comp = out.filter((l) => l.kind === "mla");
    notes.push(`${mt}: ${n2} sliding-window records (window ${swin}, ${headDim} values, one KV head on every rank) plus ${comp.length} compressed full-length caches on the kv / index source layers (compress ratios ${[...new Set(comp.map((l) => l.tokensPerState))].join(", ")}); packed record widths and page alignment are not modelled (${VLLM_SRC.split("@")[0]} vllm/models/deepseek_v41/attention.py)`);
    return { layers: out, notes, kvHeadsPerRank: 1, replicated: false, recurrentFamily: null };
  }
  const kvLora = posInt3(c.kv_lora_rank);
  const rope = posInt3(c.qk_rope_head_dim);
  const mlaBytes = kvLora != null ? (kvLora + (rope ?? 0)) * opts.kvBytes : null;
  if (kvLora != null) notes.push(`MLA: ${kvLora} + ${rope ?? 0} latent values per token per layer, the same on every TP rank (${VLLM_SRC} MLAAttentionSpec, head_size_v = 0)`);
  const ssmBytes = DTYPE_BYTES_OF(c.mamba_ssm_cache_dtype, opts.modelBytes);
  const mambaState = () => {
    const conv = posInt3(c.conv_kernel) ?? posInt3(c.mamba_d_conv) ?? posInt3(c.d_conv) ?? posInt3(c.linear_conv_kernel_dim) ?? 4;
    const kda = c.linear_attn_config;
    if (kda && posInt3(kda.num_heads) && posInt3(kda.head_dim)) {
      const h = posInt3(kda.num_heads), d = posInt3(kda.head_dim), k = posInt3(kda.short_conv_kernel_size) ?? 4;
      return 3 * h * d / tp * (k - 1) * opts.modelBytes + h / tp * d * d * ssmBytes;
    }
    const nk = posInt3(c.linear_num_key_heads), nv = posInt3(c.linear_num_value_heads);
    const hk = posInt3(c.linear_key_head_dim), hv = posInt3(c.linear_value_head_dim);
    if (nk && nv && hk && hv) {
      const convDim2 = hk * nk * 2 + hv * nv;
      return convDim2 / tp * (conv - 1) * opts.modelBytes + nv / tp * hv * hk * ssmBytes;
    }
    const nh = posInt3(c.mamba_num_heads) ?? posInt3(c.mamba_n_heads) ?? (/mamba2|nemotron_h/.test(mt) ? posInt3(c.num_heads) : null);
    const hd = posInt3(c.mamba_head_dim) ?? posInt3(c.mamba_d_head) ?? (/mamba2/.test(mt) ? posInt3(c.head_dim) : null);
    const state = posInt3(c.ssm_state_size) ?? posInt3(c.mamba_d_state) ?? posInt3(c.state_size) ?? posInt3(c.d_state);
    if (nh && hd && state) {
      let ng = posInt3(c.n_groups) ?? posInt3(c.mamba_n_groups) ?? 1;
      if (ng % tp !== 0) ng = ng + (tp - ng);
      const inter2 = nh * hd;
      return (inter2 + 2 * ng * state) / tp * (conv - 1) * opts.modelBytes + nh / tp * hd * state * ssmBytes;
    }
    const expand = num6(c.mamba_expand) ?? num6(c.expand) ?? 2;
    const inter = posInt3(c.intermediate_size) && /^(falcon_)?mamba$/.test(mt) ? posInt3(c.intermediate_size) : hidden ? hidden * expand : null;
    if (inter && state) return inter / tp * (conv - 1) * opts.modelBytes + inter / tp * state * ssmBytes;
    return null;
  };
  let kinds = null;
  const lt = Array.isArray(c.layer_types) ? c.layer_types.map((x) => String(x).toLowerCase()) : null;
  const sw = posInt3(c.sliding_window) ?? posInt3(c.attention_window);
  const pattern = typeof c.hybrid_override_pattern === "string" ? c.hybrid_override_pattern : null;
  const blockTypes = Array.isArray(c.layers_block_type) ? c.layers_block_type.map((x) => String(x).toLowerCase()) : null;
  if (pattern) {
    kinds = [...pattern].map((ch) => ch === "M" ? "mamba" : ch === "*" ? "attn" : "none");
    notes.push(`hybrid_override_pattern: ${kinds.filter((k) => k === "mamba").length} Mamba-2 layers, ${kinds.filter((k) => k === "attn").length} attention layers, the rest MLP / MoE with no cache`);
  } else if (blockTypes) {
    kinds = blockTypes.map((t) => /mamba|linear/.test(t) ? "mamba" : /attention/.test(t) ? "attn" : "none");
  } else if (lt) {
    kinds = lt.map((t) => /linear|mamba|kda|deltanet/.test(t) ? "mamba" : /sliding/.test(t) ? "sliding" : /chunk/.test(t) ? "chunked" : /attention|full/.test(t) ? "attn" : "none");
  } else if (c.linear_attn_config && Array.isArray(c.linear_attn_config.kda_layers)) {
    const kdaSet = new Set(c.linear_attn_config.kda_layers.map(Number));
    kinds = Array.from({ length: n2 }, (_, i) => kdaSet.has(i + 1) ? "mamba" : "attn");
  } else if (posInt3(c.full_attention_interval) != null && posInt3(c.linear_num_value_heads) != null) {
    const k = posInt3(c.full_attention_interval);
    kinds = Array.from({ length: n2 }, (_, i) => (i + 1) % k === 0 ? "attn" : "mamba");
  } else if (posInt3(c.attn_layer_period) != null) {
    const per = posInt3(c.attn_layer_period), off = num6(c.attn_layer_offset) ?? 0;
    kinds = Array.from({ length: n2 }, (_, i) => i % per === off ? "attn" : "mamba");
  } else if (/^(mamba|mamba2|falcon_mamba)$/.test(mt)) {
    kinds = Array.from({ length: n2 }, () => "mamba");
  } else if (/^gemma2$|cost_wise_gemma/.test(mt) && sw) {
    kinds = Array.from({ length: n2 }, (_, i) => i % 2 === 0 ? "sliding" : "attn");
  } else if (/^gemma3/.test(mt) && sw) {
    const p = posInt3(c.sliding_window_pattern) ?? 6;
    kinds = Array.from({ length: n2 }, (_, i) => (i + 1) % p ? "sliding" : "attn");
  } else if (posInt3(c.attention_chunk_size) != null) {
    const nr = Array.isArray(c.no_rope_layers) ? c.no_rope_layers : null;
    const interval = posInt3(c.no_rope_layer_interval) ?? 4;
    kinds = Array.from({ length: n2 }, (_, i) => (nr ? Number(nr[i]) !== 0 : (i + 1) % interval !== 0) ? "chunked" : "attn");
  } else if (sw && c.use_sliding_window !== false && !/qwen2|qwen3/.test(mt)) {
    kinds = Array.from({ length: n2 }, () => "sliding");
  } else if (sw && c.use_sliding_window === true) {
    const mw = posInt3(c.max_window_layers) ?? 0;
    kinds = Array.from({ length: n2 }, (_, i) => i >= mw ? "sliding" : "attn");
  } else {
    kinds = Array.from({ length: n2 }, () => "attn");
  }
  if (Array.isArray(c.cross_attention_layers)) {
    const cross = new Set(c.cross_attention_layers.map(Number));
    kinds = kinds.map((k, i) => cross.has(i) ? "cross" : k);
  }
  const whisper = /whisper/.test(mt);
  const layers = [];
  const chunk = posInt3(c.attention_chunk_size);
  const visionTokens = (() => {
    const v = top.vision_config ?? {};
    const img = posInt3(v.image_size), patch = posInt3(v.patch_size), tiles = posInt3(v.max_num_tiles);
    return img && patch ? ((img / patch) ** 2 + 1) * (tiles ?? 1) : null;
  })();
  for (const k of kinds) {
    if (k === "none") continue;
    if (k === "mamba") {
      const s = mambaState();
      if (s == null) {
        notes.push("a recurrent layer whose state size the config does not give: counted as 0");
        continue;
      }
      layers.push({ kind: "mamba", bytesPerToken: 0, stateBytes: s });
      continue;
    }
    if (mlaBytes != null) {
      layers.push({ kind: "mla", bytesPerToken: mlaBytes });
      continue;
    }
    if (attnBytes == null) return null;
    if (k === "cross") {
      layers.push({ kind: "cross", bytesPerToken: attnBytes, encoderLen: visionTokens ?? 6404 });
      continue;
    }
    if (k === "sliding" && sw) {
      layers.push({ kind: "sliding", bytesPerToken: attnBytes, window: sw });
      continue;
    }
    if (k === "chunked" && chunk) {
      layers.push({ kind: "chunked", bytesPerToken: attnBytes, window: chunk });
      continue;
    }
    layers.push({ kind: "full", bytesPerToken: attnBytes });
    if (whisper) layers.push({ kind: "cross", bytesPerToken: attnBytes, encoderLen: posInt3(c.max_source_positions) ?? 1500 });
  }
  const count = (k) => layers.filter((l) => l.kind === k).length;
  if (count("sliding")) notes.push(`${count("sliding")} sliding-window layers (window ${sw}): each holds at most the window plus the tokens in flight, not the whole sequence (${VLLM_SRC} SlidingWindowSpec)`);
  if (count("chunked")) notes.push(`${count("chunked")} chunked-local layers (chunk ${chunk}): each holds one chunk plus the tokens in flight (ChunkedLocalAttentionSpec)`);
  if (count("mamba")) notes.push(`${count("mamba")} recurrent layers: a fixed state per sequence, not per token (MambaSpec)`);
  if (count("cross")) notes.push(`${count("cross")} cross-attention layers: vLLM reserves max_num_encoder_input_tokens (default: max_num_batched_tokens) of encoder cache per request, not decoder tokens (CrossAttentionSpec)`);
  if (replicated && mlaBytes == null) notes.push(`TP ${tp} is past ${kvHeads} KV heads: each rank holds 1 full KV head, so KV per GPU stops shrinking at TP ${kvHeads} (vLLM max(1, kv_heads // tp); ${TRTLLM_SRC} heuristics.py max(tp_size, num_key_value_heads))`);
  if (/deepseek_v32|deepseek_v4|kimi_linear|minimax_m[23]/.test(mt)) notes.push(`${mt}: sparse-attention indexer or compressed caches are not modelled; KV here is the dense part only`);
  const linear = posInt3(c.linear_num_value_heads) != null || (lt ?? []).some((x) => /linear|kda|deltanet/.test(x)) || c.linear_attn_config != null;
  return { layers, notes, kvHeadsPerRank: mlaBytes != null ? 1 : kvRank, replicated: replicated && mlaBytes == null, recurrentFamily: count("mamba") ? linear ? "linear" : "mamba" : null };
}
var kvBytesPerTokenOf = (l) => l.layers.reduce((s, x) => s + (x.kind === "mamba" || x.kind === "cross" ? 0 : x.bytesPerToken / (x.tokensPerState ?? 1)), 0);
function vllmKvCapacity(layout, o) {
  const L = layout.layers;
  if (!L.length) return null;
  if (L.some((l) => l.tokensPerState != null)) return mixedPageCapacity(L, o);
  const attn = L.find((l) => l.kind !== "mamba");
  const mambaState = Math.max(0, ...L.filter((l) => l.kind === "mamba").map((l) => l.stateBytes ?? 0));
  let bs = o.blockSize ?? (L.some((l) => l.kind === "mla") ? 64 : 16);
  if (attn && mambaState > 0) {
    const align = Math.max(16, bs);
    bs = Math.max(bs, align * cdiv(mambaState, align * attn.bytesPerToken));
  }
  const page = attn ? bs * attn.bytesPerToken : mambaState;
  if (!(page > 0)) return null;
  const key = (l) => `${l.kind}:${l.window ?? ""}:${l.encoderLen ?? ""}`;
  const buckets = /* @__PURE__ */ new Map();
  for (const l of L) {
    const k = key(l);
    if (!buckets.has(k)) buckets.set(k, []);
    buckets.get(k).push(l);
  }
  const sizes = [...buckets.values()].map((b) => b.length);
  const minN = Math.min(...sizes), maxN = Math.max(...sizes);
  const groupSize = buckets.size === 1 ? maxN : maxN < minN * 1.5 ? maxN : minN;
  const bytesPerBlock = groupSize * page;
  const numBlocks = Math.max(0, Math.floor(o.budgetBytes / bytesPerBlock));
  const groups = [];
  let perReq = 0;
  for (const b of buckets.values()) {
    const spec = b[0];
    const nGroups = cdiv(b.length, groupSize);
    let blocks;
    if (spec.kind === "mamba") blocks = o.mambaAlign ? 2 : 1;
    else if (spec.kind === "sliding") blocks = cdiv(Math.min((spec.window ?? 0) - 1 + o.inFlightTokens, o.maxModelLen), bs) + 1;
    else if (spec.kind === "chunked") blocks = cdiv(Math.min((spec.window ?? 0) + o.inFlightTokens, o.maxModelLen), bs);
    else if (spec.kind === "cross") blocks = cdiv(o.encoderTokens ?? spec.encoderLen ?? 0, bs);
    else blocks = cdiv(o.maxModelLen, bs);
    groups.push({ kind: spec.kind, layers: b.length, blocksPerRequest: blocks * nGroups });
    perReq += blocks * nGroups;
  }
  const maxConcurrency = perReq > 0 ? numBlocks / perReq : 0;
  const nGroupsTotal = groups.reduce((s, g) => s + cdiv(g.layers, groupSize), 0);
  const printedTokens = o.printsTokensAs === "concurrency" ? Math.floor(maxConcurrency * o.maxModelLen) : Math.floor(numBlocks / nGroupsTotal) * bs;
  return { blockSize: bs, bytesPerBlock, numBlocks, groups, blocksPerRequest: perReq, maxConcurrency, printedTokens, poolTokens: numBlocks * bs };
}
function mixedPageCapacity(L, o) {
  const bs = o.blockSize ?? 64;
  const tokensHeld = (l) => l.kind === "sliding" ? Math.min((l.window ?? 0) - 1 + o.inFlightTokens, o.maxModelLen) : cdiv(o.maxModelLen, l.tokensPerState ?? 1);
  const buckets = /* @__PURE__ */ new Map();
  for (const l of L) {
    const k = `${l.kind}:${l.tokensPerState ?? ""}:${l.bytesPerToken}`;
    if (!buckets.has(k)) buckets.set(k, []);
    buckets.get(k).push(l);
  }
  let perReqBytes = 0, perReqBlocks = 0, bytesPerBlock = 0;
  const groups = [];
  for (const b of buckets.values()) {
    const blocks = cdiv(tokensHeld(b[0]), bs) + (b[0].kind === "sliding" ? 1 : 0);
    perReqBytes += b.length * blocks * bs * b[0].bytesPerToken;
    perReqBlocks += blocks;
    bytesPerBlock += b.length * bs * b[0].bytesPerToken;
    groups.push({ kind: b[0].kind, layers: b.length, blocksPerRequest: blocks });
  }
  const maxConcurrency = perReqBytes > 0 ? o.budgetBytes / perReqBytes : 0;
  const numBlocks = Math.max(0, Math.floor(maxConcurrency * perReqBlocks));
  return {
    blockSize: bs,
    bytesPerBlock: perReqBlocks > 0 ? perReqBytes / perReqBlocks : bytesPerBlock,
    numBlocks,
    groups,
    blocksPerRequest: perReqBlocks,
    maxConcurrency,
    printedTokens: Math.floor(maxConcurrency * o.maxModelLen),
    poolTokens: Math.floor(maxConcurrency * o.maxModelLen)
  };
}
function vllmNonKvReserve(config, o) {
  const c = config ? textConfigOf(config) : {};
  const vocab = posInt3(c.vocab_size) ?? posInt3((config ?? {}).vocab_size) ?? 32e3;
  const hidden = posInt3(c.hidden_size) ?? 4096;
  const inter = posInt3(c.intermediate_size) ?? hidden * 4;
  const moeInter = posInt3(c.moe_intermediate_size) ?? (posInt3(c.num_local_experts) || posInt3(c.num_experts) ? inter : null);
  const topk = posInt3(c.num_experts_per_tok) ?? posInt3(c.num_experts_per_token) ?? posInt3(c.moe_topk) ?? 2;
  const T = o.maxNumBatchedTokens;
  const rows = Math.min(o.maxNumSeqs, T);
  const sampler = rows * vocab * 16;
  const dense = T * (3 * inter / o.tp + 2 * hidden) * o.actBytes;
  const moe = moeInter != null ? T * topk * (3 * moeInter / o.tp + hidden) * o.actBytes : 0;
  const lmAct = Math.max(sampler, dense, moe) / GiB;
  const enc = o.encoder ?? null;
  const act = enc ? Math.max(enc.peakGib, enc.outputsGib + lmAct) : lmAct;
  const multiGpu = o.tp >= 2 && o.nvlink !== false;
  const nonTorch = multiGpu ? 2.1 : o.pp > 1 ? 0.07 : 0.27;
  const terms = [
    { what: "weights", gib: o.weightsGib, basis: 'computed from the config (vLLM reports it as "Model loading took")', measuredByEngine: false },
    { what: "activation peak", gib: act, basis: `estimate: the profile run at ${T.toLocaleString("en-US")} batched tokens and ${rows} sampler rows (${sampler >= Math.max(dense, moe) ? `sampler, ${rows} x ${vocab.toLocaleString("en-US")} vocab x 16 bytes` : moe > dense ? "fused-MoE intermediate caches" : "MLP intermediates"})${enc ? `; vision tower: ${enc.items} x ${enc.tokensPerItem}-token image items, peak ${enc.peakGib.toFixed(2)} GiB, outputs ${enc.outputsGib.toFixed(2)} GiB held through the text pass (${enc.peakGib >= enc.outputsGib + lmAct ? "the tower sets the peak" : "the text pass sets the peak"})` : ""}; vLLM measures it`, measuredByEngine: true },
    { what: "non-torch (NCCL, CUDA)", gib: nonTorch, basis: `estimate: ${o.tp >= 2 && !multiGpu ? "TP on a card without NVLink: no custom all-reduce buffers, TP 1 figure used (from two public PCIe logs)" : `llm-d-planner's measured mean for ${multiGpu ? "TP >= 2" : o.pp > 1 ? "PP only" : "TP 1"} on vLLM 0.19 / H100`}; vLLM measures it`, measuredByEngine: true }
  ];
  if (o.cudaGraphsCounted && !o.enforceEager) {
    const g = o.cudaGraph;
    if (g) terms.push({ what: "CUDA graphs", gib: g.gib, basis: g.basis, measuredByEngine: true });
    else terms.push({ what: "CUDA graphs", gib: 1.04, basis: "estimate: mean of the 0.51 to 1.85 GiB llm-d-planner observed; vLLM >= 0.20 captures throwaway graphs and measures it", measuredByEngine: true });
  }
  return terms;
}
function vllmCudaGraphSizes(version, o) {
  const q = 1 + (o.numSpeculativeTokens ?? 0);
  const mode = (o.mode ?? "FULL_AND_PIECEWISE").toUpperCase();
  const blackwell = o.sm != null && o.sm >= 100 && o.sm < 110 && vAtLeast(version, [0, 28, 0]);
  const platformMax = blackwell ? 1024 : 512;
  const mnbt = o.maxNumBatchedTokens;
  let max = o.maxCaptureSize ?? Math.min(o.maxNumSeqs * q * 2, platformMax);
  max = Math.min(mnbt, max);
  let sizes;
  if (o.captureSizes?.length) sizes = [...new Set(o.captureSizes.filter((x) => x <= mnbt))].sort((a, b) => a - b);
  else {
    sizes = [1, 2, 4].filter((x) => x <= max);
    for (let x = 8; x < Math.min(max + 1, 256); x += 8) sizes.push(x);
    for (let x = 256; x <= max; x += 16) sizes.push(x);
    if (vAtLeast(version, [0, 21, 0]) && mnbt <= max && !sizes.includes(mnbt)) sizes.push(mnbt);
    if (vAtLeast(version, [0, 29, 0]) && o.maxCaptureSize == null && q === 1 && o.maxNumSeqs <= max && o.maxNumSeqs <= mnbt) sizes.push(o.maxNumSeqs);
    sizes = [...new Set(sizes)].sort((a, b) => a - b);
  }
  if (q > 1 && sizes.length) {
    const cap = sizes[sizes.length - 1];
    const r = [...new Set(sizes.map((x) => Math.ceil(x / q) * q).filter((x) => x <= cap))].sort((a, b) => a - b);
    sizes = r.length ? r : q <= cap ? [q] : [];
  }
  const maxCaptureSize = sizes[sizes.length - 1] ?? 0;
  if (mode === "NONE") return { piecewise: [], full: [], maxCaptureSize: 0, mode, decodeQueryLen: q };
  const fullDecode = sizes.filter((x) => x <= o.maxNumSeqs * q && x >= q);
  return {
    piecewise: mode === "FULL" || mode === "FULL_DECODE_ONLY" ? [] : sizes,
    full: mode === "PIECEWISE" ? [] : mode === "FULL" ? sizes : fullDecode,
    maxCaptureSize,
    mode,
    decodeQueryLen: q
  };
}
var V2_DEFAULT_ARCHS = [
  { since: [0, 22, 0], archs: ["Qwen3ForCausalLM"] },
  { since: [0, 23, 0], archs: ["LlamaForCausalLM", "MistralForCausalLM", "Qwen3ForCausalLM"] },
  { since: [0, 24, 0], archs: ["Qwen3ForCausalLM", "DeepseekV2ForCausalLM", "Qwen2MoeForCausalLM", "GraniteMoeForCausalLM", "LlamaForCausalLM", "MistralForCausalLM"] },
  { since: [0, 25, 0], archs: ["DeepseekV2ForCausalLM", "Qwen2MoeForCausalLM", "GraniteMoeForCausalLM"] },
  { since: [0, 26, 0], archs: ["DeepseekV2ForCausalLM", "GraniteMoeForCausalLM", "InklingForCausalLM", "InklingForConditionalGeneration", "LongcatFlashNgramForCausalLM", "Qwen2MoeForCausalLM"] },
  { since: [0, 27, 0], archs: ["DeepseekV2ForCausalLM", "GraniteMoeForCausalLM", "InklingForCausalLM", "InklingForConditionalGeneration", "KimiK3ForConditionalGeneration", "LongcatFlashNgramForCausalLM", "Qwen2MoeForCausalLM"] },
  { since: [0, 28, 0], archs: ["DeepseekV2ForCausalLM", "DeepseekV4ForCausalLM", "GraniteMoeForCausalLM", "InklingForCausalLM", "InklingForConditionalGeneration", "KimiK3ForConditionalGeneration", "LongcatFlashNgramForCausalLM", "Qwen2MoeForCausalLM"] }
];
function vllmV2RunnerSkipsGraphEstimate(version, architectures) {
  if (!vAtLeast(version, [0, 22, 0]) || vAtLeast(version, [0, 29, 0])) return false;
  const row = [...V2_DEFAULT_ARCHS].reverse().find((r) => vAtLeast(version, r.since));
  const archs = Array.isArray(architectures) ? architectures.map(String) : [];
  return !!row && archs.some((a) => row.archs.includes(a));
}
var CUDA_GRAPH_FIT = { firstBaseMib: 240, perPiecewiseMibPerLayer: 0, perFullMib: 0, perFullMibPerLayer: 0.2 };
function vllmCudaGraphEstimate(config, o) {
  const a = o.args ?? {};
  const gptOss = Array.isArray(config?.architectures) && config.architectures.includes("GptOssForCausalLM") && a.cudagraphCaptureSizes == null && a.maxCudagraphCaptureSize == null;
  const sizes = vllmCudaGraphSizes(o.version, {
    maxNumSeqs: o.maxNumSeqs,
    maxNumBatchedTokens: o.maxNumBatchedTokens,
    sm: o.sm,
    numSpeculativeTokens: a.numSpeculativeTokens,
    captureSizes: a.cudagraphCaptureSizes,
    maxCaptureSize: a.maxCudagraphCaptureSize ?? (gptOss ? vAtLeast(o.version, [0, 12, 0]) ? 1024 : 992 : void 0),
    mode: a.cudagraphMode
  });
  const words = `PIECEWISE ${sizes.piecewise.length} sizes (largest ${sizes.piecewise[sizes.piecewise.length - 1] ?? 0}), FULL ${sizes.full.length} (largest ${sizes.full[sizes.full.length - 1] ?? 0})`;
  if (config && vllmV2RunnerSkipsGraphEstimate(o.version, config.architectures)) {
    return {
      gib: 0,
      sizes,
      firstGib: 0,
      perGraphGib: { piecewise: 0, full: 0 },
      skipped: "v2-runner",
      basis: `0: this architecture runs on Model Runner V2 by default in this release, whose profile_cudagraph_memory returned 0 until v0.29.0 (vllm-project/vllm#49224); the graphs (${words}) are still captured after the KV cache is sized`
    };
  }
  const c = config ? textConfigOf(config) : {};
  const layers = Math.ceil((posInt3(c.num_hidden_layers) ?? 32) / o.pp);
  const S = Math.max(sizes.piecewise[sizes.piecewise.length - 1] ?? 0, sizes.full[sizes.full.length - 1] ?? 0);
  const F = o.fit ?? CUDA_GRAPH_FIT;
  const lt = Array.isArray(c.layer_types) ? c.layer_types.map(String) : null;
  const interval = posInt3(c.full_attention_interval);
  const linearLayers = lt ? lt.filter((x) => /linear|deltanet|gdn/i.test(x)).length : interval ? (posInt3(c.num_hidden_layers) ?? 0) - Math.floor((posInt3(c.num_hidden_layers) ?? 0) / interval) : 0;
  const transientMib = vAtLeast(o.version, [0, 26, 0]) && !vAtLeast(o.version, [0, 27, 0]) && sizes.full.length > 0 && linearLayers > 0 ? o.maxNumBatchedTokens * linearLayers * (128 / 36) / 1024 / o.tp : 0;
  const firstMib = S > 0 ? F.firstBaseMib + transientMib : 0;
  const perP = Math.max(1, F.perPiecewiseMibPerLayer * (layers + 1));
  const perF = Math.max(1, F.perFullMib + (F.perFullMibPerLayer ?? 0) * (layers + 1));
  const nP = sizes.piecewise.length, nF = sizes.full.length;
  const totalMib = firstMib + (nP > 1 ? (nP - 1) * perP : 0) + (nF > 1 ? (nF - 1) * perF : 0);
  return {
    gib: totalMib / 1024,
    sizes,
    firstGib: firstMib / 1024,
    perGraphGib: { piecewise: perP / 1024, full: perF / 1024 },
    basis: `estimate of vLLM's own estimate: ${words}; first capture ${firstMib.toFixed(0)} MiB (the pool at ${S} tokens${transientMib ? `, of which ${transientMib.toFixed(0)} MiB is the v0.26 profiling transient on ${linearLayers} linear-attention layers, #50780` : ""}) + (N - 1) x ${perP.toFixed(1)} MiB per PIECEWISE graph + (N - 1) x ${perF.toFixed(1)} MiB per FULL graph, the structure of profile_cudagraph_memory with its two measured terms fitted to public logs`
  };
}
function visionTowerParams(config) {
  const v = config?.vision_config ?? null;
  if (!v || typeof v !== "object") return null;
  const mt = String(config?.model_type ?? "").toLowerCase();
  const h = posInt3(v.hidden_size) ?? posInt3(v.embed_dim);
  const depth = posInt3(v.depth) ?? posInt3(v.num_hidden_layers);
  if (!h || !depth) return null;
  const inter = posInt3(v.intermediate_size) ?? Math.round(h * (Number(v.mlp_ratio) || 4));
  const gated = /qwen2_5_vl/.test(mt);
  return depth * (4 * h * h + 4 * h + (gated ? 3 : 2) * h * inter + inter + 5 * h);
}
function vllmEncoderProfile(config, o) {
  if (!config || o.skip) return null;
  const v = config.vision_config ?? null;
  if (!v || typeof v !== "object") return null;
  const mt = String(config.model_type ?? "").toLowerCase();
  const arch = String(config.architectures?.[0] ?? "");
  const t = textConfigOf(config);
  const textHidden = posInt3(t.hidden_size) ?? posInt3(config.hidden_size) ?? 4096;
  const kw = o.mmProcessorKwargs ?? {};
  const limit = o.limitPerPrompt?.image;
  if (limit === 0) return null;
  let family;
  let tokens, patches, h, inter, gated, depth, pixelBytesPerItem;
  const notes = [];
  if (mt === "gemma3" || /Gemma3ForConditionalGeneration/.test(arch)) {
    family = "gemma3";
    const img = posInt3(v.image_size) ?? 896, patch = posInt3(v.patch_size) ?? 14;
    const perCrop = posInt3(config.mm_tokens_per_image) ?? 256;
    const crops = kw.do_pan_and_scan === true ? 1 + (posInt3(kw.pan_and_scan_max_num_crops) ?? 4) : 1;
    tokens = perCrop * crops;
    patches = (img / patch) ** 2 * crops;
    h = posInt3(v.hidden_size) ?? 1152;
    inter = posInt3(v.intermediate_size) ?? 4304;
    gated = false;
    depth = posInt3(v.num_hidden_layers) ?? 27;
    pixelBytesPerItem = crops * 3 * img * img * o.actBytes;
  } else if (/qwen2_vl|qwen2_5_vl|qwen3_vl/.test(mt) || /Qwen(2|2_5|3)VL/.test(arch)) {
    family = /qwen3_vl|Qwen3VL/.test(mt + arch) ? "qwen3-vl" : /qwen2_5_vl|Qwen2_5_VL/.test(mt + arch) ? "qwen2.5-vl" : "qwen2-vl";
    const patch = posInt3(v.patch_size) ?? (family === "qwen3-vl" ? 16 : 14);
    const merge = posInt3(v.spatial_merge_size) ?? 2;
    const temporal = posInt3(v.temporal_patch_size) ?? 2;
    const unit = (patch * merge) ** 2;
    const maxPixels = posInt3(kw.max_pixels) ?? posInt3(kw.size?.longest_edge) ?? (family === "qwen3-vl" ? 16777216 : 12845056);
    tokens = Math.max(1, Math.floor(maxPixels / unit));
    patches = tokens * merge * merge;
    h = posInt3(v.hidden_size) ?? posInt3(v.embed_dim) ?? 1280;
    if (family === "qwen2-vl") {
      h = posInt3(v.embed_dim) ?? 1280;
      inter = Math.round(h * (Number(v.mlp_ratio) || 4));
      gated = false;
    } else {
      inter = posInt3(v.intermediate_size) ?? (family === "qwen3-vl" ? 4304 : 3420);
      gated = family === "qwen2.5-vl";
    }
    depth = posInt3(v.depth) ?? posInt3(v.num_hidden_layers) ?? 32;
    pixelBytesPerItem = patches * 3 * temporal * patch * patch * o.actBytes;
    if (o.limitPerPrompt?.video !== 0 && o.maxModelLen > 2 * tokens) notes.push('video is not sized: at this max length vLLM may profile one video item instead of images, with more tokens than an image (the log says which: "profiled with N video items")');
  } else return null;
  const budget = Math.max(o.maxNumBatchedTokens, tokens);
  const perPrompt = Math.max(1, Math.min(limit ?? 999, Math.floor(o.maxModelLen / tokens)));
  const items = Math.max(1, Math.min(Math.floor(budget / tokens), o.maxNumSeqs * perPrompt));
  const P = items * patches;
  const peak = P * (2 * h + (gated ? 3 : 2) * inter) * o.actBytes + items * pixelBytesPerItem;
  const outputs = items * tokens * textHidden * o.actBytes;
  const towerParams = visionTowerParams(config) ?? depth * (4 * h * h + 4 * h + (gated ? 3 : 2) * h * inter + inter + 5 * h);
  return {
    family,
    modality: "image",
    tokensPerItem: tokens,
    patchesPerItem: patches,
    budgetTokens: budget,
    items,
    peakGib: peak / GiB,
    outputsGib: outputs / GiB,
    towerWeightsGib: towerParams * o.actBytes / GiB,
    basis: `estimate: vLLM profiles the vision tower with ${items} image item${items === 1 ? "" : "s"} of ${tokens.toLocaleString("en-US")} tokens (budget ${budget.toLocaleString("en-US")} = max(max_num_batched_tokens, one item); ${P.toLocaleString("en-US")} patches through ${depth} layers of width ${h}); the peak is one layer's MLP intermediates at that patch count plus the pixel input`,
    notes
  };
}
function sglangMemoryDefaults(o) {
  const g = o.gpuMemMiB;
  let cp, bs;
  if (g < 20 * 1024) {
    cp = 2048;
    bs = 8;
  } else if (g < 35 * 1024) {
    cp = 4096;
    bs = o.tp < 4 ? 48 : 160;
  } else if (g < 60 * 1024) {
    cp = 4096;
    bs = o.tp < 4 ? 32 : 160;
  } else if (g < 90 * 1024) {
    cp = 8192;
    bs = o.tp < 4 ? 256 : 512;
  } else if (g < 160 * 1024) {
    cp = 8192;
    bs = o.tp < 4 ? 256 : 512;
  } else {
    cp = 16384;
    bs = 512;
  }
  if (o.chunkedPrefill != null) cp = o.chunkedPrefill;
  if (o.maxRunningRequests != null && bs > o.maxRunningRequests) bs = o.maxRunningRequests;
  let prefillMax = o.mla ? 2048 : cp;
  if (/llama-2/i.test(o.modelPath ?? "")) prefillMax = Math.min(prefillMax, 4096);
  const sizes = [];
  for (let s = 4; s <= 32; s += 4) sizes.push(s);
  for (let s = 48; s <= 256; s += 16) sizes.push(s);
  for (let s = 288; s <= 512; s += 32) sizes.push(s);
  for (let s = 576; s <= 1024; s += 64) sizes.push(s);
  for (let s = 1280; s <= 4096; s += 256) sizes.push(s);
  for (let s = 4608; s <= prefillMax; s += 512) sizes.push(s);
  const prefillGraphSizes = sizes.filter((s) => s <= prefillMax).length;
  const terms = [];
  const act = Math.max(cp > 0 ? cp : 0, 2048);
  terms.push({ what: "constant (attention metadata) + activation slack", mb: 512 + act * 1.5 });
  terms.push({ what: `parallel slack (TP ${o.tp} x PP ${o.pp} / 8 x 1 GiB)`, mb: o.tp * o.pp / 8 * 1024 });
  if (!o.disableCudaGraph) {
    terms.push({ what: `decode CUDA graphs (max_bs ${bs} x 2 MB)`, mb: bs * 2 });
    if (o.dpAttention && o.dpAttention > 1) terms.push({ what: "DP-attention padding", mb: bs * o.dpAttention * 3 + (bs > 300 ? bs * o.dpAttention * 1.5 : 0) });
    terms.push({ what: o.mla ? "prefill CUDA graphs (MLA backend, 1.5 GiB)" : `prefill CUDA graphs (${prefillGraphSizes} sizes x 8 MB)`, mb: o.mla ? 1.5 * 1024 : prefillGraphSizes * 8 });
  }
  let reserved = terms.reduce((s, t) => s + t.mb, 0);
  if (g > 60 * 1024 && reserved < 10 * 1024) {
    terms.push({ what: "floor to 10 GiB above 60 GiB cards", mb: 10 * 1024 - reserved });
    reserved = 10 * 1024;
  }
  if (o.deepep && !o.disableCudaGraph) {
    terms.push({ what: "DeepEP all-to-all buffers", mb: 2048 });
    reserved += 2048;
  }
  return { chunkedPrefill: cp, decodeMaxBs: bs, prefillGraphSizes, reservedMb: reserved, memFractionStatic: Math.round((g - reserved) / g * 1e3) / 1e3, terms };
}
function llamacppSwaCells(nCtx, nSwa, nSeq, unified, nUbatch = 512) {
  return Math.min(Math.ceil(Math.min(nCtx, nSwa * (unified ? nSeq : 1) + nUbatch) / 256) * 256, Math.max(nCtx, Math.ceil(nCtx / 256) * 256));
}
var VLLM_PLUGIN_MIGRATIONS = {
  gguf: { since: "0.24.0", plugin: "vllm-gguf-plugin", pr: "#39612" },
  bitsandbytes: { since: "0.28.0", plugin: "vllm-bnb-plugin", pr: "#43529" }
};
function quantTableFor(engine, version) {
  const ts = ENGINE_QUANT_TABLES.filter((t) => t.engine === engine).map((t) => ({ t, v: readEngineVersion(t.tag) })).sort((a, b) => vAtLeast(a.v, b.v) ? 1 : -1);
  if (!ts.length) return null;
  if (!version) return ts[ts.length - 1].t;
  const fit = ts.filter((x) => vAtLeast(version, x.v));
  return (fit.length ? fit[fit.length - 1] : ts[0]).t;
}
function engineQuantKey(raw, qc) {
  if (!raw) return null;
  const m = String(raw).toLowerCase();
  if (m !== "compressed-tensors") return { method: m, scheme: null };
  const groups = qc?.config_groups ?? {};
  const g = Object.values(groups)[0];
  const w = g?.weights, a = g?.input_activations;
  let scheme = null;
  if (w?.num_bits === 8 && w.type === "float") scheme = a?.num_bits === 8 ? "w8a8fp8" : "w8a16fp8";
  else if (w?.num_bits === 4 && w.type === "float") scheme = a?.num_bits === 4 ? "w4a4fp4" : "w4a16fp4";
  else if (w?.num_bits === 8 && a?.num_bits === 8) scheme = "w8a8int8";
  else if (w?.num_bits === 4 && a?.num_bits === 8) scheme = a.type === "float" ? "w4a8fp8" : "w4a8int";
  else if (w?.num_bits) scheme = "wna16";
  return { method: "compressed-tensors", scheme };
}
var gib = (x) => `${x.toFixed(2)} GiB`;
function engineMemory(x) {
  const { engine, args, card, layout, servedLen } = x;
  const findings = [];
  const checked = [];
  const weightsGib = x.weightsPerGpuBytes / GiB;
  const total = cardTotalGiB(card.id, card.memoryGB);
  const stage = x.pp > 1 ? { ...layout, layers: layout.layers.slice(0, cdiv(layout.layers.length, x.pp)) } : layout;
  const kvPerTok = kvBytesPerTokenOf(stage);
  const statePerSeq = stage.layers.reduce((s, l) => s + (l.stateBytes ?? 0), 0);
  const base = {
    weightsBasis: x.weights.basis,
    weightsGbTotal: x.weights.bytes / 1e9,
    weightsGbPerGpu: x.weightsPerGpuBytes / 1e9,
    kvBytesPerTokenPerGpu: kvPerTok,
    stateBytesPerSequencePerGpu: statePerSeq,
    method: "engine",
    cardTotalGib: total.gib,
    cardTotalSource: total.source,
    layoutNotes: stage.notes
  };
  if (engine === "vllm" && x.vllmDefaults) {
    const D = x.vllmDefaults;
    const util = args.gpuMemoryUtilization ?? D.gpuMemoryUtilization;
    const requested = total.gib * util;
    const used = x.usedGib ?? 0;
    const kvFixed = args.kvCacheMemoryBytes != null ? args.kvCacheMemoryBytes / GiB : null;
    const mnbt = args.maxNumBatchedTokens ?? D.maxNumBatchedTokens;
    const mns = args.maxNumSeqs ?? D.maxNumSeqs;
    const vt = x.engineVersionTuple ?? readEngineVersion(ASSUMED_VLLM_VERSION);
    const encoder = vllmEncoderProfile(x.config, { maxNumBatchedTokens: mnbt, maxNumSeqs: mns, maxModelLen: servedLen, actBytes: x.actBytes, limitPerPrompt: args.limitMmPerPrompt, mmProcessorKwargs: args.mmProcessorKwargs, skip: args.skipMmProfiling });
    const cudaGraph = D.cudaGraphsCounted && !args.enforceEager ? vllmCudaGraphEstimate(x.config, { version: vt, maxNumSeqs: mns, maxNumBatchedTokens: mnbt, sm: smOfCard(card.id), tp: x.tp, pp: x.pp, actBytes: x.actBytes, args }) : null;
    const reserve = vllmNonKvReserve(x.config, { weightsGib, tp: x.tp, pp: x.pp, maxNumSeqs: mns, maxNumBatchedTokens: mnbt, cudaGraphsCounted: D.cudaGraphsCounted, enforceEager: !!args.enforceEager, actBytes: x.actBytes, nvlink: card.nvlinkGBs > 0, encoder, cudaGraph });
    const reserveGib = reserve.reduce((s, r) => s + r.gib, 0);
    const kvBudget = kvFixed ?? requested - reserveGib;
    const capOf = (len) => vllmKvCapacity(stage, {
      budgetBytes: Math.max(0, kvBudget) * GiB,
      maxModelLen: len,
      blockSize: args.blockSize,
      inFlightTokens: D.inFlightBatches * mnbt,
      printsTokensAs: D.printsTokensAs,
      encoderTokens: mnbt,
      mambaAlign: D.linearRecurrentAlign && stage.recurrentFamily === "linear" && !(args.flags ?? []).includes("--no-enable-prefix-caching")
    });
    const cap = capOf(servedLen);
    if (!cap) return null;
    const perSeqBytes = cap.blocksPerRequest * cap.bytesPerBlock;
    const conc = cap.maxConcurrency;
    const budgetWord = kvFixed != null ? `--kv-cache-memory-bytes ${gib(kvFixed)} (gpu_memory_utilization not used for the KV size)` : `${gib(total.gib)} x ${util} = ${gib(requested)}`;
    const reserveWord = reserve.map((r) => `${r.what} ${gib(r.gib)}`).join(" + ");
    const memory = {
      ...base,
      budgetGbPerGpu: requested * GiB / 1e9,
      kvGbPerSequencePerGpu: perSeqBytes / 1e9,
      kvCapacityGbPerGpu: Math.max(0, kvBudget) * GiB / 1e9,
      tokenCapacity: cap.printedTokens,
      maxConcurrentFullLength: Math.floor(conc * 100) / 100,
      reserve,
      pool: { blockSize: cap.blockSize, numBlocks: cap.numBlocks, groups: cap.groups, printedTokens: cap.printedTokens },
      notCounted: [
        "vLLM measures the activation peak, the non-torch memory and the CUDA graph pool at startup; the figures here for those three are estimates (see each term)",
        "a second process on the same GPU, and memory another tenant frees or takes during profiling",
        ...encoder ? encoder.notes : x.config.audio_config || x.config.vision_config && !args.skipMmProfiling && args.limitMmPerPrompt?.image !== 0 ? ["the multimodal encoder's profiling run for this architecture (vLLM profiles the vision / audio tower too; sized here for gemma-3 and Qwen2/2.5/3-VL images only, so on this model the activation here is low)"] : []
      ]
    };
    if (used > 0) {
      checked.push("memory-in-use");
      if (total.gib - used < requested) {
        findings.push({
          rule: "free-under-requested",
          area: "memory",
          severity: "block",
          title: `${gib(used)} already in use leaves ${gib(total.gib - used)} free; vLLM asks for ${gib(total.gib)} x ${util} = ${gib(requested)}`,
          detail: `vLLM compares free memory with the requested share at startup and refuses ("Free memory on device ... on startup is less than desired GPU memory utilization (${util}, ${requested.toFixed(2)} GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.").`,
          fix: `Lower --gpu-memory-utilization to at most ${(Math.floor((total.gib - used) / total.gib * 100) / 100).toFixed(2)}, or free the memory first.`,
          source: `${VLLM_SRC} vllm/v1/worker/gpu_worker.py init_device`
        });
      }
    }
    if (kvFixed != null) {
      checked.push("kv-cache-memory-bytes");
      const need = weightsGib + reserveGib + kvFixed;
      if (need > total.gib - used) {
        findings.push({
          rule: "kv-bytes-over-card",
          area: "memory",
          severity: "block",
          title: `Weights ${gib(weightsGib)} + reserve ${gib(reserveGib)} + --kv-cache-memory-bytes ${gib(kvFixed)} = ${gib(need)} per GPU, over the ${gib(total.gib - used)} ${used > 0 ? "left free" : "on the card"}`,
          detail: "With the KV size fixed, vLLM skips the check against the utilization budget and allocates the cache as asked; the allocation runs out of memory at startup, or the first large batch does.",
          fix: `Set --kv-cache-memory-bytes to at most ${gib(Math.max(0, total.gib - used - weightsGib - reserveGib))}, raise TP, or free memory on the card.`
        });
      }
    }
    if (kvBudget <= 0) {
      const weightsOnly = weightsGib >= requested;
      findings.push({
        rule: weightsOnly ? "weights-over-budget" : "no-kv-memory",
        area: "memory",
        severity: "block",
        title: weightsOnly ? `Weights alone are ${gib(weightsGib)} per GPU${x.shareWord}, over the ${gib(requested)} budget` : `Weights plus vLLM's own reserve leave no memory for the KV cache`,
        detail: `Budget = ${budgetWord}. Out of it: ${reserveWord}. vLLM refuses at startup ("No available memory for the cache blocks. Try increasing \`gpu_memory_utilization\`").`,
        fix: "Raise TP (or PP), quantize the weights, raise --gpu-memory-utilization, or use a larger card.",
        source: `${VLLM_SRC} vllm/v1/core/kv_cache_planning.py _check_enough_kv_cache_memory`
      });
    } else if (conc < 1) {
      let lo = 0, hi = servedLen;
      while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        const c = capOf(mid);
        if (c && c.maxConcurrency >= 1) lo = mid;
        else hi = mid - 1;
      }
      findings.push({
        rule: "kv-under-one-sequence",
        area: "memory",
        severity: "block",
        title: `One ${servedLen}-token request needs ${gib(perSeqBytes / GiB)} of KV per GPU; vLLM would have ${gib(kvBudget)}`,
        detail: `Budget = ${budgetWord}; out of it ${reserveWord}. vLLM refuses at startup ("To serve at least one request with the model's max seq len (${servedLen}), (${(perSeqBytes / GiB).toFixed(2)} GiB KV cache is needed, which is larger than the available KV cache memory (${kvBudget.toFixed(2)} GiB). Based on the available memory, the estimated maximum model length is ${lo}.").`,
        fix: `Serve at most ~${lo} tokens, raise --gpu-memory-utilization, use an fp8 KV cache (--kv-cache-dtype fp8), or raise TP.`,
        source: `${VLLM_SRC} vllm/v1/core/kv_cache_planning.py _check_enough_kv_cache_memory`
      });
    } else {
      checked.push("memory-fit");
      if (conc < 1.5) findings.push({
        rule: "kv-borderline",
        area: "memory",
        severity: "warn",
        title: `Only ~${conc.toFixed(2)} full-length requests fit after vLLM's own reserve`,
        detail: `${gib(kvBudget)} for KV against ${gib(perSeqBytes / GiB)} per ${servedLen}-token request. Three of the reserve terms are estimates of what vLLM measures, so at this margin the startup check can go either way.`,
        fix: "Leave more room: lower the max length, use an fp8 KV cache, or raise TP."
      });
      if (args.maxNumSeqs != null && args.maxNumSeqs > Math.floor(conc)) findings.push({
        rule: "seqs-over-capacity",
        area: "memory",
        severity: "warn",
        title: `--max-num-seqs ${args.maxNumSeqs}, but at most ${Math.floor(conc)} requests fit at the full ${servedLen} tokens`,
        detail: `vLLM's pool would hold ${cap.numBlocks.toLocaleString("en-US")} blocks of ${cap.blockSize} tokens; a full-length request takes ${cap.blocksPerRequest.toLocaleString("en-US")}. Shorter requests pack more, but once long ones arrive the scheduler preempts or queues past ~${Math.floor(conc)}.`,
        fix: `Set the flag to what fits (${Math.max(1, Math.floor(conc))}), shorten the max length, use an fp8 KV cache, or add GPUs as DP replicas.`
      });
      if (!findings.some((f) => f.rule === "seqs-over-capacity")) findings.push({
        rule: "kv-capacity",
        area: "memory",
        severity: "info",
        title: `vLLM would log "GPU KV cache size: ${cap.printedTokens.toLocaleString("en-US")} tokens" and "Maximum concurrency for ${servedLen.toLocaleString("en-US")} tokens per request: ${conc.toFixed(2)}x" (predicted)`,
        detail: `Per GPU: budget ${budgetWord}; ${reserveWord}; KV ${gib(kvBudget)} (${x.kvName}${stage.replicated ? ", KV heads replicated past TP" : ""}). Pool: ${cap.groups.map((g) => `${g.layers} ${g.kind} layer${g.layers === 1 ? "" : "s"}`).join(", ")} in blocks of ${cap.blockSize} tokens.`,
        fix: ""
      });
    }
    return { memory, findings, checked };
  }
  if (engine === "sglang" && x.sglDefault) {
    const def = x.sglDefault;
    const frac = args.memFractionStatic ?? def.memFractionStatic;
    const kvBudget = (total.gib - (x.usedGib ?? 0)) * frac - weightsGib;
    const cell = stage.layers.reduce((s, l) => s + (l.kind === "mamba" || l.kind === "cross" ? 0 : l.bytesPerToken), 0);
    const tokens = kvBudget > 0 && cell > 0 ? Math.floor(kvBudget * GiB / cell) : 0;
    const conc = tokens / servedLen;
    const reserveGib = total.gib * (1 - frac);
    const memory = {
      ...base,
      budgetGbPerGpu: total.gib * frac * GiB / 1e9,
      kvGbPerSequencePerGpu: (cell * servedLen + statePerSeq) / 1e9,
      kvCapacityGbPerGpu: Math.max(0, kvBudget) * GiB / 1e9,
      tokenCapacity: tokens,
      maxConcurrentFullLength: Math.floor(conc * 100) / 100,
      sglangDefault: { memFractionStatic: def.memFractionStatic, reservedMb: def.reservedMb, chunkedPrefill: def.chunkedPrefill, decodeMaxBs: def.decodeMaxBs, terms: def.terms },
      notCounted: [
        "SGLang measures free memory after loading; the CUDA context and NCCL buffers present before the load are not subtracted here",
        ...stage.layers.some((l) => l.kind === "sliding") ? ["SGLang's separate sliding-window pool: sliding layers are counted at the full length here, so the token count is a lower bound for them"] : [],
        ...stage.layers.some((l) => l.kind === "mamba") ? ["SGLang's separate Mamba state pool (max_mamba_cache_size)"] : []
      ]
    };
    if (args.memFractionStatic != null && args.memFractionStatic > def.memFractionStatic + 5e-4) {
      findings.push({
        rule: "mem-fraction-over-engine-default",
        area: "memory",
        severity: "warn",
        title: `--mem-fraction-static ${args.memFractionStatic} leaves ${gib(reserveGib)} outside the static pool; SGLang's own formula keeps ${(def.reservedMb / 1024).toFixed(2)} GiB there on this card`,
        detail: `The space outside weights + KV is where activations and CUDA graphs live. SGLang's default (${def.memFractionStatic}) reserves ${def.terms.map((t) => `${t.what} ${Math.round(t.mb)} MB`).join(", ")}. Setting a higher fraction contradicts the engine's own arithmetic for chunked_prefill_size ${def.chunkedPrefill} and decode CUDA graphs up to batch ${def.decodeMaxBs}, and is a common source of OOM under the first long prefill.`,
        fix: `Drop the flag (SGLang picks ${def.memFractionStatic}), or lower --chunked-prefill-size / --cuda-graph-max-bs alongside it.`,
        source: `${SGLANG_SRC} python/sglang/srt/arg_groups/memory_hook.py`
      });
    }
    if (kvBudget <= 0) {
      findings.push({
        rule: "weights-over-budget",
        area: "memory",
        severity: "block",
        title: `Weights ${gib(weightsGib)} per GPU${x.shareWord} are over ${gib(total.gib)} x ${frac} = ${gib(total.gib * frac)}`,
        detail: `SGLang refuses at startup ("Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=${frac}").`,
        fix: "Raise TP (or PP), quantize the weights, or use a larger card.",
        source: `${SGLANG_SRC} python/sglang/srt/mem_cache/kv_cache_configurator.py _profile_available_bytes`
      });
    } else if (conc < 1) {
      findings.push({
        rule: "kv-under-one-sequence",
        area: "memory",
        severity: "warn",
        title: `The KV pool holds ${tokens.toLocaleString("en-US")} tokens, under one ${servedLen}-token request`,
        detail: `${gib(kvBudget)} for KV at ${Math.round(cell).toLocaleString("en-US")} bytes per token (${x.kvName}). The server starts, and the longest requests it advertises cannot be scheduled.`,
        fix: `Serve at most ~${Math.floor(tokens / 1024) * 1024} tokens, use an fp8 KV cache, or raise TP.`
      });
    } else {
      checked.push("memory-fit");
      if (args.maxNumSeqs != null && args.maxNumSeqs > Math.floor(conc)) findings.push({
        rule: "seqs-over-capacity",
        area: "memory",
        severity: "warn",
        title: `--max-running-requests ${args.maxNumSeqs}, but at most ${Math.floor(conc)} requests fit at the full ${servedLen} tokens`,
        detail: `max_total_num_tokens would be ~${tokens.toLocaleString("en-US")} (${gib(kvBudget)} at ${Math.round(cell).toLocaleString("en-US")} bytes per token). Shorter requests pack more; long ones queue past ~${Math.floor(conc)}.`,
        fix: `Set the flag to what fits (${Math.max(1, Math.floor(conc))}), shorten the context, use an fp8 KV cache, or add DP replicas.`
      });
      if (!findings.some((f) => f.rule === "seqs-over-capacity")) findings.push({
        rule: "kv-capacity",
        area: "memory",
        severity: "info",
        title: `SGLang would allocate ~${tokens.toLocaleString("en-US")} KV tokens (max_total_num_tokens, predicted): ${Math.floor(conc)} full ${servedLen}-token requests`,
        detail: `Per GPU: ${gib(total.gib)} x mem_fraction_static ${frac}${args.memFractionStatic == null ? " (SGLang's own default)" : ""} = ${gib(total.gib * frac)}, less weights ${gib(weightsGib)}${x.shareWord}, at ${Math.round(cell).toLocaleString("en-US")} bytes per token (${x.kvName}).`,
        fix: ""
      });
    }
    return { memory, findings, checked };
  }
  if (engine === "llamacpp") {
    const nCtx = x.kvTokensAllocated ?? servedLen;
    const nSeq = args.nParallel ?? 1;
    const unified = args.kvUnified ?? args.nParallel == null;
    const ub = args.nUbatch ?? 512;
    const scale = x.kvBytes / 2;
    let kvAlloc = 0;
    const swaNotes = [];
    const tcfg = textConfigOf(x.config);
    const lora = num6(tcfg.kv_lora_rank) ?? 0, ropeD = num6(tcfg.qk_rope_head_dim) ?? 0;
    const mlaFactor = lora ? (2 * lora + ropeD) / (lora + ropeD) : 1;
    for (const l of stage.layers) {
      if (l.kind === "mamba") {
        kvAlloc += (l.stateBytes ?? 0) * nSeq;
        continue;
      }
      if (l.kind === "cross") continue;
      const perTok = l.bytesPerToken * scale * (l.kind === "mla" ? mlaFactor : 1);
      const cells = (l.kind === "sliding" || l.kind === "chunked") && !args.swaFull ? llamacppSwaCells(nCtx, l.window ?? nCtx, nSeq, unified, ub) : nCtx;
      if (cells < nCtx && !swaNotes.length) swaNotes.push(`sliding-window layers keep ${cells.toLocaleString("en-US")} cells, not ${nCtx.toLocaleString("en-US")} (GGML_PAD(min(n_ctx, n_swa x n_seq + n_ubatch), 256), ${LLAMACPP_SRC} src/llama-kv-cache-iswa.cpp)`);
      kvAlloc += perTok * cells;
    }
    const kvGb = kvAlloc / 1e9;
    const conc = unified ? Math.max(1, Math.floor(nCtx / servedLen)) : nSeq;
    const memory = {
      ...base,
      kvBytesPerTokenPerGpu: kvPerTok * scale,
      budgetGbPerGpu: card.memoryGB,
      kvGbPerSequencePerGpu: kvPerTok * scale * servedLen / 1e9,
      kvCapacityGbPerGpu: kvGb,
      tokenCapacity: nCtx,
      maxConcurrentFullLength: conc,
      layoutNotes: [...stage.notes, ...swaNotes],
      notCounted: ["llama.cpp's compute buffers (measured by --fit / llama-fit-params; not counted here)", "host RAM for layers left on the CPU"]
    };
    const need = x.weightsPerGpuBytes / 1e9 + kvGb;
    if (need > card.memoryGB - (x.usedGib ?? 0) * GiB / 1e9) {
      const ngl = args.nGpuLayers;
      const explicitAll = ngl === "all" || typeof ngl === "number" && ngl > 0;
      findings.push({
        rule: "memory-over-card",
        area: "memory",
        severity: explicitAll ? "block" : "warn",
        title: `Weights ${gbs(x.weightsPerGpuBytes / 1e9)} + KV for ${nCtx} tokens ${gbs(kvGb)} = ${gbs(need)} per GPU, over the ${card.label}`,
        detail: explicitAll ? "llama.cpp allocates the whole -c context up front; with these layers forced onto the GPU the allocation fails at load." : "With -ngl auto llama.cpp keeps the layers that do not fit on the CPU, which runs, at CPU memory bandwidth for those layers.",
        fix: "Lower -c, quantize the cache (-ctk q8_0 -ctv q8_0 halves it), pick a smaller quant, or offload fewer layers; llama-fit-params --model <file> prints the -c / -ngl / -ot llama.cpp itself would choose."
      });
    } else checked.push("memory-fit");
    return { memory, findings, checked };
  }
  return null;
}
var RECIPE_GPU_OF = { "h100-sxm": "h100", "h200": "h200", "b200": "b200", "l40s": "l40s", "a100-80g": "a100", "a100-40g": "a100" };
function findRecipe(modelId) {
  if (!modelId) return null;
  const want = modelId.toLowerCase();
  for (const [id, rc] of Object.entries(VLLM_RECIPES.recipes)) if (id.toLowerCase() === want) return { id, recipe: rc, variant: "default" in rc.variants ? "default" : null };
  for (const [id, rc] of Object.entries(VLLM_RECIPES.recipes)) for (const [vn, v] of Object.entries(rc.variants)) if (v.modelId && v.modelId.toLowerCase() === want) return { id, recipe: rc, variant: vn };
  return null;
}
var gbs = (x) => `${x.toFixed(x < 10 ? 2 : 1)} GB`;
function tpGeometryOf(config) {
  const c = config ? textConfigOf(config) : {};
  const n2 = (v) => typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null;
  const experts = n2(c.num_experts) ?? n2(c.num_local_experts) ?? n2(c.n_routed_experts);
  const vc = config?.vision_config ?? null;
  const q = c.quantization_config ?? config?.quantization_config ?? null;
  const wbs = q && Array.isArray(q.weight_block_size) ? n2(q.weight_block_size[0]) : null;
  const mla = n2(c.kv_lora_rank) != null;
  return {
    numHeads: n2(c.num_attention_heads) ?? n2(c.n_head),
    numKVHeads: mla ? null : n2(c.num_key_value_heads) ?? n2(c.num_attention_heads),
    visionHeads: vc ? n2(vc.num_heads) ?? n2(vc.num_attention_heads) : null,
    ffn: experts != null ? n2(c.moe_intermediate_size) ?? n2(c.intermediate_size) : n2(c.intermediate_size) ?? n2(c.ffn_dim),
    blockN: wbs
  };
}
function legalTpSizes(g, max = 8) {
  const out = [];
  for (let d = 1; d <= max; d++) {
    if (g.numHeads != null && g.numHeads % d !== 0) continue;
    if (g.numKVHeads != null && g.numKVHeads % d !== 0 && d % g.numKVHeads !== 0) continue;
    if (d > 1 && g.visionHeads != null && g.visionHeads % d !== 0) continue;
    if (d > 1 && g.ffn != null && g.ffn % d !== 0) continue;
    if (d > 1 && g.ffn != null && g.blockN != null && g.ffn / d % g.blockN !== 0) continue;
    out.push(d);
  }
  return out;
}
function visionKeptSplit(split, config, quant) {
  if (!quant || !config?.vision_config) return split;
  const qc = config.quantization_config ?? textConfigOf(config).quantization_config ?? null;
  if (!qc) return split;
  const lists = ["ignore", "modules_to_not_convert", "llm_int8_skip_modules", "ignored_layers"].map((k) => qc[k]).filter(Array.isArray).flat().map(String);
  if (!lists.some((x) => /visual|vision/i.test(x))) return split;
  const vp = visionTowerParams(config) ?? 0;
  return { ...split, edge: Math.min(split.total, split.edge + vp) };
}
function weightBytes(split, dtypeBytes, quant, bits, engine, ggufType) {
  const { total, expert, edge } = split;
  if (engine === "llamacpp") {
    if (!ggufType) return null;
    const bpw = GGUF_BPW[ggufType];
    return { bytes: total * bpw / 8, basis: `${ggufType} at ${bpw} bits per weight (${GGUF_BPW_SOURCE}); mixed per-tensor types make this approximate` };
  }
  const body = total - edge;
  switch (quant) {
    case null:
      return { bytes: total * dtypeBytes, basis: `${dtypeBytes} bytes per parameter (${dtypeBytes === 4 ? "fp32" : "16-bit"})` };
    case "fp8":
      return { bytes: body * 1 + edge * 2, basis: "fp8 weights (1 byte) with the embedding and LM head kept 16-bit" };
    case "int8":
      return { bytes: body * 1 + edge * 2, basis: "int8 weights (1 byte) with the embedding and LM head kept 16-bit" };
    case "mxfp4": {
      const ex = expert > 0 ? expert : body;
      return { bytes: ex * 4.25 / 8 + (total - ex) * dtypeBytes, basis: "MXFP4 experts at 4.25 bits (4-bit values plus a shared 8-bit scale per 32) and everything else 16-bit, as gpt-oss's modules_to_not_convert lists" };
    }
    case "nvfp4":
      return { bytes: body * 4.5 / 8 + edge * 2, basis: "NVFP4 weights at 4.5 bits (4-bit values plus an fp8 scale per 16), embedding and LM head 16-bit" };
    case "bitsandbytes": {
      const b = bits === 8 ? 8 : 4.5;
      return { bytes: body * b / 8 + edge * 2, basis: `bitsandbytes ${bits === 8 ? "int8" : "nf4"} at ${b} bits, embedding and LM head 16-bit` };
    }
    case "awq":
    case "gptq":
    case "marlin":
    default: {
      const b = (bits ?? 4) + 0.25;
      return { bytes: body * b / 8 + edge * 2, basis: `${bits ?? 4}-bit weights plus group scales (~${b} bits at group size 128), embedding and LM head 16-bit` };
    }
  }
}
function textGeometry(tc, fallback) {
  const heads = posInt3(tc.num_attention_heads) ?? posInt3(tc.n_head);
  const experts = posInt3(tc.n_routed_experts) ?? posInt3(tc.num_experts) ?? posInt3(tc.num_local_experts) ?? posInt3(tc.moe_num_experts);
  const dense = posInt3(tc.intermediate_size);
  const moeW = posInt3(tc.moe_intermediate_size);
  const hasDense = (posInt3(tc.first_k_dense_replace) ?? 0) > 0 || Array.isArray(tc.mlp_layer_types) && tc.mlp_layer_types.some((x) => /dense/i.test(String(x)));
  return {
    numHeads: heads ?? fallback.numHeads,
    numKVHeads: posInt3(tc.num_key_value_heads) ?? heads ?? fallback.numKVHeads,
    hidden: posInt3(tc.hidden_size) ?? fallback.hidden,
    ffn: (experts ? hasDense ? dense ?? moeW : moeW ?? dense : dense) ?? fallback.ffn,
    numExperts: experts ?? (heads ? null : fallback.numExperts),
    vocab: posInt3(tc.vocab_size) ?? fallback.vocab
  };
}
var TEXT_MAX_LEN_DEFAULT = [[/^gemma3_text$/, 131072]];
function textLengthConfig(tc) {
  if (posInt3(tc.max_position_embeddings) != null) return tc;
  const d = TEXT_MAX_LEN_DEFAULT.find(([re]) => re.test(String(tc.model_type ?? "")));
  return d ? { ...tc, max_position_embeddings: d[1] } : tc;
}
var PUBLISHED_WEIGHTS_TOLERANCE = 0.1;
var STORED_FLOAT_BYTES = { F64: 8, F32: 4, F16: 2, BF16: 2 };
function checkServeConfig(input) {
  const { engine, args, model } = input;
  const config = input.config ?? null;
  const checked = [];
  const skipped = [];
  const findings = [];
  const add = (f) => findings.push(f);
  const comps = model.components ?? [];
  const tc = config ? textConfigOf(config) : null;
  const multimodal = !!config && !!tc && (tc !== config || config.vision_config != null || config.audio_config != null || config.thinker_config != null);
  const geo = multimodal ? textGeometry(tc, geometryOf(model)) : geometryOf(model);
  const hasMla = comps.some((c) => c.type === "mla") || multimodal && (posInt3(tc.kv_lora_rank) != null || Array.isArray(tc.compress_ratios));
  const learned = multimodal ? /^(opt|gpt2|gpt_bigcode|gptj|gpt_neo)$/.test(String(tc.model_type ?? "")) : comps.some((c) => c.type === "learnedPositionalEmbedding" || c.type === "positionalEncoding");
  const card = input.gpu != null || input.gpuMemGiB != null ? resolveServeGpu(input.gpu, input.gpuMemGiB) : null;
  const sm = card ? smOfCard(card.id) : null;
  const lc = engine === "llamacpp";
  const tp = lc ? 1 : Math.max(1, args.tp ?? 1);
  const pp = lc ? 1 : Math.max(1, args.pp ?? 1);
  const dp = lc ? 1 : Math.max(1, args.dp ?? 1);
  const ep = !lc && args.expertParallel === true;
  const world = tp * pp * dp;
  const gpus = input.gpus ?? (lc ? 1 : world);
  const modelType = String(config?.model_type ?? "");
  const ident = `${modelType} ${input.modelId ?? ""} ${model.name ?? ""}`;
  const ckpt = checkpointQuant(config);
  const defaultsAssumed = [];
  const givenVersion = input.engineVersion != null ? readEngineVersion(input.engineVersion) : null;
  const ver = lc ? null : givenVersion ?? readEngineVersion(engine === "vllm" ? ASSUMED_VLLM_VERSION : ASSUMED_SGLANG_VERSION);
  const versionLabel = lc ? null : `${engine === "vllm" ? "vLLM" : "SGLang"} ${givenVersion ? input.engineVersion : engine === "vllm" ? ASSUMED_VLLM_VERSION : ASSUMED_SGLANG_VERSION}`;
  if (!lc && !givenVersion) defaultsAssumed.push(`${versionLabel} (no --${engine}-version given; the newest release the tables were read from)`);
  const vllmDefaults = engine === "vllm" && card && ver ? vllmVersionDefaults(ver, card, pp) : null;
  if (vllmDefaults) {
    if (args.gpuMemoryUtilization == null) defaultsAssumed.push(`gpu_memory_utilization ${vllmDefaults.gpuMemoryUtilization} (vLLM ${vllmDefaults.gpuMemoryUtilization === 0.92 ? ">= 0.20.0" : "< 0.20.0"} default)`);
    defaultsAssumed.push(`CUDA graph memory ${vllmDefaults.cudaGraphsCounted ? args.enforceEager ? "none (--enforce-eager)" : "taken from the KV budget (vLLM >= 0.20.0)" : "not taken from the KV budget (vLLM < 0.20.0)"}`);
    if (args.maxNumBatchedTokens == null) defaultsAssumed.push(`max_num_batched_tokens ${vllmDefaults.maxNumBatchedTokens} (vllm serve default on ${card.label})`);
    if (args.maxNumSeqs == null) defaultsAssumed.push(`max_num_seqs ${vllmDefaults.maxNumSeqs} (vllm serve default on ${card.label})`);
  }
  const sglMib = card ? /@/.test(card.id) ? Math.round(cardTotalGiB(card.id, card.memoryGB).gib * 1024 + TORCH_GAP_MIB) : CARD_NVSMI_MIB[card.id]?.mib ?? card.memoryGB * 1024 : null;
  const sglDefault = engine === "sglang" && sglMib != null ? sglangMemoryDefaults({
    gpuMemMiB: sglMib,
    tp,
    pp,
    chunkedPrefill: args.chunkedPrefillSize,
    maxRunningRequests: args.maxNumSeqs,
    mla: hasMla || posInt3(config ? textConfigOf(config).kv_lora_rank : null) != null,
    dpAttention: args.dpAttention ? dp : void 0,
    deepep: args.deepep,
    disableCudaGraph: args.enforceEager,
    modelPath: args.model ?? input.modelId
  }) : null;
  if (sglDefault) {
    if (args.memFractionStatic == null) defaultsAssumed.push(`mem_fraction_static ${sglDefault.memFractionStatic} (SGLang's own formula on a ${sglMib} MiB card)`);
    if (args.chunkedPrefillSize == null) defaultsAssumed.push(`chunked_prefill_size ${sglDefault.chunkedPrefill} (SGLang tier default)`);
  }
  if (lc) {
    skipped.push({ check: "parallel", reason: "llama.cpp splits whole layers (and their KV) across GPUs by default (--split-mode layer), so no head or width divisibility applies." });
  } else {
    const vcTp = config?.vision_config ?? null;
    const vHeadsTp = vcTp ? posInt3(vcTp.num_heads) ?? posInt3(vcTp.num_attention_heads) : null;
    const splitFfnTp = geo.numExperts == null || !ep;
    const legal = legalTpSizes({
      numHeads: geo.numHeads ?? null,
      numKVHeads: !hasMla ? geo.numKVHeads ?? null : null,
      visionHeads: args.mmEncoderTpMode === "data" ? null : vHeadsTp,
      ffn: splitFfnTp ? geo.ffn ?? null : null,
      blockN: splitFfnTp ? ckpt.blockN ?? null : null
    });
    const legalText = legal.length ? legal.join(", ") : "none up to 8";
    checked.push("gpu-count");
    if (input.gpus != null && world > input.gpus) {
      add({
        rule: "world-over-gpus",
        area: "parallel",
        severity: "block",
        title: `TP ${tp} x PP ${pp} x DP ${dp} = ${world} GPUs, but ${input.gpus} are available`,
        detail: `The engine needs one GPU per rank; with ${input.gpus} it fails to place the workers (or waits for them).`,
        fix: `Make TP x PP x DP at most ${input.gpus}${input.gpus % (tp * pp) === 0 ? ` (e.g. DP ${input.gpus / (tp * pp)})` : ""}.`
      });
    } else if (input.gpus != null && world < input.gpus) {
      add({
        rule: "gpus-idle",
        area: "parallel",
        severity: "info",
        title: `${input.gpus - world} of ${input.gpus} GPUs are not used by TP x PP x DP = ${world}`,
        detail: "The engine only starts as many ranks as the product of the parallel sizes.",
        fix: input.gpus % (tp * pp) === 0 ? `Add DP ${input.gpus / (tp * pp)} to use them as replicas.` : "Change the parallel sizes to use every GPU, or launch on fewer."
      });
    }
    if (geo.numHeads != null) {
      checked.push("tp-heads");
      if (geo.numHeads % tp !== 0) {
        add({
          rule: "tp-heads",
          area: "parallel",
          severity: "block",
          title: `TP ${tp} does not divide ${geo.numHeads} attention heads`,
          detail: `${engine === "vllm" ? "vLLM" : "SGLang"} gives each TP rank whole heads and asserts the division at load ("Total number of attention heads (${geo.numHeads}) must be divisible by tensor parallel size (${tp})").`,
          fix: `Use a TP size that passes every TP rule for this model: ${legalText}.`
        });
      }
      if (!hasMla && geo.numKVHeads != null) {
        checked.push("tp-kv-heads");
        if (geo.numKVHeads % tp !== 0) {
          if (tp % geo.numKVHeads === 0) {
            add({
              rule: "tp-kv-replicated",
              area: "parallel",
              severity: "warn",
              title: `TP ${tp} is past ${geo.numKVHeads} KV heads: each KV head is replicated ${tp / geo.numKVHeads}x`,
              detail: `It starts, but every GPU stores a full KV head, so the KV cache per GPU stops shrinking at TP ${geo.numKVHeads}: the extra ranks hold copies, not more sequences.`,
              fix: `TP ${geo.numKVHeads} with DP ${tp / geo.numKVHeads} serves more sequences on the same GPUs, if the weights fit at TP ${geo.numKVHeads}.`
            });
          } else {
            add({
              rule: "tp-kv-heads",
              area: "parallel",
              severity: "block",
              title: `TP ${tp} neither divides nor is a multiple of ${geo.numKVHeads} KV heads`,
              detail: "Grouped-query attention needs whole KV heads per rank or an exact replication factor; the engine asserts this at load.",
              fix: `Use a TP size that passes every TP rule for this model: ${legalText}.`
            });
          }
        }
      }
    } else skipped.push({ check: "tp-heads", reason: "no attention layer with a head count in the graph." });
    const vc = config?.vision_config ?? null;
    const vHeads = vc ? posInt3(vc.num_heads) ?? posInt3(vc.num_attention_heads) : null;
    if (vHeads != null && tp > 1) {
      checked.push("tp-vision-heads");
      const dataMode = args.mmEncoderTpMode === "data";
      if (!dataMode && vHeads % tp !== 0) {
        const known = engine === "vllm" && /^(qwen2_vl|qwen2_5_vl|qwen3_vl|qwen3_vl_moe|gemma3|paligemma|idefics3|smolvlm)$/.test(modelType);
        add({
          rule: "tp-vision-heads",
          area: "parallel",
          severity: known ? "block" : "warn",
          title: `TP ${tp} does not divide the vision tower's ${vHeads} attention heads`,
          detail: `${known ? `vLLM splits this tower's heads across TP ranks and asserts the division at load ("` + vHeads + " is not divisible by " + tp + `"), even when the language model's heads divide.` : "Engines that shard the vision tower with the language model split its heads across TP ranks and assert the division at load; whether this architecture does depends on the engine's implementation."}`,
          fix: `${engine === "vllm" ? "Add --mm-encoder-tp-mode data (the tower runs whole on every rank, which costs its weights once per GPU), or use" : "Use"} a TP size that passes every TP rule for this model: ${legalText}.`,
          source: "vllm/model_executor/models/qwen2_5_vl.py, qwen3_vl.py, siglip.py (vllm-project/vllm@9499c949): num_heads divided by the TP size unless the ViT runs data-parallel; vllm-project/vllm#55517"
        });
      }
    }
    if (geo.ffn != null && tp > 1) {
      checked.push("tp-ffn");
      const moeNoEp = geo.numExperts != null && !ep;
      const what = geo.numExperts != null ? "expert FFN width" : "FFN width";
      const splitFfn = geo.numExperts == null || moeNoEp;
      if (splitFfn && geo.ffn % tp !== 0) {
        add({
          rule: "tp-ffn",
          area: "parallel",
          severity: "block",
          title: `TP ${tp} does not divide the ${what} ${geo.ffn}`,
          detail: "Column-parallel linears split the intermediate dimension evenly across ranks and assert the division.",
          fix: `Use a TP size that passes every TP rule for this model: ${legalText}${geo.numExperts != null ? "; or enable expert parallelism so experts stay whole" : ""}.`
        });
      } else if (splitFfn && ckpt.blockN != null && geo.ffn / tp % ckpt.blockN !== 0) {
        add({
          rule: "tp-quant-block",
          area: "parallel",
          severity: "block",
          title: `TP ${tp} leaves ${geo.ffn / tp} columns of the ${what} per rank, not a multiple of the checkpoint's ${ckpt.blockN}-wide quant block`,
          detail: `The checkpoint is block-quantized (weight_block_size [${ckpt.blockN}, ...]); each rank's slice must hold whole blocks, and vLLM refuses at load ("... is not divisible by weight quantization block_n = ${ckpt.blockN}").`,
          fix: `Use a TP size where ${geo.ffn} / TP is a multiple of ${ckpt.blockN} and every other TP rule passes: ${legalText}${geo.numExperts != null ? "; or enable expert parallelism so experts stay whole" : ""}.`,
          source: "Qwen deployment docs, vLLM FP8 troubleshooting: the same block_n = 128 error, fixed by lower TP or expert parallelism (qwen.readthedocs.io/en/latest/deployment/vllm.html)"
        });
      }
    }
    const ratios = tc && Array.isArray(tc.compress_ratios) ? tc.compress_ratios.map((x) => Number(x) || 0) : null;
    const nLayers = tc ? posInt3(tc.num_hidden_layers) : null;
    if (ratios && nLayers && pp > 1) {
      checked.push("pp-kv-sharing");
      const parts = Array.from({ length: pp }, () => Math.floor(nLayers / pp));
      for (let i = 2; i < nLayers % pp + 2; i++) parts[pp - i] += 1;
      const stageOf = [];
      parts.forEach((n2, st) => {
        for (let k = 0; k < n2; k++) stageOf.push(st);
      });
      const srcs = (k) => Array.isArray(tc[k]) ? tc[k].map(Number) : [];
      const kvS = srcs("kv_source_layer_ids"), ixS = srcs("index_source_layer_ids");
      const split = [];
      for (let i = 0; i < nLayers; i++) {
        if (!(ratios[i] > 0)) continue;
        for (const list2 of [kvS, ixS]) {
          const below = list2.filter((x) => x <= i);
          if (below.length && stageOf[Math.max(...below)] !== stageOf[i]) {
            split.push(i);
            break;
          }
        }
      }
      if (split.length) {
        add({
          rule: "pp-kv-sharing",
          area: "parallel",
          severity: "block",
          title: `PP ${pp} puts ${split.length} compressed-attention layer${split.length === 1 ? "" : "s"} on a different stage from the layer that owns their cache (first: layer ${split[0]})`,
          detail: `${modelType} layers read the compressed KV and indexer caches of the nearest kv / index source layer below them; vLLM refuses a pipeline split that separates the two ("PP splits inside a v4.1 kv-sharing group are not supported"). Stages here: ${parts.join(" / ")} layers.`,
          fix: "Use a PP size whose stage boundaries fall on source layers, or TP / EP instead of PP.",
          source: "vllm-project/vllm vllm/models/deepseek_v41/attention.py"
        });
      }
    }
    if (geo.numExperts != null) {
      checked.push("expert-parallel");
      if (ep) {
        const epSize = engine === "vllm" ? tp * dp : args.epSize ?? tp;
        if (geo.numExperts % epSize !== 0) {
          add({
            rule: "ep-uneven",
            area: "parallel",
            severity: "warn",
            title: `${geo.numExperts} experts over EP ${epSize} is not even`,
            detail: `${engine === "vllm" ? "vLLM sets EP = TP x DP and" : "The engine"} gives some ranks one more expert than others, so the busiest rank sets the pace and holds the most weights.`,
            fix: `Choose TP x DP dividing ${geo.numExperts}.`
          });
        }
        if (engine === "sglang" && args.epSize != null && tp % args.epSize !== 0) {
          add({
            rule: "ep-tp",
            area: "parallel",
            severity: "block",
            title: `SGLang EP ${args.epSize} does not divide TP ${tp}`,
            detail: "SGLang carves the expert-parallel group out of the tensor-parallel ranks.",
            fix: `Use an EP size dividing ${tp}.`
          });
        }
      } else if (tp >= 4) {
        add({
          rule: "moe-no-ep",
          area: "parallel",
          severity: "info",
          title: `MoE with ${geo.numExperts} experts at TP ${tp} without expert parallelism`,
          detail: `Every expert is split ${tp} ways, so each rank multiplies ${geo.ffn != null ? `${geo.ffn / tp}-wide` : "thin"} expert slices and all-reduces after every MoE layer. Expert parallelism keeps experts whole and sends tokens to them instead.`,
          fix: engine === "vllm" ? "Try --enable-expert-parallel (EP = TP x DP) and compare throughput." : "Try --ep-size equal to TP and compare throughput."
        });
      }
    } else if (ep) {
      add({
        rule: "ep-dense",
        area: "parallel",
        severity: "warn",
        title: "Expert parallelism on a model with no MoE layer",
        detail: "There are no experts to distribute; the flag does nothing here.",
        fix: "Drop the expert-parallel flag."
      });
    }
  }
  const graphMax = (() => {
    const lp = comps.find((c) => c.type === "learnedPositionalEmbedding" || c.type === "positionalEncoding");
    const e = comps.find((c) => c.type === "embedding");
    return posInt3(lp?.params?.maxLen) ?? posInt3(e?.params?.maxSeqLen);
  })();
  let ropeOverride = args.ropeScaling;
  if (lc && (args.ropeScale != null || args.ropeScalingType != null) && String(args.ropeScalingType ?? "linear") !== "none") {
    ropeOverride = { rope_type: args.ropeScalingType ?? "linear", factor: args.ropeScale ?? 1, ...args.yarnOrigCtx ? { original_max_position_embeddings: args.yarnOrigCtx } : {} };
  }
  const maxLenRule = maxLenRuleFor(lc ? "llamacpp" : engine, ver);
  const derived = derivedMaxLen(multimodal ? textLengthConfig(tc) : config, ropeOverride, args.maxPositionOverride, multimodal ? null : graphMax, maxLenRule);
  let servedLen = null;
  let servedLenSource = "";
  let kvTokensAllocated = null;
  if (lc) {
    const total = args.ctxSize && args.ctxSize > 0 ? args.ctxSize : derived.native;
    const slots = args.nParallel ?? 1;
    const unified = args.kvUnified ?? args.nParallel == null;
    kvTokensAllocated = total;
    servedLen = total == null ? null : unified ? total : Math.floor(total / slots);
    servedLenSource = args.ctxSize && args.ctxSize > 0 ? unified ? `-c ${args.ctxSize} (one unified KV buffer, so one sequence may use all of it)` : `-c ${args.ctxSize} split over --parallel ${slots} slots = ${servedLen} per sequence` : `-c 0: the model's training context${unified ? "" : `, split over ${slots} slots`}`;
  } else if (args.maxModelLen != null) {
    servedLen = args.maxModelLen;
    servedLenSource = engine === "vllm" ? `--max-model-len ${args.maxModelLen}` : `--context-length ${args.maxModelLen}`;
  } else {
    servedLen = derived.len;
    servedLenSource = "not set: the engine derives it from the config";
  }
  if (servedLen != null && derived.native != null) {
    checked.push("context");
    const limitName = engine === "vllm" ? "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" : "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1";
    if (learned && servedLen > derived.native) {
      add({
        rule: "context-over-learned",
        area: "context",
        severity: "block",
        title: `Serving ${servedLen} tokens on a ${derived.native}-row learned position table`,
        detail: `Positions past ${derived.native - 1} index outside the table: the first long request crashes (or, if the engine clamps, attends with wrong positions). Rope scaling does not apply to learned positions.`,
        fix: `Serve at most ${derived.native} tokens.`
      });
    } else if (derived.len != null && servedLen > derived.len) {
      const ropeSet = derived.rope != null;
      add({
        rule: "context-over-derived",
        area: "context",
        severity: lc ? "warn" : "block",
        title: `Served length ${servedLen} is past the model's ${ropeSet ? `rope-scaled ${derived.len}` : `${derived.len}-token`} context`,
        detail: lc ? `llama.cpp runs past n_ctx_train with a warning; rotary positions keep computing, and quality past the trained length is not something the config vouches for.` : `The engine refuses to start ("User-specified ${engine === "vllm" ? "max_model_len" : "context_length"} (${servedLen}) is greater than the derived ${engine === "vllm" ? "max_model_len" : "context_length"} (${derived.len})") unless ${limitName} is set, and with it set, positions past the trained length are extrapolated.`,
        fix: ropeSet && derived.scaledByFactor ? `Raise the rope factor to at least ${Math.ceil(servedLen / (posInt3(derived.rope?.original_max_position_embeddings) ?? derived.native))}, or serve at most ${derived.len}.` : ropeSet && !lc ? `Serve at most ${derived.len}. ${engine === "vllm" ? "vLLM" : "SGLang"} reads this rope type as already folded into max_position_embeddings, so a larger factor alone does not raise the limit; where the model card supports a longer context, override max_position_embeddings to it together with the factor.` : `Serve at most ${derived.len}, or add rope scaling where the model card supports it, with the card's own original length (e.g. ${engine === "vllm" ? `--hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":F,"original_max_position_embeddings":N}${maxLenRule === "vllm" ? `,"max_position_embeddings":${servedLen}` : ""}}'` : engine === "sglang" ? `--json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":F,"original_max_position_embeddings":N},"max_position_embeddings":${servedLen}}'` : "--rope-scaling yarn --rope-scale F --yarn-orig-ctx N"}, with N x F >= ${servedLen}; N is the native length the card states, which can be below max_position_embeddings${maxLenRule === "vllm-legacy" || lc ? "" : `; this engine does not scale max_position_embeddings for YaRN, so it is raised in the same override`}).`
      });
    }
    if (ropeOverride && !config?.rope_scaling) {
      const type = String(ropeOverride.rope_type ?? ropeOverride.type ?? "");
      if (/yarn|linear|dynamic/.test(type)) {
        add({
          rule: "rope-static",
          area: "context",
          severity: servedLen <= derived.native ? "warn" : "info",
          title: servedLen <= derived.native ? `Rope scaling (${type}) is set but the served length ${servedLen} is within the native ${derived.native}` : `Rope scaling (${type}, factor ${ropeOverride.factor}) applies to every request, short ones included`,
          detail: "The scaling is static in these engines: the factor applies regardless of input length, which the Qwen model cards say can hurt quality on shorter texts.",
          fix: servedLen <= derived.native ? "Drop the rope scaling; it buys nothing at this length." : "Keep it only on the deployment that needs the long context.",
          source: 'Qwen3 model card, "Processing Long Texts" (static YaRN note)'
        });
      }
    }
  } else skipped.push({ check: "context", reason: derived.native == null ? "the config states no maximum length (max_position_embeddings)." : "no served length." });
  const cfgDtype = normDtype(String(config?.torch_dtype ?? config?.dtype ?? tc?.torch_dtype ?? tc?.dtype ?? "")) ?? null;
  const flagDtype = normDtype(args.dtype);
  let dtype = flagDtype && flagDtype !== "auto" ? flagDtype : cfgDtype && cfgDtype !== "auto" ? cfgDtype : "bf16";
  if (!lc && cfgDtype === "fp32" && (flagDtype == null || flagDtype === "auto")) {
    const fp16Unsafe = /^(gemma2|gemma3|gemma3_text|glm4)$/.test(modelType);
    dtype = engine === "vllm" ? sm == null || sm >= 80 ? "bf16" : fp16Unsafe ? "fp32" : "fp16" : /^gemma/.test(modelType) ? "bf16" : "fp16";
    defaultsAssumed.push(`dtype ${dtype}: the config says float32 and --dtype auto downcasts it (${engine === "vllm" ? "vLLM _resolve_auto_dtype" : "SGLang _get_and_verify_dtype"})`);
  }
  if (flagDtype === "fp32" && cfgDtype && cfgDtype !== "fp32") {
  }
  const flagQuant = normalizeQuant(args.quantization);
  const quant = lc ? "gguf" : flagQuant ?? ckpt.method;
  const gguf = lc ? input.gguf ?? null : null;
  const ggufType = lc ? gguf?.fileType && GGUF_BPW[gguf.fileType] != null ? gguf.fileType : gguf?.dominantType ?? ggufQuantOf(args.ggufFile) : null;
  if (gguf) {
    checked.push("gguf-header");
    const cfgLayers = posInt3(config ? textConfigOf(config).num_hidden_layers : null);
    if (cfgLayers != null && gguf.blockCount != null && gguf.blockCount !== cfgLayers) {
      add({
        rule: "gguf-config-mismatch",
        area: "context",
        severity: "warn",
        title: `The GGUF has ${gguf.blockCount} blocks; the config.json given has ${cfgLayers} layers`,
        detail: `The repo named for the config is not the one ${gguf.source} was converted from (or the file is a pruned / merged build), so the context, template and sampler checks read the wrong model.`,
        fix: "Give the Hugging Face id the GGUF was converted from as the model."
      });
    }
    if (gguf.contextLength != null && derivedMaxLenRaw(config) != null && gguf.contextLength !== derivedMaxLenRaw(config)) {
      add({
        rule: "gguf-context",
        area: "context",
        severity: "info",
        title: `The GGUF records a ${gguf.contextLength}-token training context (${gguf.arch}.context_length); config.json says ${derivedMaxLenRaw(config)}`,
        detail: "llama.cpp reads the GGUF value (n_ctx_train), not config.json, for -c 0 and for its past-the-trained-length warning.",
        fix: ""
      });
    }
  }
  if (!lc && flagQuant && ckpt.method && flagQuant !== ckpt.method) {
    const compatible = flagQuant === "marlin" && (ckpt.method === "gptq" || ckpt.method === "awq");
    if (!compatible) {
      add({
        rule: "quant-mismatch",
        area: "quant",
        severity: "block",
        title: `--quantization ${args.quantization} does not match the checkpoint's ${ckpt.raw}`,
        detail: `The weights are stored as ${ckpt.raw}; the engine refuses a different method ("Quantization method specified in the model config (${ckpt.raw}) does not match the quantization method specified in the \`quantization\` argument").`,
        fix: "Drop --quantization and let the engine read quantization_config, or point at an unquantized checkpoint."
      });
    }
  }
  if (sm == null) {
    skipped.push({ check: "quant-support", reason: card ? `${card.label} is not a CUDA card with a compute capability in the table.` : "no --gpu given." });
  } else if (lc) {
    skipped.push({ check: "quant-support", reason: "llama.cpp's CUDA backend runs every GGUF type on these cards; support is not a per-architecture table." });
  } else {
    checked.push("quant-support");
    if (sm < BF16_MIN_SM && (flagDtype === "bf16" || flagDtype !== "fp16" && flagDtype !== "fp32" && cfgDtype === "bf16")) {
      if (flagDtype === "bf16") {
        add({
          rule: "bf16-unsupported",
          area: "quant",
          severity: "block",
          title: `--dtype bfloat16 on ${card.label} (sm${sm})`,
          detail: "The card has no bf16 tensor cores and the engine refuses the explicit dtype.",
          fix: "Use --dtype float16 (and check the model does not overflow in fp16), or a card at sm80 or newer.",
          source: BF16_SOURCE
        });
      } else {
        add({
          rule: "bf16-auto-cast",
          area: "quant",
          severity: "warn",
          title: `The checkpoint is bf16 and ${card.label} (sm${sm}) has no bf16`,
          detail: "With --dtype auto the engine casts to fp16 (older versions refuse). fp16 has a narrower range than bf16, and some models trained in bf16 overflow in it.",
          fix: "Pass --dtype float16 knowingly and spot-check outputs, or use an sm80+ card.",
          source: BF16_SOURCE
        });
        dtype = "fp16";
      }
    }
    if (quant != null) {
      const table = quantTableFor(engine, ver);
      const rawMethod = (args.quantization ?? ckpt.raw ?? "").toLowerCase();
      const key = engineQuantKey(args.quantization ? rawMethod : ckpt.raw ?? null, args.quantization ? null : config?.quantization_config ?? null);
      const where = (at) => table && at ? `github.com/${table.repo}/blob/${table.commit.slice(0, 10)}/${at.replace("#L", "#L")}` : null;
      if (table && key) {
        const entry = table.methods[key.method] ?? (!args.quantization ? table.methods[`${key.method}_marlin`] : void 0);
        const label = `${engine === "vllm" ? "vLLM" : "SGLang"} ${table.tag}`;
        const plugin = engine === "vllm" ? VLLM_PLUGIN_MIGRATIONS[key.method] : void 0;
        if (plugin && ver && vAtLeast(ver, readEngineVersion(plugin.since))) {
          add({
            rule: "quant-method-plugin",
            area: "quant",
            severity: "warn",
            title: `${key.method} left vLLM's tree in ${plugin.since}: it loads only with ${plugin.plugin} installed`,
            detail: `${plugin.pr} moved it to an out-of-tree plugin; without the plugin the registry has no "${key.method}" and vLLM ${versionLabel?.replace("vLLM ", "")} refuses the model at load.`,
            fix: `uv pip install ${plugin.plugin}, or pin vLLM below ${plugin.since}.`,
            source: `github.com/vllm-project/vllm/pull/${plugin.pr.slice(1)}`
          });
        } else if (!entry) {
          add({
            rule: "quant-method-unknown",
            area: "quant",
            severity: "block",
            title: `${label} has no quantization method "${key.method}"`,
            detail: `The method registry at ${table.tag} (${Object.keys(table.methods).length} methods) does not list it, so the engine refuses it at load ("Invalid quantization method" / "Unknown quantization method").${engine === "vllm" && /bitsandbytes|gguf/.test(key.method) ? " vLLM removed it from the registry after 0.19." : ""}`,
            fix: `Use a release that lists it (pass --${engine}-version to check another), or a checkpoint in a method this release lists.`,
            source: `github.com/${table.repo}/blob/${table.commit.slice(0, 10)}/${engine === "vllm" ? "vllm/model_executor/layers/quantization/__init__.py" : "python/sglang/srt/layers/quantization/__init__.py"}`
          });
        } else {
          const scheme = key.scheme ? table.compressedTensorsSchemes[key.scheme] : null;
          const minSm = scheme?.minSm ?? entry.minSm;
          const at = scheme?.at ?? entry.at;
          if (minSm != null && sm < minSm) {
            add({
              rule: "quant-unsupported",
              area: "quant",
              severity: "block",
              title: `${key.method}${key.scheme ? ` (${key.scheme})` : ""} needs sm${minSm} on ${label}; ${card.label} is sm${sm}`,
              detail: `${label}'s ${entry.cls}${key.scheme ? ` / ${key.scheme} scheme` : ""} declares get_min_capability() = ${minSm}; the engine compares it with the card at load and refuses below it.`,
              fix: `Use a card at sm${minSm} or newer, a different quantization, or the unquantized checkpoint.`,
              source: where(at) ?? void 0
            });
          } else {
            const row = QUANT_SUPPORT.find((r) => r.method === quant);
            if (row && sm < row.nativeSm) {
              add({
                rule: "quant-fallback",
                area: "quant",
                severity: "warn",
                title: `${quant} on ${card.label} (sm${sm}) is not native: it runs on a fallback kernel`,
                detail: `${label} accepts it here (minimum sm${minSm ?? "?"}), but native ${quant} kernels need sm${row.nativeSm}+. On this card it ${row.fallbackNote ?? "runs on a slower path"}.`,
                fix: `Expect the memory saving and not the speedup; for native ${quant} use an sm${row.nativeSm}+ card.`,
                source: row.source
              });
            }
          }
          const other = quantTableFor(engine === "vllm" ? "sglang" : "vllm", null);
          const oe = other?.methods[key.method];
          const oMin = key.scheme ? other?.compressedTensorsSchemes[key.scheme]?.minSm ?? oe?.minSm : oe?.minSm;
          if (other && oMin != null && minSm != null && oMin !== minSm && sm >= Math.min(oMin, minSm) && sm < Math.max(oMin, minSm)) {
            add({
              rule: "quant-engine-differs",
              area: "quant",
              severity: "info",
              title: `${key.method}${key.scheme ? ` (${key.scheme})` : ""} on sm${sm}: ${label} needs sm${minSm}, ${other.engine === "vllm" ? "vLLM" : "SGLang"} ${other.tag} needs sm${oMin}`,
              detail: "The two engines gate the same method at different compute capabilities, so a launch that loads on one is refused by the other on this card.",
              fix: "Keep it in mind if the deployment moves engines.",
              source: [where(at), other && oe?.at ? `github.com/${other.repo}/blob/${other.commit.slice(0, 10)}/${oe.at}` : null].filter(Boolean).join("; ")
            });
          }
        }
      } else {
        const row = QUANT_SUPPORT.find((r) => r.method === quant);
        if (!row) skipped.push({ check: "quant-support", reason: `no support row for quant method "${quant}".` });
        else if (sm < row.fallbackSm) {
          add({
            rule: "quant-unsupported",
            area: "quant",
            severity: "block",
            title: `${quant} weights on ${card.label} (sm${sm}): not supported`,
            detail: `${quant} needs sm${row.fallbackSm} or newer${row.nativeSm !== row.fallbackSm ? ` (sm${row.nativeSm} for the native kernels)` : ""}; the engine refuses at load.`,
            fix: `Use a ${quant === "mxfp4" ? "bf16 build of the model" : "different quantization"} on this card, or an sm${row.fallbackSm}+ card.`,
            source: row.source
          });
        }
      }
    }
  }
  let memory = null;
  if (!card) skipped.push({ check: "memory", reason: "no --gpu given." });
  else if (servedLen == null) skipped.push({ check: "memory", reason: "no served length (set the max length, or give a config with max_position_embeddings)." });
  else {
    const peak = inferencePeakFor(model, { promptTokens: servedLen, genTokens: 0, batch: 1, precision: "bf16" });
    const split = paramSplit(model);
    const sizeQuant = !lc && !flagQuant && ckpt.store === "mxfp4" ? "mxfp4" : quant;
    let w = gguf ? { bytes: gguf.tensorBytes, basis: `${gguf.tensorCount} tensors in the GGUF header, mostly ${gguf.dominantType ?? "?"}${gguf.fileType ? ` (file type ${gguf.fileType})` : ""}, summed at ggml's block sizes (${LLAMACPP_SRC} ggml-common.h); read from ${gguf.source}` } : weightBytes(visionKeptSplit(split, config, quant), DTYPE_BYTES[dtype], sizeQuant, ckpt.bits, engine, ggufType);
    const pub = input.publishedWeights ?? null;
    if (w && !lc && pub && pub.totalBytes > 0) {
      if (flagQuant && flagQuant !== ckpt.method && !(flagQuant === "marlin" && (ckpt.method === "gptq" || ckpt.method === "awq"))) {
        skipped.push({ check: "weights-vs-checkpoint", reason: `--quantization ${args.quantization} changes the weights at load, so the stored size says nothing about what is on the GPU.` });
      } else {
        checked.push("weights-vs-checkpoint");
        const dp2 = pub.dtypeParams ?? null;
        let loaded = pub.totalBytes;
        let how = `${(pub.totalBytes / 1e9).toFixed(1)} GB stored (${pub.source})`;
        if (quant == null && dp2 && Object.keys(dp2).length && Object.keys(dp2).every((k) => k in STORED_FLOAT_BYTES)) {
          const n2 = Object.values(dp2).reduce((a, b) => a + b, 0);
          const stored = Object.entries(dp2).reduce((a, [k, v]) => a + v * STORED_FLOAT_BYTES[k], 0);
          loaded = n2 * DTYPE_BYTES[dtype];
          if (Math.abs(stored - loaded) > 0.01 * loaded) how = `${(n2 / 1e9).toFixed(2)}B stored parameters (${Object.keys(dp2).join(", ")}, ${(stored / 1e9).toFixed(1)} GB on disk) loaded as ${dtype}, ${(loaded / 1e9).toFixed(1)} GB (${pub.source})`;
        }
        const off = w.bytes / loaded - 1;
        if (Math.abs(off) > PUBLISHED_WEIGHTS_TOLERANCE) {
          add({
            rule: "weights-vs-checkpoint",
            area: "memory",
            severity: "warn",
            title: `Weights read from config.json come to ${gbs(w.bytes / 1e9)}; the checkpoint itself is ${gbs(loaded / 1e9)} (${off > 0 ? "+" : ""}${(off * 100).toFixed(0)}%): memory below uses the checkpoint's size`,
            detail: `Derived: ${w.basis}. Published: ${how}${pub.saveFormat ? `, save_format ${pub.saveFormat}` : ""}. A gap past ${PUBLISHED_WEIGHTS_TOLERANCE * 100}% means the config does not describe the stored tensors the way this check reads them (a storage format the quantization_config does not name, modules kept 16-bit, a tower or expert set the reading missed). The published size counts every tensor in the repo, including any the engine skips (MTP layers, a speech talker), so for those it is an upper bound.`,
            fix: "Nothing to change in the launch; the verdict below is sized on the published bytes.",
            source: pub.source
          });
          w = { bytes: loaded, basis: `the checkpoint's own size, ${how}; config.json alone gave ${gbs(w.bytes / 1e9)}` };
        } else {
          w = { ...w, basis: `${w.basis}; within ${Math.abs(off * 100).toFixed(1)}% of the checkpoint's ${gbs(loaded / 1e9)}` };
        }
      }
    }
    if (!peak || !w || !(split.total > 0)) {
      skipped.push({ check: "memory", reason: !w && lc ? `the GGUF quant type is not readable from "${args.ggufFile ?? "(no -m)"}"; name the file with its type (e.g. model-Q4_K_M.gguf).` : "the graph has no parameter count to size." });
    } else {
      checked.push("memory");
      let kvBytes = dtype === "fp32" ? 4 : 2;
      let kvName = dtype === "fp32" ? "fp32" : "16-bit";
      if (lc) {
        const k = GGUF_KV_BYTES[String(args.cacheTypeK ?? "f16").toLowerCase()] ?? 2;
        const v = GGUF_KV_BYTES[String(args.cacheTypeV ?? "f16").toLowerCase()] ?? 2;
        kvBytes = (k + v) / 2;
        kvName = `K ${args.cacheTypeK ?? "f16"} / V ${args.cacheTypeV ?? "f16"}`;
      } else if (args.kvCacheDtype && /fp8/i.test(args.kvCacheDtype)) {
        kvBytes = 1;
        kvName = args.kvCacheDtype;
      } else if (engine === "vllm" && (!args.kvCacheDtype || args.kvCacheDtype === "auto") && ckpt.kvDtype === "fp8_e4m3") {
        kvBytes = 1;
        kvName = "fp8_e4m3 (auto, from the checkpoint's kv_cache_quant_algo)";
      }
      let weightsPerGpu;
      let kvShare;
      let budgetGb;
      let fraction = null;
      if (lc) {
        const layers = posInt3(config?.num_hidden_layers) ?? Math.max(1, comps.filter((c) => /attention|mla/i.test(c.type)).length);
        const ngl = args.nGpuLayers;
        const onGpu = ngl === "all" || ngl === "auto" || ngl == null ? 1 : Math.min(1, ngl / (layers + 1));
        weightsPerGpu = w.bytes * onGpu / gpus;
        kvShare = onGpu / gpus;
        budgetGb = card.memoryGB;
      } else {
        const expertBytesShare = split.expert > 0 ? w.bytes * split.expert / split.total : 0;
        const denseBytes = w.bytes - expertBytesShare;
        const expertWays = ep && engine === "vllm" ? tp * dp * pp : tp * pp;
        weightsPerGpu = denseBytes / (tp * pp) + expertBytesShare / expertWays;
        kvShare = (hasMla ? 1 : geo.numKVHeads ? Math.max(1 / tp, 1 / geo.numKVHeads) : 1 / tp) / pp;
        fraction = engine === "vllm" ? args.gpuMemoryUtilization ?? 0.9 : args.memFractionStatic ?? 0.88;
        budgetGb = card.memoryGB * fraction;
      }
      const kvPerTokGpu = peak.kvBytesPerToken * kvBytes / 2 * kvShare;
      const statePerSeqGpu = peak.stateBytesPerSequence * kvShare;
      const weightsGb = weightsPerGpu / 1e9;
      const kvSeqGb = (kvPerTokGpu * servedLen + statePerSeqGpu) / 1e9;
      const lcTokens = kvTokensAllocated ?? servedLen;
      const capGb = lc ? kvPerTokGpu * lcTokens / 1e9 : budgetGb - weightsGb;
      const tokenCapacity = lc ? lcTokens : capGb > 0 && kvPerTokGpu > 0 ? Math.floor(capGb * 1e9 / kvPerTokGpu) : 0;
      const conc = lc ? args.kvUnified ?? args.nParallel == null ? Math.max(1, Math.floor(lcTokens / servedLen)) : args.nParallel ?? 1 : capGb > 0 && kvSeqGb > 0 ? capGb / kvSeqGb : 0;
      memory = {
        weightsBasis: w.basis,
        weightsGbTotal: w.bytes / 1e9,
        weightsGbPerGpu: weightsGb,
        budgetGbPerGpu: budgetGb,
        kvBytesPerTokenPerGpu: kvPerTokGpu,
        stateBytesPerSequencePerGpu: statePerSeqGpu,
        kvGbPerSequencePerGpu: kvSeqGb,
        kvCapacityGbPerGpu: Math.max(0, capGb),
        tokenCapacity,
        maxConcurrentFullLength: Math.floor(conc * 100) / 100,
        method: "upper-bound",
        notCounted: [
          "the engine's activation reserve from its profiling run, CUDA graphs and communication buffers (they come out of the same budget, so capacity here is an upper bound)",
          "the card's real usable memory, which is below the nominal figure",
          ...lc ? ["host RAM for layers left on the CPU"] : []
        ]
      };
      const budgetWord = lc ? `${card.memoryGB} GB on the card` : `${card.memoryGB} GB x ${fraction} = ${gbs(budgetGb)}`;
      const shareWord = lc ? gpus > 1 ? ` (layers split over ${gpus} GPUs)` : "" : ` after TP ${tp}${pp > 1 ? ` x PP ${pp}` : ""}${ep && split.expert > 0 ? " with experts spread by EP" : ""}`;
      const layout = config ? kvLayoutFromConfig(config, { tp: lc ? 1 : tp, kvBytes: lc ? 2 : kvBytes, modelBytes: DTYPE_BYTES[dtype] }) : null;
      const eng = layout && config ? engineMemory({
        engine,
        args,
        card,
        config,
        layout,
        tp,
        pp,
        ep,
        servedLen,
        kvTokensAllocated,
        weightsPerGpuBytes: weightsPerGpu,
        weights: w,
        kvBytes,
        kvName,
        actBytes: DTYPE_BYTES[dtype],
        vllmDefaults,
        engineVersionTuple: ver,
        sglDefault,
        versionLabel,
        gpus,
        shareWord,
        usedGib: input.gpuMemUsedGiB
      }) : null;
      if (eng) {
        memory = eng.memory;
        checked.push(...eng.checked);
        for (const f of eng.findings) add(f);
      } else if (lc) {
        const need = weightsGb + kvPerTokGpu * (kvTokensAllocated ?? servedLen) / 1e9;
        const ngl = args.nGpuLayers;
        if (need > budgetGb) {
          const explicitAll = ngl === "all" || typeof ngl === "number" && ngl > 0;
          add({
            rule: "memory-over-card",
            area: "memory",
            severity: explicitAll ? "block" : "warn",
            title: `Weights ${gbs(weightsGb)} + KV for ${kvTokensAllocated ?? servedLen} tokens ${gbs(need - weightsGb)} = ${gbs(need)} per GPU, over the ${card.label}`,
            detail: explicitAll ? "llama.cpp allocates the whole -c context up front; with these layers forced onto the GPU the allocation fails at load." : "With -ngl auto llama.cpp keeps the layers that do not fit on the CPU, which runs, at CPU memory bandwidth for those layers.",
            fix: `Lower -c, quantize the cache (-ctk q8_0 -ctv q8_0 halves it), pick a smaller quant, or offload fewer layers.`
          });
        } else {
          checked.push("memory-fit");
        }
      } else if (weightsGb >= budgetGb) {
        add({
          rule: "weights-over-budget",
          area: "memory",
          severity: "block",
          title: `Weights alone are ${gbs(weightsGb)} per GPU${shareWord}, over the ${gbs(budgetGb)} budget`,
          detail: `Budget = ${budgetWord}. Weights: ${w.basis}. Nothing is left for the KV cache, so the engine fails at load.`,
          fix: `Raise TP (or PP), quantize the weights, or use a larger card.`
        });
      } else if (conc < 1) {
        add({
          rule: "kv-under-one-sequence",
          area: "memory",
          severity: engine === "vllm" ? "block" : "warn",
          title: `One ${servedLen}-token sequence needs ${gbs(kvSeqGb)} of KV per GPU; at most ${gbs(capGb)} is left`,
          detail: `Budget = ${budgetWord}; weights ${gbs(weightsGb)} per GPU${shareWord}. ${engine === "vllm" ? 'vLLM checks at startup that one request at the full length fits the KV cache and refuses otherwise ("larger than the maximum number of tokens that can be stored in KV cache").' : "The KV pool cannot hold one request at the full length, so the server starts and the longest requests it advertises cannot be scheduled."} Capacity here is an upper bound: the engine's reserve comes out of it too.`,
          fix: `Serve at most ~${Math.floor(tokenCapacity / 1024) * 1024} tokens, raise ${engine === "vllm" ? "--gpu-memory-utilization" : "--mem-fraction-static"}, use an fp8 KV cache (--kv-cache-dtype fp8), or raise TP.`
        });
      } else {
        checked.push("memory-fit");
        if (conc < 1.5) {
          add({
            rule: "kv-borderline",
            area: "memory",
            severity: "warn",
            title: `Only ~${conc.toFixed(1)} full-length sequences fit, before the engine's own reserve`,
            detail: `${gbs(capGb)} per GPU for KV against ${gbs(kvSeqGb)} per ${servedLen}-token sequence. The activation profile and CUDA graphs come out of the same budget, so this can fail the startup check that one full sequence fits.`,
            fix: `Leave more room: lower the max length, use an fp8 KV cache, or raise TP.`
          });
        }
        if (args.maxNumSeqs != null && args.maxNumSeqs > Math.floor(conc)) {
          add({
            rule: "seqs-over-capacity",
            area: "memory",
            severity: "warn",
            title: `${engine === "vllm" ? "--max-num-seqs" : "--max-running-requests"} ${args.maxNumSeqs}, but at most ${Math.floor(conc)} sequences fit at the full ${servedLen} tokens`,
            detail: `KV capacity is at most ${tokenCapacity.toLocaleString("en-US")} tokens per replica (each GPU holds its share of every token; ${gbs(capGb)} at ${Math.round(kvPerTokGpu).toLocaleString("en-US")} bytes per token, ${kvName}). Shorter requests pack more, but once long ones arrive the engine preempts or queues past ~${Math.floor(conc)}; the batch size the flag promises is not there.`,
            fix: `Set the flag to what fits (${Math.max(1, Math.floor(conc))}), shorten the max length, use an fp8 KV cache, or add GPUs as DP replicas.`
          });
        }
        if (!findings.some((f) => f.rule === "seqs-over-capacity")) add({
          rule: "kv-capacity",
          area: "memory",
          severity: "info",
          title: `At most ${Math.floor(conc)} concurrent ${servedLen}-token sequences per replica (${tokenCapacity.toLocaleString("en-US")} KV tokens)`,
          detail: `Per GPU: budget ${budgetWord}; weights ${gbs(weightsGb)}${shareWord} (${w.basis}); KV ${Math.round(kvPerTokGpu).toLocaleString("en-US")} bytes per token (${kvName}${!hasMla && geo.numKVHeads && tp > geo.numKVHeads ? `, KV heads replicated past TP ${geo.numKVHeads}` : ""}${hasMla ? ", MLA latent cache is not split by TP" : ""}). Upper bound: the engine's reserve is not counted.`,
          fix: ""
        });
      }
    }
  }
  const template = chatTemplateOf(input);
  const haveTokenizerSide = input.tokenizerConfig != null || input.chatTemplateFile != null;
  if (!haveTokenizerSide && !args.chatTemplate) {
    skipped.push({ check: "chat-template", reason: "no tokenizer_config.json or chat_template.jinja was given (they are fetched for a Hugging Face id)." });
  } else {
    checked.push("chat-template");
    if (args.chatTemplate) {
      add({
        rule: "template-overridden",
        area: "template",
        severity: "info",
        title: `The chat template is replaced by ${args.chatTemplate}`,
        detail: `${template ? `The model ships its own template (hash ${templateHash(template)}); ` : ""}the override wins for every chat request. Nothing here renders the two, so whether they produce the same prompt is not checked.`,
        fix: "Render one multi-turn conversation with tool calls through both and diff the prompts before serving."
      });
    } else if (!template) {
      add({
        rule: "template-missing",
        area: "template",
        severity: "warn",
        title: "The model ships no chat template",
        detail: "Chat completions have nothing to render with: transformers 4.44+ no longer supplies a default, so vLLM and SGLang refuse /v1/chat/completions, and llama.cpp falls back to a generic format. Fine for a base model served on /v1/completions only.",
        fix: `Pass ${lc ? "--chat-template-file" : "--chat-template"} with the template the model was trained on, or serve completions only.`
      });
    }
    const effTemplate = args.chatTemplate ? null : template;
    if (effTemplate) {
      const addBos = input.tokenizerConfig?.add_bos_token === true;
      if (addBos && /bos_token/.test(effTemplate)) {
        add({
          rule: "bos-duplication",
          area: "template",
          severity: "info",
          title: "The template inserts the BOS token and the tokenizer also adds one (add_bos_token: true)",
          detail: lc ? "llama.cpp prints a warning and keeps both when a rendered prompt already starts with BOS; two BOS tokens shift every position the model saw in training." : "The chat path tokenizes the rendered template without special tokens, so chat is safe; a client that renders the template itself and posts to /v1/completions gets two BOS tokens.",
          fix: "Render with the engine's chat endpoint, or tokenize client-rendered prompts with add_special_tokens=False."
        });
      }
      const port = checkTemplatePortability(effTemplate, { bos_token: specialTokenText(input.tokenizerConfig?.bos_token), eos_token: specialTokenText(input.tokenizerConfig?.eos_token) });
      if (port.undefinedCalls.length) {
        const names = port.undefinedCalls.map((c) => c.name);
        const failed = port.cases.filter((c) => !c.ok && !c.raised).map((c) => c.case);
        add({
          rule: "template-undefined-call",
          area: "template",
          severity: "warn",
          title: `The chat template calls ${names.join(", ")} but never defines ${names.length === 1 ? "it" : "them"}`,
          detail: `Requests that never reach the call render fine; the first one that does fails (Python Jinja2 raises UndefinedError, JavaScript renderers "Cannot call something that is not a function").${failed.length ? ` Rendered here with ${port.renderer}, the ${failed.join(", ")} case failed.` : ""} LM Studio's bundled Gemma 4 template shipped this way: plain chat worked and every request with tools[] failed (lmstudio-ai/lmstudio-bug-tracker#2012).`,
          fix: `Define ${names.length === 1 ? "the macro" : "the macros"} in the template, or pass ${lc ? "--chat-template-file" : "--chat-template"} with the model repo's own template at a revision that defines ${names.length === 1 ? "it" : "them"}.`,
          source: "github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2012"
        });
      } else if (lc && (!port.parsed || port.cases.some((c) => !c.ok && !c.raised))) {
        const bad = port.cases.filter((c) => !c.ok && !c.raised);
        add({
          rule: "template-subset-renderer",
          area: "template",
          severity: "info",
          title: port.parsed ? `A non-Python Jinja renderer (${port.renderer}) could not render the ${bad.map((c) => c.case).join(", ")} case` : `A non-Python Jinja renderer (${port.renderer}) could not parse the template`,
          detail: `${port.parsed ? bad.map((c) => `${c.case}: ${c.message}`).join("; ") : port.parseError}. llama.cpp's --jinja renderer is another Jinja subset; this is a sign the template leans on a Python-only filter or construct, not proof llama.cpp fails.`,
          fix: "Render a request of this shape once against the llama.cpp server before routing traffic, or pass --chat-template-file with a template that avoids the construct."
        });
      }
      const thinking = /<think>|enable_thinking|reasoning_content/.test(effTemplate);
      if (thinking && !lc && !args.reasoningParser) {
        add({
          rule: "reasoning-parser-missing",
          area: "template",
          severity: "info",
          title: "The template has a thinking block and no --reasoning-parser is set",
          detail: "The reasoning text comes back inside message.content instead of reasoning_content, which breaks clients that hide or strip it.",
          fix: `Add --reasoning-parser ${/qwen3/i.test(ident) ? "qwen3" : /deepseek/i.test(ident) ? "deepseek_r1" : "<family>"}.`
        });
      }
    }
  }
  if (!lc && args.enableAutoToolChoice && !args.toolCallParser) {
    checked.push("tool-parser");
    add({
      rule: "tool-parser-missing",
      area: "template",
      severity: "block",
      title: "--enable-auto-tool-choice without --tool-call-parser",
      detail: "vLLM refuses to start: automatic tool choice needs a parser to turn the model's tool-call text into tool_calls.",
      fix: "Add --tool-call-parser for this family (e.g. hermes for Qwen, llama3_json for Llama 3, kimi_k2, openai for gpt-oss)."
    });
  }
  if (engine === "vllm") {
    const hit = findRecipe(input.modelId ?? args.model ?? null);
    if (!hit) skipped.push({ check: "recipe", reason: `no vLLM recipe for "${input.modelId ?? args.model ?? "(no model id)"}" in the ${VLLM_RECIPES.count}-recipe index (vllm-project/recipes@${VLLM_RECIPES.commit.slice(0, 10)}).` });
    else {
      checked.push("recipe");
      const { id, recipe: rc, variant } = hit;
      const src = `github.com/vllm-project/recipes/blob/${VLLM_RECIPES.commit.slice(0, 10)}/${rc.file}`;
      const flags = new Set(args.flags ?? []);
      if (rc.minVllmVersion && ver && !vAtLeast(ver, readEngineVersion(rc.minVllmVersion))) {
        add({
          rule: "recipe-min-version",
          area: "template",
          severity: "warn",
          title: `The vLLM recipe for ${id} needs vLLM ${rc.minVllmVersion}; this launch assumes ${versionLabel}`,
          detail: "min_vllm_version is the earliest release the recipe maintainers say supports the model; an older release may lack the architecture, parser or kernels it relies on.",
          fix: `Run vLLM ${rc.minVllmVersion} or newer${rc.nightlyRequired ? " (the recipe says a nightly build is required)" : ""}.`,
          source: src
        });
      } else if (rc.nightlyRequired) {
        add({ rule: "recipe-nightly", area: "template", severity: "info", title: `The vLLM recipe for ${id} says a nightly build is required`, detail: `min_vllm_version ${rc.minVllmVersion ?? "?"} had not shipped when the recipe was written.`, fix: "Install from the nightly index, or check a newer release.", source: src });
      }
      if ((rc.baseArgs ?? []).includes("--trust-remote-code") && !flags.has("--trust-remote-code")) {
        add({
          rule: "recipe-trust-remote-code",
          area: "template",
          severity: "warn",
          title: `The vLLM recipe for ${id} launches with --trust-remote-code; this command does not`,
          detail: "The model ships Python code in its repo (a config class, tokenizer or architecture). Without the flag vLLM refuses to load it.",
          fix: "Add --trust-remote-code (after reading the repo code).",
          source: src
        });
      }
      if (rc.toolCallParser && args.toolCallParser && args.toolCallParser !== rc.toolCallParser) {
        add({
          rule: "recipe-tool-parser",
          area: "template",
          severity: "warn",
          title: `--tool-call-parser ${args.toolCallParser}, but the vLLM recipe for ${id} uses ${rc.toolCallParser}`,
          detail: "A parser written for another family does not recognise this model's tool-call syntax: tool calls come back as plain text in content, or are cut.",
          fix: `Use --tool-call-parser ${rc.toolCallParser}.`,
          source: src
        });
      } else if (rc.toolCallParser && args.enableAutoToolChoice && !args.toolCallParser) {
        const f = findings.find((x) => x.rule === "tool-parser-missing");
        if (f) {
          f.fix = `Add --tool-call-parser ${rc.toolCallParser} (the vLLM recipe for ${id}).`;
          f.source = src;
        }
      }
      if (rc.reasoningParser && args.reasoningParser && args.reasoningParser !== rc.reasoningParser) {
        add({
          rule: "recipe-reasoning-parser",
          area: "template",
          severity: "warn",
          title: `--reasoning-parser ${args.reasoningParser}, but the vLLM recipe for ${id} uses ${rc.reasoningParser}`,
          detail: "A reasoning parser looks for the family's own thinking markers; the wrong one leaves the reasoning inside content or swallows the answer.",
          fix: `Use --reasoning-parser ${rc.reasoningParser}.`,
          source: src
        });
      } else if (rc.reasoningParser && !args.reasoningParser) {
        const f = findings.find((x) => x.rule === "reasoning-parser-missing");
        if (f) {
          f.fix = `Add --reasoning-parser ${rc.reasoningParser} (the vLLM recipe for ${id}).`;
          f.source = src;
        } else add({
          rule: "recipe-reasoning-parser",
          area: "template",
          severity: "info",
          title: `The vLLM recipe for ${id} passes --reasoning-parser ${rc.reasoningParser}; this command does not`,
          detail: "Without it the reasoning text comes back inside message.content instead of reasoning_content.",
          fix: `Add --reasoning-parser ${rc.reasoningParser} if clients read reasoning_content.`,
          source: src
        });
      }
      const va = variant ? rc.variants[variant] : null;
      if (va?.extraArgs?.includes("--kv-cache-dtype") && !args.kvCacheDtype && !(ckpt.kvDtype && /fp8/i.test(String(va.extraArgs[va.extraArgs.indexOf("--kv-cache-dtype") + 1])))) {
        const want = va.extraArgs[va.extraArgs.indexOf("--kv-cache-dtype") + 1];
        add({ rule: "recipe-variant-args", area: "template", severity: "info", title: `The vLLM recipe's ${variant} variant passes --kv-cache-dtype ${want}`, detail: `${va.modelId ?? id} is served there with: ${va.extraArgs.join(" ")}.`, fix: `Add --kv-cache-dtype ${want} unless you measured otherwise.`, source: src });
      }
      const rg = card ? RECIPE_GPU_OF[card.id] : void 0;
      if (rg && rc.unsupportedGpus?.includes(rg)) {
        add({ rule: "recipe-gpu-unsupported", area: "template", severity: "warn", title: `The vLLM recipe for ${id} marks ${card.label} unsupported`, detail: "The recipe maintainers record a confirmed blocker on this card.", fix: "Use a card the recipe lists as verified or supported.", source: src });
      }
    }
  }
  const effSampler = effectiveSampler(input);
  for (const k of KNOWN_SERVE_ISSUES) {
    if (!k.match.test(modelType) && !k.match.test(input.modelId ?? "") && !k.match.test(model.name ?? "")) continue;
    if (k.when && !k.when({ args, sampler: effSampler.values })) continue;
    checked.push(`known:${k.id}`);
    if (k.engines.includes(engine)) add({ rule: k.id, area: k.id === "qwen3-greedy" ? "sampler" : "template", severity: k.severity, title: k.title, detail: k.detail, fix: k.fix, source: k.source });
    else add({ rule: k.id, area: "template", severity: "info", title: `${k.title} (not ${engine === "vllm" ? "vLLM" : engine})`, detail: k.detail, fix: "No change needed on this engine; keep it in mind if the deployment moves.", source: k.source });
  }
  const gen = input.generationConfig ?? null;
  if (!gen) skipped.push({ check: "sampler", reason: "no generation_config.json (fetched for a Hugging Face id)." });
  else {
    checked.push("sampler");
    const rec = readSampler(gen);
    const diffs = [];
    for (const k of ["temperature", "top_p", "top_k", "repetition_penalty"]) {
      const want = rec[k];
      const got = effSampler.values[k];
      if (want == null || got == null) continue;
      const norm = (x) => k === "top_k" && x < 0 ? 0 : x;
      if (Math.abs(norm(want) - norm(got)) > 1e-9) diffs.push(`${k} ${got} (recommended ${want}, from ${effSampler.origin[k]})`);
    }
    if (diffs.length) {
      add({
        rule: "sampler-differs",
        area: "sampler",
        severity: "warn",
        title: `Sampling defaults differ from the model's generation_config.json: ${diffs.map((d) => d.split(" (")[0]).join(", ")}`,
        detail: `${diffs.join("; ")}. Applies to every request that does not send its own value. Model makers have traced host-to-host quality gaps to exactly this (Moonshot: "a significant portion" of discrepancies came from decoding-parameter misuse).`,
        fix: lc ? `Pass ${Object.entries(rec).map(([k, v]) => `--${k === "temperature" ? "temp" : k === "repetition_penalty" ? "repeat-penalty" : k.replace("_", "-")} ${v}`).join(" ")} (llama.cpp does not read generation_config.json).` : engine === "vllm" ? "Leave --generation-config at auto (the default reads generation_config.json), and do not override it in the gateway." : "Let the engine read the model's sampling defaults, and do not override them in the gateway.",
        source: "kimi.com/blog/kimi-vendor-verifier"
      });
    }
    const eos = gen.eos_token_id ?? config?.eos_token_id;
    const pad = gen.pad_token_id ?? config?.pad_token_id;
    const eosList = Array.isArray(eos) ? eos : eos != null ? [eos] : [];
    if (pad != null && eosList.some((e) => e === pad)) {
      add({
        rule: "pad-is-eos",
        area: "template",
        severity: "info",
        title: `pad_token_id ${pad} is also an EOS id`,
        detail: "Serving engines do not pad, so generation is unaffected; a fine-tune on this tokenizer that masks pad tokens out of the loss also masks every EOS, and the tuned model stops learning to stop.",
        fix: "For fine-tuning, set a distinct pad token."
      });
    }
  }
  let change = null;
  if (input.against) {
    checked.push("change");
    change = diffAgainst(input, memory, derived.len, template);
    for (const c of change.changed) {
      add({ rule: "changed", area: "change", severity: "info", title: `${c.key}: ${fmt(c.from)} -> ${fmt(c.to)}`, detail: c.effect, fix: "" });
    }
    for (const n2 of change.numbers) {
      add({ rule: "changed-number", area: "change", severity: "info", title: `${n2.what}: ${n2.from} -> ${n2.to}`, detail: "Same engine flags and card, computed for both models.", fix: "" });
    }
  }
  return {
    engine,
    resolved: {
      gpu: card?.id ?? null,
      gpuMemoryGb: card?.memoryGB ?? null,
      sm,
      gpus,
      tp,
      pp,
      dp,
      expertParallel: ep,
      dtype,
      quant: lc ? ggufType ?? "gguf" : quant,
      kvCacheDtype: lc ? `${args.cacheTypeK ?? "f16"}/${args.cacheTypeV ?? "f16"}` : (!args.kvCacheDtype || args.kvCacheDtype === "auto") && engine === "vllm" && ckpt.kvDtype ? `auto (${ckpt.kvDtype} from the checkpoint)` : args.kvCacheDtype ?? "auto",
      servedLen,
      servedLenSource,
      derivedMaxLen: derived.len,
      memFraction: lc ? null : engine === "vllm" ? args.gpuMemoryUtilization ?? vllmDefaults?.gpuMemoryUtilization ?? 0.92 : args.memFractionStatic ?? sglDefault?.memFractionStatic ?? null,
      maxNumSeqs: args.maxNumSeqs ?? null,
      engineVersion: lc ? input.engineVersion ?? null : versionLabel,
      engineVersionAssumed: !lc && input.engineVersion == null,
      defaultsAssumed
    },
    checked,
    skipped,
    findings,
    memory,
    change,
    basis: memory?.method === "upper-bound" ? `${SERVE_BASIS} ${SERVE_BASIS_UPPER_BOUND}` : SERVE_BASIS
  };
}
function effectiveSampler(input) {
  const { engine, args } = input;
  const def = ENGINE_SAMPLER_DEFAULTS[engine];
  const values = { temperature: def.temperature, top_p: def.top_p, top_k: def.top_k, repetition_penalty: def.repetition_penalty };
  const origin = { temperature: "engine default", top_p: "engine default", top_k: "engine default", repetition_penalty: "engine default" };
  const readsModel = engine === "vllm" ? !/^vllm$/i.test(args.generationConfigMode ?? "auto") : engine === "sglang" ? !/^openai$/i.test(args.generationConfigMode ?? "model") : false;
  if (readsModel && input.generationConfig) {
    const m = readSampler(input.generationConfig);
    for (const k of Object.keys(m)) {
      values[k] = m[k];
      origin[k] = "generation_config.json";
    }
  }
  if (engine !== "vllm" && engine !== "sglang" && !input.generationConfig) {
  }
  if (!readsModel && engine !== "llamacpp") {
    for (const k of Object.keys(origin)) origin[k] = engine === "vllm" ? "vLLM default (--generation-config vllm)" : "SGLang default (--sampling-defaults openai)";
  }
  if (engine === "llamacpp") for (const k of Object.keys(origin)) origin[k] = "llama.cpp default";
  for (const [src, s] of [[engine === "llamacpp" ? "llama.cpp flag" : "--override-generation-config", args.samplerFlags], ["request / gateway override", input.sampler]]) {
    if (!s) continue;
    for (const k of Object.keys(s)) {
      values[k] = s[k];
      origin[k] = src;
    }
  }
  return { values, origin };
}
var fmt = (v) => v === void 0 ? "(absent)" : typeof v === "object" ? JSON.stringify(v) : String(v);
var DIFF_KEYS = [
  ["model_type", "A different architecture family: the engine loads a different model class, and every family-specific flag (tool-call parser, reasoning parser, chat template) must be re-chosen."],
  ["num_hidden_layers", "Layer count changes weights and KV per token; PP stage splits must be redone."],
  ["hidden_size", "Width changes weights, the TP divisibility of every linear, and activation memory."],
  ["num_attention_heads", "Head count changes which TP sizes are legal."],
  ["num_key_value_heads", "KV heads change KV bytes per token and where TP starts replicating KV."],
  ["head_dim", "Head dimension changes KV bytes per token and the attention kernels that apply."],
  ["intermediate_size", "FFN width changes weights and which TP sizes divide it."],
  ["moe_intermediate_size", "Expert width changes which TP sizes divide it (and block-quant alignment)."],
  ["num_experts", "Expert count changes legal EP sizes and expert weight memory."],
  ["num_local_experts", "Expert count changes legal EP sizes and expert weight memory."],
  ["vocab_size", "Vocabulary changes the embedding and LM head; the tokenizer changes with it, so token budgets and stop ids measured on the old model no longer apply."],
  ["max_position_embeddings", "The longest servable length changes; --max-model-len / --context-length may now be past it or leave length unused."],
  ["rope_scaling", "Rope scaling changes the derived maximum length and short-context quality."],
  ["rope_theta", "Rope base changes how positions are encoded; any rope override written for the old model is wrong for this one."],
  ["sliding_window", "Sliding window changes KV per token at long lengths and which attention backends apply."],
  ["torch_dtype", "Stored dtype changes weight memory and which cards run it natively."],
  ["quantization_config", "Quantization changes weight memory, which cards run it, and --quantization must match it."],
  ["tie_word_embeddings", "Tied embeddings change weight memory by one vocab x hidden matrix."],
  ["bos_token_id", "BOS id changes: prompt templates and BOS handling must follow."],
  ["eos_token_id", "Stop ids change: a gateway or client that hard-codes stop token ids stops at the wrong place or never stops."],
  ["pad_token_id", "Pad id changes (matters for batched fine-tuning, not for serving engines)."]
];
function diffAgainst(input, mem, derivedLen, template) {
  const old = input.against;
  const a = old.config ?? {};
  const b = input.config ?? {};
  const changed = [];
  const same = (x, y) => JSON.stringify(x ?? null) === JSON.stringify(y ?? null);
  for (const [k, effect] of DIFF_KEYS) {
    const ka = k === "torch_dtype" ? a.torch_dtype ?? a.dtype : a[k];
    const kb = k === "torch_dtype" ? b.torch_dtype ?? b.dtype : b[k];
    if (ka === void 0 && kb === void 0) continue;
    if (!same(ka, kb)) changed.push({ key: k, from: ka, to: kb, effect });
  }
  const tOld = chatTemplateOf(old);
  if ((tOld ?? "") !== (template ?? "")) {
    changed.push({
      key: "chat_template",
      from: tOld ? `hash ${templateHash(tOld)}` : void 0,
      to: template ? `hash ${templateHash(template)}` : void 0,
      effect: input.args.chatTemplate ? "The model's own template changed, but --chat-template overrides both: the override was written for one of them at most." : "Every chat request renders differently: re-render a reference conversation (with tool calls) and re-run the evals that depend on format."
    });
  }
  const ga = readSampler(old.generationConfig ?? {});
  const gb2 = readSampler(input.generationConfig ?? {});
  for (const k of ["temperature", "top_p", "top_k", "repetition_penalty"]) {
    if (ga[k] != null || gb2[k] != null) {
      if (ga[k] !== gb2[k]) changed.push({ key: `generation_config.${k}`, from: ga[k], to: gb2[k], effect: "The recommended sampling changes; a gateway that pins the old value now overrides the new model's recommendation." });
    }
  }
  const numbers = [];
  const oldReport = checkServeConfig({
    engine: input.engine,
    args: input.args,
    gpu: input.gpu,
    gpus: input.gpus,
    sampler: input.sampler,
    model: old.model,
    config: old.config ?? null,
    tokenizerConfig: old.tokenizerConfig ?? null,
    chatTemplateFile: old.chatTemplateFile ?? null,
    generationConfig: old.generationConfig ?? null,
    modelId: old.modelId
  });
  const om = oldReport.memory;
  if (om && mem) {
    numbers.push({ what: "weights per GPU", from: gbs(om.weightsGbPerGpu), to: gbs(mem.weightsGbPerGpu) });
    numbers.push({ what: "KV bytes per token per GPU", from: Math.round(om.kvBytesPerTokenPerGpu).toLocaleString("en-US"), to: Math.round(mem.kvBytesPerTokenPerGpu).toLocaleString("en-US") });
    numbers.push({ what: "concurrent full-length sequences (upper bound)", from: String(Math.floor(om.maxConcurrentFullLength)), to: String(Math.floor(mem.maxConcurrentFullLength)) });
  }
  if (oldReport.resolved.derivedMaxLen !== derivedLen) {
    numbers.push({ what: "derived maximum length", from: String(oldReport.resolved.derivedMaxLen ?? "n/a"), to: String(derivedLen ?? "n/a") });
  }
  return { against: old.modelId ?? old.model.name ?? "previous model", changed, numbers };
}
var ENGINE_NAME = { vllm: "vLLM", sglang: "SGLang", llamacpp: "llama.cpp" };
function renderServeText(r, name = "model") {
  const L = [];
  const R = r.resolved;
  L.push("");
  L.push(`  ${name} on ${ENGINE_NAME[r.engine]}`);
  const par = r.engine === "llamacpp" ? "" : `TP ${R.tp} x PP ${R.pp} x DP ${R.dp}${R.expertParallel ? " + EP" : ""}, `;
  L.push(`  ${par}${R.gpus} x ${R.gpu ?? "(no card given)"}${R.sm ? ` (sm${R.sm})` : ""}${r.engine === "llamacpp" ? "" : `, dtype ${R.dtype}`}, weights ${R.quant ?? "unquantized"}, KV cache ${R.kvCacheDtype}`);
  L.push(`  served length ${R.servedLen ?? "n/a"} (${R.servedLenSource}); the model's derived maximum ${R.derivedMaxLen ?? "n/a"}`);
  if (R.engineVersion) L.push(`  engine ${R.engineVersion}${R.engineVersionAssumed ? " (assumed; pass --" + r.engine + "-version to change)" : ""}`);
  for (const d of R.defaultsAssumed.filter((x) => !/^(vLLM|SGLang) /.test(x))) L.push(`  default assumed: ${d}`);
  if (r.memory) {
    const m = r.memory;
    L.push("");
    const g = (x) => `${x.toFixed(2)} GiB`;
    const gb2gib = (x) => x * 1e9 / 1024 ** 3;
    if (m.method === "engine" && r.engine === "vllm" && m.reserve && m.pool) {
      L.push(`  memory per GPU, predicted: ${g(m.cardTotalGib)} x ${R.memFraction} = ${g(gb2gib(m.budgetGbPerGpu))}`);
      for (const t of m.reserve) L.push(`                   - ${t.what} ${g(t.gib)}${t.measuredByEngine ? " (estimate; vLLM measures this)" : ""}`);
      L.push(`                   = KV cache ${g(gb2gib(m.kvCapacityGbPerGpu))}: ${m.pool.numBlocks.toLocaleString("en-US")} blocks of ${m.pool.blockSize} tokens (${m.pool.groups.map((x) => `${x.layers} ${x.kind}`).join(", ")})`);
      L.push(`                   ${Math.round(m.kvBytesPerTokenPerGpu).toLocaleString("en-US")} KV bytes per token over the full-length layers; one full-length request ${g(gb2gib(m.kvGbPerSequencePerGpu))}`);
      L.push(`                   vLLM would print: GPU KV cache size: ${m.tokenCapacity.toLocaleString("en-US")} tokens; Maximum concurrency for ${(R.servedLen ?? 0).toLocaleString("en-US")} tokens per request: ${m.maxConcurrentFullLength.toFixed(2)}x`);
    } else if (m.method === "engine" && r.engine === "sglang") {
      L.push(`  memory per GPU, predicted: ${g(m.cardTotalGib)} x mem_fraction_static ${R.memFraction}${m.sglangDefault && m.sglangDefault.memFractionStatic !== R.memFraction ? ` (SGLang's own default here: ${m.sglangDefault.memFractionStatic})` : " (SGLang's own default)"} = ${g(gb2gib(m.budgetGbPerGpu))}`);
      L.push(`                   - weights ${g(gb2gib(m.weightsGbPerGpu))}`);
      L.push(`                   = KV cache ${g(gb2gib(m.kvCapacityGbPerGpu))}: ~${m.tokenCapacity.toLocaleString("en-US")} tokens at ${Math.round(m.kvBytesPerTokenPerGpu).toLocaleString("en-US")} bytes per token, ${Math.floor(m.maxConcurrentFullLength)} full-length requests`);
    } else if (r.engine === "llamacpp") {
      L.push(`  memory per GPU, predicted: card ${m.budgetGbPerGpu.toFixed(1)} GB, weights ${m.weightsGbPerGpu.toFixed(2)} GB, KV for ${m.tokenCapacity.toLocaleString("en-US")} tokens ${m.kvCapacityGbPerGpu.toFixed(2)} GB (allocated up front), total ${(m.weightsGbPerGpu + m.kvCapacityGbPerGpu).toFixed(2)} GB`);
      L.push(`                   ${Math.round(m.kvBytesPerTokenPerGpu).toLocaleString("en-US")} KV bytes per token; one full-length sequence ${m.kvGbPerSequencePerGpu.toFixed(2)} GB`);
    } else {
      L.push(`  memory per GPU, upper bound   budget ${m.budgetGbPerGpu.toFixed(1)} GB, weights ${m.weightsGbPerGpu.toFixed(2)} GB, KV room ${m.kvCapacityGbPerGpu.toFixed(2)} GB`);
      L.push(`                   ${Math.round(m.kvBytesPerTokenPerGpu).toLocaleString("en-US")} KV bytes per token; one full-length sequence ${m.kvGbPerSequencePerGpu.toFixed(2)} GB`);
      L.push(`                   at most ${Math.floor(m.maxConcurrentFullLength)} concurrent full-length sequences (${m.tokenCapacity.toLocaleString("en-US")} tokens)`);
    }
    L.push(`                   weights: ${m.weightsBasis}`);
    if (m.cardTotalSource && m.method === "engine" && r.engine !== "llamacpp") L.push(`                   card total: ${m.cardTotalSource}`);
    for (const n2 of m.layoutNotes ?? []) L.push(`                   ${n2}`);
  }
  const order = { block: 0, warn: 1, info: 2 };
  const fs = [...r.findings].sort((a, b) => order[a.severity] - order[b.severity]);
  L.push("");
  if (!fs.length) L.push("  no findings in the checks that ran");
  for (const f of fs) {
    const tag = f.severity === "block" ? "block" : f.severity === "warn" ? "warn " : "info ";
    L.push(`  ${tag} ${f.title}`);
    if (f.detail) L.push(`        ${f.detail}`);
    if (f.fix) L.push(`        fix: ${f.fix}`);
    if (f.source) L.push(`        source: ${f.source}`);
  }
  L.push("");
  L.push(`  checked: ${r.checked.join(", ") || "nothing"}`);
  for (const s of r.skipped) L.push(`  not checked: ${s.check}: ${s.reason}`);
  L.push("");
  L.push(`  ${r.basis}`);
  L.push("");
  return L.join("\n");
}

// src/utils/checkVerdict.ts
var VERDICT_PARTS = ["design", "serve", "layout", "data", "change"];
var RANK = { block: 3, ask: 2, warn: 1, ok: 0 };
function toneOfFindings(findings) {
  if (!Array.isArray(findings)) return "ok";
  let tone = "ok";
  for (const f of findings) {
    if (f?.severity === "block") return "block";
    if (f?.severity === "warn") tone = "warn";
  }
  return tone;
}
function toneOfChange(change) {
  const gate = change?.gate;
  if (!gate) return "ok";
  if (gate.ok === false || Array.isArray(gate.blockers) && gate.blockers.length > 0) return "block";
  return Array.isArray(gate.warnings) && gate.warnings.length > 0 ? "warn" : "ok";
}
function foldCheckVerdict(designVerdict, blocks) {
  const parts = { design: designVerdict };
  if (blocks.serve != null) parts.serve = toneOfFindings(blocks.serve.findings);
  if (blocks.layout != null) parts.layout = toneOfFindings(blocks.layout.findings);
  if (blocks.dataFit != null) parts.data = toneOfFindings(blocks.dataFit.findings);
  if (blocks.change != null) parts.change = toneOfChange(blocks.change);
  let verdict = designVerdict;
  for (const p of VERDICT_PARTS) {
    const t = parts[p];
    if (t != null && RANK[t] > RANK[verdict]) verdict = t;
  }
  const verdictSources = VERDICT_PARTS.filter((p) => parts[p] === verdict);
  return { verdict, designVerdict, verdictSources, verdictParts: parts };
}
export {
  ACTIVATION_K,
  ATTENTION_K,
  DEFAULT_ATTENTION,
  DEFAULT_GEN_TOKENS,
  DEFAULT_INFERENCE_BATCH,
  DEFAULT_OOM_TRAIN_BATCH,
  DEFAULT_SEQ_LEN,
  EDIT_RULE_IDS,
  EDIT_RULE_SEEN_IN,
  FIT_HEADROOM,
  INFERENCE_NOT_COUNTED,
  INTERNAL_K,
  LAYOUT_BASIS,
  MEMORY_CALIBRATION,
  MIG_PROFILES,
  MIG_SOURCE,
  OPTIMIZER_K,
  OUTPUT_K,
  QUANT_SUPPORT,
  RECURRENT_K,
  SERVE_BASIS,
  SHAREABLE_NOTE,
  TRANSIENT_K,
  UPPER_COVERAGE,
  UPPER_FLOOR_GB,
  UPPER_K,
  VERDICT_PARTS,
  activationGbFor,
  adapterParamsFor,
  attentionPhrase,
  attentionScoresGbFor,
  borderlineSentence,
  buildPlan,
  calibrationSentence,
  changeGateSentence,
  checkComfyWorkflow,
  checkConfigEdit,
  checkDataFit,
  checkGraphEdit,
  checkLoraTargets,
  checkParallelLayout,
  checkServeConfig,
  deviceMapFor,
  diagnoseOom,
  factsFor2 as factsFor,
  fitVerdict,
  fmtParamCount,
  foldCheckVerdict,
  historySentence,
  inferenceFactsFor,
  inferencePeakFor,
  inputContractOf,
  layerMemoryTerms,
  legalTpSizes,
  looksLikeHFConfig,
  loraTargetsFor,
  maxBatchFor,
  memoryTermsFor,
  oomFactsForModel,
  parseHumanSize,
  parseOomMessage,
  parseServeCommand,
  peakInputsOf,
  readAttention,
  readDataSpec,
  readGraph,
  readHistogram,
  readLayout,
  readPolicy,
  readSeqLen,
  readServeSpec,
  renderEditFindingsMarkdown,
  renderLayoutText,
  renderLoraText,
  renderOomText,
  renderPlanMarkdown,
  renderPlanText,
  renderServeText,
  shareLine,
  shareableFor,
  smallestCardFor,
  smallestSafeCardFor,
  storesScoresUnder,
  tokenSeqLenOf,
  tpGeometryOf,
  trainingPeakFor,
  upperBoundPhrase,
  upperPeakGb,
  verdictWord,
  withSeqLen
};
/*! @license Data derived from NeuSight (MIT) and PyTorch (BSD-3-Clause); notices below.
 * GENERATED by scripts/gemm-efficiency/fit_gemm_efficiency.py. Do not edit by hand.
 *
 * Per-card GEMM efficiency SHAPE (measured throughput over the same card's
 * large-GEMM throughput) against log2 of the GEMM's time at that plateau, in
 * microseconds. Fitted offline from NeuSight's measured linear-kernel
 * latencies (https://github.com/scai-tech/NeuSight @ 6945927d9afc, MIT,
 * Copyright (c) 2025 Georgia Institute of Technology and Microsoft Corporation).
 * Datasheet peaks transcribed from PyTorch torch/_inductor/analysis/device_info.py
 * (https://github.com/pytorch/pytorch @ 6a6db44e15f9, BSD-3-Clause,
 * Copyright (c) 2016- Facebook, Inc and the PyTorch contributors). Full notices: NOTICE.
 * NeuSight's rows for these cards are FP32 GEMMs; see the fit script for the
 * one assumption that carries their shape over to bf16.
 */
/*! @license Schedule tables and simulation ported from AMD-AGI/Primus, MIT, Copyright (c) 2025 Advanced Micro Devices, Inc. Full notice: NOTICE. */
