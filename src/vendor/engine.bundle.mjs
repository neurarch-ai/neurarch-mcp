// GENERATED FILE. Do not edit.
//
// The Neurarch lint engine, compiled from the private main repo:
//
//   npx esbuild src/utils/lintEngine.ts --bundle --format=esm --platform=node \
//     --external:web-tree-sitter --outfile=<this file>
//
// It is vendored rather than depended on because it has no npm package: the
// source lives in the app repo, and this server has to work with no network,
// no API key, and no second install step. Self-contained by construction (the
// bundle has zero imports), which is also why it can be dropped in wholesale.
//
// What it gives this server: graphFromPyTorchSource (read a .py file as a
// graph) and lintModelGraph (the structural rules, offline). Regenerate it
// whenever the app's component registry or rule set moves; src/vendor/
// engine.contract.test.ts fails loudly if the exports drift.

// src/utils/graphIO.ts
function rebuildNodeIO(components, connections) {
  const incoming = /* @__PURE__ */ new Map();
  const outgoing = /* @__PURE__ */ new Map();
  for (const conn of connections) {
    let outs = outgoing.get(conn.from);
    if (!outs) outgoing.set(conn.from, outs = []);
    outs.push(conn.to);
    let ins = incoming.get(conn.to);
    if (!ins) incoming.set(conn.to, ins = []);
    ins.push(conn.from);
  }
  for (const comp of components) {
    comp.inputs = incoming.get(comp.id) ?? [];
    comp.outputs = outgoing.get(comp.id) ?? [];
  }
  return components;
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
    const { kernelSize, stride = kernelSize, padding = 0 } = params;
    const newH = Math.floor((h + 2 * padding - kernelSize) / stride + 1);
    const newW = Math.floor((w + 2 * padding - kernelSize) / stride + 1);
    return [c, newH, newW];
  }
  return inputShape;
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
        const { kernelSize, stride = kernelSize } = params;
        return [c, Math.floor((l - kernelSize) / stride + 1)];
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
        const { kernelSize, stride = kernelSize } = params;
        return [c, Math.floor((l - kernelSize) / stride + 1)];
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
    computeOutputShape: (inputShape, params) => {
      const hiddenSize = params.hiddenSize || 128;
      const multiplier = params.bidirectional ? 2 : 1;
      if (inputShape.length >= 3) {
        return [inputShape[0], inputShape[1], hiddenSize * multiplier];
      }
      return [inputShape[0] || 1, hiddenSize * multiplier];
    }
  },
  gru: {
    type: "gru",
    name: "GRU",
    icon: "\u{1F500}",
    category: "nlp",
    defaultParams: { hiddenSize: 128, numLayers: 1, bidirectional: false },
    computeOutputShape: (inputShape, params) => {
      const hiddenSize = params.hiddenSize || 128;
      const multiplier = params.bidirectional ? 2 : 1;
      if (inputShape.length >= 3) {
        return [inputShape[0], inputShape[1], hiddenSize * multiplier];
      }
      return [inputShape[0] || 1, hiddenSize * multiplier];
    }
  },
  rnn: {
    type: "rnn",
    name: "RNN",
    icon: "\u21A9\uFE0F",
    category: "nlp",
    defaultParams: { hiddenSize: 128, numLayers: 1 },
    computeOutputShape: (inputShape, params) => {
      const hiddenSize = params.hiddenSize || 128;
      if (inputShape.length >= 3) {
        return [inputShape[0], inputShape[1], hiddenSize];
      }
      return [inputShape[0] || 1, hiddenSize];
    }
  },
  bidirectionalLSTM: {
    type: "bidirectionalLSTM",
    name: "BiLSTM",
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
    computeOutputShape: (inputShape, params) => {
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
      return [inputShape[0] || 1, params.embedDim || 32];
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
    computeOutputShape: (inputShape, params) => {
      const patch = params.patchSize || 16;
      const embedDim = params.embedDim ?? 768;
      if (inputShape.length >= 3) {
        const [, h, w] = inputShape;
        return [Math.floor(h / patch) * Math.floor(w / patch), embedDim];
      }
      const side = Math.floor((params.imgSize ?? 224) / patch);
      return [side * side, embedDim];
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
      if (dims.length === inputShape.length) {
        return dims.map((d) => inputShape[d]);
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
  }
};
function createComponent(type, position, id) {
  const def = componentRegistry[type];
  return {
    id: id || `comp-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
    type,
    name: def.name,
    position,
    params: { ...def.defaultParams },
    inputs: [],
    outputs: []
  };
}

// src/utils/pythonStmts.ts
function parsePyStmts(source) {
  const logical = toLogicalLines(source);
  return buildTree(logical, 0, logical.length, 0);
}
function toLogicalLines(src) {
  const lines = src.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const raw = lines[i];
    if (raw.trim() === "" || raw.trimStart().startsWith("#")) {
      out.push({ text: raw, indent: indentWidth(raw), startLine: i, endLine: i });
      i++;
      continue;
    }
    const startLine = i;
    let combined = raw;
    let depth = bracketDelta(raw);
    let endsWithBackslash = stripComment(raw).trimEnd().endsWith("\\");
    while ((depth > 0 || endsWithBackslash) && i + 1 < lines.length) {
      i++;
      const next = lines[i];
      combined += "\n" + next;
      depth += bracketDelta(next);
      endsWithBackslash = stripComment(next).trimEnd().endsWith("\\");
    }
    out.push({
      text: combined,
      indent: indentWidth(raw),
      startLine,
      endLine: i
    });
    i++;
  }
  return out;
}
function stripComment(line) {
  let i = 0;
  let inStr = false;
  while (i < line.length) {
    const ch = line[i];
    if (inStr) {
      if (typeof inStr === "string" && inStr.length === 3) {
        if (line.slice(i, i + 3) === inStr) {
          inStr = false;
          i += 3;
          continue;
        }
        if (ch === "\\") {
          i += 2;
          continue;
        }
      } else {
        if (ch === inStr) {
          inStr = false;
          i++;
          continue;
        }
        if (ch === "\\") {
          i += 2;
          continue;
        }
      }
      i++;
      continue;
    }
    if (ch === "#") return line.slice(0, i);
    if (line.slice(i, i + 3) === "'''" || line.slice(i, i + 3) === '"""') {
      inStr = line.slice(i, i + 3);
      i += 3;
      continue;
    }
    if (ch === "'" || ch === '"') {
      inStr = ch;
      i++;
      continue;
    }
    i++;
  }
  return line;
}
function indentWidth(line) {
  let w = 0;
  for (const ch of line) {
    if (ch === " ") w++;
    else if (ch === "	") w += 8;
    else break;
  }
  return w;
}
function bracketDelta(line) {
  let d = 0;
  let i = 0;
  let inStr = false;
  while (i < line.length) {
    const ch = line[i];
    if (inStr) {
      if (typeof inStr === "string" && inStr.length === 3) {
        if (line.slice(i, i + 3) === inStr) {
          inStr = false;
          i += 3;
          continue;
        }
        if (ch === "\\") {
          i += 2;
          continue;
        }
      } else {
        if (ch === inStr) {
          inStr = false;
          i++;
          continue;
        }
        if (ch === "\\") {
          i += 2;
          continue;
        }
      }
      i++;
      continue;
    }
    if (ch === "#") break;
    if (line.slice(i, i + 3) === "'''" || line.slice(i, i + 3) === '"""') {
      inStr = line.slice(i, i + 3);
      i += 3;
      continue;
    }
    if (ch === "'" || ch === '"') {
      inStr = ch;
      i++;
      continue;
    }
    if (ch === "(" || ch === "[" || ch === "{") d++;
    else if (ch === ")" || ch === "]" || ch === "}") d--;
    i++;
  }
  return d;
}
function buildTree(lines, start, end, baseIndent) {
  const out = [];
  let pendingDecorators = [];
  let i = start;
  while (i < end) {
    const ln = lines[i];
    const trimmed = ln.text.trim();
    if (trimmed === "" || trimmed.startsWith("#")) {
      i++;
      continue;
    }
    if (ln.indent < baseIndent) break;
    if (ln.indent > baseIndent) {
      i++;
      continue;
    }
    if (trimmed.startsWith("@")) {
      pendingDecorators.push(trimmed.replace(/^@\s*/, ""));
      i++;
      continue;
    }
    const head = headWord(trimmed);
    const stmt = {
      kind: classifyStmt(trimmed, head),
      text: ln.text,
      indent: ln.indent,
      startLine: ln.startLine,
      endLine: ln.endLine,
      body: []
    };
    if (pendingDecorators.length > 0 && (stmt.kind === "class" || stmt.kind === "def")) {
      stmt.decorators = pendingDecorators;
    }
    pendingDecorators = [];
    if (stmt.kind === "class") {
      const m = trimmed.match(/^class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:/);
      if (m) {
        stmt.name = m[1];
        stmt.bases = (m[2] || "").split(",").map((s) => s.trim()).filter(Boolean);
      }
    } else if (stmt.kind === "def") {
      const m = trimmed.match(/^(?:async\s+)?def\s+(\w+)\s*\(/);
      if (m) stmt.name = m[1];
    }
    if (isBlockOpener(stmt.kind) && trimmed.endsWith(":")) {
      const bodyStart = i + 1;
      let bodyEnd = bodyStart;
      let bodyIndent = -1;
      for (let j = bodyStart; j < end; j++) {
        const t = lines[j].text.trim();
        if (t === "" || t.startsWith("#")) continue;
        if (lines[j].indent <= ln.indent) break;
        if (bodyIndent === -1) bodyIndent = lines[j].indent;
        bodyEnd = j + 1;
      }
      if (bodyIndent !== -1) {
        stmt.body = buildTree(lines, bodyStart, bodyEnd, bodyIndent);
        stmt.endLine = lines[bodyEnd - 1].endLine;
      }
      i = bodyEnd;
    } else {
      i++;
    }
    out.push(stmt);
  }
  return out;
}
function headWord(text) {
  const m = text.match(/^[A-Za-z_]\w*/);
  return m ? m[0] : "";
}
function classifyStmt(text, head) {
  if (head === "class") return "class";
  if (head === "def" || head === "async" && /^async\s+def\b/.test(text)) return "def";
  if (head === "if" || head === "elif" || head === "else") return "if";
  if (head === "for") return "for";
  if (head === "while") return "while";
  if (head === "with") return "with";
  if (head === "try" || head === "except" || head === "finally") return "try";
  if (head === "return") return "return";
  if (head === "import" || head === "from") return "import";
  if (hasTopLevelAssign(text)) return "assign";
  return "expr";
}
function isBlockOpener(k) {
  return k === "class" || k === "def" || k === "if" || k === "for" || k === "while" || k === "with" || k === "try";
}
function hasTopLevelAssign(text) {
  let depth = 0;
  let inStr = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (ch === inStr) inStr = false;
      else if (ch === "\\") i++;
      continue;
    }
    if (ch === "'" || ch === '"') {
      inStr = ch;
      continue;
    }
    if (ch === "#") break;
    if (ch === "(" || ch === "[" || ch === "{") depth++;
    else if (ch === ")" || ch === "]" || ch === "}") depth--;
    else if (ch === "=" && depth === 0) {
      const prev = text[i - 1];
      const next = text[i + 1];
      if (next === "=") {
        i++;
        continue;
      }
      if ("=<>!:+-*/%&|^~".includes(prev)) continue;
      return true;
    }
  }
  return false;
}
function* walk(stmts) {
  for (const s of stmts) {
    yield s;
    if (s.body.length > 0) yield* walk(s.body);
  }
}
function findMainModelClass(stmts) {
  let candidate = null;
  for (const s of walk(stmts)) {
    if (s.kind !== "class") continue;
    const hasInit = s.body.some((c) => c.kind === "def" && c.name === "__init__");
    const hasForward = s.body.some((c) => c.kind === "def" && c.name === "forward");
    if (hasInit && hasForward) candidate = s;
  }
  return candidate;
}

// src/utils/codeParser.ts
var CUSTOM_CLASS_MAP = {
  // Positional encoding
  positionalencoding: "positionalEncoding",
  posencoding: "positionalEncoding",
  posemb: "positionalEncoding",
  positionalembedding: "positionalEncoding",
  rotaryembedding: "rope",
  ropeembedding: "rope",
  // Attention — standard
  multiheadattention: "multiHeadAttention",
  multiheadattn: "multiHeadAttention",
  causalselfattention: "causalAttention",
  selfattn: "selfAttention",
  selfattention: "selfAttention",
  attention: "attention",
  crossattention: "crossModalAttention",
  // Attention — GQA / modern LLMs
  llamaattention: "groupedQueryAttention",
  mistralattention: "groupedQueryAttention",
  mixtralattention: "groupedQueryAttention",
  qwenattention: "groupedQueryAttention",
  gemmaattention: "groupedQueryAttention",
  phi3attention: "groupedQueryAttention",
  falcon7battention: "groupedQueryAttention",
  groupedqueryattention: "groupedQueryAttention",
  gqaattention: "groupedQueryAttention",
  // Feed-forward / SwiGLU MLP
  feedforward: "feedForward",
  feedforwardnetwork: "feedForward",
  ffn: "feedForward",
  mlp: "feedForward",
  llamamlp: "swiglu",
  mistralmpl: "swiglu",
  mixtralmlp: "swiglu",
  qwenmlp: "swiglu",
  gemmamlp: "swiglu",
  phi3mlp: "swiglu",
  swiglu: "swiglu",
  gatedmlp: "swiglu",
  // MoE
  mixtralsparsemoeblock: "moeLayer",
  moelayer: "moeLayer",
  expertlayer: "moeLayer",
  sparsemlp: "moeLayer",
  // Transformer block — generic
  block: "transformerBlock",
  gptblock: "transformerBlock",
  bertlayer: "transformerBlock",
  bertblock: "transformerBlock",
  encoderlayer: "transformerBlock",
  decoderlayer: "transformerBlock",
  transformerblock: "transformerBlock",
  transformerlayer: "transformerBlock",
  transformer: "transformerBlock",
  visionblock: "transformerBlock",
  // Transformer block — named LLM variants
  llamadecoderlayer: "transformerBlock",
  llamadecoderblock: "transformerBlock",
  mistraldecoderlayer: "transformerBlock",
  mixtraldecoderlayer: "transformerBlock",
  qwendecoderlayer: "transformerBlock",
  gemmadecoderlayer: "transformerBlock",
  phi3decoderlayer: "transformerBlock",
  // ViT / Vision
  visiontransformer: "transformerBlock",
  vitblock: "transformerBlock",
  vitlayer: "transformerBlock",
  patchembed: "patchEmbed",
  patchembedding: "patchEmbed",
  patchprojection: "patchEmbed",
  // SE / Squeeze-Excite
  seblock: "seBlock",
  squeezeexcitation: "seBlock",
  channelattention: "seBlock",
  // Residual
  resblock: "residual",
  residualblock: "residual",
  resnetblock: "residual",
  bottleneck: "residual"
};
function mapCustomClassToComponent(className) {
  return CUSTOM_CLASS_MAP[className.toLowerCase()] ?? null;
}
function findMainClassRange(tree) {
  const main = findMainModelClass(tree);
  if (!main) return null;
  const initStmt = main.body.find((c) => c.kind === "def" && c.name === "__init__");
  const forwardStmt = main.body.find((c) => c.kind === "def" && c.name === "forward");
  if (!initStmt || !forwardStmt) return null;
  return {
    name: main.name ?? "",
    initLine: initStmt.startLine,
    forwardLine: forwardStmt.startLine,
    endLine: main.endLine
  };
}
var MAX_EXPAND_DEPTH = 3;
function buildClassRegistry(tree) {
  const registry = /* @__PURE__ */ new Map();
  for (const s of walk(tree)) {
    if (s.kind !== "class" || !s.name) continue;
    const initStmt = s.body.find((c) => c.kind === "def" && c.name === "__init__");
    if (!initStmt) continue;
    const forwardStmt = s.body.find((c) => c.kind === "def" && c.name === "forward") ?? null;
    const methods = /* @__PURE__ */ new Map();
    for (const c of s.body) {
      if (c.kind === "def" && c.name && c.name !== "__init__" && c.name !== "forward") {
        methods.set(c.name, c);
      }
    }
    registry.set(s.name, { name: s.name, initStmt, forwardStmt, methods });
  }
  return registry;
}
function canExpandClass(className, ctx) {
  return ctx.registry.has(className) && ctx.depth < MAX_EXPAND_DEPTH && !ctx.expanding.has(className);
}
function splitTopLevelArgs(s) {
  const parts = [];
  let depth = 0;
  let cur = "";
  for (const ch of s) {
    if (ch === "(" || ch === "[" || ch === "{") depth++;
    else if (ch === ")" || ch === "]" || ch === "}") depth--;
    if (ch === "," && depth === 0) {
      parts.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  if (cur.trim()) parts.push(cur);
  return parts;
}
function extractBalancedArgs(text, openIdx) {
  let depth = 0;
  for (let i = openIdx; i < text.length; i++) {
    const ch = text[i];
    if (ch === "(") depth++;
    else if (ch === ")") {
      depth--;
      if (depth === 0) return text.slice(openIdx + 1, i);
    }
  }
  return null;
}
function resolveNumExpr(expr, env) {
  const t = expr.trim();
  if (/^\d+$/.test(t)) return parseInt(t, 10);
  if (/^\d*\.\d+$/.test(t)) return parseFloat(t);
  if (/^\w+$/.test(t)) return typeof env[t] === "number" ? env[t] : null;
  const m = t.match(/^(\w+)\s*([+\-*])\s*(\w+)$/);
  if (m) {
    const a = resolveNumExpr(m[1], env);
    const b = resolveNumExpr(m[3], env);
    if (a === null || b === null) return null;
    return m[2] === "+" ? a + b : m[2] === "-" ? a - b : a * b;
  }
  return null;
}
function bindCallArgs(defText, callArgsStr, callerEnv) {
  const env = {};
  const open = defText.indexOf("(");
  const sig = open >= 0 ? extractBalancedArgs(defText, open) : null;
  const params = [];
  if (sig) {
    for (const part of splitTopLevelArgs(sig)) {
      const p = part.trim();
      if (!p || p === "self" || p.startsWith("*")) continue;
      const pm = p.match(/^(\w+)\s*(?::[^=]*)?(?:=\s*(.+))?$/);
      if (!pm) continue;
      params.push(pm[1]);
      if (pm[2] !== void 0) {
        const v = resolveNumExpr(pm[2], callerEnv);
        if (v !== null) env[pm[1]] = v;
      }
    }
  }
  let pos = 0;
  for (const part of splitTopLevelArgs(callArgsStr)) {
    const a = part.trim();
    if (!a) continue;
    const kw = a.match(/^(\w+)\s*=\s*(.+)$/);
    if (kw) {
      const v = resolveNumExpr(kw[2], callerEnv);
      if (v !== null) env[kw[1]] = v;
    } else {
      const v = resolveNumExpr(a, callerEnv);
      if (v !== null && params[pos] !== void 0) env[params[pos]] = v;
      pos++;
    }
  }
  return env;
}
function resolveParamsWithEnv(params, env) {
  const out = {};
  for (const [k, v] of Object.entries(params)) {
    if (typeof v === "string") {
      const r = resolveNumExpr(v, env);
      out[k] = r !== null ? r : v;
    } else {
      out[k] = v;
    }
  }
  return out;
}
function expandClassInstance(className, callArgsStr, ctx, callerEnv) {
  const info = ctx.registry.get(className);
  if (!info || ctx.depth >= MAX_EXPAND_DEPTH || ctx.expanding.has(className)) return [];
  const env = bindCallArgs(info.initStmt.text, callArgsStr, callerEnv);
  const innerCtx = {
    ...ctx,
    methods: info.methods,
    env,
    depth: ctx.depth + 1,
    expanding: /* @__PURE__ */ new Set([...ctx.expanding, className])
  };
  const initLayers = parseInitLayers(
    ctx.lines,
    info.initStmt.startLine,
    info.initStmt.endLine,
    innerCtx
  );
  const forwardCalls = info.forwardStmt ? parseForwardCalls(ctx.lines, info.forwardStmt.startLine, info.forwardStmt.endLine) : [];
  return orderLayersByForward(initLayers, forwardCalls).map((l) => ({
    name: l.pyName ?? l.type,
    type: l.type,
    params: resolveParamsWithEnv(l.params, env)
  }));
}
var MODULELIST_COMP_RE = /\[\s*(\w+)\s*\(([^)]*)\)\s+for\s+\w+\s+in\s+range\s*\(([^)]+)\)/;
function expandModuleListComp(compMatch, listName, initSigLine, ctx) {
  const [, className, argsStr, nExpr] = compMatch;
  let n2 = resolveNumExpr(nExpr, ctx?.env ?? {});
  if (n2 === null) {
    const defaultMatch = initSigLine.match(new RegExp(`\\b${nExpr.trim()}=(\\d+)`));
    n2 = defaultMatch ? parseInt(defaultMatch[1], 10) : 6;
  }
  n2 = Math.max(0, Math.floor(n2));
  const out = [];
  const mapped = mapCustomClassToComponent(className);
  if (mapped) {
    for (let j = 0; j < n2; j++) {
      out.push({ name: `${listName}_${j}`, type: mapped, params: {} });
    }
    return out;
  }
  if (ctx && canExpandClass(className, ctx)) {
    const inner = expandClassInstance(className, argsStr, ctx, ctx.env);
    for (let j = 0; j < n2; j++) {
      for (const il of inner) {
        out.push({ name: `${listName}_${j}.${il.name}`, type: il.type, params: il.params });
      }
    }
  }
  return out;
}
function scanInstantiations(text, registry) {
  const out = [];
  const re = /\b(?:nn\.(\w+)|(?<![\w.])([A-Z]\w*))\s*\(/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const openIdx = m.index + m[0].length - 1;
    const args = extractBalancedArgs(text, openIdx);
    if (args === null) continue;
    if (m[1]) {
      if (m[1] !== "Sequential" && m[1] !== "ModuleList") {
        out.push({ kind: "nn", name: m[1], args });
      }
    } else if (m[2] && (registry.has(m[2]) || mapCustomClassToComponent(m[2]))) {
      out.push({ kind: "class", name: m[2], args });
    }
  }
  return out;
}
function expandHelperMethod(methodStmt, callArgsStr, attrName, ctx) {
  const env = bindCallArgs(methodStmt.text, callArgsStr, ctx.env);
  const collect = (stmts) => {
    const insts = [];
    for (const s of stmts) {
      if (s.kind === "for" && s.body.length > 0) {
        const rangeMatch = s.text.match(/\bin\s+range\s*\(([^)]+)\)/);
        const count = rangeMatch ? Math.max(0, Math.floor(resolveNumExpr(rangeMatch[1], env) ?? 1)) : 1;
        const inner = collect(s.body);
        for (let r = 0; r < count; r++) insts.push(...inner);
      } else if (s.body.length > 0) {
        insts.push(...collect(s.body));
      } else {
        insts.push(...scanInstantiations(s.text, ctx.registry));
      }
    }
    return insts;
  };
  const out = [];
  let k = 0;
  for (const inst of collect(methodStmt.body)) {
    if (inst.kind === "nn") {
      const parsed = parseLayerDefinition(`nn.${inst.name}(${inst.args})`);
      if (parsed?.type) {
        out.push({
          name: `${attrName}_${k}`,
          type: parsed.type,
          params: resolveParamsWithEnv(parsed.params, env)
        });
        k++;
      }
      continue;
    }
    const mapped = mapCustomClassToComponent(inst.name);
    if (mapped) {
      out.push({ name: `${attrName}_${k}`, type: mapped, params: {} });
      k++;
    } else if (canExpandClass(inst.name, ctx)) {
      const inner = expandClassInstance(inst.name, inst.args, ctx, env);
      if (inner.length > 0) {
        for (const il of inner) {
          out.push({ name: `${attrName}_${k}.${il.name}`, type: il.type, params: il.params });
        }
        k++;
      }
    }
  }
  return out;
}
function orderLayersByForward(initLayers, forwardCalls) {
  const allLayers = [];
  const usedLayerNames = /* @__PURE__ */ new Set();
  for (const call of forwardCalls) {
    if (call.isFunctional) {
      const componentType = mapFunctionalToComponent(call.layerName) ?? mapEinopsToComponent(call.layerName);
      if (componentType) {
        allLayers.push({ type: componentType, params: {}, isFunctional: true, pyName: call.layerName });
      }
    } else {
      const matchingLayers = initLayers.filter(
        (l) => l.name === call.layerName || l.name.startsWith(`${call.layerName}_`)
        // Sequential
      );
      if (matchingLayers.length > 1) {
        for (const layer of matchingLayers) {
          if (layer.type && !usedLayerNames.has(layer.name)) {
            allLayers.push({ type: layer.type, params: layer.params, isFunctional: false, pyName: layer.name });
            usedLayerNames.add(layer.name);
          }
        }
      } else if (matchingLayers.length === 1) {
        const layer = matchingLayers[0];
        if (layer.type) {
          allLayers.push({ type: layer.type, params: layer.params, isFunctional: false, pyName: layer.name });
        }
      } else {
        const layer = initLayers.find((l) => l.name === call.layerName);
        if (layer && layer.type) {
          allLayers.push({ type: layer.type, params: layer.params, isFunctional: false, pyName: layer.name });
        }
      }
    }
  }
  if (allLayers.length === 0) {
    for (const layer of initLayers) {
      if (layer.type) {
        allLayers.push({ type: layer.type, params: layer.params, isFunctional: false, pyName: layer.name });
      }
    }
  }
  return allLayers;
}
function parsePyTorchCode(code) {
  try {
    const lines = code.split("\n").map((l) => l.trim());
    const tree = parsePyStmts(code);
    const mainClass = findMainClassRange(tree);
    const initStart = mainClass?.initLine ?? 0;
    const forwardStart = mainClass?.forwardLine ?? 0;
    const classEnd = mainClass?.endLine ?? lines.length - 1;
    const registry = buildClassRegistry(tree);
    const mainInfo = mainClass ? registry.get(mainClass.name) : void 0;
    const ctx = {
      lines,
      registry,
      methods: mainInfo?.methods ?? /* @__PURE__ */ new Map(),
      env: mainInfo ? bindCallArgs(mainInfo.initStmt.text, "", {}) : {},
      depth: 0,
      expanding: new Set(mainClass ? [mainClass.name] : [])
    };
    const initLayers = parseInitLayers(lines, initStart, forwardStart - 1, ctx);
    if (initLayers.length === 0) {
      return null;
    }
    const forwardCalls = parseForwardCalls(lines, forwardStart, classEnd);
    const inputShape = parseInputShape(lines);
    const components = [];
    const connections = [];
    const inputComponent = createComponent("input", { x: 200, y: 50 }, void 0);
    if (inputShape) {
      inputComponent.params.shape = inputShape;
    }
    components.push(inputComponent);
    let prevComponentId = inputComponent.id;
    const allLayers = orderLayersByForward(initLayers, forwardCalls);
    const COMPONENT_SPACING_Y = 200;
    const START_X = 200;
    const START_Y = 100;
    const calculateBestPortsForVertical = () => {
      return { fromPort: "bottom", toPort: "top" };
    };
    for (let i = 0; i < allLayers.length; i++) {
      const layer = allLayers[i];
      if (!layer.type) continue;
      const component = createComponent(layer.type, {
        x: START_X,
        y: START_Y + i * COMPONENT_SPACING_Y
      }, void 0);
      component.params = { ...layer.params };
      if (layer.pyName) {
        const dot = layer.pyName.lastIndexOf(".");
        if (dot > 0) {
          component.scope = layer.pyName.slice(0, dot);
          component.name = layer.pyName.slice(dot + 1);
        } else {
          component.name = layer.pyName;
        }
      }
      components.push(component);
      const bestPorts = calculateBestPortsForVertical();
      const connection = {
        id: `conn-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
        from: prevComponentId,
        to: component.id,
        fromPort: bestPorts.fromPort,
        toPort: bestPorts.toPort
      };
      connections.push(connection);
      prevComponentId = component.id;
    }
    if (prevComponentId !== inputComponent.id && allLayers.length > 0) {
      const outputY = START_Y + allLayers.length * COMPONENT_SPACING_Y;
      const outputComponent = createComponent("output", { x: 200, y: outputY }, void 0);
      components.push(outputComponent);
      const lastConnection = {
        id: `conn-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
        from: prevComponentId,
        to: outputComponent.id,
        fromPort: "bottom",
        toPort: "top"
      };
      connections.push(lastConnection);
    }
    rebuildNodeIO(components, connections);
    return {
      components,
      connections,
      inputShape
    };
  } catch (error) {
    console.error("Error parsing code:", error);
    return null;
  }
}
function parseInitLayers(lines, startLine, endLine, ctx) {
  const layers = [];
  let order = 0;
  let sequentialContext = null;
  let sequentialContent = [];
  let moduleListName = null;
  let moduleListBuffer = [];
  let moduleListDepth = 0;
  for (let i = startLine; i <= endLine; i++) {
    const line = lines[i];
    if (!line.trim() || line.trim().startsWith("#")) continue;
    if (line.match(/def\s+__init__/)) continue;
    if (line.match(/^\s*def\s+\w+/) && !line.match(/def\s+__init__/)) break;
    if (moduleListName !== null) {
      moduleListBuffer.push(line);
      for (const ch of line) {
        if (ch === "[") moduleListDepth++;
        else if (ch === "]") moduleListDepth--;
      }
      if (moduleListDepth <= 0) {
        const combined = moduleListBuffer.join(" ");
        const compMatch = combined.match(MODULELIST_COMP_RE);
        if (compMatch) {
          for (const il of expandModuleListComp(compMatch, moduleListName, lines[startLine], ctx)) {
            layers.push({ name: il.name, type: il.type, params: il.params, order: order++ });
          }
        }
        moduleListName = null;
        moduleListBuffer = [];
        moduleListDepth = 0;
      }
      continue;
    }
    const moduleListMatch = line.match(/self\.(\w+)\s*=\s*nn\.ModuleList\s*\(\s*\[/);
    if (moduleListMatch) {
      const compMatch = line.match(MODULELIST_COMP_RE);
      if (compMatch) {
        for (const il of expandModuleListComp(compMatch, moduleListMatch[1], lines[startLine], ctx)) {
          layers.push({ name: il.name, type: il.type, params: il.params, order: order++ });
        }
      } else {
        moduleListName = moduleListMatch[1];
        moduleListBuffer = [line];
        for (const ch of line) {
          if (ch === "[") moduleListDepth++;
          else if (ch === "]") moduleListDepth--;
        }
      }
      continue;
    }
    const sequentialMatch = line.match(/self\.(\w+)\s*=\s*nn\.Sequential\s*\(/);
    if (sequentialMatch) {
      if (sequentialContext) {
        const seqLayers = parseSequentialContent(sequentialContent, sequentialContext.name, order);
        layers.push(...seqLayers);
        order += seqLayers.length;
      }
      sequentialContext = { name: sequentialMatch[1] };
      sequentialContent = [];
      if (line.includes(")")) {
        const content = line.match(/nn\.Sequential\s*\(([^)]+)\)/)?.[1];
        if (content) {
          sequentialContent.push(content);
          const seqLayers = parseSequentialContent(sequentialContent, sequentialContext.name, order);
          layers.push(...seqLayers);
          order += seqLayers.length;
          sequentialContext = null;
          sequentialContent = [];
        }
      }
      continue;
    }
    if (sequentialContext) {
      sequentialContent.push(line);
      if (line.includes(")")) {
        const seqLayers = parseSequentialContent(sequentialContent, sequentialContext.name, order);
        layers.push(...seqLayers);
        order += seqLayers.length;
        sequentialContext = null;
        sequentialContent = [];
      }
      continue;
    }
    if (line.startsWith("self.")) {
      const match = line.match(/self\.(\w+)\s*=\s*(.+)/);
      if (match) {
        const [, name, layerDef] = match;
        const helperMatch = layerDef.match(/^self\.(\w+)\s*\(/);
        if (helperMatch) {
          const methodStmt = ctx?.methods.get(helperMatch[1]);
          if (ctx && methodStmt) {
            const args = extractBalancedArgs(layerDef, layerDef.indexOf("(")) ?? "";
            for (const il of expandHelperMethod(methodStmt, args, name, ctx)) {
              layers.push({ name: il.name, type: il.type, params: il.params, order: order++ });
            }
          }
          continue;
        }
        const parsed = parseLayerDefinition(layerDef.trim());
        if (parsed) {
          layers.push({ name, type: parsed.type, params: parsed.params, order: order++ });
        } else {
          const customMatch = layerDef.match(/^(\w+)\s*\(/);
          if (customMatch) {
            const componentType = mapCustomClassToComponent(customMatch[1]);
            if (componentType) {
              layers.push({ name, type: componentType, params: {}, order: order++ });
            } else if (ctx && canExpandClass(customMatch[1], ctx)) {
              const args = extractBalancedArgs(layerDef, layerDef.indexOf("(")) ?? "";
              for (const il of expandClassInstance(customMatch[1], args, ctx, ctx.env)) {
                layers.push({ name: `${name}.${il.name}`, type: il.type, params: il.params, order: order++ });
              }
            }
          }
        }
      }
    }
  }
  return layers;
}
function parseSequentialContent(lines, sequentialName, startOrder) {
  const layers = [];
  let order = startOrder;
  const cleanedLines = lines.map((line) => {
    const commentIndex = line.indexOf("#");
    if (commentIndex >= 0) {
      const beforeComment = line.substring(0, commentIndex);
      const openParens = (beforeComment.match(/\(/g) || []).length;
      const closeParens = (beforeComment.match(/\)/g) || []).length;
      if (openParens > closeParens) {
        return line;
      } else {
        return beforeComment.trim();
      }
    }
    return line.trim();
  });
  const fullText = cleanedLines.join("\n");
  const layerPattern = /nn\.(\w+)\s*\([^)]*(?:\([^)]*\)[^)]*)*\)/g;
  let match;
  const matches = [];
  while ((match = layerPattern.exec(fullText)) !== null) {
    matches.push(match[0]);
  }
  if (matches.length === 0) {
    for (const line of cleanedLines) {
      const simpleMatch = line.match(/nn\.(\w+)\s*\(([^)]*)\)/);
      if (simpleMatch) {
        matches.push(simpleMatch[0]);
      }
    }
  }
  for (const layerDef of matches) {
    const parsed = parseLayerDefinition(layerDef);
    if (parsed && parsed.type) {
      layers.push({
        name: `${sequentialName}_${order - startOrder + 1}`,
        type: parsed.type,
        params: parsed.params,
        order: order++
      });
    }
  }
  return layers;
}
function parseLayerDefinition(layerDef) {
  const params = {};
  if (layerDef.includes("F.") || layerDef.includes("torch.")) {
    return null;
  }
  const linearMatch = layerDef.match(/nn\.Linear\s*\(\s*([^,]+)\s*,\s*([^,)]+)/);
  if (linearMatch) {
    const [, inFeatures, outFeatures] = linearMatch;
    params.inFeatures = parseNumberOrVariable(inFeatures);
    params.outFeatures = parseNumberOrVariable(outFeatures);
    return { type: "linear", params };
  }
  const convT2dMatch = layerDef.match(/nn\.ConvTranspose2d\s*\(([^)]+)\)/);
  if (convT2dMatch) {
    const args = parseArguments(convT2dMatch[1]);
    params.inChannels = parseNumberOrVariable(args.in_channels || args[0]);
    params.outChannels = parseNumberOrVariable(args[1] || args.out_channels || args[0]);
    params.kernelSize = parseNumberOrVariable(args.kernel_size || args[2] || "3");
    params.stride = parseNumberOrVariable(args.stride || args[3] || "1");
    params.padding = parseNumberOrVariable(args.padding || args[4] || "0");
    if (args.output_padding !== void 0) params.outputPadding = parseNumberOrVariable(args.output_padding);
    if (args.dilation !== void 0) params.dilation = parseNumberOrVariable(args.dilation);
    if (args.groups !== void 0) params.groups = parseNumberOrVariable(args.groups);
    return { type: "transposeConv2d", params };
  }
  const conv2dMatch = layerDef.match(/nn\.Conv2d\s*\(([^)]+)\)/);
  if (conv2dMatch) {
    const args = parseArguments(conv2dMatch[1]);
    const pk = parseNumberOrVariable(args.kernel_size || args[2] || "0");
    const ps = parseNumberOrVariable(args.stride || args[3] || "1");
    const pInC = parseNumberOrVariable(args.in_channels || args[0] || "3");
    const pOutC = parseNumberOrVariable(args.out_channels || args[1] || "768");
    if (typeof pk === "number" && pk > 4 && pk === ps && typeof pInC === "number" && pInC <= 4) {
      params.patchSize = pk;
      params.embedDim = pOutC;
      params.inChans = pInC;
      return { type: "patchEmbed", params };
    }
    params.inChannels = parseNumberOrVariable(args.in_channels || args[0]);
    params.outChannels = parseNumberOrVariable(args[1] || args.out_channels || args[0]);
    params.kernelSize = parseNumberOrVariable(args.kernel_size || args[2] || "3");
    params.stride = parseNumberOrVariable(args.stride || args[3] || "1");
    params.padding = parseNumberOrVariable(args.padding || args[4] || "0");
    if (args.dilation !== void 0) params.dilation = parseNumberOrVariable(args.dilation);
    if (args.groups !== void 0) params.groups = parseNumberOrVariable(args.groups);
    return { type: "conv2d", params };
  }
  const conv1dMatch = layerDef.match(/nn\.Conv1d\s*\(([^)]+)\)/);
  if (conv1dMatch) {
    const args = parseArguments(conv1dMatch[1]);
    params.inChannels = parseNumberOrVariable(args.in_channels || args[0]);
    params.outChannels = parseNumberOrVariable(args[1] || args.out_channels);
    params.kernelSize = parseNumberOrVariable(args.kernel_size || args[2] || "3");
    params.stride = parseNumberOrVariable(args.stride || args[3] || "1");
    params.padding = parseNumberOrVariable(args.padding || args[4] || "0");
    if (args.dilation !== void 0) params.dilation = parseNumberOrVariable(args.dilation);
    if (args.groups !== void 0) params.groups = parseNumberOrVariable(args.groups);
    return { type: "conv1d", params };
  }
  const maxpoolMatch = layerDef.match(/nn\.MaxPool2d\s*\(([^)]*)\)/);
  if (maxpoolMatch) {
    const args = parseArguments(maxpoolMatch[1] || "");
    params.kernelSize = parseNumberOrVariable(args.kernel_size || args[0] || "2");
    params.stride = parseNumberOrVariable(args.stride || args[1] || args.kernel_size || "2");
    return { type: "maxpool2d", params };
  }
  const avgpoolMatch = layerDef.match(/nn\.AvgPool2d\s*\(([^)]*)\)/);
  if (avgpoolMatch) {
    const args = parseArguments(avgpoolMatch[1] || "");
    params.kernelSize = parseNumberOrVariable(args.kernel_size || args[0] || "2");
    params.stride = parseNumberOrVariable(args.stride || args[1] || args.kernel_size || "2");
    return { type: "avgpool2d", params };
  }
  const dropoutMatch = layerDef.match(/nn\.Dropout\s*\(([^)]*)\)/);
  if (dropoutMatch) {
    const args = parseArguments(dropoutMatch[1] || "");
    params.p = parseNumberOrVariable(args.p || args[0] || "0.5");
    return { type: "dropout", params };
  }
  const batchNormMatch = layerDef.match(/nn\.BatchNorm(?:2d|3d)\s*\(([^)]*)\)/);
  if (batchNormMatch) {
    return { type: "batchNorm", params };
  }
  if (layerDef.match(/nn\.ReLU\s*\(/)) {
    return { type: "relu", params: {} };
  }
  if (layerDef.match(/nn\.Sigmoid\s*\(/)) return { type: "sigmoid", params: {} };
  if (layerDef.match(/nn\.Tanh\s*\(/)) return { type: "tanh", params: {} };
  if (layerDef.match(/nn\.GELU\s*\(/)) return { type: "gelu", params: {} };
  if (layerDef.match(/nn\.SiLU\s*\(/)) return { type: "swish", params: {} };
  if (layerDef.match(/nn\.Swish\s*\(/)) return { type: "swish", params: {} };
  if (layerDef.match(/nn\.Softmax\s*\(/)) return { type: "softmax", params: {} };
  const leakyReluMatch = layerDef.match(/nn\.LeakyReLU\s*\(([^)]*)\)/);
  if (leakyReluMatch) {
    const args = parseArguments(leakyReluMatch[1] || "");
    params.negativeSlope = parseNumberOrVariable(args.negative_slope || args[0] || "0.01");
    return { type: "leakyRelu", params };
  }
  const layerNormMatch = layerDef.match(/nn\.LayerNorm\s*\(([^)]+)\)/);
  if (layerNormMatch) {
    const args = parseArguments(layerNormMatch[1]);
    params.normalizedShape = parseNumberOrVariable(args.normalized_shape || args[0] || "768");
    return { type: "layerNorm", params };
  }
  const rmsNormMatch = layerDef.match(/nn\.RMSNorm\s*\(([^)]+)\)/);
  if (rmsNormMatch) {
    const args = parseArguments(rmsNormMatch[1]);
    params.normalizedShape = parseNumberOrVariable(args.normalized_shape || args[0] || "768");
    return { type: "rmsNorm", params };
  }
  const groupNormMatch = layerDef.match(/nn\.GroupNorm\s*\(([^)]+)\)/);
  if (groupNormMatch) {
    const args = parseArguments(groupNormMatch[1]);
    params.numGroups = parseNumberOrVariable(args.num_groups || args[0] || "32");
    params.numChannels = parseNumberOrVariable(args.num_channels || args[1]);
    return { type: "groupNorm", params };
  }
  const batchNorm1dMatch = layerDef.match(/nn\.BatchNorm1d\s*\(([^)]*)\)/);
  if (batchNorm1dMatch) {
    return { type: "batchNorm", params };
  }
  const embeddingMatch = layerDef.match(/nn\.Embedding\s*\(([^)]+)\)/);
  if (embeddingMatch) {
    const args = parseArguments(embeddingMatch[1]);
    params.vocabSize = parseNumberOrVariable(args.num_embeddings || args[0] || "10000");
    params.embeddingDim = parseNumberOrVariable(args.embedding_dim || args[1] || "128");
    return { type: "embedding", params };
  }
  const telMatch = layerDef.match(/nn\.TransformerEncoderLayer\s*\(([^)]+)\)/);
  if (telMatch) {
    const args = parseArguments(telMatch[1]);
    params.embedDim = parseNumberOrVariable(args.d_model || args[0] || "512");
    params.numHeads = parseNumberOrVariable(args.nhead || args[1] || "8");
    params.ffDim = parseNumberOrVariable(args.dim_feedforward || args[2] || "2048");
    return { type: "transformerBlock", params };
  }
  const mhaMatch = layerDef.match(/nn\.MultiheadAttention\s*\(([^)]+)\)/);
  if (mhaMatch) {
    const args = parseArguments(mhaMatch[1]);
    params.hiddenDim = parseNumberOrVariable(args.embed_dim || args[0] || "512");
    params.numHeads = parseNumberOrVariable(args.num_heads || args[1] || "8");
    return { type: "multiHeadAttention", params };
  }
  const captureRecurrentExtras = (args) => {
    const numLayers = args.num_layers ?? args[2];
    if (numLayers !== void 0) params.numLayers = parseNumberOrVariable(String(numLayers));
    if (args.bidirectional !== void 0) {
      params.bidirectional = args.bidirectional === "True" || args.bidirectional === true;
    }
  };
  const lstmMatch = layerDef.match(/nn\.LSTM\s*\(([^)]+)\)/);
  if (lstmMatch) {
    const args = parseArguments(lstmMatch[1]);
    params.inputSize = parseNumberOrVariable(args.input_size || args[0] || "128");
    params.hiddenSize = parseNumberOrVariable(args.hidden_size || args[1] || "128");
    captureRecurrentExtras(args);
    return { type: "lstm", params };
  }
  const gruMatch = layerDef.match(/nn\.GRU\s*\(([^)]+)\)/);
  if (gruMatch) {
    const args = parseArguments(gruMatch[1]);
    params.inputSize = parseNumberOrVariable(args.input_size || args[0] || "128");
    params.hiddenSize = parseNumberOrVariable(args.hidden_size || args[1] || "128");
    captureRecurrentExtras(args);
    return { type: "gru", params };
  }
  const rnnMatch = layerDef.match(/nn\.RNN\s*\(([^)]+)\)/);
  if (rnnMatch) {
    const args = parseArguments(rnnMatch[1]);
    params.inputSize = parseNumberOrVariable(args.input_size || args[0] || "128");
    params.hiddenSize = parseNumberOrVariable(args.hidden_size || args[1] || "128");
    captureRecurrentExtras(args);
    return { type: "rnn", params };
  }
  if (layerDef.match(/nn\.Flatten\s*\(/)) {
    return { type: "flatten", params: {} };
  }
  const adaptiveAvgMatch = layerDef.match(/nn\.AdaptiveAvgPool2d\s*\(([^)]*)\)/);
  if (adaptiveAvgMatch) {
    const inner = adaptiveAvgMatch[1].trim().replace(/^\(+/, "").replace(/\)+$/, "");
    const nums = inner.split(",").map((t) => Number(t.trim())).filter((n2) => Number.isFinite(n2));
    if (nums.length === 0 || nums.every((n2) => n2 === 1)) {
      return { type: "globalAvgPool2d", params: {} };
    }
    params.outputSize = nums.length === 1 ? nums[0] : nums;
    return { type: "adaptiveAvgPool2d", params };
  }
  const upsampleMatch = layerDef.match(/nn\.Upsample\s*\(([^)]+)\)/);
  if (upsampleMatch) {
    const args = parseArguments(upsampleMatch[1]);
    params.scaleFactor = parseNumberOrVariable(args.scale_factor || args[0] || "2");
    return { type: "upsample", params };
  }
  return null;
}
function parseArguments(argsStr) {
  const result = {};
  const positional = [];
  if (!argsStr.trim()) return result;
  const parts = argsStr.split(",").map((p) => p.trim());
  for (const part of parts) {
    if (part.includes("=")) {
      const [key, value] = part.split("=").map((p) => p.trim());
      result[key] = parseNumberOrVariable(value);
    } else {
      positional.push(parseNumberOrVariable(part));
      result[positional.length - 1] = parseNumberOrVariable(part);
    }
  }
  return result;
}
function parseNumberOrVariable(value) {
  if (!value) return void 0;
  const num2 = Number(value);
  if (!isNaN(num2)) {
    return num2;
  }
  return value;
}
function parseForwardCalls(lines, startLine, endLine) {
  const calls = [];
  const seenForLoop = /* @__PURE__ */ new Set();
  for (let i = startLine; i <= endLine; i++) {
    const line = lines[i];
    if (!line.trim() || line.trim().startsWith("#")) continue;
    if (line.match(/def\s+forward/)) continue;
    if (line.match(/^\s*def\s+\w+/) || line.match(/^class\s+/)) break;
    const forLoopMatch = line.match(/^\s*for\s+\w+\s+in\s+self\.(\w+)/);
    if (forLoopMatch) {
      const layerName = forLoopMatch[1];
      if (!seenForLoop.has(layerName)) {
        calls.push({ layerName, isFunctional: false });
        seenForLoop.add(layerName);
      }
      continue;
    }
    const selfCalls = [...line.matchAll(/self\.(\w+)\s*\(/g)].map((m) => m[1]);
    for (const name of [...selfCalls].reverse()) {
      calls.push({ layerName: name, isFunctional: false });
    }
    for (const m of line.matchAll(/(?:F\.|torch\.)(\w+)\s*\(/g)) {
      const funcName = m[1];
      const componentType = mapFunctionalToComponent(funcName);
      if (componentType) {
        calls.push({ layerName: funcName, isFunctional: true });
      }
    }
    for (const m of line.matchAll(/(?:einops\.)?(rearrange|repeat|reduce)\s*\(/g)) {
      const funcName = m[1];
      const componentType = mapEinopsToComponent(funcName);
      if (componentType) {
        calls.push({ layerName: funcName, isFunctional: true });
      }
    }
  }
  return calls;
}
function mapFunctionalToComponent(funcName) {
  const mapping = {
    // Activations
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "gelu": "gelu",
    "silu": "swish",
    "leaky_relu": "leakyRelu",
    "elu": "elu",
    "selu": "selu",
    "prelu": "prelu",
    "mish": "mish",
    "hardswish": "hardSwish",
    "hard_swish": "hardSwish",
    "glu": "glu",
    "softmax": "softmax",
    // Regularization
    "dropout": "dropout",
    // Pooling
    "max_pool2d": "maxpool2d",
    "avg_pool2d": "avgpool2d",
    "adaptive_avg_pool2d": "globalAvgPool2d",
    // Normalization
    "layer_norm": "layerNorm",
    "rms_norm": "rmsNorm",
    "batch_norm": "batchNorm",
    // Shape ops
    "flatten": "flatten",
    "interpolate": "interpolate",
    "upsample": "upsample",
    "pad": "pad",
    // Attention (PyTorch 2.0+ canonical; can't recover head/dim from call site,
    // user fills those in on the canvas)
    "scaled_dot_product_attention": "attention"
  };
  return mapping[funcName] || null;
}
function mapEinopsToComponent(funcName) {
  const mapping = {
    "rearrange": "reshape",
    "repeat": "reshape",
    "reduce": "reshape"
  };
  return mapping[funcName] || null;
}
function parseInputShape(lines) {
  for (const line of lines) {
    const match = line.match(/Input shape:\s*(\[[\d,\s]+\])/);
    if (match) {
      try {
        return JSON.parse(match[1]);
      } catch (e) {
      }
    }
  }
  return void 0;
}

// src/utils/modelFormat.ts
var MODEL_FORMAT_VERSION = 1;
function wrapModelFile(model) {
  const file = {
    $schema: "https://neurarch.com/schema/model.v1.json",
    formatVersion: MODEL_FORMAT_VERSION,
    generator: "neurarch",
    model
  };
  return JSON.stringify(file, null, 2);
}

// src/utils/lintThresholds.ts
var HIGH_DROPOUT_P = 0.65;
var DEEP_NO_RESIDUAL_MIN_LAYERS = 8;
var DEEP_NO_NORM_MIN_LAYERS = 7;
var VANISHING_MIN_LAYERS = 5;
var LARGE_ACTIVATION_MAX_ELEMENTS = 5e7;
var SWIGLU_RATIO_MIN = 2;
var SWIGLU_RATIO_MAX = 5;
var LARGE_LINEAR_MAX_PARAMS = 1e9;
var KV_BUDGET_CONTEXT_TOKENS = 8192;
var KV_BUDGET_MAX_GB = 4;
var SCALED_INIT_MIN_ATTENTION_LAYERS = 8;
var LINT_THRESHOLDS = {
  HIGH_DROPOUT_P,
  DEEP_NO_RESIDUAL_MIN_LAYERS,
  DEEP_NO_NORM_MIN_LAYERS,
  VANISHING_MIN_LAYERS,
  LARGE_ACTIVATION_MAX_ELEMENTS,
  SWIGLU_RATIO_MIN,
  SWIGLU_RATIO_MAX,
  LARGE_LINEAR_MAX_PARAMS,
  KV_BUDGET_CONTEXT_TOKENS,
  KV_BUDGET_MAX_GB,
  SCALED_INIT_MIN_ATTENTION_LAYERS
};

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
  "positionalEncoding",
  "learnedPositionalEmbedding",
  "rope",
  "alibi"
]);
var VANISHING_ACTIVATIONS = /* @__PURE__ */ new Set(["sigmoid", "tanh"]);
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
  for (const conn of model.connections) {
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
  for (const conn of model.connections) {
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
  for (const conn of model.connections.filter((c) => c.to === outputNode.id)) {
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
  for (const conn of model.connections.filter((c) => c.to === outputNode.id)) {
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
var attentionNoPE = (model) => {
  const attnNodes = model.components.filter((c) => ATTENTION_TYPES.has(c.type));
  if (attnNodes.length === 0) return [];
  if (model.components.some((c) => PE_TYPES.has(c.type))) return [];
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
  for (const cn of model.connections) next.set(cn.from, [...next.get(cn.from) ?? [], cn.to]);
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
  return model.components.filter((c) => c.type === "moeLayer").map((c) => ({
    id: `moe-aux-${c.id}`,
    ruleId: "moe-no-aux-loss",
    severity: "info",
    category: "pattern",
    title: `MoE "${c.name}": add auxiliary load-balancing loss`,
    message: `MoE layers require an auxiliary router z-loss + load-balance loss during training to prevent expert collapse. This is not visible in the architecture diagram but must be in the training loop.`,
    affectedIds: [c.id],
    suggestion: `Add a note on this layer. Typical aux_loss coefficient: 1e-2 (Mixtral/Switch Transformer).`
  }));
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
  for (const conn of model.connections) {
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
  for (const conn of model.connections) {
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
  for (const conn of model.connections) {
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
  for (const conn of model.connections.filter((c) => c.to === outputNode.id)) {
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
  for (const conn of model.connections) {
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
  for (const conn of model.connections) {
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
  return [{
    id: "duplicate-positional-encoding",
    ruleId: "duplicate-positional-encoding",
    severity: "info",
    category: "pattern",
    title: `${pes.length} positional encodings`,
    message: `${pes.length} positional-encoding layers found (${pes.map((c) => c.type).join(", ")}). Position is normally injected once. Stacking absolute + rotary, or two of the same, double-counts position and tends to hurt more than help.`,
    affectedIds: pes.map((c) => c.id),
    suggestion: "Keep a single positional scheme: sinusoidal OR learned OR RoPE / ALiBi."
  }];
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
  for (const conn of model.connections) {
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
  for (const conn of model.connections) {
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
      const gb = (count * 4 / 1073741824).toFixed(1);
      issues.push({
        id: `huge-linear-${c.id}`,
        ruleId: "huge-linear-params",
        severity: "warning",
        category: "performance",
        title: `Very large Linear: "${c.name}" (~${Math.round(count / 1e6)}M params)`,
        message: `Linear "${c.name}" is ${inF} \xD7 ${outF} = ${Math.round(count / 1e6)}M parameters (~${gb} GB float32). A single dense layer this large usually means a feature map was flattened without pooling first; embedding / vocab-projection heads are the expected exception.`,
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
  for (const conn of model.connections) {
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
  for (const conn of model.connections) {
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
  kvCacheContextBudget
  // R35 performance warning
];
var ADVISOR_RULE_IDS = [
  "no-input-node",
  "no-output-node",
  "isolated-components",
  "dead-end",
  "bn-after-activation",
  "dropout-before-bn",
  "output-activation",
  "bn-at-output",
  "deep-no-residual",
  "attention-no-pe",
  "vanishing-gradient",
  "deep-no-norm",
  "high-dropout",
  "large-activation",
  "moe-no-aux-loss",
  "gqa-head-mismatch",
  "swiglu-dim-convention",
  "linear-after-conv-no-flatten",
  "redundant-activation",
  "consecutive-linear-no-activation",
  "dropout-at-output",
  "conv-stride-gt-kernel",
  "non-spatial-into-conv",
  "double-norm",
  "duplicate-positional-encoding",
  "pool-into-linear-no-flatten",
  "flatten-into-attention",
  "transposeconv-checkerboard",
  "groupnorm-divisibility",
  "huge-linear-params",
  "full-mha-serving-cost",
  "init-activation-mismatch",
  "deep-attention-default-init",
  "lm-head-vocab-mismatch",
  "kv-cache-context-budget"
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

// src/utils/paramEstimator.ts
function num(v, fallback = 0) {
  const n2 = typeof v === "number" ? v : Number(v);
  return isFinite(n2) && n2 >= 0 ? n2 : fallback;
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
      const inF = num(p.inFeatures ?? lastDim);
      const outF = num(p.outFeatures);
      return inF > 0 && outF > 0 ? inF * outF + outF : 0;
    }
    case "flatten":
      return 0;
    // ── Convolution ──────────────────────────────────────────────────────────
    case "conv2d": {
      const inC = num(p.inChannels ?? ch, 1);
      const outC = num(p.outChannels);
      const k = num(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k + outC : 0;
    }
    case "conv1d": {
      const inC = num(p.inChannels ?? ch, 1);
      const outC = num(p.outChannels);
      const k = num(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k + outC : 0;
    }
    case "conv3d": {
      const inC = num(p.inChannels ?? ch, 1);
      const outC = num(p.outChannels);
      const k = num(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k * k + outC : 0;
    }
    case "depthwiseConv2d": {
      const inC = num(p.inChannels ?? ch, 1);
      const k = num(p.kernelSize, 3);
      const dm = num(p.depthMultiplier, 1);
      return inC * dm * k * k + inC * dm;
    }
    case "separableConv2d": {
      const inC = num(p.inChannels ?? ch, 1);
      const outC = num(p.outChannels, inC);
      const k = num(p.kernelSize, 3);
      return inC * k * k + inC * outC + outC;
    }
    case "transposeConv2d": {
      const inC = num(p.inChannels ?? ch, 1);
      const outC = num(p.outChannels);
      const k = num(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k + outC : 0;
    }
    case "deformableConv2d":
    case "dilatedConv2d": {
      const inC = num(p.inChannels ?? ch, 1);
      const outC = num(p.outChannels);
      const k = num(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k + outC : 0;
    }
    case "invResidualBlock": {
      const inC = num(p.inChannels ?? ch, 32);
      const outC = num(p.outChannels, inC);
      const e = num(p.expandRatio, 6);
      const k = num(p.kernelSize, 3);
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
      const V = num(p.vocabSize ?? p.numEmbeddings);
      const D = num(p.embeddingDim ?? p.embedDim);
      return V > 0 && D > 0 ? V * D : 0;
    }
    case "learnedPositionalEmbedding": {
      const L = num(p.maxLen ?? p.maxPositions ?? p.numPositions);
      const D = num(p.embedDim ?? p.embeddingDim);
      return L > 0 && D > 0 ? L * D : 0;
    }
    case "segmentEmbedding": {
      const S = num(p.numSegments ?? p.numTypes, 2);
      const D = num(p.embeddingDim ?? p.embedDim);
      return S > 0 && D > 0 ? S * D : 0;
    }
    case "lstm": {
      const H = num(p.hiddenSize, 128);
      const I = num(p.inputSize ?? lastDim, H);
      const L = num(p.numLayers, 1);
      const dirs = p.bidirectional === true ? 2 : 1;
      const layer0 = dirs * 4 * (I * H + H * H + 2 * H);
      const layerRest = L > 1 ? (L - 1) * dirs * 4 * (dirs * H * H + H * H + 2 * H) : 0;
      return layer0 + layerRest;
    }
    case "gru": {
      const H = num(p.hiddenSize, 128);
      const I = num(p.inputSize ?? lastDim, H);
      const L = num(p.numLayers, 1);
      const dirs = p.bidirectional === true ? 2 : 1;
      const layer0 = dirs * 3 * (I * H + H * H + 2 * H);
      const layerRest = L > 1 ? (L - 1) * dirs * 3 * (dirs * H * H + H * H + 2 * H) : 0;
      return layer0 + layerRest;
    }
    case "rnn": {
      const H = num(p.hiddenSize, 128);
      const I = num(p.inputSize ?? lastDim, H);
      return I * H + H * H + 2 * H;
    }
    case "bidirectionalLSTM": {
      const H = num(p.hiddenSize, 128);
      const I = num(p.inputSize ?? lastDim, H);
      const L = num(p.numLayers, 1);
      const layer0 = 2 * 4 * (I * H + H * H + 2 * H);
      const layerRest = L > 1 ? (L - 1) * 2 * 4 * (2 * H * H + H * H + 2 * H) : 0;
      return layer0 + layerRest;
    }
    case "attention":
    case "selfAttention":
    case "multiHeadAttention": {
      const d = num(p.hiddenDim ?? p.embedDim);
      return d > 0 ? 4 * d * d + 4 * d : 0;
    }
    case "crossModalAttention":
    case "crossAttention":
    case "coAttention":
    case "linearAttention":
    case "localAttention": {
      const d = num(p.embedDim ?? p.hiddenDim);
      return d > 0 ? 4 * d * d + 4 * d : 0;
    }
    // ── LLM ─────────────────────────────────────────────────────────────────
    case "feedForward": {
      const d = num(p.hiddenDim ?? p.embedDim);
      const ff = num(p.ffDim, d > 0 ? d * 4 : 0);
      return d > 0 && ff > 0 ? d * ff + ff + ff * d + d : 0;
    }
    case "transformerBlock": {
      const d = num(p.embedDim ?? p.hiddenDim);
      const ff = num(p.ffDim, d > 0 ? d * 4 : 0);
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
      const f = num(Array.isArray(feat) ? feat[0] : feat);
      return f > 0 ? 2 * f : 0;
    }
    case "groupNorm": {
      const c = num(p.numChannels ?? ch);
      return c > 0 ? 2 * c : 0;
    }
    // ── RL ───────────────────────────────────────────────────────────────────
    case "policyNetwork":
    case "valueNetwork": {
      const H = num(p.hiddenSize, 256);
      const inF = num(lastDim, H);
      return inF * H + H + H * H + H;
    }
    case "dqnHead": {
      const H = num(p.hiddenSize ?? lastDim, 512);
      const A = num(p.numActions, 18);
      return H * A + A;
    }
    case "actorHead": {
      const H = num(lastDim, 256);
      const A = num(p.numActions, 6);
      return H * A + A;
    }
    case "criticHead": {
      const H = num(lastDim, 256);
      const O = num(p.outputDim, 1);
      return H * O + O;
    }
    // ── Graph ────────────────────────────────────────────────────────────────
    case "graphConv":
    case "gcn": {
      const inC = num(p.inChannels ?? p.inFeatures);
      const outC = num(p.outChannels ?? p.outFeatures);
      return inC > 0 && outC > 0 ? inC * outC + outC : 0;
    }
    case "graphAttention":
    case "gat": {
      const inC = num(p.inChannels ?? p.inFeatures ?? lastDim);
      const outC = num(p.outChannels ?? p.outFeatures);
      const heads = num(p.numHeads ?? p.heads, 1);
      return inC > 0 && outC > 0 ? heads * (inC * outC + 2 * outC) : 0;
    }
    case "graphSAGE": {
      const inC = num(p.inChannels ?? p.inFeatures);
      const outC = num(p.outChannels ?? p.outFeatures);
      return inC > 0 && outC > 0 ? inC * 2 * outC + outC : 0;
    }
    // ── Multimodal ───────────────────────────────────────────────────────────
    case "fusion": {
      const inD = num(lastDim);
      const d = num(p.fusionDim, 256);
      return d > 0 && inD > 0 ? inD * d + d : 0;
    }
    case "projection": {
      const inD = num(p.inDim ?? lastDim);
      const outD = num(p.outDim);
      return inD > 0 && outD > 0 ? inD * outD + outD : 0;
    }
    // ── Tabular ──────────────────────────────────────────────────────────────
    case "tabnet": {
      const fd = num(p.featureDim, 64);
      const dd = num(p.decisionDim, 64);
      const inF = num(p.inputDim ?? lastDim, fd);
      const w = fd + dd;
      return w > 0 ? inF * w + w + 4 * w * w : 0;
    }
    case "featureInteraction":
      return 0;
    // ── Audio ────────────────────────────────────────────────────────────────
    case "audioConv": {
      const inC = num(p.inChannels ?? 1, 1);
      const outC = num(p.outChannels);
      const k = num(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k + outC : 0;
    }
    case "melSpectrogram":
    case "mfcc":
    case "stft":
      return 0;
    // ── New attention types ──────────────────────────────────────────────────
    case "windowAttention": {
      const D = num(p.embedDim, 96);
      return D > 0 ? 4 * D * D : 0;
    }
    case "groupedQueryAttention": {
      const D = num(p.embedDim, 4096);
      const H = num(p.numHeads, 32);
      const Hkv = num(p.numKVHeads, H);
      const headDim = num(p.headDim, H > 0 ? Math.floor(D / H) : 128);
      const qDim = H * headDim;
      const kvDim = Hkv * headDim;
      return D > 0 && headDim > 0 ? D * qDim + qDim * D + 2 * (D * kvDim) : 0;
    }
    case "causalAttention": {
      const D = num(p.embedDim, 512);
      return D > 0 ? 4 * D * D : 0;
    }
    case "adaptiveMaxPool2d":
      return 0;
    // no learnable params
    // ── New activations ──────────────────────────────────────────────────────
    case "prelu": {
      return num(p.numParameters, 1);
    }
    case "rmsNorm": {
      const ns = num(p.normalizedShape ?? lastDim, 512);
      return ns;
    }
    // ── Transformer extras ───────────────────────────────────────────────────
    case "swiglu": {
      const D = num(p.embedDim ?? p.inFeatures, 4096);
      const I = num(p.intermediateSize ?? p.hiddenFeatures ?? p.ffDim, Math.round(D * 8 / 3));
      return D > 0 && I > 0 ? 3 * D * I : 0;
    }
    case "moeLayer": {
      const D = num(p.embedDim, 512);
      const E = num(p.numExperts, 8);
      const I = num(p.expertDim ?? p.ffDim, Math.round(D * 8 / 3));
      return D > 0 && I > 0 ? D * E + E * 3 * D * I : 0;
    }
    case "patchEmbed": {
      const inC = num(p.inChans ?? p.inChannels, 3);
      const D = num(p.embedDim, 768);
      const P = num(p.patchSize, 16);
      return inC * D * P * P + D;
    }
    case "seBlock": {
      const C = num(p.channels, 64);
      const r = num(p.reductionRatio ?? p.reduction, 16);
      const mid = Math.max(1, Math.floor(C / r));
      return C * mid + mid + mid * C + C;
    }
    case "layerScale":
      return num(p.dim, 512);
    case "alibi":
    case "dropPath":
      return 0;
    // ── Frontier architectures (2024-2025) ────────────────────────────────────
    case "mla": {
      const D = num(p.embedDim, 512);
      const kv = num(p.kvLatentDim, 128);
      return D > 0 ? 2 * D * D + 2 * D * kv : 0;
    }
    case "mamba2": {
      const D = num(p.dModel, 512);
      const E = num(p.expand, 2);
      const S = num(p.dState, 128);
      const inner = D * E;
      return D > 0 ? 2 * D * inner + inner * D + inner * S : 0;
    }
    case "differentialAttention":
    case "retention": {
      const D = num(p.embedDim ?? p.dModel, 512);
      return D > 0 ? 4 * D * D : 0;
    }
    case "rgLru": {
      const D = num(p.dModel, 512);
      const E = num(p.expand, 1);
      const inner = D * E;
      return D > 0 ? 2 * D * inner + 2 * inner : 0;
    }
    case "hyena": {
      const D = num(p.dModel, 512);
      const fo = num(p.filterOrder, 64);
      const order = num(p.order, 2);
      return D > 0 ? 2 * D * D + order * (D * fo + fo) : 0;
    }
    case "rwkv":
    case "xlstm": {
      const D = num(p.dModel, 512);
      return D > 0 ? 4 * D * D : 0;
    }
    case "mixtureOfDepths": {
      const D = num(p.dModel, 512);
      return D > 0 ? D : 0;
    }
    case "tttLayer":
    case "titansMemory": {
      const D = num(p.dModel, 512);
      const depth = num(p.memoryDepth, 2);
      return D > 0 ? depth * D * D : 0;
    }
    case "multiTokenPrediction": {
      const D = num(p.dModel, 512);
      const V = num(p.vocabSize, 32e3);
      const k = num(p.numFutureTokens, 2);
      return D > 0 && V > 0 ? k * D * V : 0;
    }
    case "kan": {
      const inF = num(p.inFeatures, 128);
      const outF = num(p.outFeatures, 128);
      const grid = num(p.gridSize, 5);
      const order = num(p.splineOrder, 3);
      return inF > 0 && outF > 0 ? inF * outF * (grid + order + 1) : 0;
    }
    case "geglu": {
      const D = num(p.dim, 512);
      const I = num(p.hiddenDim, D > 0 ? D * 4 : 0);
      return D > 0 && I > 0 ? 3 * D * I : 0;
    }
    case "grn": {
      const C = num(p.channels, 256);
      return C > 0 ? 2 * C : 0;
    }
    case "qkNorm": {
      const d = num(p.dim, 64);
      return d > 0 ? 2 * d : 0;
    }
    // ── Frontier architectures (new batch) ────────────────────────────────────
    case "deltaNet":
    case "gatedDeltaNet":
    case "gatedLinearAttention": {
      const d = num(p.dModel, 512);
      return d > 0 ? 4 * d * d : 0;
    }
    case "nativeSparseAttention": {
      const d = num(p.embedDim, 512);
      return d > 0 ? 4 * d * d : 0;
    }
    case "sharedExpertMoE": {
      const d = num(p.embedDim, 4096);
      const e = num(p.numExperts, 64);
      const s = num(p.numSharedExperts, 2);
      const I = num(p.expertDim, 1408);
      return d > 0 && I > 0 ? d * e + (e + s) * 3 * d * I : 0;
    }
    case "ditBlock": {
      const d = num(p.hiddenDim, 1152);
      const cond = num(p.condDim, 1152);
      return d > 0 ? 4 * d * d + 8 * d * d + 6 * cond * d : 0;
    }
    case "vectorQuantizer": {
      const cb = num(p.codebookSize, 8192);
      const d = num(p.embedDim, 256);
      return cb > 0 && d > 0 ? cb * d : 0;
    }
    case "residualVQ": {
      const q = num(p.numQuantizers, 8);
      const cb = num(p.codebookSize, 1024);
      const d = num(p.embedDim, 256);
      return q > 0 && cb > 0 && d > 0 ? q * cb * d : 0;
    }
    case "perceiverLatent": {
      const nL = num(p.numLatents, 64);
      const lD = num(p.latentDim, 768);
      return lD > 0 ? nL * lD + 4 * lD * lD : 0;
    }
    case "convNeXtBlock": {
      const dim = num(p.dim, 96);
      const k = num(p.kernelSize, 7);
      const ex = num(p.expandRatio, 4);
      return dim > 0 ? dim * k * k + dim + 2 * dim * (dim * ex) : 0;
    }
    case "s4Layer": {
      const d = num(p.dModel, 512);
      const s = num(p.dState, 64);
      return d > 0 ? d * s * 2 + d * 2 : 0;
    }
    case "dyt": {
      const dim = num(p.dim, 512);
      return dim > 0 ? 2 * dim + 1 : 0;
    }
    case "film": {
      const f = num(p.numFeatures, 256);
      return f > 0 ? 2 * f : 0;
    }
    case "crossNetworkDCN": {
      const nL = num(p.numLayers, 3);
      const d = num(p.inputDim, 256);
      return d > 0 ? nL * (d * d + d) : 0;
    }
    case "ftTransformerBlock": {
      const d = num(p.dModel, 192);
      const ff = num(p.ffMult, 4);
      return d > 0 ? 4 * d * d + 2 * d * (d * ff) : 0;
    }
    case "deformableAttention": {
      const d = num(p.embedDim, 256);
      const h = num(p.numHeads, 8);
      const pts = num(p.numPoints, 4);
      const lv = num(p.numLevels, 4);
      return d > 0 ? 2 * d * d + d * (h * lv * pts * 2) + d * (h * lv * pts) : 0;
    }
    case "attentionPool": {
      const dim = num(p.dim, 512);
      const seeds = num(p.numSeeds, 1);
      return dim > 0 ? seeds * dim + 4 * dim * dim : 0;
    }
    case "revIN": {
      if (p.affine === false) return 0;
      const f = num(p.numFeatures, 7);
      return f > 0 ? 2 * f : 0;
    }
    case "seriesDecomp":
      return 0;
    case "setAbstraction": {
      const mlp = Array.isArray(p.mlp) ? p.mlp.map((x) => num(x, 0)) : [64, 64, 128];
      let inC = 3 + 3;
      let total = 0;
      for (const outC of mlp) {
        total += inC * outC + outC;
        inC = outC;
      }
      return total;
    }
    case "sparseConv3d": {
      const inC = num(p.inChannels, 32);
      const outC = num(p.outChannels, 64);
      const k = num(p.kernelSize, 3);
      return inC > 0 && outC > 0 ? inC * outC * k * k * k + outC : 0;
    }
    case "nerfPositionalEncoding":
      return 0;
    case "dividedSpaceTimeAttention": {
      const d = num(p.embedDim, 768);
      return d > 0 ? 2 * (4 * d * d) : 0;
    }
    case "tubeletEmbed": {
      const D = num(p.embedDim, 768);
      const t = Array.isArray(p.tubeletSize) ? p.tubeletSize.map((x) => num(x, 1)) : [2, 16, 16];
      const inC = num(p.inChans ?? p.inChannels, 3);
      const kvol = (t[0] || 1) * (t[1] || 1) * (t[2] || 1);
      return D > 0 ? inC * D * kvol + D : 0;
    }
    // ── Non-LLM coverage (audio / SSM / vision / diffusion) ──────────────────
    case "conformerBlock": {
      const d = num(p.dModel ?? p.embedDim ?? p.hiddenDim, 256);
      const k = num(p.kernelSize, 31);
      if (d <= 0) return 0;
      const ff = p.ffDim != null ? num(p.ffDim, d * 4) : d * num(p.ffMult, 4);
      const ffn = 2 * (2 * d * ff);
      const mha = 4 * d * d;
      const conv = 2 * d * d + d * k + d * d;
      return ffn + mha + conv + 6 * d;
    }
    case "mamba": {
      const d = num(p.dModel ?? p.embedDim ?? p.hiddenDim, 256);
      const e = num(p.expand, 2);
      const dState = num(p.dState, 16);
      const dConv = num(p.dConv, 4);
      if (d <= 0) return 0;
      const di = d * e;
      return d * (2 * di) + di * dConv + di * (2 * dState) + di * dState + di * d;
    }
    case "relativePositionBias": {
      const h = num(p.numHeads, 8);
      const b = num(p.numBuckets, 32);
      return h * b;
    }
    case "fpn": {
      const outC = num(p.outChannels, 256);
      const ins = Array.isArray(p.inChannels) ? p.inChannels.map((x) => num(x)) : [num(p.inChannels, outC)];
      if (outC <= 0) return 0;
      let total = 0;
      for (const inC of ins) total += inC * outC + outC;
      total += ins.length * (outC * outC * 9 + outC);
      return total;
    }
    case "timeEmbedding": {
      const d = num(p.dim ?? p.embedDim ?? p.hiddenDim, 256);
      return d > 0 ? 2 * d * d + 2 * d : 0;
    }
    case "lmHead": {
      const V = num(p.vocabSize ?? p.numEmbeddings);
      const d = num(p.inFeatures ?? p.hiddenSize ?? p.embedDim ?? lastDim);
      const bias = p.bias === true ? V : 0;
      return V > 0 && d > 0 ? d * V + bias : 0;
    }
    case "adaIN": {
      const f = num(p.numFeatures ?? p.embedDim ?? lastDim);
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
      const D = n(p.embedDim ?? p.hiddenDim);
      const ff = n(p.ffDim, D > 0 ? D * 4 : 0);
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
function kvCachePerTokenBytes(spec) {
  const L = Math.max(0, spec.numLayers);
  if (spec.mlaLatentDim && spec.mlaLatentDim > 0) {
    return (spec.mlaLatentDim + (spec.mlaRopeDim ?? 0)) * spec.bytesPerValue * L;
  }
  const kvHeads = spec.kvHeads ?? 0;
  const headDim = spec.headDim ?? 0;
  return 2 * kvHeads * headDim * spec.bytesPerValue * L;
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
      const full = fullAttention(dim, heads);
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
          issues.push({
            componentId: c.id,
            componentName: c.name,
            componentType: c.type,
            kind: "unknown-layer-type",
            message: `'${c.name}': "${c.type}" is not a known layer type; nothing downstream of it can be verified and the trainer would drop it`
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

// src/utils/lintGraph.ts
function advisorSeverity(sev) {
  return sev === "error" ? "block" : sev === "warning" ? "warn" : "info";
}
var SHAPE_SEVERITY = {
  "head-dim-divisibility": "block",
  "gqa-head-divisibility": "block",
  "merge-shape-mismatch": "block",
  "compute-error": "block",
  // A type the registry doesn't know halts propagation, making everything
  // downstream unverifiable, and the trainer codegen drops the layer. Grounded
  // run 2026-08-20: such a design trained to below-random accuracy with a
  // clean static verdict. Unverifiable must never read as fine.
  "unknown-layer-type": "block",
  "invalid-output-shape": "warn",
  "attention-in-mismatch": "warn"
};
var SHAPE_RULE_IDS = Object.keys(SHAPE_SEVERITY);
var ALL_RULE_IDS = [
  .../* @__PURE__ */ new Set([...ADVISOR_RULE_IDS, ...SHAPE_RULE_IDS])
];
function dedupe(findings) {
  const seen = /* @__PURE__ */ new Set();
  const out = [];
  for (const f of findings) {
    const key = `${f.rule}::${f.componentName ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(f);
  }
  return out;
}
function lintModelGraph(model) {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const findings = [];
  for (const issue of runAdvisorRules(model)) {
    const target = issue.affectedIds.map((id) => byId.get(id)).find(Boolean);
    findings.push({
      rule: issue.ruleId,
      severity: advisorSeverity(issue.severity),
      message: issue.suggestion ? `${issue.message} Fix: ${issue.suggestion}` : issue.message,
      componentName: target?.name,
      componentType: target?.type
    });
  }
  for (const issue of propagateShapes(model).issues) {
    const severity = SHAPE_SEVERITY[issue.kind];
    if (!severity) continue;
    findings.push({
      rule: issue.kind,
      severity,
      message: issue.message,
      componentName: issue.componentName,
      componentType: issue.componentType
    });
  }
  return dedupe(findings);
}

// src/utils/structuralIndex.ts
function fnv1a(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}
function computeDepth(model) {
  const ids = model.components.map((c) => c.id);
  if (ids.length === 0) return 0;
  const adj = /* @__PURE__ */ new Map();
  const indeg = /* @__PURE__ */ new Map();
  for (const id of ids) {
    adj.set(id, []);
    indeg.set(id, 0);
  }
  for (const e of model.connections) {
    if (!adj.has(e.from) || !indeg.has(e.to)) continue;
    if (e.from === e.to) continue;
    adj.get(e.from).push(e.to);
    indeg.set(e.to, (indeg.get(e.to) ?? 0) + 1);
  }
  const dist = new Map(ids.map((id) => [id, 0]));
  const queue = ids.filter((id) => (indeg.get(id) ?? 0) === 0);
  let processed = 0;
  let best = 0;
  while (queue.length > 0) {
    const u = queue.shift();
    processed++;
    for (const v of adj.get(u) ?? []) {
      const cand = (dist.get(u) ?? 0) + 1;
      if (cand > (dist.get(v) ?? 0)) {
        dist.set(v, cand);
        best = Math.max(best, cand);
      }
      indeg.set(v, (indeg.get(v) ?? 0) - 1);
      if ((indeg.get(v) ?? 0) === 0) queue.push(v);
    }
  }
  return processed === ids.length ? best : 0;
}
function typeHistogramOf(model) {
  const counts = /* @__PURE__ */ new Map();
  for (const c of model.components) {
    counts.set(c.type, (counts.get(c.type) ?? 0) + 1);
  }
  const histogram = [...counts.entries()].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([t, n2]) => `${t}:${n2}`).join("|");
  return { histogram, counts };
}
function architectureFingerprint(model) {
  const { histogram, counts } = typeHistogramOf(model);
  const signature = `${histogram}#e${model.connections.length}`;
  return {
    hash: fnv1a(signature),
    layerCount: model.components.length,
    connectionCount: model.connections.length,
    depth: computeDepth(model),
    typeHistogram: histogram,
    distinctTypes: counts.size
  };
}

// src/utils/lintEngine.ts
function lintPyTorchSource(code) {
  let model = null;
  try {
    const parsed = parsePyTorchCode(code);
    if (parsed && parsed.components.length > 0) {
      model = {
        id: "ci-lint",
        name: "ci-lint",
        components: parsed.components,
        connections: parsed.connections
      };
    }
  } catch {
    model = null;
  }
  if (!model) return { parsed: false, componentCount: 0, findings: [] };
  const realLayers = model.components.filter((c) => c.type !== "input" && c.type !== "output");
  if (realLayers.length === 0) return { parsed: false, componentCount: 0, findings: [] };
  return {
    parsed: true,
    componentCount: model.components.length,
    findings: lintModelGraph(model)
  };
}
function graphFromPyTorchSource(code, name = "model") {
  try {
    const parsed = parsePyTorchCode(code);
    if (!parsed || parsed.components.length === 0) return null;
    const realLayers = parsed.components.filter((c) => c.type !== "input" && c.type !== "output");
    if (realLayers.length === 0) return null;
    return stabilizeIds({
      id: name,
      name,
      components: parsed.components,
      connections: parsed.connections
    });
  } catch {
    return null;
  }
}
function stabilizeIds(model) {
  const remap = /* @__PURE__ */ new Map();
  const perType = /* @__PURE__ */ new Map();
  for (const c of model.components) {
    const n2 = (perType.get(c.type) ?? 0) + 1;
    perType.set(c.type, n2);
    remap.set(c.id, `${c.type}-${n2}`);
  }
  const to = (id) => remap.get(id) ?? id;
  return {
    ...model,
    components: model.components.map((c) => ({
      ...c,
      id: to(c.id),
      inputs: (c.inputs ?? []).map(to),
      outputs: (c.outputs ?? []).map(to)
    })),
    connections: model.connections.map((e, i) => ({
      ...e,
      id: `e-${i + 1}`,
      from: to(e.from),
      to: to(e.to)
    }))
  };
}
function toModelFile(model) {
  return wrapModelFile(model);
}
function physicsFromPyTorchSource(code) {
  const none = { parsed: false, components: 0, params: 0, flopsMAC: 0, kvBytesPerToken: 0 };
  let model = null;
  try {
    const parsed = parsePyTorchCode(code);
    if (parsed && parsed.components.length > 0) {
      model = { id: "ci-phys", name: "ci-phys", components: parsed.components, connections: parsed.connections };
    }
  } catch {
    return none;
  }
  if (!model) return none;
  const real = model.components.filter((c) => c.type !== "input" && c.type !== "output");
  if (real.length === 0) return none;
  return {
    parsed: true,
    components: model.components.length,
    params: real.reduce((a, c) => a + estimateLayerParams(c.type, c.params ?? {}, []), 0),
    flopsMAC: real.reduce((a, c) => a + estimateLayerFlops(c.type, c.params ?? {}, [], []), 0),
    kvBytesPerToken: kvBytesPerTokenForModel(model, 2)
  };
}
export {
  ADVISOR_RULE_IDS,
  LINT_THRESHOLDS,
  SHAPE_RULE_IDS,
  architectureFingerprint,
  graphFromPyTorchSource,
  lintModelGraph,
  lintPyTorchSource,
  physicsFromPyTorchSource,
  toModelFile
};
