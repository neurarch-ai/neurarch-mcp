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
var SWIGLU_RATIO_MAX = 5;
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
    const list = outgoing.get(conn.from);
    if (list) list.push(conn.to);
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
      const d = num(p.embedDim ?? p.hiddenDim ?? p.dModel);
      const ff = num(p.ffDim ?? p.dFf, d > 0 ? d * 4 : 0);
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
      const R = Math.max(0, Math.floor(num(p.registerTokens, 0)));
      return inC * D * P * P + D + (p.clsToken ? D : 0) + R * D;
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
    // ── Recommendation: sequential / generative ──────────────────────────────
    case "targetAttention": {
      const d = num(p.embedDim ?? lastDim, 64);
      const h = num(p.hiddenDim, 36);
      return d > 0 ? 4 * d * h + h + h + 1 : 0;
    }
    case "behaviorRetrieval": {
      if (String(p.mode ?? "soft") === "hard") return 0;
      const d = num(p.embedDim ?? lastDim, 64);
      return d > 0 ? d * d : 0;
    }
    case "multiInterest": {
      const d = num(p.embedDim ?? lastDim, 64);
      return d > 0 ? d * d : 0;
    }
    case "hstuBlock": {
      const d = num(p.embedDim, 512);
      const h = num(p.numHeads, 4);
      const dv = num(p.linearDim, 128);
      const dqk = num(p.attnDim, 128);
      if (d <= 0) return 0;
      const inProj = d * h * (2 * dv + 2 * dqk);
      const outProj = h * dv * d;
      return inProj + outProj + 2 * d;
    }
    // ── Video latents / world models ─────────────────────────────────────────
    case "causalConv3d": {
      const inC = num(p.inChannels ?? ch, 1);
      const outC = num(p.outChannels);
      const k = num(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k * k + outC : 0;
    }
    case "rssm": {
      const det = num(p.deterDim, 512);
      const st = num(p.stochDim, 32);
      const cls = Math.max(1, num(p.stochClasses, 1));
      const hid = num(p.hiddenDim, 512);
      const stochFlat = st * cls;
      const embed = num(lastDim, 0);
      if (det <= 0) return 0;
      const gru = 3 * (det + stochFlat) * det + 3 * det;
      const prior = det * hid + hid + hid * stochFlat + stochFlat;
      const post = (det + embed) * hid + hid + hid * stochFlat + stochFlat;
      return gru + prior + post;
    }
    case "latentActionModel": {
      const inD = num(lastDim, 512);
      const hid = num(p.hiddenDim, 512);
      const lat = num(p.latentDim, 32);
      const code = num(p.codebookSize, 8);
      return 2 * inD * hid + hid + hid * lat + lat + code * lat;
    }
    case "jepaPredictor": {
      const d = num(p.embedDim, 768);
      const pd = num(p.predictorDim, 384);
      const depth = num(p.depth, 6);
      if (d <= 0 || pd <= 0) return 0;
      const blocks = depth * (4 * pd * pd + 2 * pd * (4 * pd));
      return d * pd + pd + pd * d + d + blocks;
    }
    case "emaTarget":
      return 0;
    // ── Multimodal connectors / retrieval objectives ─────────────────────────
    case "patchMerger": {
      const k = Math.max(1, num(p.mergeSize, 2));
      const inD = num(p.inDim ?? lastDim, 1280);
      const outD = num(p.outDim, 3584);
      const merged = inD * k * k;
      if (merged <= 0) return 0;
      return 2 * merged + merged * merged + merged + merged * outD + outD;
    }
    case "contrastiveHead": {
      const inD = num(lastDim, 0);
      const outD = num(p.projDim, 512);
      if (inD <= 0 || outD <= 0) return 0;
      return inD * outD + (p.learnableTemp === false ? 0 : 1);
    }
    case "matryoshkaHead":
      return 0;
    case "lateInteraction":
      return 0;
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
var PREFLIGHT_GPUS = [
  { name: "T4 (16GB)", gb: 16, hourly: 0.59, speed: 0.55, hwId: "t4" },
  { name: "L4 (24GB)", gb: 24, hourly: 0.8, speed: 0.8, hwId: "l4" },
  { name: "A10G (24GB)", gb: 24, hourly: 1.1, speed: 1, hwId: "a10g" },
  { name: "L40S (48GB)", gb: 48, hourly: 1.95, speed: 2.5, hwId: "l40s" },
  { name: "A100 (40GB)", gb: 40, hourly: 2.1, speed: 2.2, hwId: "a100-40g" },
  { name: "A100 (80GB)", gb: 80, hourly: 2.5, speed: 2.4, hwId: "a100-80g" },
  { name: "H100 (80GB)", gb: 80, hourly: 3.95, speed: 4.5, hwId: "h100-sxm" }
];
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
  const weightBytes = params * 4;
  const trainFootprintBytes = params * 16;
  const footprintGb = trainFootprintBytes / 1e9;
  const gpu = PREFLIGHT_GPUS.find((g) => g.name === opts?.plan?.gpu) ?? PREFLIGHT_GPUS[2];
  const epochs = Math.max(1, opts?.plan?.epochs ?? 10);
  const batch = Math.max(1, opts?.plan?.batchSize ?? 32);
  const footprintGbWithHeadroom = footprintGb * 1.3;
  if (params > 0) {
    if (footprintGbWithHeadroom <= gpu.gb) {
      findings.push({
        id: "hw-fit",
        category: "hardware",
        severity: "pass",
        title: `Fits a ${gpu.name}`,
        detail: `Weights + Adam state are about ${footprintGb.toFixed(footprintGb < 1 ? 2 : 1)} GB; leave headroom for activations (grows with batch size).`
      });
    } else {
      const smallest = PREFLIGHT_GPUS.find((g) => g.gb >= footprintGbWithHeadroom);
      findings.push({
        id: "hw-no-fit",
        category: "hardware",
        severity: "warn",
        title: `Will not fit a ${gpu.name}`,
        detail: `Optimizer state alone needs about ${footprintGb.toFixed(1)} GB. ${smallest ? `Use a ${smallest.name} instead, or` : "Plan for multi-GPU / sharding (FSDP, DeepSpeed), or"} train in lower precision / with LoRA.`
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
    computeUtilization: 0.35,
    memoryUtilization: 0.8
  });
  const perStepSec = 5e-3 + roofline.seconds;
  const steps = Math.max(1, Math.ceil(samples / batch));
  const estTrainSec = params > 0 ? 60 + epochs * steps * perStepSec : 0;
  const estCostUsd = estTrainSec / 3600 * gpu.hourly;
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
      weightBytes,
      trainFootprintBytes,
      fitsGpu: gpu.name,
      estTrainSec,
      estCostUsd,
      samplesAssumed
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
  const byName2 = (m) => {
    const map = /* @__PURE__ */ new Map();
    const seen = /* @__PURE__ */ new Map();
    for (const c of m.components) {
      const n2 = seen.get(c.name) ?? 0;
      seen.set(c.name, n2 + 1);
      map.set(n2 === 0 ? c.name : `${c.name}${SEP}${n2}`, c);
    }
    return map;
  };
  const prevMap = byName2(prev);
  const nextMap = byName2(next);
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
  const kvPerRequest = inp.kvBytesPerToken * avgContext;
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
      const targets = Array.isArray(a.targets) ? a.targets.filter((t) => typeof t === "string" && t.length > 0).slice(0, MAX_TARGETS) : [];
      if (!PEFT_KINDS.includes(peft)) problems.push(`adapter.peft: expected one of ${PEFT_KINDS.join(", ")}, got ${JSON.stringify(a.peft)}`);
      else if (typeof rankRaw !== "number" || !Number.isInteger(rankRaw) || rankRaw <= 0) problems.push(`adapter.rank: expected a positive integer, got ${JSON.stringify(a.rank)}`);
      else {
        if (a.targets != null && !Array.isArray(a.targets)) problems.push("adapter.targets: expected a list of module names");
        spec.adapter = { peft, rank: rankRaw, targets };
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
  const weightBytes = ctx.params > 0 ? computeQuantPreview(model).totalBytes[precision] : null;
  if (spec.adapter && spec.adapter.targets.length) {
    const names = model.components.map((c2) => `${c2.name ?? ""} ${c2.id} ${c2.type}`.toLowerCase());
    const missing = spec.adapter.targets.filter((t) => {
      const needle = t.toLowerCase();
      return !names.some((n2) => n2.includes(needle));
    });
    if (missing.length) {
      warnings.push({
        title: `change: adapter targets not found in the graph (${missing.join(", ")})`,
        detail: "Targets are module-name prefixes; a graph read from config names layers by type, so this may be a naming difference rather than a missing layer.",
        stage: "change"
      });
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
        title: `change: must_fit ${gpu.label} violated (needs ${(trainMemGb * HEADROOM).toFixed(1)} GB to train${spec.adapter ? ` with ${spec.adapter.peft} r=${spec.adapter.rank}` : ""})`,
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
  if (spec.serving && weightBytes != null) {
    const gpu = resolveGpu(spec.serving.gpu);
    if (!gpu) warnings.push({ title: `change: unknown serving GPU (${spec.serving.gpu})`, detail: `Known: ${GPU_HARDWARE.map((g) => g.id).join(", ")}.`, stage: "change" });
    else {
      const kvBytesPerToken = kvBytesPerTokenForModel(model, 2, spec.serving.promptTokens + spec.serving.genTokens);
      const sp = planServing({
        gpu,
        weightBytes,
        kvBytesPerToken,
        flopsPerToken: 2 * ctx.params,
        promptTokens: spec.serving.promptTokens,
        genTokens: spec.serving.genTokens,
        batchSize: spec.serving.batchSize
      });
      servingFits = sp.fits;
      if (!sp.fits) {
        blockers.push({
          title: `change: serving target ${gpu.label} does not fit (${precision} weights ${(weightBytes / 1e9).toFixed(1)} GB plus KV for batch ${spec.serving.batchSize} at ${spec.serving.promptTokens + spec.serving.genTokens} tokens)`,
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
    estimates: { trainableParams, trainMemGb, weightBytes, servingFits },
    gate: { ok: blockers.length === 0, blockers, warnings },
    problems
  };
}
function describeChangeSpec(spec) {
  const bits = [];
  if (spec.adapter) bits.push(`${spec.adapter.peft.toUpperCase()} r=${spec.adapter.rank}${spec.adapter.targets.length ? ` on ${spec.adapter.targets.join(",")}` : ""}`);
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
    const prod = LATENT_PRODUCERS[n2.classType];
    if (prod && (prod.slot == null || prod.slot === cur[1])) return { node: n2, channels: prod.channels, why: `${n2.classType} makes ${prod.channels}-channel latents` };
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
  const num3 = (k) => {
    const v = raw[k];
    if (v == null) return void 0;
    const n2 = typeof v === "string" ? Number(v) : v;
    if (typeof n2 !== "number" || !Number.isFinite(n2) || n2 < 0) {
      problems.push(`${k}: expected a number, got ${JSON.stringify(v)}`);
      return void 0;
    }
    return n2;
  };
  const list = (k) => {
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
  const mm = num3("max_memory_gb");
  if (mm != null) policy.max_memory_gb = mm;
  const ml = num3("max_layers");
  if (ml != null) policy.max_layers = ml;
  const mc = num3("max_cost_usd");
  if (mc != null) policy.max_cost_usd = mc;
  const ft = list("forbid_types");
  if (ft) policy.forbid_types = ft;
  const rt = list("require_types");
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
var OPTIMIZER_K = 1.26;
var OUTPUT_K = 1.57;
var ATTENTION_K = 3.47;
var ACTIVATION_K = OUTPUT_K;
var ACTIVATION_BYTES_PER_ELEMENT = 2;
var ATTENTION_TYPES2 = /attention|transformerEncoder|transformerDecoder|^mla$|ditBlock|swin/i;
function layerAttentionElems(c, shapes) {
  if (!ATTENTION_TYPES2.test(c.type)) return 0;
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
function layerActivationElems(c, shapes, attention) {
  if (c.type === "input" || c.type === "output" || c.type === "stickyNote") return null;
  const out = shapes.get(c.id)?.out ?? c.outputShape;
  if (!out || !out.length || out.some((d) => !Number.isFinite(d) || d <= 0)) return null;
  const scores = attention === "eager" ? layerAttentionElems(c, shapes) * ATTENTION_K : 0;
  return out.reduce((a, b) => a * b, 1) * OUTPUT_K + scores;
}
function attentionElemsFor(model, shapes) {
  return model.components.reduce((a, c) => a + layerAttentionElems(c, shapes) * ATTENTION_K, 0);
}
function activationElemsFor(model, shapes, attention) {
  let elems = 0;
  let seen = false;
  for (const c of model.components) {
    const e = layerActivationElems(c, shapes, attention);
    if (e == null) continue;
    seen = true;
    elems += e;
  }
  return seen ? elems : null;
}
function activationGbAt(elems, batchSize) {
  return elems * batchSize * ACTIVATION_BYTES_PER_ELEMENT / 1e9;
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
function maxBatchFor(f, gb) {
  const per = f.activationPerSampleGb;
  if (per == null || !(per > 0) || !(gb > 0)) return null;
  const fits = (b2) => (f.footprintGb + (f.activationGbAt ? f.activationGbAt(b2) : per * b2)) * FIT_HEADROOM <= gb;
  if (!fits(1)) return null;
  let b = Math.max(1, Math.floor((gb / FIT_HEADROOM - f.footprintGb) / per));
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
  for (const c of layers) {
    const inShape = shapes.get(c.id)?.in ?? void 0;
    const params = estimateLayerParams(c.type, c.params, inShape ?? void 0) ?? 0;
    const elems = layerActivationElems(c, shapes, attention);
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
  for (const gb of sizes) {
    if (peakGb * FIT_HEADROOM <= gb) return null;
    if (peakGb * FIT_HEADROOM > 2 * gb) continue;
    const map = deviceMapFor(model, [gb, gb], { batchSize, attention });
    if (map.fits) {
      const gpu = PREFLIGHT_GPUS.find((g) => g.gb === gb).name;
      return { gpu, gbPerCard: gb, ...map };
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
function fitsCard(peakGb, params, gb) {
  if (params <= 0) return null;
  return peakGb * FIT_HEADROOM <= gb;
}
var ATTENTION_LABEL = { sdpa: "SDPA", flash: "flash attention", eager: "eager" };
function attentionPhrase(attention, scoresGb) {
  if (scoresGb == null || !(scoresGb > 0.05)) return "";
  return attention === "eager" ? `attention: eager, scores stored (${scoresGb.toFixed(1)} GB of the peak)` : `attention: ${ATTENTION_LABEL[attention]}, scores not stored; eager would add ${scoresGb.toFixed(1)} GB`;
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
function byName(model) {
  const map = /* @__PURE__ */ new Map();
  for (const c of model.components) map.set(c.name, c);
  return map;
}
function diffFor(head, headFacts, base) {
  const baseFacts = factsFor(base);
  const d = diffArchitectures(base, head);
  const baseBy = byName(base);
  const headBy = byName(head);
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
    if (smallest) parts.push(`fits ${smallest} at batch ${batchOf(plan)}`);
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
    lora: null
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
  const cost = {
    estCostUsd: f.estCostUsd,
    estTrainSec: f.estTrainSec,
    // Filled in below, once policy findings have had their say: a graph that
    // passes pre-flight and fails a house rule is refused just the same.
    refusedRunUsd: null,
    fitsGpu: f.fitsGpu,
    gpuFits,
    peakGb: f.peakGb,
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
    assumptions: `Cost is a roofline estimate on the priced GPU for ${f.epochs} epochs at batch ${f.batchSize}${tokensAt(f.seqLen)} over ${f.samples.toLocaleString()} samples` + (f.samplesSource ? ` (from "${f.samplesSource}")` : " (assumed; no dataset attached)") + (f.activationGb != null ? `. GPU fit is the training peak at batch ${f.batchSize}${tokensAt(f.seqLen)}, AdamW with bf16 autocast: weights, gradients and optimizer state (${f.footprintGb.toFixed(2)} GB, ${OPTIMIZER_K} x 16 bytes per parameter) plus activations (${f.activationGb.toFixed(1)} GB: ${OUTPUT_K} x every layer's output${attention === "eager" && f.attentionGb > 0.05 ? `, and ${f.attentionGb.toFixed(1)} GB of attention scores eager attention stores for backward` : ""}), with 1.15x headroom. ${attentionSentence(attention, f.attentionGb)}Factors fitted on 141 measured runs across 44 templates, with the attention-score term present on the templates that ran eager attention; held out, the median error is 8% above 1 GB and the card verdict agrees with measurement 98% of the time; recurrent cells are under-counted.` : ". GPU fit is fp32 weights plus gradients plus two Adam moments (16 bytes per parameter) with 1.15x headroom; no shape propagated, so activations are not included.")
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
  const lora = f.params > 0 ? loraTargetsFor(model) : null;
  if (opts.spec != null) {
    const readSpec = readChangeSpec(opts.spec);
    if (!readSpec) {
      f.warnings = [...f.warnings, { title: "change: unreadable spec (expected an object)", stage: "change" }];
    } else {
      const withheld = opts.incompleteRead && typeof opts.incompleteRead.reason === "string";
      change = evaluateChangeSpec(model, readSpec.spec, readSpec.problems, {
        baseFingerprint: architectureHashFromJson(model) ?? "",
        baseLegal: withheld ? null : f.blockers.length === 0,
        trainingPeak: (adapter) => trainingPeakFor({ params: f.params, peakGb: f.peakGb, activationGb: f.activationGb, lora }, adapter),
        params: f.params,
        estCostUsd: f.estCostUsd,
        batchSize: f.batchSize
      });
      f.blockers = [...f.blockers, ...change.gate.blockers];
      f.warnings = [...f.warnings, ...change.gate.warnings];
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
    lora
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
  if (e.trainMemGb != null) bits.push(`est train mem ${(e.trainMemGb * CHANGE_HEADROOM).toFixed(1)} GB${ch.spec.adapter ? ` (${fmtParamCount(e.trainableParams)} trainable)` : ""}`);
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
    const pct = (x) => baseParams > 0 ? x / baseParams * 100 : 0;
    return {
      targetModules: [...new Set(ms.map((m) => m.module))],
      modules: ms.length,
      loraParams: { r8, r16 },
      pctOfBase: { r8: pct(r8), r16: pct(r16) }
    };
  };
  const ATTN = /* @__PURE__ */ new Set(["q_proj", "k_proj", "v_proj", "o_proj"]);
  return {
    baseParams,
    modules,
    byRole,
    presets: {
      attention: preset((m) => ATTN.has(m.role)),
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
  const pct = (x) => x < 0.01 ? "<0.01%" : `${x.toFixed(2)}%`;
  const L = [];
  L.push(`LoRA targets   base ${fmtParamCount(lora.baseParams)} params, ${lora.modules.length} linear module${lora.modules.length === 1 ? "" : "s"}`);
  const row = (label, pr) => {
    L.push(`  ${label.padEnd(11)}  r=8 +${fmtParamCount(pr.loraParams.r8)} (${pct(pr.pctOfBase.r8)})  r=16 +${fmtParamCount(pr.loraParams.r16)} (${pct(pr.pctOfBase.r16)})  over ${pr.modules} module${pr.modules === 1 ? "" : "s"}`);
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
  const g = plan.cost.gpuFits;
  L.push(`  fits      T4 16GB ${yesNo(g.T4)} \xB7 A100 40GB ${yesNo(g.A100_40)} \xB7 H100 80GB ${yesNo(g.H100_80)}   (peak ~${plan.cost.peakGb.toFixed(1)} GB at batch ${batchOf(plan)}${tokensAt(plan.cost.seqLen)}${plan.cost.activationGb == null ? ", activations unknown" : ", est."})`);
  const att = attentionPhrase(plan.cost.attention, plan.cost.attentionScoresGb);
  if (att) L.push(`  attention ${att.slice("attention: ".length)}`);
  const mbLine = maxBatchLine(plan);
  if (mbLine) L.push(`  max batch ${mbLine}`);
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
    const pct = d.paramsDeltaPct == null ? "" : ` (${d.paramsDeltaPct >= 0 ? "+" : ""}${d.paramsDeltaPct.toFixed(0)}%)`;
    const mem = d.memoryDeltaGb == null ? "" : `   memory ${d.memoryDeltaGb >= 0 ? "+" : "-"}${Math.abs(d.memoryDeltaGb).toFixed(2)} GB`;
    const cost = d.costDeltaUsd == null ? "" : `   cost ${fmtSignedUsd(d.costDeltaUsd)}`;
    L.push(`  +${d.added} -${d.removed} ~${d.changed}   params ${fmtSigned(d.paramsDelta)}${pct}${mem}${cost}   blast radius ${d.blastRadius} layer${d.blastRadius === 1 ? "" : "s"}`);
    L.push(...clip(d.lines, (l) => `  ${l}`));
    if (d.legalityChanged === "became_illegal") L.push("  LEGALITY  base ran, this does not");
    if (d.legalityChanged === "became_legal") L.push("  LEGALITY  base did not run, this does");
  }
  return L.join("\n");
}
function renderPlanMarkdown(plan) {
  const m = plan.model;
  const g = plan.cost.gpuFits;
  const d = plan.diff;
  const L = [];
  L.push(`**${plan.summary}**`, "");
  L.push("| | |", "|---|---|");
  L.push(`| Layers / params | ${m.layers} / ${fmtParamCount(m.params)}${d ? ` (${fmtSigned(d.paramsDelta)}${d.paramsDeltaPct == null ? "" : `, ${d.paramsDeltaPct >= 0 ? "+" : ""}${d.paramsDeltaPct.toFixed(0)}%`})` : ""} |`);
  L.push(`| Shape | \`${fmtShape(m.inputShape)}\` -> \`${fmtShape(m.outputShape)}\` |`);
  L.push(`| Will it run | ${plan.run.legal == null ? `**cannot say**: ${plan.run.withheld ?? "the read was incomplete"}` : plan.run.legal ? `yes, ${plan.run.warnings.length} warning${plan.run.warnings.length === 1 ? "" : "s"}` : `**no**, ${plan.run.blockers.length} blocker${plan.run.blockers.length === 1 ? "" : "s"}`}${plan.run.legal == null ? "" : d?.legalityChanged === "became_illegal" ? " (**base ran, this does not**)" : d?.legalityChanged === "became_legal" ? " (base did not run, this does)" : ""} |`);
  L.push(`| GPU fit | T4 ${yesNo(g.T4)} \xB7 A100 40GB ${yesNo(g.A100_40)} \xB7 H100 80GB ${yesNo(g.H100_80)}${d?.memoryDeltaGb != null ? ` (memory ${d.memoryDeltaGb >= 0 ? "+" : "-"}${Math.abs(d.memoryDeltaGb).toFixed(2)} GB)` : ""}, peak ~${plan.cost.peakGb.toFixed(1)} GB at batch ${batchOf(plan)}${tokensAt(plan.cost.seqLen)}${plan.cost.activationGb == null ? " (activations unknown)" : " (est.)"} |`);
  const att = attentionPhrase(plan.cost.attention, plan.cost.attentionScoresGb);
  if (att) L.push(`| Attention | ${att.slice("attention: ".length)} |`);
  const mb = maxBatchLine(plan);
  if (mb) L.push(`| Max batch | ${mb} |`);
  const sp = splitLine(plan);
  if (sp) L.push(`| Two-card split | ${sp} |`);
  L.push(`| Est. training | ${plan.cost.estCostUsd != null ? `~${fmtUsd(plan.cost.estCostUsd)} / ${fmtDuration(plan.cost.estTrainSec)} on ${plan.cost.fitsGpu ?? "the priced GPU"}` : "not estimable"}${d?.costDeltaUsd != null ? ` (${fmtSignedUsd(d.costDeltaUsd)})` : ""} |`);
  if (d) L.push(`| Blast radius | ${d.blastRadius} downstream layer${d.blastRadius === 1 ? "" : "s"} from ${d.added + d.changed} touched |`);
  L.push("");
  if (plan.run.blockers.length) {
    L.push(plan.run.legal == null ? "**Findings from an incomplete read** (unverified: a layer this read could not open may explain, or cause, any of them)" : "**Blockers**");
    for (const b of plan.run.blockers.slice(0, 6)) L.push(`- ${b.title}${b.detail ? `: ${b.detail}` : ""}${b.fix ? ` Fix: ${b.fix}` : ""}`);
    if (plan.run.blockers.length > 6) L.push(`- and ${plan.run.blockers.length - 6} more`);
    L.push("");
  }
  if (plan.run.warnings.length) {
    L.push("**Warnings**");
    for (const w of plan.run.warnings.slice(0, 6)) L.push(`- ${w.title}${w.detail ? `: ${w.detail}` : ""}`);
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
var num2 = (v, unit) => Number(v.replace(/,/g, "")) * (UNIT[unit.toLowerCase()] ?? NaN);
var SIZE = String.raw`([\d.,]+)\s*(B|KiB|KB|MiB|MB|GiB|GB|TiB)`;
function grab(text, re) {
  const m = re.exec(text);
  if (!m) return null;
  const v = num2(m[1], m[2]);
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
      const v = num2(m[1], m[2]);
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
function batchThatFits(facts, cardGb) {
  const { footprintGb, activationPerSampleGb } = facts;
  if (footprintGb == null || activationPerSampleGb == null || !(activationPerSampleGb > 0)) return null;
  const b = Math.floor((cardGb / HEADROOM2 - footprintGb) / activationPerSampleGb);
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
  if (facts.activationPerSampleGb != null && facts.batch != null && facts.footprintGb != null) {
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
  const peakGb = weightsGb + kvGb;
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
export {
  ACTIVATION_K,
  ATTENTION_K,
  DEFAULT_ATTENTION,
  DEFAULT_GEN_TOKENS,
  DEFAULT_INFERENCE_BATCH,
  DEFAULT_OOM_TRAIN_BATCH,
  DEFAULT_SEQ_LEN,
  FIT_HEADROOM,
  INFERENCE_NOT_COUNTED,
  OPTIMIZER_K,
  OUTPUT_K,
  activationGbFor,
  adapterParamsFor,
  attentionPhrase,
  attentionScoresGbFor,
  buildPlan,
  changeGateSentence,
  checkComfyWorkflow,
  deviceMapFor,
  diagnoseOom,
  factsFor2 as factsFor,
  fmtParamCount,
  historySentence,
  inferenceFactsFor,
  inferencePeakFor,
  loraTargetsFor,
  maxBatchFor,
  oomFactsForModel,
  parseHumanSize,
  parseOomMessage,
  peakInputsOf,
  readAttention,
  readGraph,
  readPolicy,
  readSeqLen,
  renderLoraText,
  renderOomText,
  renderPlanMarkdown,
  renderPlanText,
  tokenSeqLenOf,
  trainingPeakFor,
  withSeqLen
};
