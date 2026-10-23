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
      const first2 = m[1].split(",")[0];
      const rhs = m[2];
      if (first2.length !== inputShape.length) {
        throw new Error(`einsum: first operand '${first2}' expects rank ${first2.length}, input has rank ${inputShape.length} (equation "${eq}")`);
      }
      const bound = /* @__PURE__ */ new Map();
      [...first2].forEach((ch, i) => bound.set(ch, inputShape[i]));
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
    const first2 = parentShapes[0];
    for (let i = 1; i < parentShapes.length; i++) {
      if (!shapesEqual(first2, parentShapes[i])) {
        return {
          componentId: c.id,
          componentName: c.name,
          componentType: c.type,
          kind: "merge-shape-mismatch",
          message: `${c.type} '${c.name}': elementwise inputs must match, got [${first2.join(",")}] vs [${parentShapes[i].join(",")}]`,
          expected: first2,
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
  const first2 = widths[0];
  const mismatch = widths.find((w) => w !== first2);
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
function gqaParams(p) {
  const D = num(p.embedDim, 4096);
  const H = num(p.numHeads, 32);
  const Hkv = num(p.numKVHeads, H);
  const headDim = num(p.headDim, H > 0 ? Math.floor(D / H) : 128);
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
      if (type === "localAttention" && (p.numKVHeads != null || p.headDim != null)) {
        return gqaParams(p);
      }
      const d = num(p.embedDim ?? p.hiddenDim);
      return d > 0 ? 4 * d * d + 4 * d : 0;
    }
    // ── LLM ─────────────────────────────────────────────────────────────────
    case "feedForward": {
      const d = num(p.hiddenDim ?? p.embedDim);
      const ff = num(p.ffDim, d > 0 ? d * 4 : 0);
      const b = p.bias === false ? 0 : ff + d;
      return d > 0 && ff > 0 ? d * ff + ff * d + b : 0;
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
    case "groupedQueryAttention":
      return gqaParams(p);
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
      const m = p.gatedExperts === false ? 2 : 3;
      return D > 0 && I > 0 ? D * E + E * m * D * I : 0;
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
      const H = num(p.numHeads);
      const nope = num(p.qkNopeHeadDim);
      const vd = num(p.vHeadDim);
      if (D > 0 && H > 0 && nope > 0 && vd > 0) {
        const rope = num(p.ropeHeadDim);
        const qk = nope + rope;
        const qLora = num(p.qLoraRank);
        const q = qLora > 0 ? D * qLora + qLora + qLora * H * qk : D * H * qk;
        return q + D * (kv + rope) + kv + kv * H * (nope + vd) + H * vd * D;
      }
      return D > 0 ? 2 * D * D + 2 * D * kv : 0;
    }
    case "mamba2": {
      const D = num(p.dModel, 512);
      const E = num(p.expand, 2);
      const S = num(p.dState, 128);
      const H = num(p.numHeads);
      if (D > 0 && H > 0) {
        const di = H * num(p.headDim, 64);
        const G = num(p.nGroups, 1);
        const K = num(p.dConv, 4);
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
      const Hk = num(p.numKeyHeads);
      if (type === "gatedDeltaNet" && d > 0 && Hk > 0) {
        const Hv = num(p.numHeads, Hk);
        const dv = num(p.headDim, 128);
        const dk = num(p.keyHeadDim, dv);
        const K = num(p.convKernel, 4);
        const keyDim = Hk * dk;
        const valDim = Hv * dv;
        return d * (2 * keyDim + 2 * valDim) + d * 2 * Hv + (2 * keyDim + valDim) * K + 2 * Hv + dv + valDim * d;
      }
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
      const Is = num(p.sharedExpertDim, I);
      const m = p.gatedExperts === false ? 2 : 3;
      const lat = num(p.latentDim);
      if (d > 0 && I > 0 && lat > 0) return d * e + 2 * d * lat + e * m * lat * I + s * m * d * Is;
      return d > 0 && I > 0 ? d * e + e * m * d * I + s * m * d * Is : 0;
    }
    case "ditBlock": {
      const d = num(p.hiddenDim, 1152);
      const cond2 = num(p.condDim, 1152);
      return d > 0 ? 4 * d * d + 8 * d * d + 6 * cond2 * d : 0;
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
      const pts2 = num(p.numPoints, 4);
      const lv = num(p.numLevels, 4);
      return d > 0 ? 2 * d * d + d * (h * lv * pts2 * 2) + d * (h * lv * pts2) : 0;
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
      const dtRank = num(p.dtRank);
      if (dtRank > 0) {
        const pb = p.projBias === true;
        return d * 2 * di + (pb ? 2 * di : 0) + di * dConv + (p.convBias === false ? 0 : di) + di * (dtRank + 2 * dState) + dtRank * di + di + di * dState + di + di * d + (pb ? d : 0);
      }
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
function fmtParams(n2) {
  if (!isFinite(n2) || isNaN(n2)) return "?";
  if (n2 === 0) return ", ";
  if (n2 >= 1e9) return `${(n2 / 1e9).toFixed(2)}B`;
  if (n2 >= 1e6) return `${(n2 / 1e6).toFixed(2)}M`;
  if (n2 >= 1e3) return `${(n2 / 1e3).toFixed(1)}K`;
  return n2.toString();
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
var num2 = (v, d) => {
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
    const heads = num2(p.numHeads ?? p.num_heads ?? p.nhead, 8);
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
      const window2 = t === "windowAttention" ? Math.min(Tk, num2(p.windowSize, 7) ** 2) : Tk;
      const scores = heads * T * window2;
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
    const heads = num2(p.numHeads, 32);
    const kvHeads = num2(p.numKVHeads, 8);
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
    const ff = num2(p.ffDim ?? p.dFf ?? p.hiddenFeatures, 4 * d);
    const saved2 = [castIn(x, ctx, c.id), make(id("h"), T * ff, ctx.low)];
    return { saved: saved2, out: make(id("out"), O, ctx.low), backwardScratch: 0, weightCopyBytes: weightCopy, fallback: false };
  }
  if ((t === "swiglu" || t === "geglu") && finiteShape(inShape)) {
    const d = inShape[inShape.length - 1];
    const T = prod(inShape) / d;
    const I = num2(p.intermediateSize ?? p.hiddenFeatures ?? p.ffDim, Math.round(d * 8 / 3));
    const saved2 = [castIn(x, ctx, c.id), make(id("gate"), T * I, ctx.low), make(id("up"), T * I, ctx.low), make(id("act"), T * I, ctx.low), make(id("prod"), T * I, ctx.low)];
    return { saved: saved2, out: make(id("out"), O, ctx.low), backwardScratch: 0, weightCopyBytes: weightCopy, fallback: false };
  }
  if (t === "transformerBlock" && finiteShape(inShape) && inShape.length >= 2) {
    const T = inShape[inShape.length - 2];
    const d = num2(p.embedDim ?? p.hiddenDim ?? p.dModel, inShape[inShape.length - 1]);
    const heads = num2(p.numHeads, 8);
    const ff = num2(p.ffDim ?? p.dFf, 4 * d);
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
    const hidden = num2(p.hiddenSize ?? p.hidden_size, 128);
    const layers = num2(p.numLayers ?? p.num_layers, 1);
    const dirs = p.bidirectional || t === "bidirectionalLSTM" ? 2 : 1;
    const elems = steps * hidden * (gates + 2) * layers * dirs;
    return { saved: [castIn(x, ctx, c.id), make(id("rnn"), elems, ctx.low)], out: make(id("out"), O, ctx.low), backwardScratch: 0, weightCopyBytes: weightCopy, fallback: false };
  }
  const din = finiteShape(inShape) ? inShape[inShape.length - 1] : 0;
  let internal = 0;
  if (din >= 2 && tokens > 0 && !/embed|positional|^ema/i.test(t)) {
    let pp = params;
    if (/moe/i.test(t)) {
      const e = num2(p.numExperts, 8), k = num2(p.topK, 2), s = Number(p.numSharedExperts ?? 0);
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
function trainingMemoryFor(model, opts = {}, shapesIn) {
  const batch = Math.max(1, Math.round(opts.batchSize ?? 32));
  const precision = opts.precision ?? DEFAULT_PRECISION;
  const optimizer = opts.optimizer ?? DEFAULT_OPTIMIZER;
  const mixed = precision === "bf16-mixed" || precision === "fp16-mixed";
  const ctx = { low: precision === "fp32" ? "f32" : "half", mixed, storesScores: opts.storesScores ?? (() => false) };
  const shapes = shapesIn ?? propagateShapes(model).shapes;
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

// src/utils/runEvidence.ts
var MEASURED_RUN_MODES = ["gpu", "colab", "kaggle"];
function isMeasuredRunMode(mode) {
  return typeof mode === "string" && MEASURED_RUN_MODES.includes(mode);
}
var WHERE = {
  gpu: "a real GPU",
  colab: "Colab",
  kaggle: "Kaggle"
};
function runEvidence(run) {
  if (!run) {
    return {
      measured: false,
      reason: "never-run",
      sentence: "Nothing has been trained yet"
    };
  }
  if (isMeasuredRunMode(run.mode)) {
    return {
      measured: true,
      mode: run.mode,
      where: WHERE[run.mode],
      sentence: `Trained on ${WHERE[run.mode]}`
    };
  }
  if (run.mode === "simulation") {
    return {
      measured: false,
      reason: "simulated",
      sentence: "This run was simulated, so its curves come from the architecture rather than from data"
    };
  }
  return {
    measured: false,
    reason: "unknown-mode",
    sentence: "This run does not record how it ran, so it cannot be read as a measurement"
  };
}

// src/utils/peakCheck.ts
var PEAK_MISS_TOLERANCE = 0.25;
var PEAK_GRADE_FLOOR_GB = 1;
function normalizePrecision(p) {
  const s = String(p ?? "").trim().toLowerCase();
  if (s === "bf16-mixed") return "bf16-mixed";
  if (s === "16-mixed" || s === "fp16-mixed" || s === "16") return "fp16-mixed";
  if (s === "bf16-true" || s === "bf16") return "bf16";
  if (s === "32-true" || s === "32" || s === "fp32") return "fp32";
  return null;
}
function normalizeOptimizer(o) {
  const s = String(o ?? "").trim().toLowerCase();
  if (s === "adamw" || s === "adam") return s;
  if (s === "sgd" || s === "sgd-momentum") return "sgd-momentum";
  return null;
}
function comparePeak(model, run) {
  const mem = run.memory;
  if (!mem || !(mem.peakMb > 0)) return null;
  if (!runEvidence(run).measured) return null;
  const precision = normalizePrecision(mem.precision);
  const optimizer = normalizeOptimizer(mem.optimizer);
  if (!precision || !optimizer || !(mem.batchSize > 0)) return null;
  let predictedGb;
  try {
    const t = trainingMemoryFor(model, { batchSize: mem.batchSize, precision, optimizer });
    if (!t.activationsKnown) return null;
    predictedGb = t.peakGb;
  } catch {
    return null;
  }
  if (!(predictedGb > 0)) return null;
  const measuredGb = mem.peakMb * 1024 * 1024 / 1e9;
  const miss = (measuredGb - predictedGb) / predictedGb;
  const grade = measuredGb < PEAK_GRADE_FLOOR_GB ? "ungraded" : miss > PEAK_MISS_TOLERANCE ? "low" : miss < -PEAK_MISS_TOLERANCE ? "high" : "within";
  return { predictedGb, measuredGb, miss, batchSize: mem.batchSize, precision, optimizer, grade, source: mem.source, gpu: run.gpu };
}
var gb = (x) => x >= 10 ? x.toFixed(1) : x.toFixed(2);
var pctSigned = (x) => `${x >= 0 ? "+" : ""}${Math.round(x * 100)}%`;
function peakNote(c) {
  const setting = `batch ${c.batchSize}, ${c.precision}, ${c.optimizer}${c.gpu ? `, on ${c.gpu}` : ""}`;
  const base = `Pre-flight predicted ${gb(c.predictedGb)} GB for max_memory_allocated at ${setting}; the run measured ${gb(c.measuredGb)} GB (${c.source}).`;
  switch (c.grade) {
    case "within":
      return {
        severity: "good",
        ruleId: "peak-vs-measured",
        agentFixable: false,
        title: `Peak memory: predicted ${gb(c.predictedGb)} GB, measured ${gb(c.measuredGb)} GB (${pctSigned(c.miss)})`,
        detail: `${base} Within ${Math.round(PEAK_MISS_TOLERANCE * 100)}%, so fit verdicts for this design can be read at face value.`
      };
    case "low":
      return {
        severity: "warn",
        ruleId: "peak-vs-measured",
        agentFixable: false,
        title: `Peak memory ran ${pctSigned(c.miss)} over the estimate on this design`,
        detail: `${base} The estimate was low here, so a "fits" verdict for this structure at a larger batch or a smaller card is optimistic by about that much. This is a fact about the estimate, not about the design.`
      };
    case "high":
      return {
        severity: "info",
        ruleId: "peak-vs-measured",
        agentFixable: false,
        title: `Peak memory came in ${pctSigned(c.miss)} under the estimate`,
        detail: `${base} The estimate was conservative here, so a "does not fit" verdict for this structure may be too cautious.`
      };
    case "ungraded":
      return {
        severity: "info",
        ruleId: "peak-vs-measured",
        agentFixable: false,
        title: `Peak memory: predicted ${gb(c.predictedGb)} GB, measured ${gb(c.measuredGb)} GB`,
        detail: `${base} Under ${PEAK_GRADE_FLOOR_GB} GB the gap is not graded: allocator rounding and kernel workspaces are a large share of a run this small.`
      };
  }
}

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
    const list = embInputsByAdd.get(cn.to) ?? [];
    if (!list.includes(cn.from)) list.push(cn.from);
    embInputsByAdd.set(cn.to, list);
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
      const pts2 = n(p.numPoints, 4);
      const lv = n(p.numLevels, 4);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 ? 2 * T * D * D + T * (h * lv * pts2) * D : 0;
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
function kvCachePerTokenBytes(spec) {
  const L = Math.max(0, spec.numLayers);
  if (spec.mlaLatentDim && spec.mlaLatentDim > 0) {
    return (spec.mlaLatentDim + (spec.mlaRopeDim ?? 0)) * spec.bytesPerValue * L;
  }
  const kvHeads = spec.kvHeads ?? 0;
  const headDim = spec.headDim ?? 0;
  return 2 * kvHeads * headDim * spec.bytesPerValue * L;
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
function modalityFromShape(shape) {
  if (!shape || shape.length === 0) return "tabular";
  if (shape.length >= 3) return "vision";
  if (shape.length === 1 && shape[0] > 256) return "audio";
  return "tabular";
}
function detectInputModality(model) {
  const types = new Set(model.components.map((c) => c.type));
  const input = model.components.find((c) => c.type === "input");
  const shape = input?.params?.shape ?? [];
  if (types.has("embedding") || types.has("embeddingBag")) return "nlp";
  if ((types.has("transformerBlock") || types.has("multiHeadAttention") || types.has("causalAttention") || types.has("attention")) && !types.has("conv2d")) return "nlp";
  if (types.has("melSpectrogram") || types.has("audioConv") || types.has("mfcc")) return "audio";
  if (types.has("conv1d") && shape.length === 1 && (shape[0] ?? 0) > 256) return "audio";
  if (types.has("conv2d") || types.has("conv3d") || shape.length >= 3) return "vision";
  return modalityFromShape(shape);
}
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
    const first2 = missingParams[0];
    const others = missingParams.length - 1;
    const alsoText = others > 0 ? ` ${others} other layer${others !== 1 ? "s" : ""} ${others !== 1 ? "are" : "is"} incomplete too.` : "";
    findings.push({
      id: "structure-missing-params",
      category: "structure",
      severity: "block",
      title: missingParams.length === 1 ? `"${first2.componentName}" is missing ${first2.params.join(", ")}` : `${missingParams.length} layers are missing required parameters`,
      detail: `"${first2.componentName}" (${first2.componentType}) has no value for ${first2.params.join(", ")}.${alsoText} A layer without these cannot be built, so the run fails at construction rather than part way through.`,
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
  const memory2 = {
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
      memory: memory2,
      fitsCards,
      compute: params > 0 ? { trainFlops, peakTflops, assumedMfu: ASSUMED_MFU, computeSec } : void 0,
      weightBytes,
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

// src/utils/dataSource.ts
var DATA_SOURCE_KEY = "pipeline-data-source";
var HF_ID_MIGRATIONS = {
  cifar10: "uoft-cs/cifar10",
  mnist: "ylecun/mnist",
  imdb: "stanfordnlp/imdb",
  glue: "nyu-mll/glue",
  speech_commands: "google/speech_commands"
};
function migrateDataSource(s) {
  if (!s || s.kind !== "huggingface") return s;
  const id = (s.hfDatasetId ?? "").trim();
  const moved = HF_ID_MIGRATIONS[id];
  return moved ? { ...s, hfDatasetId: moved } : s;
}
function getDataSource() {
  try {
    const raw = localStorage.getItem(DATA_SOURCE_KEY);
    if (!raw) return null;
    const stored = JSON.parse(raw);
    const migrated = migrateDataSource(stored);
    if (migrated !== stored) {
      try {
        localStorage.setItem(DATA_SOURCE_KEY, JSON.stringify(migrated));
      } catch {
      }
    }
    return migrated;
  } catch {
    return null;
  }
}
var PIPELINE_STATE_CHANGED = "pipeline:state-changed";
function hasRealData(s) {
  if (!s || s.kind === "synthetic") return false;
  if (s.kind === "huggingface") return !!s.hfDatasetId?.trim();
  return !!s.path?.trim();
}
function describeDataSource(s) {
  if (!hasRealData(s)) return "Synthetic (no real data wired yet)";
  switch (s.kind) {
    case "huggingface":
      return `\u{1F917} ${s.hfDatasetId}${s.hfConfig ? ` (${s.hfConfig})` : ""}`;
    case "csv":
      return `CSV \xB7 ${s.path}`;
    case "image_zip":
      return `Image folder \xB7 ${s.path}`;
    default:
      return "Synthetic";
  }
}

// src/utils/dataContractExporter.ts
function sanitizeModelName(arch) {
  return String(arch.name ?? "MyModel").replace(/[^a-zA-Z0-9]/g, "") || "MyModel";
}
function numericShape(raw) {
  return Array.isArray(raw) ? raw.filter((n2) => typeof n2 === "number" && Number.isFinite(n2) && n2 > 0) : [];
}
function targetsOf(arch, id) {
  const ids = arch.connections.filter((c) => c.from === id).map((c) => c.to);
  return arch.components.filter((c) => ids.includes(c.id));
}
function isEmbeddingType(t) {
  return /embedding/i.test(t);
}
function feedsEmbedding(arch, input) {
  if (!input) return arch.components.some((c) => isEmbeddingType(c.type));
  const next = targetsOf(arch, input.id);
  if (next.some((c) => isEmbeddingType(c.type))) return true;
  return next.some((c) => targetsOf(arch, c.id).some((d) => isEmbeddingType(d.type)));
}
function terminalLayers(arch) {
  const hasOut = new Set(arch.connections.map((c) => c.from));
  const terminals = arch.components.filter(
    (c) => c.type !== "output" && c.type !== "input" && !hasOut.has(c.id)
  );
  if (terminals.length > 0) return terminals;
  const out = arch.components.find((c) => c.type === "output");
  if (out) {
    const fromIds = arch.connections.filter((c) => c.to === out.id).map((c) => c.from);
    return arch.components.filter((c) => fromIds.includes(c.id));
  }
  return [];
}
function heuristicTask(arch) {
  const terminals = terminalLayers(arch);
  const isLMHead = (c) => /lmhead|languagemodel/i.test(c.type ?? "") || /lm.?head|language.?model/i.test(c.name ?? "");
  if (terminals.some(isLMHead) || arch.components.some(isLMHead)) return "language-modeling";
  const DIFF_RE = /latent|timestep|noisy|noise|diffus|denois|\bvae\b|autoencoder/i;
  const inputs = arch.components.filter((c) => c.type === "input");
  if (inputs.some((c) => DIFF_RE.test(c.name ?? "")) || arch.components.some((c) => DIFF_RE.test(c.type) || /diffus|denois/i.test(c.name ?? ""))) {
    return "diffusion";
  }
  return "classification";
}
function outFeatureLike(params) {
  if (!params) return null;
  for (const key of ["outFeatures", "outChannels", "numClasses", "vocabSize", "units", "outputDim"]) {
    const v = params[key];
    if (typeof v === "number" && Number.isFinite(v) && v > 0) return v;
  }
  return null;
}
function deriveOutputDim(arch) {
  try {
    const { shapes } = propagateShapes(arch);
    for (const t of terminalLayers(arch)) {
      const out = shapes.get(t.id)?.out;
      if (out && out.length > 0) {
        const last = out[out.length - 1];
        if (typeof last === "number" && Number.isFinite(last) && last > 0) return last;
      }
    }
  } catch {
  }
  for (const t of terminalLayers(arch)) {
    const v = outFeatureLike(t.params);
    if (v != null) return v;
  }
  for (let i = arch.components.length - 1; i >= 0; i--) {
    const v = outFeatureLike(arch.components[i].params);
    if (v != null) return v;
  }
  return null;
}
function deriveModelContract(arch, opts = {}) {
  const input = arch.components.find((c) => c.type === "input");
  const overrideShape = numericShape(opts.inputShape);
  return {
    modelName: sanitizeModelName(arch),
    inputShape: overrideShape.length ? overrideShape : numericShape(input?.params?.shape),
    inputDtype: feedsEmbedding(arch, input) ? "long" : "float32",
    outputDim: typeof opts.numClasses === "number" && opts.numClasses > 0 ? opts.numClasses : deriveOutputDim(arch),
    taskType: opts.taskType ?? heuristicTask(arch)
  };
}

// src/utils/hfDatasetMeta.ts
var HF_VIEWER_BASE = "https://datasets-server.huggingface.co";
var HF_META_CACHE_KEY = "hf-dataset-meta-cache";
var OK_TTL_MS = 24 * 3600 * 1e3;
var FAIL_TTL_MS = 5 * 60 * 1e3;
var memory = /* @__PURE__ */ new Map();
var inFlight = /* @__PURE__ */ new Map();
function hfCacheKey(id, config) {
  return `${id.trim().toLowerCase()}::${(config ?? "").trim().toLowerCase()}`;
}
function readPersisted() {
  try {
    const raw = typeof localStorage !== "undefined" ? localStorage.getItem(HF_META_CACHE_KEY) : null;
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}
function persist(key, probe) {
  try {
    if (typeof localStorage === "undefined") return;
    const all = readPersisted();
    all[key] = probe;
    const keys = Object.keys(all);
    if (keys.length > 20) {
      keys.sort((a, b) => probeTime(all[a]) - probeTime(all[b]));
      for (const k of keys.slice(0, keys.length - 20)) delete all[k];
    }
    localStorage.setItem(HF_META_CACHE_KEY, JSON.stringify(all));
  } catch {
  }
}
var probeTime = (p) => p.status === "ok" ? p.meta.fetchedAt : p.at;
function fresh(p, now) {
  if (!p) return false;
  return now - probeTime(p) < (p.status === "ok" ? OK_TTL_MS : FAIL_TTL_MS);
}
function readHfDatasetMeta(source, now = Date.now()) {
  if (!source || source.kind !== "huggingface" || !source.hfDatasetId?.trim()) return null;
  const key = hfCacheKey(source.hfDatasetId, source.hfConfig);
  const m = memory.get(key);
  if (fresh(m, now)) return m;
  const p = readPersisted()[key];
  if (fresh(p, now)) {
    memory.set(key, p);
    return p;
  }
  return null;
}
async function getJson(fetchImpl, path, params, signal) {
  const qs = new URLSearchParams(params).toString();
  const res = await fetchImpl(`${HF_VIEWER_BASE}${path}?${qs}`, { signal });
  const body = await res.json().catch(() => null);
  if (!res.ok || !body || typeof body !== "object") {
    const msg = body && typeof body.error === "string" ? body.error : `HTTP ${res.status}`;
    throw new Error(`${path}: ${msg}`);
  }
  return body;
}
async function fetchHfDatasetMeta(source, fetchImpl, opts = {}) {
  const now = opts.now ?? Date.now;
  const id = (source.hfDatasetId ?? "").trim();
  const key = hfCacheKey(id, source.hfConfig);
  const running = inFlight.get(key);
  if (running) return running;
  const work = (async () => {
    try {
      if (!id) throw new Error("no dataset id");
      const infoBody = await getJson(fetchImpl, "/info", { dataset: id, ...source.hfConfig ? { config: source.hfConfig } : {} }, opts.signal);
      const parsed = parseInfo(infoBody, source.hfConfig ?? null);
      if (!parsed) throw new Error("the dataset viewer returned no schema for it");
      let splits = parsed.splits;
      if (splits.length === 0 || splits.every((s) => s.numExamples == null)) {
        try {
          const sizeBody = await getJson(fetchImpl, "/size", { dataset: id, ...parsed.config ? { config: parsed.config } : {} }, opts.signal);
          splits = parseSizeSplits(sizeBody, parsed.config) ?? splits;
        } catch {
        }
      }
      const cfg = parsed.config ?? "default";
      const { inputColumn, inputKind, labelColumn } = pickColumns(parsed.features);
      const wanted = orderSplits(splits.map((s) => s.name)).slice(0, 3);
      const stats = [];
      for (const split of wanted) {
        try {
          const body = await getJson(fetchImpl, "/statistics", { dataset: id, config: cfg, split }, opts.signal);
          stats.push(parseStatistics(body, split, inputColumn, labelColumn, splits.find((s) => s.name === split)?.numExamples ?? null));
        } catch {
        }
      }
      const meta = {
        datasetId: id,
        config: parsed.config,
        ...parsed.otherConfigs ? { otherConfigs: parsed.otherConfigs } : {},
        features: parsed.features,
        inputColumn,
        inputKind,
        labelColumn,
        classNames: parsed.features.find((f) => f.name === labelColumn)?.classNames ?? null,
        numClasses: null,
        splits,
        stats,
        fetchedAt: now()
      };
      meta.numClasses = meta.classNames?.length ?? observedClassCount(stats);
      if (inputColumn && (inputKind === "image" || inputKind === "tokens") && wanted[0]) {
        try {
          const rowsBody = await getJson(fetchImpl, "/first-rows", { dataset: id, config: cfg, split: wanted[0] }, opts.signal);
          Object.assign(meta, parseFirstRows(rowsBody, inputColumn, inputKind));
        } catch {
        }
      }
      return { status: "ok", meta };
    } catch (e) {
      return { status: "unavailable", reason: e instanceof Error ? e.message : String(e), at: now() };
    }
  })();
  inFlight.set(key, work);
  try {
    const probe = await work;
    memory.set(key, probe);
    persist(key, probe);
    return probe;
  } finally {
    inFlight.delete(key);
  }
}
var autoFetch = null;
var onAutoFetched = null;
function requestHfDatasetMeta(source) {
  if (!autoFetch || !source || source.kind !== "huggingface" || !source.hfDatasetId?.trim()) return false;
  if (readHfDatasetMeta(source)) return false;
  const key = hfCacheKey(source.hfDatasetId, source.hfConfig);
  if (inFlight.has(key)) return true;
  void fetchHfDatasetMeta(source, autoFetch).then(() => {
    if (onAutoFetched) onAutoFetched();
    else if (typeof window !== "undefined") {
      try {
        window.dispatchEvent(new Event(PIPELINE_STATE_CHANGED));
      } catch {
      }
    }
  }).catch(() => {
  });
  return true;
}
var isObj = (v) => !!v && typeof v === "object" && !Array.isArray(v);
var num3 = (v) => typeof v === "number" && Number.isFinite(v) ? v : null;
function featureKind(f) {
  if (!isObj(f)) return { kind: "other" };
  const t = f._type;
  if (t === "ClassLabel") {
    return { kind: "class_label", classNames: Array.isArray(f.names) ? f.names.map(String) : void 0 };
  }
  if (t === "Image") return { kind: "image" };
  if (t === "Audio") return { kind: "audio" };
  if (t === "Value") {
    const d = String(f.dtype ?? "");
    if (d === "string" || d === "large_string") return { kind: "text" };
    if (/^u?int/.test(d)) return { kind: "int" };
    if (/^float|^double|^decimal/.test(d)) return { kind: "float" };
    if (d === "bool") return { kind: "bool" };
    return { kind: "other" };
  }
  if (t === "Sequence" || t === "List" || t === "LargeList") {
    const inner = featureKind(f.feature);
    if (inner.kind === "int" || inner.kind === "class_label") return { kind: "sequence_int" };
    if (inner.kind === "float") return { kind: "sequence_float" };
    return { kind: "other" };
  }
  if (Array.isArray(f)) {
    const inner = featureKind(f[0]);
    if (inner.kind === "int") return { kind: "sequence_int" };
    if (inner.kind === "float") return { kind: "sequence_float" };
  }
  return { kind: "other" };
}
function parseFeatures(features) {
  if (!isObj(features)) return [];
  return Object.entries(features).map(([name, f]) => {
    const k = Array.isArray(f) ? featureKind(f) : featureKind(f);
    return { name, kind: k.kind, ...k.classNames ? { classNames: k.classNames } : {} };
  });
}
function parseInfo(body, wantedConfig) {
  const di = body.dataset_info;
  if (!isObj(di)) return null;
  let info;
  let config = wantedConfig;
  let otherConfigs;
  if ("features" in di) {
    info = di;
    config = typeof di.config_name === "string" ? di.config_name : wantedConfig;
  } else {
    const names = Object.keys(di);
    if (names.length === 0) return null;
    const pick = wantedConfig && names.includes(wantedConfig) ? wantedConfig : names.includes("default") ? "default" : names[0];
    info = isObj(di[pick]) ? di[pick] : {};
    config = pick;
    if (names.length > 1 && !wantedConfig) otherConfigs = names.filter((n2) => n2 !== pick);
  }
  const features = parseFeatures(info.features);
  const splits = [];
  if (isObj(info.splits)) {
    for (const [name, s] of Object.entries(info.splits)) {
      splits.push({ name, numExamples: isObj(s) ? num3(s.num_examples) : null });
    }
  }
  return { config, ...otherConfigs ? { otherConfigs } : {}, features, splits };
}
function parseSizeSplits(body, config) {
  const size = isObj(body.size) ? body.size : null;
  if (!size || !Array.isArray(size.splits)) return null;
  return size.splits.filter(isObj).filter((s) => !config || s.config === config).map((s) => ({ name: String(s.split), numExamples: num3(s.num_rows) }));
}
function orderSplits(names) {
  const rank = (n2) => {
    const s = n2.toLowerCase();
    if (s === "train") return 0;
    if (s === "test") return 1;
    if (s === "validation" || s === "valid" || s === "val" || s === "dev") return 2;
    if (s.startsWith("train")) return 3;
    return 4;
  };
  return [...names].sort((a, b) => rank(a) - rank(b));
}
var LABEL_NAMES = /^(label|labels|target|targets|class|y|category|sentiment)$/i;
var TEXT_NAMES = /^(text|sentence|sentence1|content|review|document|question|prompt|input|inputs|tweet|comment)$/i;
var TOKEN_NAMES = /(input_ids|token|ids)$/i;
function pickColumns(features) {
  const label = features.find((f) => f.kind === "class_label") ?? features.find((f) => LABEL_NAMES.test(f.name) && (f.kind === "int" || f.kind === "text" || f.kind === "float" || f.kind === "bool"));
  const rest = features.filter((f) => f !== label && !/^(idx|id|index)$/i.test(f.name));
  const image = rest.find((f) => f.kind === "image");
  if (image) return { inputColumn: image.name, inputKind: "image", labelColumn: label?.name ?? null };
  const audio = rest.find((f) => f.kind === "audio");
  if (audio) return { inputColumn: audio.name, inputKind: "audio", labelColumn: label?.name ?? null };
  const tokens = rest.find((f) => f.kind === "sequence_int" && TOKEN_NAMES.test(f.name));
  if (tokens) return { inputColumn: tokens.name, inputKind: "tokens", labelColumn: label?.name ?? null };
  const text = rest.find((f) => f.kind === "text" && TEXT_NAMES.test(f.name)) ?? rest.find((f) => f.kind === "text");
  if (text) return { inputColumn: text.name, inputKind: "text", labelColumn: label?.name ?? null };
  const vector = rest.find((f) => f.kind === "sequence_float" || f.kind === "sequence_int");
  if (vector) return { inputColumn: vector.name, inputKind: "tabular", labelColumn: label?.name ?? null };
  if (rest.some((f) => f.kind === "float" || f.kind === "int")) return { inputColumn: null, inputKind: "tabular", labelColumn: label?.name ?? null };
  return { inputColumn: null, inputKind: null, labelColumn: label?.name ?? null };
}
function parseStatistics(body, split, inputColumn, labelColumn, numExamples) {
  const out = {
    split,
    numExamples: num3(body.num_examples) ?? numExamples,
    partial: body.partial === true
  };
  const cols = Array.isArray(body.statistics) ? body.statistics.filter(isObj) : [];
  for (const c of cols) {
    const name = c.column_name;
    const st = isObj(c.column_statistics) ? c.column_statistics : {};
    const type = String(c.column_type ?? "");
    if (name === labelColumn) {
      if (isObj(st.frequencies)) {
        const freq = {};
        for (const [k, v] of Object.entries(st.frequencies)) if (num3(v) != null) freq[k] = v;
        out.labelFrequencies = freq;
      }
      if (num3(st.no_label_proportion) != null) out.labelNoLabelProportion = st.no_label_proportion;
      if (num3(st.nan_proportion) != null) out.labelNanProportion = st.nan_proportion;
    } else if (name === inputColumn) {
      if (num3(st.nan_proportion) != null) out.inputNanProportion = st.nan_proportion;
      const hist = isObj(st.histogram) ? st.histogram : null;
      if ((type === "string_text" || type === "string_label") && hist && Array.isArray(hist.hist) && Array.isArray(hist.bin_edges)) {
        out.textLength = {
          min: num3(st.min) ?? 0,
          max: num3(st.max) ?? 0,
          median: num3(st.median),
          hist: hist.hist.map((v) => num3(v) ?? 0),
          edges: hist.bin_edges.map((v) => num3(v) ?? 0)
        };
      } else if (type === "image") {
        out.imageWidth = { min: num3(st.min) ?? 0, max: num3(st.max) ?? 0, median: num3(st.median) };
      }
    }
  }
  return out;
}
function observedClassCount(stats) {
  const train = stats.find((s) => s.split.toLowerCase() === "train") ?? stats[0];
  const f = train?.labelFrequencies;
  if (!f) return null;
  const n2 = Object.values(f).filter((v) => v > 0).length;
  return n2 > 1 ? n2 : null;
}
var median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
function parseFirstRows(body, inputColumn, inputKind) {
  const rows = Array.isArray(body.rows) ? body.rows.filter(isObj) : [];
  const cells = rows.map((r) => isObj(r.row) ? r.row[inputColumn] : void 0);
  if (inputKind === "image") {
    const hs = [];
    const ws = [];
    for (const c of cells) {
      if (isObj(c) && num3(c.height) != null && num3(c.width) != null) {
        hs.push(c.height);
        ws.push(c.width);
      }
    }
    return hs.length ? { imageSize: { height: median(hs), width: median(ws), rows: hs.length } } : {};
  }
  if (inputKind === "tokens") {
    let mx = -1;
    let seen = 0;
    for (const c of cells) {
      if (!Array.isArray(c)) continue;
      seen++;
      for (const t of c) if (typeof t === "number" && t > mx) mx = t;
    }
    return seen && mx >= 0 ? { maxTokenId: { value: mx, rows: seen } } : {};
  }
  return {};
}

// src/utils/datasetContract.ts
var KNOWN_DATASETS = {
  mnist: { key: "mnist", label: "MNIST", inputShape: [1, 28, 28], classes: 10 },
  cifar10: { key: "cifar10", label: "CIFAR-10", inputShape: [3, 32, 32], classes: 10 },
  imdb: { key: "imdb", label: "IMDB", inputShape: [], classes: 2 },
  glue: { key: "glue", label: "GLUE / SST-2", inputShape: [], classes: 2 }
};
function knownDatasetFor(source) {
  if (!source || source.kind !== "huggingface") return null;
  const id = (source.hfDatasetId ?? "").trim().toLowerCase();
  if (!id) return null;
  const tail = id.includes("/") ? id.slice(id.lastIndexOf("/") + 1) : id;
  return KNOWN_DATASETS[tail] ?? null;
}
function contractMismatches(model, source) {
  const ds = knownDatasetFor(source);
  if (!ds) return [];
  const contract = deriveModelContract(model);
  const out = [];
  if (ds.inputShape.length > 0) {
    const got = contract.inputShape;
    const same = got.length === ds.inputShape.length && got.every((d, i) => d === ds.inputShape[i]);
    if (!same) {
      out.push({
        kind: "input-shape",
        title: `Input shape does not match ${ds.label}`,
        detail: `The graph declares [${got.join(", ") || "nothing"}] and ${ds.label} is [${ds.inputShape.join(", ")}]. The run would resize and re-channel every sample to fit the graph, so the model would train on something other than the data you chose.`,
        fix: `Set the input layer's shape to [${ds.inputShape.join(", ")}].`,
        expected: ds.inputShape,
        actual: got
      });
    }
  }
  const outputDim = contract.outputDim;
  if (typeof outputDim === "number" && outputDim > 0 && outputDim !== ds.classes) {
    out.push({
      kind: "output-classes",
      title: `Head is ${outputDim} wide, ${ds.label} has ${ds.classes} classes`,
      detail: outputDim > ds.classes ? `The last layer predicts ${outputDim} classes and only ${ds.classes} exist. Cross-entropy over ${outputDim - ds.classes} classes that never appear makes the model slower to learn and its reported accuracy hard to read.` : `The last layer predicts ${outputDim} classes and the data has ${ds.classes}. Labels above ${outputDim - 1} cannot be represented at all.`,
      fix: `Set the final layer's outFeatures to ${ds.classes}.`,
      expected: ds.classes,
      actual: outputDim
    });
  }
  return out;
}
function derivedDataSpec(source) {
  if (!source || source.kind !== "huggingface") return null;
  const known = knownDatasetFor(source);
  const probe = readHfDatasetMeta(source);
  const meta = probe?.status === "ok" ? probe.meta : null;
  const split = (source.hfSplit ?? "train").toLowerCase();
  const rows = meta ? meta.splits.find((s) => s.name.toLowerCase() === split)?.numExamples ?? meta.splits.find((s) => s.name.toLowerCase() === "train")?.numExamples ?? null : null;
  const classes = known?.classes ?? meta?.numClasses ?? null;
  if (rows == null && classes == null) return null;
  return {
    ...rows != null && rows > 0 ? { samples: rows } : {},
    ...classes != null && classes > 1 ? { numClasses: classes } : {},
    from: known?.label ?? meta?.datasetId ?? (source.hfDatasetId ?? "the wired dataset")
  };
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

// src/utils/pipelineFlow.ts
function deriveRunInsights(run) {
  const out = [];
  const { trainAcc, valAcc, trainLoss, valLoss } = run;
  const measured = runEvidence(run).measured;
  const sev = (s) => measured ? s : "info";
  const said = measured ? "" : `${runEvidence(run).sentence}. This is a prediction, not a measurement. `;
  const haveAcc = typeof trainAcc === "number" && typeof valAcc === "number";
  const gap = haveAcc ? trainAcc - valAcc : null;
  const lossGap = !haveAcc && typeof trainLoss === "number" && typeof valLoss === "number" && trainLoss > 0.05 ? (valLoss - trainLoss) / trainLoss : null;
  if (gap != null && gap > 0.12 || lossGap != null && lossGap > 0.35) {
    out.push({
      severity: sev("warn"),
      title: measured ? "Overfit in training" : "Overfit predicted",
      detail: said + (gap != null ? `Train accuracy ${(trainAcc * 100).toFixed(0)}% vs validation ${(valAcc * 100).toFixed(0)}%. Add dropout / weight decay or more data before serving, or it will underperform on real inputs.` : "Validation loss ran well above training loss. Add regularization or more data before serving.")
    });
  }
  if (typeof valAcc === "number" && valAcc < 0.6) {
    out.push({
      severity: sev("warn"),
      title: measured ? "Low validation accuracy" : "Low validation accuracy predicted",
      detail: said + `Validation accuracy is ${(valAcc * 100).toFixed(0)}%. Train longer, raise capacity, or revisit the data before relying on this in production.`
    });
  }
  const ev = runEvidence(run);
  if (!ev.measured) {
    out.push({
      severity: "info",
      title: ev.reason === "simulated" ? "Results are simulated" : "Results are unmeasured",
      detail: `${ev.sentence}. Run real training before trusting deploy readiness.`
    });
  } else if (out.length === 0 && typeof valAcc === "number" && valAcc >= 0.6) {
    out.push({
      severity: "good",
      title: "Training looks healthy",
      detail: `Validation accuracy ${(valAcc * 100).toFixed(0)}% with no large train/val gap. Reasonable to proceed to serving.`
    });
  }
  return out;
}
var PIPELINE_LAST_RUN_KEY = "pipeline-last-run";
function designFingerprint(model) {
  const comps = (model?.components ?? []).filter((c) => c && c.type !== "stickyNote");
  if (comps.length === 0) return null;
  const ids = new Set(comps.map((c) => c.id));
  const conns = (model.connections ?? []).filter((c) => c && ids.has(c.from) && ids.has(c.to));
  return architectureFingerprint({ ...model, components: comps, connections: conns }).hash;
}
function getLastRun() {
  try {
    const raw = localStorage.getItem(PIPELINE_LAST_RUN_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}
function getLastRunFor(model) {
  const run = getLastRun();
  if (!run) return { run: null, otherDesign: null };
  if (run.archHash) {
    return run.archHash === designFingerprint(model) ? { run, otherDesign: null } : { run: null, otherDesign: run };
  }
  return { run, otherDesign: null };
}
function readDataSpec() {
  let declared;
  try {
    const r = localStorage.getItem("preflight-data-spec");
    declared = r ? JSON.parse(r) : void 0;
  } catch {
    declared = void 0;
  }
  const derived = derivedDataSpec(getDataSource());
  if (!derived) return declared;
  const merged = { ...declared ?? {} };
  if (!(merged.samples && merged.samples > 0) && derived.samples) merged.samples = derived.samples;
  if (!(merged.numClasses && merged.numClasses > 0) && derived.numClasses) merged.numClasses = derived.numClasses;
  return merged;
}
function readTrainPlan() {
  try {
    const r = localStorage.getItem("preflight-plan");
    return r ? JSON.parse(r) : void 0;
  } catch {
    return void 0;
  }
}

// src/utils/runDiagnosis.ts
var AT_CHANCE_TOLERANCE = 0.05;
var FLAT_EPSILON = 5e-3;
var OVERFIT_GAP = 0.12;
var PLATEAU_SKILL_FLOOR = 0.75;
var DIVERGENCE_RISE = 1.2;
var ROWS_PER_PARAM_FLOOR = 0.01;
var SPIKE_SIGMAS = 6;
var SPIKE_WINDOW = 8;
var SPIKE_MIN_HISTORY = 3;
var FIRST_LOSS_RATIO = 2;
function expectedInitialLoss(model, source) {
  const contract = deriveModelContract(model);
  if (contract.taskType === "regression" || contract.taskType === "diffusion") return null;
  const dataset = knownDatasetFor(source);
  const classes = dataset?.classes ?? (typeof contract.outputDim === "number" ? contract.outputDim : null);
  if (!classes || classes < 2) return null;
  const basis = contract.taskType === "language-modeling" ? "vocabulary" : "classes";
  const value = Math.log(classes);
  const what = basis === "vocabulary" ? `ln ${classes.toLocaleString("en-US")}, a ${classes.toLocaleString("en-US")}-token vocabulary` : `ln ${classes}, ${classes}-way softmax cross-entropy`;
  return { value, classes, basis, sentence: `about ${value.toFixed(2)} (${what})` };
}
function firstNonFinite(values) {
  return values.findIndex((v) => v !== void 0 && (typeof v !== "number" || !Number.isFinite(v)));
}
function findLossSpike(values, opts = {}) {
  const w = opts.window ?? SPIKE_WINDOW;
  const k = opts.sigmas ?? SPIKE_SIGMAS;
  const minH = opts.minHistory ?? SPIKE_MIN_HISTORY;
  for (let i = minH; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) return null;
    const hist = values.slice(Math.max(0, i - w), i).filter(Number.isFinite);
    if (hist.length < minH) continue;
    const mean = hist.reduce((a, b) => a + b, 0) / hist.length;
    const std = Math.sqrt(hist.reduce((a, b) => a + (b - mean) ** 2, 0) / hist.length);
    if (std < 1e-10) continue;
    if (v > mean + k * std) return { index: i, value: v, mean, std };
  }
  return null;
}
var pct = (v) => `${(v * 100).toFixed(1)}%`;
function belowSkillFloor(valAcc, classes) {
  if (!classes || classes <= 1) return false;
  const chance = 1 / classes;
  return (valAcc - chance) / (1 - chance) < PLATEAU_SKILL_FLOOR;
}
function tailOf(curve) {
  if (curve.length < 4) return [];
  return curve.slice(-Math.max(2, Math.round(curve.length / 3)));
}
function diagnoseRun(ev) {
  if (!ev.measured) return [];
  const out = [];
  const last = ev.curve.length > 0 ? ev.curve[ev.curve.length - 1] : ev.final && (typeof ev.final.valAcc === "number" || typeof ev.final.valLoss === "number") ? {
    epoch: 0,
    totalEpochs: 0,
    lr: Number.NaN,
    epochMs: 0,
    trainAcc: ev.final.trainAcc ?? Number.NaN,
    valAcc: ev.final.valAcc ?? Number.NaN,
    trainLoss: ev.final.trainLoss ?? Number.NaN,
    valLoss: ev.final.valLoss ?? Number.NaN
  } : null;
  const contract = deriveModelContract(ev.model);
  const dataset = knownDatasetFor(ev.source);
  const mismatches = contractMismatches(ev.model, ev.source);
  if (mismatches.length > 0) {
    out.push({
      kind: "contract-mismatch",
      title: "The graph does not match the dataset it trained on",
      evidence: mismatches.map((m) => m.title).join(". ") + ".",
      fix: mismatches[0].fix,
      agentFixable: true,
      explainsEverything: true
    });
    return out;
  }
  const trainSeries = ev.curve.map((e) => e.trainLoss);
  const valSeries = ev.curve.map((e) => e.valLoss);
  const nanAtTrain = firstNonFinite(trainSeries);
  const nanAtVal = firstNonFinite(valSeries);
  const nanAt = nanAtTrain < 0 ? nanAtVal : nanAtVal < 0 ? nanAtTrain : Math.min(nanAtTrain, nanAtVal);
  if (ev.curve.length > 0 && nanAt >= 0) {
    const at = ev.curve[nanAt];
    const before = nanAt > 0 ? ev.curve[nanAt - 1] : null;
    const bad = nanAt === nanAtTrain ? at.trainLoss : at.valLoss;
    out.push({
      kind: "nan-loss",
      // A stored run went through JSON, where NaN and Infinity both become
      // null, so only a live curve can say which of the two it was.
      title: `The loss became ${typeof bad !== "number" ? "non-finite (NaN or infinite)" : Number.isNaN(bad) ? "NaN" : "infinite"} at epoch ${at.epoch}`,
      evidence: `The ${nanAt === nanAtTrain ? "training" : "validation"} loss is not a finite number from epoch ${at.epoch}` + (before && Number.isFinite(before.trainLoss) ? `; the epoch before it was ${before.trainLoss.toFixed(4)}.` : ", the first point logged.") + " The curve is logged per epoch, so the step inside that epoch is not known.",
      fix: `Lower the learning rate. It ran at ${at.lr ?? "the configured rate"}; a tenth of that is the usual first try.`,
      agentFixable: false,
      explainsEverything: true
    });
    return out;
  }
  const climbing = ev.curve.length >= 3 && ev.curve.every((e, i) => i === 0 || e.trainLoss > ev.curve[i - 1].trainLoss) && last.trainLoss > ev.curve[0].trainLoss * DIVERGENCE_RISE;
  if (climbing) {
    out.push({
      kind: "diverged",
      title: "Training diverged",
      evidence: `Training loss rose every epoch, ${ev.curve[0].trainLoss.toFixed(4)} to ${last.trainLoss.toFixed(4)}.`,
      fix: `Lower the learning rate. It ran at ${last?.lr ?? "the configured rate"}; a tenth of that is the usual first try.`,
      agentFixable: false,
      explainsEverything: true
    });
    return out;
  }
  const spike = findLossSpike(trainSeries);
  if (spike) {
    const at = ev.curve[spike.index];
    out.push({
      kind: "loss-spike",
      title: `The training loss spiked at epoch ${at.epoch}`,
      evidence: `Training loss ${spike.value.toFixed(4)} at epoch ${at.epoch}, against a mean of ${spike.mean.toFixed(4)} (standard deviation ${spike.std.toFixed(4)}) over the ${Math.min(SPIKE_WINDOW, spike.index)} epochs before it: more than ${SPIKE_SIGMAS} standard deviations above. Per-epoch points, so this is an approximation of a per-step monitor.`,
      fix: "Add learning-rate warmup or clip the gradient norm; a spike that recurs at a lower rate points at the architecture instead.",
      agentFixable: false
    });
  }
  const init = expectedInitialLoss(ev.model, ev.source);
  const first2 = ev.curve[0];
  if (init && first2 && Number.isFinite(first2.trainLoss) && first2.trainLoss > init.value * FIRST_LOSS_RATIO) {
    out.push({
      kind: "first-loss-high",
      title: "The first loss was far above what guessing scores",
      evidence: `The first logged training loss was ${first2.trainLoss.toFixed(4)}; an uninformative head scores ${init.sentence}, and this is ${(first2.trainLoss / init.value).toFixed(1)} times that after a whole epoch of training.`,
      fix: "Check the input scale and the loss inputs: cross-entropy expects raw logits and integer class labels, and very large initial logits put the loss here.",
      agentFixable: false
    });
  }
  const classes = dataset?.classes ?? (typeof contract.outputDim === "number" ? contract.outputDim : null);
  if (last && classes && classes > 1) {
    const chance = 1 / classes;
    if (Number.isFinite(last.valAcc) && last.valAcc <= chance + AT_CHANCE_TOLERANCE) {
      out.push({
        kind: "at-chance",
        title: "The model is at chance level",
        evidence: `Validation accuracy ${pct(last.valAcc)} against ${pct(chance)} for guessing among ${classes} classes.`,
        fix: "Nothing was learned, so more epochs and more capacity are both the wrong lever. Check that labels reach the loss correctly and that the input is normalised the way the data expects.",
        agentFixable: false
      });
    }
  }
  const tail = tailOf(ev.curve);
  if (tail.length >= 2 && last && !out.some((d) => d.kind === "at-chance")) {
    const move = tail[tail.length - 1].valAcc - tail[0].valAcc;
    if (move > FLAT_EPSILON) {
      out.push({
        kind: "still-improving",
        title: "The run stopped while it was still learning",
        evidence: `Validation accuracy rose ${pct(move)} over the last ${tail.length} epochs (${pct(tail[0].valAcc)} to ${pct(last.valAcc)}) and had not levelled off.`,
        fix: `Train longer. At that rate the next ${tail.length} epochs are worth roughly ${pct(move)} more.`,
        agentFixable: false
      });
    } else if (move <= FLAT_EPSILON && belowSkillFloor(last.valAcc, classes)) {
      out.push({
        kind: "plateaued",
        title: "The run had stopped learning before it ended",
        evidence: `Validation accuracy moved ${pct(Math.abs(move))} over the last ${tail.length} epochs (${pct(tail[0].valAcc)} to ${pct(last.valAcc)}).`,
        // The explicit negative is the point. "Train longer" was the advice
        // this product gave for exactly this curve.
        fix: "Training longer will not help; this is the ceiling of this graph on this data. Change the architecture or the data, not the schedule.",
        agentFixable: true
      });
    }
  }
  if (last && Number.isFinite(last.trainAcc) && Number.isFinite(last.valAcc)) {
    const gap = last.trainAcc - last.valAcc;
    if (gap > OVERFIT_GAP) {
      out.push({
        kind: "overfit",
        title: "Overfit",
        evidence: `Train accuracy ${pct(last.trainAcc)} against validation ${pct(last.valAcc)}, a gap of ${pct(gap)}.`,
        fix: "Add dropout or weight decay, or train on more data. The model has memorised what it saw.",
        agentFixable: true
      });
    }
  }
  const params = estimateParams(ev.model);
  if (ev.datasetSize && params && ev.datasetSize / params < ROWS_PER_PARAM_FLOOR) {
    out.push({
      kind: "tiny-data",
      title: "There is far more model than data",
      evidence: `${ev.datasetSize.toLocaleString()} training rows against roughly ${params.toLocaleString()} parameters.`,
      fix: "More data, or a much smaller model. At this ratio the architecture is not what is limiting the result.",
      agentFixable: false
    });
  }
  return out;
}
function estimateParams(model) {
  const total = (model.components ?? []).reduce(
    (s, c) => s + estimateLayerParams(c.type, c.params, c.inputShape ?? []),
    0
  );
  return total > 0 ? total : null;
}

// src/utils/trainStability.ts
var ATTENTION = /attention|^transformerBlock$|transformerEncoder|transformerDecoder|^mla$|^ditBlock$|^hstuBlock$/i;
var QK_NORM_INFO_WIDTH = 1024;
var QK_NORM_WARN_WIDTH = 4096;
var Z_LOSS_VOCAB = 32e3;
var PAPER = "Wortsman et al., arXiv 2309.14322";
function widthOf(c, inShape) {
  const p = c.params ?? {};
  const v = Number(p.embedDim ?? p.hiddenDim ?? p.dModel ?? p.hiddenSize);
  if (Number.isFinite(v) && v > 0) return v;
  return inShape && inShape.length ? inShape[inShape.length - 1] : 0;
}
function hasQkNorm(c) {
  const p = c.params ?? {};
  return Boolean(p.qkNorm ?? p.qk_norm ?? p.qkLayerNorm);
}
function stabilityFindings(model) {
  const out = [];
  const comps = model.components ?? [];
  const shapes = propagateShapes(model).shapes;
  const graphHasQkNorm = comps.some((c) => c.type === "qkNorm");
  const attn = comps.filter((c) => ATTENTION.test(c.type) && c.type !== "qkNorm" && !hasQkNorm(c));
  if (!graphHasQkNorm && attn.length) {
    const wide = attn.map((c) => ({ c, d: widthOf(c, shapes.get(c.id)?.in) })).filter((x) => x.d >= QK_NORM_INFO_WIDTH);
    if (wide.length) {
      const d = Math.max(...wide.map((x) => x.d));
      out.push({
        ruleId: "attn-no-qk-norm",
        severity: d >= QK_NORM_WARN_WIDTH ? "warn" : "info",
        title: `${wide.length} attention layer${wide.length === 1 ? "" : "s"} at width ${d.toLocaleString("en-US")} without QK-norm`,
        detail: `Attention-logit growth is one of the two instabilities ${PAPER} reproduce at small scale, and it gets more likely as width and learning rate grow; normalising q and k removed it in their runs. Shown from width ${QK_NORM_INFO_WIDTH.toLocaleString("en-US")}, a warning from ${QK_NORM_WARN_WIDTH.toLocaleString("en-US")}. Many models train stably without it at a tuned learning rate.`,
        fix: "Add a qkNorm node (RMSNorm on q and k per head) to the attention blocks.",
        agentFixable: true,
        affectedIds: wide.map((x) => x.c.id)
      });
    }
  }
  const contract = deriveModelContract(model);
  const V = contract.outputDim ?? 0;
  if (contract.taskType === "language-modeling" && V >= Z_LOSS_VOCAB) {
    out.push({
      ruleId: "softmax-no-z-loss",
      severity: "info",
      title: `A ${V.toLocaleString("en-US")}-way output softmax with no z-loss`,
      detail: `Output logits drifting away from log-probabilities is the second instability in ${PAPER}; an auxiliary z-loss (1e-4 x log^2 Z) removed it. The graph has no loss node, so this is a line in the training loop, not a layer.`,
      fix: "Add z-loss with coefficient 1e-4 to the training loss.",
      agentFixable: false,
      affectedIds: []
    });
  }
  return out;
}

// src/utils/datasetMetaChecks.ts
var IMBALANCE_FRACTION_OF_UNIFORM = 0.1;
var CHARS_PER_TOKEN = 4;
var LABEL_DRIFT_LINF = 0.1;
var TRUNCATION_WARN_SHARE = 0.1;
var fmt = (n2) => n2.toLocaleString("en-US");
var pct2 = (v, digits = 0) => `${(v * 100).toFixed(digits)}%`;
function modelMaxSeqLen(model) {
  let best = null;
  for (const c of model.components) {
    const p = c.params ?? {};
    for (const k of ["maxLen", "maxSeqLen", "maxPositionEmbeddings", "contextLength", "blockSize", "seqLen"]) {
      const v = p[k];
      if (typeof v === "number" && Number.isFinite(v) && v > 0) best = best == null ? v : Math.min(best, v);
    }
  }
  if (best != null) return best;
  const input = model.components.find((c) => c.type === "input");
  const shape = input?.params?.shape ?? [];
  const last = shape[shape.length - 1];
  return typeof last === "number" && last > 1 ? last : null;
}
function modelEmbeddingVocab(model) {
  const emb = model.components.find((c) => /embedding/i.test(c.type) && !/positional/i.test(c.type));
  const p = emb?.params ?? {};
  const v = p.numEmbeddings ?? p.vocabSize ?? p.vocab_size;
  return typeof v === "number" && v > 0 ? v : null;
}
function shareAbove(h, limit) {
  const total = h.hist.reduce((a, b) => a + b, 0);
  if (total <= 0) return 0;
  let above = 0;
  for (let i = 0; i < h.hist.length; i++) {
    const lo = h.edges[i];
    const hi = h.edges[i + 1] ?? lo;
    if (lo >= limit) above += h.hist[i];
    else if (hi > limit && hi > lo) above += h.hist[i] * ((hi - limit) / (hi - lo));
  }
  return Math.min(1, above / total);
}
var MODALITY_FOR = {
  image: ["vision"],
  audio: ["audio"],
  text: ["nlp"],
  tokens: ["nlp"],
  tabular: ["tabular"]
};
function trainAndEval(meta) {
  const train = meta.stats.find((s) => s.split.toLowerCase() === "train") ?? null;
  const evals = meta.stats.filter((s) => s !== train);
  return { train, evals };
}
function datasetMetaFindings(model, meta, ctx) {
  const out = [];
  const name = meta.datasetId + (meta.config && meta.config !== "default" ? ` (${meta.config})` : "");
  const { train, evals } = trainAndEval(meta);
  if (meta.inputKind && !MODALITY_FOR[meta.inputKind].includes(ctx.modality)) {
    out.push({
      severity: "warn",
      ruleId: "dataset-modality",
      title: `${name} is ${meta.inputKind === "tokens" ? "tokenized text" : meta.inputKind} data; the graph reads as ${ctx.modality}`,
      detail: `The dataset's input column "${meta.inputColumn ?? "?"}" is ${meta.inputKind}, and the graph's input stem looks like a ${ctx.modality} model. A loader would have to turn one into the other, and nothing in the generated code does that.`,
      fix: "Pick a dataset whose input matches the graph, or change the input stem.",
      // Could be either side; changing the dataset is not a graph edit, and
      // rebuilding a stem is not a one-finding fix. Not offered to the agent.
      agentFixable: false
    });
  }
  const outDim = ctx.contract.outputDim;
  if (!ctx.headAlreadyChecked && ctx.contract.taskType === "classification" && meta.numClasses && meta.numClasses > 1 && typeof outDim === "number" && outDim > 0 && outDim !== meta.numClasses) {
    const from = meta.classNames?.length ? `its ClassLabel names` : "the labels its train split contains";
    out.push({
      severity: "block",
      ruleId: "dataset-output-classes",
      title: `Head is ${outDim} wide, ${name} has ${meta.numClasses} classes`,
      detail: `The class count comes from ${from}, read from the Hugging Face dataset viewer. ` + (outDim > meta.numClasses ? `${outDim - meta.numClasses} outputs would be trained toward classes that never appear.` : `Labels above ${outDim - 1} cannot be represented at all, and cross-entropy fails on the first one.`),
      fix: `Set the final layer's output width to ${meta.numClasses}.`,
      agentFixable: true
    });
  }
  const shape = ctx.contract.inputShape;
  if (meta.imageSize && shape.length === 3) {
    const [, h, w] = shape;
    const up = Math.max(h / meta.imageSize.height, w / meta.imageSize.width);
    const down = Math.max(meta.imageSize.height / h, meta.imageSize.width / w);
    const sizeLine = `The graph expects ${h}x${w}; the median image in the first ${meta.imageSize.rows} rows is ${meta.imageSize.height}x${meta.imageSize.width}.`;
    if (up >= 4) {
      out.push({
        severity: "block",
        ruleId: "dataset-image-size",
        title: `Images would be upscaled ${up.toFixed(0)}x to fit the graph`,
        detail: `${sizeLine} Upscaling adds no information and multiplies compute; this is the shape of the 2026-09-05 MNIST run that trained a 224x224 graph on 28x28 digits.`,
        fix: `Set the input layer's spatial size to ${meta.imageSize.height}x${meta.imageSize.width}.`,
        agentFixable: true
      });
    } else if (down >= 4) {
      out.push({
        severity: "warn",
        ruleId: "dataset-image-size",
        title: `Images would be downscaled ${down.toFixed(0)}x to fit the graph`,
        detail: `${sizeLine} Detail lost at that factor does not come back.`,
        fix: "Raise the input resolution if fine detail matters for the task.",
        agentFixable: true
      });
    } else if (up > 1.01 || down > 1.01) {
      out.push({ severity: "info", ruleId: "dataset-image-size", title: "Images will be resized to the graph input", detail: sizeLine, agentFixable: false });
    }
  }
  const vocab = modelEmbeddingVocab(model);
  if (meta.maxTokenId && vocab) {
    if (meta.maxTokenId.value >= vocab) {
      out.push({
        severity: "block",
        ruleId: "dataset-vocab",
        title: `Token id ${fmt(meta.maxTokenId.value)} is past the ${fmt(vocab)}-row embedding`,
        detail: `The first ${meta.maxTokenId.rows} rows of "${meta.inputColumn}" already contain id ${fmt(meta.maxTokenId.value)}. The embedding lookup indexes out of range on that batch and the run stops there.`,
        fix: `Set the embedding's vocabulary size to at least ${fmt(meta.maxTokenId.value + 1)}, or use the tokenizer the dataset was built with.`,
        agentFixable: true
      });
    } else {
      out.push({
        severity: "info",
        ruleId: "dataset-vocab",
        title: "Token ids in the first rows fit the embedding",
        detail: `Largest id in the first ${meta.maxTokenId.rows} rows is ${fmt(meta.maxTokenId.value)}, under the ${fmt(vocab)}-row table. A sample, not a proof about the whole split.`,
        agentFixable: false
      });
    }
  }
  if (train) {
    const noLabel = Math.max(train.labelNoLabelProportion ?? 0, train.labelNanProportion ?? 0);
    if (noLabel > 0) {
      out.push({
        severity: "block",
        ruleId: "dataset-empty-labels",
        title: `${pct2(noLabel, noLabel < 0.01 ? 2 : 0)} of train rows have no label`,
        detail: `The "${meta.labelColumn}" column of the train split is empty or -1 on those rows. Cross-entropy refuses a -1 target, so the run fails on the first batch that holds one.`,
        fix: "Filter unlabeled rows in data.py, or set ignore_index on the loss if it is semi-supervised on purpose.",
        agentFixable: false
      });
    }
    if ((train.inputNanProportion ?? 0) > 0) {
      out.push({
        severity: "warn",
        ruleId: "dataset-empty-inputs",
        title: `${pct2(train.inputNanProportion, train.inputNanProportion < 0.01 ? 2 : 0)} of train inputs are empty`,
        detail: `"${meta.inputColumn}" is null on those rows. The loader will either crash on them or feed an empty sample.`,
        fix: "Drop or impute the empty rows in data.py.",
        agentFixable: false
      });
    }
  }
  const freq = train?.labelFrequencies;
  if (freq) {
    const entries = Object.entries(freq).filter(([, n2]) => n2 > 0);
    const total = entries.reduce((a, [, n2]) => a + n2, 0);
    const K = entries.length;
    if (K > 1 && total > 0) {
      const [rareName, rareN] = entries.reduce((a, b) => b[1] < a[1] ? b : a);
      const pMin = rareN / total;
      const line = IMBALANCE_FRACTION_OF_UNIFORM / K;
      if (pMin < line) {
        out.push({
          severity: "warn",
          ruleId: "dataset-imbalance",
          title: `Class "${rareName}" is ${pct2(pMin, 2)} of train, under a tenth of an even share`,
          detail: `${K} classes, so an even share is ${pct2(1 / K, 1)}; "${rareName}" has ${fmt(rareN)} of ${fmt(total)} rows. Rule: rarest class under 10% of 1/K, from cleanlab's imbalance check. Plain accuracy will mostly measure the common classes.`,
          fix: "Weight the loss by class, or resample; read macro-F1 or balanced accuracy, not accuracy.",
          agentFixable: false
        });
      }
    }
  }
  if (train?.labelFrequencies) {
    const trainLabels = new Set(Object.entries(train.labelFrequencies).filter(([, n2]) => n2 > 0).map(([k]) => k));
    for (const ev of evals) {
      if (!ev.labelFrequencies) continue;
      if ((ev.labelNoLabelProportion ?? 0) >= 0.999) {
        out.push({
          severity: "info",
          ruleId: "dataset-unlabeled-split",
          title: `The ${ev.split} split carries no labels`,
          detail: `Every "${meta.labelColumn}" value in ${ev.split} is empty, as on a benchmark that keeps its test labels private. Evaluate on another split.`,
          agentFixable: false
        });
        continue;
      }
      const evLabels = Object.entries(ev.labelFrequencies).filter(([, n2]) => n2 > 0).map(([k]) => k);
      const unseen = evLabels.filter((l) => !trainLabels.has(l));
      const missing = [...trainLabels].filter((l) => !evLabels.includes(l));
      if (unseen.length) {
        out.push({
          severity: "block",
          ruleId: "dataset-unseen-labels",
          title: `${ev.split} has ${unseen.length} label${unseen.length > 1 ? "s" : ""} train never shows`,
          detail: `${unseen.slice(0, 5).map((l) => `"${l}"`).join(", ")}${unseen.length > 5 ? ", and more" : ""} appear in ${ev.split} and not in train. The model is graded on classes it was never taught, so every one of those rows is a guaranteed miss.`,
          fix: "Re-split so every class appears in train, or drop those classes from the evaluation split.",
          agentFixable: false
        });
      }
      if (!unseen.length && !missing.length) {
        const drift = labelShareDrift(train.labelFrequencies, ev.labelFrequencies);
        if (drift && drift.linf > LABEL_DRIFT_LINF) {
          out.push({
            severity: "warn",
            ruleId: "dataset-label-drift",
            title: `Class shares differ between train and ${ev.split} by up to ${Math.round(drift.linf * 100)} percentage points`,
            detail: `"${drift.label}" is ${pct2(drift.trainShare, 1)} of train and ${pct2(drift.evalShare, 1)} of ${ev.split}. L-infinity distance between the two splits' class shares (the categorical measure TFDV uses); the ${Math.round(LABEL_DRIFT_LINF * 100)}-point line is ours. Accuracy on ${ev.split} then weights the classes differently from the data the model learned, so a train-to-${ev.split} gap is partly the split, not the model.`,
            fix: "Re-split with stratification by label, or read per-class recall and balanced accuracy rather than accuracy.",
            agentFixable: false
          });
        }
      }
      if (missing.length) {
        out.push({
          severity: "warn",
          ruleId: "dataset-missing-labels",
          title: `${ev.split} is missing ${missing.length} class${missing.length > 1 ? "es" : ""} train has`,
          detail: `${missing.slice(0, 5).map((l) => `"${l}"`).join(", ")}${missing.length > 5 ? ", and more" : ""} never appear in ${ev.split}, so per-class recall for them is undefined.`,
          agentFixable: false
        });
      }
    }
  }
  const seqLen = modelMaxSeqLen(model);
  const text = train?.textLength;
  if (text && seqLen && meta.inputKind === "text") {
    const limit = seqLen * CHARS_PER_TOKEN;
    const share = shareAbove(text, limit);
    const basis = `The graph takes ${fmt(seqLen)} tokens, about ${fmt(limit)} characters at ${CHARS_PER_TOKEN} characters per token (an assumption, not a tokenizer run). Train texts run ${fmt(text.min)} to ${fmt(text.max)} characters${text.median != null ? `, median ${fmt(text.median)}` : ""}.`;
    if (share >= TRUNCATION_WARN_SHARE) {
      out.push({
        severity: "warn",
        ruleId: "dataset-truncation",
        title: `About ${pct2(share)} of train texts would be cut off (approximate)`,
        detail: `${basis} Truncated rows lose their ending, which is often where a review's verdict is. The 10% line is ours.`,
        fix: `Raise the sequence length toward ${fmt(Math.round((text.median ?? limit) / CHARS_PER_TOKEN) * 2)} tokens, or chunk long inputs.`,
        agentFixable: true
      });
    } else {
      out.push({
        severity: "info",
        ruleId: "dataset-truncation",
        title: share > 0 ? `About ${pct2(share, 1)} of train texts would be cut off (approximate)` : "No train text is longer than the graph accepts (approximate)",
        detail: basis,
        agentFixable: false
      });
    }
  }
  const partial = meta.stats.filter((s) => s.partial).map((s) => s.split);
  if (partial.length) {
    out.push({
      severity: "info",
      ruleId: "dataset-partial-stats",
      title: "Statistics cover the first 5 GB only",
      detail: `The dataset viewer computed ${partial.join(", ")} statistics on the first 5 GB, which is not a random sample. Counts above describe that slice.`,
      agentFixable: false
    });
  }
  return out;
}
function labelShareDrift(a, b) {
  const total = (f) => Object.values(f).reduce((x, n2) => x + (n2 > 0 ? n2 : 0), 0);
  const ta = total(a), tb = total(b);
  if (ta <= 0 || tb <= 0) return null;
  let best = null;
  for (const label of /* @__PURE__ */ new Set([...Object.keys(a), ...Object.keys(b)])) {
    const pa = Math.max(0, a[label] ?? 0) / ta;
    const pb = Math.max(0, b[label] ?? 0) / tb;
    const d = Math.abs(pa - pb);
    if (!best || d > best.linf) best = { linf: d, label, trainShare: pa, evalShare: pb };
  }
  return best;
}
function describeMeta(meta) {
  const parts = [];
  if (meta.inputColumn) parts.push(`input "${meta.inputColumn}" (${meta.inputKind})`);
  if (meta.labelColumn) parts.push(`label "${meta.labelColumn}"${meta.numClasses ? `, ${meta.numClasses} classes` : ""}`);
  const sized = meta.splits.filter((s) => s.numExamples != null);
  if (sized.length) parts.push(`splits ${sized.map((s) => `${s.name} ${fmt(s.numExamples)}`).join(", ")}`);
  if (meta.config && meta.otherConfigs?.length) parts.push(`config "${meta.config}" of ${meta.otherConfigs.length + 1}`);
  return parts.join("; ") + ".";
}

// src/utils/evalStats.ts
var Z95 = 1.959964;
var MDE_FACTOR = 1.959964 + 0.841621;
function wilsonInterval(p, n2, z = Z95) {
  if (!(n2 > 0) || !Number.isFinite(p)) return { lo: 0, hi: 1 };
  const q = Math.min(1, Math.max(0, p));
  const z2 = z * z;
  const denom = 1 + z2 / n2;
  const centre = (q + z2 / (2 * n2)) / denom;
  const half = z * Math.sqrt(q * (1 - q) / n2 + z2 / (4 * n2 * n2)) / denom;
  return { lo: Math.max(0, centre - half), hi: Math.min(1, centre + half) };
}
function minimumDetectableDifference(p, n2) {
  if (!(n2 > 0)) return 1;
  const q = Math.min(1, Math.max(0, p));
  const variance = Math.max(q * (1 - q), 1 / n2 * (1 - 1 / n2));
  return MDE_FACTOR * Math.sqrt(2 * variance / n2);
}
function within(i, value) {
  return value >= i.lo && value <= i.hi;
}
var pts = (v, digits = 1) => `${(v * 100).toFixed(digits)}`;
function evalSampleCount(evalMetrics) {
  if (!evalMetrics) return null;
  for (const k of ["n_samples", "num_samples", "n", "n_test", "samples"]) {
    const v = evalMetrics[k];
    if (typeof v === "number" && Number.isFinite(v) && v >= 1) return Math.round(v);
  }
  return null;
}
function pickHeadlineMetric(evalMetrics, opts) {
  if (!evalMetrics) return null;
  const has = (k) => typeof evalMetrics[k] === "number" && Number.isFinite(evalMetrics[k]);
  if (opts.imbalanced) {
    if (has("balanced_accuracy")) return { key: "balanced_accuracy", label: "balanced accuracy", value: evalMetrics.balanced_accuracy, because: "the classes are imbalanced, so every class counts equally" };
    if (has("f1_macro")) return { key: "f1_macro", label: "macro-F1", value: evalMetrics.f1_macro, because: "the classes are imbalanced, so every class counts equally" };
    if (has("recall_macro")) return { key: "recall_macro", label: "macro recall (balanced accuracy)", value: evalMetrics.recall_macro, because: "the classes are imbalanced; macro recall is balanced accuracy" };
  }
  if (has("accuracy")) return { key: "accuracy", label: "accuracy", value: evalMetrics.accuracy, because: opts.imbalanced ? "only accuracy was reported" : "the classes are balanced enough for accuracy to mean what it says" };
  if (has("mae")) return { key: "mae", label: "MAE", value: evalMetrics.mae, because: "a regression task" };
  if (has("perplexity")) return { key: "perplexity", label: "perplexity", value: evalMetrics.perplexity, because: "a language model" };
  if (has("mean_iou")) return { key: "mean_iou", label: "mean IoU", value: evalMetrics.mean_iou, because: "a segmentation task" };
  return null;
}

// src/utils/evalStore.ts
var PREFIX = "neurarch-eval-suite-v1:";
function modelIdFor(arch) {
  const raw = (arch.name ?? "untitled").trim().toLowerCase();
  const slug = raw.replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return slug || "untitled";
}
function loadSuite(arch) {
  const id = modelIdFor(arch);
  try {
    const raw = localStorage.getItem(PREFIX + id);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && Array.isArray(parsed.cases) && Array.isArray(parsed.lastResults)) {
        return parsed;
      }
    }
  } catch {
  }
  return {
    modelId: id,
    modelName: arch.name ?? "Untitled",
    cases: [],
    lastResults: [],
    updatedAt: Date.now()
  };
}
function summarize(suite) {
  const total = suite.cases.length;
  const byCase = new Map(suite.lastResults.map((r) => [r.caseId, r]));
  let passed = 0, failed = 0, pending = 0, errors = 0;
  let latencySum = 0, latencyN = 0;
  let lastRan = 0;
  for (const c of suite.cases) {
    const r = byCase.get(c.id);
    if (!r || r.status === "pending") pending++;
    else if (r.status === "pass") {
      passed++;
      if (r.latencyMs) {
        latencySum += r.latencyMs;
        latencyN++;
      }
    } else if (r.status === "fail") failed++;
    else errors++;
    if (r?.ranAt && r.ranAt > lastRan) lastRan = r.ranAt;
  }
  return {
    total,
    passed,
    failed,
    pending,
    errors,
    avgLatencyMs: latencyN > 0 ? latencySum / latencyN : 0,
    ranAt: lastRan || null
  };
}

// src/utils/llmAnalyzer.ts
var LLM_GPU_IDS = ["h100-sxm", "a100-80g", "a100-40g", "rtx-4090", "rtx-3090", "t4", "m3-max", "m3-pro"];
var LLM_GPUS = LLM_GPU_IDS.map((id) => {
  const hw = GPU_HARDWARE.find((g) => g.id === id);
  return {
    name: hw.label,
    vramBytes: hw.memoryGB * 1e9,
    fp16Tflops: hw.bf16TFLOPS,
    memBandwidthGBs: hw.memBandwidthGBs,
    tier: hw.tier,
    hw
  };
});
var TRANSFORMER_TYPES = /* @__PURE__ */ new Set([
  "transformerBlock",
  "multiHeadAttention",
  "selfAttention",
  "attention",
  "groupedQueryAttention",
  "causalAttention",
  "localAttention",
  "mla"
]);
function detectLLMShape(model) {
  const blocks = model.components.filter((c) => TRANSFORMER_TYPES.has(c.type));
  if (blocks.length === 0) return null;
  const ref = blocks.find((c) => c.params?.numKVHeads !== void 0) ?? blocks[0];
  const embedDim = Number(
    ref.params?.embedDim ?? ref.params?.hiddenDim ?? ref.params?.dModel ?? 512
  );
  const numHeads = Number(ref.params?.numHeads ?? ref.params?.nHead ?? 8);
  const numKVHeads = Math.max(1, Math.min(numHeads, Number(ref.params?.numKVHeads ?? numHeads)));
  const headDim = embedDim > 0 && numHeads > 0 ? embedDim / numHeads : 64;
  const ffDim = Number(ref.params?.ffDim ?? ref.params?.ffnDim ?? embedDim * 4);
  const windowLayer = model.components.find((c) => c.type === "localAttention" && Number(c.params?.windowSize) > 0) ?? blocks.find((c) => Number(c.params?.windowSize) > 0);
  const windowSize = windowLayer ? Number(windowLayer.params?.windowSize) : void 0;
  const ropeLayer = model.components.find(
    (c) => c.type === "rope" && typeof c.params?.scalingType === "string" && c.params.scalingType !== "none"
  );
  const ropeScaling = ropeLayer ? {
    type: String(ropeLayer.params?.scalingType),
    factor: Math.max(1, Number(ropeLayer.params?.scalingFactor ?? 1)),
    originalMaxPos: Math.max(1, Number(ropeLayer.params?.originalMaxPos ?? 2048))
  } : void 0;
  const flashAttention = blocks.some((c) => c.params?.flashAttention === true);
  const mlaLayer = blocks.find((c) => c.type === "mla");
  const mlaLatentDim = mlaLayer ? Math.max(1, Number(mlaLayer.params?.kvLatentDim ?? 128)) : void 0;
  const mlaRopeDim = mlaLayer && Number(mlaLayer.params?.ropeHeadDim) > 0 ? Number(mlaLayer.params?.ropeHeadDim) : void 0;
  let denseParams = 0;
  let moeTotal = 0;
  let moeActivated = 0;
  let moe;
  for (const c of model.components) {
    if (c.type === "transformerBlock") {
      const d = Number(c.params?.embedDim ?? embedDim);
      const ff = Number(c.params?.ffDim ?? d * 4);
      denseParams += 4 * d * d + 4 * d + d * ff + ff + ff * d + d + 4 * d;
    } else if (c.type === "mla") {
      const d = Number(c.params?.embedDim ?? embedDim);
      const kv = Number(c.params?.kvLatentDim ?? 128);
      denseParams += 2 * d * d + 2 * d * kv;
    } else if (TRANSFORMER_TYPES.has(c.type)) {
      const d = Number(c.params?.embedDim ?? embedDim);
      denseParams += 4 * d * d + 4 * d;
    } else if (c.type === "sharedExpertMoE" || c.type === "moeLayer") {
      const d = Number(c.params?.embedDim ?? embedDim);
      const E = Math.max(1, Number(c.params?.numExperts ?? (c.type === "sharedExpertMoE" ? 64 : 8)));
      const S = c.type === "sharedExpertMoE" ? Math.max(0, Number(c.params?.numSharedExperts ?? 2)) : 0;
      const I = Number(c.params?.expertDim ?? c.params?.ffDim ?? Math.round(d * 8 / 3));
      const K = Math.max(1, Math.min(E, Number(c.params?.topK ?? (c.type === "sharedExpertMoE" ? 6 : 2))));
      const expertParams = 3 * d * I;
      moeTotal += d * E + (E + S) * expertParams;
      moeActivated += d * E + (K + S) * expertParams;
      moe = { layers: (moe?.layers ?? 0) + 1, numExperts: E, topK: K, numShared: S, expertDim: I };
    } else if (c.type === "linear") {
      const inF = Number(c.params?.inFeatures ?? embedDim);
      const outF = Number(c.params?.outFeatures ?? embedDim);
      if (inF > 0 && outF > 0) denseParams += inF * outF + outF;
    } else if (c.type === "embedding") {
      const V = Number(c.params?.vocabSize ?? c.params?.numEmbeddings ?? 32e3);
      const D = Number(c.params?.embeddingDim ?? embedDim);
      if (V > 0 && D > 0) denseParams += V * D;
    } else if (c.type === "layerNorm" || c.type === "batchNorm") {
      const f = Number(c.params?.normalizedShape ?? c.params?.numFeatures ?? embedDim);
      denseParams += 2 * f;
    }
  }
  const fallbackDense = blocks.length * (4 * embedDim * embedDim + 2 * embedDim * ffDim);
  const totalParams2 = (denseParams || fallbackDense) + moeTotal;
  const activatedParams = (denseParams || fallbackDense) + moeActivated;
  const hasLM = model.components.some((c) => c.type === "embedding");
  return {
    numLayers: blocks.length,
    embedDim,
    numHeads,
    numKVHeads,
    headDim,
    ffDim,
    totalParams: totalParams2,
    activatedParams,
    moe,
    hasLM,
    windowSize,
    ropeScaling,
    flashAttention,
    mlaLatentDim,
    mlaRopeDim
  };
}
function effectiveKvPositions(shape, seqLen) {
  return shape.windowSize && shape.windowSize > 0 ? Math.min(seqLen, shape.windowSize) : seqLen;
}
function kvBytesPerToken(shape, bytesPerParam) {
  return kvCachePerTokenBytes({
    numLayers: shape.numLayers,
    bytesPerValue: bytesPerParam,
    kvHeads: shape.numKVHeads,
    headDim: shape.headDim,
    mlaLatentDim: shape.mlaLatentDim,
    mlaRopeDim: shape.mlaRopeDim
  });
}

// src/utils/deployChecks.ts
var FAMILY_OF = {
  conv1d: "conv12d",
  conv2d: "conv12d",
  depthwiseConv2d: "conv12d",
  separableConv2d: "conv12d",
  dilatedConv2d: "conv12d",
  depthwiseConv1d: "conv12d",
  audioConv: "conv12d",
  patchEmbed: "conv12d",
  conv3d: "conv3d",
  causalConv3d: "conv3d",
  tubeletEmbed: "conv3d",
  transposeConv2d: "convTranspose",
  linear: "linear",
  projection: "linear",
  lmHead: "linear",
  feedForward: "linear",
  swiglu: "linear",
  geglu: "linear",
  glu: "linear",
  batchNorm: "batchNorm",
  layerNorm: "layerNorm",
  qkNorm: "layerNorm",
  groupNorm: "groupNorm",
  instanceNorm: "instanceNorm",
  rmsNorm: "rmsNorm",
  relu: "activation",
  relu6: "activation",
  leakyRelu: "activation",
  prelu: "activation",
  gelu: "activation",
  swish: "activation",
  sigmoid: "activation",
  tanh: "activation",
  softplus: "activation",
  hardSigmoid: "activation",
  selu: "activation",
  elu: "elu",
  hardSwish: "hardSwish",
  mish: "mish",
  softmax: "softmax",
  logSoftmax: "softmax",
  maxpool2d: "maxPool2d",
  adaptiveMaxPool2d: "maxPool2d",
  avgpool2d: "avgPool2d",
  adaptiveAvgPool2d: "avgPool2d",
  maxpool1d: "pool1d",
  avgpool1d: "pool1d",
  maxpool3d: "pool3d",
  avgpool3d: "pool3d",
  globalAvgPool2d: "globalPool",
  globalMaxPool2d: "globalPool",
  globalAvgPool1d: "globalPool",
  mean: "globalPool",
  dropout: "identity",
  dropPath: "identity",
  flatten: "structural",
  reshape: "structural",
  residual: "structural",
  skipConnection: "structural",
  concatenate: "structural",
  add: "structural",
  multiply: "structural",
  split: "structural",
  permute: "structural",
  squeeze: "structural",
  unsqueeze: "structural",
  pad: "structural",
  matmul: "structural",
  clamp: "structural",
  layerScale: "structural",
  stack: "structural",
  embedding: "embedding",
  segmentEmbedding: "embedding",
  embeddingBag: "embedding",
  positionalEncoding: "positional",
  learnedPositionalEmbedding: "positional",
  rope: "positional",
  timeEmbedding: "positional",
  attention: "attention",
  selfAttention: "attention",
  crossAttention: "attention",
  multiHeadAttention: "attention",
  groupedQueryAttention: "attention",
  causalAttention: "attention",
  transformerBlock: "attention",
  localAttention: "attention",
  windowAttention: "attention",
  mla: "attention",
  differentialAttention: "attention",
  ditBlock: "attention",
  conformerBlock: "attention",
  lstm: "recurrent",
  gru: "recurrent",
  rnn: "recurrent",
  bidirectionalLSTM: "recurrent",
  bidirectionalGRU: "recurrent",
  upsample: "upsample",
  interpolate: "upsample",
  graphConv: "graph",
  graphAttention: "graph",
  graphSAGE: "graph",
  gcn: "graph",
  gat: "graph",
  gin: "graph",
  edgeConv: "graph",
  scatter: "graph",
  mamba: "ssm",
  mamba2: "ssm",
  s4Layer: "ssm",
  hyena: "ssm",
  rwkv: "ssm",
  retention: "ssm",
  rgLru: "ssm",
  deltaNet: "ssm",
  gatedDeltaNet: "ssm",
  xlstm: "ssm",
  moeLayer: "moe",
  sharedExpertMoE: "moe",
  topK: "moe",
  gather: "gather"
};
function familyOf(type) {
  return FAMILY_OF[type] ?? "unknown";
}
var BACKEND_LABEL = {
  coreml: "CoreML (ML Program, via ONNX Runtime)",
  xnnpack: "XNNPACK (ExecuTorch)",
  tensorrt: "TensorRT",
  "onnx-cpu": "ONNX Runtime CPU",
  pytorch: "PyTorch"
};
var BACKEND_SOURCE = {
  coreml: "https://github.com/microsoft/onnxruntime/blob/main/tools/ci_build/github/apple/coreml_supported_mlprogram_ops.md",
  xnnpack: "https://github.com/pytorch/executorch/tree/main/backends/xnnpack/partition/config",
  tensorrt: "https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md",
  "onnx-cpu": "https://github.com/microsoft/onnxruntime/blob/main/docs/OperatorKernels.md",
  pytorch: "https://pytorch.org/docs/stable/nn.html"
};
var BACKEND_FALLBACK = {
  coreml: true,
  xnnpack: true,
  tensorrt: true,
  "onnx-cpu": false,
  pytorch: false
};
var TARGET_BACKEND = {
  "mobile-ios": "coreml",
  "mobile-android": "xnnpack",
  "edge-cpu": "onnx-cpu",
  "edge-gpu": "tensorrt",
  // The wasm build of ONNX Runtime Web compiles the CPU kernels.
  browser: "onnx-cpu",
  // The WebGPU EP covers a subset with a CPU fallback; we hold no verified
  // list of that subset, so it is not assessed.
  "browser-webgpu": null,
  // edgetpu_compiler's op list is not transcribed here yet.
  "edge-tpu": null,
  "cloud-t4": "pytorch",
  "cloud-a100": "pytorch"
};
var yes = (note) => ({ ok: true, ...note ? { note } : {} });
var no = (reason) => ({ ok: false, reason });
var cond = (check, note) => ({ ok: true, check, ...note ? { note } : {} });
var pv = (c, ...keys) => {
  const p = c.params ?? {};
  for (const k of keys) if (p[k] !== void 0) return p[k];
  return void 0;
};
var first = (v) => Array.isArray(v) ? typeof v[0] === "number" ? v[0] : null : typeof v === "number" ? v : null;
var CONV_TYPES2 = /* @__PURE__ */ new Set(["conv1d", "conv2d", "depthwiseConv2d", "separableConv2d", "dilatedConv2d", "transposeConv2d", "conv3d", "depthwiseConv1d", "audioConv"]);
var bnAfterConv = (_c, { prevType }) => prevType && CONV_TYPES2.has(prevType) ? null : "BatchNorm that does not follow a conv cannot be fused into it, and this backend has no standalone BatchNormalization";
var TABLE = {
  // onnxruntime coreml_supported_mlprogram_ops.md
  coreml: {
    conv12d: yes("Only 1D/2D Conv is supported; bias must be constant."),
    conv3d: no("Only 1D/2D Conv is supported."),
    convTranspose: yes("Weight and bias must be constant; output_shape is not supported."),
    linear: yes("Gemm: input B must be constant."),
    batchNorm: cond(bnAfterConv),
    layerNorm: yes(),
    groupNorm: yes(),
    instanceNorm: yes(),
    rmsNorm: yes("Lowered to Pow, ReduceMean, Add, Sqrt, Div, Mul below opset 23, all supported."),
    activation: yes(),
    elu: yes(),
    hardSwish: no("HardSwish is not in the CoreML ML Program op list."),
    mish: no("Mish is not in the CoreML ML Program op list."),
    softmax: yes(),
    maxPool2d: yes(),
    avgPool2d: yes("Only 2D Pool is supported."),
    pool1d: no("Only 2D Pool is supported currently."),
    pool3d: no("Only 2D Pool is supported currently."),
    globalPool: cond((c) => /1d/i.test(c.type) ? "GlobalAveragePool: only 2D Pool is supported." : null),
    identity: yes(),
    structural: yes(),
    embedding: no("Gather is not in the CoreML ML Program op list (only GatherND)."),
    positional: yes(),
    attention: yes("MatMul, Softmax, Transpose and Reshape are all supported."),
    recurrent: no("LSTM and GRU are not in the CoreML ML Program op list."),
    upsample: yes("Resize: some mode combinations are refused; see resize_op_builder.cc."),
    graph: no("Scatter ops are not in the CoreML op list."),
    ssm: no("A selective scan lowers to a Loop, which CoreML does not take."),
    moe: no("TopK is not in the CoreML op list."),
    gather: no("Gather is not in the CoreML ML Program op list.")
  },
  // executorch backends/xnnpack/partition/config: ALL_PARTITIONER_CONFIGS
  xnnpack: {
    conv12d: yes("ConvolutionConfig: 1D and 2D only."),
    conv3d: no('ConvolutionConfig: "Only support 1D + 2D Conv".'),
    convTranspose: yes(),
    linear: yes(),
    batchNorm: cond(bnAfterConv),
    layerNorm: no("native_layer_norm has no XNNPACK partitioner config; it runs on the portable CPU kernels."),
    groupNorm: no("group_norm has no XNNPACK partitioner config."),
    instanceNorm: no("instance_norm has no XNNPACK partitioner config."),
    rmsNorm: yes("Lowered to pow, mean, rsqrt, mul, each with a config."),
    activation: yes(),
    elu: no("EluConfig is commented out of ALL_PARTITIONER_CONFIGS."),
    hardSwish: yes(),
    mish: no("Mish has no XNNPACK partitioner config."),
    softmax: cond((c) => {
      const d = pv(c, "dim", "axis");
      return typeof d === "number" && d !== -1 ? "SoftmaxConfig: the softmax dim must be the last dimension." : null;
    }),
    maxPool2d: cond((c) => {
      if (pv(c, "ceilMode", "ceil_mode") === true) return "MaxPool2dConfig: ceil mode is not supported.";
      const k = first(pv(c, "kernelSize", "kernel_size"));
      const s = first(pv(c, "stride"));
      return k != null && s != null && s > k ? "MaxPool2dConfig: stride must not exceed kernel_size." : null;
    }),
    avgPool2d: yes(),
    pool1d: no("Only 2D pooling has an XNNPACK config."),
    pool3d: no("Only 2D pooling has an XNNPACK config."),
    globalPool: yes("MeanDimConfig."),
    identity: yes(),
    structural: yes(),
    embedding: no("embedding has no XNNPACK partitioner config."),
    positional: yes(),
    attention: yes("BMM, softmax, permute and view all have configs."),
    recurrent: no("LSTM and GRU have no XNNPACK partitioner config."),
    upsample: cond((c) => {
      const mode = String(pv(c, "mode") ?? "nearest");
      return /bilinear/i.test(mode) ? null : "Only upsample_bilinear2d has an XNNPACK config.";
    }),
    graph: no("Scatter ops have no XNNPACK config."),
    ssm: no("A selective scan has no XNNPACK config."),
    moe: no("topk has no XNNPACK config."),
    gather: no("gather has no XNNPACK config.")
  },
  // onnx-tensorrt docs/operators.md
  tensorrt: {
    conv12d: yes(),
    conv3d: yes(),
    convTranspose: yes(),
    linear: yes(),
    batchNorm: yes(),
    layerNorm: yes("Only the first output Y is supported."),
    groupNorm: yes(),
    instanceNorm: yes(),
    rmsNorm: yes(),
    activation: yes(),
    elu: yes(),
    hardSwish: yes(),
    mish: yes(),
    softmax: yes(),
    maxPool2d: yes("2D or 3D pooling only; dilations must be all ones."),
    avgPool2d: yes(),
    pool1d: no("MaxPool: 2D or 3D pooling only."),
    pool3d: yes(),
    globalPool: yes(),
    identity: yes(),
    structural: yes(),
    embedding: yes(),
    positional: yes(),
    attention: yes("Lowered to MatMul and Softmax; the fused Attention op needs 4D Q, K, V."),
    recurrent: yes("Bidirectional: activations must match both directions; layout must be 0."),
    upsample: yes("Resize: nearest or linear; antialiasing is not supported."),
    graph: no("ScatterND: reduction other than none is not supported, and message passing sums."),
    ssm: no("A fused selective-scan kernel has no ONNX export to hand TensorRT."),
    moe: yes("TopK: K must be under 3840 and sorted must be 1."),
    gather: yes()
  },
  // ONNX Runtime CPU EP: every standard op has a kernel. What cannot run is
  // what cannot be exported to standard ONNX at all.
  "onnx-cpu": {
    graph: yes("Exports through ScatterElements with reduction, opset 16 or later."),
    ssm: no("A fused selective-scan kernel has no ONNX symbolic; only a pure PyTorch scan exports.")
  },
  pytorch: {}
};
function supportFor(backend, family) {
  if (family === "unknown") return null;
  if (backend === "pytorch") return yes();
  const row = TABLE[backend][family];
  if (row) return row;
  return backend === "onnx-cpu" ? yes() : null;
}
function isLayer(c) {
  return c.type !== "input" && c.type !== "output" && c.type !== "stickyNote";
}
function layerSupport(model, backend) {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const prevOf = /* @__PURE__ */ new Map();
  for (const e of model.connections ?? []) {
    const from = byId.get(e.from);
    if (from && !prevOf.has(e.to)) prevOf.set(e.to, from.type);
  }
  return model.components.filter(isLayer).map((c) => {
    const family = familyOf(c.type);
    const s = supportFor(backend, family);
    if (!s) return { componentId: c.id, name: c.name, type: c.type, family, supported: null };
    if (!s.ok) return { componentId: c.id, name: c.name, type: c.type, family, supported: false, reason: s.reason };
    const why = s.check?.(c, { prevType: prevOf.get(c.id) ?? null }) ?? null;
    return why ? { componentId: c.id, name: c.name, type: c.type, family, supported: false, reason: why } : { componentId: c.id, name: c.name, type: c.type, family, supported: true };
  });
}
function supportedPartitions(model, support) {
  const supportedIds = new Set(support.filter((s) => s.supported !== false).map((s) => s.componentId));
  const nodes = model.components.filter(isLayer).map((c) => c.id);
  const nodeSet = new Set(nodes);
  const producers = /* @__PURE__ */ new Map();
  const consumers = /* @__PURE__ */ new Map();
  for (const id of nodes) {
    producers.set(id, /* @__PURE__ */ new Set());
    consumers.set(id, /* @__PURE__ */ new Set());
  }
  for (const e of model.connections ?? []) {
    if (!nodeSet.has(e.from) || !nodeSet.has(e.to) || e.from === e.to) continue;
    producers.get(e.to).add(e.from);
    consumers.get(e.from).add(e.to);
  }
  const inDegree = new Map(nodes.map((id) => [id, producers.get(id).size]));
  let toProcess = nodes.filter((id) => inDegree.get(id) === 0);
  let withNext = [];
  const groups = [];
  let group = [];
  let border = /* @__PURE__ */ new Set();
  const done = /* @__PURE__ */ new Set();
  const close = () => {
    if (group.length) groups.push(group);
    group = [];
    border = /* @__PURE__ */ new Set();
  };
  let guard = 0;
  while ((toProcess.length || withNext.length) && guard++ < nodes.length * 4 + 10) {
    if (!toProcess.length) {
      close();
      toProcess = withNext;
      withNext = [];
      continue;
    }
    const id = toProcess.shift();
    if (done.has(id)) continue;
    const ok = supportedIds.has(id);
    if (!ok && border.has(id)) {
      withNext.push(id);
      continue;
    }
    done.add(id);
    if (ok) {
      group.push(id);
      border.delete(id);
      for (const c of consumers.get(id)) border.add(c);
    }
    for (const c of consumers.get(id)) {
      const d = inDegree.get(c) - 1;
      inDegree.set(c, d);
      if (d === 0) toProcess.push(c);
    }
  }
  close();
  const missed = nodes.filter((id) => !done.has(id) && supportedIds.has(id));
  if (missed.length) groups.push(missed);
  return groups;
}
function suitabilityFrom(partitions, pctSupported) {
  if (partitions === 1) return pctSupported > 75 ? "YES" : pctSupported > 50 ? "MAYBE" : "NO";
  if (partitions === 2) return pctSupported > 75 ? "MAYBE" : "NO";
  return "NO";
}
function assessBackend(model, backend) {
  const support = layerSupport(model, backend);
  const assessedRows = support.filter((s) => s.supported !== null);
  const supportedCount = assessedRows.filter((s) => s.supported).length;
  const share = assessedRows.length ? supportedCount / assessedRows.length : 1;
  const unsupported = support.filter((s) => s.supported === false).map((s) => ({ componentId: s.componentId, name: s.name, type: s.type, reason: s.reason ?? "Not supported." }));
  const notAssessedTypes = [...new Set(support.filter((s) => s.supported === null).map((s) => s.type))];
  const groups = supportedPartitions(model, support);
  const label = BACKEND_LABEL[backend];
  const pctTxt = `${Math.round(share * 100)}%`;
  const typesTxt = [...new Set(unsupported.map((u) => u.type))].slice(0, 3).join(", ");
  let verdict;
  let reason;
  if (!BACKEND_FALLBACK[backend]) {
    verdict = unsupported.length === 0 ? "YES" : "NO";
    reason = unsupported.length === 0 ? `${label} runs every layer${assessedRows.length < support.length ? " our table covers" : ""}.` : `${label} cannot run ${typesTxt}, and there is no fallback device, so the model does not run there as built.`;
  } else {
    verdict = suitabilityFrom(groups.length, share * 100);
    const parts = `${groups.length} partition${groups.length === 1 ? "" : "s"}`;
    if (unsupported.length === 0) {
      reason = `${label} takes every layer${assessedRows.length < support.length ? " our table covers" : ""} as one partition.`;
    } else if (verdict === "YES") {
      reason = `${label} takes ${pctTxt} of layers in one partition; ${typesTxt} run on the CPU after it.`;
    } else if (verdict === "MAYBE") {
      reason = `${label} takes ${pctTxt} of layers in ${parts}; ${typesTxt} fall back to the CPU. Worth measuring before relying on it.`;
    } else {
      reason = `Cut into ${parts}, ${label} covers ${pctTxt} of layers; ${typesTxt} fall back to the CPU, and with more than two partitions the copies between devices likely cost more than the accelerator saves, so the CPU alone is probably faster.`;
    }
  }
  return {
    backend,
    backendLabel: label,
    source: BACKEND_SOURCE[backend],
    verdict,
    reason,
    layers: support.length,
    assessed: assessedRows.length,
    supported: supportedCount,
    supportedShare: share,
    partitions: groups.length,
    partitionSizes: groups.map((g) => g.length),
    unsupported,
    notAssessedTypes
  };
}
function assessTarget(model, target) {
  const backend = TARGET_BACKEND[target] ?? null;
  if (!backend) {
    return { backend: null, verdict: null, reason: "No verified op table is held for this runtime, so op coverage is not assessed here." };
  }
  return assessBackend(model, backend);
}
var QUANTIZABLE = /* @__PURE__ */ new Set(["linear", "conv12d", "conv3d", "convTranspose", "attention"]);
function quantizedSize(model) {
  let total = 0;
  let q = 0;
  for (const c of model.components) {
    const n2 = estimateLayerParams(c.type, c.params, c.inputShape ?? []);
    total += n2;
    if (QUANTIZABLE.has(familyOf(c.type))) q += n2;
  }
  const MB = 1024 * 1024;
  return {
    fp32MB: total * 4 / MB,
    int8MB: (q * 1 + (total - q) * 4) / MB,
    quantizableShare: total > 0 ? q / total : 0,
    quantizableParams: q,
    totalParams: total
  };
}
var OPSET = {
  attention: { min: 14, native: 23, why: "scaled_dot_product_attention exports from opset 14 (torch symbolic_opset14); the causal mask uses Trilu, since 14; the native Attention op is 23" },
  upsample: { min: 11, why: "Resize gained the coordinate modes PyTorch interpolation needs at opset 11" },
  pixelShuffle: { min: 11, why: "DepthToSpace CRD mode, which pixel_shuffle needs, is opset 11" },
  graph: { min: 16, why: "ScatterElements with reduction, which message passing sums through, is opset 16" },
  gridSample: { min: 16, why: "GridSample is opset 16" },
  einsum: { min: 12, why: "Einsum is opset 12" },
  layerNorm: { native: 17, why: "LayerNormalization is native from opset 17; below it torch.onnx decomposes it" },
  groupNorm: { native: 18, why: "GroupNormalization is native from opset 18" },
  rmsNorm: { native: 23, why: "RMSNormalization is native from opset 23" },
  gelu: { native: 20, why: "Gelu is native from opset 20; below it, Erf" },
  mish: { native: 18, why: "Mish is native from opset 18" },
  hardSwish: { native: 14, why: "HardSwish is native from opset 14" },
  rope: { native: 23, why: "RotaryEmbedding is native from opset 23" }
};
function opsetRowFor(type) {
  if (OPSET[type]) return OPSET[type];
  const fam = familyOf(type);
  if (fam === "attention") return OPSET.attention;
  if (fam === "upsample") return OPSET.upsample;
  if (fam === "graph") return OPSET.graph;
  if (fam === "layerNorm") return OPSET.layerNorm;
  return void 0;
}
function onnxOpsetReport(model) {
  const seen = /* @__PURE__ */ new Set();
  const requiredBy = [];
  const nativeFrom = [];
  for (const c of model.components) {
    if (!isLayer(c) || seen.has(c.type)) continue;
    seen.add(c.type);
    const row = opsetRowFor(c.type);
    if (!row) continue;
    if (row.min) requiredBy.push({ type: c.type, opset: row.min, why: row.why });
    if (row.native) nativeFrom.push({ type: c.type, opset: row.native, why: row.why });
  }
  requiredBy.sort((a, b) => b.opset - a.opset);
  return { minOpset: requiredBy.length ? requiredBy[0].opset : null, requiredBy, nativeFrom };
}
var VLLM_GPU_MEMORY_UTILIZATION = 0.92;
var GiB = 1024 ** 3;
var gib = (b) => (b / GiB).toFixed(2);
function kvCacheFit(model, memoryMB, opts) {
  if (!opts.isDecoderLM) return null;
  const shape = detectLLMShape(model);
  const maxModelLen = modelMaxSeqLen(model);
  if (!shape || !maxModelLen) return null;
  const perToken = kvBytesPerToken(shape, 2);
  if (!(perToken > 0)) return null;
  const memory2 = memoryMB * 1024 * 1024;
  const weightBytes = quantizedSize(model).totalParams * 2;
  const reserve = Math.min(GiB, memory2 * 0.1);
  const available = memory2 * VLLM_GPU_MEMORY_UTILIZATION - weightBytes - reserve;
  const positions = effectiveKvPositions(shape, maxModelLen);
  const needed = perToken * positions;
  const longest = available > 0 ? shape.windowSize && shape.windowSize > 0 && perToken * shape.windowSize <= available ? maxModelLen : Math.min(maxModelLen, Math.floor(available / perToken)) : 0;
  const fits = available > 0 && needed <= available;
  let vllmError = null;
  if (available <= 0) {
    vllmError = "No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.";
  } else if (!fits) {
    vllmError = `To serve at least one request with the model's max seq len (${maxModelLen}), (${gib(needed)} GiB KV cache is needed, which is larger than the available KV cache memory (${gib(available)} GiB). ` + (longest > 0 ? `Based on the available memory, the estimated maximum model length is ${longest}. ` : "") + "Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.";
  }
  return {
    maxModelLen,
    kvBytesPerToken: perToken,
    weightBytes,
    activationReserveBytes: reserve,
    availableBytes: Math.max(0, available),
    neededBytes: needed,
    fits,
    longestContext: longest,
    concurrentAtMax: fits ? Math.floor(available / needed) : 0,
    vllmError
  };
}

// src/utils/deployAdvisor.ts
var PLATFORM_PROFILES = [
  {
    id: "mobile-ios",
    label: "iOS / CoreML",
    icon: "mobile",
    maxParamsMB: 50,
    hardParamLimit: 5e7,
    maxMemoryMB: 256,
    description: "iPhone/iPad inference via CoreML. Supports ANE (neural engine) for quantized conv/linear ops.",
    runtime: "CoreML / PyTorch Mobile",
    effectiveMacsPerSec: 5e11,
    baseLatencyMs: 3,
    incompatibleOps: {
      conv3d: "Conv3D is not supported by CoreML on older devices",
      graphConv: "GNN ops not supported by CoreML, requires custom layer",
      gat: "GAT not natively supported in CoreML",
      gcn: "GCN not natively supported in CoreML",
      graphSAGE: "GraphSAGE not natively supported in CoreML"
    },
    slowOps: {
      lstm: "LSTM is significantly slower than GRU on ANE, consider replacing",
      bidirectionalLSTM: "Bidirectional LSTM not accelerated on ANE",
      multiHeadAttention: "Attention is fast only with the Transformer op set (iOS 16+)",
      transformerBlock: "Transformer blocks work best with CoreML Transformer API (iOS 17+)"
    }
  },
  {
    id: "mobile-android",
    label: "Android / TFLite",
    icon: "mobile",
    maxParamsMB: 50,
    hardParamLimit: 5e7,
    maxMemoryMB: 256,
    description: "Android inference via TensorFlow Lite or ONNX Runtime Mobile.",
    runtime: "TFLite / ONNX Runtime Mobile",
    effectiveMacsPerSec: 15e10,
    baseLatencyMs: 5,
    incompatibleOps: {
      conv3d: "Conv3D has limited TFLite delegate support",
      graphConv: "GNN ops not supported in TFLite",
      gat: "GAT not in TFLite op set",
      gcn: "GCN not in TFLite op set",
      graphSAGE: "GraphSAGE not in TFLite op set"
    },
    slowOps: {
      lstm: "Use LSTM with TFLite optimized kernels or switch to GRU for speed",
      bidirectionalLSTM: "Bidirectional LSTM is slow: consider unidirectional",
      transformerBlock: "Full Transformer blocks require XNNPACK delegate (Android 8+)"
    }
  },
  {
    id: "edge-cpu",
    label: "Edge CPU (Pi/x86)",
    icon: "cpu",
    maxParamsMB: 100,
    hardParamLimit: 1e8,
    maxMemoryMB: 512,
    description: "CPU-only inference on Raspberry Pi, Jetson CPU, or x86 edge servers.",
    runtime: "ONNX Runtime / OpenVINO",
    effectiveMacsPerSec: 1e10,
    baseLatencyMs: 2,
    incompatibleOps: {},
    slowOps: {
      conv3d: "Conv3D is extremely slow on CPU, avoid for real-time use",
      multiHeadAttention: "Attention is O(n\xB2): keep sequence lengths short",
      transformerBlock: "Transformer blocks are CPU-intensive, minimize depth",
      lstm: "Consider 1-layer LSTM or switch to CNN-based sequence model",
      bidirectionalLSTM: "Bidirectional LSTM doubles compute on CPU"
    }
  },
  {
    id: "edge-gpu",
    label: "Edge GPU (Jetson)",
    icon: "cpu",
    maxParamsMB: 200,
    hardParamLimit: 2e8,
    maxMemoryMB: 4096,
    description: "NVIDIA Jetson (Nano/Orin) GPU inference via TensorRT.",
    runtime: "TensorRT / ONNX",
    effectiveMacsPerSec: 15e11,
    baseLatencyMs: 5,
    incompatibleOps: {
      graphConv: "Custom GNN kernels required for TensorRT",
      gat: "Custom GAT kernel required"
    },
    slowOps: {
      depthwiseConv2d: "Depthwise conv has poor TensorRT throughput on Jetson Nano, use regular Conv2D"
    }
  },
  {
    id: "browser",
    label: "Browser / ONNX.js",
    icon: "globe",
    maxParamsMB: 20,
    hardParamLimit: 2e7,
    maxMemoryMB: 512,
    description: "In-browser inference via ONNX Runtime Web or TensorFlow.js.",
    runtime: "ONNX Runtime Web / TF.js",
    effectiveMacsPerSec: 1e10,
    baseLatencyMs: 15,
    incompatibleOps: {
      conv3d: "Conv3D not supported in ONNX Runtime Web",
      graphConv: "GNN ops not supported in ONNX Runtime Web",
      gat: "GAT not in ONNX Runtime Web",
      gcn: "GCN not in ONNX Runtime Web",
      graphSAGE: "GraphSAGE not in ONNX Runtime Web",
      bidirectionalLSTM: "Bidirectional LSTM unsupported in ONNX Runtime Web"
    },
    slowOps: {
      lstm: "LSTM is slow in WebAssembly: consider replacing with CNN",
      transformerBlock: "Transformer is memory-intensive in browser"
    }
  },
  {
    id: "browser-webgpu",
    label: "Browser / WebGPU",
    icon: "globe",
    maxParamsMB: 100,
    hardParamLimit: 1e8,
    maxMemoryMB: 2048,
    description: "In-browser inference via WebGPU, Chrome 113+ / Edge / Safari TP. Significantly faster than WASM for compute-heavy models.",
    runtime: "ONNX Runtime Web (WebGPU EP) / TF.js (webgpu backend)",
    // Calibrated against measured ResNet50 ~80ms in Chrome WebGPU on M1.
    effectiveMacsPerSec: 5e10,
    baseLatencyMs: 8,
    incompatibleOps: {
      conv3d: "Conv3D not yet implemented in ONNX Runtime Web WebGPU EP",
      graphConv: "GNN ops not supported in browser runtimes",
      gat: "GAT not in browser WebGPU op set",
      gcn: "GCN not in browser WebGPU op set"
    },
    slowOps: {
      bidirectionalLSTM: "BiLSTM is slower in WebGPU than WASM, consider GRU",
      transformerBlock: "Large transformer blocks may exceed browser GPU buffer limits (~2GB per tensor)"
    }
  },
  {
    id: "edge-tpu",
    label: "Edge TPU (Coral)",
    icon: "cpu",
    // Coral USB / dev board with INT8 quantization. Hard cap is the on-chip
    // SRAM size — anything past ~8 MB starts paging to host and tanks throughput.
    maxParamsMB: 8,
    hardParamLimit: 8e6,
    maxMemoryMB: 8,
    description: "Google Coral Edge TPU (USB / Dev Board). Requires INT8 quantization and a fully-supported op set; fastest INT8 inference per dollar.",
    runtime: "TensorFlow Lite + edgetpu_compiler",
    effectiveMacsPerSec: 4e12,
    // 4 TOPS INT8 — but only if the model compiles cleanly
    baseLatencyMs: 2,
    incompatibleOps: {
      // Coral's edgetpu_compiler refuses to compile these — they fall back to CPU
      // and erase the latency advantage.
      conv3d: "Conv3D not supported by edgetpu_compiler",
      transformerBlock: "Transformer ops not supported, model will run entirely on CPU",
      multiHeadAttention: "Attention not in Edge TPU op set",
      lstm: "LSTM falls back to CPU on Coral, defeats the purpose",
      bidirectionalLSTM: "BiLSTM falls back to CPU on Coral",
      gru: "GRU falls back to CPU on Coral",
      graphConv: "GNN ops not supported",
      gat: "GAT not in Edge TPU op set",
      gcn: "GCN not in Edge TPU op set"
    },
    slowOps: {
      depthwiseConv2d: "Depthwise conv compiles but underutilizes the matrix unit, regular Conv2D is often faster on Coral"
    }
  },
  {
    id: "cloud-t4",
    label: "Cloud GPU T4",
    icon: "cloud",
    maxParamsMB: 1e4,
    hardParamLimit: 2e9,
    maxMemoryMB: 16384,
    description: "NVIDIA T4 GPU (16GB VRAM): common on GCP, AWS g4dn, Azure NC.",
    runtime: "PyTorch / TensorFlow",
    effectiveMacsPerSec: 15e11,
    baseLatencyMs: 1,
    incompatibleOps: {},
    slowOps: {
      graphConv: "GNNs require torch_geometric: additional install needed"
    }
  },
  {
    id: "cloud-a100",
    label: "Cloud GPU A100",
    icon: "cloud",
    maxParamsMB: 8e4,
    hardParamLimit: 2e10,
    maxMemoryMB: 8e4,
    description: "NVIDIA A100 (40/80GB VRAM): best for LLMs and large batch training.",
    runtime: "PyTorch / TensorFlow",
    effectiveMacsPerSec: 25e12,
    baseLatencyMs: 1,
    incompatibleOps: {},
    slowOps: {}
  }
];
var KV_TARGETS = /* @__PURE__ */ new Set(["cloud-t4", "cloud-a100", "edge-gpu"]);
function fmtLatency(ms) {
  if (ms < 1) return "<1ms";
  if (ms < 10) return `~${ms.toFixed(1)}ms`;
  if (ms < 1e3) return `~${Math.round(ms)}ms`;
  return `~${(ms / 1e3).toFixed(1)}s`;
}
function totalParams(model) {
  return model.components.reduce((sum, c) => sum + estimateLayerParams(c.type, c.params, c.inputShape ?? []), 0);
}
function totalFlops(model) {
  return model.components.reduce((sum, c) => {
    return sum + estimateLayerFlops(c.type, c.params, c.inputShape ?? [], c.outputShape ?? []);
  }, 0);
}
function paramsMB(params) {
  return params * 4 / (1024 * 1024);
}
function analyzeForDeployment(model, target) {
  const profile = PLATFORM_PROFILES.find((p) => p.id === target);
  const recommendations = [];
  const comps = model.components.filter((c) => c.type !== "input" && c.type !== "output");
  const params = totalParams(model);
  const flops = totalFlops(model);
  const sizeMB = paramsMB(params);
  const quantized = quantizedSize(model);
  const int8MB = quantized.int8MB;
  const int8Line = `~${int8MB.toFixed(1)} MB (int8 for the ${Math.round(quantized.quantizableShare * 100)}% of weights in linear, conv and attention layers; the rest stays float32)`;
  if (params > profile.hardParamLimit) {
    recommendations.push({
      id: "params-too-large",
      severity: "blocker",
      title: "Model too large",
      message: `Model has ${fmtParams(params)} params (${sizeMB.toFixed(1)} MB). ${profile.label} recommends \u2264 ${fmtParams(profile.hardParamLimit)} (${profile.maxParamsMB} MB). Consider pruning, quantization, or replacing large layers.`,
      affectedIds: []
    });
  } else if (sizeMB > profile.maxParamsMB * 0.7) {
    recommendations.push({
      id: "params-large-warn",
      severity: "warning",
      title: "Model approaching size limit",
      message: `${sizeMB.toFixed(1)} MB is above 70% of the ${profile.maxParamsMB} MB target. Consider INT8 quantization to reduce to ${int8Line}.`,
      affectedIds: []
    });
  }
  for (const comp of comps) {
    const reason = profile.incompatibleOps[comp.type];
    if (reason) {
      recommendations.push({
        id: `incompat-${comp.id}`,
        severity: "blocker",
        title: `Incompatible op: ${comp.type}`,
        message: reason,
        affectedIds: [comp.id]
      });
    }
  }
  const slowGroups = /* @__PURE__ */ new Map();
  for (const comp of comps) {
    const reason = profile.slowOps[comp.type];
    if (reason) {
      if (!slowGroups.has(comp.type)) slowGroups.set(comp.type, []);
      slowGroups.get(comp.type).push(comp);
    }
  }
  for (const [type, group] of slowGroups.entries()) {
    const reason = profile.slowOps[type];
    recommendations.push({
      id: `slow-${type}`,
      severity: "warning",
      title: `Slow op on ${profile.label}: ${type}`,
      message: reason ?? `${type} has known performance issues on ${profile.label}.`,
      affectedIds: group.map((c) => c.id),
      fix: type === "lstm" ? group.map((c) => ({
        componentId: c.id,
        updates: { type: "gru", name: c.name.replace(/lstm/gi, "gru") },
        description: `Replace ${c.name} (LSTM) \u2192 GRU (fewer gates, faster on-device)`
      })) : type === "bidirectionalLSTM" ? group.map((c) => ({
        componentId: c.id,
        updates: { type: "lstm", name: c.name.replace(/bi.*lstm/gi, "lstm") },
        description: `Replace ${c.name} (BiLSTM) \u2192 LSTM (remove backward pass)`
      })) : void 0
    });
  }
  if (["mobile-ios", "mobile-android", "edge-cpu", "browser"].includes(target)) {
    const largeLinear = comps.filter(
      (c) => c.type === "linear" && ((c.params.outFeatures ?? 0) > 1024 || (c.params.inFeatures ?? 0) > 1024)
    );
    if (largeLinear.length > 0) {
      recommendations.push({
        id: "large-linear",
        severity: "warning",
        title: "Large linear layers on constrained device",
        message: `${largeLinear.length} linear layer(s) have dimension >1024, which is slow on ${profile.label}. Consider factorization; the one-click fix clamps in/out features to 512.`,
        affectedIds: largeLinear.map((c) => c.id),
        fix: largeLinear.map((c) => ({
          componentId: c.id,
          updates: {
            params: {
              ...c.params,
              outFeatures: Math.min(c.params.outFeatures ?? 512, 512),
              inFeatures: Math.min(c.params.inFeatures ?? 512, 512)
            }
          },
          description: `Reduce ${c.name} dimensions to \u2264512`
        }))
      });
    }
    const largeLSTM = comps.filter(
      (c) => ["lstm", "gru", "rnn"].includes(c.type) && (c.params.hiddenSize ?? 0) > 256
    );
    if (largeLSTM.length > 0) {
      recommendations.push({
        id: "large-rnn",
        severity: "warning",
        title: "Large recurrent hidden size",
        message: `${largeLSTM.length} RNN layer(s) have hiddenSize >${256}. For ${profile.label} target, reduce to \u2264128 for real-time performance.`,
        affectedIds: largeLSTM.map((c) => c.id),
        fix: largeLSTM.map((c) => ({
          componentId: c.id,
          updates: { params: { ...c.params, hiddenSize: 128 } },
          description: `Reduce ${c.name} hiddenSize \u2192 128`
        }))
      });
    }
  }
  if (["mobile-ios", "mobile-android"].includes(target)) {
    const hasBN = comps.some((c) => c.type === "batchNorm");
    if (!hasBN && params > 5e5) {
      recommendations.push({
        id: "no-bn-mobile",
        severity: "suggestion",
        title: "Add BatchNorm before quantization",
        message: "BatchNorm before Linear/Conv layers helps INT8 quantization maintain accuracy on mobile. Consider adding BatchNorm layers.",
        affectedIds: []
      });
    }
    const hasDropout = comps.some((c) => c.type === "dropout");
    if (hasDropout) {
      recommendations.push({
        id: "remove-dropout",
        severity: "suggestion",
        title: "Remove Dropout for inference",
        message: "Dropout should be disabled at inference time. Ensure your export pipeline disables dropout (model.eval() in PyTorch).",
        affectedIds: comps.filter((c) => c.type === "dropout").map((c) => c.id)
      });
    }
  }
  const hasConv3d = comps.some((c) => c.type === "conv3d");
  if (hasConv3d && ["mobile-ios", "mobile-android", "edge-cpu"].includes(target)) {
    recommendations.push({
      id: "conv3d-factorize",
      severity: "suggestion",
      title: "Factorize Conv3D \u2192 (2+1)D convolution",
      message: "Decompose Conv3D into a spatial Conv2D followed by a temporal Conv1D to reduce FLOPs by ~30% with minimal accuracy loss.",
      affectedIds: comps.filter((c) => c.type === "conv3d").map((c) => c.id)
    });
  }
  const highDropout2 = comps.filter((c) => c.type === "dropout" && (c.params.p ?? 0.5) > 0.5);
  if (highDropout2.length > 0 && ["mobile-ios", "mobile-android", "browser"].includes(target)) {
    recommendations.push({
      id: "high-dropout",
      severity: "suggestion",
      title: "High dropout rate",
      message: "Dropout > 0.5 can hurt inference accuracy. For deployment, ensure model.eval() is called, or consider reducing p to \u22640.3 during fine-tuning.",
      affectedIds: highDropout2.map((c) => c.id)
    });
  }
  if (["mobile-ios", "mobile-android", "edge-cpu", "browser"].includes(target) && params > 1e6) {
    const hasNorm = comps.some((c) => ["batchNorm", "layerNorm", "groupNorm", "instanceNorm", "rmsNorm"].includes(c.type));
    recommendations.push({
      id: "quantize-hint",
      severity: "suggestion",
      title: "Apply INT8 quantization",
      message: `Quantizing to INT8 would reduce model size from ${sizeMB.toFixed(1)} MB to ${int8Line} and speed up inference by 2-4\xD7. ${hasNorm ? "Normalization layers detected: good candidate for Post-Training Quantization (PTQ)." : "Consider adding BatchNorm layers first to improve quantization accuracy."}`,
      affectedIds: []
    });
  }
  let kvCache = null;
  if (KV_TARGETS.has(target)) {
    let isDecoderLM = false;
    try {
      isDecoderLM = deriveModelContract(model).taskType === "language-modeling";
    } catch {
    }
    kvCache = kvCacheFit(model, profile.maxMemoryMB, { isDecoderLM });
    if (kvCache && !kvCache.fits) {
      recommendations.push({
        id: "kv-cache-fit",
        severity: "blocker",
        title: kvCache.longestContext > 0 ? `KV cache does not fit at ${kvCache.maxModelLen.toLocaleString("en-US")} tokens; longest context that fits is ${kvCache.longestContext.toLocaleString("en-US")}` : "Weights leave no memory for the KV cache",
        message: `vLLM would refuse to start: "${kvCache.vllmError}" Assumes fp16 weights and cache, gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION}, and a ${(kvCache.activationReserveBytes / 1024 ** 3).toFixed(2)} GiB activation reserve that vLLM would measure and we cannot.`,
        affectedIds: []
      });
    }
  }
  const blockers = recommendations.filter((r) => r.severity === "blocker").length;
  const warnings = recommendations.filter((r) => r.severity === "warning").length;
  const suggestions = recommendations.filter((r) => r.severity === "suggestion").length;
  const score = Math.max(0, 100 - blockers * 30 - warnings * 10 - suggestions * 3);
  const assessment = assessTarget(model, target);
  const latencyMs = flops / profile.effectiveMacsPerSec * 1e3 + profile.baseLatencyMs;
  const HOURLY_USD = {
    "cloud-t4": 0.59,
    // Modal T4 on-demand
    "cloud-a100": 4.2
    // Modal A100 80GB on-demand
  };
  const SPOT_HOURLY_USD = {
    "cloud-t4": 0.15,
    // ~75% off T4 on-demand
    "cloud-a100": 1.2
    // ~71% off A100 80GB on-demand
  };
  const hourly = HOURLY_USD[target];
  const spotHourly = SPOT_HOURLY_USD[target];
  const costPerMillionUsd = hourly !== void 0 ? latencyMs / 1e3 / 3600 * hourly * 1e6 : null;
  const costPerMillionSpotUsd = spotHourly !== void 0 ? latencyMs / 1e3 / 3600 * spotHourly * 1e6 : null;
  const fmtUsd2 = (v) => v < 0.01 ? "<$0.01" : v < 1 ? `~$${v.toFixed(2)}` : v < 100 ? `~$${v.toFixed(1)}` : `~$${Math.round(v)}`;
  const costPerMillionLabel = costPerMillionUsd === null ? "free (on-device)" : fmtUsd2(costPerMillionUsd);
  const costPerMillionSpotLabel = costPerMillionSpotUsd === null ? null : fmtUsd2(costPerMillionSpotUsd);
  const POWER_W = {
    "mobile-ios": 1.5,
    // A17 ANE during sustained inference
    "mobile-android": 1.8,
    // Mid-range NPU
    "edge-cpu": 5,
    // Pi 4 / x86 server CPU
    "edge-gpu": 15,
    // Jetson Orin Nano sustained
    "browser": 3,
    // Laptop CPU running WASM
    "browser-webgpu": 8,
    // Laptop GPU running WebGPU shaders
    "edge-tpu": 2
    // Coral USB sustained
  };
  const powerW = POWER_W[target];
  const energyPerInferenceMJ = powerW !== void 0 ? powerW * latencyMs : null;
  const energyLabel = energyPerInferenceMJ === null ? ", " : energyPerInferenceMJ < 0.1 ? "<0.1 mJ" : energyPerInferenceMJ < 1e3 ? `~${energyPerInferenceMJ.toFixed(1)} mJ` : `~${(energyPerInferenceMJ / 1e3).toFixed(2)} J`;
  const SMARTPHONE_BATTERY_MJ = 5544e4;
  const batteryPctPer1k = energyPerInferenceMJ === null ? null : energyPerInferenceMJ * 1e3 / SMARTPHONE_BATTERY_MJ * 100;
  const batteryLabel = batteryPctPer1k === null ? ", " : batteryPctPer1k < 1e-3 ? "<0.001% / 1k" : batteryPctPer1k < 1 ? `~${batteryPctPer1k.toFixed(3)}% / 1k` : `~${batteryPctPer1k.toFixed(2)}% / 1k`;
  return {
    target,
    profile,
    score,
    totalParams: params,
    totalParamsMB: sizeMB,
    totalParamsMBQuantized: int8MB,
    estimatedFlops: flops,
    estimatedLatencyMs: latencyMs,
    estimatedLatencyLabel: fmtLatency(latencyMs),
    costPerMillionUsd,
    costPerMillionLabel,
    costPerMillionSpotUsd,
    costPerMillionSpotLabel,
    energyPerInferenceMJ,
    energyLabel,
    batteryPctPer1k,
    batteryLabel,
    blockers,
    warnings,
    suggestions,
    recommendations,
    suitability: assessment.verdict,
    suitabilityReason: assessment.reason,
    backend: assessment.backend ? assessment : null,
    quantized,
    onnx: onnxOpsetReport(model),
    kvCache
  };
}
var DEPLOY_TARGET_KEY = "deploy-target";
function getChosenDeployTarget() {
  try {
    const raw = localStorage.getItem(DEPLOY_TARGET_KEY);
    return raw && PLATFORM_PROFILES.some((p) => p.id === raw) ? raw : null;
  } catch {
    return null;
  }
}
var DEPLOY_TIE_ORDER = [
  "cloud-t4",
  "cloud-a100",
  "edge-gpu",
  "edge-cpu",
  "browser",
  "browser-webgpu",
  "mobile-android",
  "mobile-ios",
  "edge-tpu"
];
function pickBestDeployTarget(results) {
  const rank = (t) => {
    const i = DEPLOY_TIE_ORDER.indexOf(t);
    return i < 0 ? DEPLOY_TIE_ORDER.length : i;
  };
  const suit = (r) => r.suitability === "YES" ? 0 : r.suitability === "MAYBE" ? 1 : r.suitability === null ? 2 : 3;
  return [...results].sort((a, b) => a.blockers - b.blockers || suit(a) - suit(b) || b.score - a.score || rank(a.target) - rank(b.target))[0];
}

// src/utils/pipelineStages.ts
var STAGE_ORDER = ["preflight", "data", "train", "evaluate", "deploy"];
var STAGE_LABEL = {
  preflight: "Pre-flight",
  data: "Data",
  train: "Train",
  evaluate: "Evaluate",
  deploy: "Deploy"
};
var fmtUsd = (v) => v < 0.01 ? "<$0.01" : v < 1 ? `$${v.toFixed(2)}` : `$${v.toFixed(1)}`;
var fmtDuration = (sec) => {
  const h = sec / 3600;
  return h >= 1 ? `${h.toFixed(1)}h` : `${Math.max(1, Math.ceil(sec / 60))}m`;
};
var fmtCount = (n2, one, many = `${one}s`) => `${n2} ${n2 === 1 ? one : many}`;
var pct3 = (v) => `${(v * 100).toFixed(0)}%`;
function isDesignLayer(c) {
  return !!c && c.type !== "input" && c.type !== "output" && c.type !== "stickyNote";
}
function realLayerCount(model) {
  return model.components.filter((c) => isDesignLayer({ type: c.type })).length;
}
function hasDesign(model) {
  try {
    return !!model && realLayerCount(model) > 0;
  } catch {
    return false;
  }
}
function emptyCanvasResult(stage) {
  return {
    stage,
    status: "blocked",
    headline: "Nothing on the canvas yet, so there is nothing to check.",
    notes: [{
      severity: "block",
      title: "Empty design",
      detail: "Describe what you want to build, or load a template, and run this again."
    }],
    data: { empty: true },
    evidence: evidenceFor(stage)
  };
}
var VOWEL_SOUND_INITIALS = /* @__PURE__ */ new Set(["A", "E", "F", "H", "I", "L", "M", "N", "O", "R", "S", "X"]);
function gpuArticle(name) {
  return VOWEL_SOUND_INITIALS.has((name?.[0] ?? "").toUpperCase()) ? "an" : "a";
}
function evidenceFor(stage) {
  const event = {
    preflight: "preflight:open",
    data: "open:data-step",
    train: "open:training-runner",
    evaluate: "open:evaluate-step",
    deploy: "open:deploy-advisor"
  }[stage];
  return { label: `Open ${STAGE_LABEL[stage]}`, event };
}
function runPreflightStage(model, workload) {
  if (realLayerCount(model) === 0) return emptyCanvasResult("preflight");
  const report = buildPreflightReport(model, {
    dataSpec: readDataSpec(),
    plan: readTrainPlan(),
    // A workload the caller read off the task sentence. Absent for every caller
    // that has only a graph, which is most of them, and pre-flight then says
    // "assumed" over its 50,000-row default exactly as before.
    workload
  });
  const notes = report.findings.filter((f) => f.severity !== "pass").map((f) => ({
    severity: f.severity === "block" ? "block" : f.severity === "warn" ? "warn" : "info",
    ruleId: f.ruleId ?? f.id,
    title: f.title,
    detail: f.detail,
    fix: f.fix,
    // Every pre-flight finding above `info` is a statement about the graph and
    // is therefore the agent's to act on. Class imbalance is the exception:
    // it is a fact about the user's dataset, and no edit to the layers can
    // change it.
    agentFixable: f.severity !== "info" && f.id !== "data-imbalance"
  }));
  const { block, warn } = report.counts;
  const m = report.metrics;
  const costLine = m.estTrainSec > 0 ? `${fmtUsd(m.estCostUsd)} / ${fmtDuration(m.estTrainSec)} on ${gpuArticle(m.fitsGpu)} ${m.fitsGpu}` : null;
  const assumed = [];
  if (m.samplesAssumed) assumed.push(`${(m.samples ?? 5e4).toLocaleString("en-US")} samples`);
  if (m.planAssumed) assumed.push(`${m.epochs ?? 10} epochs, batch ${m.batchSize ?? 32}`);
  const assumedTail = assumed.length ? `, assuming ${assumed.join(", ")}` : "";
  const noFit = m.gpuFits === false;
  const fitClause = noFit ? `does not fit ${gpuArticle(m.fitsGpu)} ${m.fitsGpu}${m.smallestFitGpu ? `; the smallest card that holds it is ${gpuArticle(m.smallestFitGpu)} ${m.smallestFitGpu}` : "; no single offered card holds it, so it needs sharding"}` : "";
  let status;
  let headline;
  if (block > 0) {
    status = "blocked";
    headline = costLine ? assumedTail ? `${fmtCount(block, "blocker")} would have failed a run priced at ${costLine}${assumedTail}.` : `${fmtCount(block, "blocker")} would have failed a ${costLine} run.` : `${fmtCount(block, "blocker")} stop this model from training.`;
  } else if (warn > 0) {
    status = "attention";
    headline = noFit ? `Trainable, but it ${fitClause}. ${fmtCount(warn, "warning")} in all.` : `Trainable, with ${fmtCount(warn, "warning")}. Estimated ${costLine ?? "cost unknown"}.`;
  } else {
    status = "ok";
    headline = `Ready to train: ${fmtCount(m.layers, "layer")}, ${(m.params / 1e6).toFixed(m.params < 1e6 ? 2 : 1)}M params${costLine ? `, about ${costLine}` : ""}.`;
  }
  return {
    stage: "preflight",
    status,
    headline,
    notes,
    data: {
      score: report.score,
      verdict: report.verdict,
      blockers: report.findings.filter((f) => f.severity === "block").map((f) => f.title),
      warnings: report.findings.filter((f) => f.severity === "warn").map((f) => f.title),
      layers: m.layers,
      params: m.params,
      flops: m.flops,
      // The card the cost is priced on. Read `gpuFits` before calling it the
      // card it trains on: it is the plan's choice whether or not it fits.
      gpu: m.fitsGpu,
      gpuFits: m.gpuFits ?? null,
      smallestFitGpu: m.smallestFitGpu ?? null,
      estCostUsd: m.estCostUsd,
      estTrainSec: m.estTrainSec,
      costAssumes: {
        samples: m.samples ?? null,
        epochs: m.epochs ?? null,
        batchSize: m.batchSize ?? null,
        samplesAssumed: m.samplesAssumed,
        planAssumed: m.planAssumed ?? false
      },
      // The memory numbers and the "was the sample count a guess" flag were
      // already computed here and thrown away at the stage boundary. Autopilot
      // shows the arithmetic beside each candidate, and a footprint with no
      // provenance is the half of that which makes it a claim again.
      weightBytes: m.weightBytes,
      trainFootprintBytes: m.trainFootprintBytes,
      samplesAssumed: m.samplesAssumed,
      // Model states, activations at the planned batch, overhead and the peak,
      // each with what it assumed; ZeRO per-card numbers for 2, 4 and 8 cards.
      memory: m.memory ?? null,
      fitsCards: m.fitsCards ?? [],
      // MACs, forward FLOPs and training FLOPs, with the convention spelled out.
      flopsDetail: m.flopsDetail ?? null
    },
    evidence: evidenceFor("preflight")
  };
}
function datasetOptionsFor(modality, numClasses) {
  const upload = modality === "vision" ? [{ label: "Upload my own images", value: "upload:image_zip", hint: "ImageFolder .zip, goes to your private storage" }] : [{ label: "Upload my own file", value: "upload:csv", hint: "CSV / JSON / JSONL, goes to your private storage" }];
  const synthetic = {
    label: "No data yet, smoke-test it",
    value: "synthetic",
    hint: "Random tensors matching the input contract, so train.py runs end to end"
  };
  switch (modality) {
    case "vision":
      return [
        { label: "CIFAR-10", value: "hf:uoft-cs/cifar10", hint: "60k 32\xD732 colour images, 10 classes" },
        { label: "MNIST", value: "hf:ylecun/mnist", hint: "70k 28\xD728 greyscale digits, 10 classes" },
        ...upload,
        synthetic
      ];
    case "nlp":
      return [
        { label: "IMDB reviews", value: "hf:stanfordnlp/imdb", hint: "50k reviews, binary sentiment" },
        // GLUE is eleven configs in one repo and load_dataset refuses it without
        // one; HF_OPTION_CONFIG supplies `sst2`, which is what this label means.
        { label: "GLUE / SST-2", value: "hf:nyu-mll/glue", hint: "Sentence-level sentiment, the standard small benchmark" },
        ...upload,
        synthetic
      ];
    case "audio":
      return [
        ...upload,
        synthetic
      ];
    case "tabular":
    default: {
      const benchmark = numClasses && numClasses > 2 ? { label: "Covertype", value: "hf:jxie/covtype", hint: "Forest cover type, 54 features, 7 classes" } : { label: "Adult / census income", value: "hf:scikit-learn/adult-census-income", hint: "A standard tabular benchmark to check the wiring against" };
      return [
        ...upload,
        benchmark,
        synthetic
      ];
    }
  }
}
function runDataStage(model) {
  if (realLayerCount(model) === 0) return emptyCanvasResult("data");
  const source = getDataSource();
  const contract = deriveModelContract(model);
  const modality = detectInputModality(model);
  const notes = [];
  notes.push({
    severity: "info",
    title: "Input contract from the graph",
    detail: `${contract.inputShape.length ? `shape (${contract.inputShape.join(", ")})` : "input shape not set"}, dtype ${contract.inputDtype}, ${contract.outputDim != null ? `${contract.outputDim} output${contract.outputDim !== 1 ? "s" : ""}` : "output size unknown"}, task ${contract.taskType}.`,
    fix: "validate_data.py enforces this before a GPU-minute is spent, so a mismatch fails at batch 0."
  });
  if (hasRealData(source)) {
    const mismatches = contractMismatches(model, source);
    for (const m of mismatches) {
      notes.push({
        severity: "block",
        blocks: true,
        ruleId: `dataset-${m.kind}`,
        title: m.title,
        detail: m.detail,
        fix: m.fix,
        // A graph edit resolves it, which is exactly what the agent can do.
        agentFixable: true
      });
    }
    let meta = null;
    let unchecked;
    if (source.kind === "huggingface") {
      const probe = readHfDatasetMeta(source);
      if (probe?.status === "ok") {
        meta = probe.meta;
        notes.push({
          severity: "info",
          ruleId: "dataset-viewer",
          title: "Read from the Hugging Face dataset viewer",
          detail: describeMeta(meta)
        });
        const findings = datasetMetaFindings(model, meta, {
          contract,
          modality,
          headAlreadyChecked: mismatches.some((m) => m.kind === "output-classes")
        });
        for (const f of findings) {
          notes.push({ ...f, blocks: f.severity === "block" });
        }
        const typed = (() => {
          try {
            const r = localStorage.getItem("preflight-data-spec");
            return r ? JSON.parse(r) : null;
          } catch {
            return null;
          }
        })();
        if (typed?.numClasses && meta.numClasses && typed.numClasses !== meta.numClasses) {
          notes.push({
            severity: "warn",
            ruleId: "dataset-declared-classes",
            title: `You declared ${typed.numClasses} classes; ${meta.datasetId} has ${meta.numClasses}`,
            detail: "Pre-flight uses the number you typed, because what you type wins. The dataset viewer reads the dataset's own label names.",
            fix: `Change the class count in Pre-flight's "Your data" to ${meta.numClasses}, or clear it to use the dataset's.`,
            agentFixable: false
          });
        }
      } else {
        const pending = requestHfDatasetMeta(source);
        unchecked = pending ? "Reading the dataset's labels, splits and statistics from Hugging Face." : probe ? `The Hugging Face dataset viewer could not be read (${probe.reason}), so labels, splits and class balance were not checked.` : "The dataset's metadata has not been read in this environment, so labels, splits and class balance were not checked.";
        notes.push({
          severity: pending ? "info" : "warn",
          blocks: false,
          ruleId: "dataset-unchecked",
          title: pending ? "Checking the dataset" : "Could not check the dataset",
          detail: unchecked,
          fix: pending ? void 0 : "Re-run when online. Nothing here was assumed to pass.",
          agentFixable: false
        });
      }
    }
    const blocked = notes.some((n2) => n2.severity === "block");
    const warned = notes.some((n2) => n2.severity === "warn");
    return {
      stage: "data",
      status: blocked ? "blocked" : warned || unchecked ? "attention" : "ok",
      headline: blocked ? `${describeDataSource(source)} is wired, but the graph does not match it.` : unchecked ? `Training on ${describeDataSource(source)}; its labels and splits are not checked yet.` : `Training on ${describeDataSource(source)}.`,
      notes,
      data: {
        wired: true,
        source,
        modality,
        contract,
        mismatches,
        // The dataset's own metadata, when it was read; null is "not read".
        datasetMeta: meta
      },
      evidence: evidenceFor("data"),
      ...unchecked ? { unchecked } : {}
    };
  }
  if (source?.kind === "synthetic") {
    notes.push({
      severity: "warn",
      title: "Synthetic data: proves it runs, says nothing about accuracy",
      detail: "train.py will feed random tensors shaped to the input contract. A run on them checks the wiring end to end; its loss and accuracy describe noise, not your task.",
      fix: "Pick a real dataset in the Data step when you have one.",
      // A dataset choice, not a graph edit.
      agentFixable: false
    });
    return {
      stage: "data",
      status: "attention",
      headline: "Synthetic data: proves it runs, says nothing about accuracy.",
      notes,
      data: { wired: false, synthetic: true, source, modality, contract },
      evidence: evidenceFor("data")
    };
  }
  return {
    stage: "data",
    status: "needs_input",
    headline: "No dataset is wired yet, so a run would train on random tensors.",
    notes,
    data: { wired: false, modality, contract },
    question: {
      id: "dataset",
      question: "Which data should this train on?",
      because: `Everything else about the run is derived from the graph. This is not: the design says it wants ${modality} input${contract.inputShape.length ? ` of shape (${contract.inputShape.join(", ")})` : ""}, but not which corpus you have.`,
      options: datasetOptionsFor(modality, contract.outputDim)
    },
    evidence: evidenceFor("data")
  };
}
function otherDesignNote(other) {
  return {
    severity: "info",
    title: "The last run was for a different design",
    detail: `${other.modelName ? `"${other.modelName}"` : "Another design"} trained last, and its structure differs from the one on the canvas. Its numbers say nothing about this design, so this one counts as untrained.`
  };
}
function diagnoseStoredRun(model, run) {
  return diagnoseRun({
    curve: run.curve ?? [],
    model,
    source: getDataSource(),
    measured: runEvidence(run).measured,
    final: { trainAcc: run.trainAcc, valAcc: run.valAcc, trainLoss: run.trainLoss, valLoss: run.valLoss },
    // The run's own row count when it recorded one, else the sample count the
    // user declared in pre-flight. Without either, tiny-data cannot fire, and
    // it used to be unable to fire at all because nothing passed this.
    datasetSize: run.datasetSize ?? readDataSpec()?.samples ?? null
  });
}
function runTrainStage(model) {
  if (realLayerCount(model) === 0) return emptyCanvasResult("train");
  const { run, otherDesign } = getLastRunFor(model);
  const plan = readTrainPlan();
  const report = buildPreflightReport(model, { dataSpec: readDataSpec(), plan });
  const cost = report.metrics.estCostUsd;
  const dur = report.metrics.estTrainSec;
  const gpu = report.metrics.fitsGpu;
  const costLine = dur > 0 ? `${fmtUsd(cost)} / ${fmtDuration(dur)} on ${gpuArticle(gpu)} ${gpu}` : "cost not estimable";
  const init = expectedInitialLoss(model, getDataSource());
  const stabilityNotes = stabilityFindings(model).map((f) => ({
    severity: f.severity,
    ruleId: f.ruleId,
    title: f.title,
    detail: f.detail,
    fix: f.fix,
    // Set per rule in trainStability.ts: QK-norm is a node, z-loss is a loss term.
    agentFixable: f.agentFixable
  }));
  const compute = report.metrics.compute;
  const computeNote = compute && compute.trainFlops > 0 ? {
    severity: "info",
    title: `Compute: ${compute.trainFlops.toExponential(2)} training FLOPs, about ${fmtDuration(compute.computeSec)} at an assumed ${Math.round(compute.assumedMfu * 100)}% utilisation`,
    detail: `${report.metrics.flopsDetail?.convention ?? "FLOPs count a multiply-add as 2"}. Per sample that is 3 x the forward FLOPs (about 6 x parameters per token for a dense model), times ${(report.metrics.samples ?? 0).toLocaleString("en-US")} samples and ${report.metrics.epochs ?? 10} epochs, over the ${gpu}'s ${compute.peakTflops} TFLOPS dense bf16 peak. The ${Math.round(compute.assumedMfu * 100)}% is an assumption, not a measurement; a short timed run replaces it.`
  } : null;
  if (run) {
    const ev = runEvidence(run);
    const measured = ev.measured;
    const diagnoses = diagnoseStoredRun(model, run);
    const notes = diagnoses.map((d) => ({
      // A cause that explains everything (the loss diverged, the graph does not
      // match its data) makes every number on the run a measurement of the
      // wrong thing. That is a stop, not a note to scroll past on the way to
      // Evaluate grading those numbers.
      severity: d.explainsEverything ? "block" : "warn",
      ruleId: `run-${d.kind}`,
      title: d.title,
      detail: `${d.evidence} ${d.fix}`,
      // Set per cause in runDiagnosis, never inferred from severity: "the
      // dataset is 300 rows" and "the learning rate blew up" are warnings the
      // agent cannot act on, and a fix button on either promises nothing.
      agentFixable: d.agentFixable
    }));
    for (const i of deriveRunInsights(run).filter((i2) => i2.severity !== "warn")) {
      notes.push({ severity: i.severity === "good" ? "good" : "info", title: i.title, detail: i.detail });
    }
    const peak = comparePeak(model, run);
    if (peak) notes.push(peakNote(peak));
    notes.push(...stabilityNotes);
    const accPart = typeof run.valAcc === "number" ? `val acc ${pct3(run.valAcc)}` : "no accuracy reported";
    const fatal = diagnoses.find((d) => d.explainsEverything);
    return {
      stage: "train",
      status: fatal ? "blocked" : diagnoses.length > 0 ? "attention" : measured ? "ok" : "attention",
      headline: fatal ? `${ev.sentence}, but ${fatal.title.toLowerCase()}, so its ${accPart} measures nothing useful.` : measured ? `${ev.sentence}: ${accPart} over ${fmtCount(run.epochs, "epoch")}.` : `${ev.sentence}. ${accPart}, and nothing has been measured on real data.`,
      notes,
      data: { lastRun: run, plan: plan ?? null, estCostUsd: cost, estTrainSec: dur, gpu, expectedInitialLoss: init, peakCheck: peak },
      evidence: evidenceFor("train")
    };
  }
  return {
    stage: "train",
    status: "needs_input",
    headline: `Not trained yet. A real run is estimated at ${costLine}.`,
    notes: [
      ...otherDesign ? [otherDesignNote(otherDesign)] : [],
      ...init ? [{
        severity: "info",
        ruleId: "expected-first-loss",
        title: `Expected first loss: ${init.sentence}`,
        detail: "What an uninformative head scores. A first logged loss far above it means the start was worse than guessing, which is checked against the run when one exists."
      }] : [],
      ...stabilityNotes,
      ...computeNote ? [computeNote] : [],
      {
        severity: "info",
        title: "Nothing is started without you",
        detail: "Picking an option here opens the runner with that choice made. The run itself stays behind its own confirm, because it spends time or money."
      }
    ],
    data: {
      lastRun: null,
      ...otherDesign ? { lastRunOtherDesign: { modelName: otherDesign.modelName ?? null, finishedAt: otherDesign.finishedAt } } : {},
      plan: plan ?? null,
      estCostUsd: cost,
      estTrainSec: dur,
      gpu,
      expectedInitialLoss: init,
      compute: compute ?? null
    },
    question: {
      id: "train-mode",
      question: "How do you want to train it?",
      because: "The estimate and the hardware fit are derived. Whether you want to spend the time or the money is not.",
      options: [
        { label: "Simulate it (free, instant)", value: "simulate", hint: "Architecture-based curves, useful to check the wiring, not a measurement" },
        {
          label: "Free GPU (Colab / Kaggle)",
          value: "free_gpu",
          // The cheapest real evidence that the graph can learn, offered as text:
          // nothing here launches it.
          hint: `A notebook you run yourself; results report back into the app. Worth a smoke test there first: train one fixed batch for about 100 steps and check the loss falls from ${init ? init.value.toFixed(2) : "its first value"} toward 0, then train once on zeroed inputs, a baseline the real inputs must beat`
        },
        {
          label: `Managed GPU (about ${fmtUsd(cost)})`,
          value: "managed_gpu",
          hint: `${fmtDuration(dur)} on ${gpuArticle(gpu)} ${gpu}` + (report.metrics.gpuFits === false ? ", which it does not fit as planned" : "") + (getDataSource()?.kind === "synthetic" ? ", on synthetic data, so it proves the wiring and measures nothing" : "") + ", confirmed in the runner before anything starts"
        }
      ]
    },
    evidence: evidenceFor("train")
  };
}
function classificationBaseline(model, source) {
  const contract = deriveModelContract(model);
  if (contract.taskType !== "classification") return null;
  const spec = readDataSpec();
  const ds = knownDatasetFor(source);
  const probe = readHfDatasetMeta(source);
  const meta = probe?.status === "ok" ? probe.meta : null;
  const classes = ds?.classes ?? (meta?.numClasses && meta.numClasses > 1 ? meta.numClasses : null) ?? (spec?.numClasses && spec.numClasses > 1 ? spec.numClasses : null) ?? (typeof contract.outputDim === "number" && contract.outputDim > 1 ? contract.outputDim : null);
  if (!classes || classes < 2) return null;
  const evalSplit = meta ? labelledEvalSplit(meta) : null;
  const fromMeta = evalSplit ?? meta?.stats.find((st) => st.split.toLowerCase() === "train") ?? null;
  const metaCounts = fromMeta?.labelFrequencies ? Object.values(fromMeta.labelFrequencies).filter((n2) => n2 > 0) : [];
  const counts = metaCounts.length > 1 ? metaCounts : (spec?.classCounts ?? []).filter((n2) => typeof n2 === "number" && n2 > 0);
  const total = counts.reduce((a, b) => a + b, 0);
  const majority = counts.length > 1 && total > 0 ? Math.max(...counts) / total : null;
  return {
    classes,
    chance: 1 / classes,
    majority,
    from: ds ? ds.label : meta?.numClasses ? meta.datasetId : spec?.numClasses ? "your declared data" : "the width of the output layer",
    ...metaCounts.length > 1 && fromMeta ? { majorityFrom: `${meta.datasetId} ${fromMeta.split}` } : {}
  };
}
function labelledEvalSplit(meta) {
  return meta.stats.find((st) => st.split.toLowerCase() !== "train" && st.labelFrequencies && (st.labelNoLabelProportion ?? 0) < 0.999) ?? null;
}
function classesImbalanced(meta, classCounts) {
  const train = meta?.stats.find((st) => st.split.toLowerCase() === "train");
  const counts = train?.labelFrequencies ? Object.values(train.labelFrequencies).filter((n2) => n2 > 0) : (classCounts ?? []).filter((n2) => typeof n2 === "number" && n2 > 0);
  if (counts.length < 2) return false;
  const total = counts.reduce((a, b) => a + b, 0);
  const pMin = Math.min(...counts) / total;
  const pMax = Math.max(...counts) / total;
  return pMin < IMBALANCE_FRACTION_OF_UNIFORM / counts.length || pMax > 0.7;
}
var EVAL_QUALIFIER_KEYS = /* @__PURE__ */ new Set([
  "n_samples",
  "ece_bins",
  "bootstrap_resamples",
  "f1_macro_ci_lo",
  "f1_macro_ci_hi"
]);
var ECE_WARN = 0.1;
function runEvaluateStage(model) {
  if (realLayerCount(model) === 0) return emptyCanvasResult("evaluate");
  const { run, otherDesign } = getLastRunFor(model);
  const suite = summarize(loadSuite(model));
  const source = getDataSource();
  const notes = [];
  if (suite.total === 0) {
    notes.push({
      severity: "warn",
      title: "No test cases",
      detail: "Nothing would catch a regression between one run and the next.",
      fix: "Starter cases can be derived from the input contract right now, for free. Refining them into real assertions is yours.",
      // Not agentFixable: adding test cases is not an edit to the graph. It is
      // something WE can do, which is a different and better answer.
      selfAction: { id: "suggest-eval-cases", label: "Add starter cases" }
    });
  } else {
    const ran = suite.passed + suite.failed + suite.errors;
    if (suite.failed > 0) {
      notes.push({
        severity: "block",
        title: `${fmtCount(suite.total, "test case")}: ${suite.passed} passed, ${suite.failed} failed`,
        detail: "A failing assertion means the model no longer does what the suite says it should."
      });
    } else if (ran === 0) {
      notes.push({
        severity: "warn",
        title: `${fmtCount(suite.total, "test case")}, none run yet`,
        detail: "Every case is still pending, so the suite has not checked anything about this model.",
        fix: "Run the suite from the Evaluate step after the next training run."
      });
    } else {
      notes.push({
        severity: suite.pending > 0 || suite.errors > 0 ? "info" : "good",
        title: `${fmtCount(suite.total, "test case")}: ${suite.passed} passed${suite.errors ? `, ${suite.errors} errored` : ""}${suite.pending ? `, ${suite.pending} pending` : ""}`,
        detail: suite.pending > 0 ? `${suite.pending} still pending a run.` : void 0
      });
    }
  }
  if (source?.kind === "synthetic") {
    notes.push({
      severity: "warn",
      title: "The data is synthetic",
      detail: "A run on random tensors proves the graph trains end to end. Its accuracy describes noise, so nothing here can say whether the model is any good."
    });
  }
  if (!run) {
    if (otherDesign) notes.unshift(otherDesignNote(otherDesign));
    return {
      stage: "evaluate",
      status: notes.some((n2) => n2.severity === "block") ? "blocked" : "attention",
      headline: "Nothing measured yet: there is no training run of this design to judge.",
      notes,
      data: { run: null, suite },
      evidence: evidenceFor("evaluate"),
      unchecked: "There is no training run of this design, so nothing has been evaluated."
    };
  }
  const measured = runEvidence(run).measured;
  const evalMetrics = run.evalMetrics ?? {};
  const metricNames = Object.keys(evalMetrics);
  const listed = metricNames.filter((k) => !EVAL_QUALIFIER_KEYS.has(k));
  if (metricNames.length) {
    notes.push({
      severity: "info",
      title: "evaluate.py on the trained checkpoint",
      detail: listed.slice(0, 6).map((k) => `${k.replace(/_/g, " ")} ${Math.abs(evalMetrics[k]) >= 100 ? evalMetrics[k].toFixed(1) : evalMetrics[k].toFixed(4)}`).join(", ")
    });
  }
  const diagnoses = diagnoseStoredRun(model, run);
  if (diagnoses.length > 0) {
    const fatal = diagnoses.some((d) => d.explainsEverything);
    notes.push({
      severity: fatal ? "block" : "warn",
      ruleId: "run-diagnosed",
      title: diagnoses.length === 1 ? `The run has a diagnosed cause: ${diagnoses[0].title.toLowerCase()}` : `The run has ${diagnoses.length} diagnosed causes`,
      detail: `${diagnoses.map((d) => d.title).join("; ")}. The evidence and the fix for each are on the Train row.`
    });
  }
  const metaProbe = readHfDatasetMeta(source);
  const meta = metaProbe?.status === "ok" ? metaProbe.meta : null;
  const testN = evalSampleCount(evalMetrics);
  const evalAcc = typeof evalMetrics.accuracy === "number" ? evalMetrics.accuracy : null;
  const scored = evalAcc != null && testN != null ? { acc: evalAcc, n: testN, label: "evaluate.py accuracy" } : null;
  const interval = scored ? wilsonInterval(scored.acc, scored.n) : null;
  const mde = scored ? minimumDetectableDifference(scored.acc, scored.n) : null;
  if (scored && interval && mde != null) {
    notes.push({
      severity: "info",
      ruleId: "eval-interval",
      title: `Accuracy ${pts(scored.acc)}%, 95% interval ${pts(interval.lo)} to ${pts(interval.hi)} (n = ${scored.n.toLocaleString("en-US")})`,
      detail: `Wilson score interval on ${scored.n.toLocaleString("en-US")} test rows. This test set cannot distinguish differences under ${pts(mde)} points between two models scored on separate sets of this size (5% significance, 80% power). Single run: seed variance is not measured.`
    });
  } else if (measured) {
    notes.push({
      severity: "info",
      ruleId: "eval-interval",
      title: "Test-set size not recorded, so no interval",
      detail: evalAcc != null ? "evaluate.py reported accuracy but not how many rows it scored, so there is no error bar and no smallest detectable difference. evaluate.py writes n_samples from this version on." : "The run reported validation accuracy without the number of rows behind it, so there is no error bar on it."
    });
  }
  const f1 = typeof evalMetrics.f1_macro === "number" ? evalMetrics.f1_macro : null;
  const f1Lo = evalMetrics.f1_macro_ci_lo;
  const f1Hi = evalMetrics.f1_macro_ci_hi;
  if (f1 != null && typeof f1Lo === "number" && typeof f1Hi === "number") {
    notes.push({
      severity: "info",
      ruleId: "eval-f1-interval",
      title: `Macro-F1 ${f1.toFixed(3)}, 95% bootstrap interval ${f1Lo.toFixed(3)} to ${f1Hi.toFixed(3)}`,
      detail: `${typeof evalMetrics.bootstrap_resamples === "number" ? evalMetrics.bootstrap_resamples.toLocaleString("en-US") : "Bootstrap"} resamples of the test rows with replacement, by evaluate.py. Two models whose intervals overlap this much are not separated by this test set; a paired comparison on the same rows (predictions.jsonl from each) is the sharper test.`
    });
  }
  const ece = typeof evalMetrics.ece === "number" ? evalMetrics.ece : null;
  if (ece != null) {
    const high = ece > ECE_WARN;
    notes.push({
      severity: high ? "warn" : "info",
      ruleId: "eval-calibration",
      title: `Calibration: expected calibration error ${ece.toFixed(3)}`,
      detail: `The average gap between the model's top-class confidence and how often it is right, over ${typeof evalMetrics.ece_bins === "number" ? evalMetrics.ece_bins : 15} equal-width confidence bins on the test rows. ` + (high ? `Above ${ECE_WARN} (our line, not a standard): its probabilities should not be read as probabilities, for thresholds or for deferring to a person.` : "Its confidence can be read roughly at face value on data like this test set."),
      ...high ? {
        fix: "Fit a single temperature on a held-out split after training (temperature scaling); it changes confidence, not predictions.",
        // A post-training step, not an edit to the graph.
        agentFixable: false
      } : {}
    });
  }
  const baseline = measured ? classificationBaseline(model, source) : null;
  const point = scored?.acc ?? (typeof run.valAcc === "number" ? run.valAcc : null);
  if (baseline && point != null) {
    const accName = scored ? "Test accuracy" : "Validation accuracy";
    const parts = [`chance is ${pct3(baseline.chance)} for ${baseline.classes} classes (from ${baseline.from})`];
    if (baseline.majority != null) parts.push(`always predicting the commonest class scores ${pct3(baseline.majority)}${baseline.majorityFrom ? ` (${baseline.majorityFrom})` : ""}`);
    const hasMajority = baseline.majority != null && baseline.majority > baseline.chance;
    const atMajority = hasMajority && (interval ? within(interval, baseline.majority) || scored.acc < baseline.majority : point <= baseline.majority + 0.01);
    const atChance = !atMajority && (interval ? within(interval, baseline.chance) || scored.acc < baseline.chance : false);
    const how = interval ? `the baseline is inside the 95% interval (${pts(interval.lo)} to ${pts(interval.hi)}), so this test set cannot tell the model from it` : "a point comparison only: the test-set size was not recorded, so this is not a statistical statement";
    if (atMajority) {
      notes.push({
        severity: "warn",
        ruleId: "eval-at-majority",
        title: "No better than predicting the commonest class",
        detail: `${accName} ${pct3(point)}; ${parts.join(", and ")}; ${how}. The model may have learned the class balance and nothing else.`,
        fix: "Look at per-class recall before reading the accuracy, and weight the loss by class.",
        agentFixable: false
      });
    } else if (atChance) {
      notes.push({
        severity: "warn",
        ruleId: "eval-at-chance",
        title: "Indistinguishable from chance",
        detail: `${accName} ${pct3(point)}; ${parts.join(", and ")}; ${how}.`,
        agentFixable: false
      });
    } else {
      notes.push({
        severity: "info",
        title: `${scored ? "Test" : "Val"} acc ${pct3(point)} against a ${pct3(Math.max(baseline.chance, baseline.majority ?? 0))} baseline`,
        detail: `${parts.join("; ")}${interval ? `; the baseline is outside the 95% interval` : ""}.`
      });
    }
  }
  const imbalanced = classesImbalanced(meta, readDataSpec()?.classCounts);
  const headlineMetric = metricNames.length ? pickHeadlineMetric(evalMetrics, { imbalanced }) : null;
  if (headlineMetric && headlineMetric.key !== "accuracy") {
    notes.push({
      severity: "info",
      ruleId: "eval-headline-metric",
      title: `Headline metric: ${headlineMetric.label} ${headlineMetric.value.toFixed(3)}`,
      detail: `Chosen because ${headlineMetric.because}.`
    });
  } else if (imbalanced && metricNames.length) {
    notes.push({
      severity: "warn",
      ruleId: "eval-headline-metric",
      title: "Only accuracy exists, and the classes are imbalanced",
      detail: "On imbalanced classes accuracy mostly measures the common ones. evaluate.py reported no macro-F1 or balanced accuracy for this run.",
      fix: "Re-export evaluate.py: it now writes balanced_accuracy and f1_macro.",
      agentFixable: false
    });
  } else if (imbalanced && measured) {
    notes.push({
      severity: "warn",
      ruleId: "eval-headline-metric",
      title: "Only accuracy exists, and the classes are imbalanced",
      detail: "The run reported accuracy from training only. On imbalanced classes that mostly measures the common ones.",
      fix: "Run evaluate.py on the checkpoint; it writes balanced_accuracy and f1_macro.",
      agentFixable: false
    });
  }
  const insights = deriveRunInsights(run).filter((i) => i.severity !== "warn");
  for (const i of insights) {
    notes.push({
      severity: i.severity === "good" ? "good" : "info",
      title: i.title,
      detail: i.detail
    });
  }
  const accPart = typeof run.valAcc === "number" ? `val acc ${pct3(run.valAcc)}` : "no accuracy reported";
  const status = notes.some((n2) => n2.severity === "block") ? "blocked" : notes.some((n2) => n2.severity === "warn") || !measured ? "attention" : "ok";
  return {
    stage: "evaluate",
    status,
    headline: measured ? `${accPart}${suite.total ? `, ${suite.passed}/${suite.total} test cases passing` : ", no test cases"}.` : `${accPart}, but simulated: this describes the architecture, not a trained model.`,
    notes,
    data: {
      run,
      suite,
      evalMetrics,
      baseline,
      // Added fields: the size behind the numbers and what follows from it.
      testN,
      interval,
      minimumDetectableDifference: mde,
      headlineMetric,
      imbalanced,
      ece,
      f1Interval: f1 != null && typeof f1Lo === "number" && typeof f1Hi === "number" ? { lo: f1Lo, hi: f1Hi } : null
    },
    evidence: evidenceFor("evaluate"),
    ...!measured ? { unchecked: "The run was simulated, so nothing about the trained model has been measured." } : {}
  };
}
function runDeployStage(model) {
  if (realLayerCount(model) === 0) return emptyCanvasResult("deploy");
  const all = PLATFORM_PROFILES.map((p) => analyzeForDeployment(model, p.id));
  const best = pickBestDeployTarget(all);
  const chosenId = getChosenDeployTarget();
  const chosen = chosenId ? all.find((r) => r.profile.id === chosenId) : void 0;
  const target = chosen ?? best;
  const blockers = target.recommendations.filter((r) => r.severity === "blocker");
  const warnings = target.recommendations.filter((r) => r.severity === "warning");
  const suggestions = target.recommendations.filter((r) => r.severity === "suggestion");
  const notes = [
    ...blockers.map((r) => ({ severity: "block", blocks: true, ...r.id === "kv-cache-fit" ? { ruleId: "deploy-kv-cache" } : {}, title: r.title, detail: r.message, agentFixable: true })),
    ...warnings.map((r) => ({ severity: "warn", title: r.title, detail: r.message, agentFixable: true }))
  ];
  const suitLabel = (r) => r.suitability ?? "not assessed";
  if (target.backend) {
    const b = target.backend;
    const unsupportedTypes = [...new Set(b.unsupported.map((u) => u.type))];
    notes.push({
      severity: target.suitability === "NO" ? "warn" : "info",
      ruleId: "deploy-suitability",
      title: `${b.backendLabel}: ${target.suitability}`,
      detail: `${target.suitabilityReason} ${b.supported} of ${b.assessed} layers supported, ${b.partitions} partition${b.partitions === 1 ? "" : "s"}${b.partitionSizes.length > 1 ? ` (sizes ${b.partitionSizes.join(", ")})` : ""}.` + (b.unsupported.length ? ` Unsupported: ${b.unsupported.slice(0, 4).map((u) => `${u.name} (${u.type}): ${u.reason}`).join(" ")}` : "") + (b.notAssessedTypes.length ? ` Not in our table, so not counted: ${b.notAssessedTypes.slice(0, 5).join(", ")}.` : "") + ` Source: ${b.source}`,
      ...target.suitability === "NO" && unsupportedTypes.length ? { fix: `Replace ${unsupportedTypes.slice(0, 3).join(", ")} with layers ${b.backendLabel} runs, or pick a target that runs them.` } : {},
      // A NO names layers to replace, which is a graph edit.
      agentFixable: target.suitability === "NO" && unsupportedTypes.length > 0
    });
  } else {
    notes.push({
      severity: "info",
      ruleId: "deploy-suitability",
      title: `${target.profile.label}: op coverage not assessed`,
      detail: target.suitabilityReason
    });
  }
  const q = target.quantized;
  notes.push({
    severity: "info",
    title: `int8 size ~${q.int8MB.toFixed(1)} MB, from ${q.fp32MB.toFixed(1)} MB float32`,
    detail: `${Math.round(q.quantizableShare * 100)}% of parameters sit in linear, conv and attention layers, the ones a post-training weight quantizer converts; norms, embeddings and the rest stay float32. Not a flat quarter of the float size.`
  });
  if (target.onnx.minOpset != null) {
    const top = target.onnx.requiredBy[0];
    notes.push({
      severity: "info",
      ruleId: "deploy-onnx-opset",
      title: `ONNX export needs opset ${target.onnx.minOpset} or later`,
      detail: `${top.type}: ${top.why}.` + (target.onnx.nativeFrom.length ? ` Native single ops from: ${target.onnx.nativeFrom.slice(0, 4).map((n2) => `${n2.type} ${n2.opset}`).join(", ")}; below those they are decomposed.` : "")
    });
  }
  if (target.kvCache?.fits) {
    const kv = target.kvCache;
    notes.push({
      severity: "info",
      ruleId: "deploy-kv-cache",
      title: `KV cache fits: ${kv.concurrentAtMax} concurrent request${kv.concurrentAtMax === 1 ? "" : "s"} at ${kv.maxModelLen.toLocaleString("en-US")} tokens`,
      detail: `${(kv.kvBytesPerToken / 1024).toFixed(0)} KiB of fp16 KV per token; ${(kv.availableBytes / 1024 ** 3).toFixed(2)} GiB left after weights at vLLM's default gpu_memory_utilization, less an assumed ${(kv.activationReserveBytes / 1024 ** 3).toFixed(2)} GiB activation reserve that vLLM measures and we cannot.`
    });
  }
  if (chosen) {
    notes.push({
      severity: "info",
      title: `Target is ${chosen.profile.label}, as chosen in the Deploy panel`,
      detail: chosen.profile.id !== best.profile.id ? `The best fit for this graph is ${best.profile.label} (${suitLabel(best)}).` : "It is also the best fit for this graph."
    });
  }
  const order = (r) => r.suitability === "YES" ? 0 : r.suitability === "MAYBE" ? 1 : r.suitability === null ? 2 : 3;
  const runners = all.filter((r) => r.profile.id !== target.profile.id).sort((a, b) => a.blockers - b.blockers || order(a) - order(b) || b.score - a.score).slice(0, 3);
  if (runners.length) {
    notes.push({
      severity: "info",
      title: "Other targets",
      detail: runners.map((r) => `${r.profile.label} ${suitLabel(r)}${r.blockers ? ` (${fmtCount(r.blockers, "blocker")})` : ""}`).join(", ") + ". Change the target in the Deploy panel to see its own analysis."
    });
  }
  const { run, otherDesign } = getLastRunFor(model);
  const deployEv = runEvidence(run);
  if (otherDesign) notes.push(otherDesignNote(otherDesign));
  if (!deployEv.measured) {
    notes.push({
      severity: "warn",
      title: run ? "Readiness is unmeasured" : "Never trained",
      detail: run ? `${deployEv.sentence}, so nothing below is evidence about the trained model, only about the architecture.` : "This analysis is about the architecture. Nothing has been trained, so there are no weights to ship."
    });
  }
  const who = chosen ? `your target, ${target.profile.label}` : `${target.profile.label}, the best fit`;
  const verdictTxt = target.backend ? `${target.backend.backendLabel}: ${target.suitability}` : "op coverage not assessed";
  return {
    stage: "deploy",
    status: blockers.length ? "blocked" : warnings.length || !deployEv.measured || target.suitability === "NO" ? "attention" : "ok",
    headline: blockers.length ? `${fmtCount(blockers.length, "blocker")} for ${who}.` : chosen ? `Your target is ${target.profile.label} (${verdictTxt}): ${target.estimatedLatencyLabel} per inference, ${target.totalParamsMB.toFixed(1)} MB.` : `Best target is ${target.profile.label} (${verdictTxt}): ${target.estimatedLatencyLabel} per inference, ${target.totalParamsMB.toFixed(1)} MB.`,
    notes,
    data: {
      target: target.profile.id,
      targetLabel: target.profile.label,
      // Who picked it: the user in the Deploy panel, or the ranking. The value
      // 'score' is kept for API compatibility; the ranking now orders by
      // blockers, then suitability, then score.
      targetChosenBy: chosen ? "user" : "score",
      bestTarget: best.profile.id,
      score: target.score,
      latencyMs: target.estimatedLatencyMs,
      sizeMB: target.totalParamsMB,
      sizeMBQuantized: target.totalParamsMBQuantized,
      blockers: blockers.map((r) => r.title),
      warnings: warnings.map((r) => r.title),
      suggestions: suggestions.map((r) => r.title),
      alternatives: all.map((r) => ({ id: r.profile.id, label: r.profile.label, score: r.score, suitability: r.suitability })),
      // Added fields.
      suitability: target.suitability,
      suitabilityReason: target.suitabilityReason,
      backend: target.backend ? {
        id: target.backend.backend,
        label: target.backend.backendLabel,
        source: target.backend.source,
        supportedShare: target.backend.supportedShare,
        supported: target.backend.supported,
        assessed: target.backend.assessed,
        partitions: target.backend.partitions,
        partitionSizes: target.backend.partitionSizes,
        unsupported: target.backend.unsupported,
        notAssessedTypes: target.backend.notAssessedTypes
      } : null,
      quantized: target.quantized,
      onnx: target.onnx,
      kvCache: target.kvCache
    },
    evidence: evidenceFor("deploy"),
    ...!deployEv.measured ? { unchecked: run ? "Readiness is unmeasured: the run was not a real one." : "Never trained, so there are no weights to judge." } : {}
  };
}
function runStage(stage, model, opts) {
  switch (stage) {
    case "preflight":
      return runPreflightStage(model, opts?.workload);
    case "data":
      return runDataStage(model);
    case "train":
      return runTrainStage(model);
    case "evaluate":
      return runEvaluateStage(model);
    case "deploy":
      return runDeployStage(model);
  }
}
function runStageSafely(stage, model, opts) {
  try {
    return runStage(stage, model, opts);
  } catch (e) {
    return {
      stage,
      status: "blocked",
      headline: `${STAGE_LABEL[stage]} could not be checked: the design could not be read.`,
      notes: [{
        severity: "block",
        title: "This check failed to run",
        detail: e instanceof Error ? e.message : String(e),
        fix: `Open ${STAGE_LABEL[stage]} to see the full state.`,
        // Not the agent's to fix: this is our code failing, not a graph edit.
        agentFixable: false
      }],
      data: { error: true },
      evidence: evidenceFor(stage)
    };
  }
}
var GRAPH_ONLY_STAGES = /* @__PURE__ */ new Set(["data", "deploy"]);
function provisionalAfter(stoppedAt, model, opts) {
  const out = [];
  for (const stage of STAGE_ORDER.slice(STAGE_ORDER.indexOf(stoppedAt) + 1)) {
    if (!GRAPH_ONLY_STAGES.has(stage)) continue;
    if (stage === "data" && !hasRealData(getDataSource())) continue;
    const r = runStageSafely(stage, model, opts);
    if (r.data.error) continue;
    out.push({ ...r, provisional: true });
  }
  return out;
}
function runPipeline(model, opts) {
  const results = [];
  let stoppedAt = null;
  let question;
  if (!hasDesign(model) && Array.isArray(model?.components) && model.components.every((c) => c && typeof c === "object")) {
    const r = emptyCanvasResult("preflight");
    return {
      results: [r],
      provisional: [],
      stoppedAt: "preflight",
      outcome: "blocked",
      headline: r.headline,
      empty: true,
      ranAt: Date.now()
    };
  }
  for (const stage of STAGE_ORDER) {
    const r = runStageSafely(stage, model, opts);
    results.push(r);
    if (r.status === "needs_input" && !question) question = r.question;
    if (r.status === "blocked" || r.status === "needs_input" && !opts?.continuePastQuestions) {
      stoppedAt = stage;
      break;
    }
  }
  const blocked = results.some((r) => r.status === "blocked");
  const outcome = blocked ? "blocked" : question ? "needs_input" : "complete";
  const lead = (stoppedAt ? results.find((r) => r.stage === stoppedAt) : void 0) ?? results.find((r) => r.status === "blocked") ?? results.find((r) => r.status === "needs_input") ?? results[results.length - 1];
  return {
    results,
    provisional: stoppedAt ? provisionalAfter(stoppedAt, model, opts) : [],
    stoppedAt,
    outcome,
    headline: lead.headline,
    question,
    ranAt: Date.now()
  };
}
function summarizeVerdict(report) {
  const stopping = report.stoppedAt ? report.results.find((r) => r.stage === report.stoppedAt) : void 0;
  const blockerCount = stopping ? stopping.notes.filter((n2) => n2.severity === "block").length : 0;
  const statuses = new Set(report.results.map((r) => r.status));
  const tone = statuses.has("blocked") ? "block" : statuses.has("needs_input") ? "ask" : statuses.has("attention") ? "warn" : "ok";
  let label;
  if (report.results.length === 1 && report.results[0].data.empty) {
    label = "no design";
  } else if (stopping?.data.error) {
    label = "check failed";
  } else if (report.outcome === "blocked") {
    label = blockerCount > 0 ? fmtCount(blockerCount, "blocker") : "blocked";
  } else if (report.stoppedAt === "data") {
    label = "no data";
  } else if (report.stoppedAt === "train") {
    label = "not trained";
  } else if (report.outcome === "needs_input") {
    label = "your call";
  } else {
    const noisy = report.results.filter((r) => r.status === "attention").length;
    label = noisy > 0 ? fmtCount(noisy, "note") : "ready";
  }
  return {
    outcome: report.outcome,
    label,
    tone,
    stoppedAt: report.stoppedAt,
    blockerCount,
    detail: report.headline
  };
}
function checkDesign(model) {
  const report = runPipeline(model, { continuePastQuestions: true });
  const verdict = summarizeVerdict(report);
  const findings = [];
  for (const r of report.results) {
    for (const n2 of r.notes) {
      if (n2.severity !== "block" && n2.severity !== "warn") continue;
      findings.push({ stage: r.stage, severity: n2.severity, blocks: n2.blocks ?? n2.severity === "block", ruleId: n2.ruleId, title: n2.title, detail: n2.detail, fix: n2.fix });
    }
  }
  findings.sort((a, b) => (a.severity === "block" ? 0 : 1) - (b.severity === "block" ? 0 : 1));
  return {
    verdict: verdict.tone,
    outcome: report.outcome,
    summary: report.headline,
    stoppedAt: report.stoppedAt,
    findings,
    stages: report.results.map((r) => ({
      stage: r.stage,
      status: r.status,
      headline: r.headline,
      data: r.data
    })),
    provisionalStages: (report.provisional ?? []).map((r) => ({
      stage: r.stage,
      status: r.status,
      headline: r.headline,
      data: r.data,
      findings: r.notes.filter((n2) => n2.severity === "block" || n2.severity === "warn").map((n2) => ({ stage: r.stage, severity: n2.severity, blocks: false, ruleId: n2.ruleId, title: n2.title, detail: n2.detail, fix: n2.fix }))
    })),
    decision: report.question ? {
      question: report.question.question,
      because: report.question.because,
      options: report.question.options
    } : void 0
  };
}

// lib/ruleProvenance.ts
var STRUCTURAL_STUDY = "https://neurarch.com/docs/structural-checks";
var CRASH_STUDY = {
  evidence: "In a 264-graph study (two seeds, torch 2.8), all 96 graphs blocked by the structural checks crashed PyTorch forward and all 80 that passed ran clean.",
  source: STRUCTURAL_STUDY,
  kind: "crash"
};
var RULE_PROVENANCE = {
  "head-dim-divisibility": CRASH_STUDY,
  "gqa-head-divisibility": CRASH_STUDY,
  "merge-shape-mismatch": CRASH_STUDY,
  "compute-error": CRASH_STUDY,
  "linear-in-mismatch": CRASH_STUDY,
  "invalid-output-shape": CRASH_STUDY,
  "unknown-layer-type": {
    evidence: "Outcome-derived: in a grounded training campaign a statically-clean design carrying an unknown layer type trained to below-random accuracy because the trainer dropped the layer; the same design trained to 66.9% once the type resolved.",
    source: STRUCTURAL_STUDY,
    kind: "outcome"
  },
  "deep-no-norm": {
    evidence: "Outcome-mined from the production training corpus: unnormalized conv-run length correlates at Spearman -0.53 with trained accuracy (n=15 distinct designs), and every run-of-6 design reached only 15-27% of its reference under an identical budget.",
    source: STRUCTURAL_STUDY,
    kind: "outcome"
  }
};
var OUTCOME_RULE_IDS = Object.entries(RULE_PROVENANCE).filter(([, p]) => p.kind === "outcome").map(([id]) => id);
function provenanceFor(ruleIds) {
  const out = {};
  for (const id of ruleIds) {
    if (id && RULE_PROVENANCE[id] && !out[id]) out[id] = RULE_PROVENANCE[id];
  }
  return out;
}

// lib/rankCandidates.ts
var RANK_CALIBRATION = {
  exclusion: {
    claim: "A design with a pre-flight blocker does not forward-pass.",
    evidence: "264-graph study, torch 2.8: 96 of 96 blocked graphs crashed PyTorch forward, 80 of 80 passes ran clean."
  },
  ordering: {
    claim: "Ordering two legal designs is weakly supported and usually abstains.",
    pairwiseAccuracy: 0.514,
    coverage: 0.083,
    sampleSize: 36,
    /**
     * How many of those 36 pairs the score actually took a position on. Three.
     * The accuracy above is 33 abstentions scored as coin flips plus three real
     * comparisons, and it is shipped with `quotable: false` because a caller
     * comparing 51.4% to the papers' 61.5% would be comparing an anecdote to a
     * measurement. See MIN_DECIDED_PAIRS in scripts/rl-benchmark/selection.ts.
     */
    decidedPairs: 3,
    quotable: false,
    basis: "in-sample: 24 designs over 6 tasks trained end to end on one T4 (scripts/rl-benchmark/grounded-results.json)",
    comparators: {
      chance: 0.5,
      "zheng-2026-best": 0.615,
      "foster-2026-best": 0.6935
    }
  },
  /**
   * The out-of-sample campaign exists and did not move this. 2026-09-01, five
   * held-out tasks and two held-out designers on a T4: fifteen designs trained,
   * fifteen within-task pairs, four decided, three of them correct. Still under
   * MIN_DECIDED_PAIRS, so the accuracy is not promoted here.
   *
   * What it did reproduce is the abstention: the score declined to separate 11
   * of 15 pairs of legal designs on tasks and designers it had never seen. That
   * is the claim this endpoint's `recommended: null` is built on, and unlike an
   * accuracy it does not need a sample size to be believed.
   */
  outOfSample: {
    ran: "2026-09-01",
    trainedRows: 15,
    judgedPairs: 15,
    decidedPairs: 4,
    abstained: 11,
    quotable: false,
    note: "Held-out campaign reproduced the abstention (11 of 15 pairs) but decided too few pairs to promote an accuracy. See scripts/rl-benchmark/CALIBRATION.md.",
    /**
     * The objection, measured. 2026-09-01: two frontier models read the
     * PyTorch exported from the same fifteen pairs, asked in both orderings and
     * tallied by the same rules (`npm run bench:llm-baseline`). Both decided
     * every pair, neither flipped on ordering, and both were right about three
     * quarters of the time. This score decided four. Shipped here so a caller
     * choosing between this ordering and a model call has both numbers, and so
     * this endpoint cannot imply a ranking ability the harness has measured it
     * not to have. Pinned to llm-baseline-results.json by test.
     */
    codeReadingJudges: {
      basis: "same 15 held-out pairs, exported PyTorch, both orderings, position flips scored as abstentions (scripts/rl-benchmark/llm-baseline-results.json)",
      "claude-opus-5": { decidedPairs: 15, correct: 12, pairwiseAccuracy: 0.8, coverage: 1 },
      "grok-4.3": { decidedPairs: 15, correct: 11, pairwiseAccuracy: 0.7333, coverage: 1 }
    },
    /**
     * The rule with no model in it, on the same fifteen pairs: always pick the
     * design with more parameters. LLMRouter's largest_llm baseline, Zheng et
     * al.'s complexity heuristic. It ties Grok 4.3 here, which is why the
     * judges above are not shipped without it. Pinned to what
     * scripts/rl-benchmark/selectors.ts computes (SELECTORS.md).
     */
    trivialBaseline: {
      id: "largest-params",
      decidedPairs: 15,
      correct: 11,
      pairwiseAccuracy: 0.7333,
      coverage: 1,
      note: "A no-model rule within a few pairs of both judges on this set. Neither judge number is evidence that reading code sees what parameter count does not."
    }
  },
  source: "https://neurarch.com/docs/structural-checks"
};
var cmp = (a, b) => (a.tier === b.tier ? 0 : a.tier === "blocked" ? 1 : -1) || a.outcomeFlags.length - b.outcomeFlags.length || a.warnings - b.warnings;
function reasonsFor(c, tier) {
  const out = [];
  if (tier === "blocked") {
    out.push(`${c.blocking} blocking finding${c.blocking === 1 ? "" : "s"}: this graph does not forward-pass, so executing it spends budget on a crash.`);
    return out;
  }
  if (c.outcomeFlags.length > 0) {
    out.push(`Carries ${c.outcomeFlags.length} finding${c.outcomeFlags.length === 1 ? "" : "s"} measured against trained outcomes (${c.outcomeFlags.join(", ")}), which is the only evidence here that separates two runnable designs.`);
  } else {
    out.push("Runs, and trips no rule with a trained-outcome measurement behind it.");
  }
  if (c.warnings > 0) out.push(`${c.warnings} pre-flight warning${c.warnings === 1 ? "" : "s"} with no outcome measurement behind them; used only to break ties.`);
  for (const b of c.otherStageBlockers ?? []) {
    out.push(`Blocked at ${b.stage}: ${b.title}. Reported, not ranked on: it does not bear on whether training this design is worth the budget.`);
  }
  return out;
}
function signalsFromCheck(id, check, firedRuleIds, outcomeRuleIds) {
  const c = check;
  if (!c || typeof c !== "object") return null;
  if (!Array.isArray(c.findings) || !Array.isArray(c.stages)) return null;
  const findings = c.findings.filter(
    (f) => !!f && typeof f === "object"
  );
  const stages = c.stages.filter((s) => !!s && typeof s === "object");
  const outcome = new Set(outcomeRuleIds);
  const preflight = stages.find((s) => s.stage === "preflight");
  const data = preflight?.data ?? {};
  const num4 = (k) => typeof data[k] === "number" && Number.isFinite(data[k]) ? data[k] : null;
  const blocks = findings.filter((f) => f.severity === "block");
  return {
    id,
    blocking: blocks.filter((f) => f.stage === "preflight").length,
    warnings: findings.filter((f) => f.severity === "warn" && f.stage === "preflight").length,
    otherStageBlockers: blocks.filter((f) => f.stage !== "preflight").map((f) => ({ stage: f.stage, title: f.title })),
    outcomeFlags: [...new Set([...firedRuleIds].filter((r) => r && outcome.has(r)))],
    params: num4("params"),
    estCostUsd: num4("estCostUsd"),
    fitsGpu: typeof data.gpu === "string" ? data.gpu : null,
    summary: typeof c.summary === "string" ? c.summary : void 0
  };
}
var TIE_KEY = {
  cost: { of: (c) => c.estCostUsd, noun: "estimated cost" },
  params: { of: (c) => c.params, noun: "parameter count" }
};
function rankCandidates(candidates, opts = {}) {
  const tieBreak = opts.tieBreak ?? "none";
  const withTier = candidates.map((c) => {
    const tier = c.blocking > 0 ? "blocked" : "legal";
    return { ...c, tier, rank: 0, measuredRank: 0, tiedWith: 1, reasons: reasonsFor(c, tier) };
  });
  const sorted = [...withTier].sort(cmp);
  let rank = 0;
  for (let i = 0; i < sorted.length; i++) {
    if (i === 0 || cmp(sorted[i - 1], sorted[i]) !== 0) rank = i + 1;
    sorted[i].measuredRank = rank;
    sorted[i].rank = rank;
  }
  let decidedByTieBreak = 0;
  if (tieBreak !== "none") {
    const key = TIE_KEY[tieBreak];
    const groups = /* @__PURE__ */ new Map();
    for (const c of sorted) if (c.tier === "legal") groups.set(c.measuredRank, [...groups.get(c.measuredRank) ?? [], c]);
    for (const [measuredRank, group] of groups) {
      if (group.length < 2) continue;
      const known = group.filter((c) => key.of(c) != null).sort((a, b) => key.of(a) - key.of(b));
      const unknown = group.filter((c) => key.of(c) == null);
      const ordered = [...known, ...unknown];
      let sub = 0;
      for (let i = 0; i < ordered.length; i++) {
        const prev = i > 0 ? ordered[i - 1] : null;
        const same = prev != null && key.of(prev) != null && key.of(ordered[i]) != null && key.of(prev) === key.of(ordered[i]);
        const bothUnknown = prev != null && key.of(prev) == null && key.of(ordered[i]) == null;
        if (i === 0 || !(same || bothUnknown)) sub = i;
        ordered[i].rank = measuredRank + sub;
      }
      const distinct = new Set(ordered.map((c) => c.rank)).size;
      if (distinct > 1) {
        decidedByTieBreak += ordered.length;
        for (const c of ordered) {
          const v = key.of(c);
          c.reasons.push(
            v == null ? `Tied at measured rank ${measuredRank} with ${group.length - 1} other${group.length === 2 ? "" : "s"}; placed last among them because its ${key.noun} is unknown, on your tieBreak rule, not on any measurement.` : `Tied at measured rank ${measuredRank} with ${group.length - 1} other${group.length === 2 ? "" : "s"}; placed by ${key.noun} (${v}) on your tieBreak rule. That is a budget decision, not a claim about the design.`
          );
        }
      }
    }
    sorted.sort((a, b) => a.rank - b.rank || cmp(a, b));
  }
  const groupSize = /* @__PURE__ */ new Map();
  for (const c of sorted) groupSize.set(c.rank, (groupSize.get(c.rank) ?? 0) + 1);
  for (const c of sorted) c.tiedWith = groupSize.get(c.rank) ?? 1;
  const legal = sorted.filter((c) => c.tier === "legal");
  const blocked = sorted.filter((c) => c.tier === "blocked");
  let recommended = null;
  let recommendation;
  if (legal.length === 0) {
    recommendation = blocked.length === 0 ? "No candidates were submitted." : "Every candidate has a blocking finding. None of them will forward-pass, so none should be given execution budget as submitted.";
  } else if (legal[0].tiedWith > 1) {
    recommendation = `${legal[0].tiedWith} of ${legal.length} legal candidates are tied at the top and nothing measured separates them. ` + (tieBreak === "none" ? 'Pick on your own budget (params, cost and GPU fit are returned per candidate), pass tieBreak: "cost" or "params" to have that done for you, or run more than one: ' : `Your tieBreak rule (${tieBreak}) could not separate them either. Run more than one: `) + "a verifier that abstains is telling you the truth about what it can see.";
  } else if (legal.length > 1 && legal[1].measuredRank === legal[0].measuredRank) {
    recommended = legal[0].id;
    const tiedAtMeasured = legal.filter((c) => c.measuredRank === legal[0].measuredRank).length;
    recommendation = `${legal[0].id} sits alone at the top because you asked ties to be broken on ${tieBreak}. Nothing measured separates it from ${tiedAtMeasured - 1} other${tiedAtMeasured === 2 ? "" : "s"} at measured rank ${legal[0].measuredRank}; the choice is your budget's, not the verifier's.`;
  } else {
    recommended = legal[0].id;
    recommendation = `${legal[0].id} is the only candidate at the top rank. ${legal[0].reasons[0]}`;
  }
  return {
    ranked: sorted,
    recommended,
    recommendation,
    budget: {
      candidates: sorted.length,
      blocked: blocked.length,
      legal: legal.length,
      wouldNotRun: blocked.map((c) => c.id),
      reclaimed: sorted.length === 0 ? 0 : blocked.length / sorted.length
    },
    tieBreak: {
      rule: tieBreak,
      decided: decidedByTieBreak,
      note: tieBreak === "none" ? 'No tie-break requested: ties are reported as ties. Pass tieBreak: "cost" or "params" to break them on your own budget.' : decidedByTieBreak === 0 ? `tieBreak: "${tieBreak}" was requested and decided nothing: no measured tie among legal candidates could be separated by ${TIE_KEY[tieBreak].noun}.` : `${decidedByTieBreak} candidate${decidedByTieBreak === 1 ? "" : "s"} were placed by ${TIE_KEY[tieBreak].noun} inside a measured tie. Those positions are your rule, not a measurement; measuredRank is what the verifier said.`
    },
    calibration: RANK_CALIBRATION
  };
}

// lib/normalizeGraph.ts
function normalizeGraphConnections(graph) {
  const g = graph;
  if (!g || typeof g !== "object") return graph;
  const components = g.components;
  const connections = g.connections;
  if (!Array.isArray(components) || !Array.isArray(connections)) return graph;
  const ids = /* @__PURE__ */ new Set();
  for (const c of components) {
    if (c && typeof c.id === "string") ids.add(c.id);
  }
  const nameCounts = /* @__PURE__ */ new Map();
  for (const c of components) {
    if (c && typeof c.name === "string") {
      nameCounts.set(c.name, (nameCounts.get(c.name) ?? 0) + 1);
    }
  }
  const idByName = /* @__PURE__ */ new Map();
  for (const c of components) {
    if (!c || typeof c.name !== "string" || typeof c.id !== "string") continue;
    if (nameCounts.get(c.name) === 1 && !ids.has(c.name)) idByName.set(c.name, c.id);
  }
  const resolve = (endpoint) => {
    if (typeof endpoint !== "string") return endpoint;
    if (ids.has(endpoint)) return endpoint;
    return idByName.get(endpoint) ?? endpoint;
  };
  let changed = false;
  const rewritten = connections.map((conn) => {
    if (!conn || typeof conn !== "object") return conn;
    const from = resolve(conn.from);
    const to = resolve(conn.to);
    if (from === conn.from && to === conn.to) return conn;
    changed = true;
    return { ...conn, from, to };
  });
  return changed ? { ...g, connections: rewritten } : graph;
}
function ensureComponentIds(graph) {
  const g = graph;
  if (!g || typeof g !== "object" || !Array.isArray(g.components)) return graph;
  const components = g.components;
  if (components.length === 0) return graph;
  if (components.some((c) => c && typeof c.id === "string" && c.id)) return graph;
  return {
    ...g,
    components: components.map((c, i) => c && typeof c === "object" ? { ...c, id: `n${i}` } : c)
  };
}
function ensureComponentParams(graph) {
  const g = graph;
  if (!g || typeof g !== "object" || !Array.isArray(g.components)) return graph;
  const components = g.components;
  if (components.every((c) => !c || typeof c !== "object" || c.params)) return graph;
  return {
    ...g,
    components: components.map((c) => c && typeof c === "object" && !c.params ? { ...c, params: {} } : c)
  };
}
function normalizeGraphForVerification(graph) {
  return normalizeGraphConnections(ensureComponentParams(ensureComponentIds(graph)));
}
export {
  OUTCOME_RULE_IDS,
  RANK_CALIBRATION,
  RULE_PROVENANCE,
  checkDesign,
  normalizeGraphForVerification,
  provenanceFor,
  rankCandidates,
  signalsFromCheck
};
