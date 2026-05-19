/**
 * FLOPs estimator — Multiply-Accumulate Operations (MACs) per layer.
 *
 * Convention: 1 MAC = 2 FLOPs (one multiply + one add).
 * We return MACs; the caller can display as "MACs" or "FLOPs × 2".
 *
 * All estimates assume batch_size = 1 (caller multiplies by batch if needed).
 */

/** Safe number coercion */
function n(v: any, fallback = 0): number {
  const x = typeof v === 'number' ? v : Number(v);
  return isFinite(x) && x > 0 ? x : fallback;
}

/** Spatial element count from shape (everything after batch + channel dims) */
function spatialElements(shape: number[]): number {
  if (shape.length < 3) return 1;
  return shape.slice(2).reduce((a, b) => a * b, 1);
}

/**
 * Estimate MACs (Multiply-Accumulate Operations) for a single layer.
 * Returns 0 if the layer has no multiply-accumulate ops (activations, pooling, etc.)
 */
export function estimateLayerFlops(
  type: string,
  params: Record<string, any>,
  inputShape: number[],  // no batch dim — e.g. [3, 224, 224] or [512]
  outputShape: number[], // same convention
): number {
  const p = params;

  switch (type) {
    // ── Linear / Dense ──────────────────────────────────────────────────────
    case 'linear': {
      const inF  = n(p.inFeatures  ?? inputShape[inputShape.length - 1]);
      const outF = n(p.outFeatures ?? outputShape[outputShape.length - 1]);
      // Batched: all dims except the last are "batch" — just multiply them through
      const batch = outputShape.slice(0, -1).reduce((a, b) => a * b, 1) || 1;
      return inF * outF * batch; // MACs
    }

    // ── Convolutions ─────────────────────────────────────────────────────────
    case 'conv2d': {
      const Cin  = n(p.inChannels  ?? inputShape[0],  1);
      const Cout = n(p.outChannels ?? outputShape[0], 1);
      const kH   = n(p.kernelSize, 3);
      const kW   = typeof p.kernelSize === 'object' ? n(p.kernelSize[1], kH) : kH;
      const groups = n(p.groups, 1);
      const Hout = outputShape[1] ?? 1;
      const Wout = outputShape[2] ?? 1;
      return (Cin / groups) * Cout * kH * kW * Hout * Wout;
    }
    case 'conv1d': {
      const Cin  = n(p.inChannels  ?? inputShape[0], 1);
      const Cout = n(p.outChannels ?? outputShape[0], 1);
      const kL   = n(p.kernelSize, 3);
      const Lout = outputShape[1] ?? 1;
      return Cin * Cout * kL * Lout;
    }
    case 'conv3d': {
      const Cin  = n(p.inChannels  ?? inputShape[0], 1);
      const Cout = n(p.outChannels ?? outputShape[0], 1);
      const k    = n(p.kernelSize, 3);
      const spat = spatialElements(outputShape);
      return Cin * Cout * k * k * k * spat;
    }
    case 'depthwiseConv2d': {
      const Cin = n(p.inChannels ?? inputShape[0], 1);
      const dm  = n(p.depthMultiplier, 1);
      const k   = n(p.kernelSize, 3);
      const spat = spatialElements(outputShape);
      return Cin * dm * k * k * spat;
    }
    case 'separableConv2d': {
      const Cin  = n(p.inChannels  ?? inputShape[0], 1);
      const Cout = n(p.outChannels ?? outputShape[0], Cin);
      const k    = n(p.kernelSize, 3);
      const spat = spatialElements(outputShape);
      const depthwise  = Cin * k * k * spat;
      const pointwise  = Cin * Cout * spat;
      return depthwise + pointwise;
    }
    case 'transposeConv2d': {
      // Same flop formula as conv (input and output swap roles)
      const Cin  = n(p.inChannels  ?? inputShape[0], 1);
      const Cout = n(p.outChannels ?? outputShape[0], 1);
      const k    = n(p.kernelSize, 3);
      const spat = spatialElements(outputShape);
      return Cin * Cout * k * k * spat;
    }

    // ── Attention ─────────────────────────────────────────────────────────────
    case 'multiHeadAttention':
    case 'attention':
    case 'selfAttention':
    case 'crossModalAttention':
    case 'causalAttention': {
      const D  = n(p.embedDim ?? p.hiddenDim ?? inputShape[inputShape.length - 1]);
      const T  = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      if (D === 0) return 0;
      // QKV projections: 3 × T × D × D
      const qkv = 3 * T * D * D;
      // Attention matrix: T × T × D (softmax negligible)
      const attn = T * T * D;
      // Output projection: T × D × D
      const out = T * D * D;
      return qkv + attn + out;
    }
    case 'windowAttention': {
      // Per-window attention: W²×W² instead of T×T; W = windowSize
      const D  = n(p.embedDim, 96);
      const W  = n(p.windowSize, 7);
      const T  = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : W * W;
      const numWindows = Math.max(1, Math.round(T / (W * W)));
      const wsq = W * W;
      return numWindows * (3 * wsq * D * D + wsq * wsq * D + wsq * D * D);
    }
    case 'groupedQueryAttention': {
      const D   = n(p.embedDim, 4096);
      const H   = n(p.numHeads, 32);
      const Hkv = n(p.numKVHeads, 8);
      const T   = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      if (D === 0) return 0;
      const headDim = H > 0 ? Math.floor(D / H) : 128;
      // Q proj + KV proj (reduced) + attn + out proj
      const q_proj = T * D * D;
      const kv_proj = 2 * T * D * (headDim * Hkv);
      const attn_w = T * T * D;
      const o_proj = T * D * D;
      return q_proj + kv_proj + attn_w + o_proj;
    }

    // ── Recurrent ─────────────────────────────────────────────────────────────
    case 'lstm': {
      const H = n(p.hiddenSize, 128);
      const I = n(p.inputSize ?? inputShape[inputShape.length - 1], H);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      const L = n(p.numLayers, 1);
      // Per gate, per timestep: I→H + H→H; 4 gates for LSTM
      const layer0 = 4 * T * (I * H + H * H);
      const layerRest = L > 1 ? (L - 1) * 4 * T * 2 * H * H : 0;
      return layer0 + layerRest;
    }
    case 'gru': {
      const H = n(p.hiddenSize, 128);
      const I = n(p.inputSize ?? inputShape[inputShape.length - 1], H);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      const L = n(p.numLayers, 1);
      const layer0 = 3 * T * (I * H + H * H);
      const layerRest = L > 1 ? (L - 1) * 3 * T * 2 * H * H : 0;
      return layer0 + layerRest;
    }
    case 'rnn': {
      const H = n(p.hiddenSize, 128);
      const I = n(p.inputSize ?? inputShape[inputShape.length - 1], H);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return T * (I * H + H * H);
    }
    case 'bidirectionalLSTM': {
      const H = n(p.hiddenSize, 128);
      const I = n(p.inputSize ?? inputShape[inputShape.length - 1], H);
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return 2 * 4 * T * (I * H + H * H);
    }

    // ── Transformer block ─────────────────────────────────────────────────────
    case 'transformerBlock': {
      const D  = n(p.embedDim ?? p.hiddenDim);
      const ff = n(p.ffDim, D > 0 ? D * 4 : 0);
      const T  = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      if (D === 0) return 0;
      const mha = 4 * T * D * D + T * T * D; // QKV+O proj + attn weights
      const ffn = 2 * T * D * ff;             // up + down projections
      return mha + ffn;
    }

    // ── Feed-forward (MLP block) ──────────────────────────────────────────────
    case 'feedForward': {
      const D  = n(p.embedDim ?? p.hiddenDim);
      const ff = n(p.ffDim, D > 0 ? D * 4 : 0);
      const T  = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      return D > 0 && ff > 0 ? 2 * T * D * ff : 0;
    }

    // ── Normalization (lightweight but non-zero) ──────────────────────────────
    case 'batchNorm':
    case 'layerNorm':
    case 'rmsNorm':
    case 'instanceNorm':
    case 'groupNorm': {
      // normalize + scale + shift ≈ 4–5 ops/element; we use 4
      const elems = outputShape.reduce((a, b) => a * b, 1);
      return Math.round(elems * 2); // 2 MACs: (x - μ)/σ * γ + β
    }

    // ── Pooling ───────────────────────────────────────────────────────────────
    case 'maxpool2d':
    case 'avgpool2d': {
      const k    = n(p.kernelSize, 2);
      const spat = spatialElements(outputShape);
      const C    = outputShape[0] ?? 1;
      return C * k * k * spat; // compare k² elements per output
    }
    case 'adaptiveAvgPool2d':
    case 'globalAvgPool2d': {
      const Cin   = inputShape[0] ?? 1;
      const inSpat = spatialElements(inputShape);
      return Cin * inSpat; // sum over spatial, divide (1 add per element)
    }

    // ── Activations (element-wise, ~1 MAC each) ───────────────────────────────
    case 'relu':
    case 'leakyRelu':
    case 'sigmoid':
    case 'tanh':
    case 'softmax': {
      return outputShape.reduce((a, b) => a * b, 1);
    }
    case 'gelu':
    case 'swish':
    case 'silu': {
      // GELU/Swish are heavier (~8 ops each), approximate
      return outputShape.reduce((a, b) => a * b, 1) * 4;
    }

    // ── Embedding lookup (0 MACs — just a gather) ────────────────────────────
    case 'embedding':
    case 'embeddingBag':
    case 'positionalEncoding':
    case 'rope':
      return 0;

    // ── Transformer extras ────────────────────────────────────────────────────
    case 'swiglu': {
      const D = n(p.embedDim ?? p.inFeatures, 4096);
      const I = n(p.intermediateSize ?? p.hiddenFeatures ?? p.ffDim, Math.round(D * 8 / 3));
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      // gate(D→I) + up(D→I) + element-wise SiLU+mul + down(I→D)
      return T > 0 && D > 0 && I > 0 ? T * (2 * D * I + I + I * D) : 0;
    }
    case 'moeLayer': {
      const D = n(p.embedDim, 512);
      const E = n(p.numExperts, 8);
      const K = n(p.topK, 2);
      const I = n(p.expertDim ?? p.ffDim, Math.round(D * 8 / 3));
      const T = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 1;
      // router: T·D·E  +  topK experts × swiglu cost per token
      return T > 0 && D > 0 && I > 0 ? T * D * E + K * T * (2 * D * I + I + I * D) : 0;
    }
    case 'patchEmbed': {
      const P    = n(p.patchSize, 16);
      const D    = n(p.embedDim, 768);
      const inC  = n(p.inChans ?? p.inChannels, 3);
      const H    = inputShape[1] ?? 224;
      const W    = inputShape[2] ?? 224;
      const nPat = Math.floor(H / P) * Math.floor(W / P);
      return inC * D * P * P * nPat; // convolution as patch proj
    }
    case 'seBlock': {
      const C    = n(p.channels, 64);
      const r    = n(p.reductionRatio ?? p.reduction, 16);
      const mid  = Math.max(1, Math.floor(C / r));
      const spat = spatialElements(inputShape);
      // GAP(C·spat) + FC1(C·mid) + FC2(mid·C)
      return C * spat + C * mid + mid * C;
    }
    case 'alibi':
    case 'dropPath':
    case 'layerScale':
      return 0;

    // ── Dropout / Flatten / IO (0 MACs) ──────────────────────────────────────
    case 'dropout':
    case 'flatten':
    case 'input':
    case 'output':
      return 0;

    default:
      return 0;
  }
}

/**
 * Format a MAC/FLOPs count with G / M / K suffixes.
 * Returns "—" for zero.
 */
export function fmtFlops(macs: number): string {
  if (!isFinite(macs) || isNaN(macs) || macs <= 0) return '—';
  if (macs >= 1e12) return `${(macs / 1e12).toFixed(2)} T`;
  if (macs >= 1e9)  return `${(macs / 1e9).toFixed(2)} G`;
  if (macs >= 1e6)  return `${(macs / 1e6).toFixed(1)} M`;
  if (macs >= 1e3)  return `${(macs / 1e3).toFixed(1)} K`;
  return macs.toString();
}

/** Format bytes with MB / KB suffixes */
export function fmtBytes(bytes: number): string {
  if (!isFinite(bytes) || bytes <= 0) return '—';
  if (bytes >= 1_073_741_824) return `${(bytes / 1_073_741_824).toFixed(2)} GB`;
  if (bytes >= 1_048_576)     return `${(bytes / 1_048_576).toFixed(1)} MB`;
  if (bytes >= 1_024)         return `${(bytes / 1_024).toFixed(1)} KB`;
  return `${bytes} B`;
}
