import { ComponentType } from './types.js';

/** Safe number coercion — returns fallback for NaN / Infinity / negative */
function num(v: any, fallback = 0): number {
  const n = typeof v === 'number' ? v : Number(v);
  return isFinite(n) && n >= 0 ? n : fallback;
}

/**
 * Estimate learnable parameter count for a single layer.
 * Returns 0 for parameter-free ops (activations, pooling, dropout, …).
 */
export function estimateLayerParams(
  type: ComponentType,
  params: Record<string, any>,
  inputShape?: number[],
): number {
  const p = params;
  const inp = inputShape ?? [];
  const lastDim = inp.length > 0 ? inp[inp.length - 1] : 0;
  const ch = inp.length >= 2 ? inp[1] : 1; // channel dim (NCHW layout)

  switch (type) {
    // ── Basic ────────────────────────────────────────────────────────────────
    case 'linear': {
      const inF = num(p.inFeatures ?? lastDim);
      const outF = num(p.outFeatures);
      return inF > 0 && outF > 0 ? inF * outF + outF : 0;
    }
    case 'flatten':
      return 0;

    // ── Convolution ──────────────────────────────────────────────────────────
    case 'conv2d': {
      const inC = num(p.inChannels ?? ch, 1);
      const outC = num(p.outChannels);
      const k = num(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k + outC : 0;
    }
    case 'conv1d': {
      const inC = num(p.inChannels ?? ch, 1);
      const outC = num(p.outChannels);
      const k = num(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k + outC : 0;
    }
    case 'conv3d': {
      const inC = num(p.inChannels ?? ch, 1);
      const outC = num(p.outChannels);
      const k = num(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k * k + outC : 0;
    }
    case 'depthwiseConv2d': {
      const inC = num(p.inChannels ?? ch, 1);
      const k = num(p.kernelSize, 3);
      const dm = num(p.depthMultiplier, 1);
      return inC * dm * k * k + inC * dm;
    }
    case 'separableConv2d': {
      const inC = num(p.inChannels ?? ch, 1);
      const outC = num(p.outChannels, inC);
      const k = num(p.kernelSize, 3);
      return inC * k * k + inC * outC + outC; // depthwise + pointwise
    }
    case 'transposeConv2d': {
      const inC = num(p.inChannels ?? ch, 1);
      const outC = num(p.outChannels);
      const k = num(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k * k + outC : 0;
    }

    // ── Pooling (no learnable params) ────────────────────────────────────────
    case 'maxpool2d':
    case 'avgpool2d':
    case 'adaptiveAvgPool2d':
    case 'globalAvgPool2d':
    case 'upsample':
      return 0;

    // ── NLP ─────────────────────────────────────────────────────────────────
    case 'embedding':
    case 'embeddingBag': {
      const V = num(p.vocabSize ?? p.numEmbeddings);
      const D = num(p.embeddingDim ?? p.embedDim);
      return V > 0 && D > 0 ? V * D : 0;
    }
    case 'lstm': {
      const H = num(p.hiddenSize, 128);
      const I = num(p.inputSize ?? lastDim, H);
      const L = num(p.numLayers, 1);
      const layer0 = 4 * (I * H + H * H + 2 * H);
      const layerRest = L > 1 ? (L - 1) * 4 * (2 * H * H + 2 * H) : 0;
      return layer0 + layerRest;
    }
    case 'gru': {
      const H = num(p.hiddenSize, 128);
      const I = num(p.inputSize ?? lastDim, H);
      const L = num(p.numLayers, 1);
      const layer0 = 3 * (I * H + H * H + 2 * H);
      const layerRest = L > 1 ? (L - 1) * 3 * (2 * H * H + 2 * H) : 0;
      return layer0 + layerRest;
    }
    case 'rnn': {
      const H = num(p.hiddenSize, 128);
      const I = num(p.inputSize ?? lastDim, H);
      return I * H + H * H + 2 * H;
    }
    case 'bidirectionalLSTM': {
      const H = num(p.hiddenSize, 128);
      const I = num(p.inputSize ?? lastDim, H);
      return 2 * (4 * (I * H + H * H + 2 * H)); // 2× single LSTM
    }
    case 'attention':
    case 'selfAttention':
    case 'multiHeadAttention': {
      const d = num(p.hiddenDim ?? p.embedDim);
      return d > 0 ? 4 * d * d + 4 * d : 0; // Q/K/V/O projections + biases
    }
    case 'crossModalAttention': {
      const d = num(p.embedDim);
      return d > 0 ? 4 * d * d + 4 * d : 0;
    }

    // ── LLM ─────────────────────────────────────────────────────────────────
    case 'feedForward': {
      const d = num(p.hiddenDim ?? p.embedDim);
      const ff = num(p.ffDim, d > 0 ? d * 4 : 0);
      return d > 0 && ff > 0 ? d * ff + ff + ff * d + d : 0;
    }
    case 'transformerBlock': {
      const d = num(p.embedDim ?? p.hiddenDim);
      const ff = num(p.ffDim, d > 0 ? d * 4 : 0);
      if (d <= 0) return 0;
      // MHA (4d²+4d) + FFN (d·ff+ff + ff·d+d) + 2×LayerNorm (4d)
      return 4 * d * d + 4 * d + d * ff + ff + ff * d + d + 4 * d;
    }
    case 'lmHead': {
      // Output projection hidden -> vocab. Default bias=false (and often weight-
      // tied with the embedding, but we count it untied unless told otherwise).
      const d = num(p.embedDim ?? p.hiddenDim ?? lastDim);
      const V = num(p.vocabSize ?? p.numEmbeddings);
      const bias = p.bias === true ? V : 0;
      return d > 0 && V > 0 ? d * V + bias : 0;
    }
    case 'positionalEncoding':
    case 'rope':
      return 0; // learned or fixed, no gradient params

    // ── Normalization ────────────────────────────────────────────────────────
    case 'layerNorm':
    case 'batchNorm':
    case 'instanceNorm': {
      const feat = p.normalizedShape ?? p.numFeatures ?? lastDim;
      const f = num(Array.isArray(feat) ? feat[0] : feat);
      return f > 0 ? 2 * f : 0; // scale + bias
    }
    case 'groupNorm': {
      const c = num(p.numChannels ?? ch);
      return c > 0 ? 2 * c : 0;
    }

    // ── RL ───────────────────────────────────────────────────────────────────
    case 'policyNetwork':
    case 'valueNetwork': {
      const H = num(p.hiddenSize, 256);
      const inF = num(lastDim, H);
      // Approx 2-layer MLP
      return inF * H + H + H * H + H;
    }
    case 'dqnHead': {
      const H = num(p.hiddenSize ?? lastDim, 512);
      const A = num(p.numActions, 18);
      return H * A + A;
    }
    case 'actorHead': {
      const H = num(lastDim, 256);
      const A = num(p.numActions, 6);
      return H * A + A;
    }
    case 'criticHead': {
      const H = num(lastDim, 256);
      const O = num(p.outputDim, 1);
      return H * O + O;
    }

    // ── Graph ────────────────────────────────────────────────────────────────
    case 'graphConv':
    case 'gcn': {
      const inC = num(p.inChannels ?? p.inFeatures);
      const outC = num(p.outChannels ?? p.outFeatures);
      return inC > 0 && outC > 0 ? inC * outC + outC : 0;
    }
    case 'graphAttention':
    case 'gat': {
      const inC = num(p.inChannels ?? p.inFeatures);
      const outC = num(p.outChannels ?? p.outFeatures);
      const heads = num(p.heads, 1);
      return inC > 0 && outC > 0 ? heads * (inC * outC + 2 * outC) : 0;
    }
    case 'graphSAGE': {
      const inC = num(p.inChannels ?? p.inFeatures);
      const outC = num(p.outChannels ?? p.outFeatures);
      return inC > 0 && outC > 0 ? inC * 2 * outC + outC : 0;
    }

    // ── Multimodal ───────────────────────────────────────────────────────────
    case 'fusion': {
      const inD = num(lastDim);
      const d = num(p.fusionDim, 256);
      return d > 0 && inD > 0 ? inD * d + d : 0;
    }
    case 'projection': {
      const inD = num(p.inDim ?? lastDim);
      const outD = num(p.outDim);
      return inD > 0 && outD > 0 ? inD * outD + outD : 0;
    }

    // ── Tabular ──────────────────────────────────────────────────────────────
    case 'tabnet': {
      const inF = num(p.inputDim ?? lastDim);
      const N = num(p.numAttentionEmbeddings, 8);
      return inF > 0 ? inF * N + N : 0;
    }
    case 'featureInteraction':
      return 0;

    // ── Audio ────────────────────────────────────────────────────────────────
    case 'audioConv': {
      const inC = num(p.inChannels ?? 1, 1);
      const outC = num(p.outChannels);
      const k = num(p.kernelSize, 3);
      return outC > 0 ? inC * outC * k + outC : 0;
    }
    case 'melSpectrogram':
    case 'mfcc':
    case 'stft':
      return 0;

    // ── New attention types ──────────────────────────────────────────────────
    case 'windowAttention': {
      const D = num(p.embedDim, 96);
      return D > 0 ? 4 * D * D : 0; // QKV + output projection (no KV reduction)
    }
    case 'groupedQueryAttention': {
      const D = num(p.embedDim, 4096);
      const H = num(p.numHeads, 32);
      const Hkv = num(p.numKVHeads, 8);
      // Q: D×D; K,V: D × (D/H * Hkv) each; O: D×D
      const headDim = H > 0 ? Math.floor(D / H) : 128;
      return D > 0 ? D * D + 2 * (D * headDim * Hkv) + D * D : 0;
    }
    case 'causalAttention': {
      const D = num(p.embedDim, 512);
      return D > 0 ? 4 * D * D : 0; // same as MHA params
    }
    case 'adaptiveMaxPool2d':
      return 0; // no learnable params

    // ── New activations ──────────────────────────────────────────────────────
    case 'prelu': {
      // PReLU has one learnable parameter per channel (or one global)
      return num(p.numParameters, 1);
    }
    case 'rmsNorm': {
      const ns = num(p.normalizedShape ?? lastDim, 512);
      return ns; // weight only (no bias)
    }

    // ── Transformer extras ───────────────────────────────────────────────────
    case 'swiglu': {
      const D = num(p.embedDim ?? p.inFeatures, 4096);
      const I = num(p.intermediateSize ?? p.hiddenFeatures ?? p.ffDim, Math.round(D * 8 / 3));
      // gate_proj(D→I) + up_proj(D→I) + down_proj(I→D), all bias=False
      return D > 0 && I > 0 ? 3 * D * I : 0;
    }
    case 'moeLayer': {
      const D = num(p.embedDim, 512);
      const E = num(p.numExperts, 8);
      const I = num(p.expertDim ?? p.ffDim, Math.round(D * 8 / 3));
      // router: D×E + E experts each shaped like a SwiGLU FFN
      return D > 0 && I > 0 ? D * E + E * 3 * D * I : 0;
    }
    case 'patchEmbed': {
      // Conv2d(inChans, embedDim, kernel=patchSize, stride=patchSize) + embedDim bias
      const inC = num(p.inChans ?? p.inChannels, 3);
      const D   = num(p.embedDim, 768);
      const P   = num(p.patchSize, 16);
      return inC * D * P * P + D;
    }
    case 'seBlock': {
      // FC1: C→C/r, FC2: C/r→C (both with bias)
      const C = num(p.channels, 64);
      const r = num(p.reductionRatio ?? p.reduction, 16);
      const mid = Math.max(1, Math.floor(C / r));
      return C * mid + mid + mid * C + C;
    }
    case 'layerScale':
      // Learnable per-dim scale vector (γ), no bias
      return num(p.dim, 512);
    case 'alibi':
    case 'dropPath':
      return 0;

    // ── No learnable params ──────────────────────────────────────────────────
    default:
      return 0;
  }
}

/**
 * Format a parameter count with K / M / B suffixes.
 * Returns "—" for zero, "?" for invalid.
 */
export function fmtParams(n: number): string {
  if (!isFinite(n) || isNaN(n)) return '?';
  if (n === 0) return '—';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toString();
}
