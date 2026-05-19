// Tensor shape — e.g. [batch, channels, height, width]
export type TensorShape = number[];

// ML component types
export type ComponentType = 
  // Basic
  | 'input'
  | 'output'
  | 'linear'
  | 'flatten'
  // CV - Computer Vision
  | 'conv2d'
  | 'conv3d'
  | 'depthwiseConv2d'
  | 'separableConv2d'
  | 'transposeConv2d'
  | 'maxpool2d'
  | 'avgpool2d'
  | 'adaptiveAvgPool2d'
  | 'adaptiveMaxPool2d'
  | 'globalAvgPool2d'
  | 'globalMaxPool2d'
  | 'dilatedConv2d'
  | 'roiAlign'
  | 'maxpool3d'
  | 'avgpool3d'
  | 'upsample'
  | 'pixelShuffle'
  // NLP - Natural Language Processing
  | 'conv1d'
  | 'maxpool1d'
  | 'avgpool1d'
  | 'embedding'
  | 'segmentEmbedding'
  | 'lstm'
  | 'gru'
  | 'rnn'
  | 'bidirectionalLSTM'
  | 'bidirectionalGRU'
  | 'attention'
  | 'selfAttention'
  | 'crossAttention'
  | 'globalAvgPool1d'
  // LLM - Large Language Models
  | 'multiHeadAttention'
  | 'groupedQueryAttention'
  | 'causalAttention'
  | 'transformerBlock'
  | 'positionalEncoding'
  | 'feedForward'
  | 'rope'
  | 'lmHead'
  | 'timeEmbedding'
  | 'mamba'
  | 'relativePositionBias'
  | 'learnedPositionalEmbedding'
  | 'localAttention'
  | 'linearAttention'
  // Audio
  | 'melSpectrogram'
  | 'mfcc'
  | 'stft'
  | 'audioConv'
  | 'conformerBlock'
  | 'depthwiseConv1d'
  // Tabular
  | 'featureInteraction'
  | 'embeddingBag'
  | 'tabnet'
  // Reinforcement Learning
  | 'dqnHead'
  | 'actorHead'
  | 'criticHead'
  | 'policyNetwork'
  | 'valueNetwork'
  // Graph ML
  | 'graphConv'
  | 'graphAttention'
  | 'graphSAGE'
  | 'gcn'
  | 'gat'
  | 'gin'
  | 'edgeConv'
  // Multimodal
  | 'crossModalAttention'
  | 'fusion'
  | 'projection'
  | 'coAttention'
  // Activation
  | 'relu'
  | 'relu6'
  | 'leakyRelu'
  | 'elu'
  | 'prelu'
  | 'gelu'
  | 'swish'
  | 'selu'
  | 'mish'
  | 'hardSwish'
  | 'hardSigmoid'
  | 'logSoftmax'
  | 'glu'
  | 'softplus'
  | 'sigmoid'
  | 'tanh'
  | 'softmax'
  | 'gumbelSoftmax'
  // Normalization
  | 'batchNorm'
  | 'layerNorm'
  | 'instanceNorm'
  | 'groupNorm'
  | 'rmsNorm'
  | 'adaIN'
  | 'spectralNorm'
  | 'pixelNorm'
  | 'weightNorm'
  | 'localResponseNorm'
  // LLM extras
  | 'swiglu'
  | 'moeLayer'
  | 'alibi'
  // CV extras
  | 'seBlock'
  | 'patchEmbed'
  | 'windowAttention'
  | 'fpn'
  | 'invResidualBlock'
  | 'deformableConv2d'
  | 'interpolate'
  | 'channelShuffle'
  | 'gridSample'
  | 'spatialPyramidPool'
  // Utility
  | 'dropout'
  | 'reshape'
  | 'residual'
  | 'skipConnection'
  | 'concatenate'
  | 'add'
  | 'multiply'
  | 'dropPath'
  | 'layerScale'
  | 'split'
  | 'permute'
  | 'customModule'
  | 'stickyNote'
  | 'squeeze'
  | 'unsqueeze'
  | 'pad'
  | 'mean'
  | 'matmul'
  | 'clamp'
  | 'norm'
  | 'vaeBottleneck'
  | 'miniBatchStdDev'
  | 'topK'
  | 'gather'
  | 'scatter'
  | 'stack'
  | 'einsum';

// ML component interface
export interface MLComponent {
  id: string;
  type: ComponentType;
  name: string;
  scope?: string;            // dotted-path module hierarchy, e.g. "encoder.layer.0.attention". Used by ScopeFolderPanel; falls back to parsing `name` when absent.
  position: { x: number; y: number };
  params: Record<string, any>;
  inputShape?: TensorShape;  // computed from upstream connections
  outputShape?: TensorShape; // computed output shape
  inputs: string[];          // connected input component IDs
  outputs: string[];         // connected output component IDs
  tensorValue?: number[];    // current tensor values (1D flat array)
  initializationMethod?: string; // input component only
  notes?: string;            // user annotations for documentation
  augmentations?: string[];  // layer-level overlays: 'freeze' | 'quantize_int8' | 'gradient_checkpoint' | 'amp'
  colorTag?: string;         // user-assigned color label for visual grouping
  locked?: boolean;          // locked nodes cannot be moved or deleted
  lockedParams?: string[];   // param keys frozen from edits
  /**
   * Where this layer came from. Populated by importers (paper / code / HF /
   * agent) and surfaced in the inspector + academic export. Letting layers
   * carry their own citation lets us auto-generate .bib entries and the
   * Methods section without re-asking the user.
   */
  provenance?: LayerProvenance;
}

/**
 * Origin metadata for a single layer. Plain interface so we can serialize/
 * deserialize without losing fields. `kind` is the discriminator the UI uses
 * to format hover cards and bibliography entries.
 */
export interface LayerProvenance {
  /** Where the layer came from: which importer / which manual action. */
  kind: 'paper' | 'code' | 'hf-model' | 'agent' | 'block-library' | 'manual';
  /** One-line human label, e.g. "He et al. 2016 — ResNet" or "imported from resnet.py:42". */
  label?: string;
  /**
   * Canonical source URL: arXiv abstract page, HuggingFace model page,
   * github.com permalink, etc. Used as the bibliography URL field.
   */
  url?: string;
  /** Free-form section/file citation, e.g. "Section 3.2" or "models/resnet.py". */
  section?: string;
  /** Source file line number when known (1-based, matching most editors). */
  line?: number;
  /** When this layer was imported (UTC ms), for "imported 2 days ago" affordances. */
  importedAt?: number;
  /** Bibliography entry id — stable, slug-safe (e.g. "he2016resnet"). */
  bibKey?: string;
  /** Authors / title / venue / year if known — used to render .bib entries. */
  authors?: string;
  title?: string;
  venue?: string;
  year?: number;
}

// Port direction
export type PortSide = 'top' | 'right' | 'bottom' | 'left';

// Component connection
export type EdgeRouting = 'bezier' | 'step' | 'straight';

export interface ComponentConnection {
  id: string;
  from: string;      // source component ID
  to: string;        // target component ID
  fromPort: PortSide;
  toPort: PortSide;
  label?: string;    // optional user-defined edge label
  routing?: EdgeRouting; // per-connection routing override
  bend?: number;     // bezier control-point lateral offset (0 = default tension)
}

// Canvas annotation (sticky note anchored to canvas coordinates)
export interface CanvasAnnotation {
  id: string;
  x: number;
  y: number;
  text: string;
  color?: 'yellow' | 'blue' | 'green' | 'pink';
  width?: number;
  targetComponentId?: string;  // optional arrow callout target component ID
}

// Component group — collection of layers that can be collapsed into a single node
export interface ComponentGroup {
  id: string;
  name: string;
  componentIds: string[];
  collapsed: boolean;
  color?: string;                    // header accent color (defaults to blue)
  position?: { x: number; y: number }; // position used when collapsed (centroid of members)
}

// Model architecture definition
export interface HyperparamDef {
  value: number | string;
  type: 'float' | 'int' | 'str';
  description?: string;
}

export interface ModelArchitecture {
  id: string;
  name: string;
  description?: string;
  components: MLComponent[];
  connections: ComponentConnection[];
  annotations?: CanvasAnnotation[];
  groups?: ComponentGroup[];
  hyperparams?: Record<string, HyperparamDef>;
  /** Paper-import metadata — used by exporters to emit task-aware code. Optional. */
  paperMeta?: {
    taskSpec?: any;
    dataset?: any;
    backbone?: any;
    /** Source paper citation — populated when imported via PaperImporter.
     *  Drives the BibTeX exporter and the citation footer in TikZ figures. */
    citation?: {
      title?: string;
      authors?: string[];
      year?: number;
      arxivId?: string;
      venue?: string;
      doi?: string;
      url?: string;
    };
  };
  /** Promoted AI outputs (pinned agent suggestions, accepted advisor rationales).
   *  Persisted with the model so the design rationale survives across sessions
   *  and gets included in the Model Card export. */
  designNotes?: DesignNote[];
  /** Advisor rule IDs the user has explicitly accepted as known/intentional for
   *  this model. The advisor still computes them but the UI suppresses them. */
  acceptedAdvisorRules?: string[];
}

/** A single promoted AI output. Sources: 'agent' (pinned chat reply),
 *  'advisor' (accepted lint rationale), 'manual' (user-typed note). */
export interface DesignNote {
  id: string;
  source: 'agent' | 'advisor' | 'manual';
  /** Short title shown in compact lists (model card section header, etc.) */
  title: string;
  /** Full markdown body — agent reply text, advisor message + suggestion, or user note. */
  body: string;
  /** ISO timestamp when this was pinned. */
  createdAt: string;
  /** Component IDs this note refers to, if any. Used to pin notes to layers. */
  affectedIds?: string[];
  /** When source === 'advisor', the rule that fired. Used to dedupe / link back. */
  ruleId?: string;
}

// Tensor visualization config
export interface TensorVisualizationConfig {
  shape: TensorShape;
  color?: string;
  showValues?: boolean;
  scale?: number;
}

