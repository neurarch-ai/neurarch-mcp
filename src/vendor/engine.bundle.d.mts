/**
 * Types for the vendored engine bundle (engine.bundle.mjs), hand-written
 * because the bundle ships as compiled JavaScript with no declarations.
 *
 * Only what this server actually calls is declared. Adding a declaration for
 * something the bundle does not export would compile fine and fail at runtime,
 * so engine.contract.test.ts asserts the real module against this list.
 *
 * ModelArchitecture is deliberately this package's own type rather than the
 * app's: the two are the same shape on the wire (that is the whole point of
 * .neurarch.json), and importing the app's would drag its type graph in here.
 */
import type { ModelArchitecture } from '../lib/types.js';

export type EngineSeverity = 'block' | 'warn' | 'info';

export interface EngineFinding {
  /** Stable rule id, e.g. 'high-dropout' or 'head-dim-divisibility'. */
  rule: string;
  severity: EngineSeverity;
  message: string;
  /** The offending layer, when the rule pins one. Absent for whole-model rules. */
  componentName?: string;
  componentType?: string;
}

/** Options for graphFromPyTorchSource. */
export interface GraphFromSourceOptions {
  /** Parse this class instead of the auto-selected root class (the class no
   *  other class in the file instantiates, with the most layers). */
  className?: string;
}

/**
 * Reconstruct a model graph from PyTorch source. Returns null when the file has
 * no recognizable model class, or parses to nothing but an input/output stub.
 * `name` is usually the file path; its stem is a tie-break between root classes.
 */
export function graphFromPyTorchSource(
  code: string,
  name?: string,
  opts?: GraphFromSourceOptions,
): ModelArchitecture | null;

/** Run the structural rule set over a graph. */
export function lintModelGraph(model: ModelArchitecture): EngineFinding[];

/** Rule ids in the advisor set, for coverage assertions. */
export const ADVISOR_RULE_IDS: readonly string[];

/** Rule ids in the shape set. */
export const SHAPE_RULE_IDS: readonly string[];

/**
 * Render the graph as a runnable PyTorch module. Same generator as the app's
 * Export panel. Layer types the generator has no template for are reported on
 * stderr and left as a comment in the output rather than silently dropped.
 */
export function generatePyTorchCode(model: ModelArchitecture): string;

/** A Hugging Face config.json, plus the loader's own annotations. */
export interface HFModelConfig {
  model_type?: string;
  architectures?: string[];
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  intermediate_size?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  /** Where the dimensions came from. 'fallback' means a generic family template, not the real model. */
  _configSource?: 'config' | 'diffusers' | 'fallback';
  /** Authoritative parameter count from safetensors metadata, when HF publishes one. */
  _realParamCount?: number;
  [key: string]: unknown;
}

/**
 * Fetch and normalise a model's config.json from the Hub. The only function in
 * this bundle that opens a socket; the server exposes it behind --hf. Reads
 * HF_TOKEN from the environment for gated repos. Null when nothing usable
 * could be fetched or inferred.
 */
export function fetchHFModelConfig(modelId: string): Promise<HFModelConfig | null>;

/** Build the graph for a config. Pure: no network. */
export function convertHFConfigToModel(
  modelId: string,
  config: HFModelConfig,
): { components: ModelArchitecture['components']; connections: ModelArchitecture['connections'] };
