/**
 * Does the data a job will feed the model fit what the model accepts? A
 * sequence longer than a learned position table, a tokenizer with more ids
 * than the embedding has rows, a label id past the classification head, an
 * image size or channel count the first layer was not built for: each raises
 * on the first batch, after the GPU is reserved, and each is readable from the
 * config before anything is.
 *
 * No arithmetic here. The checks are the vendored plan bundle's
 * `inputContractOf` / `readDataSpec` / `checkDataFit` (the app's
 * src/utils/configDataFit.ts, neurarch-ai/neurarch #314), the same functions
 * behind `npx neurarch <id> --seq-len ... --tokenizer-size ...` and the
 * `data` field of POST /api/v1/check. This file adds only where the Hugging
 * Face config comes from, mirroring the CLI's `dataFitHFModel`: the graph
 * built from a config carries no classification head, so the label count is
 * read from the config itself.
 */
import type { ModelArchitecture } from './types.js';
import { inputContractOf, readDataSpec, checkDataFit, type DataFitReport, type DataSpec } from '../vendor/plan.bundle.mjs';
import { loadHFModel } from '../sources.js';

export type { DataFitReport, DataSpec };

/**
 * JSON schema for the `data` argument. Deliberately terse: the tool listing
 * has a token budget (toolDocs.test.ts), and the fields are validated by the
 * bundle's readDataSpec, which refuses a present field that is not a positive
 * integer. The tool descriptions spell out what each field means.
 */
export const DATA_SCHEMA = {
  type: 'object',
  description: 'seq_len, tokenizer_size, num_classes, image_size, channels',
} as const;

/** The `data` argument as a DataSpec; a present field that is not a positive integer is refused, not ignored. */
export function parseDataArg(tool: string, raw: unknown): DataSpec {
  const spec = readDataSpec(raw);
  if ('error' in spec) throw new Error(`${tool}: ${spec.error}`);
  return spec;
}

/**
 * The config.json behind a graph built from hf:<org/name>, or null. Served
 * from the cache loadHFModel wrote when it built the graph, so it opens no
 * socket of its own; a graph from anywhere else has no config.
 */
export async function hfConfigOf(model: ModelArchitecture): Promise<Record<string, unknown> | null> {
  const m = /^hf-([\w.-]+\/[\w.-]+)$/.exec(model.id ?? '');
  if (!m) return null;
  try {
    return (await loadHFModel(m[1])).config ?? null;
  } catch {
    return null;
  }
}

/** The bundle's report for this graph, this config (when there is one) and this data. */
export function dataFitFor(model: ModelArchitecture, data: DataSpec, config?: Record<string, unknown> | null): DataFitReport {
  return checkDataFit(inputContractOf(model, config ?? null), data);
}

/** One line for the verdict when the data cannot load, else null. */
export function dataBlockSentence(report: DataFitReport): string | null {
  const blocks = report.findings.filter(f => f.severity === 'block');
  if (!blocks.length) return null;
  return `the data will not load: ${blocks[0].title}${blocks.length > 1 ? ` (and ${blocks.length - 1} more blocking data-fit finding${blocks.length > 2 ? 's' : ''})` : ''}`;
}
