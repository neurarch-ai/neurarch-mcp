/**
 * Reading a model off disk, in either of the two forms it comes in.
 *
 * For most of this server's life there was one form: a `.neurarch.json` the app
 * had written. That is a fine format and a bad precondition. An agent is
 * standing in a repository that contains `model.py`, and telling it "first open
 * the app, draw the graph, File → Save, then come back" is a wall between the
 * install and the first useful answer. So a `.py` file is loaded by parsing it
 * into the same graph, with the same parser the app and CI use.
 *
 * What you get from Python source is real but not complete: layers, types,
 * hyperparameters and wiring, all of it; tensor shapes, none of it, because the
 * source does not say what goes in. Parameter counts still come out (they are
 * derived from layer params) and FLOPs largely do not. That is stated by
 * `sourceKind` rather than papered over, so a tool can report "unknown" instead
 * of a confident zero.
 */
import { readFile, stat } from 'node:fs/promises';
import { basename, resolve } from 'node:path';
import type { ModelArchitecture } from './lib/types.js';
import { graphFromPyTorchSource } from './vendor/engine.bundle.mjs';

/** How a loaded graph was obtained. Governs what can be trusted and written. */
export type ModelSourceKind = 'neurarch-json' | 'pytorch-source';

/**
 * Dispatch is on the extension, not on content sniffing.
 *
 * A JSON file that fails to parse should say "this JSON is malformed", not
 * silently fall through to the Python parser and then report "no model class
 * found", which would send the user hunting for the wrong bug.
 */
export function sourceKindFor(path: string): ModelSourceKind {
  return path.toLowerCase().endsWith('.py') ? 'pytorch-source' : 'neurarch-json';
}

export async function loadModelFile(path: string): Promise<ModelArchitecture> {
  const abs = resolve(path);
  const info = await stat(abs).catch((e) => {
    throw new Error(`Cannot read model file: ${abs} (${e.code ?? e.message})`);
  });
  if (!info.isFile()) {
    throw new Error(`Not a file: ${abs}`);
  }
  const raw = await readFile(abs, 'utf-8');

  if (sourceKindFor(abs) === 'pytorch-source') return parsePythonModel(abs, raw);

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    throw new Error(
      `File is not valid JSON: ${abs} (${(e as Error).message}). `
      + 'If this is PyTorch source, give it a .py extension and it will be parsed as one.',
    );
  }
  if (!parsed || typeof parsed !== 'object') {
    throw new Error(`Model file is not a JSON object: ${abs}`);
  }
  const candidate = parsed as Partial<ModelArchitecture>;
  if (!Array.isArray(candidate.components) || !Array.isArray(candidate.connections)) {
    throw new Error(
      `Model file is missing required fields. Expected JSON with "components" and "connections" arrays. ` +
      `Use File → Save (.json) in the Neurarch app to produce a valid file.`,
    );
  }
  // Every tool indexes components by id; a missing/non-string id would surface
  // later as confusing "layer not found" errors, so reject up front.
  const badIdx = candidate.components.findIndex(
    (c) => !c || typeof c !== 'object' || typeof (c as { id?: unknown }).id !== 'string',
  );
  if (badIdx !== -1) {
    throw new Error(
      `Model file is corrupt: components[${badIdx}] is missing a string "id". ` +
      `Re-export from the Neurarch app with File → Save (.json).`,
    );
  }
  return parsed as ModelArchitecture;
}

/**
 * Parse PyTorch source into a graph.
 *
 * A failure here is the common case for a file that simply is not a model
 * (a training loop, a dataset module), so the message says what was looked for
 * rather than just "parse failed".
 */
function parsePythonModel(abs: string, code: string): ModelArchitecture {
  const name = basename(abs).replace(/\.py$/i, '');
  const model = graphFromPyTorchSource(code, name);
  if (!model) {
    throw new Error(
      `No PyTorch model found in ${abs}. The parser looks for an nn.Module subclass whose layers `
      + 'are assigned in __init__ (self.fc = nn.Linear(...), nn.Sequential(...), and so on). '
      + 'A file that only defines a training loop, a dataset, or a bare forward() has no graph to read. '
      + 'If the architecture is built dynamically, export it from the Neurarch app as .neurarch.json instead.',
    );
  }
  return model;
}
