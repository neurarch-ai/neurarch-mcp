import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import type { ModelArchitecture } from './lib/types.js';

export async function loadModelFile(path: string): Promise<ModelArchitecture> {
  const abs = resolve(path);
  const info = await stat(abs).catch((e) => {
    throw new Error(`Cannot read model file: ${abs} (${e.code ?? e.message})`);
  });
  if (!info.isFile()) {
    throw new Error(`Not a file: ${abs}`);
  }
  const raw = await readFile(abs, 'utf-8');
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    throw new Error(`File is not valid JSON: ${abs} (${(e as Error).message})`);
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
  return parsed as ModelArchitecture;
}
