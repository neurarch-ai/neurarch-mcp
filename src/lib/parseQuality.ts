/**
 * How much of a graph that came out of Python source is actually numbers.
 *
 * docs/REAL_REPOS_STUDY.md measured the static parser against 116 files from
 * 59 popular repositories: it returned a graph for 63% of them, a graph a
 * person would recognise as the model for 8%, and every finding it raised on
 * the result was its own artefact or a design choice. The biggest single
 * cause was a dimension the parser could not evaluate (`config.hidden_size`,
 * `dims[0]`) stored as text where a number belongs, then propagated as if it
 * were one: 54 of 76 warnings, all `invalid-output-shape`, all false.
 *
 * So a graph from source carries a quality note, and the rules that depend
 * on a resolved dimension are held back on layers that do not have one. The
 * note names the fix: neurarch-trace, which reads the numbers at runtime.
 */
import type { ModelArchitecture, MLComponent } from './types.js';

/** A param value that is source text rather than a number the parser evaluated. */
export function isUnresolved(value: unknown): boolean {
  if (typeof value !== 'string') return false;
  const v = value.trim();
  if (v === '' || /^-?\d+(\.\d+)?$/.test(v)) return false;
  if (/^(true|false|none|relu|gelu|silu|tanh|sigmoid|same|valid|zeros|reflect|replicate|circular|mean|sum)$/i.test(v)) return false;
  // Attribute access, indexing, calls, arithmetic, or a bare identifier where
  // a number belongs: all of them are code the parser did not run.
  return /[.\[\](),+\-*/ ]/.test(v) || /^[A-Za-z_]\w*$/.test(v);
}

export function unresolvedParamsOf(c: MLComponent): string[] {
  return Object.entries(c.params ?? {}).filter(([, v]) => isUnresolved(v)).map(([k, v]) => `${k}=${String(v)}`);
}

export interface ParseQuality {
  layers: number;
  /** Layers with at least one param left as source text. */
  layersWithUnresolvedParams: number;
  unresolvedParams: number;
  /** 'full' is the shape of a graph any tool can trust; 'partial' means numbers are missing; 'thin' means the parser saw almost nothing. */
  grade: 'full' | 'partial' | 'thin';
  note?: string;
}

export function parseQuality(model: ModelArchitecture): ParseQuality {
  const real = model.components.filter(c => c.type !== 'input' && c.type !== 'output');
  let layersWithUnresolvedParams = 0;
  let unresolvedParams = 0;
  for (const c of real) {
    const u = unresolvedParamsOf(c);
    if (u.length) { layersWithUnresolvedParams++; unresolvedParams += u.length; }
  }
  const thin = real.length <= 3;
  const grade: ParseQuality['grade'] = thin ? 'thin' : unresolvedParams > 0 ? 'partial' : 'full';
  const note = grade === 'full' ? undefined
    : grade === 'thin'
      ? `The parser found only ${real.length} layer(s). Real model files usually build their layers through other classes, factories or loops the static parser cannot follow; trace the model at runtime instead (pip install neurarch-trace; neurarch-trace <module>:<class> --input ...) and point this server at the .neurarch.json it writes.`
      : `${layersWithUnresolvedParams} of ${real.length} layers carry ${unresolvedParams} parameter(s) as source text the parser could not evaluate (e.g. config.hidden_size). Parameter counts, shapes and shape rules on those layers are not trustworthy. Trace the model at runtime for real numbers (pip install neurarch-trace).`;
  return { layers: real.length, layersWithUnresolvedParams, unresolvedParams, grade, note };
}

/** Rules whose verdict depends on a dimension being a number. Held back where it is not. */
export const DIMENSION_RULES = new Set(['invalid-output-shape', 'head-dim-divisibility', 'gqa-divisibility', 'linear-in-features-mismatch']);
