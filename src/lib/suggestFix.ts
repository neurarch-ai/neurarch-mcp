/**
 * suggest_fix: turn a lint finding into an edit to the source it came from.
 *
 * lint_model says "embed dim 258 is not divisible by 8 heads" and stops. The
 * agent that asked is sitting in an editor with the .py open, and the thing it
 * does next is change a line. So this hands it the line, as a unified diff it
 * can apply, and is honest about which fixes are mechanical and which are
 * proposals:
 *
 *   exact     the rule pins a number or an order, and the source carries that
 *             number as a literal. The diff makes the model pass the rule and
 *             touches every literal that has to move with it (a width change
 *             is not one line).
 *   proposal  the rule says something is missing (a positional encoding, an
 *             activation between two linears). The diff inserts a plausible
 *             layer and says what forward() needs; it is a starting point, and
 *             it is labelled as one.
 *
 * Findings this cannot help with (the parser's own placeholder, a rule about
 * training recipe) come back under `notFixable` with the reason, never
 * silently dropped.
 */
import type { ModelArchitecture, MLComponent } from './types.js';
import type { EngineFinding } from '../vendor/engine.bundle.mjs';
import { unifiedDiff } from './unifiedDiff.js';

export interface Fix {
  rule: string;
  layer?: string;
  confidence: 'exact' | 'proposal';
  summary: string;
  /** Unified diff against the source file. Empty when the fix is words only. */
  diff: string;
  /** 1-based source lines the diff touches. */
  lines: number[];
  /** What the agent still has to do by hand, when the diff is not the whole fix. */
  followUp?: string;
}

export interface SuggestFixResult {
  path: string;
  fixes: Fix[];
  notFixable: Array<{ rule: string; layer?: string; reason: string }>;
  /** The whole file after every exact fix, so an agent can write it in one go. */
  patchedSource?: string;
}

const NUMERIC = /^-?\d+(\.\d+)?$/;

/** 1-based line of `self.<name> = ...`, from the parser when it says, else by name. */
function locate(c: MLComponent | undefined, lines: string[]): number | null {
  if (!c) return null;
  const fromParser = (c as MLComponent & { sourceLine?: number }).sourceLine;
  if (typeof fromParser === 'number' && fromParser > 0 && fromParser <= lines.length) return fromParser;
  const re = new RegExp(`^\\s*self\\.${escape(c.name)}\\s*=`);
  const idx = lines.findIndex(l => re.test(l));
  return idx === -1 ? null : idx + 1;
}

/** The statement `x = self.<name>(` inside forward(), 1-based. */
function locateCall(name: string, lines: string[]): number | null {
  const re = new RegExp(`^\\s*\\w+\\s*=\\s*self\\.${escape(name)}\\(`);
  const idx = lines.findIndex(l => re.test(l));
  return idx === -1 ? null : idx + 1;
}

function escape(s: string): string { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

/** Replace one literal number on one line, only where it appears as a whole token. */
function replaceLiteral(line: string, from: number, to: number): string | null {
  const re = new RegExp(`(?<![\\w.])${from}(?![\\w.])`);
  if (!re.test(line)) return null;
  return line.replace(re, String(to));
}

function nearestMultiple(value: number, of: number): number {
  const down = Math.floor(value / of) * of;
  const up = down + of;
  return value - down <= up - value && down > 0 ? down : up;
}

function largestDivisorAtMost(n: number, cap: number): number {
  for (let d = Math.min(cap, n); d >= 1; d--) if (n % d === 0) return d;
  return 1;
}

export function suggestFixes(
  model: ModelArchitecture,
  findings: EngineFinding[],
  source: string,
  path: string,
): SuggestFixResult {
  const original = source.split('\n');
  // Every diff is against the original file, so each applies on its own.
  // `lines` accumulates the exact fixes only (they never change the line
  // count, except a swap, which keeps it), for `patchedSource`.
  let lines = [...original];
  const base = () => [...original];
  const byName = new Map(model.components.map(c => [c.name, c]));
  const fixes: Fix[] = [];
  const notFixable: SuggestFixResult['notFixable'] = [];

  const num = (v: unknown): number | null => (typeof v === 'number' ? v : typeof v === 'string' && NUMERIC.test(v) ? Number(v) : null);

  for (const f of findings) {
    const c = f.componentName ? byName.get(f.componentName) : undefined;
    const at = locate(c, original);
    const fail = (reason: string) => notFixable.push({ rule: f.rule, layer: f.componentName, reason });

    switch (f.rule) {
      case 'head-dim-divisibility': {
        const dim = num(c?.params.hiddenDim ?? c?.params.embedDim);
        const heads = num(c?.params.numHeads);
        if (!c || at === null || dim === null || heads === null) { fail('the embed dim or head count is not a literal in the source'); break; }
        const target = nearestMultiple(dim, heads);
        // The width is shared with every layer that carries the same literal.
        const touched: number[] = [];
        const next = base();
        for (const other of model.components) {
          if (!Object.values(other.params ?? {}).some(v => num(v) === dim)) continue;
          const ln = locate(other, original);
          if (ln === null) continue;
          const replaced = replaceLiteral(next[ln - 1], dim, target);
          if (replaced) { next[ln - 1] = replaced; touched.push(ln); }
        }
        if (!touched.includes(at)) { fail(`could not find the literal ${dim} on line ${at}`); break; }
        for (const ln of touched) lines[ln - 1] = replaceLiteral(lines[ln - 1], dim, target) ?? lines[ln - 1];
        fixes.push({
          rule: f.rule, layer: c.name, confidence: 'exact',
          summary: `Change the width ${dim} to ${target} (divisible by ${heads} heads) on ${touched.length} line(s) that share it.`,
          diff: unifiedDiff(path, source, next.join('\n')), lines: touched.sort((a, b) => a - b),
          followUp: touched.length === 1 ? undefined : 'Every layer that carried the old width was changed together; check any width passed through a variable rather than a literal.',
        });
        break;
      }
      case 'gqa-head-divisibility':
      case 'gqa-head-mismatch': {
        const heads = num(c?.params.numHeads);
        const kv = num(c?.params.numKVHeads ?? c?.params.numKvHeads);
        if (!c || at === null || heads === null || kv === null) { fail('head counts are not literals in the source'); break; }
        const target = largestDivisorAtMost(heads, kv);
        const replaced = replaceLiteral(original[at - 1], kv, target);
        if (!replaced) { fail(`could not find the literal ${kv} on line ${at}`); break; }
        const next = base(); next[at - 1] = replaced;
        lines[at - 1] = replaceLiteral(lines[at - 1], kv, target) ?? lines[at - 1];
        fixes.push({ rule: f.rule, layer: c.name, confidence: 'exact', summary: `Set KV heads to ${target}, the largest divisor of ${heads} at or below ${kv}.`, diff: unifiedDiff(path, source, next.join('\n')), lines: [at] });
        break;
      }
      case 'high-dropout': {
        const p = num(c?.params.p);
        if (!c || at === null || p === null) { fail('dropout rate is not a literal'); break; }
        const replaced = replaceLiteral(original[at - 1], p, 0.5);
        if (!replaced) { fail(`could not find the literal ${p} on line ${at}`); break; }
        const next = base(); next[at - 1] = replaced;
        lines[at - 1] = replaceLiteral(lines[at - 1], p, 0.5) ?? lines[at - 1];
        fixes.push({ rule: f.rule, layer: c.name, confidence: 'exact', summary: `Clamp dropout from ${p} to 0.5.`, diff: unifiedDiff(path, source, next.join('\n')), lines: [at] });
        break;
      }
      case 'conv-stride-gt-kernel': {
        const k = num(Array.isArray(c?.params.kernelSize) ? c?.params.kernelSize[0] : c?.params.kernelSize);
        const s = num(Array.isArray(c?.params.stride) ? c?.params.stride[0] : c?.params.stride);
        if (!c || at === null || k === null || s === null) { fail('kernel or stride is not a literal'); break; }
        const replaced = original[at - 1].replace(new RegExp(`stride\\s*=\\s*${s}(?![\\d.])`), `stride=${k}`);
        if (replaced === original[at - 1]) { fail(`stride=${s} is not written as a keyword on line ${at}`); break; }
        const next = base(); next[at - 1] = replaced;
        lines[at - 1] = lines[at - 1].replace(new RegExp(`stride\\s*=\\s*${s}(?![\\d.])`), `stride=${k}`);
        fixes.push({ rule: f.rule, layer: c.name, confidence: 'exact', summary: `Reduce stride ${s} to the kernel size ${k} so no input is skipped.`, diff: unifiedDiff(path, source, next.join('\n')), lines: [at] });
        break;
      }
      case 'groupnorm-divisibility': {
        const groups = num(c?.params.numGroups);
        const ch = num(c?.params.numChannels);
        if (!c || at === null || groups === null || ch === null) { fail('groups or channels is not a literal'); break; }
        const target = largestDivisorAtMost(ch, groups);
        const replaced = replaceLiteral(original[at - 1], groups, target);
        if (!replaced) { fail(`could not find the literal ${groups} on line ${at}`); break; }
        const next = base(); next[at - 1] = replaced;
        lines[at - 1] = replaceLiteral(lines[at - 1], groups, target) ?? lines[at - 1];
        fixes.push({ rule: f.rule, layer: c.name, confidence: 'exact', summary: `Use ${target} groups, which divides ${ch} channels.`, diff: unifiedDiff(path, source, next.join('\n')), lines: [at] });
        break;
      }
      case 'dropout-before-bn':
      case 'bn-after-activation': {
        // The rule names the first of two layers in the wrong order; the
        // second is its single downstream. Swapping their two consecutive
        // statements in forward() is exact when they are consecutive.
        if (!c) { fail('no layer named'); break; }
        const nextId = c.outputs?.[0];
        const nextC = model.components.find(x => x.id === nextId);
        const l1 = locateCall(c.name, original);
        const l2 = nextC ? locateCall(nextC.name, original) : null;
        if (l1 === null || l2 === null || Math.abs(l1 - l2) !== 1) { fail('the two calls are not consecutive statements in forward(), reorder them by hand'); break; }
        const next = base(); [next[l1 - 1], next[l2 - 1]] = [next[l2 - 1], next[l1 - 1]];
        [lines[l1 - 1], lines[l2 - 1]] = [lines[l2 - 1], lines[l1 - 1]];
        fixes.push({ rule: f.rule, layer: c.name, confidence: 'exact', summary: `Swap ${c.name} and ${nextC!.name} in forward().`, diff: unifiedDiff(path, source, next.join('\n')), lines: [Math.min(l1, l2), Math.max(l1, l2)] });
        break;
      }
      case 'consecutive-linear-no-activation': {
        if (!c || at === null) { fail('could not locate the layer'); break; }
        const indent = original[at - 1].match(/^\s*/)?.[0] ?? '        ';
        const actName = `${c.name}_act`;
        const next = base();
        next.splice(at, 0, `${indent}self.${actName} = nn.GELU()`);
        const call = locateCall(c.name, next);
        if (call !== null) {
          const callIndent = next[call - 1].match(/^\s*/)?.[0] ?? indent;
          const lhs = next[call - 1].match(/^\s*(\w+)\s*=/)?.[1] ?? 'x';
          next.splice(call, 0, `${callIndent}${lhs} = self.${actName}(${lhs})`);
        }
        fixes.push({
          rule: f.rule, layer: c.name, confidence: 'proposal',
          summary: `Insert a GELU after ${c.name}; two stacked linears without one collapse into a single matrix.`,
          diff: unifiedDiff(path, source, next.join('\n')), lines: [at + 1],
          followUp: call === null ? 'Add a call to the new activation in forward() after the linear.' : 'If the stack is a deliberate low-rank factorisation, drop this instead.',
        });
        break;
      }
      case 'attention-no-pe': {
        if (!c || at === null) { fail('could not locate the attention layer'); break; }
        const dim = num(c.params.hiddenDim ?? c.params.embedDim);
        const indent = original[at - 1].match(/^\s*/)?.[0] ?? '        ';
        const next = base();
        next.splice(at - 1, 0, `${indent}self.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, ${dim ?? 'EMBED_DIM'}))  # learned positional embedding`);
        fixes.push({
          rule: f.rule, layer: c.name, confidence: 'proposal',
          summary: 'Add a learned positional embedding before the first attention layer.',
          diff: unifiedDiff(path, source, next.join('\n')), lines: [at],
          followUp: 'Replace SEQ_LEN with the sequence length, add `x = x + self.pos_embed` in forward() before the attention call, and import torch if the file only imports torch.nn.',
        });
        break;
      }
      case 'dropout-at-output':
      case 'output-activation':
      case 'bn-at-output': {
        if (!c) { fail('no layer named'); break; }
        const call = locateCall(c.name, original);
        if (call === null) { fail(`no statement calling self.${c.name} in forward()`); break; }
        const next = base(); next.splice(call - 1, 1);
        fixes.push({
          rule: f.rule, layer: c.name, confidence: 'proposal',
          summary: `Drop the ${c.name} call at the output.`,
          diff: unifiedDiff(path, source, next.join('\n')), lines: [call],
          followUp: 'Keep the layer definition or remove it too; a definition nothing calls is harmless.',
        });
        break;
      }
      case 'invalid-output-shape':
      case 'compute-error':
      case 'unknown-layer-type':
      case 'merge-shape-mismatch':
      case 'attention-in-mismatch':
        fail('a shape finding; on parsed source these are usually the parser\'s own unevaluated dimension, not a bug in the model');
        break;
      default:
        fail('no mechanical fix for this rule; read the finding\'s message, it names the change');
    }
  }

  const patched = lines.join('\n');
  return {
    path,
    fixes,
    notFixable,
    ...(fixes.some(x => x.confidence === 'exact') && patched !== source ? { patchedSource: patched } : {}),
  };
}
