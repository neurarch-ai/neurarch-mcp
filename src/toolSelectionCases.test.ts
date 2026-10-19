/**
 * The first-call eval (scripts/eval-first-tool.mjs) costs tokens and is not in
 * CI, but its case file can be kept honest for free: a question that names a
 * tool measures string matching, and an accept set naming a tool that does
 * not exist can never pass.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { listedTools } from './cli.js';

const { cases } = JSON.parse(readFileSync(new URL('../scripts/tool-selection-cases.json', import.meta.url), 'utf8')) as {
  cases: { id: string; ask: string; accept: string[] }[];
};

describe('tool-selection cases', () => {
  const names = listedTools(true, 'full').map(t => t.name);
  it('has 30 to 50 cases with unique ids', () => {
    expect(cases.length).toBeGreaterThanOrEqual(30);
    expect(cases.length).toBeLessThanOrEqual(50);
    expect(new Set(cases.map(c => c.id)).size).toBe(cases.length);
  });
  it('never names a tool in the question', () => {
    for (const c of cases) for (const n of names) expect(new RegExp(`\\b${n}\\b`, "i").test(c.ask), `${c.id} names ${n}`).toBe(false);
  });
  it('accepts only real tools, and never more than three', () => {
    for (const c of cases) {
      expect(c.accept.length, c.id).toBeGreaterThan(0);
      expect(c.accept.length, c.id).toBeLessThanOrEqual(3);
      for (const n of c.accept) expect(names, `${c.id}: ${n}`).toContain(n);
    }
  });
});
