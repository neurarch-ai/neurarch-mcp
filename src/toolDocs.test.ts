import { describe, it, expect } from 'vitest';
import { SHORT } from './toolDocs.js';
import { TOOLS } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';
import { HF_TOOLS } from './extraTools.js';
import { listedTools, CORE_TOOLS } from './cli.js';

describe('tool descriptions', () => {
  it('every tool has a short description under 220 characters', () => {
    for (const t of [...TOOLS, ...WRITE_TOOLS, ...HF_TOOLS]) {
      expect(SHORT[t.name], `${t.name} has no SHORT entry`).toBeDefined();
      expect(SHORT[t.name].length, `${t.name} short description too long`).toBeLessThanOrEqual(220);
    }
  });
  it('the core listing stays under 4.8k tokens and the full one under 6.7k (0.13 was 6.9k for the full set)', () => {
    // The budget moved once, in 0.16, when `plan` and `history` joined the core
    // set: two tools an agent should reach for first are worth ~0.7k a turn,
    // and the cap is raised deliberately rather than quietly. It is still a
    // cap: the next tool pays for itself out of somebody else's description.
    const core = JSON.stringify(listedTools(true, 'core')).length / 4;
    const full = JSON.stringify(listedTools(true, 'full')).length / 4;
    expect(core).toBeLessThan(4800);
    expect(full).toBeLessThan(6700);
  });
  it('core tools all exist', () => {
    const names = new Set(TOOLS.map(t => t.name));
    for (const c of CORE_TOOLS) expect(names.has(c), c).toBe(true);
  });
});
