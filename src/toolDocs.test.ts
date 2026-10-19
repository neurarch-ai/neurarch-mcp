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
  it('the core listing stays under 5.5k tokens and the full one under 7.8k (0.13 was 6.9k for the full set)', () => {
    // The budget moved once, in 0.16, when `plan` and `history` joined the core
    // set: two tools an agent should reach for first are worth ~0.7k a turn,
    // and the cap is raised deliberately rather than quietly. It is still a
    // cap: the next tool pays for itself out of somebody else's description.
    // size_training_job did (~0.35k): rank_designs left the core listing, and
    // the plan, trace, rank and find_layers schemas were trimmed to fit.
    // check_edit (~0.25k) did too: the model_path description injected on
    // every read tool was cut to a third of its length.
    //
    // 0.20 raised it again, deliberately, by ~0.6k: output schemas on the
    // tools whose shape is stable (the contract a client validates against,
    // trimmed to the keys every code path returns) and check_pins plus
    // model_changes in the core set, which mermaid_diagram and export_pytorch
    // left for it (both are also resources).
    //
    // 0.21 raised the core cap by 0.1k for `response_format` on the five
    // tools whose answers run to kilobytes: ~90 tokens of listing a turn
    // against up to ~2.2k tokens saved on each lint_model call on a 7B model.
    const core = JSON.stringify(listedTools(true, 'core')).length / 4;
    const full = JSON.stringify(listedTools(true, 'full')).length / 4;
    expect(core).toBeLessThan(5500);
    expect(full).toBeLessThan(7800);
  });
  it('core tools all exist', () => {
    const names = new Set(TOOLS.map(t => t.name));
    for (const c of CORE_TOOLS) expect(names.has(c), c).toBe(true);
  });
  it('listed descriptions make no quality or accuracy claim and no superlative', () => {
    // A tool description is also an attack surface and a selection signal:
    // boastful wording is what poisoned descriptions use to win selection
    // (arXiv 2509.24272), and this server's numbers are estimates and
    // predictions, never a statement about how well a model will train.
    for (const [name, text] of Object.entries(SHORT)) {
      expect(text, name).not.toMatch(/\b(best|better|optimal|guarantee[sd]?|accurate|accuracy|always works)\b/i);
    }
  });
});
