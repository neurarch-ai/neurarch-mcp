/**
 * The calibration numbers are quoted in prose in six places the code cannot
 * write (README, package.json, manifest.json, AGENTS.md, the skill's tool
 * reference). The tool's own sentence is written from the bundle's
 * MEMORY_CALIBRATION; this test holds the prose copies to the same object, so
 * a new bundle with a refit fails here until every copy moves with it.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { MEMORY_CALIBRATION as C } from './vendor/plan.bundle.mjs';
import { ACCELERATE_FLAGGED_AT_MOST } from './lib/sizeJob.js';

const FILES = [
  'README.md',
  'package.json',
  'manifest.json',
  'AGENTS.md',
  'skills/neurarch-skill/references/neurarch-mcp-tools.md',
];

const read = (f: string) => readFileSync(new URL(`../${f}`, import.meta.url), 'utf-8').replace(/\s+/g, ' ');

describe('calibration numbers in prose match the vendored bundle', () => {
  for (const f of FILES) {
    it(f, () => {
      const s = read(f);
      expect(s, 'fitted runs').toMatch(new RegExp(`${C.fittedRuns} measured runs across ${C.architectures} architectures`));
      expect(s, 'held-out median').toContain(`${C.heldOutMedianPct}%`);
      expect(s, 'flagged').toContain(`${C.flagged} of the ${C.noFit}`);
      expect(s, 'accelerate').toMatch(new RegExp(`accelerate estimate-memory\`? (flagged )?at most ${ACCELERATE_FLAGGED_AT_MOST}`));
      // A held-out median is never quoted without the after-fit miss beside it.
      expect(s, 'after-fit miss').toContain(`${C.afterFitMedianPct}%`);
      expect(s, 'after-fit runs').toMatch(new RegExp(`${C.afterFitRuns} runs measured after the fit`));
      expect(s, 'stale three-factor numbers').not.toMatch(/flagged 54|54 of the 90|at most 9\b|median miss is 42%|141 measured runs/);
    });
  }
});
