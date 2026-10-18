/**
 * The README's "What it looks like" block quotes size_training_job's output for
 * two calls on the phi3-mini template. It is prose the code cannot write, and
 * after the 0.17.1 refit it went on quoting the 0.17.0 numbers. This test runs
 * the same two calls and holds the quoted verdicts, peaks and maxBatch row to
 * the tool's current answer, so the next refit fails here until the example
 * moves with it.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { TOOLS } from './tools.js';
import type { ModelArchitecture } from './lib/types.js';

const tool = TOOLS.find(t => t.name === 'size_training_job')!;
const model = JSON.parse(
  readFileSync(new URL('./test/templates/phi3-mini.neurarch.json', import.meta.url), 'utf-8'),
) as ModelArchitecture;
const call = (args: Record<string, unknown>) =>
  tool.handler(args, model, { modelPath: '' }) as Promise<any>;

const readme = readFileSync(new URL('../README.md', import.meta.url), 'utf-8');
const section = readme.slice(readme.indexOf('## What it looks like'));
const block = section.slice(section.indexOf('```') + 3, section.indexOf('```', section.indexOf('```') + 3));
const [t4Part, eagerPart] = block.split('(same job, attention: "eager", gpu: "A100-40")');

/** A labelled field, with its indented continuation lines joined by one space. */
function field(part: string, label: string): string {
  const lines = part.split('\n');
  const i = lines.findIndex(l => l.trimStart().startsWith(`${label} `));
  if (i === -1) return '';
  const out = [lines[i].trimStart().slice(label.length).trim()];
  for (let j = i + 1; j < lines.length && /^\s{10,}\S/.test(lines[j]); j++) out.push(lines[j].trim());
  return out.join(' ');
}

describe('README "What it looks like" quotes the current size_training_job output', () => {
  it('has both calls', () => {
    expect(t4Part).toContain('size_training_job { batch_size: 32, gpu: "T4" }');
    expect(eagerPart).toBeDefined();
  });

  it('batch 32 on a T4', async () => {
    const r = await call({ batch_size: 32, gpu: 'T4' });
    expect(field(t4Part, 'verdict')).toBe(r.verdict);
    expect(field(t4Part, 'memory')).toContain(
      `peak ${r.memory.peakGb} GB = ${r.memory.weightsAndOptimizerGb} weights + optimizer state, ${r.memory.activationGb} activations`,
    );
    expect(field(t4Part, 'memory')).toContain(`the ${r.memory.attentionScoresDroppedGb} GB of attention scores eager would keep`);
    const quoted = Object.fromEntries(
      field(t4Part, 'maxBatch').split(', ').map(p => { const k = p.lastIndexOf(' '); return [p.slice(0, k), Number(p.slice(k + 1))]; }),
    );
    expect(quoted).toEqual(r.maxBatch);
    expect(field(t4Part, 'cost')).toContain(`$${r.cost.usd} on a T4`);
    expect(field(t4Part, 'lastRun')).toContain(r.lastRun.fingerprint);
  });

  it('the same job under eager attention on an A100-40', async () => {
    const r = await call({ batch_size: 32, gpu: 'A100-40', attention: 'eager' });
    expect(field(eagerPart, 'verdict')).toBe(r.verdict);
    expect(field(eagerPart, 'memory')).toContain(
      `peak ${r.memory.peakGb} GB = ${r.memory.weightsAndOptimizerGb} weights + optimizer state, ${r.memory.activationGb} activations`,
    );
    expect(field(eagerPart, 'memory')).toContain(`${r.memory.attentionScoresGb} GB of them`);
    // Quote a two-card split only when the tool returns one.
    if (r.twoCardSplit) expect(eagerPart).toMatch(/twoCardSplit/);
    else expect(eagerPart).not.toMatch(/twoCardSplit/);
  });
});
