import { describe, it, expect } from 'vitest';
import { resolve } from 'node:path';
import { runLintCommand, runCheckCommand } from './commands.js';

function io() {
  const out: string[] = []; const err: string[] = [];
  return { out: (s: string) => out.push(s), err: (s: string) => err.push(s), text: () => out.join(''), errors: () => err.join('') };
}

const VIT = resolve(__dirname, '../examples/tiny-vit.py');

describe('neurarch-mcp lint', () => {
  it('exits 1 on a block and prints rule, layer and message', async () => {
    const o = io();
    expect(await runLintCommand([VIT], false, o)).toBe(1);
    expect(o.text()).toMatch(/1 block/);
    expect(o.text()).toMatch(/block head-dim-divisibility \[attn:multiHeadAttention\]/);
  });
  it('exits 0 on a clean zoo entry and 2 on an unreadable input', async () => {
    expect(await runLintCommand(['zoo:resnet-50'], false, io())).toBe(0);
    const o = io();
    expect(await runLintCommand(['/nonexistent/model.py'], false, o)).toBe(2);
    expect(o.errors()).toMatch(/Cannot read/);
  });
  it('--json emits one object for one file and a map for several', async () => {
    const one = io();
    await runLintCommand([VIT], true, one);
    expect(JSON.parse(one.text()).blocks).toBe(1);
    const two = io();
    await runLintCommand([VIT, 'zoo:resnet-50'], true, two);
    expect(Object.keys(JSON.parse(two.text()))).toEqual([VIT, 'zoo:resnet-50']);
  });
  it('prints usage with no files', async () => {
    const o = io();
    expect(await runLintCommand([], false, o)).toBe(2);
    expect(o.errors()).toMatch(/usage/);
  });
});

describe('neurarch-mcp check', () => {
  it('prints the verdict and the stages walked', async () => {
    const o = io();
    const code = await runCheckCommand(['zoo:bert-base'], false, o);
    expect([0, 1]).toContain(code);
    expect(o.text()).toMatch(/preflight/);
  });
  it('--json emits the raw DesignCheck', async () => {
    const o = io();
    await runCheckCommand(['zoo:bert-base'], true, o);
    const r = JSON.parse(o.text());
    expect(r.stages[0].stage).toBe('preflight');
  });
  it('takes exactly one file', async () => {
    expect(await runCheckCommand([], false, io())).toBe(2);
    expect(await runCheckCommand(['a', 'b'], false, io())).toBe(2);
  });
});
