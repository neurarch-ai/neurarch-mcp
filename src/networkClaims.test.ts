/**
 * Every sentence that tells a reader when this server opens a socket must name
 * every tool that does. The source of truth is the tool's own openWorldHint:
 * a tool that reaches the network says so in its annotation, and the HELP text,
 * the README and the server instructions must agree with the annotations.
 *
 * Found 2026-11-05: model_changes (0.20) and plan (0.16) reach the network, but
 * --help, the README and two error messages still described --hf as "the one
 * network call this server can make", and neurarch.com/mcp told buyers the
 * graph never leaves the machine (neurarch #460).
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { TOOLS } from './tools.js';
import { HF_TOOLS } from './extraTools.js';
import { SERVER_INSTRUCTIONS } from './server.js';

const root = path.resolve(__dirname, '..');
const read = (p: string) => fs.readFileSync(path.join(root, p), 'utf8');
const hf = new Set(HF_TOOLS.map((t) => t.name));
/** Tools that open a socket whenever they are called, independent of --hf. */
const SOCKET_TOOLS = TOOLS.filter((t) => t.annotations?.openWorldHint === true && !hf.has(t.name)).map((t) => t.name);

describe('network claims', () => {
  it('finds the tools that reach the network on their own', () => {
    expect([...SOCKET_TOOLS].sort()).toEqual(['history', 'model_changes', 'plan']);
  });

  it('--help names every one of them in its "no network calls at all" paragraph', () => {
    const help = read('src/index.ts');
    const para = help.slice(help.indexOf('reach the network when you call them') - 40, help.indexOf('no network calls at all') + 40);
    for (const t of SOCKET_TOOLS) expect(para, t).toContain(t);
  });

  it('the README condition for "no network calls at all" names every one of them', () => {
    const readme = read('README.md');
    const i = readme.indexOf('makes **no network calls at all**');
    expect(i).toBeGreaterThan(0);
    const sentence = readme.slice(readme.lastIndexOf('\n', i), i);
    for (const t of SOCKET_TOOLS) expect(sentence, t).toContain(`\`${t}\``);
  });

  it('the server instructions name every one of them', () => {
    const line = /Offline except[^.]*\./.exec(SERVER_INSTRUCTIONS)?.[0] ?? '';
    expect(line).not.toBe('');
    for (const t of SOCKET_TOOLS) expect(line, t).toContain(t);
  });

  it('nothing calls --hf the only network call', () => {
    const files = ['README.md', ...fs.readdirSync(path.join(root, 'src')).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts')).map((f) => `src/${f}`)];
    const bad = /the one network (call|switch)|only two in this server that reach the network|no tool opens a socket/i;
    expect(files.filter((f) => bad.test(read(f)))).toEqual([]);
  });
});
