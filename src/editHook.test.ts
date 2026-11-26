import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { runEditHook, editedFile, decideFromGate, type GateRunner } from './editHook.js';

let dir: string;
beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), 'nx-edit-'));
  await writeFile(join(dir, 'serve.sh'), 'vllm serve Qwen/Qwen2.5-7B-Instruct -tp 8\n');
  await writeFile(join(dir, 'notes.md'), 'vllm serve Qwen/Qwen2.5-7B-Instruct\n');
  await writeFile(join(dir, 'util.py'), 'def add(a, b):\n    return a + b\n');
});
afterAll(async () => { await rm(dir, { recursive: true, force: true }); });

const payload = (file: string, tool = 'Edit') => JSON.stringify({ hook_event_name: 'PostToolUse', tool_name: tool, tool_input: { file_path: file }, cwd: dir });
const gateSays = (json: object, calls: string[] = []): GateRunner => async (file) => { calls.push(file); return { code: 0, stdout: JSON.stringify(json) }; };
async function run(text: string, runner: GateRunner) {
  let out = ''; let err = '';
  const code = await runEditHook(text, { out: (s) => { out += s; }, err: (s) => { err += s; } }, runner);
  return { code, out, err };
}
const REFUSE = { verdict: 'refuse', because: ['1 launch blocker'], models: [], launches: [{ command: 'vllm serve Qwen/Qwen2.5-7B-Instruct -tp 8', findings: [{ severity: 'block', title: 'TP 8 does not divide 28 attention heads' }] }] };

describe('the edit hook', () => {
  it('a refused edit comes back to the agent as exit 2 with the reasons', async () => {
    const r = await run(payload(join(dir, 'serve.sh')), gateSays(REFUSE));
    expect(r.code).toBe(2);
    expect(r.err).toMatch(/refuses serve\.sh as edited \(1 launch blocker\)/);
    expect(r.err).toMatch(/TP 8 does not divide 28 attention heads/);
  });
  it('a review is context, not a stop; an admit is silence', async () => {
    const review = await run(payload(join(dir, 'serve.sh'), 'Write'), gateSays({ verdict: 'review', because: ['1 medium scan finding'], models: [{ id: 'Qwen/Qwen2.5-7B-Instruct', findings: [{ severity: 'medium', message: 'Not pinned to a commit' }] }], launches: [] }));
    expect(review.code).toBe(0);
    expect(JSON.parse(review.out).hookSpecificOutput.additionalContext).toMatch(/review \(1 medium scan finding\)[\s\S]*Not pinned/);
    const admit = await run(payload(join(dir, 'serve.sh')), gateSays({ verdict: 'admit', because: [] }));
    expect(admit).toEqual({ code: 0, out: '', err: '' });
  });
  it('never runs the gate on files that name no model, other kinds of file, or other tools', async () => {
    const calls: string[] = [];
    for (const p of [payload(join(dir, 'util.py')), payload(join(dir, 'notes.md')), JSON.stringify({ tool_name: 'Bash', tool_input: { command: 'ls' } }), 'not json']) {
      expect((await run(p, gateSays(REFUSE, calls))).code).toBe(0);
    }
    expect(calls).toEqual([]);
    expect(editedFile({ tool_name: 'Edit', tool_input: { file_path: 'k8s/Dockerfile' }, cwd: '/r' })?.file).toBe('/r/k8s/Dockerfile');
  });
  it('fails open: no gate, unparsable output, an exception, or the off switch', async () => {
    expect((await run(payload(join(dir, 'serve.sh')), async () => null)).code).toBe(0);
    expect((await run(payload(join(dir, 'serve.sh')), async () => ({ code: 1, stdout: 'npm ERR!' }))).code).toBe(0);
    expect((await run(payload(join(dir, 'serve.sh')), async () => { throw new Error('boom'); })).code).toBe(0);
    process.env.NEURARCH_ALLOW_EDIT = '1';
    try { expect((await run(payload(join(dir, 'serve.sh')), gateSays(REFUSE))).code).toBe(0); } finally { delete process.env.NEURARCH_ALLOW_EDIT; }
  });
  it('shows only what decided the verdict', () => {
    const d = decideFromGate('/r/a.py', { verdict: 'refuse', because: ['1 high scan finding'], models: [{ id: 'x/y', findings: [{ severity: 'high', message: 'runs code from another repo' }, { severity: 'medium', message: 'not pinned' }] }] });
    expect(d.stderr).toMatch(/runs code from another repo/);
    expect(d.stderr).not.toMatch(/not pinned/);
  });
});
