import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { checkPinsTool } from './pinTools.js';
import type { ToolContext } from './tools.js';

let dir: string;
const call = (args: Record<string, unknown>, ctx: Partial<ToolContext> = {}) =>
  checkPinsTool.handler(args, undefined as never, { modelPath: join(dir, 'model.neurarch.json'), ...ctx }) as Promise<any>;

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'nx-pins-'));
  await mkdir(join(dir, 'configs'));
  await mkdir(join(dir, '.venv', 'lib'), { recursive: true });
  await writeFile(join(dir, 'train.py'), [
    'from transformers import AutoModel, AutoTokenizer',
    'm = AutoModel.from_pretrained("Qwen/Qwen3-8B")',
    't = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision="main")',
    'p = AutoModel.from_pretrained(path)',
    'ok = AutoModel.from_pretrained("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")',
  ].join('\n'));
  await writeFile(join(dir, 'serve.sh'), 'vllm serve meta-llama/Llama-3.1-8B -tp 2\n');
  await writeFile(join(dir, 'configs', 'sft.yaml'), 'base_model: mistralai/Mistral-7B-v0.1\n');
  await writeFile(join(dir, 'README.md'), 'from_pretrained("not/scanned")\n');
  // A virtualenv is not the user's code, as in the CLI.
  await writeFile(join(dir, '.venv', 'lib', 'x.py'), 'AutoModel.from_pretrained("vendored/model")\n');
});
afterEach(async () => { await rm(dir, { recursive: true, force: true }); });

describe('check_pins', () => {
  it('walks the model\'s directory and reports each unpinned download with a fix for its file', async () => {
    const r = await call({});
    expect(r.filesRead).toBe(3);
    expect(r.counts).toEqual({ warn: 4, info: 1 });
    expect(r.verdict).toBe('4 downloads not pinned to a commit in 3 files');
    expect(r.findings.map((f: any) => `${f.path}:${f.line} ${f.rule}`)).toEqual([
      'configs/sft.yaml:1 pin/hf-unpinned',
      'serve.sh:1 pin/hf-unpinned',
      'train.py:2 pin/hf-unpinned',
      'train.py:3 pin/hf-floating-revision',
    ]);
    expect(r.findings.find((f: any) => f.path === 'serve.sh').fix).toContain('--revision <sha>');
    expect(r.findings.find((f: any) => f.path === 'configs/sft.yaml').fix).toContain('model_revision: <sha>');
    expect(r.findings.every((f: any) => !f.fix.includes('Then extend'))).toBe(true);
    expect(r.hiddenDynamic).toBe(1);
  });

  it('shows dynamic ids on request, and pages', async () => {
    const all = await call({ include_dynamic: true });
    expect(all.findings).toHaveLength(5);
    const p1 = await call({ include_dynamic: true, limit: 3 });
    const p2 = await call({ include_dynamic: true, limit: 3, cursor: p1.nextCursor });
    expect([...p1.findings, ...p2.findings]).toEqual(all.findings);
    expect(p2.nextCursor).toBeNull();
  });

  it('reads inline source by filename, and refuses a name it cannot read', async () => {
    const r = await call({ source: 'from huggingface_hub import snapshot_download\nsnapshot_download(repo_id="org/m")\n', filename: 'dl.py' });
    expect(r.findings[0].rule).toBe('pin/hf-unpinned');
    await expect(call({ source: 'x', filename: 'notes.txt' })).rejects.toThrow(/not a file kind/);
    await expect(call({ source: 'x' })).rejects.toThrow(/needs filename/);
  });

  it('says where a missing path was resolved', async () => {
    await expect(call({ paths: ['nope'] })).rejects.toThrow(new RegExp(`nope: no such file or directory \\(resolved against ${dir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`));
  });

  it('reports progress while walking a large tree', async () => {
    for (let i = 0; i < 260; i++) await writeFile(join(dir, `f${i}.py`), 'x = 1\n');
    const seen: number[] = [];
    await call({}, { progress: async (p) => { seen.push(p); } });
    expect(seen).toEqual([250]);
  });
});
