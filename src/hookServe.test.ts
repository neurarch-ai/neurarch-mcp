/**
 * The hook's serving half: a vLLM or SGLang launch whose own config.json and
 * flags say it will not start (or will serve wrongly) is blocked with the
 * finding; everything else runs untouched. Offline, on the real Qwen3-8B
 * files (src/test/hfStub.ts).
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { runHookCommand, parseServeLaunch } from './hook.js';
import { setHfEnabled } from './sources.js';
import { stubHuggingFace } from './test/hfStub.js';

let cache: string;
const realCache = process.env.NEURARCH_MCP_CACHE;
beforeEach(async () => {
  cache = await mkdtemp(join(tmpdir(), 'nx-hook-serve-'));
  process.env.NEURARCH_MCP_CACHE = cache;
  stubHuggingFace();
  setHfEnabled(true);
});
afterEach(async () => {
  setHfEnabled(false);
  vi.restoreAllMocks();
  if (realCache === undefined) delete process.env.NEURARCH_MCP_CACHE; else process.env.NEURARCH_MCP_CACHE = realCache;
  await rm(cache, { recursive: true, force: true });
});

async function bash(command: string): Promise<{ code: number; err: string }> {
  let err = '';
  const code = await runHookCommand(JSON.stringify({ tool_name: 'Bash', tool_input: { command }, cwd: cache }), { out: () => undefined, err: s => { err += s; } });
  return { code, err };
}

describe('parseServeLaunch', () => {
  it('finds vLLM and SGLang launches by the command word, past env prefixes and separators', () => {
    expect(parseServeLaunch('vllm serve Qwen/Qwen3-8B -tp 4')?.engine).toBe('vllm');
    expect(parseServeLaunch('CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve Qwen/Qwen3-8B')?.words.slice(0, 2)).toEqual(['vllm', 'serve']);
    expect(parseServeLaunch('cd /srv && python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B')?.engine).toBe('vllm');
    expect(parseServeLaunch('python -m sglang.launch_server --model-path Qwen/Qwen3-8B --tp 2')?.engine).toBe('sglang');
  });
  it('is not fooled by a mention', () => {
    expect(parseServeLaunch('echo "vllm serve Qwen/Qwen3-8B"')).toBeNull();
    expect(parseServeLaunch('pip install vllm sglang')).toBeNull();
    expect(parseServeLaunch('grep -r "vllm serve" docs/')).toBeNull();
    expect(parseServeLaunch('llama-server -m Qwen3-8B-Q4_K_M.gguf -c 8192')).toBeNull();
  });
});

describe('neurarch-mcp hook, on a serving launch', () => {
  it('blocks TP 3 on Qwen3-8B (32 heads, 8 KV heads), with the finding, the fix and the override', async () => {
    const r = await bash('vllm serve Qwen/Qwen3-8B --tensor-parallel-size 3');
    expect(r.code).toBe(2);
    expect(r.err).toContain('neurarch: this vLLM launch has 2 blocking findings for Qwen/Qwen3-8B.');
    expect(r.err).toContain('TP 3 does not divide');
    expect(r.err).toMatch(/fix: /);
    expect(r.err).toContain('Read from hf:Qwen/Qwen3-8B (config.json) and the flags');
    expect(r.err).toContain('NEURARCH_ALLOW_SERVE=1');
  });

  it('blocks a served length past the 40960-token context, unless the engine\'s own override is set', async () => {
    const r = await bash('vllm serve Qwen/Qwen3-8B --max-model-len 131072');
    expect(r.code).toBe(2);
    expect(r.err).toContain('past the model\'s 40960-token context');
    expect((await bash('VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3-8B --max-model-len 131072')).code).toBe(0);
  });

  it('blocks on SGLang too', async () => {
    const r = await bash('python -m sglang.launch_server --model-path Qwen/Qwen3-8B --tp 3');
    expect(r.code).toBe(2);
    expect(r.err).toContain('this SGLang launch');
  });

  it('lets a clean launch, an override, a mention and llama.cpp through silently', async () => {
    for (const cmd of [
      'vllm serve Qwen/Qwen3-8B --tensor-parallel-size 4 --max-model-len 32768',
      'NEURARCH_ALLOW_SERVE=1 vllm serve Qwen/Qwen3-8B --tensor-parallel-size 3',
      'echo "vllm serve Qwen/Qwen3-8B --tensor-parallel-size 3"',
      'llama-server -m Qwen3-8B-Q4_K_M.gguf -c 131072',
    ]) {
      const r = await bash(cmd);
      expect(r, cmd).toEqual({ code: 0, err: '' });
    }
  });

  it('fails open when the model cannot be read: no --hf and nothing cached, a repo that is not there', async () => {
    expect((await bash('vllm serve no-such-org/no-such-model --tensor-parallel-size 3')).code).toBe(0);
    setHfEnabled(false);
    expect((await bash('vllm serve Qwen/Qwen3-8B --tensor-parallel-size 3')).code).toBe(0);
  });
});
