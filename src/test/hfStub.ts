/**
 * An offline Hugging Face for tests: a `fetch` stub that serves the files under
 * src/test/hf/<name>/ for the repo ids mapped to them, and 404s everything
 * else. With it, the --hf path (loadHFModel, loadHFServeFiles) runs end to end
 * and no test opens a socket.
 *
 * src/test/hf/Qwen3-8B holds config.json, tokenizer_config.json and
 * generation_config.json exactly as huggingface.co/Qwen/Qwen3-8B served them at
 * revision b968826d9c46dd6066d109eabc6255188de91218 (fetched 2026-10-18). The
 * repo ships no chat_template.jinja; the template is inside tokenizer_config.
 * Its safetensors total, 8,190,735,360, is the Hub API's at the same revision.
 */
import { readFile } from 'node:fs/promises';
import { vi } from 'vitest';

export const QWEN3_8B = 'Qwen/Qwen3-8B';
export const QWEN3_8B_PARAMS = 8190735360;

const REPOS: Record<string, { dir: string; params: number }> = {
  [QWEN3_8B]: { dir: 'Qwen3-8B', params: QWEN3_8B_PARAMS },
};

export async function fixtureFile(dir: string, file: string): Promise<string | null> {
  return readFile(new URL(`./hf/${dir}/${file}`, import.meta.url), 'utf-8').catch(() => null);
}

/** Replace globalThis.fetch with the fixture server. Returns the spy, so a test can count calls. */
export function stubHuggingFace() {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: string | URL | Request) => {
    const url = String(input instanceof Request ? input.url : input);
    const api = /^https:\/\/huggingface\.co\/api\/models\/([^/]+\/[^/?]+)/.exec(url);
    if (api && REPOS[api[1]]) {
      return new Response(JSON.stringify({ id: api[1], safetensors: { total: REPOS[api[1]].params } }), { status: 200 });
    }
    const file = /^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/resolve\/main\/(.+)$/.exec(url);
    if (file && REPOS[file[1]]) {
      const text = await fixtureFile(REPOS[file[1]].dir, file[2]);
      if (text != null) return new Response(text, { status: 200 });
    }
    return new Response('Entry not found', { status: 404 });
  });
}
