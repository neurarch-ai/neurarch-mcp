/**
 * A fetch stub serving the slice of www.neurarch.com/changes-feeds/ that
 * model_changes reads. Qwen/Qwen3-8B is copied from the live feed on
 * 2026-10-19 (index row and revision list as published); acme/demo is made up
 * to exercise the filters: two changes, one breaking.
 */
const BASE = 'https://www.neurarch.com/changes-feeds';

const row = (o: Record<string, unknown>) => ({
  modified: '2026-10-19T02:33:38.839Z',
  published: '2026-10-18T09:00:52.565Z',
  impact: { memory: false, serving: true, tokenizer: true, licence: null },
  ...o,
});

export const INDEX = {
  what: 'stub',
  license: 'CC-BY-4.0',
  count: 3,
  withdrawn: 0,
  advisories: [
    row({ id: 'x_NXMC-2025-0065', model: 'Qwen/Qwen3-8B', from: '47719a242beab8f9aecc40ce3928b034dd5dd559', to: '895c8d171bc03c30e113cd7a28c02494b5e068b7', date: '2025-05-19T07:46:12.000Z', kind: 'template', breaking: false, url: `${BASE}/advisories/x_NXMC-2025-0065.json` }),
    row({ id: 'x_NXMC-2026-9001', model: 'acme/demo', from: 'aaaaaaa1111111111111111111111111111111aa', to: 'bbbbbbb2222222222222222222222222222222bb', date: '2026-01-01T00:00:00.000Z', kind: 'structural', breaking: true, impact: { memory: true, serving: true, tokenizer: false, licence: null }, url: `${BASE}/advisories/x_NXMC-2026-9001.json` }),
    row({ id: 'x_NXMC-2026-9002', model: 'acme/demo', from: 'bbbbbbb2222222222222222222222222222222bb', to: 'ccccccc3333333333333333333333333333333cc', date: '2026-02-01T00:00:00.000Z', kind: 'generation', breaking: false, url: `${BASE}/advisories/x_NXMC-2026-9002.json` }),
  ],
};

const RELEASES: Record<string, unknown> = {
  'Qwen/Qwen3-8B': {
    changelogUrl: `${BASE}/models/Qwen/Qwen3-8B.html`,
    releases: [
      { version: '47719a242beab8f9aecc40ce3928b034dd5dd559', digest: '47719a242beab8f9aecc40ce3928b034dd5dd559', releaseTimestamp: '2025-04-28T06:31:52.000Z' },
      { version: '895c8d171bc03c30e113cd7a28c02494b5e068b7', digest: '895c8d171bc03c30e113cd7a28c02494b5e068b7', releaseTimestamp: '2025-05-19T07:46:12.000Z' },
      { version: 'main', digest: 'b968826d9c46dd6066d109eabc6255188de91218', releaseTimestamp: '2025-07-26T03:49:13.000Z' },
    ],
  },
  'acme/demo': {
    changelogUrl: `${BASE}/models/acme/demo.html`,
    releases: [
      { version: 'aaaaaaa1111111111111111111111111111111aa', digest: 'aaaaaaa1111111111111111111111111111111aa', releaseTimestamp: '2025-12-01T00:00:00.000Z' },
      { version: 'bbbbbbb2222222222222222222222222222222bb', digest: 'bbbbbbb2222222222222222222222222222222bb', releaseTimestamp: '2026-01-01T00:00:00.000Z' },
      { version: 'main', digest: 'ccccccc3333333333333333333333333333333cc', releaseTimestamp: '2026-02-01T00:00:00.000Z' },
    ],
  },
};

export interface FeedFetch {
  (input: string | URL | Request, init?: RequestInit): Promise<Response>;
  urls: string[];
}

/** `delayMs` holds every answer back (for cancellation tests); an aborted signal rejects as fetch does. */
export function feedFetch(opts: { delayMs?: number; fail?: boolean } = {}): FeedFetch {
  const urls: string[] = [];
  const f = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input instanceof Request ? input.url : input);
    urls.push(url);
    if (opts.fail) throw new TypeError('fetch failed');
    if (opts.delayMs) {
      await new Promise<void>((res, rej) => {
        const t = setTimeout(res, opts.delayMs);
        init?.signal?.addEventListener('abort', () => { clearTimeout(t); rej(Object.assign(new Error('aborted'), { name: 'AbortError' })); }, { once: true });
      });
    }
    const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
    if (url === `${BASE}/advisories/index.json`) return json(INDEX);
    const rel = /\/renovate\/(.+)\.json$/.exec(url);
    if (rel) return RELEASES[rel[1]] ? json(RELEASES[rel[1]]) : json({ error: 'not found' }, 404);
    const adv = /\/advisories\/(x_NXMC-[\d-]+)\.json$/.exec(url);
    if (adv) return json({ id: adv[1], summary: `stub summary for ${adv[1]}`, database_specific: { changed_keys: [{ file: 'config', key: 'max_position_embeddings', before: 514, after: 512, class: 'structural' }] } });
    return json({ error: 'not found' }, 404);
  }) as FeedFetch;
  f.urls = urls;
  return f;
}
