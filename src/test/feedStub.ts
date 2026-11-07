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

/** Shape of /changes/lineage.json (neurarch #387); acme/demo is made up. */
export const LINEAGE = {
  checked: 2,
  findings: [
    { id: 'acme/demo', base: 'acme/base', kinds: ['edit-declared-as-quantization'], basis: ['Declares itself a quantization of acme/base, whose id names no edit, while its own id or tags say "abliterated".'], sha: 'c', at: '2026-10-19T00:00:00Z' },
  ],
};

/** Shape of /changes-feeds/lineage.json (neurarch #459, #463); rows copied from the live feed on 2026-11-04 and 2026-11-05. */
export const WEIGHTS = {
  what: 'stub',
  license: 'CC-BY-4.0',
  models: [
    { id: 'ISxOdin/vit-base-oxford-iiit-pets', sha: '6fcb4ce433d4b43a19bd9b4d2d906b2646858d45', base: 'google/vit-base-patch16-224', baseSha: '3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3', verdict: 'head-only', common: 200, changed: 2, onlyBase: 0, onlyDerivative: 0, changedElementShare: 0.000332, examples: ['classifier.bias', 'classifier.weight'], sentence: 'Weights match google/vit-base-patch16-224 except the head: 2 head tensors changed, the other 198 identical.' },
    { id: 'Qwen/Qwen3-8B-FP8', sha: '220b46e3b2180893580a4454f21f22d3ebb187d3', base: 'Qwen/Qwen3-8B', baseSha: 'b968826d9c46dd6066d109eabc6255188de91218', verdict: 'requantized', common: 399, changed: 252, onlyBase: 0, onlyDerivative: 252, changedElementShare: 0.848003, examples: [], recast: 252, casts: [], requant: { direction: 'quantized', layers: 252, kept: 147, keptChanged: 0, keptRecast: 0 }, sentence: 'A quantization of Qwen/Qwen3-8B: 252 layers re-encoded; of the 147 tensors left unquantized, 147 byte-identical to the base.' },
    { id: 'Qwen/Qwen2.5-7B-Instruct-AWQ', sha: 'b25037543e9394b818fdfca67ab2a00ecc7dd641', base: 'Qwen/Qwen2.5-7B-Instruct', baseSha: 'a09a35458c702b33eeacc393d103063234e8bc28', verdict: 'requantized', common: 143, changed: 143, onlyBase: 196, onlyDerivative: 588, changedElementShare: 1, examples: [], recast: 143, casts: ['BF16>F16'], requant: { direction: 'quantized', layers: 196, kept: 59, keptChanged: 59, keptRecast: 59, keptChangedKinds: {} }, sentence: 'A quantization of Qwen/Qwen2.5-7B-Instruct: 196 layers re-encoded; of the 59 tensors left unquantized, 59 cast to another dtype (BF16 to F16), so their hashes cannot compare values.' },
    { id: 'nvidia/MiniMax-M2.5-NVFP4', sha: 'n', base: 'MiniMaxAI/MiniMax-M2.5', baseSha: 'm', verdict: 'requantized', common: 48239, changed: 47988, onlyBase: 47864, onlyDerivative: 142896, changedElementShare: 1, examples: [], recast: 47988, casts: ['F32>BF16'], requant: { direction: 'converted', layers: 47864, kept: 375, keptChanged: 124, keptRecast: 124 }, sentence: 'A conversion of MiniMaxAI/MiniMax-M2.5 from one quantized encoding to another: 47864 layers re-encoded; of the 375 tensors left unquantized, 251 byte-identical to the base, 124 cast to another dtype (F32 to BF16), so their hashes cannot compare values.' },
    { id: 'unsloth/Llama-3.2-1B-Instruct', sha: '5a8abab4a5d6f164389b1079fb721cfab8d7126c', base: 'meta-llama/Llama-3.2-1B-Instruct', baseSha: '9213176726f574b556790deb65791e0c5aa438b6', verdict: 'identical', common: 146, changed: 0, onlyBase: 0, onlyDerivative: 0, changedElementShare: 0, examples: [], sentence: 'Weights are byte-identical to meta-llama/Llama-3.2-1B-Instruct: all 146 tensors hash the same.' },
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
    if (url === 'https://www.neurarch.com/changes/lineage.json') return json(LINEAGE);
    if (url === `${BASE}/lineage.json`) return json(WEIGHTS);
    const rel = /\/renovate\/(.+)\.json$/.exec(url);
    if (rel) return RELEASES[rel[1]] ? json(RELEASES[rel[1]]) : json({ error: 'not found' }, 404);
    const adv = /\/advisories\/(x_NXMC-[\d-]+)\.json$/.exec(url);
    if (adv) return json({ id: adv[1], summary: `stub summary for ${adv[1]}`, database_specific: { changed_keys: [{ file: 'config', key: 'max_position_embeddings', before: 514, after: 512, class: 'structural' }] } });
    return json({ error: 'not found' }, 404);
  }) as FeedFetch;
  f.urls = urls;
  return f;
}
