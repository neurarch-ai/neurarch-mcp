/**
 * The property that matters most here is not "does it parse the response".
 *
 * It is that this server's README promise holds: with no NEURARCH_API_KEY set,
 * nothing opens a socket. That promise is why someone is willing to point this
 * at a private model file, and a tool that quietly broke it would be the worst
 * possible regression in this package.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { checkDesign, apiKey, NO_KEY_MESSAGE, DEFAULT_CHECK_URL } from './checkDesign.js';
import type { ModelArchitecture } from './types.js';

const model = (): ModelArchitecture => ({
  id: 'm',
  name: 'T',
  components: [
    { id: 'i', name: 'input', type: 'input', position: { x: 0, y: 0 }, params: { shape: [3, 32, 32] }, inputs: [], outputs: ['c'] },
    { id: 'c', name: 'conv', type: 'conv2d', position: { x: 0, y: 0 }, params: { inChannels: 3, outChannels: 16, kernelSize: 3 }, inputs: ['i'], outputs: [] },
  ],
  connections: [{ id: 'e', from: 'i', to: 'c', fromPort: 'right', toPort: 'left' }],
} as unknown as ModelArchitecture);

const OK_BODY = {
  verdict: 'ask',
  outcome: 'needs_input',
  summary: 'No dataset is wired yet, so a run would train on random tensors.',
  stoppedAt: null,
  findings: [],
  stages: [{ stage: 'preflight', status: 'ok', headline: 'Ready to train', data: { params: 448 } }],
  decision: { question: 'Which data should this train on?', because: '…', options: [{ label: 'CIFAR-10', value: 'hf:cifar10' }] },
};

let prevKey: string | undefined;
beforeEach(() => { prevKey = process.env.NEURARCH_API_KEY; });
afterEach(() => {
  if (prevKey === undefined) delete process.env.NEURARCH_API_KEY;
  else process.env.NEURARCH_API_KEY = prevKey;
  vi.restoreAllMocks();
});

describe('the offline promise', () => {
  it('opens no socket at all when no key is set', async () => {
    delete process.env.NEURARCH_API_KEY;
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    const out = await checkDesign(model());
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(out).toEqual({ error: NO_KEY_MESSAGE });
  });

  it('treats an empty or whitespace key as no key', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    for (const blank of ['', '   ']) {
      process.env.NEURARCH_API_KEY = blank;
      expect(apiKey()).toBeUndefined();
      await checkDesign(model());
    }
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('says what is missing, where to get it, and what still works', async () => {
    delete process.env.NEURARCH_API_KEY;
    const out = await checkDesign(model()) as { error: string };
    expect(out.error).toMatch(/Settings → Developer API/);
    expect(out.error).toMatch(/NEURARCH_API_KEY/);
    expect(out.error).toMatch(/still works offline/);
    expect(out.error).toMatch(/validate_model/);
  });
});

describe('with a key', () => {
  it('posts the graph with the key and returns the verdict', async () => {
    process.env.NEURARCH_API_KEY = 'nrk_test';
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(OK_BODY), { status: 200, headers: { 'Content-Type': 'application/json' } }),
    );
    const out = await checkDesign(model());
    expect(out).toMatchObject({ verdict: 'ask', summary: OK_BODY.summary });

    const [url, init] = fetchSpy.mock.calls[0];
    expect(String(url)).toMatch(/\/api\/v1\/check$/);
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers.Authorization).toBe('Bearer nrk_test');
    expect(JSON.parse((init as RequestInit).body as string).model.components).toHaveLength(2);
  });

  it('passes the server\'s own message through instead of flattening it', async () => {
    process.env.NEURARCH_API_KEY = 'nrk_test';
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ error: 'Invalid or missing API key.' }), { status: 401 }),
    );
    const out = await checkDesign(model()) as { error: string };
    expect(out.error).toMatch(/HTTP 401/);
    expect(out.error).toMatch(/Invalid or missing API key/);
  });

  it('survives a non-JSON error body rather than throwing on it', async () => {
    process.env.NEURARCH_API_KEY = 'nrk_test';
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('<html>502</html>', { status: 502 }));
    const out = await checkDesign(model()) as { error: string };
    expect(out.error).toMatch(/HTTP 502/);
  });

  it('reports a network failure as an error, never as a verdict', async () => {
    process.env.NEURARCH_API_KEY = 'nrk_test';
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('ECONNREFUSED'));
    const out = await checkDesign(model()) as { error: string };
    expect(out.error).toMatch(/could not reach/);
    expect(out).not.toHaveProperty('verdict');
  });

  it('honours NEURARCH_API_URL for self-hosted or test endpoints', async () => {
    // Reads the env at call time, so setting it after import works. A module
    // -scope capture would need a re-import here, which is the smell that
    // says the module would also ignore a host configuring it late.
    process.env.NEURARCH_API_KEY = 'nrk_test';
    const prevUrl = process.env.NEURARCH_API_URL;
    process.env.NEURARCH_API_URL = 'http://localhost:3000/api/v1/check';
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(OK_BODY), { status: 200 }),
    );
    await checkDesign(model());
    expect(String(fetchSpy.mock.calls[0][0])).toBe('http://localhost:3000/api/v1/check');
    if (prevUrl === undefined) delete process.env.NEURARCH_API_URL;
    else process.env.NEURARCH_API_URL = prevUrl;
  });
  it('defaults to a host that does not redirect, so the key survives the hop', () => {
    // 0.10.0 shipped pointing at the apex, which 307s to www. Node's fetch
    // follows the hop and strips Authorization across origins, so every user
    // with a perfectly good key was told the key was invalid. The default host
    // is therefore part of the contract, not a cosmetic choice.
    expect(DEFAULT_CHECK_URL).toBe('https://www.neurarch.com/api/v1/check');
    expect(new URL(DEFAULT_CHECK_URL).hostname).not.toBe('neurarch.com');
  });

  it('sends the key on the request it actually makes', async () => {
    process.env.NEURARCH_API_KEY = 'nrk_live';
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(OK_BODY), { status: 200 }),
    );
    await checkDesign(model());
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(String(url)).toBe(DEFAULT_CHECK_URL);
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer nrk_live');
  });
});
