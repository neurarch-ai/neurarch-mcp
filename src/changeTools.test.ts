import { describe, it, expect, afterEach, vi } from 'vitest';
import { modelChangesTool, clearChangeCache } from './changeTools.js';
import { feedFetch } from './test/feedStub.js';

const call = (args: Record<string, unknown>) => modelChangesTool.handler(args, undefined as never, { modelPath: '' }) as Promise<any>;

afterEach(() => { vi.unstubAllGlobals(); clearChangeCache(); delete process.env.NEURARCH_FEEDS; });

describe('model_changes', () => {
  it('lists a tracked model\'s recorded changes newest first, with main\'s commit', async () => {
    vi.stubGlobal('fetch', feedFetch());
    const r = await call({ hf_id: 'acme/demo' });
    expect(r.tracked).toBe(true);
    expect(r.changes.map((c: any) => c.id)).toEqual(['x_NXMC-2026-9002', 'x_NXMC-2026-9001']);
    expect(r.counts).toEqual({ total: 2, breaking: 1 });
    expect(r.verdict).toBe('2 recorded changes, 1 breaking; main is ccccccc33333 (2026-02-01)');
    expect(r.basis).toMatch(/not that anything broke/);
  });

  it('since_revision keeps only changes after the pin, and says when the pin is unknown', async () => {
    vi.stubGlobal('fetch', feedFetch());
    const after = await call({ hf_id: 'acme/demo', since_revision: 'bbbbbbb2222' });
    expect(after.pin).toEqual({ revision: 'bbbbbbb2222', known: true, date: '2026-01-01T00:00:00.000Z' });
    expect(after.changes.map((c: any) => c.id)).toEqual(['x_NXMC-2026-9002']);
    const unknown = await call({ hf_id: 'acme/demo', since_revision: 'deadbeef' });
    expect(unknown.pin.known).toBe(false);
    expect(unknown.changes).toHaveLength(2);
    expect(unknown.verdict).toContain('deadbeef is not a revision the record holds');
  });

  it('breaking_only and detail: full', async () => {
    const f = feedFetch();
    vi.stubGlobal('fetch', f);
    const r = await call({ hf_id: 'acme/demo', breaking_only: true, detail: 'full' });
    expect(r.changes).toHaveLength(1);
    expect(r.changes[0].changedKeys[0].key).toBe('max_position_embeddings');
    expect(f.urls.filter(u => /x_NXMC-2026-9001\.json$/.test(u))).toHaveLength(1);
  });

  it('carries the lineage finding for the model, empty when it has none, null when the record cannot be read', async () => {
    vi.stubGlobal('fetch', feedFetch());
    const hit = await call({ hf_id: 'acme/demo' });
    expect(hit.lineage).toEqual({ base: 'acme/base', findings: [{ kind: 'edit-declared-as-quantization', basis: expect.stringMatching(/quantization of acme\/base/) }] });
    const none = await call({ hf_id: 'Qwen/Qwen3-8B' });
    expect(none.lineage).toEqual({ base: null, findings: [] });
    clearChangeCache();
    const f = feedFetch();
    vi.stubGlobal('fetch', (async (u: string, i?: RequestInit) => (String(u).endsWith('/changes/lineage.json') ? new Response('nope', { status: 500 }) : f(u, i))) as typeof fetch);
    const down = await call({ hf_id: 'acme/demo' });
    expect(down.lineage).toBeNull();
    expect(down.changes).toHaveLength(2); // the change list still answers
  });

  it('weights: a derivative against its declared base, a base\'s tracked derivatives, and nothing claimed when unread', async () => {
    vi.stubGlobal('fetch', feedFetch());
    const d = await call({ hf_id: 'ISxOdin/vit-base-oxford-iiit-pets' });
    expect(d.weights.vsBase).toMatchObject({ base: 'google/vit-base-patch16-224', verdict: 'head-only', changed: 2, common: 200, added: 0, dropped: 0, examples: ['classifier.bias', 'classifier.weight'] });
    expect(d.weights.note).toMatch(/never what it does to behaviour/);
    const b = await call({ hf_id: 'meta-llama/Llama-3.2-1B-Instruct' });
    expect(b.weights.vsBase).toBeNull();
    expect(b.weights.derivatives).toEqual([{ id: 'unsloth/Llama-3.2-1B-Instruct', verdict: 'identical', changed: 0, common: 146, sha: '5a8abab4a5d6f164389b1079fb721cfab8d7126c', baseSha: '9213176726f574b556790deb65791e0c5aa438b6' }]);
    const none = await call({ hf_id: 'acme/demo' });
    expect(none.weights).toEqual({ vsBase: null, derivatives: [], note: expect.stringMatching(/Not fingerprinted \(4 tracked derivatives are\)\. No comparison is not the same as identical weights/) });
    clearChangeCache();
    const f = feedFetch();
    vi.stubGlobal('fetch', (async (u: string, i?: RequestInit) => (String(u).endsWith('/changes-feeds/lineage.json') ? new Response('nope', { status: 503 }) : f(u, i))) as typeof fetch);
    const down = await call({ hf_id: 'ISxOdin/vit-base-oxford-iiit-pets' });
    expect(down.weights).toBeNull();
    expect(down.tracked).toBe(false);
  });

  it('weights: a quantization is judged on what it left unquantized, and a cast is never an edit (neurarch #463)', async () => {
    vi.stubGlobal('fetch', feedFetch());
    const fp8 = await call({ hf_id: 'Qwen/Qwen3-8B-FP8' });
    expect(fp8.weights.vsBase.verdict).toBe('requantized');
    expect(fp8.weights.vsBase.unquantized).toEqual({ direction: 'quantized', layersReencoded: 252, tensors: 147, identical: 147, cast: 0, changedSameDtype: 0 });
    const awq = await call({ hf_id: 'Qwen/Qwen2.5-7B-Instruct-AWQ' });
    expect(awq.weights.vsBase.unquantized).toMatchObject({ tensors: 59, identical: 0, cast: 59, changedSameDtype: 0 });
    expect(awq.weights.vsBase.casts).toEqual(['BF16>F16']);
    expect(awq.weights.note).toMatch(/cannot hash equal, so neither is evidence of an edit/);
    const vit = await call({ hf_id: 'ISxOdin/vit-base-oxford-iiit-pets' });
    expect(vit.weights.vsBase.unquantized).toBeNull();
  });

  it('an untracked model is not "no change"', async () => {
    vi.stubGlobal('fetch', feedFetch());
    const r = await call({ hf_id: 'nobody/untracked' });
    expect(r.tracked).toBe(false);
    expect(r.verdict).toMatch(/not in the change record.*No record is not the same as no change/);
  });

  it('refuses a malformed id or a branch name as a pin before any request', async () => {
    const f = feedFetch();
    vi.stubGlobal('fetch', f);
    await expect(call({ hf_id: 'just-a-name' })).rejects.toThrow(/org\/name/);
    await expect(call({ hf_id: 'acme/demo', since_revision: 'main' })).rejects.toThrow(/branch or tag name moves/);
    expect(f.urls).toHaveLength(0);
  });

  it('turns an unreachable feed into a sentence with the browsable URL, and honours NEURARCH_FEEDS', async () => {
    const f = feedFetch({ fail: true });
    vi.stubGlobal('fetch', f);
    process.env.NEURARCH_FEEDS = 'https://mirror.example';
    await expect(call({ hf_id: 'acme/demo' })).rejects.toThrow(/could not reach https:\/\/mirror\.example\/changes-feeds\/advisories\/index\.json.*www\.neurarch\.com\/changes-feeds/);
  });

  it('caches the index between calls', async () => {
    const f = feedFetch();
    vi.stubGlobal('fetch', f);
    await call({ hf_id: 'acme/demo' });
    await call({ hf_id: 'Qwen/Qwen3-8B' });
    expect(f.urls.filter(u => u.endsWith('/advisories/index.json'))).toHaveLength(1);
  });
});
