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
