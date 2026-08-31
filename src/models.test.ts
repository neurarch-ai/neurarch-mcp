/**
 * The cache exists to avoid re-parsing, and it is only allowed to do that while
 * it can prove the file has not moved under it. Both halves are tested here;
 * the routing that uses it is covered end-to-end in server.test.ts.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, writeFile, rm, utimes } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadModelCached, clearModelCache, modelCacheSize } from './models.js';

let dir: string;

async function write(name: string, layerCount: number): Promise<string> {
  const path = join(dir, name);
  const components = Array.from({ length: layerCount }, (_, i) => ({
    id: `l${i}`, type: 'linear', name: `fc${i}`, position: { x: 0, y: i },
    params: { inFeatures: 4, outFeatures: 4 }, inputs: [], outputs: [],
  }));
  await writeFile(path, JSON.stringify({ id: name, name, components, connections: [] }), 'utf-8');
  return path;
}

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'nx-mcp-cache-'));
  clearModelCache();
});
afterEach(async () => { await rm(dir, { recursive: true, force: true }); });

describe('loadModelCached', () => {
  it('returns the identical object while the file is unchanged', async () => {
    const path = await write('a.json', 2);
    const first = await loadModelCached(path);
    const second = await loadModelCached(path);
    expect(second).toBe(first);
  });

  it('re-reads once the file changes', async () => {
    const path = await write('a.json', 2);
    const first = await loadModelCached(path);
    await write('a.json', 5);
    const future = new Date(Date.now() + 2000);
    await utimes(path, future, future);
    const second = await loadModelCached(path);
    expect(second).not.toBe(first);
    expect(second.components).toHaveLength(5);
  });

  it('treats the same file reached by two spellings as one entry', async () => {
    const path = await write('a.json', 2);
    await loadModelCached(path);
    await loadModelCached(join(dir, '.', 'a.json'));
    expect(modelCacheSize()).toBe(1);
  });

  it('stays bounded when pointed at a repository full of models', async () => {
    for (let i = 0; i < 12; i++) await loadModelCached(await write(`m${i}.json`, 1));
    expect(modelCacheSize()).toBeLessThanOrEqual(8);
  });

  it('surfaces the loader\'s own message for a bad path, and caches nothing', async () => {
    await expect(loadModelCached(join(dir, 'missing.json'))).rejects.toThrow(/Cannot read model file/);
    expect(modelCacheSize()).toBe(0);
  });

  it('surfaces the loader\'s own message for a file that is not a model', async () => {
    const path = join(dir, 'notamodel.json');
    await writeFile(path, JSON.stringify({ hello: 'world' }), 'utf-8');
    await expect(loadModelCached(path)).rejects.toThrow(/missing required fields/);
  });
});
