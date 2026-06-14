import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { mkdtemp, writeFile, rm, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadModelFile } from './loader.js';

let dir: string;
beforeAll(async () => { dir = await mkdtemp(join(tmpdir(), 'neurarch-mcp-')); });
afterAll(async () => { await rm(dir, { recursive: true, force: true }); });

async function write(name: string, contents: string): Promise<string> {
  const p = join(dir, name);
  await writeFile(p, contents, 'utf-8');
  return p;
}

describe('loadModelFile', () => {
  it('loads a valid model file', async () => {
    const p = await write('ok.json', JSON.stringify({ id: 'a', name: 'n', components: [], connections: [] }));
    const model = await loadModelFile(p);
    expect(model.name).toBe('n');
    expect(model.components).toEqual([]);
  });

  it('rejects a non-existent file', async () => {
    await expect(loadModelFile(join(dir, 'missing.json'))).rejects.toThrow(/Cannot read model file/);
  });

  it('rejects a directory', async () => {
    const sub = join(dir, 'adir');
    await mkdir(sub, { recursive: true });
    await expect(loadModelFile(sub)).rejects.toThrow(/Not a file/);
  });

  it('rejects malformed JSON', async () => {
    const p = await write('bad.json', '{ not json');
    await expect(loadModelFile(p)).rejects.toThrow(/not valid JSON/);
  });

  it('rejects a JSON value that is not an object', async () => {
    const p = await write('arr.json', '[1,2,3]');
    await expect(loadModelFile(p)).rejects.toThrow();
  });

  it('rejects an object missing components/connections', async () => {
    const p = await write('incomplete.json', JSON.stringify({ id: 'a', name: 'n' }));
    await expect(loadModelFile(p)).rejects.toThrow(/missing required fields/);
  });

  it('rejects a component without a string id', async () => {
    const p = await write('corrupt.json', JSON.stringify({
      id: 'a', name: 'n',
      components: [{ type: 'linear', name: 'l' }],
      connections: [],
    }));
    await expect(loadModelFile(p)).rejects.toThrow(/missing a string "id"/);
  });
});
