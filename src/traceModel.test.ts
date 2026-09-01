/**
 * trace_model shells out to Python. The tests use a fake interpreter (a shell
 * script) so they run without torch: one that writes a valid graph, one that
 * says the module is missing, one that hangs.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, writeFile, rm, chmod } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { TOOLS } from './tools.js';
import { makeModel } from './test/fixtures.js';

let dir: string;
const tool = () => TOOLS.find(t => t.name === 'trace_model')!;
async function fakePython(body: string): Promise<string> {
  const p = join(dir, 'python');
  await writeFile(p, `#!/bin/sh\n${body}\n`);
  await chmod(p, 0o755);
  return p;
}
beforeEach(async () => { dir = await mkdtemp(join(tmpdir(), 'nx-trace-')); process.env.NEURARCH_MCP_CACHE = join(dir, 'cache'); });
afterEach(async () => { delete process.env.NEURARCH_PYTHON; delete process.env.NEURARCH_MCP_CACHE; await rm(dir, { recursive: true, force: true }); });

describe('trace_model', () => {
  it('returns the traced graph as a model_path any tool accepts', async () => {
    const graph = JSON.stringify(makeModel());
    // Find the -o argument and write the graph there.
    process.env.NEURARCH_PYTHON = await fakePython(`while [ $# -gt 0 ]; do if [ "$1" = "-o" ]; then out="$2"; fi; shift; done; printf '%s' '${graph.replace(/'/g, "'\\''")}' > "$out"`);
    const r: any = await tool().handler({ target: 'm.py:Net', input: ['1,4'] }, makeModel(), { modelPath: join(dir, 'x.py') });
    expect(r.modelPath).toMatch(/traces\/Net-\d+\.neurarch\.json$/);
    expect(r.layers).toBe(7);
    expect(r.pipeline[0]).toBe('input');
  });
  it('names the install command when neurarch-trace is missing', async () => {
    process.env.NEURARCH_PYTHON = await fakePython(`echo "No module named neurarch_trace" >&2; exit 1`);
    await expect(tool().handler({ target: 'm.py:Net', input: ['1,4'] }, makeModel(), { modelPath: '' })).rejects.toThrow(/pip install neurarch-trace/);
  });
  it('kills a trace that overruns its timeout', async () => {
    process.env.NEURARCH_PYTHON = await fakePython(`exec sleep 30`);
    await expect(tool().handler({ target: 'm.py:Net', input: ['1,4'], timeout_s: 5 }, makeModel(), { modelPath: '' })).rejects.toThrow(/did not finish/);
  }, 15000);
});
