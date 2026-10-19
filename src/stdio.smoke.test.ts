/**
 * The built server over stdio, driven by the SDK's own client: the same path
 * Claude Code, Cursor and the MCP Inspector take. Everything else in the suite
 * uses an in-memory transport; this is the one test that would notice a
 * bundling or startup regression in dist/index.js. CI builds before it tests;
 * locally run `npm run build` first or this skips.
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Client } from '@modelcontextprotocol/client';
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio';
import pkg from '../package.json';

const entry = fileURLToPath(new URL('../dist/index.js', import.meta.url));

describe.skipIf(!existsSync(entry))('stdio smoke (dist/index.js)', () => {
  it('initializes, lists, calls with a validated schema, completes and reads a resource', async () => {
    const transport = new StdioClientTransport({ command: process.execPath, args: [entry, 'zoo:bert-base'], stderr: 'ignore' });
    const c = new Client({ name: 'smoke', version: '0' }, { capabilities: {} });
    await c.connect(transport);
    try {
      expect(c.getServerVersion()).toMatchObject({ name: 'neurarch-mcp', version: pkg.version, title: 'Neurarch' });
      expect(c.getInstructions()).toMatch(/size_training_job/);
      const { tools } = await c.listTools();
      expect(tools.map(t => t.name)).toEqual(expect.arrayContaining(['size_training_job', 'check_pins', 'model_changes', 'lint_model']));
      const lint = await c.callTool({ name: 'lint_model', arguments: {} });
      expect((lint.structuredContent as { counts: unknown }).counts).toBeDefined();
      const pins = await c.callTool({ name: 'check_pins', arguments: { source: 'vllm serve Qwen/Qwen3-8B\n', filename: 'serve.sh' } });
      expect((pins.structuredContent as { findings: unknown[] }).findings).toHaveLength(1);
      const done = await c.complete({ ref: { type: 'ref/resource', uri: 'neurarch://zoo/{id}' }, argument: { name: 'id', value: 'bert' } });
      expect(done.completion.values).toContain('bert-base');
      const rules = await c.readResource({ uri: 'neurarch://rules/catalogue' });
      expect(JSON.parse((rules.contents[0] as { text: string }).text).count).toBeGreaterThan(40);
      const prompts = await c.listPrompts();
      expect(prompts.prompts.map(p => p.name)).toEqual(expect.arrayContaining(['precheck_change', 'precheck_serving', 'pin_downloads']));
    } finally {
      await c.close();
    }
  }, 30_000);
});
