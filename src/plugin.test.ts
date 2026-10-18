/**
 * The Claude Code plugin (.claude-plugin/, hooks/hooks.json) is three JSON
 * files nothing else reads at test time, so a release that bumps package.json
 * and forgets plugin.json, or a hook command that drifts from the one the
 * README documents, would ship silently. `claude plugin validate .` checks the
 * schema; this checks that the files say what this package does.
 */
import { describe, it, expect } from 'vitest';
import { readFile } from 'node:fs/promises';

const json = async (p: string) => JSON.parse(await readFile(new URL(`../${p}`, import.meta.url), 'utf-8'));

describe('Claude Code plugin', () => {
  it('carries the package version, so a release cannot leave the plugin behind', async () => {
    const pkg = await json('package.json');
    const plugin = await json('.claude-plugin/plugin.json');
    expect(plugin.name).toBe('neurarch');
    expect(plugin.version).toBe(pkg.version);
  });

  it('starts the server with no model of its own (--hosted), so every tool takes model_path', async () => {
    const plugin = await json('.claude-plugin/plugin.json');
    const s = plugin.mcpServers.neurarch;
    expect(s.command).toBe('npx');
    expect(s.args.slice(0, 2)).toEqual(['-y', 'neurarch-mcp']);
    expect(s.args).toContain('--hosted');
  });

  it('runs the same hook command the README documents, on Bash only', async () => {
    const hooks = await json('hooks/hooks.json');
    const pre = hooks.hooks.PreToolUse;
    expect(pre).toHaveLength(1);
    expect(pre[0].matcher).toBe('Bash');
    expect(pre[0].hooks[0].command).toBe('npx -y neurarch-mcp hook --hf');
    const readme = await readFile(new URL('../README.md', import.meta.url), 'utf-8');
    expect(readme).toContain('"command": "npx -y neurarch-mcp hook --hf"');
  });

  it('the marketplace lists this repository as the plugin', async () => {
    const m = await json('.claude-plugin/marketplace.json');
    expect(m.name).toBe('neurarch-ai');
    expect(m.plugins).toEqual([expect.objectContaining({ name: 'neurarch', source: './' })]);
  });
});
