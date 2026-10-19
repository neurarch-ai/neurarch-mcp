import { describe, it, expect } from 'vitest';
import { parseFlags, selectTools, resolveToolCall } from './cli.js';
import { TOOLS } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';

describe('parseFlags', () => {
  it('picks out the model path and recognises flags in any order', () => {
    const f = parseFlags(['--write', './model.json', '--watch']);
    expect(f.modelArg).toBe('./model.json');
    expect(f.writeEnabled).toBe(true);
    expect(f.watchEnabled).toBe(true);
    expect(f.versionRequested).toBe(false);
    expect(f.helpRequested).toBe(false);
  });

  it('detects version and help aliases', () => {
    expect(parseFlags(['-v']).versionRequested).toBe(true);
    expect(parseFlags(['--version']).versionRequested).toBe(true);
    expect(parseFlags(['-h']).helpRequested).toBe(true);
    expect(parseFlags(['--help']).helpRequested).toBe(true);
  });

  it('reports no model path when only flags are present', () => {
    expect(parseFlags(['--write']).modelArg).toBeUndefined();
    expect(parseFlags([]).modelArg).toBeUndefined();
  });

  it('collects unknown flags', () => {
    expect(parseFlags(['--frobnicate', 'm.json']).unknownFlags).toEqual(['--frobnicate']);
    expect(parseFlags(['--write', 'm.json']).unknownFlags).toEqual([]);
  });

  it('does not mutate the input array', () => {
    const argv = ['--write', 'm.json'];
    parseFlags(argv);
    expect(argv).toEqual(['--write', 'm.json']);
  });

  it('parses --http as a bare flag (no port) without stealing the model path', () => {
    const f = parseFlags(['m.json', '--http']);
    expect(f.httpEnabled).toBe(true);
    expect(f.httpPort).toBeUndefined();
    expect(f.modelArg).toBe('m.json');
    expect(f.unknownFlags).toEqual([]);
  });

  it('parses --http=PORT and --host=ADDR key/value flags', () => {
    const f = parseFlags(['m.json', '--http=9000', '--host=0.0.0.0']);
    expect(f.httpEnabled).toBe(true);
    expect(f.httpPort).toBe(9000);
    expect(f.httpHost).toBe('0.0.0.0');
    expect(f.unknownFlags).toEqual([]);
  });

  it('rejects an out-of-range or non-numeric http port (falls back to default)', () => {
    expect(parseFlags(['m.json', '--http=0']).httpPort).toBeUndefined();
    expect(parseFlags(['m.json', '--http=70000']).httpPort).toBeUndefined();
    expect(parseFlags(['m.json', '--http=abc']).httpPort).toBeUndefined();
  });

  it('defaults http off and leaves host/port unset', () => {
    const f = parseFlags(['m.json']);
    expect(f.httpEnabled).toBe(false);
    expect(f.httpPort).toBeUndefined();
    expect(f.httpHost).toBeUndefined();
  });
});

describe('selectTools', () => {
  it('exposes the core read tools by default, all of them with --tools=full', () => {
    expect(selectTools(false).every(t => TOOLS.includes(t))).toBe(true);
    expect(selectTools(false).length).toBeLessThan(TOOLS.length);
    expect(selectTools(false, 'full')).toEqual(TOOLS);
  });
  it('adds write tools with --write', () => {
    const all = selectTools(true, 'full');
    expect(all).toHaveLength(TOOLS.length + WRITE_TOOLS.length);
    expect(all.some(t => t.name === 'add_layer')).toBe(true);
  });
});

describe('resolveToolCall', () => {
  it('resolves a read tool regardless of write mode', () => {
    expect(resolveToolCall('get_layer', false).tool?.name).toBe('get_layer');
  });

  it('explains how to enable a gated write tool', () => {
    const r = resolveToolCall('add_layer', false);
    expect(r.tool).toBeUndefined();
    expect(r.errorText).toMatch(/--write/);
  });

  it('resolves a write tool when writes are enabled', () => {
    expect(resolveToolCall('add_layer', true).tool?.name).toBe('add_layer');
  });

  it('reports a plain unknown-tool error for a bogus name', () => {
    const r = resolveToolCall('not_a_tool', true);
    expect(r.tool).toBeUndefined();
    expect(r.errorText).toMatch(/Unknown tool/);
    expect(r.errorText).not.toMatch(/--write/);
  });
});

describe('0.13 flags and subcommands', () => {
  it('parses --hf, --tools=core and --json', () => {
    const f = parseFlags(['m.py', '--hf', '--tools=core', '--json']);
    expect(f.hfEnabled).toBe(true);
    expect(f.toolSet).toBe('core');
    expect(f.json).toBe(true);
    expect(f.unknownFlags).toEqual([]);
    expect(parseFlags(['m.py']).toolSet).toBe('core');
    expect(parseFlags(['m.py', '--tools=full']).toolSet).toBe('full');
    expect(parseFlags(['m.py', '--tools=bogus']).toolSet).toBe('core');
  });
  it('parses --hosted', () => {
    expect(parseFlags(['--hosted']).hostedRequested).toBe(true);
    expect(parseFlags(['--hosted']).unknownFlags).toEqual([]);
  });
  it('recognises lint and check as commands and keeps their files positional', () => {
    const f = parseFlags(['lint', 'a.py', 'zoo:bert-base', '--json']);
    expect(f.command).toBe('lint');
    expect(f.positional).toEqual(['a.py', 'zoo:bert-base']);
    expect(f.modelArg).toBe('a.py');
    expect(parseFlags(['check', 'a.py']).command).toBe('check');
    expect(parseFlags(['a.py']).command).toBeUndefined();
  });
  it('--tools=core narrows what is advertised, never what resolves', () => {
    const core = selectTools(false, 'core').map(t => t.name);
    expect(core.length).toBeLessThan(TOOLS.length);
    expect(core).toContain('check_design');
    expect(resolveToolCall('list_connections', false).tool?.name).toBe('list_connections');
  });
  it('points at --hf when load_hf_model is called without it', () => {
    expect(resolveToolCall('load_hf_model', false).errorText).toMatch(/--hf/);
  });
});

describe('0.21 --toolsets and --read-only', () => {
  it('advertises only the named groups, core order first, and leaves the default alone', () => {
    const f = parseFlags(['m.json', '--toolsets=sizing,serving']);
    expect(f.toolSet).toEqual({ groups: ['sizing', 'serving'] });
    const names = selectTools(false, f.toolSet).map(t => t.name);
    expect(names[0]).toBe('size_training_job');
    expect(names).toContain('check_serving_config');
    expect(names).not.toContain('check_design');
    expect(new Set(names).size).toBe(names.length);
    expect(parseFlags(['m.json']).toolSet).toBe('core');
    expect(parseFlags(['m.json', '--toolsets=sizing']).unknownFlags).toEqual([]);
  });
  it('names every group member as a real tool', async () => {
    const { TOOL_GROUPS } = await import('./cli.js');
    const { HF_TOOLS } = await import('./extraTools.js');
    const all = new Set([...TOOLS, ...HF_TOOLS].map(t => t.name));
    for (const [g, names] of Object.entries(TOOL_GROUPS)) for (const n of names) expect(all.has(n), `${g}: ${n}`).toBe(true);
  });
  it('treats write as --write, and --read-only beats both', () => {
    expect(parseFlags(['m.json', '--toolsets=design,write']).writeEnabled).toBe(true);
    expect(parseFlags(['m.json', '--toolsets=design,write', '--read-only']).writeEnabled).toBe(false);
    expect(parseFlags(['m.json', '--write', '--read-only']).writeEnabled).toBe(false);
    const f = parseFlags(['m.json', '--write', '--read-only', '--toolsets=write']);
    expect(selectTools(f.writeEnabled, f.toolSet).some(t => WRITE_TOOLS.includes(t))).toBe(false);
  });
  it('reports a group name it does not know', () => {
    const f = parseFlags(['m.json', '--toolsets=sizing,bogus']);
    expect(f.unknownToolsets).toEqual(['bogus']);
    expect(f.toolSet).toEqual({ groups: ['sizing'] });
  });
});
