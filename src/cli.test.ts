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
  it('exposes only read tools by default', () => {
    expect(selectTools(false)).toEqual(TOOLS);
  });
  it('adds write tools with --write', () => {
    const all = selectTools(true);
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
