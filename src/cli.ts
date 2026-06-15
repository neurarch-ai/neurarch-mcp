/**
 * Pure CLI helpers, split out from index.ts so the argument parsing and the
 * tool-dispatch gating can be unit-tested without spinning up a stdio server.
 * index.ts wires these into the live MCP server and the process lifecycle.
 */
import { TOOLS, type ToolDef } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';

export interface ParsedFlags {
  versionRequested: boolean;
  helpRequested: boolean;
  writeEnabled: boolean;
  watchEnabled: boolean;
  /** First non-flag token — the model file path, or undefined if none given. */
  modelArg?: string;
  /** Flags we did not recognise (e.g. "--frobnicate"). */
  unknownFlags: string[];
}

const KNOWN_FLAGS = new Set(['--version', '-v', '--help', '-h', '--write', '--watch']);

/** Parse argv (already sliced past `node script`). Pure — does not mutate input. */
export function parseFlags(argv: string[]): ParsedFlags {
  const flags = argv.filter(a => a.startsWith('-'));
  const has = (...names: string[]) => names.some(n => flags.includes(n));
  return {
    versionRequested: has('--version', '-v'),
    helpRequested: has('--help', '-h'),
    writeEnabled: has('--write'),
    watchEnabled: has('--watch'),
    modelArg: argv.find(a => !a.startsWith('-')),
    unknownFlags: flags.filter(f => !KNOWN_FLAGS.has(f)),
  };
}

/** The tools exposed for a given mode: read-only by default, +writes with --write. */
export function selectTools(writeEnabled: boolean): ToolDef[] {
  return writeEnabled ? [...TOOLS, ...WRITE_TOOLS] : TOOLS;
}

export interface ToolResolution {
  tool?: ToolDef;
  /** Set when no tool matched; carries a user-facing explanation. */
  errorText?: string;
}

/**
 * Resolve a tool call by name against the active tool set. When the name is a
 * known write tool but writes are disabled, the error explains how to enable it
 * rather than just saying "unknown tool".
 */
export function resolveToolCall(name: string, writeEnabled: boolean): ToolResolution {
  const tool = selectTools(writeEnabled).find(t => t.name === name);
  if (tool) return { tool };
  const isGatedWriteTool = !writeEnabled && WRITE_TOOLS.some(t => t.name === name);
  const hint = isGatedWriteTool
    ? ' Restart the MCP server with --write to enable mutation tools.'
    : '';
  return { errorText: `Unknown tool: ${name}.${hint}` };
}
