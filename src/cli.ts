/**
 * Pure CLI helpers, split out from index.ts so the argument parsing and the
 * tool-dispatch gating can be unit-tested without spinning up a stdio server.
 * index.ts wires these into the live MCP server and the process lifecycle.
 */
import { TOOLS, type ToolAnnotations, type ToolDef } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';

export interface ParsedFlags {
  versionRequested: boolean;
  helpRequested: boolean;
  writeEnabled: boolean;
  watchEnabled: boolean;
  /** Serve over Streamable HTTP instead of stdio (`--http` / `--http=PORT`). */
  httpEnabled: boolean;
  /** Port from `--http=PORT`; undefined falls back to the default. */
  httpPort?: number;
  /** Bind address from `--host=ADDR`; undefined falls back to loopback. */
  httpHost?: string;
  /** First non-flag token — the model file path, or undefined if none given. */
  modelArg?: string;
  /** Flags we did not recognise (e.g. "--frobnicate"). */
  unknownFlags: string[];
}

const KNOWN_FLAGS = new Set(['--version', '-v', '--help', '-h', '--write', '--watch', '--http', '--host']);

/** Base name of a flag token, dropping any `=value` (so `--http=8787` → `--http`). */
function flagName(token: string): string {
  const eq = token.indexOf('=');
  return eq === -1 ? token : token.slice(0, eq);
}

/** Parse argv (already sliced past `node script`). Pure — does not mutate input. */
export function parseFlags(argv: string[]): ParsedFlags {
  const flags = argv.filter(a => a.startsWith('-'));
  const has = (...names: string[]) => flags.some(f => names.includes(flagName(f)));
  const valueOf = (name: string): string | undefined => {
    const hit = flags.find(f => flagName(f) === name && f.includes('='));
    return hit ? hit.slice(hit.indexOf('=') + 1) : undefined;
  };

  const portRaw = valueOf('--http');
  const port = portRaw !== undefined ? Number(portRaw) : undefined;

  return {
    versionRequested: has('--version', '-v'),
    helpRequested: has('--help', '-h'),
    writeEnabled: has('--write'),
    watchEnabled: has('--watch'),
    httpEnabled: has('--http'),
    httpPort: port !== undefined && Number.isInteger(port) && port > 0 && port < 65536 ? port : undefined,
    httpHost: valueOf('--host') || undefined,
    modelArg: argv.find(a => !a.startsWith('-')),
    unknownFlags: flags.filter(f => !KNOWN_FLAGS.has(flagName(f))),
  };
}

/** The tools exposed for a given mode: read-only by default, +writes with --write. */
export function selectTools(writeEnabled: boolean): ToolDef[] {
  return writeEnabled ? [...TOOLS, ...WRITE_TOOLS] : TOOLS;
}

/** Argument name for "ask this about a different file". Read tools only. */
export const MODEL_PATH_PARAM = 'model_path';

const MODEL_PATH_PROPERTY = {
  type: 'string',
  description:
    'Optional. Answer this about a different model file instead of the one this server was '
    + 'started with, so one server can cover a whole repository of architectures. Absolute, or '
    + 'relative to the server working directory. Read tools only: mutations always target the '
    + 'file passed on the command line, so an agent cannot write to a path it just made up.',
};

/**
 * Add `model_path` to a read tool's schema.
 *
 * Done here rather than in each of the two dozen tool definitions for the
 * obvious reason (one place to change) and one less obvious one: a tool that
 * already has a `model_path` of its own must keep it, and a single guard
 * enforces that everywhere. `diff_models` and `save_model` take a `path`, which
 * means something different in both cases, which is exactly why the injected
 * argument is not called `path`.
 */
export function withModelPath(schema: Record<string, unknown>): Record<string, unknown> {
  const props = (schema.properties ?? {}) as Record<string, unknown>;
  if (MODEL_PATH_PARAM in props) return schema;
  return { ...schema, properties: { ...props, [MODEL_PATH_PARAM]: MODEL_PATH_PROPERTY } };
}

/**
 * What a read tool is, unless it says otherwise: it looks, it does not touch,
 * it gives the same answer twice, and it does not reach off the machine.
 * `check_design` is the single tool that overrides part of this.
 */
const READ_TOOL_DEFAULTS: ToolAnnotations = {
  readOnlyHint: true,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: false,
};

/** The wire form of a tool: what ListTools sends to the client. */
export interface ListedTool {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  annotations: ToolAnnotations;
}

/**
 * Build the ListTools payload: schemas with `model_path` where it applies, and
 * annotations resolved from the per-tool overrides over the read defaults.
 */
export function listedTools(writeEnabled: boolean): ListedTool[] {
  const reads = TOOLS.map(t => ({
    name: t.name,
    description: t.description,
    inputSchema: withModelPath(t.inputSchema),
    annotations: { ...READ_TOOL_DEFAULTS, ...(t.annotations ?? {}) },
  }));
  if (!writeEnabled) return reads;
  const writes = WRITE_TOOLS.map(t => ({
    name: t.name,
    description: t.description,
    inputSchema: t.inputSchema,
    // Write tools always declare their own; the fallback exists so a new one
    // added without annotations is still never advertised as read-only.
    annotations: t.annotations ?? { readOnlyHint: false, destructiveHint: true },
  }));
  return [...reads, ...writes];
}

/** True when the name belongs to a mutation tool, whether or not writes are on. */
export function isWriteTool(name: string): boolean {
  return WRITE_TOOLS.some(t => t.name === name);
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
