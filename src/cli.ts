/**
 * Pure CLI helpers, split out from index.ts so the argument parsing and the
 * tool-dispatch gating can be unit-tested without spinning up a stdio server.
 * index.ts wires these into the live MCP server and the process lifecycle.
 */
import { TOOLS, type ToolAnnotations, type ToolDef } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';
import { HF_TOOLS, hfToolsIfEnabled } from './extraTools.js';
import { shortDescription, TITLES } from './toolDocs.js';

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
  /** Allow hf:<org/name> model refs and list load_hf_model: the one network switch. */
  hfEnabled: boolean;
  /** Which read tools to advertise: the 'core' dozen (default) or 'full'. */
  toolSet: ToolSetName;
  /** Serve with no model on stdio too (a container with nothing mounted); every call names its model. */
  hostedRequested: boolean;
  /** A subcommand (`lint`, `check`) instead of serving, when the first token is one. */
  command?: Command;
  /** Emit JSON from a subcommand instead of a human report. */
  json: boolean;
  /** First non-flag token — the model file path, or undefined if none given. */
  modelArg?: string;
  /** Every non-flag token after the command, for subcommands that take several files. */
  positional: string[];
  /** Flags we did not recognise (e.g. "--frobnicate"). */
  unknownFlags: string[];
}

const KNOWN_FLAGS = new Set(['--version', '-v', '--help', '-h', '--write', '--watch', '--http', '--host', '--hf', '--tools', '--json', '--hosted']);

export type ToolSetName = 'core' | 'full';
export type Command = 'lint' | 'check';
const COMMANDS = new Set<Command>(['lint', 'check']);

/**
 * The tools that answer the questions an agent actually asks, without the
 * long tail. Twenty-five tool descriptions ride along on every turn of every
 * conversation the server is attached to, and some clients cap the count;
 * `--tools=core` advertises these and keeps the rest callable by name.
 *
 * The ORDER is the listing order (`selectTools` reads this array, not the
 * registry), and it is the order the tools are meant to be reached for. An
 * agent picks by scanning a list, so the list is the recommendation: trace
 * first because a static parse of a real repository is the weak path
 * (recognisable on 41% of 116 real files, zero true findings hand-judged),
 * then plan, then the local ladder.
 */
export const CORE_TOOLS: readonly string[] = [
  'trace_model',
  'plan',
  'history',
  'check_design',
  'lint_model',
  'describe_architecture',
  'rank_designs',
  'suggest_fix',
  'layer_impact',
  'get_layer',
  'find_layers',
  'mermaid_diagram',
  'export_pytorch',
  'list_architectures',
  'find_models',
];

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
  const positionalAll = argv.filter(a => !a.startsWith('-'));
  const command = COMMANDS.has(positionalAll[0] as Command) ? (positionalAll[0] as Command) : undefined;
  const positional = command ? positionalAll.slice(1) : positionalAll;
  const toolsRaw = valueOf('--tools');

  return {
    hfEnabled: has('--hf'),
    hostedRequested: has('--hosted'),
    toolSet: toolsRaw === 'full' ? 'full' : 'core',
    command,
    json: has('--json'),
    positional,
    versionRequested: has('--version', '-v'),
    helpRequested: has('--help', '-h'),
    writeEnabled: has('--write'),
    watchEnabled: has('--watch'),
    httpEnabled: has('--http'),
    httpPort: port !== undefined && Number.isInteger(port) && port > 0 && port < 65536 ? port : undefined,
    httpHost: valueOf('--host') || undefined,
    modelArg: positional[0],
    unknownFlags: flags.filter(f => !KNOWN_FLAGS.has(flagName(f))),
  };
}

/**
 * The tools exposed for a given mode: read-only by default, +writes with
 * --write, +load_hf_model with --hf. `toolSet` narrows what is ADVERTISED, not
 * what is callable: resolveToolCall still finds a full-set tool by name, so an
 * agent that read the docs can use one the listing left out.
 */
export function selectTools(writeEnabled: boolean, toolSet: ToolSetName = 'core'): ToolDef[] {
  const reads = [...TOOLS, ...hfToolsIfEnabled()];
  // Ordered by CORE_TOOLS rather than by the registry: the listing order is
  // what an agent scans, so it carries the recommendation. A name in
  // CORE_TOOLS with no tool behind it is skipped rather than crashing the
  // listing, since the two lists are maintained by hand.
  const advertised = toolSet === 'core'
    ? CORE_TOOLS.map(name => reads.find(t => t.name === name)).filter((t): t is ToolDef => !!t)
    : reads;
  return writeEnabled ? [...advertised, ...WRITE_TOOLS] : advertised;
}

/** Argument name for "ask this about a different file". Read tools only. */
export const MODEL_PATH_PARAM = 'model_path';

const MODEL_PATH_PROPERTY = {
  type: 'string',
  description: 'Optional: answer about another model instead (a .py/.neurarch.json path, zoo:<id>, or hf:<org/name>).',
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
  outputSchema?: Record<string, unknown>;
}

/**
 * Output schemas, declared loosely on purpose. A client validates
 * structuredContent against these and fails the call on a mismatch, so each
 * one names the keys an agent should rely on and leaves the rest open
 * (additionalProperties: true). A drift between handler and schema then
 * costs a missing hint, not a broken tool.
 */
const OUTPUT_SCHEMAS: Record<string, Record<string, unknown>> = {
  plan: { type: 'object', additionalProperties: true, properties: { text: { type: 'string' }, plan: { type: ['object', 'null'] }, policy: { type: 'object' }, history: { type: 'object' } } },
  history: { type: 'object', additionalProperties: true, properties: { fingerprint: { type: 'string' }, ledger: { type: 'string' }, runs: { type: ['number', 'null'] }, summary: { type: 'string' } } },
  check_design: { type: 'object', additionalProperties: true, properties: { verdict: { type: 'string' }, outcome: { type: 'string' }, summary: { type: 'string' }, findings: { type: 'array' } } },
  lint_model: { type: 'object', additionalProperties: true, properties: { clean: { type: 'boolean' }, counts: { type: 'object' }, findings: { type: 'array' } } },
  validate_model: { type: 'object', additionalProperties: true, properties: { ok: { type: 'boolean' } } },
  rank_designs: { type: 'object', additionalProperties: true, properties: { ranked: { type: 'array' }, recommended: { type: ['string', 'null'] }, recommendation: { type: 'string' }, calibration: { type: 'object' } } },
  suggest_fix: { type: 'object', additionalProperties: true, properties: { fixes: { type: 'array' }, notFixable: { type: 'array' } } },
  describe_architecture: { type: 'object', additionalProperties: true, properties: { pipeline: { type: 'array' }, layerCount: { type: 'number' } } },
  layer_impact: { type: 'object', additionalProperties: true },
  export_pytorch: { type: 'object', additionalProperties: true, properties: { code: { type: 'string' }, language: { type: 'string' } } },
  list_architectures: { type: 'object', additionalProperties: true, properties: { architectures: { type: 'array' } } },
  find_models: { type: 'object', additionalProperties: true, properties: { models: { type: 'array' } } },
};

/**
 * Build the ListTools payload: schemas with `model_path` where it applies, and
 * annotations resolved from the per-tool overrides over the read defaults.
 */
export function listedTools(writeEnabled: boolean, toolSet: ToolSetName = 'core'): ListedTool[] {
  const reads = selectTools(false, toolSet).map(t => ({
    name: t.name,
    description: shortDescription(t.name, t.description),
    inputSchema: withModelPath(t.inputSchema),
    annotations: { title: TITLES[t.name] ?? t.name, ...READ_TOOL_DEFAULTS, ...(t.annotations ?? {}) },
    ...(OUTPUT_SCHEMAS[t.name] ? { outputSchema: OUTPUT_SCHEMAS[t.name] } : {}),
  }));
  if (!writeEnabled) return reads;
  const writes = WRITE_TOOLS.map(t => ({
    name: t.name,
    description: shortDescription(t.name, t.description),
    inputSchema: t.inputSchema,
    // Write tools always declare their own; the fallback exists so a new one
    // added without annotations is still never advertised as read-only.
    annotations: { title: TITLES[t.name] ?? t.name, readOnlyHint: false, destructiveHint: true, ...(t.annotations ?? {}) },
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
  const tool = selectTools(writeEnabled, 'full').find(t => t.name === name);
  if (tool) return { tool };
  const isGatedWriteTool = !writeEnabled && WRITE_TOOLS.some(t => t.name === name);
  const isGatedHfTool = HF_TOOLS.some(t => t.name === name);
  const hint = isGatedWriteTool
    ? ' Restart the MCP server with --write to enable mutation tools.'
    : isGatedHfTool
      ? ' Restart the MCP server with --hf to allow the Hugging Face lookup (the one network call this server can make).'
      : '';
  return { errorText: `Unknown tool: ${name}.${hint}` };
}
