/**
 * Pure CLI helpers, split out from index.ts so the argument parsing and the
 * tool-dispatch gating can be unit-tested without spinning up a stdio server.
 * index.ts wires these into the live MCP server and the process lifecycle.
 */
import { TOOLS, type ToolAnnotations, type ToolDef } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';
import { HF_TOOLS, hfToolsIfEnabled } from './extraTools.js';
import { isHfEnabled as hfEnabled } from './sources.js';
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
  /** Which read tools to advertise: the 'core' dozen (default), 'full', or named groups from `--toolsets`. */
  toolSet: ToolSetName;
  /** `--read-only`: no write tools, whatever --write or --toolsets=write asked for. */
  readOnly: boolean;
  /** Names given to `--toolsets` that are not a group; the server refuses to start on any. */
  unknownToolsets: string[];
  /** Serve with no model on stdio too (a container with nothing mounted); every call names its model. */
  hostedRequested: boolean;
  /** A subcommand (`lint`, `check`, `hook`) instead of serving, when the first token is one. */
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

const KNOWN_FLAGS = new Set(['--version', '-v', '--help', '-h', '--write', '--watch', '--http', '--host', '--hf', '--tools', '--toolsets', '--read-only', '--json', '--hosted']);

/**
 * Groups `--toolsets=` can name, the way the GitHub and Sentry servers split
 * theirs: a team that only sizes training jobs loads six tools, not nineteen.
 * A tool may sit in more than one group (trace_model feeds every one of them).
 * `write` is the mutation tools, and asking for it is the same as --write,
 * including every startup refusal --write carries; --read-only beats both.
 * Group membership narrows what is ADVERTISED, like --tools=core: every read
 * tool stays callable by name.
 */
export const TOOL_GROUPS = {
  sizing: ['size_training_job', 'diagnose_oom', 'check_edit', 'trace_model', 'plan', 'history'],
  parallel: ['check_parallel_layout', 'size_training_job', 'trace_model'],
  serving: ['check_serving_config', 'trace_model'],
  changes: ['model_changes', 'check_pins', 'diff_models', 'check_edit', 'plan', 'history'],
  design: [
    'check_design', 'lint_model', 'validate_model', 'suggest_fix', 'rank_designs', 'trace_model',
    'describe_architecture', 'get_model_summary', 'get_layer', 'find_layers', 'layer_impact',
    'param_count_by_block', 'flops_by_block', 'list_blocks', 'get_block', 'compare_layers', 'find_path',
    'list_connections', 'list_hyperparams', 'get_design_notes', 'mermaid_diagram', 'export_pytorch',
  ],
  models: ['list_architectures', 'load_architecture', 'find_models', 'load_hf_model'],
  write: [] as string[],
} as const satisfies Record<string, readonly string[]>;
export type ToolGroup = keyof typeof TOOL_GROUPS;

export type ToolSetName = 'core' | 'full' | { groups: readonly ToolGroup[] };
export type Command = 'lint' | 'check' | 'hook';
const COMMANDS = new Set<Command>(['lint', 'check', 'hook']);

/**
 * The tools that answer the questions an agent actually asks, without the
 * long tail. Twenty-five tool descriptions ride along on every turn of every
 * conversation the server is attached to, and some clients cap the count;
 * `--tools=core` advertises these and keeps the rest callable by name.
 *
 * The ORDER is the listing order (`selectTools` reads this array, not the
 * registry), and it is the order the tools are meant to be reached for. An
 * agent picks by scanning a list, so the list is the recommendation: size the
 * job first, because the failure an agent actually hits before a training run
 * is running out of memory on the card it reserved, not an illegal structure;
 * then trace, because a static parse of a real repository is the weak path
 * (recognisable on 41% of 116 real files, zero true findings hand-judged);
 * then plan, then the local legality ladder.
 *
 * 0.20 added check_pins and model_changes (what changed upstream is half of a
 * pre-check) and moved mermaid_diagram and export_pytorch out to keep the
 * count at 19: both are also served as resources (neurarch://model/mermaid,
 * neurarch://model/pytorch), and both stay callable by name.
 */
export const CORE_TOOLS: readonly string[] = [
  'size_training_job',
  'trace_model',
  'plan',
  'history',
  'diagnose_oom',
  'check_edit',
  'check_parallel_layout',
  'check_serving_config',
  'check_pins',
  'model_changes',
  'check_design',
  'lint_model',
  'describe_architecture',
  'suggest_fix',
  'layer_impact',
  'get_layer',
  'find_layers',
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
  const groupsRaw = valueOf('--toolsets');
  const named = groupsRaw === undefined ? [] : groupsRaw.split(',').map(g => g.trim()).filter(Boolean);
  const groups = named.filter((g): g is ToolGroup => g in TOOL_GROUPS);
  const readOnly = has('--read-only');

  return {
    hfEnabled: has('--hf'),
    hostedRequested: has('--hosted'),
    // --toolsets wins over --tools when both are given: it is the narrower ask.
    toolSet: groups.length > 0 ? { groups } : toolsRaw === 'full' ? 'full' : 'core',
    readOnly,
    unknownToolsets: named.filter(g => !(g in TOOL_GROUPS)),
    command,
    json: has('--json'),
    positional,
    versionRequested: has('--version', '-v'),
    helpRequested: has('--help', '-h'),
    writeEnabled: !readOnly && (has('--write') || groups.includes('write')),
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
  let advertised: ToolDef[];
  if (toolSet === 'core') {
    advertised = CORE_TOOLS.map(name => reads.find(t => t.name === name)).filter((t): t is ToolDef => !!t);
  } else if (toolSet === 'full') {
    advertised = reads;
  } else {
    // Core order first (it is the recommendation), then the rest in registry order.
    const wanted = new Set<string>(toolSet.groups.flatMap(g => TOOL_GROUPS[g]));
    const order = (name: string) => { const i = CORE_TOOLS.indexOf(name); return i === -1 ? CORE_TOOLS.length : i; };
    advertised = reads.filter(t => wanted.has(t.name)).sort((a, b) => order(a.name) - order(b.name));
  }
  return writeEnabled ? [...advertised, ...WRITE_TOOLS] : advertised;
}

/** Argument name for "ask this about a different file". Read tools only. */
export const MODEL_PATH_PARAM = 'model_path';

const MODEL_PATH_PROPERTY = {
  type: 'string',
  // Short on purpose: it rides on every read tool's schema in every ListTools,
  // so each character here is paid ~30 times a turn (check_edit paid for itself here).
  description: 'File, zoo:<id>, hf:<org/name>',
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

/**
 * Under --hf, any read tool can be handed `model_path: "hf:<org/name>"` (or an
 * `hf_id`) and fetch a config from huggingface.co, so its honest annotation is
 * open-world. Without --hf an hf: model is served from the cache or refused,
 * and the same tool is closed-world. Every read tool takes model_path, so the
 * flag flips them all; the function exists so the rule has one name.
 */
function reachesHub(_t: ToolDef): boolean {
  return true;
}

/** The wire form of a tool: what ListTools sends to the client. */
export interface ListedTool {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  annotations: ToolAnnotations;
  outputSchema?: Record<string, unknown>;
  title?: string;
}

/**
 * Output schemas, declared loosely on purpose. A client validates
 * structuredContent against these and fails the call on a mismatch, so each
 * one names the keys an agent should rely on and leaves the rest open (no
 * additionalProperties, whose JSON Schema default is true: spelling it out
 * cost ~100 listing tokens and said nothing the default does not). A drift between handler and schema then
 * costs a missing hint, not a broken tool.
 */
const S = { type: 'string' } as const;
const B = { type: 'boolean' } as const;
const N = { type: 'number' } as const;
const A = { type: 'array' } as const;
const O = { type: 'object' } as const;
const CURSOR = { type: ['string', 'null'] } as const;
/** An object schema whose `required` keys every code path of the handler returns. */
const obj = (props: Record<string, unknown>, required: string[] = []) =>
  required.length
    ? { type: 'object', properties: Object.fromEntries(required.map(k => [k, props[k] ?? {}])), required }
    : { type: 'object', properties: props };

/**
 * `required` lists only keys every code path returns, and
 * outputSchema.test.ts calls each of these tools through a real SDK client,
 * which validates structuredContent against them, on four fixture models.
 */
const OUTPUT_SCHEMAS: Record<string, Record<string, unknown>> = {
  size_training_job: obj({ verdict: S, fits: { type: ['boolean', 'null'] }, accuracy: S }, ['verdict', 'fits']),
  check_edit: obj({ verdict: S, ok: B, findings: A, memory: { type: ['object', 'null'] } }, ['verdict', 'ok', 'findings']),
  check_parallel_layout: obj({ verdict: S, legal: B, predicted: O, findings: A, stages: A, basis: S }, ['verdict', 'legal', 'predicted', 'findings', 'basis']),
  check_serving_config: obj({ verdict: S, blocked: B, findings: A, checked: A, skipped: A, caveats: A, basis: S }, ['verdict', 'blocked', 'findings', 'checked', 'skipped']),
  check_pins: obj({ verdict: S, counts: O, findings: A, nextCursor: CURSOR }, ['verdict', 'findings', 'nextCursor']),
  model_changes: obj({ model: S, tracked: B, verdict: S, changes: A, nextCursor: CURSOR }, ['model', 'tracked', 'verdict', 'changes', 'nextCursor']),
  diagnose_oom: obj({ cause: S, summary: S, fixes: A }, ['cause', 'summary', 'fixes']),
  plan: obj({ text: S, plan: { type: ['object', 'null'] }, policy: O, history: O }),
  history: obj({ fingerprint: S, ledger: S, runs: { type: ['number', 'null'] }, summary: S }),
  check_design: obj({ verdict: S, outcome: S, summary: S, findings: A }, ['verdict', 'findings']),
  lint_model: obj({ clean: B, counts: O, findings: A }, ['clean', 'counts', 'findings']),
  validate_model: obj({ ok: B, findings: A }, ['ok', 'findings']),
  rank_designs: obj({ ranked: A, recommended: { type: ['string', 'null'] }, recommendation: S, calibration: O }, ['ranked', 'recommended']),
  suggest_fix: obj({ fixes: A, notFixable: A }, ['fixes', 'notFixable']),
  describe_architecture: obj({ pipeline: A, layerCount: N, totalParameters: N }, ['pipeline', 'layerCount']),
  get_model_summary: obj({ layerCount: N, totalParameters: N }, ['layerCount', 'totalParameters']),
  layer_impact: O,
  find_layers: obj({ count: N, layers: A, nextCursor: CURSOR }, ['count', 'layers', 'nextCursor']),
  list_connections: obj({ count: N, connections: A, nextCursor: CURSOR }, ['count', 'connections', 'nextCursor']),
  export_pytorch: obj({ code: S, language: S }, ['code']),
  list_architectures: obj({ architectures: A, nextCursor: CURSOR }, ['architectures', 'nextCursor']),
  find_models: obj({ models: A }),
};

/** True when `name` declares an outputSchema, so every success must carry structuredContent. */
export const declaresOutputSchema = (name: string): boolean => name in OUTPUT_SCHEMAS;

/**
 * Build the ListTools payload: schemas with `model_path` where it applies, and
 * annotations resolved from the per-tool overrides over the read defaults.
 */
export function listedTools(writeEnabled: boolean, toolSet: ToolSetName = 'core'): ListedTool[] {
  const reads = selectTools(false, toolSet).map(t => ({
    name: t.name,
    description: shortDescription(t.name, t.description),
    title: TITLES[t.name] ?? t.name,
    inputSchema: withModelPath(t.inputSchema),
    annotations: { ...READ_TOOL_DEFAULTS, ...(hfEnabled() && reachesHub(t) ? { openWorldHint: true } : {}), ...(t.annotations ?? {}) },
    ...(OUTPUT_SCHEMAS[t.name] ? { outputSchema: OUTPUT_SCHEMAS[t.name] } : {}),
  }));
  if (!writeEnabled) return reads;
  const writes = WRITE_TOOLS.map(t => ({
    name: t.name,
    description: shortDescription(t.name, t.description),
    title: TITLES[t.name] ?? t.name,
    inputSchema: t.inputSchema,
    // Write tools always declare their own; the fallback exists so a new one
    // added without annotations is still never advertised as read-only.
    annotations: { readOnlyHint: false, destructiveHint: true, ...(t.annotations ?? {}) },
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
  if (hint) return { errorText: `Unknown tool: ${name}.${hint}` };
  // A near miss (lint vs lint_model, check_serving vs check_serving_config)
  // gets the names it was probably reaching for, so the retry is one call.
  const all = [...selectTools(true, 'full'), ...HF_TOOLS].map(t => t.name);
  const q = name.toLowerCase();
  const near = all.filter(n => n.includes(q) || q.includes(n) || n.split('_').some(w => w.length > 3 && q.includes(w))).slice(0, 5);
  return { errorText: `Unknown tool: ${name}.${near.length ? ` Did you mean: ${near.join(', ')}?` : ''} neurarch://docs lists every tool.` };
}
