/**
 * The organisation's house rules, read from the `.neurarch.yml` that is
 * already in the repository.
 *
 * A policy is the half of a plan that Neurarch does not own: `max_params`,
 * `must_fit`, `forbid_types` are somebody's own lines, and the plan endpoint
 * turns them into blockers and warnings and quotes them back as written. The
 * CI bot already reads them from `.neurarch.yml`; if the `plan` tool made an
 * agent retype them, the agent's plan and the pull request's plan would
 * disagree on the same graph, which is worse than having no policy at all.
 *
 * ── Why a parser and not a dependency ──────────────────────────────────────
 * `@modelcontextprotocol/sdk` is this package's only runtime dependency and is
 * meant to stay that way. So this reads the subset of YAML that `.neurarch.yml`
 * actually is: mappings, indentation, block and flow sequences of scalars,
 * sequences of mappings, `#` comments, quoted strings. Anchors, block scalars,
 * flow mappings and multi-document files are refused BY NAME rather than
 * mis-read, because a policy that silently parses to something narrower than
 * what is written is a plan that passes a rule the repository meant to fail.
 *
 * For the same reason a file that exists and cannot be read is an error rather
 * than a skip: planning without the house rules and saying "will run" is the
 * confident wrong answer this file exists to prevent.
 */
import { readFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, resolve } from 'node:path';

/** The lines `POST /api/v1/plan` understands. Anything else in the block is ignored, and named. */
export const POLICY_KEYS = [
  'max_params',
  'max_memory_gb',
  'must_fit',
  'forbid_types',
  'require_types',
  'max_layers',
  'max_cost_usd',
] as const;

export type PolicyKey = (typeof POLICY_KEYS)[number];
export type Policy = Partial<Record<PolicyKey, string | number | boolean | string[]>>;

/** How far up from the model to look. A repository root is rarely deeper than this from a model file. */
export const MAX_SEARCH_DEPTH = 8;

const CONFIG_NAMES = ['.neurarch.yml', '.neurarch.yaml'];

export interface PolicyLookup {
  /** Absolute path of the config that was read, or null when none was found. */
  path: string | null;
  /** The merged policy, or null when the file carried none. */
  policy: Policy | null;
  /** The `models:` entry whose `path:` names this model file, when one did. */
  matchedModel: string | null;
  /** Keys inside a `policy:` block that the plan endpoint does not accept. Reported, never sent. */
  ignoredKeys: string[];
  /** One sentence for the tool result: what was read, from where, and what applied. */
  note: string;
}

/**
 * Find and read the policy that applies to one model file.
 *
 * Walks up from the model's directory, because a repository keeps one
 * `.neurarch.yml` at its root and the models under `models/`. The first file
 * found wins; nothing merges across directories, since two configs in one tree
 * is a mistake to surface rather than a feature to support.
 */
export async function readPolicyFor(modelPath: string): Promise<PolicyLookup> {
  const none: PolicyLookup = {
    path: null, policy: null, matchedModel: null, ignoredKeys: [],
    note: 'No .neurarch.yml found near this model, so the plan carries no policy.',
  };
  if (!modelPath) return none;

  const found = await findConfig(dirname(resolve(modelPath)));
  if (!found) return none;

  let text: string;
  try {
    text = await readFile(found, 'utf-8');
  } catch (e) {
    throw new Error(`plan: found ${found} but could not read it: ${(e as Error).message}`);
  }

  let config: Record<string, unknown>;
  try {
    config = parseYamlSubset(text);
  } catch (e) {
    throw new Error(
      `plan: ${found} could not be read (${(e as Error).message}). `
      + 'Fix the file, or pass `policy` on the call to plan without it; planning while ignoring the '
      + 'repository\'s own rules would report a verdict those rules do not agree with.',
    );
  }
  return resolvePolicy(config, found, resolve(modelPath));
}

/** Which config applies, given a parsed one. Split out so the merge is testable without a filesystem. */
export function resolvePolicy(config: Record<string, unknown>, configPath: string, modelPath: string): PolicyLookup {
  const dir = dirname(configPath);
  const merged: Record<string, unknown> = {};
  const top = config.policy;
  if (isMap(top)) Object.assign(merged, top);

  // A model's own `policy:` merges over the top-level block key by key, the
  // same rule the bot applies: an entry that tightens one line must not
  // silently drop the rest of the house rules.
  let matchedModel: string | null = null;
  const models = config.models;
  if (Array.isArray(models)) {
    for (const entry of models) {
      if (!isMap(entry) || typeof entry.path !== 'string') continue;
      const file = entry.path.includes(':') ? entry.path.slice(0, entry.path.lastIndexOf(':')) : entry.path;
      if (resolve(isAbsolute(file) ? file : join(dir, file)) !== modelPath) continue;
      matchedModel = entry.path;
      if (isMap(entry.policy)) Object.assign(merged, entry.policy);
      break;
    }
  }

  const policy: Policy = {};
  const ignoredKeys: string[] = [];
  for (const [k, v] of Object.entries(merged)) {
    if ((POLICY_KEYS as readonly string[]).includes(k) && isPolicyValue(v)) policy[k as PolicyKey] = v;
    else ignoredKeys.push(k);
  }
  const lines = Object.keys(policy);

  return {
    path: configPath,
    policy: lines.length ? policy : null,
    matchedModel,
    ignoredKeys,
    note: lines.length
      ? `Policy read from ${configPath}: ${lines.join(', ')}`
        + (matchedModel ? ` (including the \`${matchedModel}\` entry's own lines).` : '.')
      : `${configPath} was read but declares no policy lines, so the plan carries none.`,
  };
}

function isMap(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** A policy line is a scalar or a list of layer type names. Nothing else is sent. */
function isPolicyValue(v: unknown): v is string | number | boolean | string[] {
  if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') return true;
  return Array.isArray(v) && v.every(x => typeof x === 'string');
}

async function findConfig(startDir: string): Promise<string | null> {
  let dir = startDir;
  for (let depth = 0; depth < MAX_SEARCH_DEPTH; depth++) {
    for (const name of CONFIG_NAMES) {
      const candidate = join(dir, name);
      try {
        await readFile(candidate, 'utf-8');
        return candidate;
      } catch {
        /* not here; keep walking */
      }
    }
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

// ── the parser ───────────────────────────────────────────────────────────────

interface Line { indent: number; text: string; n: number }

const REFUSED: Array<[RegExp, string]> = [
  [/^---\s*$|^\.\.\.\s*$/, 'multi-document files are not supported'],
  [/:\s*[|>][-+0-9]*\s*$/, 'block scalars (| and >) are not supported'],
  [/:\s*[&*]\S/, 'anchors and aliases are not supported'],
  [/:\s*\{/, 'flow mappings ({...}) are not supported'],
];

/**
 * Parse the subset of YAML `.neurarch.yml` is written in.
 *
 * Deliberately small and deliberately loud: everything it does not handle
 * throws with the line number, so an unreadable config is a message rather
 * than a policy that quietly lost half its lines.
 */
export function parseYamlSubset(text: string): Record<string, unknown> {
  const lines = scan(text);
  for (const line of lines) {
    for (const [re, why] of REFUSED) {
      if (re.test(line.text)) throw new Error(`line ${line.n}: ${why}`);
    }
  }
  if (!lines.length) return {};
  const [value, consumed] = parseNode(lines, 0, lines[0].indent);
  if (consumed !== lines.length) throw new Error(`line ${lines[consumed].n}: unexpected indentation`);
  return isMap(value) ? value : {};
}

function scan(text: string): Line[] {
  const out: Line[] = [];
  text.split(/\r?\n/).forEach((raw, i) => {
    if (raw.includes('\t')) throw new Error(`line ${i + 1}: tabs cannot indent YAML; use spaces`);
    const stripped = stripComment(raw);
    if (!stripped.trim()) return;
    out.push({ indent: stripped.length - stripped.trimStart().length, text: stripped.trimEnd().trimStart(), n: i + 1 });
  });
  return out;
}

/** Drop a trailing `#` comment, leaving one inside quotes alone. */
function stripComment(line: string): string {
  let quote: string | null = null;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quote) {
      if (c === quote) quote = null;
      continue;
    }
    if (c === '"' || c === "'") { quote = c; continue; }
    if (c === '#' && (i === 0 || /\s/.test(line[i - 1]))) return line.slice(0, i);
  }
  return line;
}

function parseNode(lines: Line[], start: number, indent: number): [unknown, number] {
  if (start >= lines.length) return [null, start];
  return lines[start].text.startsWith('-') ? parseSeq(lines, start, indent) : parseMap(lines, start, indent);
}

const KEY_RE = /^([A-Za-z_][\w.-]*)\s*:\s*(.*)$/;

function parseMap(lines: Line[], start: number, indent: number): [Record<string, unknown>, number] {
  const map: Record<string, unknown> = {};
  let i = start;
  while (i < lines.length && lines[i].indent === indent && !lines[i].text.startsWith('-')) {
    const line = lines[i];
    const m = KEY_RE.exec(line.text);
    if (!m) throw new Error(`line ${line.n}: expected "key: value"`);
    const [, key, rest] = m;
    if (rest === '') {
      const childIndent = i + 1 < lines.length ? lines[i + 1].indent : -1;
      if (childIndent > indent) {
        const [value, next] = parseNode(lines, i + 1, childIndent);
        map[key] = value;
        i = next;
      } else {
        map[key] = null;
        i++;
      }
    } else {
      map[key] = parseValue(rest, line.n);
      i++;
    }
  }
  return [map, i];
}

function parseSeq(lines: Line[], start: number, indent: number): [unknown[], number] {
  const arr: unknown[] = [];
  let i = start;
  while (i < lines.length && lines[i].indent === indent && lines[i].text.startsWith('-')) {
    const line = lines[i];
    if (line.text !== '-' && !line.text.startsWith('- ')) throw new Error(`line ${line.n}: expected "- value"`);
    const rest = line.text.slice(1).trimStart();
    if (rest === '') {
      const childIndent = i + 1 < lines.length ? lines[i + 1].indent : -1;
      if (childIndent > indent) {
        const [value, next] = parseNode(lines, i + 1, childIndent);
        arr.push(value);
        i = next;
      } else {
        arr.push(null);
        i++;
      }
    } else if (KEY_RE.test(rest)) {
      // `- path: x` opens a mapping whose siblings are indented to the key
      // column, not to the dash. Re-present that first key as a line at the
      // key's own indent and the ordinary mapping parser handles the rest;
      // `virtual[k]` is `lines[i + k]`, so its cursor translates directly.
      const keyIndent = line.indent + (line.text.length - rest.length);
      const virtual: Line[] = [{ indent: keyIndent, text: rest, n: line.n }, ...lines.slice(i + 1)];
      const [value, consumed] = parseMap(virtual, 0, keyIndent);
      arr.push(value);
      i += consumed;
    } else {
      arr.push(parseValue(rest, line.n));
      i++;
    }
  }
  return [arr, i];
}

function parseValue(raw: string, n: number): unknown {
  const s = raw.trim();
  if (s.startsWith('[')) {
    if (!s.endsWith(']')) throw new Error(`line ${n}: a flow sequence must close on the same line`);
    const inner = s.slice(1, -1).trim();
    if (!inner) return [];
    if (inner.includes('[')) throw new Error(`line ${n}: nested flow sequences are not supported`);
    return inner.split(',').map(part => parseScalar(part.trim()));
  }
  return parseScalar(s);
}

function parseScalar(s: string): string | number | boolean | null {
  if (s.length > 1 && ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'")))) {
    return s.slice(1, -1);
  }
  if (s === 'true') return true;
  if (s === 'false') return false;
  if (s === 'null' || s === '~' || s === '') return null;
  // `200M` and `1,3,32,32` stay strings, which is what the plan endpoint wants.
  if (/^-?\d+$/.test(s)) return Number(s);
  if (/^-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?$/.test(s)) return Number(s);
  return s;
}
