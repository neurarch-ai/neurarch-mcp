/**
 * model_changes: what changed in a tracked open-weight model on the Hugging
 * Face Hub, revision by revision, from the public change record.
 *
 * The record is neurarch.com/changes-feeds/ (neurarch-ai/neurarch #341): one
 * OSV-format advisory per change event to a tracked model's config, tokenizer,
 * chat template, generation defaults or weights, with the keys that changed,
 * whether it touches memory, serving or the tokenizer, and whether it is
 * breaking under a written rule. The same files feed the Renovate preset and
 * the shields.io badges. This tool reads two of them and filters; it computes
 * nothing about the model itself.
 *
 * ── What leaves the machine ────────────────────────────────────────────────
 * Two GETs of public static files: the advisory index, and the one model's
 * revision list (whose path is the model id). No key, no graph, no repository
 * content. That is why this is the third tool, after plan and history, marked
 * open-world: it reaches the network, and the description says so first.
 *
 * ── What it is not ─────────────────────────────────────────────────────────
 * A change record says what a revision does mechanically. It is not a report
 * that anything broke for anyone, and a model that is not tracked has no
 * record, which is not the same as having no changes. Both sentences travel
 * in the result.
 */
import type { ToolDef, ToolContext } from './tools.js';
import { pageOf, PAGE_PARAMS } from './paging.js';

export const DEFAULT_FEEDS = 'https://www.neurarch.com';
export const FEEDS_PATH = '/changes-feeds';
const INDEX_TTL_MS = 60 * 60 * 1000;
const TIMEOUT_MS = 20_000;

/** `NEURARCH_FEEDS` overrides the host (tests, mirrors); read at call time. */
export function feedsBase(): string {
  return (process.env.NEURARCH_FEEDS || DEFAULT_FEEDS).replace(/\/+$/, '') + FEEDS_PATH;
}

interface IndexRow {
  id: string;
  model: string;
  from: string;
  to: string;
  date: string;
  kind: string;
  breaking: boolean;
  impact: { memory: boolean; serving: boolean; tokenizer: boolean; licence: boolean | null };
  url: string;
  withdrawn?: boolean;
}
interface AdvisoryIndex { count: number; license?: string; breaking_rule?: string; advisories: IndexRow[] }
interface Releases { changelogUrl?: string; releases: Array<{ version: string; digest: string; releaseTimestamp: string }> }

interface LineageFile { checked?: number; findings: Array<{ id: string; base: string; kinds: string[]; basis: string[] }> }

let indexCache: { at: number; base: string; data: AdvisoryIndex } | null = null;
let lineageCache: { at: number; url: string; data: LineageFile | null } | null = null;

/** For tests: forget the cached index. */
export function clearChangeCache(): void { indexCache = null; lineageCache = null; }

/** `/changes/lineage.json` on the same host as the feeds. */
export function lineageUrl(): string {
  return (process.env.NEURARCH_FEEDS || DEFAULT_FEEDS).replace(/\/+$/, '') + '/changes/lineage.json';
}

/**
 * Tracked models whose card disagrees with the base it declares. Supplementary:
 * a missing or unreachable file leaves `lineage` null in the result rather than
 * failing the call, because the change list is the answer and this is context.
 */
async function lineageFile(ctx: ToolContext): Promise<LineageFile | null> {
  const url = lineageUrl();
  if (lineageCache && lineageCache.url === url && Date.now() - lineageCache.at < INDEX_TTL_MS) return lineageCache.data;
  let data: LineageFile | null = null;
  try {
    data = await getJson<LineageFile>(url, ctx, true);
    if (data && !Array.isArray(data.findings)) data = null;
  } catch (e) {
    if (ctx.signal?.aborted) throw e;
    data = null;
  }
  lineageCache = { at: Date.now(), url, data };
  return data;
}

async function getJson<T>(url: string, ctx: ToolContext, allow404 = false): Promise<T | null> {
  const signals = [AbortSignal.timeout(TIMEOUT_MS), ...(ctx.signal ? [ctx.signal] : [])];
  let res: Response;
  try {
    res = await fetch(url, { signal: AbortSignal.any(signals), headers: { accept: 'application/json' } });
  } catch (e) {
    if (ctx.signal?.aborted) throw new Error('model_changes: cancelled.');
    throw new Error(`model_changes: could not reach ${url} (${(e as Error).message}). The same record is browsable at ${DEFAULT_FEEDS}${FEEDS_PATH}/.`);
  }
  if (allow404 && res.status === 404) return null;
  if (!res.ok) throw new Error(`model_changes: ${url} answered HTTP ${res.status}. Try again later; the record is browsable at ${DEFAULT_FEEDS}${FEEDS_PATH}/.`);
  return (await res.json()) as T;
}

async function advisoryIndex(ctx: ToolContext): Promise<AdvisoryIndex> {
  const base = feedsBase();
  if (indexCache && indexCache.base === base && Date.now() - indexCache.at < INDEX_TTL_MS) return indexCache.data;
  const data = await getJson<AdvisoryIndex>(`${base}/advisories/index.json`, ctx);
  if (!data || !Array.isArray(data.advisories)) throw new Error('model_changes: the advisory index had no advisories array; the feed format may have changed. Upgrade neurarch-mcp.');
  indexCache = { at: Date.now(), base, data };
  return data;
}

const HF_ID = /^[A-Za-z0-9][\w.-]*\/[\w.-]+$/;

interface ChangeArgs {
  hf_id: string;
  since_revision?: string;
  breaking_only?: boolean;
  detail?: 'concise' | 'full';
  limit?: number;
  cursor?: string;
}

export const modelChangesTool: ToolDef = {
  name: 'model_changes',
  description:
    'Reaches the network: GETs three public files from www.neurarch.com (no key, nothing about your code is sent). '
    + 'What changed in a tracked open-weight model on the Hugging Face Hub, revision by revision: config, tokenizer, '
    + 'chat template, generation defaults or weights, with the keys that changed, whether the change touches memory, '
    + 'serving or the tokenizer, and whether it is breaking under the published rule. Call it before moving a pinned '
    + 'revision, before redeploying an unpinned model, or when a job that used to work behaves differently. Pass '
    + 'since_revision (a commit, 7+ hex characters) to list only changes after your pin. detail "full" adds each '
    + 'change\'s keys before and after (one extra GET per change on the page). A change record says what a revision '
    + 'does mechanically, not that anything broke; an untracked model has no record, which is not the same as no '
    + 'change. lineage lists where the model\'s card disagrees with the base it declares (an edit declared as a '
    + 'quantization, a licence outside the base\'s family): statements about the card and tags, not the weights. '
    + 'Not for scanning your code for unpinned downloads: use check_pins.',
  modelOptional: true,
  annotations: { openWorldHint: true, idempotentHint: true },
  inputSchema: {
    type: 'object',
    properties: {
      hf_id: { type: 'string', description: 'e.g. "Qwen/Qwen3-8B".' },
      since_revision: { type: 'string', description: 'Your pinned commit.' },
      breaking_only: { type: 'boolean' },
      detail: { type: 'string', enum: ['concise', 'full'] },
      ...PAGE_PARAMS,
    },
    required: ['hf_id'],
    additionalProperties: false,
  },
  handler: async (a: ChangeArgs, _model, ctx) => {
    const id = a.hf_id.replace(/^hf:/i, '').trim();
    if (!HF_ID.test(id)) throw new Error(`model_changes: "${a.hf_id}" is not a Hub repo id. Pass org/name, e.g. "Qwen/Qwen3-8B".`);
    if (a.since_revision != null && !/^[0-9a-f]{7,40}$/i.test(a.since_revision)) {
      throw new Error(`model_changes: since_revision "${a.since_revision}" is not a commit. Pass 7 to 40 hex characters; a branch or tag name moves, which is the problem this tool exists for.`);
    }
    await ctx.progress?.(0, 3, 'reading the advisory index');
    const index = await advisoryIndex(ctx);
    await ctx.progress?.(1, 3, `reading the revision list for ${id}`);
    const releases = await getJson<Releases>(`${feedsBase()}/renovate/${id}.json`, ctx, true);
    await ctx.progress?.(2, 3, 'reading the lineage record');
    const lin = await lineageFile(ctx);
    await ctx.progress?.(3, 3);
    const linHit = lin?.findings.find(f => f.id.toLowerCase() === id.toLowerCase());
    // null: the record could not be read, so nothing is claimed either way.
    const lineage = lin == null ? null : linHit
      ? { base: linHit.base, findings: linHit.kinds.map((kind, i) => ({ kind, basis: linHit.basis[i] ?? '' })) }
      : { base: null, findings: [] as Array<{ kind: string; basis: string }> };

    const lower = id.toLowerCase();
    let rows = index.advisories.filter(r => r.model.toLowerCase() === lower && !r.withdrawn);
    const changelog = releases?.changelogUrl ?? `${DEFAULT_FEEDS}${FEEDS_PATH}/`;

    if (!releases && !rows.length) {
      return {
        model: id,
        tracked: false,
        verdict: `${id} is not in the change record (${new Set(index.advisories.map(r => r.model)).size} models have recorded changes). No record is not the same as no change.`,
        revisions: [],
        pin: null,
        counts: { total: 0, breaking: 0 },
        changes: [],
        total: 0,
        nextCursor: null,
        lineage,
        changelog,
        license: index.license ?? 'CC-BY-4.0',
        basis: 'Neurarch model change record. A model is recorded only once it is tracked.',
      };
    }

    const revisions = (releases?.releases ?? []).map(r => ({ sha: r.digest, date: r.releaseTimestamp, main: r.version === 'main' }));
    let pin: { revision: string; known: boolean; date: string | null } | null = null;
    if (a.since_revision) {
      const rev = a.since_revision.toLowerCase();
      const hit = revisions.find(r => r.sha.toLowerCase().startsWith(rev));
      pin = { revision: a.since_revision, known: !!hit, date: hit?.date ?? null };
      if (hit) rows = rows.filter(r => Date.parse(r.date) > Date.parse(hit.date));
    }
    if (a.breaking_only) rows = rows.filter(r => r.breaking);
    rows = [...rows].sort((x, y) => Date.parse(y.date) - Date.parse(x.date));

    const page = pageOf(rows, a, 20);
    const changes = await Promise.all(page.items.map(async r => {
      const base = { id: r.id, date: r.date, from: r.from, to: r.to, kind: r.kind, breaking: r.breaking, impact: r.impact, url: r.url };
      if (a.detail !== 'full') return base;
      const full = await getJson<{ summary?: string; details?: string; database_specific?: { changed_keys?: unknown[] } }>(r.url, ctx, true);
      return { ...base, summary: full?.summary ?? null, changedKeys: full?.database_specific?.changed_keys ?? null };
    }));
    const breaking = rows.filter(r => r.breaking).length;
    const scope = pin ? (pin.known ? ` since ${a.since_revision!.slice(0, 12)} (${pin.date!.slice(0, 10)})` : '') : '';
    const head = revisions.find(r => r.main);
    const verdict = (rows.length
      ? `${rows.length} recorded change${rows.length === 1 ? '' : 's'}${scope}, ${breaking} breaking`
      : `no recorded change${scope}${a.breaking_only ? ' that is breaking' : ''}`)
      + (pin && !pin.known ? `; ${a.since_revision} is not a revision the record holds, so every change is listed` : '')
      + (head ? `; main is ${head.sha.slice(0, 12)} (${head.date.slice(0, 10)})` : '');

    return {
      model: id,
      tracked: true,
      verdict,
      revisions,
      pin,
      counts: { total: rows.length, breaking },
      changes,
      total: page.total,
      nextCursor: page.nextCursor,
      lineage,
      changelog,
      license: index.license ?? 'CC-BY-4.0',
      basis: 'Neurarch model change record (OSV format). It says what a revision does mechanically, not that anything broke for anyone.',
    };
  },
};

export const CHANGE_TOOLS: ToolDef[] = [modelChangesTool];
