/**
 * The two tools that make an organisation's memory visible inside a coding
 * agent, and the only two in this server that reach the network on purpose.
 *
 *   plan      the front-door card: will it run, what it costs, where it fits,
 *             which of the repository's own rules it breaks, and what happened
 *             last time this structure trained here
 *   history   that last line on its own, for a structure named by fingerprint
 *
 * ── Why these are not offline like everything else ─────────────────────────
 * Every other tool here answers from a vendored engine, because a rule about
 * `embedDim % numHeads` is true on any machine. These two are not that. "This
 * exact structure trained here three days ago and got 98.65%" is a fact about
 * one organisation's runs, and it is the single thing a frontier model
 * structurally cannot produce: not in the weights, not in the repository, not
 * derivable from the graph. It exists on the other side of an API key or it
 * does not exist at all.
 *
 * So the honest design is not to fake it locally. It is to make the reach
 * explicit (`openWorldHint`), to send the least that answers the question
 * (`history` sends an 8-character hash and never the graph), and to say
 * plainly, in the tool result, when there is no key: an empty ledger and an
 * unread ledger are different answers, and only one of them means "nobody has
 * trained this".
 *
 * ── What leaves the machine ────────────────────────────────────────────────
 * `history`: the fingerprint and the key. Nothing else, ever.
 * `plan`:    the graph, the base graph when one is named, and the policy
 *            lines. This is the one tool in the package that sends the model,
 *            which its description says in its first sentence so an agent
 *            cannot reach for it without the user being able to see that.
 *            `share` is pinned to false: an agent must not be able to publish
 *            a public link to someone's design.
 */
import type { ModelArchitecture } from './lib/types.js';
import type { ToolDef, ToolContext } from './tools.js';
import { loadModelCached } from './models.js';
import { structuralFingerprint } from './lib/corpusReport.js';
import { apiKey, apiRequest, apiBase, KEY_HELP } from './lib/neurarchApi.js';
import { readPolicyFor, type Policy } from './lib/policyFile.js';
import { renderRunTable, summariseRuns, type HistoryRow } from './lib/historyText.js';
import pkg from '../package.json';

const FINGERPRINT_RE = /^[0-9a-f]{8}$/;

/** Said the same way wherever a caller has no key, so the fix is never half-stated. */
const NO_KEY_NOTE =
  'The training ledger is per organisation: it answers "what happened when this structure trained HERE", '
  + 'so it cannot be read without your organisation\'s key. No request was made. '
  + 'This is not the same as "never trained": nothing was looked at. '
  + KEY_HELP;

// ── history ──────────────────────────────────────────────────────────────────
export const historyTool: ToolDef = {
  name: 'history',
  description:
    'What happened the last time this exact structure trained inside your organisation: the metric it '
    + 'reached, how many epochs, how long, what it cost. Keyed by the graph\'s 8-character structural '
    + 'fingerprint, so it answers about the shape rather than the file, and two people who arrived at the '
    + 'same architecture see the same runs. Pass model_path for a model, or fingerprint to ask about one '
    + 'directly (a plan result carries it as `plan.model.fingerprint`). Rows are newest first, at most 20. '
    + 'Sends the fingerprint and your API key, never the graph. Without NEURARCH_API_KEY it reads nothing '
    + 'and says so rather than returning an empty list, because an unread ledger and an empty one are '
    + 'different answers.',
  annotations: { openWorldHint: true },
  // A fingerprint names its own subject, so a server started with no model
  // still has an answer to give. Every other tool needs the graph first.
  modelOptional: true,
  inputSchema: {
    type: 'object',
    properties: {
      fingerprint: {
        type: 'string',
        description: '8 hex characters, from a plan result (plan.model.fingerprint). Skips loading a graph.',
      },
    },
    additionalProperties: false,
  },
  handler: async ({ fingerprint }: { fingerprint?: string }, model: ModelArchitecture | undefined) => {
    const given = fingerprint?.trim().toLowerCase();
    if (given !== undefined && !FINGERPRINT_RE.test(given)) {
      throw new Error(
        `history: "${fingerprint}" is not a fingerprint. It is 8 hex characters, as returned by plan `
        + '(`plan.model.fingerprint`). Omit it to use the model this call names instead.',
      );
    }
    if (!given && !model) {
      throw new Error(
        'history: name a structure. Pass model_path (or model_source) for a model, or fingerprint for one '
        + 'you already have. This server was started without a model of its own.',
      );
    }
    const fp = given ?? structuralFingerprint(model as ModelArchitecture);
    const from = given ? 'argument' : 'model';

    const key = apiKey();
    if (!key) {
      return { fingerprint: fp, fingerprintFrom: from, ledger: 'no-key', runs: null, rows: null, summary: NO_KEY_NOTE };
    }

    const data = await apiRequest<{ fingerprint?: string; rows?: HistoryRow[] }>(
      `/api/v1/history?fingerprint=${encodeURIComponent(fp)}`,
      { method: 'GET', key },
    );
    const rows = Array.isArray(data.rows) ? data.rows : [];
    if (!rows.length) {
      return {
        fingerprint: fp,
        fingerprintFrom: from,
        ledger: 'read',
        runs: 0,
        rows: [],
        summary:
          `No run of this structure (${fp}) is recorded under your organisation's key. The ledger holds runs `
          + 'reported through Neurarch; one trained elsewhere leaves no row, so this means "not recorded here" '
          + 'rather than "never trained".',
      };
    }
    return {
      fingerprint: fp,
      fingerprintFrom: from,
      ledger: 'read',
      runs: rows.length,
      summary: summariseRuns(rows),
      table: renderRunTable(rows),
      rows,
    };
  },
};

// ── plan ─────────────────────────────────────────────────────────────────────
interface PlanResponse {
  text: string;
  markdown?: string;
  plan?: {
    model?: { fingerprint?: string };
    run?: { legal?: boolean; blockers?: unknown[]; warnings?: unknown[] };
    policy?: { applied?: boolean; violations?: number } | null;
    history?: { rows?: unknown[] } | null;
  };
}

export const planTool: ToolDef = {
  name: 'plan',
  description:
    'The plan card for this design, rendered by Neurarch: will it run, what it will cost, which GPUs it '
    + 'fits, which of the repository\'s own rules it breaks, what changed against a base design, and what '
    + 'happened last time this structure trained here. This is the one tool that sends the graph off the '
    + 'machine (POST /api/v1/plan), which is what buys the last two lines; everything else in this server '
    + 'is local. It is the same card the Neurarch CI bot posts on a pull request and the neurarch-trace CLI '
    + 'prints, so an agent and a reviewer read the same words about the same graph. `text` comes back '
    + 'verbatim. Policy lines are read from the repository\'s .neurarch.yml unless you pass `policy`; '
    + 'base_path adds a diff. NEURARCH_API_KEY is optional and adds the history line. Nothing is shared '
    + 'publicly: the tool pins `share` to false.',
  annotations: { openWorldHint: true },
  inputSchema: {
    type: 'object',
    properties: {
      policy: {
        type: 'object',
        description:
          'House rules: max_params, max_memory_gb, must_fit, forbid_types, require_types, max_layers, '
          + 'max_cost_usd. Hard lines block, budgets warn. Replaces the .neurarch.yml lookup.',
        additionalProperties: true,
      },
      base_path: {
        type: 'string',
        description: 'A second model to diff against (a path, zoo:<id> or hf:<org/name>).',
      },
    },
    additionalProperties: false,
  },
  handler: async (
    { policy, base_path }: { policy?: Policy; base_path?: string },
    model: ModelArchitecture,
    ctx: ToolContext,
  ) => {
    const modelFile = ctx.currentPath || ctx.modelPath;
    let policyNote: string;
    let policyPath: string | null = null;
    let matchedModel: string | null = null;
    let ignoredKeys: string[] = [];
    let effective: Policy | null = null;

    if (policy && Object.keys(policy).length) {
      effective = policy;
      policyNote = 'Policy taken from the call, so no .neurarch.yml was read.';
    } else if (policy) {
      policyNote = 'Policy explicitly empty on the call, so no .neurarch.yml was read.';
    } else {
      const found = await readPolicyFor(modelFile);
      effective = found.policy;
      policyPath = found.path;
      matchedModel = found.matchedModel;
      ignoredKeys = found.ignoredKeys;
      policyNote = found.note;
    }

    const base = base_path ? await loadModelCached(base_path) : undefined;
    const key = apiKey();
    const data = await apiRequest<PlanResponse>('/api/v1/plan', {
      method: 'POST',
      key,
      body: {
        model,
        ...(base ? { base } : {}),
        ...(effective ? { policy: effective } : {}),
        // Never true from a tool call: publishing someone's design is a
        // decision for the person, not for the agent holding their file.
        share: false,
        source: { kind: 'mcp', tool: `neurarch-mcp/${pkg.version}` },
      },
    });
    if (typeof data.text !== 'string') {
      throw new Error(`plan: ${apiBase()} returned a response with no plan text.`);
    }

    const runs = data.plan?.history?.rows?.length ?? 0;
    return {
      // Verbatim, and first, because it is the answer. The structured plan
      // under it is for an agent that wants to branch on a number.
      text: data.text,
      markdown: data.markdown ?? null,
      plan: data.plan ?? null,
      policy: {
        source: policyPath ? 'file' : effective ? 'argument' : 'none',
        path: policyPath,
        lines: effective ? Object.keys(effective) : [],
        matchedModel,
        ...(ignoredKeys.length ? { ignoredKeys } : {}),
        violations: data.plan?.policy?.violations ?? 0,
        note: policyNote,
      },
      history: key
        ? { included: runs > 0, runs }
        : {
          included: false,
          runs: null,
          note: `The card carries no history line. ${NO_KEY_NOTE}`,
        },
      sentTo: `${apiBase()}/api/v1/plan`,
      ...(base_path ? { comparedWith: base_path } : {}),
    };
  },
};

/**
 * The two ledger tools, `plan` first: it is the one an agent should reach for
 * on arrival, and `history` is the line inside it asked on its own.
 */
export const LEDGER_TOOLS: ToolDef[] = [planTool, historyTool];
