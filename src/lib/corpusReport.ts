/**
 * Opt-in corpus reporting: when NEURARCH_REPORT=1 is set, a graph-grading call
 * shares one anonymous structure+verdict row with the Neurarch corpus.
 *
 * ── One vocabulary, or the rows cannot be pooled ───────────────────────────
 * The findings on the row come from `lintModelGraph`, never from the tool the
 * agent happened to call. That is deliberate and it is the whole point: the CI
 * action grades with `lintModelGraph`, the app grades with `lintModelGraph`,
 * and until this file did too, an `mcp` row said `cycle` and `orphan` where a
 * `ci` row said `head-dim-divisibility`. Three channels grading with three
 * vocabularies cannot be pooled, and a corpus that cannot be pooled is a corpus
 * of one channel wearing three names.
 *
 * So the row is a property of the GRAPH, not of the call. validate_model,
 * lint_model and check_design all produce the same row for the same graph,
 * which is also why sending one is safe from three different tools: the
 * fingerprint dedupes them server-side rather than triple-counting a design.
 *
 * What leaves the process: the structural fingerprint (8-char FNV-1a hash of
 * the layer-type histogram + edge count), the histogram and edge count that
 * let the server verify the fingerprint, and (ruleId, severity) pairs. Never
 * the graph, parameter values, layer names, file paths, or any identity; the
 * payload shape cannot carry them.
 *
 * Off by default. Without NEURARCH_REPORT=1 this server makes no network
 * calls at all. Errors are swallowed and the request is capped at 5 seconds:
 * reporting can never fail or slow a tool call (it is fire-and-forget).
 *
 * The hash MUST stay byte-identical to the app's architectureFingerprint
 * (src/utils/structuralIndex.ts) and the server's lib/archFingerprint.ts in
 * the main repo, or rows stop joining the corpus and the server rejects them.
 */
import { lintModelGraph } from '../vendor/engine.bundle.mjs';
import type { ModelArchitecture } from './types.js';

/**
 * `www`, not the apex, for the same reason check_design uses it: the apex 307s
 * every /api route to www. This request carries no Authorization header, so a
 * followed redirect would still deliver the row, but it would pay a wasted
 * round trip inside a 5s fire-and-forget budget for no reason at all.
 *
 * Read at call time rather than frozen at import time, so a host that
 * configures the process after loading modules is still honoured.
 */
export const DEFAULT_REPORT_URL = 'https://www.neurarch.com/api/v1/report';

function reportUrl(): string {
  return process.env.NEURARCH_REPORT_URL || DEFAULT_REPORT_URL;
}

/** Tiny deterministic string hash (FNV-1a, 32-bit) -> 8-char hex. */
function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

export interface CorpusReport {
  fingerprint: string;
  typeHistogram: string;
  connectionCount: number;
  layerCount: number;
  channel: 'mcp';
  findings: Array<{ ruleId: string; severity: 'warn' | 'block' }>;
}

/** The layer-type histogram, sorted so the same graph always hashes the same. */
export function typeHistogramOf(model: ModelArchitecture): string {
  const counts = new Map<string, number>();
  for (const c of model.components) counts.set(c.type, (counts.get(c.type) ?? 0) + 1);
  return [...counts.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([t, n]) => `${t}:${n}`)
    .join('|');
}

/**
 * The 8-char structural id of a graph: what the corpus joins on, and what the
 * training ledger is keyed by.
 *
 * Exported because `history` asks the server "what happened last time THIS
 * structure trained", and the only way that question can be asked without
 * sending the graph is to send this. Two callers computing it two ways would
 * be two callers asking about two different structures, so there is one
 * function, and the cross-repo lock vector in corpusReport.test.ts holds it.
 */
export function structuralFingerprint(model: ModelArchitecture): string {
  return fnv1a(`${typeHistogramOf(model)}#e${model.connections.length}`);
}

export function reportingEnabled(): boolean {
  return process.env.NEURARCH_REPORT === '1';
}

/**
 * Build the privacy-scoped row for one graph.
 *
 * Takes no tool result on purpose: see the vocabulary note at the top. The
 * findings are recomputed here with the engine every channel shares, so the
 * same graph produces the same row whichever tool was called.
 */
export function buildCorpusReport(model: ModelArchitecture): CorpusReport {
  const typeHistogram = typeHistogramOf(model);
  const connectionCount = model.connections.length;

  return {
    fingerprint: structuralFingerprint(model),
    typeHistogram,
    connectionCount,
    layerCount: model.components.length,
    channel: 'mcp',
    // `info` is dropped rather than folded into `warn`: the corpus schema has
    // no info severity, and promoting a note to a warning would inflate every
    // pooled count that reads this channel.
    findings: lintModelGraph(model)
      .filter(f => f.severity === 'block' || f.severity === 'warn')
      .map(f => ({ ruleId: f.rule, severity: f.severity as 'warn' | 'block' })),
  };
}

/**
 * In-flight sends, so a process that is about to exit can wait for them.
 *
 * Fire-and-forget is right for the tool call (a corpus row must never slow or
 * fail a verdict) and wrong for the process: a client that spawns this server,
 * makes one call and closes it exits while the POST is still in the socket, and
 * the row is lost with no error anywhere. That is not a hypothetical, it is how
 * the first hand-run probe of this channel disappeared, and it is invisible
 * exactly where it matters, because a channel that reports nothing looks
 * identical whether it is broken or unused.
 */
const pending = new Set<Promise<void>>();

/** Fire-and-forget send. Swallows everything; 5s cap. */
export function sendCorpusReport(payload: CorpusReport): void {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 5000);
  const p: Promise<void> = fetch(reportUrl(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: ctrl.signal,
  })
    .then(() => undefined)
    .catch(() => { /* reporting must never surface */ })
    .finally(() => { clearTimeout(timer); pending.delete(p); });
  pending.add(p);
}

/**
 * Wait for in-flight rows, for a process that is about to exit.
 *
 * Never throws and never waits longer than `ms`: shutdown must not be held
 * hostage by a reporting endpoint, and the same rule that says a row may not
 * fail a tool call says it may not hang an exit either. Safe to call when
 * nothing is pending, and safe to call twice.
 */
export async function flushCorpusReports(ms = 2000): Promise<void> {
  if (pending.size === 0) return;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<void>(resolve => {
    timer = setTimeout(resolve, ms);
    // Do not let the deadline itself keep the event loop alive.
    (timer as { unref?: () => void }).unref?.();
  });
  try {
    await Promise.race([Promise.allSettled([...pending]).then(() => undefined), deadline]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** Test seam: how many rows are still in flight. */
export function pendingCorpusReports(): number {
  return pending.size;
}
