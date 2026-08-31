/**
 * Opt-in corpus reporting: when NEURARCH_REPORT=1 is set, a `validate_model`
 * call shares one anonymous structure+verdict row with the Neurarch corpus.
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
import type { ModelArchitecture } from './types.js';
import type { ValidationReport } from './validation.js';

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

export function reportingEnabled(): boolean {
  return process.env.NEURARCH_REPORT === '1';
}

/** Build the privacy-scoped row for one validate_model result. */
export function buildCorpusReport(model: ModelArchitecture, report: ValidationReport): CorpusReport {
  const counts = new Map<string, number>();
  for (const c of model.components) {
    counts.set(c.type, (counts.get(c.type) ?? 0) + 1);
  }
  const typeHistogram = [...counts.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([t, n]) => `${t}:${n}`)
    .join('|');
  const connectionCount = model.connections.length;

  return {
    fingerprint: fnv1a(`${typeHistogram}#e${connectionCount}`),
    typeHistogram,
    connectionCount,
    layerCount: model.components.length,
    channel: 'mcp',
    // The validator's 'error' is the corpus's 'block': both mean "structurally
    // broken", and the corpus schema speaks the app's severity vocabulary.
    findings: report.findings.map(f => ({
      ruleId: f.rule,
      severity: f.severity === 'error' ? 'block' as const : 'warn' as const,
    })),
  };
}

/** Fire-and-forget send. Swallows everything; 5s cap. */
export function sendCorpusReport(payload: CorpusReport): void {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 5000);
  fetch(reportUrl(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: ctrl.signal,
  })
    .catch(() => { /* reporting must never surface */ })
    .finally(() => clearTimeout(timer));
}
