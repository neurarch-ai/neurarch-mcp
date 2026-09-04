/**
 * Turning ledger rows into the sentence a person would actually say.
 *
 * "This structure trained here 3 days ago: 98.65% in 3 epochs, 34s, under a
 * cent." That line is the only thing a frontier model structurally cannot
 * produce: it is not in the weights, it is not in the repository, and it is
 * not derivable from the graph. Rendering it badly, as a JSON array of ISO
 * timestamps and floats, wastes the one advantage the row has.
 *
 * Everything here is pure and takes `now` as an argument, so the relative day
 * is testable rather than a snapshot of whenever the suite happened to run.
 */

export interface HistoryRow {
  /** ISO timestamp of the run. */
  at: string;
  /** What was measured, as the run reported it: `valAcc`, `valLoss`, ... */
  metric: string;
  value: number;
  epochs: number | null;
  wallSec: number | null;
  estCostUsd: number | null;
  /** Free-form provenance the server assembled: `via=v1/train repo=... pr=8`. */
  source: string;
  jobId?: string | null;
}

/** An accuracy in 0..1 reads as a percentage; anything else is quoted as it was measured. */
export function formatMetric(metric: string, value: number): string {
  if (/acc|f1|auc|precision|recall|iou|dice/i.test(metric) && value >= 0 && value <= 1) {
    return `${(value * 100).toFixed(2)}%`;
  }
  return `${metric} ${trimNumber(value)}`;
}

export function formatDuration(sec: number | null): string | null {
  if (sec === null || !Number.isFinite(sec) || sec < 0) return null;
  if (sec < 60) return `${Math.round(sec)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  if (m < 60) return s ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

/**
 * Cost, rounded the way it would be said out loud.
 *
 * "under a cent" rather than "$0.0075": the point of the number is whether it
 * is worth thinking about, and three significant figures on three quarters of
 * a cent invites a decision nobody should be making.
 */
export function formatCost(usd: number | null): string | null {
  if (usd === null || !Number.isFinite(usd) || usd < 0) return null;
  if (usd === 0) return 'free';
  if (usd < 0.01) return 'under a cent';
  if (usd < 1) return `$${usd.toFixed(2)}`;
  return `$${usd.toFixed(usd < 100 ? 2 : 0)}`;
}

/** "today", "yesterday", "3 days ago", or the date once it is far enough back to be a date. */
export function relativeDay(at: string, now: number): string {
  const then = Date.parse(at);
  if (!Number.isFinite(then)) return 'at an unknown time';
  const days = Math.floor((startOfUtcDay(now) - startOfUtcDay(then)) / 86_400_000);
  if (days <= 0) return 'today';
  if (days === 1) return 'yesterday';
  if (days < 30) return `${days} days ago`;
  return `on ${new Date(then).toISOString().slice(0, 10)}`;
}

function startOfUtcDay(ms: number): number {
  const d = new Date(ms);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
}

/**
 * The one sentence. Clauses the row does not carry are left out rather than
 * filled with a zero: a run with no recorded cost did not cost nothing.
 */
export function summariseRuns(rows: HistoryRow[], now: number = Date.now()): string {
  const [latest] = rows;
  if (!latest) return '';
  const parts = [formatMetric(latest.metric, latest.value)];
  if (latest.epochs !== null && Number.isFinite(latest.epochs)) {
    parts.push(`${latest.epochs} epoch${latest.epochs === 1 ? '' : 's'}`);
  }
  const dur = formatDuration(latest.wallSec);
  if (dur) parts.push(dur);
  const cost = formatCost(latest.estCostUsd);
  if (cost) parts.push(cost);

  const head = `This structure trained here ${relativeDay(latest.at, now)}: ${parts.join(', ')}.`;
  if (rows.length === 1) return head;
  return `${head} ${rows.length} runs of this exact structure are on record; the most recent is quoted.`;
}

/** Fixed-width, newest first, one line per run. Wide enough to read, narrow enough to paste. */
export function renderRunTable(rows: HistoryRow[]): string {
  const body = rows.map(r => [
    (r.at || '').replace('T', ' ').slice(0, 16),
    formatMetric(r.metric, r.value),
    r.epochs === null ? '-' : String(r.epochs),
    formatDuration(r.wallSec) ?? '-',
    formatCost(r.estCostUsd) ?? '-',
    truncate(r.source ?? '', 48),
  ]);
  const header = ['when', 'result', 'epochs', 'wall', 'cost', 'source'];
  const widths = header.map((h, c) => Math.max(h.length, ...body.map(row => row[c].length)));
  const line = (cells: string[]) => cells.map((c, i) => c.padEnd(widths[i])).join('  ').trimEnd();
  return [line(header), ...body.map(line)].join('\n');
}

function truncate(s: string, max: number): string {
  return s.length <= max ? s : `${s.slice(0, max - 1)}…`;
}

function trimNumber(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(6)));
}
