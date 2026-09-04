/**
 * The one place that knows how to reach neurarch.com.
 *
 * Nineteen of this server's tools answer from a vendored engine and never open
 * a socket. Two cannot: `plan` renders the card the CI bot posts, and `history`
 * reads what happened the last time a structure trained inside your
 * organisation. Neither answer exists on this machine, which is the whole
 * point of them, so both go through here and nothing else does.
 *
 * ── www, not the apex ──────────────────────────────────────────────────────
 * The apex 307s every /api route to www, and a redirect drops the
 * Authorization header. For corpus reporting that costs a wasted round trip;
 * here it costs the answer, silently, and looks exactly like an empty ledger.
 * `DEFAULT_API` is pinned to www and a test holds it there.
 *
 * ── The key ────────────────────────────────────────────────────────────────
 * `NEURARCH_API_KEY` is read here and nowhere else. It is required for
 * `history` (the ledger is per organisation) and optional for `plan` (with a
 * key the server folds the history line into the card). It is never sent
 * anywhere but the host in `NEURARCH_API`.
 */

/** Production. `NEURARCH_API` overrides it, the same variable neurarch-trace uses. */
export const DEFAULT_API = 'https://www.neurarch.com';

/** Anything slower than this is not worth an agent's turn. */
export const DEFAULT_TIMEOUT_MS = 30_000;

/** What to tell someone who has no key, in one sentence, wherever it is needed. */
export const KEY_HELP =
  'Set NEURARCH_API_KEY to your organisation\'s key (it looks like nrk_...); '
  + 'https://www.neurarch.com/developer issues one.';

/** Read at call time, never frozen at import: a host may configure the process after loading modules. */
export function apiBase(): string {
  return (process.env.NEURARCH_API || DEFAULT_API).replace(/\/+$/, '');
}

/** The key, or undefined when it is unset or blank. A blank key is not a key. */
export function apiKey(): string | undefined {
  const k = process.env.NEURARCH_API_KEY?.trim();
  return k || undefined;
}

/** The host as a person would name it in an error message. */
export function apiHost(base: string = apiBase()): string {
  try {
    return new URL(base).host;
  } catch {
    return base;
  }
}

interface ApiRequestOptions {
  method: 'GET' | 'POST';
  /** JSON body, for POST. */
  body?: unknown;
  /** Bearer token. Omitted rather than sent empty. */
  key?: string;
  timeoutMs?: number;
}

/**
 * One request, with every failure turned into a sentence the agent can act on.
 *
 * Throws rather than returning a result union: the server wraps a thrown tool
 * error as `<tool> failed: <message>`, which is the right shape for "the
 * network was not there". A 401 or 429 names the fix, because those are the
 * two an agent can actually resolve.
 */
export async function apiRequest<T>(path: string, opts: ApiRequestOptions): Promise<T> {
  const base = apiBase();
  const host = apiHost(base);
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(base + path, {
      method: opts.method,
      headers: {
        Accept: 'application/json',
        ...(opts.body === undefined ? {} : { 'Content-Type': 'application/json' }),
        ...(opts.key ? { Authorization: `Bearer ${opts.key}` } : {}),
      },
      ...(opts.body === undefined ? {} : { body: JSON.stringify(opts.body) }),
      signal: ctrl.signal,
    });
  } catch (e) {
    const why = (e as Error).name === 'AbortError'
      ? `did not answer within ${(opts.timeoutMs ?? DEFAULT_TIMEOUT_MS) / 1000}s`
      : `could not be reached (${(e as Error).message})`;
    throw new Error(`${host} ${why}. Every other tool in this server works offline.`);
  } finally {
    clearTimeout(timer);
  }

  const raw = await res.text();
  if (!res.ok) {
    const detail = serverMessage(raw) ?? res.statusText ?? 'error';
    if (res.status === 401 || res.status === 403) {
      throw new Error(`${host} refused the key: ${detail}. ${KEY_HELP}`);
    }
    if (res.status === 429) {
      throw new Error(`${host} rate-limited this request: ${detail}. ${KEY_HELP}`);
    }
    throw new Error(`${host} returned HTTP ${res.status}: ${detail}`);
  }

  try {
    return JSON.parse(raw) as T;
  } catch {
    throw new Error(`${host} returned a response that is not JSON.`);
  }
}

/** The server's own `error` string when there is one, so its wording reaches the agent. */
function serverMessage(raw: string): string | null {
  try {
    const data = JSON.parse(raw) as { error?: unknown };
    return typeof data.error === 'string' && data.error ? data.error : null;
  } catch {
    return null;
  }
}
