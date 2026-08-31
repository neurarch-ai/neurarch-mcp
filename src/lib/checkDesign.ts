/**
 * check_design — ask Neurarch's verifier for the whole verdict on this graph.
 *
 * Every other tool in this server answers a question about the model from the
 * file in front of it: how many parameters, what does this layer touch, is
 * there a cycle. Useful, and all of it is inspection. None of it answers the
 * question an agent actually needs answered before it edits someone's
 * architecture, which is "is this design sound, what will it cost to train, and
 * where can it actually run".
 *
 * That verdict is one HTTP call away, and this is the call.
 *
 * ── The network promise ────────────────────────────────────────────────────
 * This server's README promises no network calls at all unless you opt in.
 * That promise is kept, precisely: `check_design` requires NEURARCH_API_KEY,
 * and with no key set it never opens a socket. It returns an explanation
 * instead. Setting a key is the opt-in, the same way NEURARCH_REPORT=1 is for
 * corpus reporting, and it is the only thing in this file that can cause
 * traffic.
 *
 * ── What is sent ───────────────────────────────────────────────────────────
 * The model graph, because a verdict about a graph cannot be computed without
 * it. That is materially different from corpus reporting, which is deliberately
 * incapable of carrying the model, so it is stated plainly here and in the
 * README rather than buried. If you do not want the graph leaving the machine,
 * do not set a key: every other tool in this server keeps working.
 */
import type { ModelArchitecture } from './types.js';

/**
 * Read at call time, not at import time.
 *
 * A module-scope `const` would freeze whatever the environment happened to be
 * when this file was first imported, which is wrong twice over: a host that
 * configures the process after loading modules gets ignored, and a test cannot
 * exercise the override without re-importing the module.
 */
function checkUrl(): string {
  return process.env.NEURARCH_API_URL || DEFAULT_CHECK_URL;
}

/**
 * The host is `www`, and the `www` is load-bearing.
 *
 * The apex `neurarch.com` answers every /api route with a 307 to
 * `www.neurarch.com`. Node's fetch follows it, keeps the method and the body
 * (that is what 307 means), and then does the one thing that breaks this tool:
 * per the fetch spec it strips `Authorization` when a redirect crosses
 * origins. The request that finally lands carries no key, the server answers
 * 401, and the agent is told its key is invalid when the key was fine.
 *
 * So the default points at the host that actually serves the route. Anyone
 * setting NEURARCH_API_URL by hand should do the same.
 */
export const DEFAULT_CHECK_URL = 'https://www.neurarch.com/api/v1/check';

/** Foreground tool, so it gets longer than the fire-and-forget reporter's 5s,
 *  but still a cap: an agent waiting forever on us is worse than an error. */
const TIMEOUT_MS = 20_000;

export interface DesignCheckFinding {
  stage: string;
  severity: 'block' | 'warn';
  title: string;
  detail?: string;
  fix?: string;
}

export interface DesignCheck {
  verdict: 'ok' | 'warn' | 'block' | 'ask';
  outcome: 'complete' | 'blocked' | 'needs_input';
  summary: string;
  stoppedAt: string | null;
  findings: DesignCheckFinding[];
  stages: Array<{ stage: string; status: string; headline: string; data: Record<string, unknown> }>;
  decision?: { question: string; because: string; options: Array<{ label: string; value: string; hint?: string }> };
}

export function apiKey(): string | undefined {
  const k = process.env.NEURARCH_API_KEY?.trim();
  return k ? k : undefined;
}

/**
 * The message shown when no key is configured.
 *
 * Written for the agent that hit it, not for a log: it says what is missing,
 * where to get it, and what still works without it, so the agent can tell the
 * user something useful instead of reporting a bare failure.
 */
export const NO_KEY_MESSAGE =
  'check_design needs a Neurarch API key, and none is set. '
  + 'Create one at https://www.neurarch.com under Settings → Developer API, then start this server with '
  + 'NEURARCH_API_KEY=nrk_... in its environment. '
  + 'Without a key this server makes no network calls, and every other tool here still works offline; '
  + 'validate_model is the local subset (cycles, dangling refs, orphans) of what check_design returns.';

export async function checkDesign(model: ModelArchitecture): Promise<DesignCheck | { error: string }> {
  const key = apiKey();
  // Checked before anything else: no key means no socket, which is the whole
  // basis of the offline promise above.
  if (!key) return { error: NO_KEY_MESSAGE };

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const url = checkUrl();
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
      body: JSON.stringify({ model }),
      signal: ctrl.signal,
    });

    if (!res.ok) {
      // The server's own message is more specific than anything we could
      // invent here (an expired key, a graph over the size cap), so it is
      // passed through rather than flattened into "request failed".
      const body = await res.text().catch(() => '');
      let detail = body.slice(0, 400);
      try {
        const parsed = JSON.parse(body) as { error?: string };
        if (parsed?.error) detail = parsed.error;
      } catch { /* not JSON: the truncated body is the best we have */ }
      return { error: `check_design failed (HTTP ${res.status}): ${detail || 'no response body'}` };
    }

    return (await res.json()) as DesignCheck;
  } catch (e) {
    const err = e as Error;
    if (err.name === 'AbortError') {
      return { error: `check_design timed out after ${TIMEOUT_MS / 1000}s. The graph may be very large, or the service unreachable.` };
    }
    return { error: `check_design could not reach ${checkUrl()}: ${err.message}` };
  } finally {
    clearTimeout(timer);
  }
}
