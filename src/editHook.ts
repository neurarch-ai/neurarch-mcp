/**
 * `neurarch-mcp hook --edit`: a Claude Code PostToolUse hook for file edits.
 *
 * More of the code that serves models is now written by an agent: a `vllm serve` line, a KServe
 * manifest, a `from_pretrained` call with no revision. When an Edit, MultiEdit or Write touches a file
 * that names a Hugging Face model or a serving launch, this runs `neurarch gate` on that one file (the
 * same command CI runs) and hands the verdict back to the agent while it still has the file open:
 *
 *   refuse  exit 2 with the reasons on stderr. The edit has happened; Claude Code shows the agent the
 *           reasons as feedback, so it fixes the file in the same turn instead of a reviewer finding it.
 *   review  exit 0 with `additionalContext`: what to read, without stopping anything.
 *   admit   exit 0 and silence.
 *
 * It fails open, always, like the Bash hook: a file it does not recognise, a gate it cannot run, a
 * timeout, output it cannot parse, any exception at all is exit 0 and silence. A hook that breaks an
 * agent's turn is worse than no hook. `NEURARCH_ALLOW_EDIT=1` in the environment turns it off.
 *
 * No check logic here: the verdict is the CLI's own (`npx neurarch gate --json`), so the agent sees
 * exactly what the pull request will see.
 */
import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { basename, extname, isAbsolute, join } from 'node:path';

const EDIT_TOOLS = new Set(['Edit', 'MultiEdit', 'Write']);
const KINDS = new Set(['.py', '.sh', '.bash', '.yaml', '.yml', '.toml', '.json', '.ipynb', '.env', '.cfg']);

/** A model id, a serving launch or a manifest field that hands a model to a runtime. Cheap, before spawning anything. */
const SIGNAL = /from_pretrained\s*\(|vllm\s+serve|sglang\.launch_server|text-generation-launcher|llama-server|storageUri|hf:\/\/|--model(?:[-_]path|[-_]id)?[\s=]|\bmodel(?:_id|_name|Name)?\s*[:=]\s*["']?[\w.-]+\/[\w.-]+/;

export interface GateRunResult { code: number; stdout: string }
export type GateRunner = (file: string, cwd: string) => Promise<GateRunResult | null>;

export interface EditDecision { exit: 0 | 2; stderr?: string; additionalContext?: string }

/** The file an edit touched, or null when the payload is not an edit this hook reads. */
export function editedFile(payload: unknown): { file: string; cwd: string } | null {
  if (!payload || typeof payload !== 'object') return null;
  const p = payload as { tool_name?: unknown; tool_input?: { file_path?: unknown }; cwd?: unknown };
  if (typeof p.tool_name !== 'string' || !EDIT_TOOLS.has(p.tool_name)) return null;
  const fp = p.tool_input?.file_path;
  if (typeof fp !== 'string' || !fp) return null;
  const cwd = typeof p.cwd === 'string' && p.cwd ? p.cwd : process.cwd();
  const file = isAbsolute(fp) ? fp : join(cwd, fp);
  const name = basename(file);
  if (!KINDS.has(extname(name).toLowerCase()) && !/^(Dockerfile|Containerfile|Makefile)$/i.test(name)) return null;
  return { file, cwd };
}

interface GateJson { verdict?: string; because?: string[]; models?: Array<{ id: string; findings?: Array<{ severity: string; message: string }> }>; launches?: Array<{ command: string; findings?: Array<{ severity: string; title: string }> }> }

/** What the agent is told, from the gate's own JSON. Only lines at or above what decided the verdict. */
export function decideFromGate(file: string, gate: GateJson): EditDecision {
  const shown: string[] = [];
  const hi = gate.verdict === 'refuse' ? new Set(['high', 'block']) : new Set(['high', 'medium', 'block', 'warn']);
  for (const m of gate.models ?? []) for (const f of m.findings ?? []) if (hi.has(f.severity)) shown.push(`${m.id}: ${f.message}`);
  for (const l of gate.launches ?? []) for (const f of l.findings ?? []) if (hi.has(f.severity)) shown.push(`${l.command.slice(0, 100)}: ${f.title}`);
  const list = shown.slice(0, 8).map((s) => `- ${s.length > 300 ? `${s.slice(0, 297)}...` : s}`).join('\n');
  const because = (gate.because ?? []).join(', ');
  if (gate.verdict === 'refuse') {
    return { exit: 2, stderr: `neurarch gate refuses ${basename(file)} as edited (${because}). The pull request's gate will refuse it too:\n${list}\nFix these in the file, or set NEURARCH_ALLOW_EDIT=1 to turn this hook off.\n` };
  }
  if (gate.verdict === 'review') {
    return { exit: 0, additionalContext: `neurarch gate on ${basename(file)}: review (${because}). Nothing blocks; worth reading before this goes live:\n${list || '- see `npx neurarch gate ' + basename(file) + '`'}` };
  }
  return { exit: 0 };
}

export async function decideEdit(payload: unknown, run: GateRunner, env: Record<string, string | undefined> = process.env): Promise<EditDecision> {
  if (env.NEURARCH_ALLOW_EDIT === '1') return { exit: 0 };
  const target = editedFile(payload);
  if (!target) return { exit: 0 };
  let text: string;
  try { text = await readFile(target.file, 'utf-8'); } catch { return { exit: 0 }; }
  if (text.length > 2_000_000 || !SIGNAL.test(text)) return { exit: 0 };
  const r = await run(target.file, target.cwd);
  if (!r) return { exit: 0 };
  let gate: GateJson;
  try { gate = JSON.parse(r.stdout); } catch { return { exit: 0 }; }
  return decideFromGate(target.file, gate);
}

/** `npx -y neurarch gate <file> --json` with a deadline; null on anything but a parsed run. */
export const npxGate: GateRunner = (file, cwd) => new Promise((done) => {
  let out = '';
  let child;
  try {
    child = spawn('npx', ['-y', 'neurarch', 'gate', file, '--json'], { cwd, stdio: ['ignore', 'pipe', 'ignore'], env: { ...process.env, NO_COLOR: '1' } });
  } catch { done(null); return; }
  const timer = setTimeout(() => { child.kill('SIGKILL'); done(null); }, 100_000);
  child.stdout.on('data', (b: Buffer) => { out += b.toString('utf-8'); });
  child.on('error', () => { clearTimeout(timer); done(null); });
  child.on('close', (code: number | null) => { clearTimeout(timer); done({ code: code ?? 1, stdout: out }); });
});

export async function runEditHook(stdinText: string, io: { out: (t: string) => void; err: (t: string) => void }, run: GateRunner = npxGate): Promise<number> {
  let payload: unknown;
  try { payload = JSON.parse(stdinText); } catch { return 0; }
  let d: EditDecision;
  try { d = await decideEdit(payload, run); } catch { return 0; }
  if (d.exit === 2 && d.stderr) { io.err(d.stderr); return 2; }
  if (d.additionalContext) io.out(JSON.stringify({ hookSpecificOutput: { hookEventName: 'PostToolUse', additionalContext: d.additionalContext } }));
  return 0;
}
