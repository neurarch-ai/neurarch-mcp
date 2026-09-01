#!/usr/bin/env node
// Runs the vendored parser + linter over hand-written PyTorch model files from
// popular public repositories and reports exactly what happened: which files
// parsed, which did not, what the linter said, and (merged from a hand-written
// verdict file) whether each finding was real, a parser artefact, or a true
// structural fact that is not a bug.
//
//   node scripts/lint-real-repos.mjs --repos-dir <dir> [--clone] [--manifest scripts/real-repos.json]
//                                    [--out docs/real-repos-results.json]
//                                    [--verdicts scripts/real-repos-verdicts.json]
//
// <dir> holds one checkout per repo at <dir>/<owner>__<name>/. With --clone,
// missing repos are shallow-cloned blob-less and only the manifest files are
// checked out, so the run does not need a full copy of transformers.
//
// Failures are results. A file the parser cannot read is recorded with the
// reason, never skipped.

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { graphFromPyTorchSource, lintModelGraph } from '../src/vendor/engine.bundle.mjs';
import { provenanceFor } from '../src/vendor/verifier.bundle.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  if (i === -1) return fallback;
  const v = process.argv[i + 1];
  return v === undefined || v.startsWith('--') ? true : v;
}

const reposDir = arg('--repos-dir', null);
const doClone = process.argv.includes('--clone');
const manifestPath = resolve(arg('--manifest', join(root, 'scripts', 'real-repos.json')));
const outPath = resolve(arg('--out', join(root, 'docs', 'real-repos-results.json')));
const verdictsPath = resolve(arg('--verdicts', join(root, 'scripts', 'real-repos-verdicts.json')));

if (!reposDir) {
  console.error('usage: node scripts/lint-real-repos.mjs --repos-dir <dir> [--clone]');
  process.exit(2);
}

const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
const verdicts = existsSync(verdictsPath) ? JSON.parse(readFileSync(verdictsPath, 'utf8')) : {};

const engineBytes = readFileSync(join(root, 'src', 'vendor', 'engine.bundle.mjs'));
const engineIdentity = {
  firstLine: engineBytes.toString('utf8').split('\n')[0],
  bytes: engineBytes.length,
  sha256: createHash('sha256').update(engineBytes).digest('hex'),
};

// nn.<Layer>( constructor calls in a source range, excluding containers and
// non-layer helpers. This is the denominator for "how much of the file did the
// parser actually turn into nodes"; a file with 40 constructor calls that
// parses to 1 node is a parse in name only.
const NOT_LAYERS = new Set(['Module', 'ModuleList', 'ModuleDict', 'Sequential', 'Parameter', 'ParameterList',
  'ParameterDict', 'functional', 'init', 'Identity', 'DataParallel', 'utils']);
function nnConstructorCalls(text) {
  let n = 0;
  for (const m of text.matchAll(/\bnn\.(\w+)\s*\(/g)) if (!NOT_LAYERS.has(m[1])) n++;
  return n;
}

// A param the parser stored as source text it could not evaluate, e.g.
// "config.hidden_size" or "dims[0]". Plain enum-like strings ("same", "gelu")
// are legitimate values and are not counted.
const ENUM_STRINGS = /^(true|false|none|same|valid|relu|gelu|silu|swish|tanh|sigmoid|mean|sum|zeros|circular|reflect|replicate|nearest|bilinear|bicubic|linear|trilinear|area|batch|layer|group|instance|max|avg)$/i;
function unresolvedParamCount(model) {
  let n = 0;
  for (const c of model.components) {
    for (const v of Object.values(c.params ?? {})) {
      if (typeof v !== 'string') continue;
      if (/[.\[\]()*/+-]/.test(v) || (/^[A-Za-z_]\w*$/.test(v) && !ENUM_STRINGS.test(v))) n++;
    }
  }
  return n;
}

function repoDir(repo) {
  return join(reposDir, repo.replace('/', '__'));
}

function ensureClone(entry) {
  const dir = repoDir(entry.repo);
  if (!existsSync(join(dir, '.git'))) {
    if (!doClone) return `not cloned (run with --clone): ${dir}`;
    mkdirSync(dirname(dir), { recursive: true });
    try {
      execFileSync('git', ['clone', '--depth', '1', '--quiet', '--filter=blob:none', '--no-checkout',
        `https://github.com/${entry.repo}.git`, dir], { stdio: ['ignore', 'ignore', 'pipe'] });
    } catch (e) {
      return `clone failed: ${String(e.stderr ?? e.message).trim().split('\n').pop()}`;
    }
  }
  const missing = entry.files.filter((f) => !existsSync(join(dir, f)));
  if (missing.length) {
    try {
      execFileSync('git', ['-C', dir, 'checkout', 'HEAD', '--', ...missing], { stdio: ['ignore', 'ignore', 'pipe'] });
    } catch (e) {
      return `checkout failed: ${String(e.stderr ?? e.message).trim().split('\n').pop()}`;
    }
  }
  return null;
}

// Mirror of the engine's main-class heuristic (findMainModelClass): the LAST
// class in the file that defines both __init__ and forward. Computed here so the
// report can say which class the parser actually read, which on a file with
// twenty classes is the difference between "parsed the model" and "parsed the
// token-classification head at the bottom of the file".
function classInventory(code) {
  const lines = code.split('\n');
  const classes = [];
  let cur = null;
  for (let i = 0; i < lines.length; i++) {
    // Class headers may span lines (`class X(\n    A, B\n):`), so match the
    // opening only and take whatever bases sit on the first line.
    const m = lines[i].match(/^class\s+(\w+)\s*(?:\(([^)]*)\)?)?\s*:?\s*$/);
    if (m) {
      cur = { name: m[1], bases: m[2] ?? '', line: i + 1, init: false, forward: false };
      classes.push(cur);
      continue;
    }
    if (cur && /^\s+def\s+__init__\s*\(/.test(lines[i])) cur.init = true;
    if (cur && /^\s+def\s+forward\s*\(/.test(lines[i])) cur.forward = true;
  }
  const moduleClasses = classes.filter((c) => /Module|Model|PreTrainedModel|Mixin|Block|Layer/.test(c.bases) || (c.init && c.forward));
  const candidates = classes.filter((c) => c.init && c.forward);
  return {
    total: classes.length,
    nnModuleSubclasses: classes.filter((c) => /nn\.Module|Module\b/.test(c.bases)).length,
    withInitAndForward: candidates.length,
    mainClass: candidates.length ? candidates[candidates.length - 1] : null,
    moduleClasses: moduleClasses.length,
  };
}

function verdictKey(repo, file, f) {
  return `${repo}|${file}|${f.rule}|${f.componentName ?? ''}`;
}

const results = [];
const t0 = Date.now();

for (const entry of manifest) {
  const cloneErr = ensureClone(entry);
  for (const file of entry.files) {
    const abs = join(repoDir(entry.repo), file);
    const rec = { repo: entry.repo, file, status: null };
    if (cloneErr && !existsSync(abs)) {
      rec.status = 'unavailable';
      rec.reason = cloneErr;
      results.push(rec);
      continue;
    }
    if (!existsSync(abs)) {
      rec.status = 'unavailable';
      rec.reason = 'file not present in checkout';
      results.push(rec);
      continue;
    }
    const code = readFileSync(abs, 'utf8');
    rec.lines = code.split('\n').length;
    rec.classes = classInventory(code);
    rec.nnCallsInFile = nnConstructorCalls(code);
    const start = performance.now();
    let model = null;
    try {
      model = graphFromPyTorchSource(code, `${entry.repo}:${file}`);
    } catch (e) {
      rec.status = 'parser-error';
      rec.reason = e?.message ?? String(e);
    }
    rec.parseMs = Math.round((performance.now() - start) * 10) / 10;
    if (rec.status === 'parser-error') {
      results.push(rec);
      continue;
    }
    if (!model) {
      rec.status = rec.classes.mainClass ? 'no-model' : 'no-model-class';
      // Hand-attributed construct behind the failure, from the verdict file
      // under the key `<repo>|<file>|@cause`.
      rec.cause = verdicts[`${entry.repo}|${file}|@cause`]?.cause ?? null;
      rec.reason = rec.classes.mainClass
        ? `main class ${rec.classes.mainClass.name} (line ${rec.classes.mainClass.line}) yielded zero layers`
        : 'no class defines both __init__ and forward';
      results.push(rec);
      continue;
    }
    rec.status = 'parsed';
    rec.mainClass = rec.classes.mainClass?.name ?? null;
    const layers = model.components.filter((c) => c.type !== 'input' && c.type !== 'output');
    rec.layers = layers.length;
    rec.connections = model.connections.length;
    rec.unresolvedParams = unresolvedParamCount(model);
    // Nodes produced per nn.* constructor call in the file. Above 1 is possible
    // (loops, expansion of helper classes); far below 1 means most of the file
    // was skipped.
    rec.coverage = rec.nnCallsInFile ? Math.round((100 * rec.layers) / rec.nnCallsInFile) / 100 : null;
    const hist = {};
    for (const c of layers) hist[c.type] = (hist[c.type] ?? 0) + 1;
    rec.layerTypes = Object.fromEntries(Object.entries(hist).sort((a, b) => b[1] - a[1]));
    let findings = [];
    try {
      findings = lintModelGraph(model);
    } catch (e) {
      rec.status = 'lint-error';
      rec.reason = e?.message ?? String(e);
      results.push(rec);
      continue;
    }
    const prov = provenanceFor(findings.map((f) => f.rule));
    rec.findings = findings.map((f) => {
      const v = verdicts[verdictKey(entry.repo, file, f)];
      return {
        rule: f.rule,
        severity: f.severity,
        component: f.componentName ?? null,
        componentType: f.componentType ?? null,
        message: f.message,
        measured: Boolean(prov[f.rule]),
        verdict: v?.verdict ?? null,
        reason: v?.reason ?? null,
      };
    });
    rec.blocks = findings.filter((f) => f.severity === 'block').length;
    rec.warns = findings.filter((f) => f.severity === 'warn').length;
    rec.infos = findings.filter((f) => f.severity === 'info').length;
    results.push(rec);
  }
}

// Aggregates.
const repos = [...new Set(manifest.map((e) => e.repo))];
const attempted = results.filter((r) => r.status !== 'unavailable');
const parsed = results.filter((r) => r.status === 'parsed');
const reposWithParse = new Set(parsed.map((r) => r.repo));
const reposAttempted = new Set(attempted.map((r) => r.repo));
const byStatus = {};
for (const r of results) byStatus[r.status] = (byStatus[r.status] ?? 0) + 1;

const allFindings = parsed.flatMap((r) => r.findings.map((f) => ({ ...f, repo: r.repo, file: r.file })));
const byRule = {};
for (const f of allFindings) {
  byRule[f.rule] ??= { severity: f.severity, count: 0, measured: f.measured, real: 0, artefact: 0, notABug: 0, unjudged: 0 };
  byRule[f.rule].count++;
  if (f.verdict === 'real') byRule[f.rule].real++;
  else if (f.verdict === 'artefact') byRule[f.rule].artefact++;
  else if (f.verdict === 'not-a-bug') byRule[f.rule].notABug++;
  else byRule[f.rule].unjudged++;
}
const unparsedByCause = {};
for (const r of results) if (r.status === 'no-model' || r.status === 'no-model-class') {
  const k = r.cause ?? 'unattributed';
  unparsedByCause[k] = (unparsedByCause[k] ?? 0) + 1;
}
const bySeverity = {};
for (const f of allFindings) bySeverity[f.severity] = (bySeverity[f.severity] ?? 0) + 1;

function judged(sev) {
  const fs = allFindings.filter((f) => f.severity === sev);
  return {
    total: fs.length,
    real: fs.filter((f) => f.verdict === 'real').length,
    artefact: fs.filter((f) => f.verdict === 'artefact').length,
    notABug: fs.filter((f) => f.verdict === 'not-a-bug').length,
    unjudged: fs.filter((f) => !f.verdict).length,
  };
}

const summary = {
  date: new Date().toLocaleDateString('en-CA'),
  engine: engineIdentity,
  command: `node scripts/lint-real-repos.mjs --repos-dir <dir>${doClone ? ' --clone' : ''}`,
  repos: repos.length,
  reposAttempted: reposAttempted.size,
  reposWithAtLeastOneParsedFile: reposWithParse.size,
  filesListed: results.length,
  filesAttempted: attempted.length,
  filesParsed: parsed.length,
  filesByStatus: byStatus,
  parsedGraphsWithBlock: parsed.filter((r) => r.blocks > 0).length,
  parsedGraphsWithAnyFinding: parsed.filter((r) => r.findings.length > 0).length,
  totalLayers: parsed.reduce((a, r) => a + r.layers, 0),
  layersPerParsedFile: (() => {
    const xs = parsed.map((r) => r.layers).sort((a, b) => a - b);
    return { min: xs[0] ?? 0, median: xs[Math.floor(xs.length / 2)] ?? 0, max: xs[xs.length - 1] ?? 0 };
  })(),
  thinParses: parsed.filter((r) => r.layers <= 3).length,
  substantialParses: parsed.filter((r) => r.layers >= 10).length,
  parsedFilesWithUnresolvedParams: parsed.filter((r) => r.unresolvedParams > 0).length,
  nnCallsInParsedFiles: parsed.reduce((a, r) => a + r.nnCallsInFile, 0),
  unparsedByCause: Object.fromEntries(Object.entries(unparsedByCause).sort((a, b) => b[1] - a[1])),
  findingsBySeverity: bySeverity,
  findingsByRule: byRule,
  handJudged: { block: judged('block'), warn: judged('warn'), info: judged('info') },
  wallMs: Date.now() - t0,
};

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify({ summary, results }, null, 2) + '\n');

// Markdown table.
const pct = (a, b) => (b ? `${Math.round((1000 * a) / b) / 10}%` : 'n/a');
const out = [];
out.push(`Engine: ${engineIdentity.firstLine} (${engineIdentity.bytes} bytes, sha256 ${engineIdentity.sha256.slice(0, 12)})`);
out.push(`Repos: ${repos.length} listed, ${reposAttempted.size} attempted, ${reposWithParse.size} with at least one parsed file (${pct(reposWithParse.size, reposAttempted.size)})`);
out.push(`Files: ${results.length} listed, ${attempted.length} attempted, ${parsed.length} parsed (${pct(parsed.length, attempted.length)})`);
out.push(`Status: ${Object.entries(byStatus).map(([k, v]) => `${k}=${v}`).join(', ')}`);
out.push(`Parsed graph size: min ${summary.layersPerParsedFile.min}, median ${summary.layersPerParsedFile.median}, max ${summary.layersPerParsedFile.max} layers; thin (<=3 layers) ${summary.thinParses}, substantial (>=10) ${summary.substantialParses}; ${summary.totalLayers} nodes from ${summary.nnCallsInParsedFiles} nn.* constructor calls`);
out.push(`Findings: ${Object.entries(bySeverity).map(([k, v]) => `${k}=${v}`).join(', ') || 'none'}; parsed graphs with a block: ${summary.parsedGraphsWithBlock}/${parsed.length}`);
out.push('');
out.push('| Repo | File | Status | Main class | Layers | nn.* calls | Unresolved | Findings | Verdicts |');
out.push('|---|---|---|---|---|---|---|---|---|');
for (const r of results) {
  const findingsCell = r.findings?.length
    ? Object.entries(r.findings.reduce((m, f) => ((m[`${f.rule}${f.severity === 'block' ? ' (block)' : ''}`] = (m[`${f.rule}${f.severity === 'block' ? ' (block)' : ''}`] ?? 0) + 1), m), {}))
        .map(([k, v]) => (v > 1 ? `${k} x${v}` : k)).join(', ')
    : r.status === 'parsed' ? 'clean' : (r.reason ?? '');
  const verdictCell = r.findings?.length
    ? ['real', 'artefact', 'not-a-bug'].map((v) => [v, r.findings.filter((f) => f.verdict === v).length]).filter(([, n]) => n).map(([v, n]) => `${v}=${n}`).join(' ') || ''
    : '';
  out.push(`| ${r.repo} | ${r.file} | ${r.status} | ${r.mainClass ?? r.classes?.mainClass?.name ?? ''} | ${r.layers ?? ''} | ${r.nnCallsInFile ?? ''} | ${r.unresolvedParams ?? ''} | ${findingsCell} | ${verdictCell || r.cause || ''} |`);
}
out.push('');
out.push('| Unparsed file cause | Files |');
out.push('|---|---|');
for (const [k, v] of Object.entries(summary.unparsedByCause)) out.push(`| ${k} | ${v} |`);
out.push('');
out.push('| Rule | Severity | Measured | Count | Real | Artefact | Not a bug | Unjudged |');
out.push('|---|---|---|---|---|---|---|---|');
for (const [rule, s] of Object.entries(byRule).sort((a, b) => b[1].count - a[1].count)) {
  out.push(`| ${rule} | ${s.severity} | ${s.measured ? 'yes' : 'no'} | ${s.count} | ${s.real} | ${s.artefact} | ${s.notABug} | ${s.unjudged} |`);
}
out.push('');
out.push(`Hand-judged blocks: ${JSON.stringify(summary.handJudged.block)}`);
out.push(`Hand-judged warns: ${JSON.stringify(summary.handJudged.warn)}`);
out.push(`Wrote ${outPath} in ${summary.wallMs} ms`);
console.log(out.join('\n'));
