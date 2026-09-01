/**
 * `neurarch-mcp lint <files...>` and `neurarch-mcp check <file>`: the two
 * grading tools as a command line, for the moments an agent is not involved.
 *
 * The neurarch-lint package on npm did this already, as a second install with
 * its own name to learn. One package that is both the server and the CLI is
 * fewer things to remember and one place for the stars to land; neurarch-lint
 * stays published as an alias.
 *
 * Exit code is the finding: 0 clean, 1 at least one block, 2 could not read
 * the input. `--json` prints the raw result for CI to parse.
 */
import { loadModelCached } from './models.js';
import { lintModelGraph } from './vendor/engine.bundle.mjs';
import { checkDesign, provenanceForRules } from './lib/checkDesign.js';

export interface CommandIO {
  out: (s: string) => void;
  err: (s: string) => void;
}

export async function runLintCommand(files: string[], json: boolean, io: CommandIO): Promise<number> {
  if (files.length === 0) {
    io.err('usage: neurarch-mcp lint <model.py | model.neurarch.json | zoo:<id>> [...more] [--json]\n');
    return 2;
  }
  let exit = 0;
  const all: Record<string, unknown> = {};
  for (const file of files) {
    let findings;
    try {
      findings = lintModelGraph(await loadModelCached(file));
    } catch (e) {
      io.err(`${file}: ${(e as Error).message}\n`);
      exit = Math.max(exit, 2);
      continue;
    }
    const blocks = findings.filter(f => f.severity === 'block').length;
    const warns = findings.filter(f => f.severity === 'warn').length;
    if (blocks > 0) exit = Math.max(exit, 1);
    all[file] = { blocks, warns, findings, provenance: provenanceForRules(findings.map(f => f.rule)) };
    if (json) continue;
    io.out(`${file}: ${blocks} block, ${warns} warn, ${findings.length - blocks - warns} info\n`);
    for (const f of findings) {
      const where = f.componentName ? ` [${f.componentName}${f.componentType ? `:${f.componentType}` : ''}]` : '';
      io.out(`  ${f.severity.padEnd(5)} ${f.rule}${where}: ${f.message}\n`);
    }
  }
  if (json) io.out(JSON.stringify(files.length === 1 ? all[files[0]] : all, null, 2) + '\n');
  return exit;
}

export async function runCheckCommand(files: string[], json: boolean, io: CommandIO): Promise<number> {
  if (files.length !== 1) {
    io.err('usage: neurarch-mcp check <model.py | model.neurarch.json | zoo:<id>> [--json]\n');
    return 2;
  }
  let result;
  try {
    result = checkDesign(await loadModelCached(files[0]));
  } catch (e) {
    io.err(`${files[0]}: ${(e as Error).message}\n`);
    return 2;
  }
  if ('error' in result) {
    io.err(`${files[0]}: ${result.error}\n`);
    return 2;
  }
  if (json) {
    io.out(JSON.stringify(result, null, 2) + '\n');
  } else {
    io.out(`${files[0]}: ${result.verdict} (${result.outcome})\n  ${result.summary}\n`);
    for (const s of result.stages) io.out(`  ${s.status.padEnd(8)} ${s.stage}: ${s.headline}\n`);
    for (const f of result.findings) io.out(`  ${f.severity.padEnd(5)} ${f.stage}: ${f.title}${f.fix ? ` (fix: ${f.fix})` : ''}\n`);
    if (result.decision) io.out(`  ask: ${result.decision.question}\n`);
  }
  return result.verdict === 'block' ? 1 : 0;
}
