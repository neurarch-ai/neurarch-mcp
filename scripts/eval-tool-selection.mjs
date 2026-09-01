#!/usr/bin/env node
/**
 * Does an agent, given these tools, reach for the right ones?
 *
 * Every description edit and every prompt edit changes that, and until now
 * nothing measured it. This drives Claude Code headless (`claude -p`) against
 * the built server on examples/tiny-vit.py with a fixed set of asks, records
 * which tools were called in what order, and grades each ask against a rule
 * like "lint_model or check_design was called before any edit was proposed".
 *
 *   npm run build && node scripts/eval-tool-selection.mjs [--only <id>] [--model <name>]
 *
 * Costs real tokens (one short conversation per case). Not in CI. The result
 * goes to docs/tool-selection-eval.json with the date, so regressions are
 * visible as a diff.
 */
import { spawn } from 'node:child_process';
import { writeFileSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const root = resolve(new URL('..', import.meta.url).pathname);
const args = process.argv.slice(2);
const only = args.includes('--only') ? args[args.indexOf('--only') + 1] : null;
const model = args.includes('--model') ? args[args.indexOf('--model') + 1] : null;

const CASES = [
  {
    id: 'orient',
    ask: 'What is this model and where does its parameter budget live? Use the Neurarch tools; do not guess numbers.',
    grade: (calls) => calls.includes('describe_architecture') || calls.includes('get_model_summary'),
    why: 'orientation should start with describe_architecture (or the cheaper summary)',
  },
  {
    id: 'ready',
    ask: 'Is this model ready to train? Give me a verdict.',
    grade: (calls) => calls.includes('check_design') || calls.includes('lint_model'),
    why: 'a readiness question is check_design or lint_model, not inspection tools',
  },
  {
    id: 'lint-before-edit',
    ask: 'Something is wrong with the attention layer. Find it and tell me exactly what to change in the file.',
    grade: (calls) => {
      const lint = calls.findIndex(c => c === 'lint_model' || c === 'check_design');
      return lint !== -1 && (calls.includes('suggest_fix') || calls.includes('get_layer'));
    },
    why: 'find the finding with lint_model, then suggest_fix (or at least get_layer) rather than guessing',
  },
  {
    id: 'rank',
    ask: 'I have two candidate designs, examples/tiny-gpt.neurarch.json and examples/tiny-cnn.neurarch.json. Which one should I spend my training budget on?',
    grade: (calls) => calls.includes('rank_designs'),
    why: 'a which-of-k question is rank_designs, not two check_design calls and a guess',
  },
  {
    id: 'compare',
    ask: 'How does this model compare with a published ViT? Use the reference library.',
    grade: (calls) => calls.includes('list_architectures') || calls.includes('load_architecture'),
    why: 'the reference library is list_architectures / load_architecture',
  },
  {
    id: 'blast-radius',
    ask: 'If I halve the width of the mlp layer, what else breaks?',
    grade: (calls) => calls.includes('layer_impact'),
    why: 'a what-breaks question is layer_impact',
  },
];

const dir = mkdtempSync(join(tmpdir(), 'nx-eval-'));
const mcpConfig = join(dir, 'mcp.json');
writeFileSync(mcpConfig, JSON.stringify({
  mcpServers: { neurarch: { command: 'node', args: [join(root, 'dist', 'index.js'), join(root, 'examples', 'tiny-vit.py')] } },
}));

function runCase(c) {
  return new Promise((res) => {
    const argv = ['-p', c.ask, '--output-format', 'stream-json', '--verbose', '--mcp-config', mcpConfig, '--strict-mcp-config',
      '--allowedTools', 'mcp__neurarch__*',
      // The agent must answer through the server, not by reading the file
      // itself: the question is whether it picks the right MCP tool, and
      // `cat model.py` would make every case pass for the wrong reason.
      '--disallowedTools', 'Bash', 'Read', 'Grep', 'Glob', 'Edit', 'Write', 'WebFetch', 'WebSearch', 'Task', 'Agent',
      '--max-turns', '8'];
    if (model) argv.push('--model', model);
    const child = spawn('claude', argv, { cwd: root, env: { ...process.env, CLAUDECODE: '' } });
    let out = '', err = '';
    child.stdout.on('data', d => { out += d; });
    child.stderr.on('data', d => { err += d; });
    const timer = setTimeout(() => child.kill('SIGKILL'), 180000);
    child.on('close', (code) => {
      clearTimeout(timer);
      const calls = [];
      let text = '';
      for (const line of out.split('\n')) {
        if (!line.trim()) continue;
        let ev; try { ev = JSON.parse(line); } catch { continue; }
        const content = ev?.message?.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            if (block.type === 'tool_use' && typeof block.name === 'string') calls.push(block.name.replace(/^mcp__neurarch__/, ''));
            if (block.type === 'text' && typeof block.text === 'string') text += block.text;
          }
        }
        if (ev?.type === 'result' && typeof ev.result === 'string') text = ev.result;
      }
      res({ id: c.id, ask: c.ask, calls, pass: c.grade(calls), why: c.why, exit: code, answer: text.slice(0, 400), stderr: err.slice(-400) });
    });
  });
}

const results = [];
for (const c of CASES) {
  if (only && c.id !== only) continue;
  process.stderr.write(`running ${c.id}...\n`);
  const r = await runCase(c);
  results.push(r);
  process.stderr.write(`  ${r.pass ? 'PASS' : 'FAIL'} calls=[${r.calls.join(' > ')}]\n`);
}
const passed = results.filter(r => r.pass).length;
const report = { date: new Date().toISOString().slice(0, 10), model: model ?? 'default', passed, total: results.length, results };
if (!only) writeFileSync(join(root, 'docs', 'tool-selection-eval.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ passed, total: results.length, cases: results.map(r => ({ id: r.id, pass: r.pass, calls: r.calls })) }, null, 2));
process.exit(passed === results.length ? 0 : 1);
