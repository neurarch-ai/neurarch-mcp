#!/usr/bin/env node
/**
 * First-call tool selection: given the listing this server actually sends, and
 * a question that never names a tool, which tool does a model reach for first?
 *
 * One Messages API request per case, no tool is executed: the first tool_use
 * block is the answer, graded against the case's `accept` set in
 * scripts/tool-selection-cases.json. That is the number a description edit
 * moves, so run it before and after one (MCP-Bench grades the same way; a
 * description "improvement" that lowers it is the Smelly 16.67% case).
 *
 *   npm run build
 *   ANTHROPIC_API_KEY=... npm run eval:tools -- [--tools core|full] [--model claude-haiku-5-5]
 *                                               [--max-usd 2] [--dry-run] [--only <id>]
 *   OPENROUTER_API_KEY=... npm run eval:tools -- --provider openrouter [--model qwen/qwen3-next-80b-a3b-instruct]
 *
 * The OpenRouter path is for open models (the default is the one the hosted
 * Neurarch agent runs on the free plan); Claude models go through the Anthropic
 * SDK, never through a compatibility shim.
 *
 * The listing comes from the built server over stdio (tools/list), so what is
 * graded is what a client is shown. Before spending anything it counts the
 * tokens of every request and refuses to start when the estimate exceeds
 * --max-usd; during the run it stops as soon as the measured spend would.
 * Results go to docs/tool-selection-first-call.<provider>.json.
 *
 * Costs real tokens. Not in CI. The multi-turn Claude Code version of this
 * eval (does the agent lint before it edits) is `npm run eval:tools:agent`.
 */
import Anthropic from '@anthropic-ai/sdk';
import { Client } from '@modelcontextprotocol/client';
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const root = resolve(new URL('..', import.meta.url).pathname);
const argv = process.argv.slice(2);
const opt = (name, fallback) => (argv.includes(name) ? argv[argv.indexOf(name) + 1] : fallback);
const provider = opt('--provider', 'anthropic');
if (!['anthropic', 'openrouter'].includes(provider)) { console.error(`--provider is anthropic or openrouter, not ${provider}`); process.exit(2); }
const model = opt('--model', provider === 'openrouter' ? 'qwen/qwen3-next-80b-a3b-instruct' : 'claude-haiku-5-5');
if (provider === 'openrouter' && model.startsWith('anthropic/')) { console.error('Claude models run with --provider anthropic.'); process.exit(2); }
const toolSet = opt('--tools', 'core');
const maxUsd = Number(opt('--max-usd', '2'));
const only = opt('--only', null);
const dryRun = argv.includes('--dry-run');

// USD per million tokens, prompts under 100K (models.md, 2026-10-06). A model
// missing here has to be priced on the command line rather than guessed.
const PRICES = { 'claude-haiku-5-5': [0.10, 0.50], 'claude-sonnet-5-5': [2, 10], 'claude-opus-5-5': [4, 20] };
const OPENROUTER = 'https://openrouter.ai/api/v1';
async function openRouterPrice(id) {
  // OpenRouter publishes its own per-token price per model; read it rather than keep a copy.
  const m = (await (await fetch(`${OPENROUTER}/models`)).json()).data.find(x => x.id === id);
  return m ? [Number(m.pricing.prompt) * 1e6, Number(m.pricing.completion) * 1e6] : undefined;
}
const price = (provider === 'openrouter' ? await openRouterPrice(model) : PRICES[model])
  ?? [Number(opt('--price-in', NaN)), Number(opt('--price-out', NaN))];
if (!price.every(Number.isFinite)) {
  console.error(`No price on file for ${model}; pass --price-in and --price-out (USD per million tokens).`);
  process.exit(2);
}
const usd = (input, output) => (input * price[0] + output * price[1]) / 1e6;

const { cases: allCases } = JSON.parse(readFileSync(join(root, 'scripts', 'tool-selection-cases.json'), 'utf8'));
const cases = only ? allCases.filter(c => c.id === only) : allCases;

// The listing exactly as a client receives it.
const mcp = new Client({ name: 'eval-first-tool', version: '0' });
await mcp.connect(new StdioClientTransport({
  command: process.execPath,
  args: [join(root, 'dist', 'index.js'), join(root, 'examples', 'tiny-vit.py'), `--tools=${toolSet}`],
  stderr: 'ignore',
}));
const listed = (await mcp.listTools()).tools;
const instructions = mcp.getInstructions() ?? '';
await mcp.close();
const tools = listed.map(t => ({ name: t.name, description: t.description ?? '', input_schema: t.inputSchema }));
const toolNames = new Set(tools.map(t => t.name));
for (const c of cases) {
  const missing = c.accept.filter(n => !toolNames.has(n));
  if (missing.length === c.accept.length) console.error(`warning: ${c.id} accepts only tools not in the ${toolSet} listing (${missing.join(', ')})`);
}

// What a coding agent's harness says, plus the server's own instructions the
// way a client injects them. It asks for a tool call; it never names one.
const system = [
  'You are a coding agent working in the user\'s repository. The neurarch MCP server is connected, with the user\'s model (examples/tiny-vit.py) loaded.',
  'Answer the user\'s request by first calling the single tool that addresses it; do not answer from memory.',
  instructions ? `Server instructions:\n${instructions}` : '',
].filter(Boolean).join('\n\n');

const client = provider === 'anthropic' ? new Anthropic() : null;
const request = (c) => ({
  model,
  max_tokens: 2048,
  system,
  tools,
  tool_choice: { type: 'auto', disable_parallel_tool_use: true },
  output_config: { effort: 'low' },
  messages: [{ role: 'user', content: c.ask }],
});

// Estimate from counted input tokens and the full max_tokens of output, so the
// estimate is an upper bound on what the run can spend.
// OpenRouter has no token counter for every model, so there the estimate is
// characters / 3, which over-counts JSON schemas (an upper bound again).
const perCaseInput = client
  ? (await client.messages.countTokens({ model, system, tools, messages: [{ role: 'user', content: cases[0].ask }] })).input_tokens
  : Math.ceil((JSON.stringify(tools).length + system.length) / 3);
const estimate = cases.reduce((s, c) => s + usd(perCaseInput + Math.ceil(c.ask.length / 3), 2048), 0);
console.error(`${cases.length} cases, ${tools.length} tools listed (${toolSet}), ~${perCaseInput} input tokens each, upper-bound estimate $${estimate.toFixed(4)} (cap $${maxUsd}).`);
if (estimate > maxUsd) {
  console.error('Estimate exceeds the cap; not running. Raise --max-usd or narrow with --only.');
  process.exit(2);
}
if (dryRun) process.exit(0);

let spent = 0;
const results = [];
for (const c of cases) {
  let first = null;
  let error = null;
  try {
    if (client) {
      const res = await client.messages.create(request(c));
      spent += usd(res.usage.input_tokens + (res.usage.cache_read_input_tokens ?? 0), res.usage.output_tokens);
      if (res.stop_reason === 'refusal') error = 'refusal';
      first = res.content.find(b => b.type === 'tool_use')?.name ?? null;
    } else {
      const r = await callOpenRouter(c);
      spent += r.usd;
      first = r.first;
      error = r.error;
    }
  } catch (e) {
    if (e instanceof Anthropic.APIError) error = `${e.status ?? ''} ${e.message}`.trim();
    else throw e;
  }
  const pass = first !== null && c.accept.includes(first);
  results.push({ id: c.id, ask: c.ask, accept: c.accept, first, pass, ...(error ? { error } : {}) });
  console.error(`${pass ? 'PASS' : 'FAIL'} ${c.id}: ${first ?? '(no tool)'}${pass ? '' : `  [accept: ${c.accept.join(' | ')}]`}${error ? `  ${error}` : ''}`);
  if (spent > maxUsd) {
    console.error(`Stopped: measured spend $${spent.toFixed(4)} passed the cap.`);
    break;
  }
}

async function callOpenRouter(c) {
  const res = await fetch(`${OPENROUTER}/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`, 'content-type': 'application/json' },
    body: JSON.stringify({
      model,
      max_tokens: 2048,
      messages: [{ role: 'system', content: system }, { role: 'user', content: c.ask }],
      tools: tools.map(t => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.input_schema } })),
      tool_choice: 'auto',
      parallel_tool_calls: false,
      usage: { include: true },
    }),
  });
  const body = await res.json();
  if (!res.ok || body.error) return { first: null, usd: 0, error: `${res.status} ${body.error?.message ?? ''}`.trim() };
  const call = body.choices?.[0]?.message?.tool_calls?.[0];
  const u = body.usage ?? {};
  return { first: call?.function?.name ?? null, usd: typeof u.cost === 'number' ? u.cost : usd(u.prompt_tokens ?? 0, u.completion_tokens ?? 0), error: null };
}

const passed = results.filter(r => r.pass).length;
const report = {
  date: new Date().toISOString().slice(0, 10),
  provider,
  model,
  ...(provider === 'anthropic' ? { effort: 'low' } : {}),
  listing: toolSet,
  toolsListed: tools.length,
  passed,
  total: results.length,
  spentUsd: Number(spent.toFixed(4)),
  results,
};
if (!only) writeFileSync(join(root, 'docs', `tool-selection-first-call.${provider}.json`), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ model, listing: toolSet, passed, total: results.length, spentUsd: report.spentUsd }, null, 2));
