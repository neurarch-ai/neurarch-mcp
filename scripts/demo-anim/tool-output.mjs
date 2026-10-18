// Prints one real tool result, the source for every number in scripts.mjs.
// Run from the repo root after `npm run build`:
//
//   node scripts/demo-anim/tool-output.mjs <model> <tool> '<json args>'
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

const [model, tool, args = '{}'] = process.argv.slice(2);
if (!model || !tool) {
  console.error("usage: node scripts/demo-anim/tool-output.mjs <model> <tool> '<json args>'");
  process.exit(2);
}
const transport = new StdioClientTransport({ command: 'node', args: ['dist/index.js', model] });
const client = new Client({ name: 'demo-anim', version: '0' });
await client.connect(transport);
const result = await client.callTool({ name: tool, arguments: JSON.parse(args) });
for (const part of result.content) console.log(part.text);
await client.close();
