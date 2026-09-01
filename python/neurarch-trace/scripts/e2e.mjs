// End-to-end: feed a traced graph to the built MCP server and call two tools.
//   node python/neurarch-trace/scripts/e2e.mjs path/to/traced.neurarch.json
// Run from the neurarch-mcp repo root after `npm run build`.
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';

const graph = process.argv[2];
if (!graph) { console.error('usage: node e2e.mjs <traced.neurarch.json>'); process.exit(2); }
const server = resolve(dirname(fileURLToPath(import.meta.url)), '../../../dist/index.js');

const client = new Client({ name: 'neurarch-trace-e2e', version: '0.1.0' });
await client.connect(new StdioClientTransport({ command: 'node', args: [server, resolve(graph)] }));
for (const name of ['describe_architecture', 'check_design']) {
  const r = await client.callTool({ name, arguments: {} });
  console.log(`=== ${name} ===`);
  for (const c of r.content) console.log(c.type === 'text' ? c.text : JSON.stringify(c));
}
await client.close();
