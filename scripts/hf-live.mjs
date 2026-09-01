// One real call through the built server with --hf, against a public repo.
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
const transport = new StdioClientTransport({ command: 'node', args: ['dist/index.js', 'examples/tiny-gpt.neurarch.json', '--hf'] });
const client = new Client({ name: 'live', version: '0' }, { capabilities: {} });
await client.connect(transport);
const t = await client.listTools();
console.log('tools:', t.tools.length, t.tools.some(x => x.name === 'load_hf_model'));
const r = await client.callTool({ name: 'load_hf_model', arguments: { model_id: process.argv[2] ?? 'Qwen/Qwen2.5-0.5B' } });
const s = r.structuredContent;
console.log(r.isError ? r.content[0].text : JSON.stringify({ configSource: s.configSource, realParamCount: s.realParamCount, layers: s.layerCount, params: s.totalParametersFormatted ?? s.totalParameters, cached: s.cached }, null, 1));
const r2 = await client.callTool({ name: 'lint_model', arguments: { model_path: 'hf:' + (process.argv[2] ?? 'Qwen/Qwen2.5-0.5B') } });
console.log('lint via hf: ref:', r2.isError ? r2.content[0].text : JSON.stringify(r2.structuredContent.counts));
await client.close();
