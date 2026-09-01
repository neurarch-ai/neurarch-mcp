#!/usr/bin/env node
// Write scripts/tool-schemas.json: the input schema of every read tool, as
// ListTools would send it. build-mcpb.mjs copies these into the Smithery
// bundle's manifest, which requires one per listed tool.
//   npx tsx scripts/tool-schemas.mjs
import { writeFileSync } from 'node:fs';
import { listedTools } from '../src/cli.ts';
const out = Object.fromEntries(listedTools(false, 'full').map(t => [t.name, t.inputSchema]));
writeFileSync(new URL('./tool-schemas.json', import.meta.url), JSON.stringify(out, null, 2) + '\n');
console.log(`tool-schemas.json: ${Object.keys(out).length} tools`);
