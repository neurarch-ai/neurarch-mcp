#!/usr/bin/env node
/**
 * Copy the reference architectures from a checkout of
 * github.com/neurarch-ai/awesome-llm-model-zoo into zoo/, gzipped.
 *
 * The zoo ships inside this package (about 320KB compressed for 81 graphs)
 * rather than being fetched, because "no network unless you turn one on" is
 * the promise this server makes, and a reference library that needs a socket
 * to open would be the first exception to it.
 *
 *   node scripts/sync-zoo.mjs /path/to/awesome-llm-model-zoo
 */
import { readFile, writeFile, readdir, mkdir, rm } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { join, resolve } from 'node:path';

const src = process.argv[2];
if (!src) {
  console.error('usage: node scripts/sync-zoo.mjs <path-to-model-zoo-checkout>');
  process.exit(1);
}
const root = resolve(src);
const out = resolve('zoo');
const index = JSON.parse(await readFile(join(root, 'index.json'), 'utf-8'));

await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });

const entries = [];
for (const m of index.models) {
  const dir = join(root, 'architectures', m.id);
  const names = await readdir(dir).catch(() => []);
  if (!names.includes('model.json')) {
    console.error(`skip ${m.id}: no model.json`);
    continue;
  }
  const raw = await readFile(join(dir, 'model.json'));
  const graph = JSON.parse(raw.toString('utf-8'));
  await writeFile(join(out, `${m.id}.json.gz`), gzipSync(raw, { level: 9 }));
  entries.push({
    id: m.id,
    title: m.title,
    domain: m.domain,
    type: m.type,
    params: m.params,
    layers: m.layers,
    nodes: graph.components?.length ?? m.nodes,
    attention: m.attention ?? [],
    paramCheck: m.paramCheck ?? null,
    urls: m.urls ?? {},
  });
}
entries.sort((a, b) => a.id.localeCompare(b.id));
await writeFile(join(out, 'index.json'), JSON.stringify({
  source: 'https://github.com/neurarch-ai/awesome-llm-model-zoo',
  syncedAt: new Date().toISOString().slice(0, 10),
  count: entries.length,
  architectures: entries,
}, null, 2) + '\n');
console.log(`zoo: ${entries.length} architectures written to ${out}`);
