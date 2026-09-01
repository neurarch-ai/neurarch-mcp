#!/usr/bin/env node
/**
 * Pack the Claude Desktop one-click bundle (.mcpb): manifest + built server +
 * the vendored engine and zoo, with the only runtime dependency installed.
 *
 *   npm run build && node scripts/build-mcpb.mjs   ->  neurarch-mcp.mcpb
 *
 * Uses the official packer when it is installed (npx @anthropic-ai/mcpb pack),
 * which also validates the manifest; falls back to a plain zip of the same
 * files so the bundle can be built offline.
 */
import { spawnSync } from 'node:child_process';
import { mkdtempSync, cpSync, readFileSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const root = resolve('.');
const manifest = JSON.parse(readFileSync(join(root, 'manifest.json'), 'utf-8'));
const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf-8'));
if (manifest.version !== pkg.version) {
  console.error(`manifest.json says ${manifest.version}, package.json says ${pkg.version}; they must match.`);
  process.exit(1);
}
if (!existsSync(join(root, 'dist', 'index.js'))) {
  console.error('dist/index.js missing: run npm run build first.');
  process.exit(1);
}

const stage = mkdtempSync(join(tmpdir(), 'neurarch-mcpb-'));
for (const f of ['manifest.json', 'package.json', 'README.md', 'LICENSE', 'dist', 'zoo']) {
  cpSync(join(root, f), join(stage, f), { recursive: true });
}
// The bundle carries its own node_modules: Claude Desktop does not npm install.
writeFileSync(join(stage, 'package.json'), JSON.stringify({ ...pkg, devDependencies: {}, scripts: {} }, null, 2));
const install = spawnSync('npm', ['install', '--omit=dev', '--ignore-scripts', '--no-audit', '--no-fund'], { cwd: stage, stdio: 'inherit' });
if (install.status !== 0) process.exit(install.status ?? 1);

const out = join(root, `neurarch-mcp-${pkg.version}.mcpb`);
const official = spawnSync('npx', ['--yes', '@anthropic-ai/mcpb', 'pack', stage, out], { stdio: 'inherit' });
if (official.status !== 0) {
  console.error('official packer unavailable; falling back to zip');
  rmSync(out, { force: true });
  const zip = spawnSync('zip', ['-qr', out, '.'], { cwd: stage, stdio: 'inherit' });
  if (zip.status !== 0) process.exit(zip.status ?? 1);
}
// Smithery reads the same bundle but validates the manifest more strictly
// than Anthropic's packer: prompt arguments must be objects and every tool
// needs an inputSchema. Emit a second bundle with those two edits rather
// than fork the manifest, so the two never drift.
//   npx @smithery/cli mcp publish neurarch-mcp-<v>.smithery.mcpb -n neurarch-ai/neurarch-mcp
const smitheryManifest = JSON.parse(readFileSync(join(stage, 'manifest.json'), 'utf-8'));
for (const p of smitheryManifest.prompts ?? []) {
  if (Array.isArray(p.arguments)) {
    p.arguments = p.arguments.map((a) => (typeof a === 'string' ? { name: a, description: a, required: true } : a));
  }
}
// Every tool, not the eight headline ones: Smithery scores a listing on what
// the manifest declares, and the short descriptions are what ListTools sends.
const schemas = JSON.parse(readFileSync(join(root, 'scripts', 'tool-schemas.json'), 'utf-8'));
const shorts = JSON.parse(readFileSync(join(root, 'scripts', 'tool-short.json'), 'utf-8'));
smitheryManifest.tools = Object.keys(schemas).map((name) => ({
  name,
  description: shorts[name] ?? (smitheryManifest.tools ?? []).find((t) => t.name === name)?.description ?? name,
  inputSchema: schemas[name],
}));
writeFileSync(join(stage, 'manifest.json'), JSON.stringify(smitheryManifest, null, 2));
const outSmithery = join(root, `neurarch-mcp-${pkg.version}.smithery.mcpb`);
rmSync(outSmithery, { force: true });
const zip2 = spawnSync('zip', ['-qr', outSmithery, '.'], { cwd: stage, stdio: 'inherit' });
if (zip2.status !== 0) process.exit(zip2.status ?? 1);

rmSync(stage, { recursive: true, force: true });
console.log(`wrote ${out}`);
console.log(`wrote ${outSmithery}`);
