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
rmSync(stage, { recursive: true, force: true });
console.log(`wrote ${out}`);
