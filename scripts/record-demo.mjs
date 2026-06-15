// Renders docs/demo.webm (and a demo-poster.png of the final frame): a scripted
// terminal transcript of an agent answering a model-structure question, with the
// numbers taken from the real neurarch-mcp tool output for examples/tiny-gpt.
//
// Not part of the published package or CI. Run it manually after installing the
// two render-only deps (kept out of package.json to keep installs lean):
//
//   npm i --no-save @napi-rs/canvas webm-writer
//   CHUNK=3 QUALITY=0.62 node scripts/record-demo.mjs docs/demo.webm
//
// Tunable via env: SCALE, FPS, QUALITY, CHUNK. macOS-only as written (it loads
// Menlo from /System/Library/Fonts); point it at any monospace .ttf elsewhere.
import { createCanvas, GlobalFonts } from '@napi-rs/canvas';
import { createRequire } from 'node:module';
import { writeFileSync } from 'node:fs';
const require = createRequire(import.meta.url);
globalThis.window = globalThis; // webm-writer feature-checks browser globals
const WebMWriter = require('webm-writer');

GlobalFonts.registerFromPath('/System/Library/Fonts/Menlo.ttc', 'Mono');

// ── config ───────────────────────────────────────────────────────────────────
const SCALE = Number(process.env.SCALE || 1);
const W = Math.round(1280 * SCALE), H = Math.round(720 * SCALE), FPS = Number(process.env.FPS || 24), FONT = 'Mono';
const QUALITY = Number(process.env.QUALITY || 0.9);
const SIZE = Math.round(22 * SCALE), LH = Math.round(32 * SCALE), PADX = Math.round(48 * SCALE), TITLE_H = Math.round(56 * SCALE), BODY_TOP = TITLE_H + Math.round(28 * SCALE);
const C = {
  bg: '#0d1117', title: '#161b22', border: '#21262d',
  text: '#c9d1d9', dim: '#8b949e', green: '#3fb950', cyan: '#58a6ff',
  purple: '#bc8cff', white: '#f0f6fc', yellow: '#d29922',
};
const out = process.argv[2] || 'demo.webm';

// ── conversation script ──────────────────────────────────────────────────────
// seg = [text, color]. Numbers/labels verified against the real server output
// for examples/tiny-gpt.neurarch.json.
const P = (t) => [[t, C.dim]];
const seg = (...s) => s;
const script = [
  { type: 'pause', ms: 500 },
  { type: 'type', cps: 42, line: seg(['❯ ', C.dim], ['Where does the parameter budget live, and is it safe to drop block_1?', C.white]) },
  { type: 'pause', ms: 450 },
  { type: 'line', line: seg(['', C.text]) },
  { type: 'line', line: seg(['● Claude', C.purple]) },
  { type: 'pause', ms: 250 },
  { type: 'line', line: seg(['  ▸ ', C.dim], ['describe_architecture()', C.cyan]) },
  { type: 'pause', ms: 350 },
  { type: 'line', line: seg(['    tiny-gpt · 8 layers · depth 7 · ', C.dim], ['57.8M', C.green], [' params · ', C.dim], ['8.27 GMACs', C.green]) },
  { type: 'line', line: seg(['    heaviest params: ', C.dim], ['token_embedding 25.7M (44%)', C.green], ['  lm_head 25.7M  block_0/1 3.1M', C.dim]) },
  { type: 'line', line: seg(['    heaviest compute: ', C.dim], ['lm_head', C.green], [', then the two decoder blocks · valid ', C.dim], ['✓', C.green]) },
  { type: 'pause', ms: 500 },
  { type: 'line', line: seg(['', C.text]) },
  { type: 'line', line: seg(['  ▸ ', C.dim], ["layer_impact({ names: ['block_1'] })", C.cyan]) },
  { type: 'pause', ms: 350 },
  { type: 'line', line: seg(['    downstream: ', C.dim], ['final_norm, lm_head', C.yellow], [' (shape-sensitive), then output', C.dim]) },
  { type: 'pause', ms: 650 },
  { type: 'line', line: seg(['', C.text]) },
  { type: 'type', cps: 50, line: seg(['The budget is ', C.text], ['~89% embedding + LM head', C.white], [', not the decoder blocks.', C.text]) },
  { type: 'type', cps: 50, line: seg(['Dropping block_1 frees only ', C.text], ['~3.1M params (~5%)', C.white], [' and is structurally safe —', C.text]) },
  { type: 'type', cps: 50, line: seg(['the wrong lever for shrinking. ', C.text], ['Quantize the embedding/head instead.', C.cyan]) },
  { type: 'pause', ms: 2200 },
];

// ── renderer ─────────────────────────────────────────────────────────────────
const canvas = createCanvas(W, H);
const ctx = canvas.getContext('2d');
ctx.textBaseline = 'alphabetic';

function segWidth(text) { ctx.font = `${SIZE}px ${FONT}`; return ctx.measureText(text).width; }

function drawFrame(committed, typing, frameIdx) {
  // background + window chrome
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = C.title; ctx.fillRect(0, 0, W, TITLE_H);
  ctx.strokeStyle = C.border; ctx.beginPath(); ctx.moveTo(0, TITLE_H + 0.5); ctx.lineTo(W, TITLE_H + 0.5); ctx.stroke();
  const dots = ['#ff5f56', '#ffbd2e', '#27c93f'];
  dots.forEach((d, i) => { ctx.fillStyle = d; ctx.beginPath(); ctx.arc(28 + i * 22, TITLE_H / 2, 7, 0, Math.PI * 2); ctx.fill(); });
  ctx.fillStyle = C.dim; ctx.font = `${SIZE - 4}px ${FONT}`; ctx.textAlign = 'center';
  ctx.fillText('neurarch-mcp · Claude Code', W / 2, TITLE_H / 2 + 6); ctx.textAlign = 'left';

  // assemble visible lines (committed + the in-progress typing line)
  const lines = committed.slice();
  if (typing) lines.push(typing.partial);

  // auto-scroll: keep the last lines in view
  const maxRows = Math.floor((H - BODY_TOP - 24) / LH);
  const start = Math.max(0, lines.length - maxRows);
  const visible = lines.slice(start);

  ctx.font = `${SIZE}px ${FONT}`;
  let y = BODY_TOP + SIZE;
  for (const segs of visible) {
    let x = PADX;
    for (const [t, color] of segs) { ctx.fillStyle = color; ctx.fillText(t, x, y); x += segWidth(t); }
    y += LH;
  }

  // blinking cursor at the end of the typing line (or the last committed line during pauses)
  const blinkOn = (frameIdx % FPS) < FPS * 0.6;
  if (blinkOn) {
    const lastSegs = typing ? typing.partial : visible[visible.length - 1];
    if (lastSegs) {
      let x = PADX; for (const [t] of lastSegs) x += segWidth(t);
      const cy = BODY_TOP + SIZE + (visible.length - 1) * LH;
      ctx.fillStyle = C.text; ctx.fillRect(x + 1, cy - SIZE + 4, 11, SIZE);
    }
  }
}

// reveal first n chars of a multi-segment line, preserving per-segment color
function revealSegs(segs, n) {
  const out = []; let left = n;
  for (const [t, c] of segs) {
    if (left <= 0) break;
    if (t.length <= left) { out.push([t, c]); left -= t.length; }
    else { out.push([t.slice(0, left), c]); left = 0; }
  }
  return out.length ? out : [['', C.text]];
}
const segsLen = (segs) => segs.reduce((s, [t]) => s + t.length, 0);

// ── timeline → frames ────────────────────────────────────────────────────────
// One encoded frame per distinct visual state, held for its real duration via
// overrideFrameDuration — so static pauses cost one frame, not FPS×seconds.
const writer = new WebMWriter({ quality: QUALITY, frameRate: FPS });
const committed = [];
let frame = 0, totalMs = 0;
const CHUNK = Number(process.env.CHUNK || 2); // characters revealed per typing frame
const show = (typing, durationMs) => {
  drawFrame(committed, typing, frame);
  const d = Math.max(1, Math.round(durationMs));
  writer.addFrame(canvas, null, d);
  frame++; totalMs += d;
};

for (const step of script) {
  if (step.type === 'pause') {
    show(null, step.ms);
  } else if (step.type === 'line') {
    committed.push(step.line);
    show(null, 130);
  } else if (step.type === 'type') {
    const total = segsLen(step.line);
    const perChar = 1000 / step.cps;
    for (let shown = CHUNK; shown < total; shown += CHUNK) {
      show({ partial: revealSegs(step.line, shown) }, perChar * CHUNK);
    }
    committed.push(step.line);
    show(null, 160);
  }
}

// snapshot the final composited frame for visual QA
drawFrame(committed, null, 0);
writeFileSync(out.replace(/\.webm$/, '.png'), canvas.toBuffer('image/png'));

const blob = await writer.complete();
const buf = (blob && blob.arrayBuffer) ? Buffer.from(await blob.arrayBuffer()) : Buffer.from(blob);
writeFileSync(out, buf);
console.log(`wrote ${out}: ${(buf.length / 1024).toFixed(0)} KB, ${frame} frames, ${(totalMs / 1000).toFixed(1)}s playback`);
