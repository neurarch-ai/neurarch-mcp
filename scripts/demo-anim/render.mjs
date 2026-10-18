// Renders the README GIFs (docs/<name>.gif) from the transcripts in scripts.mjs.
// See README.md in this folder. Frames go to a temp dir and are deleted after encoding.
//
//   npm i --no-save @napi-rs/canvas
//   node scripts/demo-anim/render.mjs                  # all three
//   node scripts/demo-anim/render.mjs diagnose-oom     # one
//
// Needs ffmpeg on PATH. macOS-only as written (Menlo from /System/Library/Fonts).
import { createCanvas, GlobalFonts } from '@napi-rs/canvas';
import { writeFileSync, mkdtempSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { SCRIPTS, C } from './scripts.mjs';

GlobalFonts.registerFromPath('/System/Library/Fonts/Menlo.ttc', 'Mono');
const DOCS = resolve(dirname(fileURLToPath(import.meta.url)), '../../docs');
const W = 1040, SIZE = 17, LH = 25, PADX = 28, TITLE_H = 40, BODY_TOP = TITLE_H + 18;

const reveal = (segs, n) => { const o = []; let l = n; for (const [t, c] of segs) { if (l <= 0) break; o.push([t.slice(0, l), c]); l -= t.length; } return o; };
const len = (segs) => segs.reduce((s, [t]) => s + t.length, 0);

function render(name) {
  const { title, steps, rows } = SCRIPTS[name];
  const H = BODY_TOP + rows * LH + 16;
  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');
  const wOf = (t) => { ctx.font = `${SIZE}px Mono`; return ctx.measureText(t).width; };

  function draw(committed, partial) {
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = C.title; ctx.fillRect(0, 0, W, TITLE_H);
    ctx.strokeStyle = C.border; ctx.beginPath(); ctx.moveTo(0, TITLE_H + 0.5); ctx.lineTo(W, TITLE_H + 0.5); ctx.stroke();
    ['#ff5f56', '#ffbd2e', '#27c93f'].forEach((d, i) => { ctx.fillStyle = d; ctx.beginPath(); ctx.arc(20 + i * 18, TITLE_H / 2, 6, 0, Math.PI * 2); ctx.fill(); });
    ctx.fillStyle = C.dim; ctx.font = `${SIZE - 3}px Mono`; ctx.textAlign = 'center';
    ctx.fillText(title, W / 2, TITLE_H / 2 + 5); ctx.textAlign = 'left';
    const lines = committed.slice(); if (partial) lines.push(partial);
    const visible = lines.slice(Math.max(0, lines.length - rows));
    ctx.font = `${SIZE}px Mono`;
    let y = BODY_TOP + SIZE;
    for (const segs of visible) { let x = PADX; for (const [t, c] of segs) { ctx.fillStyle = c; ctx.fillText(t, x, y); x += wOf(t); } y += LH; }
    if (visible.length) {
      let x = PADX; for (const [t] of visible[visible.length - 1]) x += wOf(t);
      ctx.fillStyle = C.text; ctx.fillRect(x + 1, BODY_TOP + 3 + (visible.length - 1) * LH, 9, SIZE);
    }
  }

  const dir = mkdtempSync(join(tmpdir(), `demo-anim-${name}-`));
  try {
    let i = 0; const list = []; const committed = [];
    // One frame per distinct visual state, held for its real duration (concat demuxer).
    const emit = (partial, ms) => {
      draw(committed, partial);
      const f = `${String(i++).padStart(4, '0')}.png`;
      writeFileSync(join(dir, f), canvas.toBuffer('image/png'));
      list.push(`file '${f}'`, `duration ${(ms / 1000).toFixed(3)}`);
    };
    for (const s of steps) {
      if (s.pause) emit(null, s.pause);
      else if (s.line) { committed.push(s.line); emit(null, s.ms ?? 110); }
      else if (s.type) {
        const n = len(s.type), chunk = s.chunk ?? 3, per = (1000 / (s.cps ?? 45)) * chunk;
        for (let k = chunk; k < n; k += chunk) emit(reveal(s.type, k), per);
        committed.push(s.type); emit(null, 140);
      }
    }
    list.push(`file '${String(i - 1).padStart(4, '0')}.png'`); // concat drops the last duration otherwise
    writeFileSync(join(dir, 'list.txt'), list.join('\n') + '\n');
    const out = join(DOCS, `${name}.gif`);
    execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-f', 'concat', '-i', 'list.txt', '-vf',
      'fps=15,split[a][b];[a]palettegen=max_colors=64:reserve_transparent=0:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle',
      '-loop', '0', out], { cwd: dir, stdio: 'inherit' });
    console.log(`${out}: ${i} frames, ${W}x${H}, ${(statSync(out).size / 1024).toFixed(0)} KB`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

const names = process.argv.slice(2);
for (const n of names.length ? names : Object.keys(SCRIPTS)) {
  if (!SCRIPTS[n]) throw new Error(`unknown animation "${n}"; one of ${Object.keys(SCRIPTS).join(', ')}`);
  render(n);
}
