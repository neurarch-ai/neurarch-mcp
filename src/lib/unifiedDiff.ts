/**
 * A unified diff between two texts, with 3 lines of context. Line-based LCS,
 * quadratic in the file length, which is fine for a model file and would not
 * be for a repository: suggest_fix hands a diff for one .py to an agent that
 * already knows how to apply one.
 */
export function unifiedDiff(path: string, before: string, after: string, context = 3): string {
  const a = before.split('\n');
  const b = after.split('\n');
  // LCS table over lines.
  const n = a.length, m = b.length;
  const dp: Uint32Array[] = [];
  for (let i = 0; i <= n; i++) dp.push(new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  type Op = { kind: ' ' | '-' | '+'; text: string; ai: number; bi: number };
  const ops: Op[] = [];
  let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) { ops.push({ kind: ' ', text: a[i], ai: i, bi: j }); i++; j++; }
    else if (i < n && (j >= m || dp[i + 1][j] >= dp[i][j + 1])) { ops.push({ kind: '-', text: a[i], ai: i, bi: j }); i++; }
    else { ops.push({ kind: '+', text: b[j], ai: i, bi: j }); j++; }
  }
  if (!ops.some(o => o.kind !== ' ')) return '';

  // Group changes into hunks separated by more than 2*context equal lines.
  const hunks: Op[][] = [];
  let current: Op[] | null = null;
  let equalRun = 0;
  for (const op of ops) {
    if (op.kind === ' ') {
      equalRun++;
      if (current) {
        current.push(op);
        if (equalRun > context * 2) {
          // Close the hunk, keeping only `context` trailing equal lines.
          const trailing = current.splice(current.length - equalRun, equalRun);
          current.push(...trailing.slice(0, context));
          hunks.push(current);
          current = null;
        }
      }
    } else {
      if (!current) {
        current = [];
        // Leading context: the last `context` equal ops before this change.
        const start = ops.indexOf(op);
        for (let k = Math.max(0, start - context); k < start; k++) if (ops[k].kind === ' ') current.push(ops[k]);
      }
      equalRun = 0;
      current.push(op);
    }
  }
  if (current) {
    if (equalRun > context) current.splice(current.length - equalRun + context, equalRun - context);
    hunks.push(current);
  }

  const out = [`--- a/${path}`, `+++ b/${path}`];
  for (const h of hunks) {
    const aStart = (h.find(o => o.kind !== '+')?.ai ?? h[0].ai) + 1;
    const bStart = (h.find(o => o.kind !== '-')?.bi ?? h[0].bi) + 1;
    const aLen = h.filter(o => o.kind !== '+').length;
    const bLen = h.filter(o => o.kind !== '-').length;
    out.push(`@@ -${aStart},${aLen} +${bStart},${bLen} @@`);
    for (const o of h) out.push(`${o.kind}${o.text}`);
  }
  return out.join('\n') + '\n';
}
