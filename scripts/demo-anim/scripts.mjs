// Every number below is copied from the real neurarch-mcp 0.17.1 (origin/main a23497d)
// tool output for the named model; nothing is invented for the animation.
export const C = {
  bg: '#0d1117', title: '#161b22', border: '#21262d',
  text: '#c9d1d9', dim: '#8b949e', green: '#3fb950', cyan: '#58a6ff',
  purple: '#bc8cff', white: '#f0f6fc', yellow: '#d29922', red: '#f85149',
};
const B = ['', C.text];
const you = (t) => ({ type: [['❯ ', C.dim], [t, C.white]], cps: 40 });
const claude = { line: [['● Claude', C.purple]] };
const call = (t) => ({ line: [['  ▸ ', C.dim], [t, C.cyan]], ms: 500 });
const out = (...segs) => ({ line: [['    ', C.dim], ...segs], ms: 140 });
const say = (...segs) => ({ type: segs, cps: 70 });
const blank = { line: [B], ms: 60 };
const bash = (a, b) => [
  { line: [['● ', C.purple], ['Bash', C.white], ['(' + a, C.cyan]], ms: 120 },
  { line: [['       ' + b + ')', C.cyan]], ms: 600 },
];
const hk = (...segs) => ({ line: [['    ', C.dim], ...segs], ms: 120 });

export const SCRIPTS = {
  // phi3-mini template, batch 32 on a T4, then the agent re-sizes at the batch the tool named.
  'size-training-job': {
    title: 'neurarch-mcp · size_training_job · phi3-mini (310M params)',
    rows: 15,
    steps: [
      { pause: 400 },
      you('Launch the Phi-3 fine-tune on the T4, batch 32, seq 2048.'),
      { pause: 300 }, blank, claude, { pause: 200 },
      call('size_training_job({ batch_size: 32, gpu: "T4" })'),
      out(['✗ will not fit a T4', C.red], [': needs ~30 GB with headroom (peak 26 GB)', C.text]),
      out(['peak     ', C.dim], ['26.38 GB', C.green], [' = 5.96 weights + optimizer, 20.42 activations', C.dim]),
      out(['maxBatch ', C.dim], ['T4 12', C.green], [' · L4 23 · A10G 23 · A100-40 45 · smallest fit: ', C.dim], ['L40S', C.green]),
      out(['cost     ', C.dim], ['$9.68', C.green], [' on a T4, 10 epochs over 50,000 samples (assumed)', C.dim]),
      out(['forward  ', C.dim], ['legal, no blockers', C.green]),
      { pause: 900 }, blank,
      call('size_training_job({ batch_size: 12, gpu: "T4" })'),
      out(['✓ fits a T4', C.green], [' with 15% headroom at batch 12 (peak ~14 GB of 16 GB)', C.text]),
      { pause: 700 }, blank,
      say(['Not launching at 32: it would run out of memory on the T4.', C.text]),
      say(['Batch 12 fits the T4; holding 32 takes an ', C.text], ['L40S', C.white], ['. Which do you want?', C.text]),
      { pause: 2600 },
    ],
  },

  // The OOM text a T4 prints for that same job, read against the same estimate.
  'diagnose-oom': {
    title: 'neurarch-mcp · diagnose_oom · phi3-mini, batch 32',
    rows: 15,
    steps: [
      { pause: 400 },
      you('Training died on step 1:'),
      { line: [['  torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB.', C.dim]], ms: 80 },
      { line: [['  GPU 0 has a total capacity of 14.74 GiB of which 412.06 MiB is free. ...', C.dim]], ms: 80 },
      { pause: 300 }, blank, claude, { pause: 200 },
      call('diagnose_oom({ error, batch: 32, seq_len: 2048 })'),
      out(['cause    ', C.dim], ['too-big-for-card', C.red]),
      out(['         predicted peak 26.4 GB (30.3 GB with headroom) on a 15.8 GB card', C.text]),
      out(['evidence ', C.dim], ['14.7 GB live tensors · 0.52 GB reserved but empty · asked 1.61 GB', C.dim]),
      out(['fix 1    ', C.dim], ['batch 8 x 4 gradient-accumulation steps', C.green], [' (effective batch stays 32)', C.dim]),
      out(['fix 2    ', C.dim], ['activation checkpointing', C.green], [' (activations are 20.4 GB of the peak)', C.dim]),
      { pause: 900 }, blank,
      say(['Not fragmentation: only 0.52 GB sits reserved and empty, so the allocator', C.text]),
      say(['setting will not save it. The job is bigger than the card. ', C.text], ['Batch 8 with 4', C.white]),
      say(['accumulation steps', C.white], [' keeps the run the same and fits.', C.text]),
      { pause: 2600 },
    ],
  },

  // examples/tiny-vit.py, the shipped example with a deliberate head-count bug.
  'lint-model': {
    title: 'neurarch-mcp · lint_model · examples/tiny-vit.py',
    rows: 11,
    steps: [
      { pause: 400 },
      you('Check tiny-vit.py before I queue it.'),
      { pause: 300 }, blank, claude, { pause: 200 },
      call('lint_model()'),
      out(['block ', C.red], ['head-dim-divisibility  ', C.white], ["attn: embed dim 258 not divisible by", C.text]),
      out(['      ', C.dim], ['numHeads 8 (head_dim would be 32.25)', C.text]),
      out(['warn  ', C.yellow], ['attention-no-pe        ', C.white], ['no positional encoding before attention', C.dim]),
      out(['info  ', C.dim], ['2 more                 ', C.white], ['dropout-before-bn, consecutive-linear-no-activation', C.dim]),
      { pause: 900 }, blank,
      say(['This does not reach step 1: nn.MultiheadAttention(embed_dim=258, num_heads=8)', C.text]),
      say(['fails when it is built. ', C.text], ['Use 256', C.white], [' in the seven places 258 appears.', C.text]),
      { pause: 2600 },
    ],
  },

  // The PreToolUse hook (`neurarch-mcp hook`) on the same job launched as a Bash call:
  // stderr is the hook's own, wrapped to the frame; exit 2 blocks, exit 0 lets it run.
  'hook': {
    title: 'neurarch-mcp hook · Claude Code PreToolUse · phi3-mini on a T4',
    rows: 19,
    steps: [
      { pause: 400 },
      you('Start the Phi-3 fine-tune on the T4.'),
      { pause: 300 }, blank,
      ...bash('python train.py --model src/test/templates/phi3-mini.neurarch.json', '--batch_size 32 --max_seq_length 2048 --gpu T4'),
      { line: [['  └ ', C.dim], ['PreToolUse:Bash hook blocked the call (exit 2)', C.red]], ms: 400 },
      hk(['neurarch: this training job will not fit T4.', C.white]),
      hk(['  will not fit a T4: needs ~30 GB with headroom (peak 26 GB) at batch 32,', C.text]),
      hk(['  seq 2048, use L40S or batch <= 12.', C.text]),
      hk(['  peak ~26.38 GB (30.33 GB with headroom), card T4 (16 GB).', C.dim]),
      hk(['  largest batch that fits T4: ', C.dim], ['12', C.green], [' at seq 2048.', C.dim]),
      hk(['  smallest card that holds it at batch 32: ', C.dim], ['L40S', C.green], ['.', C.dim]),
      hk(['  (2 more lines: where the job was read from, the NEURARCH_ALLOW_OOM=1 override)', C.dim]),
      { pause: 1100 }, blank,
      ...bash('python train.py --model src/test/templates/phi3-mini.neurarch.json', '--batch_size 12 --max_seq_length 2048 --gpu T4'),
      { line: [['  └ ', C.dim], ['hook exit 0: the call goes ahead', C.green]], ms: 500 },
      { pause: 500 }, blank,
      say(['The hook refused batch 32 before it took the card. Relaunched at batch 12,', C.text]),
      say(['which it let through; an ', C.text], ['L40S', C.white], [' would hold 32 if you would rather keep it.', C.text]),
      { pause: 2600 },
    ],
  },
};
