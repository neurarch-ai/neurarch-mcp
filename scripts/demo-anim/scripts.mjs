// Every number below is copied from the real neurarch-mcp 0.17.0 (origin/main 68cf3ec)
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
      out(['✗ will not fit a T4', C.red], [': needs ~21 GB with headroom (peak 18 GB)', C.text]),
      out(['peak     ', C.dim], ['17.91 GB', C.green], [' = 6.26 weights + optimizer, 11.66 activations', C.dim]),
      out(['maxBatch ', C.dim], ['T4 21', C.green], [' · L4 40 · A10G 40 · A100-40 78 · smallest fit: ', C.dim], ['L4', C.green]),
      out(['cost     ', C.dim], ['$9.68', C.green], [' on a T4, 10 epochs over 50,000 samples (assumed)', C.dim]),
      out(['forward  ', C.dim], ['legal, no blockers', C.green]),
      { pause: 900 }, blank,
      call('size_training_job({ batch_size: 21, gpu: "T4" })'),
      out(['✓ fits a T4', C.green], [' with 13% headroom at batch 21 (peak ~14 GB of 16 GB)', C.text]),
      { pause: 700 }, blank,
      say(['Not launching at 32: it would run out of memory on the T4.', C.text]),
      say(['Batch 21 fits the T4; an ', C.text], ['L4', C.white], [' holds 32 as asked. Which do you want?', C.text]),
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
      out(['         predicted peak 17.9 GB (20.6 GB with headroom) on a 15.8 GB card', C.text]),
      out(['evidence ', C.dim], ['14.7 GB live tensors · 0.52 GB reserved but empty · asked 1.61 GB', C.dim]),
      out(['fix 1    ', C.dim], ['batch 16 x 2 gradient-accumulation steps', C.green], [' (effective batch stays 32)', C.dim]),
      out(['fix 2    ', C.dim], ['activation checkpointing', C.green], [' (activations are 11.7 GB of the peak)', C.dim]),
      { pause: 900 }, blank,
      say(['Not fragmentation: only 0.52 GB sits reserved and empty, so the allocator', C.text]),
      say(['setting will not save it. The job is bigger than the card. ', C.text], ['Batch 16 with 2', C.white]),
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
};
