// Every number below is copied from real tool output for the named model; nothing is
// invented for the animation. The first four are neurarch-mcp 0.17.1 (origin/main a23497d);
// the rest say which version and command they were captured from.
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
const sh = (t) => ({ type: [['$ ', C.green], [t, C.white]], cps: 40 });
const o = (...segs) => ({ line: [['  ', C.dim], ...segs], ms: 120 });

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
  // neurarch-mcp 0.17.2 (origin/main c417843) with --hf, hf_id "Qwen/Qwen3-8B" (8,190,878,080
  // params from config.json), seq 2048 default, SDPA default:
  //   size_training_job({ hf_id: "Qwen/Qwen3-8B", batch_size: 8, gpu: "A100-40" })
  //   size_training_job({ hf_id: "Qwen/Qwen3-8B", batch_size: 8, gpu: "8xA100-40" })
  'size-qwen3-8b': {
    title: 'neurarch-mcp · size_training_job · Qwen/Qwen3-8B from Hugging Face',
    rows: 17,
    steps: [
      { pause: 400 },
      you('Will a full fine-tune of Qwen3-8B fit one A100-40 at batch 8?'),
      { pause: 300 }, blank, claude, { pause: 200 },
      call('size_training_job({ hf_id: "Qwen/Qwen3-8B", batch_size: 8, gpu: "A100-40" })'),
      out(['✗ will not fit an A100-40', C.red], [': needs ~298 GB with headroom (peak 259 GB)', C.text]),
      out(['peak     ', C.dim], ['259.08 GB', C.green], [' = 157.26 weights + optimizer, 101.82 activations', C.dim]),
      out(['maxBatch ', C.dim], ['none', C.red], [' on T4 · L4 · A10G · L40S · A100-40 · A100-80 · H100-80', C.dim]),
      out(['         ', C.dim], ['no batch fits, weights and optimizer state alone are 157 GB', C.text]),
      { pause: 900 }, blank,
      call('size_training_job({ hf_id: "Qwen/Qwen3-8B", batch_size: 8, gpu: "8xA100-40" })'),
      out(['✓ fits 8xA100-40 only sharded', C.green], [' (FSDP / ZeRO-3): ~298 GB across 320 GB', C.text]),
      out(['✗ plain DDP', C.red], [': each A100-40 would need ~195 GB', C.text]),
      out(['sharded  ', C.dim], ['max global batch 9', C.green], [' · $258.55 for 10 epochs over 50,000 samples (assumed)', C.dim]),
      { pause: 700 }, blank,
      say(['No single card holds it: weights and optimizer state alone are 157 GB.', C.text]),
      say(['On ', C.text], ['8 A100-40s', C.white], [' it fits only sharded with FSDP or ZeRO-3, not under plain DDP.', C.text]),
      { pause: 2600 },
    ],
  },

  // neurarch-mcp 0.17.2 (origin/main c417843), phi3-mini template at batch 8 on a T4, which
  // size_training_job says fits (peak 11.06 GB). The error text is a T4-shaped example:
  //   diagnose_oom({ error: "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
  //     GPU 0 has a total capacity of 14.74 GiB of which 1.12 GiB is free. Including non-PyTorch
  //     memory, this process has 13.62 GiB memory in use. Of the allocated memory 9.84 GiB is
  //     allocated by PyTorch, and 3.46 GiB is reserved by PyTorch but unallocated. If reserved but
  //     unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  //     to avoid fragmentation.", batch: 8, seq_len: 2048 })
  'oom-fragmentation': {
    title: 'neurarch-mcp · diagnose_oom · phi3-mini, batch 8 on a T4',
    rows: 16,
    steps: [
      { pause: 400 },
      you('OOM again, and batch 8 already fits the T4:'),
      { line: [['  torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. ...', C.dim]], ms: 80 },
      { line: [['  ... 9.84 GiB is allocated by PyTorch, and 3.46 GiB is reserved by PyTorch but unallocated.', C.dim]], ms: 80 },
      { pause: 300 }, blank, claude, { pause: 200 },
      call('diagnose_oom({ error, batch: 8, seq_len: 2048 })'),
      out(['cause    ', C.dim], ['fragmentation', C.yellow]),
      out(['         Memory is fragmented: 3.72 GB is reserved and empty, more than', C.text]),
      out(['         the 2.15 GB request that failed.', C.text]),
      out(['evidence ', C.dim], ['10.6 GB live tensors · predicted peak 11.1 GB · 15.8 GB card', C.dim]),
      out(['fix 1    ', C.dim], ['PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True', C.green], [' (training unchanged)', C.dim]),
      out(['fix 2    ', C.dim], ['keep tensor shapes steady between steps', C.green], [' (training unchanged)', C.dim]),
      { pause: 900 }, blank,
      say(['The job fits the card; the allocator is what failed. Setting', C.text]),
      say(['expandable_segments', C.white], [' before the process starts keeps batch 8 as it is.', C.text]),
      { pause: 2600 },
    ],
  },

  // The sibling CLI `npx neurarch`, data-fit check from neurarch-ai/neurarch branch
  // feat/config-data-fit (not yet in the published CLI). Output trimmed to the data-fit
  // block; the sentences are the CLI's, wrapped, with "(a device-side assert on GPU)" and
  // the "Any of the 2 ids past 30521 ..." sentence left out. Exit code 1.
  //   node packages/neurarch-cli/bin/neurarch.mjs bert-base-uncased --seq-len 1024 --tokenizer-size 30524
  'data-fit': {
    title: 'npx neurarch · does the data fit the config · bert-base-uncased',
    rows: 15,
    steps: [
      { pause: 400 },
      sh('npx neurarch bert-base-uncased --seq-len 1024 --tokenizer-size 30524'),
      { pause: 500 },
      o(['bert-base-uncased  ', C.white], ['bert · fill-mask · 110.1M published, 108.9M derived', C.dim]),
      o(['checks  ', C.dim], ['clean', C.green]),
      { pause: 600 }, blank,
      o(['data fit checked: sequence length, tokenizer size', C.text]),
      { pause: 400 },
      o(['  block ', C.red], ['Sequences of 1024 tokens are longer than the 512-row position table', C.white]),
      o(['        Position ids past 511 index outside a learned table, so the first batch', C.dim]),
      o(['        that long raises an IndexError after the reservation is taken.', C.dim]),
      o(['        fix: ', C.dim], ['Truncate or chunk to 512 tokens', C.green], [', or pick a model whose context covers 1024.', C.dim]),
      { pause: 700 },
      o(['  block ', C.red], ['The tokenizer has 30524 ids and the embedding has 30522 rows', C.white]),
      o(['        The usual cause is special tokens added to the tokenizer without resizing the model.', C.dim]),
      o(['        fix: ', C.dim], ['model.resize_token_embeddings(len(tokenizer))', C.green], [' before training.', C.dim]),
      { pause: 400 }, blank,
      o(['exit 1', C.red]),
      { pause: 2800 },
    ],
  },

  // The sibling CLI `npx neurarch` 0.1.1 from npm, on the jobs.sample.csv it ships in examples/.
  // The headline and the six rows are its markdown report, columns cut to job, requested,
  // needed, verdict and right-sized request. Model configs are read from huggingface.co.
  //   npx neurarch rightsize examples/jobs.sample.csv
  'rightsize': {
    title: 'npx neurarch rightsize · requested vs needed over past jobs',
    rows: 15,
    steps: [
      { pause: 400 },
      sh('npx neurarch rightsize jobs.sample.csv'),
      { pause: 600 },
      o(['6 jobs, 174 GPU-hours, ', C.text], ['$507.85', C.white], [' spent; ', C.text], ['$328.53', C.yellow], [' of that was reserved', C.text]),
      o(['capacity no job needed, claiming nothing on 1 row marked under or unknown.', C.text]),
      { pause: 600 }, blank,
      o(['job              requested  needed   verdict                   right-sized request', C.dim]),
      o(['ft-qwen-small    8xH100     211 GB   ', C.text], ['over (borderline)         ', C.yellow], ['2xH200', C.green]),
      o(['lora-qwen-8b     A100-80     78 GB   ', C.text], ['right-sized (borderline)  ', C.green], ['as requested', C.dim]),
      o(['bert-classifier  A100-80    8.4 GB   ', C.text], ['over                      ', C.yellow], ['T4', C.green]),
      o(['tinyllama-lora   H100        35 GB   ', C.text], ['over (borderline)         ', C.yellow], ['L40S', C.green]),
      o(['phi3-sft         A100-40    230 GB   ', C.text], ['under                     ', C.red], ['-  (needs 230 GB, has 40)', C.dim]),
      o(['gpt2-pretrain    4xA100-80   29 GB   ', C.text], ['over                      ', C.yellow], ['L40S', C.green]),
      { pause: 500 }, blank,
      o(['borderline: the upper bound does not fit; measure once', C.dim]),
      { pause: 2800 },
    ],
  },
};
