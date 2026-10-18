# README animations

Renders the GIFs in the top-level README: the "See it" section's
`docs/size-training-job.gif`, `docs/size-qwen3-8b.gif`, `docs/diagnose-oom.gif`, `docs/oom-fragmentation.gif`,
`docs/lint-model.gif` and `docs/hook.gif`, and the `npx neurarch` section's `docs/data-fit.gif` and `docs/rightsize.gif`.

- `scripts.mjs` holds the transcripts: what the user types, which tool is called with which arguments, the tool's output lines, and the model's closing sentences.
- `render.mjs` draws those transcripts as terminal frames and encodes them with ffmpeg. Frames go to a temp dir that is deleted afterwards.
- `tool-output.mjs` prints a real tool result, which is where every number in `scripts.mjs` comes from.

## Regenerate

From the repo root, with ffmpeg on PATH (macOS as written; the font is Menlo):

```bash
npm run build
npm i --no-save @napi-rs/canvas          # render-only dep, kept out of package.json like scripts/record-demo.mjs
node scripts/demo-anim/render.mjs        # all of them, or name one: render.mjs diagnose-oom
```

No Playwright or browser is involved. Each GIF comes out at about 35 to 60 KB.

## The numbers must come from real tool output

Only the model's closing sentences are scripted, and the README caption says so. Every other number has to be copied from a real tool call. When a release changes the estimate, re-run these calls, update `scripts.mjs`, and update the version and commit in its header comment:

```bash
M=src/test/templates/phi3-mini.neurarch.json
node scripts/demo-anim/tool-output.mjs $M size_training_job '{"batch_size":32,"gpu":"T4"}'
node scripts/demo-anim/tool-output.mjs $M size_training_job '{"batch_size":12,"gpu":"T4"}'
node scripts/demo-anim/tool-output.mjs $M diagnose_oom '{"error":"torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB. GPU 0 has a total capacity of 14.74 GiB of which 412.06 MiB is free. Including non-PyTorch memory, this process has 14.33 GiB memory in use. Of the allocated memory 13.71 GiB is allocated by PyTorch, and 498.20 MiB is reserved by PyTorch but unallocated.","batch":32,"seq_len":2048}'
node scripts/demo-anim/tool-output.mjs examples/tiny-vit.py lint_model '{}'
```

The Qwen3-8B and fragmentation GIFs (0.17.2). An `hf:` model starts the server with `--hf`, so the first call reads `config.json` from huggingface.co; the fragmentation error text is in the comment above `'oom-fragmentation'` in `scripts.mjs`:

```bash
Q=hf:Qwen/Qwen3-8B
node scripts/demo-anim/tool-output.mjs $Q size_training_job '{"batch_size":8,"gpu":"A100-40"}'
node scripts/demo-anim/tool-output.mjs $Q size_training_job '{"batch_size":8,"gpu":"8xA100-40"}'
node scripts/demo-anim/tool-output.mjs $M diagnose_oom '{"error":"<the text in scripts.mjs>","batch":8,"seq_len":2048}'
```

`data-fit` and `rightsize` quote the sibling CLI, [`neurarch`](https://www.npmjs.com/package/neurarch), not this server:

```bash
npx neurarch bert-base-uncased --seq-len 1024 --tokenizer-size 30524
npx neurarch rightsize jobs.sample.csv      # the examples/jobs.sample.csv the neurarch package ships
```

The hook GIF quotes the hook's own stderr and exit code, for batch 32 and then the batch that fits:

```bash
for b in 32 12; do
  echo "{\"cwd\":\"$PWD\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"python train.py --model $M --batch_size $b --max_seq_length 2048 --gpu T4\"}}" \
    | node dist/index.js hook; echo "exit $?"
done
```

If the batch that fits changes (12 as of 0.17.1), change the second `size_training_job` call and the second hook launch to match. The closing sentences have to stay true of the new output too.
