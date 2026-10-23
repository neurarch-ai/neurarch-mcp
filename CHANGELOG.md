# Changelog

All notable changes to `neurarch-mcp` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.21.1] - 2026-10-22

### Fixed
- **Vendored engine, verifier and plan bundles refreshed** from neurarch main
  (after #399), so the tools carry these fixes:
  - **Derived max length per engine and release** (neurarch #386): a
    `deepseek_yarn` (and similar) rope config no longer multiplies
    `max_position_embeddings` by the factor on vLLM 0.30+; sarvam-105b was
    reported as 5,242,880 tokens. vLLM below 0.30 and SGLang keep their own rules.
  - **ModelOpt checkpoints** (neurarch #393): `hf_quant_config.json` is read
    and merged the way vLLM does, so NVIDIA FP8 / NVFP4 checkpoints are no
    longer sized at 16 bits, and `--kv-cache-dtype auto` resolves to the
    checkpoint's fp8 KV cache.
  - **Vision tower TP** (neurarch #388): `tp-vision-heads` blocks a TP size
    that does not divide the vision encoder's heads (Qwen2.5-VL, Qwen3-VL,
    SigLIP), unless `--mm-encoder-tp-mode data`.
  - **Chat template portability** (neurarch #384): `template-undefined-call`
    warns when a template calls a macro it never defines.
  - **YaRN on vLLM 0.30+**: following vllm-project/vllm#56446, the derived
    length is `max_position_embeddings`, never multiplied by the YaRN factor.
    Adding only `rope_scaling` to Qwen3-8B as its model card shows now refuses
    131072; the fix names raising `max_position_embeddings` in the same override.
  - **The vendored verifier carries no network call again** (neurarch #400):
    the Data stage's dataset-metadata fetch had been bundled in.
  - **LoRA change specs say what they assumed**: the batch and sequence length
    a change spec was sized at are carried in the result.

### Added
- `model_changes` returns a `lineage` field: where a derivative's model card
  disagrees with the base it declares (#50).
- First `eval:tools` baseline committed (#49).

## [0.21.0] - 2026-10-19

> The v2 SDK entries below shipped in the 0.21.0 package (its dependencies are `@modelcontextprotocol/server` 2.3.1 and `@modelcontextprotocol/node` 2.1.1) but were left under Unreleased; moved here on 2026-10-22.


### Added
- **`--toolsets=GROUP[,GROUP]`** advertises only the named groups: `sizing`,
  `parallel`, `serving`, `changes`, `design`, `models`, `write`. `--toolsets=sizing`
  lists six tools (about 1.6k tokens a turn against 5.5k for the core set).
  `write` is the same as `--write`, with the same startup refusals; an unknown
  group name stops the server at startup. Every read tool stays callable by
  name. The default (`--tools=core`) is unchanged.
- **`--read-only`**: no write tools, whatever `--write` or `--toolsets=write` asked for.
- **`response_format: "concise" | "detailed"`** on `lint_model`, `find_layers`,
  `layer_impact`, `describe_architecture` and `check_design` (where `detailed`
  is the old `verbose: true`, which still works). Concise is the default. It
  always says what it left out: `omitted.count` with what and how to get it,
  or `fields` for find_layers. On qwen2.5-7b, `lint_model` goes from 10.6 KB to
  1.4 KB: info findings are counted per rule in `infoByRule` instead of
  listed, and `counts` still counts every finding.
- **Server icon**: the Neurarch mark as a 64 px PNG data URI in
  `serverInfo.icons`, rasterised from the site's `icon-n.svg`; `icon.png` in
  the MCPB bundle.
- **`npm run eval:tools`** is now a first-call tool-selection eval: 42 asks that
  never name a tool, graded on the first tool a model calls against the
  listing the built server actually sends. It estimates the cost before
  spending and refuses above `--max-usd` (default $2). The multi-turn Claude
  Code eval moved to `eval:tools:agent`.

### Changed
- The core listing is about 5.5k tokens a turn (was 5.4k) and the full one 7.8k
  (was 7.7k): the new `response_format` argument on five tools.


### Changed (v2 SDK)
- **On the v2 MCP SDK** (`@modelcontextprotocol/server` and `client` 2.3.1, plus
  `@modelcontextprotocol/node` 2.1.1 for the HTTP adapter), so the server speaks
  protocol revision **2026-07-28** as well as the 2025 revisions it already
  spoke. v1 (`@modelcontextprotocol/sdk`) only implements up to 2025-11-25 and
  is maintained until about 2027-01.
- **stdio** is `serveStdio`: the revision is negotiated per connection, from
  the same factory for both eras.
- **HTTP** routes by era with the SDK's own classifier. 2026-07-28 requests are
  stateless, one server instance per request (`createMcpHandler`). 2025-era
  requests keep the session flow they had (`Mcp-Session-Id`, GET for the SSE
  stream, DELETE to close), because a per-request instance cannot know that a
  2025 client declared elicitation at initialize. The bearer-token and Host
  checks sit in front of both paths, unchanged.
- **Elicitation** (`check_design` with `ask_user`): a 2025 connection gets the
  pushed `elicitation/create` as before; a 2026-07-28 request gets
  `input_required`, and the client retries the call with the answer. No
  `requestState` is minted, because the retry recomputes the same decision from
  the graph and nothing from the first round is carried.

### Added (v2 SDK)
- `src/eras.test.ts`: the same calls through the real HTTP server under a
  2025 client and a client pinned to 2026-07-28, compared result for result,
  plus elicitation (accept and decline), progress and cancellation on 2026.

## [0.20.0] - 2026-10-19

### Added
- **`check_pins`: Hugging Face downloads not pinned to a commit.** Reads
  Python (`from_pretrained`, `load_dataset`, `hf_hub_download`,
  `snapshot_download`, `pipeline`), notebooks, shell scripts and Dockerfiles
  (`vllm serve`, the SGLang server, `huggingface-cli` / `hf download`) and YAML
  model keys, with the fix spelled for each file. The scanner is the CLI's
  `neurarch pin-check` (neurarch-ai/neurarch #346, aligned with Bandit B615),
  vendored as `src/vendor/policy.bundle.mjs`. Offline; paged.
- **`model_changes`: what changed upstream in a tracked Hugging Face model**,
  revision by revision (config, tokenizer, chat template, generation defaults,
  weights), whether each change is breaking under the published rule, and with
  `since_revision` only what landed after your pin. Reads two public files from
  the change record at www.neurarch.com/changes-feeds/ (neurarch-ai/neurarch
  #341); no key, nothing from your repository. The third tool that opens a
  socket, annotated open-world. `detail: "full"` adds each change's keys.
- **Prompts** `precheck_change`, `precheck_serving` and `pin_downloads`.
- **Resource** `neurarch://rules/catalogue`: every rule id a tool here can
  report (lint, shape, edit, pin) with its source and, where the verifier has
  one, the measurement behind it.
- **Server instructions** at initialize: which tool answers which moment, and
  that memory figures are estimates and times are predictions. Server `title`
  and `websiteUrl`; every tool has a top-level `title`.
- **Completions** (`completion/complete`) for zoo ids, tool names, rule ids and
  card ids, on prompt arguments and resource templates.
- **Progress notifications and cancellation.** `trace_model` reports elapsed
  seconds against its timeout and is killed on cancel; `plan`, `history` and
  `model_changes` abort their request on cancel; `check_pins` reports files
  read on large trees.
- **`resource_link` results**: `list_architectures` links each entry to
  `neurarch://zoo/<id>`.

### Changed
- **Output schemas on every tool whose result shape is stable** (22 tools, up
  from 14), each naming only keys every code path returns. A new test calls
  every schema'd tool through the SDK client, which validates
  `structuredContent` against the schema, on four models.
- **Arguments are checked before a handler runs**: a missing required argument,
  an unknown argument, a wrong type or a value outside an enum now comes back
  as an error naming the argument and what is accepted. `rank_designs {}` and
  `load_architecture {}` used to answer "Cannot read properties of undefined".
  An unknown tool name suggests the tools it was probably reaching for.
- **Errors are errors**: a handler result carrying `error` is returned with
  `isError: true`, and `get_layer`, `compare_layers` and `get_block` report a
  name that matches nothing as an error with the closest names, instead of
  `null`.
- **Long lists are paged** with an opaque `nextCursor`: `list_architectures`
  (20 per page, was every entry, 41 KB), `find_layers` (50, was 100),
  `list_connections` (100, was 200), `check_pins`, `model_changes`.
  `list_architectures` keeps paper links and the parameter check behind
  `detail: "full"`.
- **Under `--hf` every read tool is annotated open-world**, since any of them
  can be handed an `hf:` model; without `--hf` they stay closed-world.
- The core listing swaps `mermaid_diagram` and `export_pytorch` (both also
  served as resources, both still callable) for `check_pins` and
  `model_changes`. It costs about 5.3k tokens a turn (was 4.7k), the full
  listing 7.6k (was 6.6k), almost all of it the output schemas.
- `@modelcontextprotocol/sdk` floor raised to `^1.29.0`, the version tested;
  the MCPB manifest moves to `manifest_version` 0.3.
- **A tool that declares an output schema can no longer drop
  `structuredContent`.** If such a handler ever returns something other than
  an object, the call fails with an internal error instead of a result a
  validating client has to reject.
- `size_training_job` and `check_serving_config` descriptions now say what
  they do not answer: not a convergence check; static, starts no server.
- Vendored engine, verifier and plan bundles rebuilt from neurarch-ai/neurarch
  `main` at 1ffaa2d8 (`npm run check:mcp-vendor` in sync).

## [0.19.0] - 2026-10-18

### Changed
- **`check_parallel_layout` now times each stage per card and simulates the
  pipeline schedule.** Matmul efficiency comes from a per-card table fitted on
  public GEMM measurements instead of a flat 40% of peak, and the 1F1B step is
  simulated op by op (a port of Primus's simulator) instead of the closed-form
  formula, so `predicted.formula` now reads `1F1B schedule simulated op by op`.
  On the 2 x H100 Qwen3-4B pipeline A/B the predicted gain is within 1.5 points
  of the measured one (neurarch-ai/neurarch #335); absolute step time is less
  reliable than the gain. Recommended splits move
  accordingly: Qwen3-8B at TP4 PP4 on H100 now recommends `10,10,9,7`.
- **`check_serving_config` sizes memory with each engine's own arithmetic**:
  vLLM's KV cache groups and blocks, SGLang's `mem_fraction_static` formula,
  llama.cpp's sliding-window cells, plus per-engine quantisation support
  tables. On 21 public vLLM startup logs (out of sample) the KV token count is
  within 1.07% (neurarch-ai/neurarch #337). `basis` now starts "Predicted from
  config.json" rather than "Derived from config.json".
- **`check_serving_config` prices the startup terms vLLM measures**: the CUDA
  graph pool from vLLM's own capture-size list and estimate (against the
  figure vLLM printed on 10 held-out public logs, mean absolute error 0.38
  GiB), and the vision encoder's profiling pass for gemma-3 and Qwen-VL
  (vision-language launches end to end on 10 public logs, median 5.96%)
  (neurarch-ai/neurarch #344).

### Fixed
- Multimodal configs are read as their language model: the vision tower's
  position table, head count and FFN no longer block TP or context checks
  (InternVL, gemma-3, Qwen-Omni). fp32 configs follow each engine's own
  dtype default. Derived weights are checked against the checkpoint's
  published `total_size` and the published size is used past a 10% gap
  (MXFP4, 4-bit and MLA checkpoints). New cards: A40, GH200 96/144 GB, RTX
  PRO 6000 Blackwell, Ascend 910B (memory only) (neurarch-ai/neurarch #348).

## [0.18.0] - 2026-10-18

### Added
- **`check_parallel_layout`: is this multi-GPU split legal, balanced and worth
  running, before it takes a reservation.** Inputs: `micro_batch`, `seq_len`,
  and optionally `tp`, `pp`, `dp`, `cp`, `ep`, `global_batch` or `grad_accum`,
  `gpu`, `schedule` (`1f1b` / `interleaved`) with `virtual_stages`,
  `layers_per_stage`, `zero_stage`, `activation_checkpointing` and a
  `seq_len_histogram`. Returns `legal`, the findings (TP against heads, KV
  heads, hidden and FFN width; EP against experts; world size; batch
  arithmetic; stage memory), a per-stage table, and a `predicted` block: the
  step slowdown against a balanced split through the 1F1B formula
  `m x slowest + (sum - slowest) / v` at the layout's own microbatch count, the
  slowest stage, the contiguous split that minimises it, the bubble, and with a
  histogram the DP sequence imbalance. Every time is predicted from the graph
  (matmul FLOPs at 40% of peak plus bandwidth-aware memory-bound bytes, fp32
  logits on the loss stage), never measured, and the result says so. The check
  is the app's `checkParallelLayout` (neurarch-ai/neurarch #319, #323, #325).
- **`check_serving_config`: will this vLLM, SGLang or llama.cpp launch start,
  hold what it was asked to hold, and serve the model as published.** Pass the
  launch line (`command`) or parsed `args`, with `gpu`, `gpus` and optionally
  `against` (the model it replaces). It reads `config.json`,
  `tokenizer_config.json`, `chat_template.jinja` and `generation_config.json`
  (from a checkpoint directory, or an `hf:` id through the existing `--hf`
  path, each file cached for a day and served from the cache without `--hf`)
  and returns findings on parallel legality, served length against the derived
  context, weights plus KV memory and the concurrent full-length sequences that
  fit (an upper bound), quantization against the card, and the chat template
  and sampler defaults. The check is the app's `checkServeConfig`
  (neurarch-ai/neurarch #322).
- **The hook pre-checks vLLM and SGLang launches.** `vllm serve`,
  `python -m vllm.entrypoints...` and `python -m sglang.launch_server` run
  through the same `checkServeConfig`, and the hook blocks only on a `block`
  finding read from the model's own `config.json` and the flags (the card is
  not on a serve line, so memory and quant support never block here). A family
  template config never blocks, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and
  `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` clear the context finding, and
  `NEURARCH_ALLOW_SERVE=1` lets a launch through. It still fails open.
- Tests for both tools and the hook on the real Qwen3-8B `config.json`,
  `tokenizer_config.json` and `generation_config.json`
  (`src/test/hf/Qwen3-8B`, revision b968826d), served by a `fetch` stub, so the
  `--hf` path runs end to end with no socket.

- **A Claude Code plugin.** `/plugin marketplace add neurarch-ai/neurarch-mcp`
  then `/plugin install neurarch@neurarch-ai` installs the server
  (`npx -y neurarch-mcp --hosted --hf`), the PreToolUse launch hook
  (`hooks/hooks.json`, the same `npx -y neurarch-mcp hook --hf` the README
  configures by hand) and `skills/neurarch-skill`. This repository is the
  marketplace (`.claude-plugin/marketplace.json`); `claude plugin validate .
  --strict` passes. `plugin.json` carries the package version and
  `src/plugin.test.ts` fails a release that bumps one without the other.

### Changed
- **The vendored plan bundle** is rebuilt from neurarch-ai/neurarch at eea7a68c,
  adding `checkParallelLayout`, `checkServeConfig` and their readers, and the
  data-fit fixes of #315. The engine and verifier bundles are unchanged
  (`npm run check:mcp-vendor` reports all three in sync).
- **Tool listing kept under its cap, not raised.** The two new tools cost ~460
  tokens; property descriptions across the read tools were shortened, the
  output schemas dropped `additionalProperties: true` (the JSON Schema
  default), and four short descriptions were tightened. The core listing is
  ~4.72k tokens (cap 4.8k) and the full one ~6.63k (cap 6.7k), both with more
  headroom than 0.17.3 had.

## [0.17.3] - 2026-10-18

### Added
- **`size_training_job` answers fits / borderline / does not fit per card.** The
  vendored plan bundle now carries neurarch-ai/neurarch #312's conservative
  peak: the estimate x 1.43, at or above the measured peak on 95% of 91
  held-out runs. Every card gets the plan's own `fitVerdict` on this job's peak
  (`gpuVerdict`), `upperBound` says what the bound was read from, and
  `smallestSafeGpu` is the cheapest card that holds it with 1.15x headroom,
  beside `smallestFitGpu` (the point estimate) and `maxBatchSafe` beside
  `maxBatch`. A card that holds only the point estimate is `borderline`: the
  verdict line says so in the CLI's words ("measure once (one step at this
  batch) before reserving") and names the card that holds the upper bound. An
  N-card request reports the bound for the DDP or sharded reading that decided
  it. `fits` is unchanged (point estimate x 1.15).
- **Data fit, on `size_training_job` and `check_design`.** An optional `data`
  argument (`seq_len`, `tokenizer_size`, `num_classes`, `image_size`,
  `channels`) returns `dataFit`: the checks that ran, the ones skipped and why,
  and findings for a sequence past a learned position table, a tokenizer
  larger than the embedding, labels past the classification head, and an image
  size or channel count the first layer was not built for. These are the
  bundle's own `inputContractOf` / `checkDataFit` (neurarch-ai/neurarch #314),
  the functions behind `npx neurarch <id> --seq-len ... --tokenizer-size ...`
  and the `data` field of `POST /api/v1/check`. For an `hf:` model the label
  count comes from the config's classification head (`architectures` +
  `id2label`), as the CLI's `dataFitHFModel` reads it; `check_design` takes a
  `config` object for any other graph. As with the CLI's `--seq-len`,
  `size_training_job`'s `seq_len` is checked against the position scheme
  whenever it is given, and a blocking finding leads the verdict line.

### Changed
- The Hugging Face config cache keeps the `config.json` a graph was built
  from, which is where the label count comes from. An entry written before
  0.17.3 is still served offline as it is; its label check reports itself
  skipped.
- Vendored `plan.bundle.mjs` rebuilt from neurarch-ai/neurarch 0a440c73
  (`engine` and `verifier` bundles unchanged).
- The `check_design` property descriptions were shortened to keep the tool
  listing inside its token budget.

## [0.17.2] - 2026-10-17

### Changed
- **`neurarch-mcp hook` reads the `.neurarch.yml` `job:` block neurarch-bot reads** (its
  `scripts/job.py`): a top-level `job:` and a model entry's own `job:` merged
  over it key by key; `gpu`, `batch` (per step), `seq_len`, `grad_accum`,
  `attention`, `epochs`, `samples`, each a value or a `file:key` pointer into a
  YAML or JSON training config read relative to the repository root, dotted
  keys allowed; unknown keys ignored. When the job does not fit and
  `grad_accum` is set, the reason names the batch and accumulation steps that
  keep the effective batch, by the bot's own split. `model` is the hook's only
  extra key: the bot reads the model from its traced file list. The tests run
  on the bot's `examples/.neurarch.yml`, copied verbatim, and the resolved job
  matches what the bot's `resolve_job` reads from the same files.
  0.17.1's hook read a `job:` block of its own (top level only, literal
  values, `batch` as a global batch); this replaces it. 0.17.1 was never
  published, so no released version reads the old shape.

## [0.17.1] - 2026-10-17

### Added
- **`check_edit`**, the tool to call after editing a model's config or
  architecture and before training. Given the base (`base_model_path`, or
  `base_config`) and the changed model (`model_path`, `model_source`, or
  `config`; a Hugging Face `config.json` as an object, JSON text or a path),
  it returns the edits that load, run and are wrong, each under a stable rule
  id with the public upstream reports of it: `vocab-resize-incomplete`,
  `weight-tying-lost`, `attention-config-drift`, `rope-context-drift`, and,
  with `lora_target_modules`, `lora-target-no-match`, `lora-target-partial`
  and `lora-target-wrapper`; config divisibility comes back under the lint ids
  `head-dim-divisibility` and `gqa-head-divisibility`. Then the memory delta:
  the training peak before and after at `batch` (default 32) and `seq_len`,
  and whether the job still fits `gpu`. No logic of its own: the checks are
  the app's `checkGraphEdit`, `checkConfigEdit` and `checkLoraTargets`, the
  memory is `size_training_job`'s, and a config with no graph is read by the
  engine's `convertHFConfigToModel`. In the core listing after `diagnose_oom`.
- **`neurarch-mcp hook`**, a Claude Code PreToolUse hook. It reads the hook
  JSON on stdin; when a Bash command launches training (`python train.py`,
  `accelerate launch`, `torchrun`, `deepspeed`, `sky launch`) and the model,
  batch and card can be read off the command line or a `job:` block in the
  nearest `.neurarch.yml`, it sizes the job as `size_training_job` does and
  exits 2 with the peak, the card, the largest batch that fits and the
  smallest card that holds it when the job will not fit the declared card.
  Every other case exits 0 and prints nothing. It fails open: an unrecognised
  command, a missing card or batch, a model it cannot load, an ambiguous card,
  or any error lets the call run, and so do jobs the estimate does not model
  (gradient checkpointing, LoRA / PEFT, 4- and 8-bit loading, CPU offload).
  `NEURARCH_ALLOW_OOM=1` in the command lets a refused launch through. The
  `.claude/settings.json` snippet is in the README.

### Changed
- **The memory estimate is the six-term refit** (neurarch-ai/neurarch #306,
  vendored from `a1bac74e`): 1.2 x 16 bytes per parameter, plus, per sample in
  bf16, 0.88 x every layer output, 3.64 x stored attention scores (eager
  only), 1.34 x each layer's weight-matrix intermediates, 5.2 x recurrent gate
  states and 2.02 x the largest single tensor. Fitted on 178 measured runs
  across 41 architectures with out-of-memory runs as constraints; held out one
  architecture at a time, the median error is 7% above 1 GB and it flagged
  113 of the 131 times a run did not fit a card, where accelerate
  estimate-memory flagged at most 8; on 25 runs measured after the fit, at the
  edge of a 24 GB card, the median miss is 14%. Numbers move with it: phi3-mini
  at batch 32 under SDPA now peaks at 26 GB (was 18), and its smallest card is
  an L40S (was an L4).
- **The accuracy sentence cannot drift from the bundle.** `size_training_job`'s
  sentence is written from the bundle's six factors and its
  `calibrationSentence()` (from `MEMORY_CALIBRATION`); the tool description
  reads the same object; the fit headroom and default sequence length are the
  bundle's `FIT_HEADROOM` and `DEFAULT_SEQ_LEN`. A test holds the README,
  package.json, manifest.json, AGENTS.md and the skill's tool reference to
  `MEMORY_CALIBRATION`, so a refit fails CI until every copy moves.
- The `model_path` description injected on every read tool is shorter
  ("Path, zoo:<id> or hf:<org/name>."), which is what pays for `check_edit`
  under the tool-list token caps.
- The engine, verifier and plan bundles are refreshed from neurarch-ai/neurarch
  `a1bac74e`; `plan.bundle.d.mts` declares the new exports (the six factors,
  `MEMORY_CALIBRATION`, `calibrationSentence`, the edit checks and their rule
  ids). `plan.contract.test.ts` pins the six-term numbers, each printed by the
  app's own TypeScript source and by this bundle, which agree to the last digit.

### Removed
- The caveat that recurrent cells are under-counted by 70 to 80%: that was the
  three-factor fit's miss, and the six-term model counts gate states.

## [0.17.0] - 2026-10-17

### Added
- **`size_training_job`**, the tool to call before a job takes a GPU
  reservation. Given the model, a batch size and optionally the card
  (`"A100-40"`, `"8xH100"`), the sequence length and the attention
  implementation, it returns the training peak with its weights-and-optimizer,
  activation and attention-score parts, the smallest card that holds it, the
  largest batch on each of seven cards, a two-card split when one card does
  not, one run's cost, and a verdict line (`will not fit a T4: needs ~21 GB
  with headroom (peak 18 GB) at batch 32, seq 2048, use L4 or batch <= 21`).
  It is a wrapper over the app's own `buildPlan`, vendored as
  `src/vendor/plan.bundle.mjs` (a byte-for-byte copy of the app's
  `lib/modelPlan.bundle.mjs`), so the number is the one the CI bot and
  `POST /api/v1/plan` print, computed offline. That number is the 2026-10-17
  memory refit: 141 measured runs across 44 architectures, held-out median
  error 8% above 1 GB, fits / does-not-fit verdict agreeing with measurement
  98% of the time. It adds two things the plan does not: the stored attention
  scores are removed under SDPA (the default) and flash, and an N-card request
  is read both as plain DDP and as sharded. A graph that feeds a hidden-state
  tensor into a token embedding (several bundled LLM references declare their
  input as `[1, L, D]`) is refused rather than sized at terabytes.
- **`diagnose_oom`**, the tool to call after a CUDA out-of-memory error,
  before changing batch size. Given the error text as printed (PyTorch 1.x or
  2.x wording) and, optionally, the model (`model_path`, the server's, or
  `hf_id`), `batch`, `seq_len`, `step`, `attention`, and `inference` with
  `gen_tokens` for a generation job, it reads the allocated,
  reserved-but-empty, other-process and failing-request numbers out of the
  message, compares them with the predicted peak `size_training_job` uses, and
  returns one cause (`too-big-for-card`, `fragmentation`, `other-process`,
  `large-logits`, `undeclared-memory`, `grows-during-run`, `unknown`), the
  evidence lines it was read from, fixes ordered least invasive first with a
  `changesTraining` flag on each, and the job facts used. A smaller batch is
  offered only when the job is too big for the card, and then with the
  gradient accumulation that keeps the effective batch. The logic is the app's
  `npx neurarch oom` (`diagnoseOom`, `oomFactsForModel`), carried in the
  vendored plan bundle since neurarch-ai/neurarch #297; this server adds no
  arithmetic of its own. In the core listing after `history`.
- **A cached Hugging Face config is served without `--hf`.** `hf_id` and
  `hf:<org/name>` read `~/.cache/neurarch-mcp/hf` (or `NEURARCH_MCP_CACHE`)
  first, at any age, because reading a file is not the network call the flag
  guards; only a fetch needs the flag. With `--hf`, a stale cached config now
  beats a failed fetch or a family template.

### Changed
- **Sizing leads.** `size_training_job` is first in the core listing and in
  the README, AGENTS.md, the skill and the `pre_train_checklist` prompt; the
  structural checks follow it as "and whether it forward-passes".
  `rank_designs` left the core listing (still callable by name, and listed
  under `--tools=full`), and several schema descriptions were trimmed so the
  listing stays under its token caps. For `diagnose_oom` the injected
  `model_path` description and the `add_layer` schema were shortened.
- **Vendored `engine`, `verifier` and `plan` bundles refreshed** from
  neurarch-ai/neurarch `96254170`, and the three again from the #297 branch
  (on `814c98a2`) for `diagnose_oom`. What moved with them: the plan assumes SDPA
  attention by default and counts the eager attention-score term only under
  `attention: 'eager'` (neurarch #285); a token graph is sized at 2048 tokens
  unless a sequence length is given, instead of at its declared context
  (#286); the ConvNeXt, DINOv2, Swin and BEiT config readers are fixed (#286);
  and right-sizing stops pooling memory across small cards, so a job is
  matched to one card that holds it (#290). `size_training_job` numbers are
  unchanged by this: it already sized SDPA by default and already defaulted
  to 2048 tokens.
- **`plan` takes `seq_len`** (#28). Optional; when given it rides in the POST
  body to `/api/v1/plan`, and when omitted the server sizes a token graph at
  2048 tokens.
- **`NEURARCH_TRACE=1 python train.py`** (neurarch-trace 0.2.0). No target, no
  `--input`, no edit to the script. The install puts one line at the root of
  site-packages, Python runs it before the first line of any script in the
  environment, and with the variable set it waits for torch and records the
  first forward pass the process runs. A real training entry point rarely has a
  model you can name on a command line, because it sits behind a config, a
  registry, a `from_pretrained` and an `if args.variant ==`; the running process
  has already resolved all of that. With `NEURARCH_TRACE_PLAN=1
  NEURARCH_TRACE_FAIL_ON_BLOCK=1` it becomes a gate: a design that will not
  forward-pass stops the process after one step instead of at the shape error
  two epochs in on a GPU that has been billing the whole time.

  It is a guest in someone's run and behaves like one. It installs the same
  hooks the CLI uses and drives nothing: no `eval()`, no grad enabled, no
  `requires_grad` set on the caller's batch, no second forward pass, and
  `nn.Module.__call__` is put back after the one capture. Nothing leaves the
  machine unless `PLAN` or `SHARE` is set, and any failure inside the tracer is
  one line on stderr while the run continues.

- **`AGENTS.md`** at the repository root: what an agent that lands here should
  reach for, in what order, and the one rule worth loading first, which is that
  a finding from a statically parsed `.py` is not actionable and a traced graph
  is.

- **`plan` takes a change spec.** `spec: { adapter?, quant?, serving?, constraints? }`
  rides beside `policy` to `POST /api/v1/plan`, and the card gains a change
  section: what is allowed to move on this base, a spec id that is a join key
  beside the structural fingerprint (never inside it), and a gate that approves
  or refuses on memory, fit and policy. The tool's result carries
  `change: { specId, approved, blockers }`. The gate never says whether an
  adapter will train well; the server says so in as many words.

### Fixed
- **`size_training_job` asks the plan for eager attention explicitly.** The
  refreshed plan bundle defaults to SDPA, so its peak no longer contains the
  attention-score term that `size_training_job` removes itself for an SDPA or
  flash job. Left alone the term would have been removed twice: phi3-mini at
  batch 32 read 6.3 GB instead of 17.9 GB, a job that does not fit a T4 was
  reported as fitting, and `seq_len` stopped moving the peak on a long-context
  model. The two-card split and the price lookup are pinned the same way, and
  a test now checks that the SDPA peak lands on the plan's own SDPA peak.
- **The Docker image builds from source** (#26). The single-stage `COPY dist`
  only worked on a machine that had already run `npm run build`, so a registry
  building straight from git (Glama, Docker MCP Toolkit) failed at that line
  and the server never started there. Two stages now: `npm ci` and `tsup` in
  the build stage, production dependencies plus `dist/` in the runtime stage.
- **neurarch-trace probe fixes** (#23, #24, #25; the PyPI package, 0.2.1).
  The autopatch no longer holds a strong reference to every module output
  until the forward pass ends (a 50M GPT at batch 128 x 2048 went from OOM on
  an H100 to normal), and `NEURARCH_TRACE_EXIT=1` ends a probe process once
  the graph is written, with saved tensors kept on the CPU so a probe of a
  depth 11 GPT no longer needs 73 GB. An edge whose consumer received a
  different shape than its producer emitted is marked
  `inferred: unmodelled-op`, including after a shape-less merge node, and an
  input no layer consumes is left out of the graph instead of reported as an
  orphan.
- **Opt-in corpus rows survive the exit.** `sendCorpusReport` was fire-and-forget
  with nowhere to wait, so a process that exited before the POST left the socket
  dropped the row with no error anywhere. Harmless under a chat client, where the
  server is long-lived; always wrong under an agent that spawns the server, asks
  once and closes the pipe, which is how the first hand-run probe of this channel
  disappeared. The call site is still fire-and-forget (a corpus row must never
  slow or fail a verdict); `flushCorpusReports()` waits for in-flight rows on
  `beforeExit`, `SIGINT` and `SIGTERM`, for at most two seconds, and never
  throws. A signal is now re-raised after the flush instead of being turned into
  `exit(0)`, so a supervisor still reads the right exit code.

## [0.16.0] - 2026-09-03

### Added
- **`plan`**: the front-door artifact, missing from the agent surface until now.
  The same card the Neurarch CI bot posts on a pull request and the
  `neurarch-trace` CLI prints, so an agent and a reviewer read the same words
  about the same graph: will it run, what it costs, which GPUs it fits, which
  of the repository's own rules it breaks, what changed against a base design,
  and what happened last time this structure trained here. The server's `text`
  comes back verbatim. Policy lines are read from the repository's
  `.neurarch.yml` (walking up from the model, merging a model's own `policy:`
  over the top-level block key by key, exactly as the bot does) unless the call
  passes `policy`; `base_path` adds a diff. `share` is pinned to false, so an
  agent cannot publish someone's design. A `.neurarch.yml` that exists and
  cannot be parsed is an error rather than a skip: planning while ignoring the
  house rules and reporting "will run" is the confident wrong answer.

- **`history`**: what this exact structure scored the last time it trained
  inside your organisation, keyed by the graph's 8-character structural
  fingerprint, so it answers about the shape rather than the file. Metric,
  epochs, wall time and cost, newest first, rendered as a short table and one
  sentence a person would say ("This structure trained here 3 days ago: 98.65%
  in 3 epochs, 34s, under a cent"). Takes `model_path` like every read tool, or
  a `fingerprint` from a plan result, which is the one call this server answers
  without a model at all.

  **Without `NEURARCH_API_KEY` it makes no request and says so**, rather than
  returning an empty list: the ledger is per organisation, and an unread ledger
  and an empty one are different answers. Only one of them means nobody has
  trained this, and the distinction is the whole value of the row.

  It sends the fingerprint and the key. Never the graph.

- `NEURARCH_API_KEY` and `NEURARCH_API` are read, by these two tools and
  nowhere else. `NEURARCH_API` is the same variable `neurarch-trace` uses.

### Changed
- **These two are the only tools in the package that reach the network**, and
  they say so: `openWorldHint: true`, the reach named in the first sentence of
  each description, and a rewritten README network section and privacy policy.
  Every other tool still runs offline against the vendored engine, and with
  `--hf` off, `NEURARCH_REPORT` unset and neither of these called, the server
  makes no network calls at all.
- **`NEURARCH_REPORT` is unchanged and still off by default.** `plan` joins
  `validate_model` / `lint_model` / `check_design` as a tool that sends the
  anonymous structure row when reporting is on, so opting in means one thing
  across every tool that grades rather than leaving a gap an agent falls into
  by reaching for the newest tool. `history` grades nothing and sends no row.
- **Tracing is the primary path in the docs now, and static parsing is the
  fallback**, with the measurement next to the claim rather than the demotion
  asserted: the parser returns a graph a person would recognise for 41% of 116
  real model files, and every block and warn it raised on them was hand-judged
  as an artefact rather than a defect. `trace_model` reads the numbers at
  runtime, which is the only place they exist. The README's "three ways in" is
  reordered accordingly.
- **The core tool listing is ordered, and the order is the recommendation.**
  `selectTools` now reads `CORE_TOOLS` in order instead of filtering the
  registry, so an agent scanning the list sees `trace_model`, `plan`,
  `history`, then the offline ladder. The core set is fifteen tools and about
  4.6k tokens a turn, up from thirteen and 3.9k; the budget test's cap was
  raised deliberately rather than quietly.
- `structuralFingerprint` is exported from `lib/corpusReport.ts` and used by
  both the corpus row and `history`, so the two can never ask about different
  structures.

## [0.15.1] - 2026-09-03

### Fixed
- **neurarch-trace 0.1.3: in-place residual adds.** `out += identity` (torchvision
  BasicBlock) kept the tensor object, so the identity rule wired the consumer to the
  pre-add producer and every downsample branch ended as a dead end; resnet18 traced
  with 7 false warnings and no `add` nodes. The tracer now notices a grad_fn that
  changed since the module returned the tensor and walks autograd instead.

### Added
- **neurarch-trace 0.1.2: `--plan` and `--share`** (`python/neurarch-trace`).
  One command from a `.py` file to a plan card in the terminal, and with
  `--share` a public link on the last line. The CLI POSTs the traced graph to
  `/api/v1/plan` (the endpoint ships separately in the neurarch app) and prints
  the server's fixed-width `text` verbatim; `--base` sends a second graph to
  diff against, `--fail-on-block` exits 2 on a blocker, `NEURARCH_API_KEY` and
  `NEURARCH_API` are honoured. Nothing is sent without one of the two flags.

### Changed
- Vendored engine bundle rebuilt from neurarch `main` (60eae74). It carries
  neurarch #128, the parser pass for real model files (same-file classes
  inlined, root class chosen, source lines kept), which the 0.13 to 0.15
  bundles predated. `docs/REAL_REPOS_STUDY.md` is re-measured against it, on
  the same 116 files and with the `lint_model` dimension filter applied for
  the first time: a graph for 86% of files (was 63%), a recognisable one for
  41% (was 8%), 91 dimension findings held back, and every one of the 92
  block/warn findings that remain hand-judged as an artefact of construction
  order read as data flow or as a design choice. Real defects found: zero,
  as in August. The August study and results stay under `-2026-08-31` names.
- Hosted instance: one machine stays warm (`min_machines_running = 1`; a cold
  start answered the first `initialize` in 7.5s, warm is 77ms) and runs with
  `NEURARCH_REPORT=1`. Stated in `docs/HOSTED.md` and the README privacy
  section; a local server is unaffected.

## [0.15.0] - 2026-09-02

### Added
- **`rank_designs` takes `tie_break`** (`"cost"` or `"params"`): where nothing
  measured separates two legal candidates, the caller's stated rule breaks the
  tie, cheapest or smallest first. It never crosses a measured rank and never
  lifts a blocked design. The result echoes the rule under `tieBreak` with how
  many positions it decided, every candidate carries `measuredRank` (the
  verifier's own answer), and each reason the rule decided says it was the
  caller's budget, not a measurement.

### Changed
- Vendored verifier bundle rebuilt from neurarch `main` (#135, #136).
  `calibration.outOfSample` now carries the two code-reading judges (Claude
  Opus 5 12 of 15, Grok 4.3 11 of 15) and the no-model `largest-params` rule
  measured on the same held-out pairs (11 of 15), so neither judge number can
  be read alone.

## [0.14.0]

### Added
- **`suggest_fix`**: lint findings as unified diffs against the .py they came
  from. `exact` where the rule pins a number or an order (a width that does
  not divide by the head count is changed on every line that shares it; two
  forward() statements in the wrong order are swapped), `proposal` where a
  layer is missing (inserted, with what forward() still needs). Findings with
  no mechanical fix come back as `notFixable` with the reason. On tiny-vit.py
  the head-dim crash becomes a 7-line diff that lints clean.

- **`trace_model`**: runs neurarch-trace in the user's Python with the input
  dims and returns a graph with real shapes as a `model_path`. The way past
  the static parser for code it cannot follow.

- **`check_design` `ask_user`**: when the verdict ends in a decision only the
  human can make and the client supports MCP elicitation, ask them through
  the client and return their answer with the verdict.

- **`neurarch://docs` and `neurarch://docs/{tool}`**: the full contract of
  every tool as a resource, including the ones the default listing omits.

- **`scripts/eval-tool-selection.mjs`** (`npm run eval:tools`): six fixed asks
  run through Claude Code headless against the built server, graded on which
  tools were called (lint before proposing an edit, rank_designs for a
  which-of-k question, layer_impact for what-breaks). Results land in
  `docs/tool-selection-eval.json` with the date. Costs tokens; not in CI.

### Changed
- **The default listing is the core set.** Thirteen tools instead of
  twenty-six; `--tools=full` advertises all. Every tool stays callable by
  name. Measured on 0.13 the full list cost about 6.9k tokens per turn; the
  default is now about 3.9k and the full set 5.8k, most of the rest being
  input schemas.
- **Every description is one or two sentences.** The long form moved to the
  docs resource. The injected `model_path` / `model_source` descriptions,
  repeated on every read tool, were the single largest cost and are a line
  each now.
- **`check_design` returns the trimmed verdict by default**: verdict,
  findings, stage headlines, decision. `verbose:true` adds each stage's data.
- **Loose `outputSchema`** on the core tools: the keys an agent should rely
  on, `additionalProperties: true`, so a drift costs a hint and not a call.
- The Smithery bundle lists all 26 tools with the short descriptions.

## [0.13.0]

### Added
- **`rank_designs`**: which of k candidate designs to spend a training run on,
  computed locally with the same contract as `POST /api/v1/rank`. Candidates
  are paths, `zoo:`/`hf:` refs or inline graphs; `include_current` adds the
  server's model. Blocked candidates (a pre-flight finding that means the graph
  will not forward-pass, the measured part: 96/96 crashed, 80/80 ran) rank
  last and come back as reclaimable budget. Legal candidates are ordered only
  by rules with a trained outcome behind them, and a tie stays a tie:
  `recommended` is null when nothing measured separates the top, which is the
  common answer and the honest one. Params, cost and GPU fit are returned per
  candidate and never used to order. `calibration` ships inside every result.

- **`export_pytorch`**: the graph as a runnable `nn.Module`, the app's own
  generator. `save_to` is honoured only under `--write` and never over the file
  the server was started from.

- **81 reference architectures bundled** (`zoo/`, 320KB gzipped, synced from
  github.com/neurarch-ai/awesome-llm-model-zoo): DeepSeek-V3, Qwen2.5, Llama,
  Mixtral, Gemma, Whisper, CLIP, BERT, ViT, ResNet and more, with real
  dimensions from each model's config and a parameter count checked against
  the published one. `list_architectures` searches them, `load_architecture`
  opens one, and any read tool accepts `model_path: "zoo:<id>"`. Offline.

- **Hugging Face input, behind `--hf`**: `load_hf_model` and `model_path:
  "hf:<org/name>"` build a graph from a repo's config.json. It is the one tool
  that opens a socket, which is why it is listed only when the flag is on;
  `HF_TOKEN` is sent for gated repos, results are cached for a day under
  `~/.cache/neurarch-mcp`. The result says whether it came from the real config
  or a family template (`configSource`) and quotes the parameter count HF
  publishes, so the agent can see how close the graph is (Qwen2.5-0.5B: graph
  494.00M against 494,032,768 published).

- **`find_models`**: walk a directory for nn.Module definitions and saved
  graphs, try the parser on each, and say which need a runtime trace instead.

- **`neurarch-trace`** (`python/neurarch-trace`, PyPI): instantiate the model,
  run one forward with hooks, write a `.neurarch.json` with real shapes. Covers
  what static parsing cannot: `from_pretrained`, timm, models spread across
  files, dynamic construction. `pip install neurarch-trace`, then
  `neurarch-trace my_pkg.model:build --input 1,3,224,224`.

- **MCP prompts**: `review_design`, `pre_train_checklist`, `shrink_for_target`,
  `compare_with_reference`, `explain_finding`. These render as slash commands
  in Claude Desktop, Cursor and VS Code, and each is the tool ladder written
  out in order, with the rule that every number comes from a tool result.

- **MCP resources**: `neurarch://model` (the graph), `neurarch://model/mermaid`,
  `neurarch://model/pytorch`, `neurarch://zoo`, `neurarch://zoo/{id}`,
  `neurarch://rules` (the provenance table). A resource is pinned as context;
  a tool result scrolls away.

- **`model_source`**: any read tool accepts inline model text (a
  `.neurarch.json` document or PyTorch source) instead of a path, for clients
  with no shared filesystem.

- **Hosted mode**: `--http` with no model argument serves a server that answers
  about whatever each call names (`model_path` zoo:/hf:, or `model_source`).
  `Dockerfile`, `fly.toml` and `docs/HOSTED.md` carry the deploy.

- **`neurarch-mcp lint <files...>` and `neurarch-mcp check <file>`**: the two
  grading tools as a CLI, exit 1 on a block, `--json` for CI. One package that
  is both the server and the linter; `neurarch-lint` on npm stays as an alias.

- **`--tools=core`**: advertise eleven tools instead of twenty-five. Every tool
  stays callable by name; this trims what rides along in the agent's context.

- **Claude Desktop bundle**: `manifest.json` and `npm run build:mcpb` produce
  the one-click `.mcpb`.

- **`skills/neurarch-skill`**: the evidence-gated edit loop skill now lives in
  this repo (`npx skills add neurarch-ai/neurarch-mcp`).

- **`docs/REAL_REPOS_STUDY.md`**: the static parser and linter measured against
  116 model files from 59 popular repositories. A graph came back for 63% of
  files, a recognisable one for 8%, and all 103 findings on real code were
  hand-judged to be parser artefacts or design choices: zero real bugs. It is
  in the repo because the alternative is a README that says "point it at a
  .py and it works" about code where it mostly does not.

  Three consequences shipped with it: `describe_architecture` and `lint_model`
  return a `parseQuality` grade (`full` / `partial` / `thin`) with the fix
  named; dimension rules are held back on layers whose dimension is still
  source text (54 of 54 such `invalid-output-shape` warnings were false) and
  reported as `suppressed` rather than dropped; `find_models` marks thin
  parses `partial`.

### Changed
- The README leads with a GIF (GitHub strips `<video>`), has Cursor and VS Code
  install buttons, and is written for the search terms people use.
- stdio mode routes `console.log` to stderr for the life of the process, so
  nothing in the vendored engine can corrupt a JSON-RPC frame.
- `ToolContext` carries `writeEnabled`; tools that can create files
  (`save_to`) refuse without it, the same way `save_model` is gated.

## [0.12.0]

### Changed
- **`check_design` no longer needs an API key, or a network call.** The verifier
  is vendored (`src/vendor/verifier.bundle.mjs`) and runs here, in about 13ms,
  against the graph already in memory.

  The key was never buying what a key is supposed to buy. `checkDesign` is a
  pure, synchronous function of the graph; the server around it validated the
  key, capped the graph at 2000 components, and rewrote the error text. The
  corpus row that would have justified metering is written by a different
  endpoint, which requires no key. So the key bought no compute and captured no
  measurement: it put a signup between a stranger and the one tool this product
  is built around, while the other eighteen tools ran offline on first install.

  The size cap is kept, because a runaway generated graph should come back with
  a sentence rather than stall the tool call. The hosted endpoint is unchanged
  for callers who are not running this server.

- **The network promise has no exceptions now.** Unset `NEURARCH_REPORT` and
  this server opens no socket, from any tool. `NEURARCH_API_KEY` and
  `NEURARCH_API_URL` are read nowhere in the package.

- **Corpus reporting covers all three grading tools**, not just
  `validate_model`, and grades with `lintModelGraph` rule ids rather than the
  validator's own vocabulary.

  Both halves were making the channel unusable. Coverage was an accident of
  which tool an agent happened to reach for: an agent that asked for the design
  rules or the whole verdict recorded nothing. And an `mcp` row said `cycle`
  where a `ci` row said `head-dim-divisibility`, so rows from the two channels
  could not be pooled even when they were about the same graph. The row is now
  derived from the graph rather than from the tool result, so the same graph
  produces the same row whichever tool asked, and the server's fingerprint
  dedupes rather than triple-counting.

### Added
- **`lint_model` returns `provenance`**: for each rule that fired and has a
  published measurement behind it, the measurement, quoted with its public
  source. Local, from a static table, no key. Rules with no number behind them
  are absent rather than dressed up, which is the same honesty contract the
  table has always carried.

## [0.11.0]

### Added
- **PyTorch source is now a first-class input.** `npx neurarch-mcp model.py` works.
  The server parses the file into the same graph the app and the CI action use,
  with the same parser, which removes the precondition that made this server a
  tool for people who were already Neurarch users: there is no longer a "first
  go open the app, draw your model and export it" step between installing this
  and getting an answer.

  What comes out of source is exact for layers, types, hyperparameters and
  wiring. Tensor shapes do not, because the source never says what goes in, so
  parameter counts still come out while FLOPs and shape contracts report as
  unknown rather than as a confident zero. `--write` is refused on a `.py` file
  and `save_model` refuses a `.py` target, because the graph was derived from
  the source and this server does not generate Python.

- **`lint_model`** — the design rules, offline and with no API key: attention
  head-dim and GQA divisibility, normalization and activation ordering, dropout
  and feature ranges, missing residuals in deep stacks, and the shape rules that
  can be decided statically. The same rule set the Neurarch CI action reports,
  so a clean result here is a clean CI run.

  It completes a ladder that was missing its middle rung. `validate_model` asks
  whether the thing is a well-formed graph at all; `lint_model` runs the design
  rules; `check_design` adds readiness, cost and deployment fit and is the only
  one that needs a key and a network call. An agent that reached for the network
  first was paying for an answer two thirds of which was computable locally.

- **`model_path` on every read tool.** One server now covers a repository
  instead of one file: ask about `baseline.py`, then about
  `variant_b.neurarch.json`, without registering a second server. Files are
  cached and re-read whenever their mtime moves, so an answer is never stale.
  Write tools refuse `model_path` outright: honouring it would let an agent
  mutate, and then save over, a file nobody pointed this server at, and ignoring
  it would edit a different model than the agent believes it is editing.

- **Tool annotations** (`readOnlyHint`, `destructiveHint`, `idempotentHint`,
  `openWorldHint`). All thirty tools were previously advertised as equally
  consequential, so a client had to confirm all of them or none. Now the read
  tools declare themselves read-only and closed-world, `check_design` declares
  the one network call, and `delete_layer` / `delete_connection` /
  `modify_layer` / `save_model` declare that they destroy something.

- **`structuredContent` on every result**, alongside the JSON text that clients
  predating it still read. No `outputSchema` is declared: a client fails a call
  whose structured result does not match a declared schema, so hand-writing two
  dozen schemas would convert every drift between schema and handler into a
  broken tool. The data is worth shipping without that; the contract can follow
  when the schemas are generated rather than typed twice.

- `examples/tiny-vit.py`, so the 30-second path needs no app and no export. It
  carries a real planted bug (`embed_dim=258`, `num_heads=8`) and a test asserts
  `lint_model` still finds it, because an example that quietly stops
  demonstrating anything is worse than no example.

### Changed
- The vendored Neurarch engine (`src/vendor/engine.bundle.mjs`) is compiled from
  the app repo and carries the component registry, the PyTorch parser and the
  rule set. A contract test asserts its exports and its lack of imports, so a
  regenerated bundle whose surface moved fails here rather than in a user's
  agent.
- `server.ts` had no test before this release, which is why the routing it now
  does is covered end to end through a real MCP client over an in-memory
  transport.

## [0.10.1]

### Fixed
- **`check_design` returned "Invalid or missing API key" to everyone who had a
  valid one.** 0.10.0 defaulted to `https://neurarch.com/api/v1/check`. The apex
  answers every `/api` route with a 307 to `www.neurarch.com`; Node's `fetch`
  follows it, keeps the method and body as a 307 requires, and then strips the
  `Authorization` header because the redirect crosses origins (this is the fetch
  spec, not a bug in undici). The request that landed carried no key, the server
  answered 401, and the error text sent the user off to re-create a key that was
  never the problem. The default is now `https://www.neurarch.com/api/v1/check`,
  and a test pins the host so the trap cannot be walked into again.

  No action needed beyond upgrading. Anyone who worked around this by setting
  `NEURARCH_API_URL` to the `www` host can drop the override.

- Corpus reporting (`NEURARCH_REPORT=1`) posted to the same apex host. It has no
  `Authorization` header to lose, so rows were still delivered, but each one paid
  a redirect inside a 5-second fire-and-forget budget. It now posts to `www`
  directly, and reads `NEURARCH_REPORT_URL` at call time rather than freezing it
  at import time, matching `check_design`.

## [0.10.0]

### Added
- **`check_design`** — the one tool here that returns a *verdict* rather than an
  inspection. Every other tool answers a question about the file in front of it
  (how many parameters, what does this layer touch, is there a cycle); this one
  answers the question an agent needs before it edits someone's architecture:
  is the design sound, what will a training run cost, which deployment target
  fits it and at what latency, and which decisions are still the human's.
  `validate_model` is the local structural subset of it.

  Requires `NEURARCH_API_KEY` (create one at neurarch.com under Settings →
  Developer API) and makes one network call. **With no key set it opens no
  socket and returns an explanation**, so the "no network calls by default"
  property is unchanged for anyone who has not configured one.

  Note that unlike corpus reporting, this **sends the model graph**, because a
  verdict about a graph cannot be computed without it. That difference is called
  out in its own README section rather than buried: if it is not acceptable for
  a given model, do not set a key and every other tool keeps working.

- `NEURARCH_API_URL` to point `check_design` at a self-hosted or test endpoint.

### Added in 0.9.1
- **Opt-in corpus reporting (`NEURARCH_REPORT=1`)** — when explicitly enabled,
  a `validate_model` call shares one anonymous structure+verdict row with the
  Neurarch corpus: structural fingerprint (8-char hash), layer-type histogram,
  edge count, and (rule id, severity) pairs. Never the graph, parameter
  values, layer names, or any identity; the payload shape cannot carry them.
  Off by default (no network calls at all without the flag); fire-and-forget
  with a 5s cap, so reporting can never slow or fail a tool call.

## [0.9.0]

### Added
- **Remote transport (`--http`)** — serve the same MCP tools over Streamable
  HTTP instead of stdio, so a hosted or phone-based agent can drive a model
  running on your machine (e.g. behind a Cloudflare/Tailscale tunnel). Stateful
  `Mcp-Session-Id` sessions, `GET /health` liveness probe, zero new runtime
  deps. `--host=ADDR` sets the bind address (default `127.0.0.1`).
  - **Auth & safety:** loopback-only by default with DNS-rebinding protection and
    no CORS; `NEURARCH_MCP_TOKEN` enables constant-time bearer auth and is
    required before `--write` may bind to a non-loopback host (the server
    refuses to start otherwise).

### Changed
- Server wiring extracted into a transport-agnostic `createMcpServer`
  (`src/server.ts`), shared by the stdio and HTTP paths; the model is now read
  through a getter so `--watch` reloads are always seen. `parseFlags` gains
  `--http` / `--host` parsing (unit-tested).

## [0.8.0]

### Added
- **`diff_models` tool** — structurally diff the current model against another
  `.neurarch.json` file: layers only in current / only in the other / modified
  (with field-level changes), plus added and removed connections. Layers are
  matched by name; duplicate names are reported as ambiguous rather than guessed.
  Use to review what an agent changed, or how the model differs from a checkpoint.
- **README "what it looks like"** transcript showing a real tool-grounded exchange.

### Changed
- CLI argument parsing and tool-dispatch gating extracted into a pure,
  unit-tested `src/cli.ts` (`parseFlags`, `selectTools`, `resolveToolCall`). The
  write-tool-gating hint ("restart with --write") is now covered by tests.

## [0.7.0]

### Added
- **`compare_layers` tool** — structural diff of two layers: same-type, the
  parameter-count delta, whether input/output shapes match, and exactly which
  param keys differ (only-in-a, only-in-b, changed values). Answers "are
  block_0 and block_1 identical?" before mirroring an edit or sharing an
  implementation.
- **`mermaid_diagram` renders groups as subgraphs** — grouped layers are wrapped
  in labelled Mermaid `subgraph` blocks so the diagram mirrors the model's
  blocks; ungrouped layers stay at the top level.

### Fixed
- `conv2d` parameter count now divides input channels by `groups`, matching the
  FLOPs estimator (grouped convs were overstated by `groups`×).
- `bidirectionalLSTM` params and FLOPs now honor `numLayers` (the unidirectional
  `lstm` case already did); `numLayers: 1` is unchanged.
- `lmHead` returns no new parameters when `weightTied`/`tied` is set, so
  weight-tied LLMs no longer inflate the headline parameter total.
- `mermaid_diagram` truncation keeps the topological head instead of an
  arbitrary file-order prefix, so the truncated diagram starts at the input.

## [0.6.0]

### Added
- **`find_layers` filters** — beyond type and name regex, the tool now filters
  by `scope` prefix (e.g. `encoder.layer.3` and anything nested under it) and by
  applied `augmentation` (e.g. `freeze`, `quantize_int8`), and can rank results
  by parameter count via `sortByParams`. Results now also carry each layer's
  scope, parameter count, and augmentations. Answers "which layers under X are
  frozen?" and "what are the biggest layers?" directly. Backward compatible.
- **`examples/resnet-mini.neurarch.json`** — a residual block with a skip/merge
  (`add`) node, giving the example set a branchier graph for the impact, path,
  and validation tools.

## [0.5.0]

### Added
- **`get_block` tool** — drill into one block (named group or scope prefix):
  member layers with per-layer params/FLOPs, block totals, and the edges
  crossing the block boundary (what feeds the block and what it feeds). Pairs
  with `list_blocks` for high-level → detail navigation.
- **Runnable example models** under `examples/` (`tiny-gpt`, `tiny-cnn`) so the
  server can be tried without exporting from the app. A test guards that every
  shipped example loads and validates with zero errors.
- **`--version` flag** (alias `-v`) prints the version and exits. The version is
  now sourced from `package.json`, so the CLI, the MCP handshake, and the
  package no longer drift.

## [0.4.0]

### Added
- **`describe_architecture` tool** — one-call orientation that returns the
  topologically-ordered layer pipeline, model depth (longest path), input/output
  shapes, total params and MACs, the top-5 heaviest layers by parameters **and**
  by compute, and a validation rollup. Replaces the common
  `get_model_summary` + `param_count_by_block` + `flops_by_block` +
  `validate_model` chain with a single round-trip.
- **Test suite (Vitest, ~100 unit tests)** covering validation, write ops,
  impact analysis, the param/FLOPs estimators, the loader, Mermaid rendering,
  the regex guard, and every tool handler. Run with `npm test`.
- **GitHub Actions CI** — `typecheck` + `build` + `test` on Node 20 and 22 for
  every push and pull request.
- `CONTRIBUTING.md` with a 3-step "add a tool" guide.
- `npm run typecheck` script (`tsc --noEmit`).

### Changed
- **Hardened user-supplied regex** (`find_layers`, `layer_impact`, the `/regex/`
  form of name resolution) behind a shared guard: caps pattern length and
  rejects nested-unbounded-quantifier shapes that risk catastrophic
  backtracking.
- **Stricter model loading** — files whose components lack a string `id` are now
  rejected at load time with a clear message instead of surfacing later as
  confusing "layer not found" errors.

### Fixed
- `mermaid_diagram` truncation no longer does an O(n²) `findIndex` scan per
  connection; it uses a precomputed id set.
- `lmHead` layers now report parameters (`embedDim × vocabSize`, optional bias)
  and FLOPs (`tokens × embedDim × vocabSize`); previously the estimators had no
  case for them and silently returned 0, undercounting LM parameter budgets.

## [0.3.0]
- `validate_model`, `find_path`, `list_connections`, hyperparameters and design
  notes tools, layer/connection delete tools, and `--watch` live reload.

## [0.2.0]
- Write tools: `add_layer`, `modify_layer`, `add_connection`, `save_model`.

## [0.1.1]
- Initial release: read-only structural tools over a Neurarch model file.
