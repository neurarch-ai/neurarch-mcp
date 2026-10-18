import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { mkdtemp, writeFile, readFile, rm, copyFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runHookCommand, parseLaunch, shellWords, findJobBlock, resolveJob, pickJob, accumulationSplit } from './hook.js';
import { parseYamlSubset } from './lib/policyFile.js';
import { sizeJob } from './lib/sizeJob.js';
import { parseFlags } from './cli.js';
import type { ModelArchitecture } from './lib/types.js';

/** src/test: the fixtures name models as templates/<name>.neurarch.json relative to it. */
const TEST_DIR = fileURLToPath(new URL('./test/', import.meta.url));
let jobDir: string;

async function fixture(name: string): Promise<string> {
  const raw = await readFile(new URL(`./test/hooks/${name}.json`, import.meta.url), 'utf-8');
  return raw.replace('__CWD__', TEST_DIR).replace('__JOBDIR__', jobDir);
}

async function run(text: string): Promise<{ code: number; err: string; out: string }> {
  let err = '';
  let out = '';
  const code = await runHookCommand(text, { out: s => { out += s; }, err: s => { err += s; } });
  return { code, err, out };
}

async function template(name: string): Promise<ModelArchitecture> {
  return JSON.parse(await readFile(new URL(`./test/templates/${name}.neurarch.json`, import.meta.url), 'utf-8'));
}

beforeAll(async () => {
  jobDir = await mkdtemp(join(tmpdir(), 'nx-hook-'));
  await copyFile(new URL('./test/templates/phi3-mini.neurarch.json', import.meta.url), join(jobDir, 'phi3-mini.neurarch.json'));
  await writeFile(join(jobDir, '.neurarch.yml'), [
    '# the job this repository launches',
    'job:',
    '  model: phi3-mini.neurarch.json',
    '  gpu: L4',
    '  batch: 32',
    '  seq_len: 2048',
    '',
  ].join('\n'));
});
afterAll(async () => { await rm(jobDir, { recursive: true, force: true }); });

describe('neurarch-mcp hook, on fixture payloads', () => {
  it('blocks a python train.py that will not fit its T4, with the peak, the largest batch and the smallest card', async () => {
    const r = await run(await fixture('block-python-train'));
    const s = sizeJob(await template('phi3-mini'), { batchSize: 32, gpu: 'T4', seqLen: 2048 });
    expect(s.fits).toBe(false);
    expect(r.code).toBe(2);
    expect(r.out).toBe('');
    expect(r.err).toContain('neurarch: this training job will not fit T4.');
    expect(r.err).toContain(s.verdict);
    expect(r.err).toContain(`peak ~${s.memory!.peakGb} GB (${s.memory!.needGb} GB with headroom), card T4 (16 GB)`);
    expect(r.err).toContain(`largest batch that fits T4: ${s.maxBatch!.T4} at seq 2048`);
    expect(r.err).toContain(`smallest card that holds it at batch 32: ${s.smallestFitGpu}`);
    expect(r.err).toMatch(/NEURARCH_ALLOW_OOM=1/);
  });

  it('lets a job that fits run, silently', async () => {
    expect(await run(await fixture('fits-python-train'))).toEqual({ code: 0, err: '', out: '' });
  });

  it('ignores commands that do not launch training, and tools that are not Bash', async () => {
    expect(await run(await fixture('not-training'))).toEqual({ code: 0, err: '', out: '' });
    expect(await run(await fixture('not-bash'))).toEqual({ code: 0, err: '', out: '' });
  });

  it('stays out of the way of an adapter run, which the full fine-tune peak does not describe', async () => {
    expect(await run(await fixture('lora-train'))).toEqual({ code: 0, err: '', out: '' });
  });

  it('reads torchrun\'s process count: 8 x per-device 32 on 8 H100s fits', async () => {
    const s = sizeJob(await template('phi3-mini'), { batchSize: 256, gpu: '8xH100' });
    expect(s.fits).toBe(true);
    expect(await run(await fixture('torchrun-8xh100'))).toEqual({ code: 0, err: '', out: '' });
  });

  it('fills what the command leaves out from the .neurarch.yml job block', async () => {
    const r = await run(await fixture('job-block-accelerate'));
    const s = sizeJob(await template('phi3-mini'), { batchSize: 32, gpu: 'L4', seqLen: 2048 });
    expect(s.fits).toBe(false);
    expect(r.code).toBe(2);
    expect(r.err).toContain('will not fit L4');
    expect(r.err).toContain(join(jobDir, '.neurarch.yml'));
    expect(r.err).toContain(`largest batch that fits L4: ${s.maxBatch!.L4}`);
  });

  it('lets NEURARCH_ALLOW_OOM=1 through, because the estimate is a fit check', async () => {
    expect(await run(await fixture('allow-override'))).toEqual({ code: 0, err: '', out: '' });
  });

  it('fails open: a model it cannot load, a payload it cannot parse, an empty stdin', async () => {
    expect(await run(await fixture('unknown-model'))).toEqual({ code: 0, err: '', out: '' });
    expect(await run('not json')).toEqual({ code: 0, err: '', out: '' });
    expect(await run('')).toEqual({ code: 0, err: '', out: '' });
    expect(await run('{"tool_name":"Bash","tool_input":{}}')).toEqual({ code: 0, err: '', out: '' });
  });

  it('never blocks without a declared card or batch', async () => {
    const payload = (command: string) => JSON.stringify({ tool_name: 'Bash', cwd: TEST_DIR, tool_input: { command } });
    expect((await run(payload('python train.py --model templates/phi3-mini.neurarch.json --batch_size 32'))).code).toBe(0);
    expect((await run(payload('python train.py --model templates/phi3-mini.neurarch.json --gpu T4'))).code).toBe(0);
    expect((await run(payload('python train.py --model templates/phi3-mini.neurarch.json --batch_size 32 --gpu A100'))).code).toBe(0);
  });
});

describe('parseLaunch', () => {
  it('recognises the launchers and reads the flags, with = or a space', () => {
    expect(parseLaunch('python3 train.py --model gpt2 --batch_size=16 --max_seq_length 512')).toMatchObject({ launcher: 'python', model: 'gpt2', batch: 16, seqLen: 512 });
    expect(parseLaunch('accelerate launch --num_processes 4 sft.py --per_device_train_batch_size 8')).toMatchObject({ launcher: 'accelerate', processes: 4, perDeviceBatch: 8 });
    expect(parseLaunch('torchrun --nproc-per-node 2 pretrain.py --attn_implementation eager')).toMatchObject({ launcher: 'torchrun', processes: 2, attention: 'eager' });
    expect(parseLaunch('python -m torch.distributed.run --nproc_per_node 8 x.py')).toMatchObject({ launcher: 'torchrun', processes: 8 });
    expect(parseLaunch('deepspeed --num_gpus 2 run.py --batch 4')).toMatchObject({ launcher: 'deepspeed', processes: 2, batch: 4 });
    expect(parseLaunch('sky launch task.yaml --gpus A100-80GB:8')).toMatchObject({ launcher: 'sky', gpu: '8xA100-80' });
    expect(parseLaunch('cd repo && CUDA_VISIBLE_DEVICES=0 python finetune.py --model "my org/model"')).toMatchObject({ launcher: 'python', model: 'my org/model' });
  });

  it('is null for commands that are not a training launch', () => {
    for (const c of ['ls', 'python eval.py --batch 32', 'pip install accelerate', 'accelerate config', 'sky status', 'python -c "import train"']) {
      expect(parseLaunch(c), c).toBeNull();
    }
  });

  it('marks flags the estimate does not model, unless they are switched off', () => {
    expect(parseLaunch('python train.py --gradient_checkpointing')?.unmodelled).toBe('--gradient_checkpointing');
    expect(parseLaunch('python train.py --load_in_4bit True')?.unmodelled).toBe('--load_in_4bit');
    expect(parseLaunch('python train.py --gradient_checkpointing false')?.unmodelled).toBeUndefined();
  });

  it('stops at the end of the launch: flags of a later command are not read', () => {
    expect(parseLaunch('python train.py --batch 8 && python other.py --batch 64')?.batch).toBe(8);
  });

  it('splits shell words with quotes and operators', () => {
    expect(shellWords(`a 'b c' "d\\"e" f&&g;h | i`)).toEqual(['a', 'b c', 'd"e', 'f', '&&', 'g', ';', 'h', '|', 'i']);
  });
});

describe('the job block', () => {
  it('is found above the working directory and read as typed values', async () => {
    const b = await findJobBlock(jobDir);
    expect(b?.job).toEqual({ model: 'phi3-mini.neurarch.json', gpu: 'L4', batch: 32, seq_len: 2048, attention: undefined });
  });
});

describe('the subcommand', () => {
  it('parses as a command, with --hf as its one flag', () => {
    expect(parseFlags(['hook']).command).toBe('hook');
    expect(parseFlags(['hook', '--hf'])).toMatchObject({ command: 'hook', hfEnabled: true, unknownFlags: [] });
  });
});

/**
 * The block neurarch-bot reads (scripts/job.py in neurarch-ai/neurarch-bot):
 * `src/test/hooks/bot-repo/.neurarch.yml` is that repository's
 * examples/.neurarch.yml copied verbatim (at 1e5da05), beside the training
 * config its `file:key` pointers name.
 */
const BOT_REPO = fileURLToPath(new URL('./test/hooks/bot-repo/', import.meta.url));

describe('the job block is neurarch-bot\'s', () => {
  it('reads the bot\'s example: the one model entry with a job, its file:key pointers resolved against the repo root', async () => {
    const b = await findJobBlock(join(BOT_REPO, 'configs'));
    expect(b?.root).toBe(BOT_REPO.replace(/\/$/, ''));
    expect(b?.job).toEqual({ gpu: 'A10G', batch: 32, seq_len: 2048, grad_accum: 4, epochs: 3, samples: 400000 });
    expect(b?.sources).toEqual({
      batch: 'configs/pretrain.yaml:batch_size',
      seq_len: 'configs/pretrain.yaml:seq_len',
      grad_accum: 'configs/pretrain.yaml:grad_accum',
    });
    expect(b?.problems).toEqual([]);
  });

  it('blocks a launch in that repository, sized at the per-step batch, with the effective batch kept by accumulation', async () => {
    const payload = JSON.stringify({ tool_name: 'Bash', cwd: BOT_REPO, tool_input: { command: 'python train.py --model phi3-mini.neurarch.json' } });
    const r = await run(payload);
    const s = sizeJob(await template('phi3-mini'), { batchSize: 32, gpu: 'A10G', seqLen: 2048 });
    expect(s.fits).toBe(false);
    expect(r.code).toBe(2);
    expect(r.err).toContain('will not fit A10G');
    expect(r.err).toContain('batch from configs/pretrain.yaml:batch_size');
    expect(r.err).toContain('batch 32 x 4 accumulation steps (effective 128)');
    const [micro, steps] = accumulationSplit(128, s.maxBatch!.A10G!);
    expect(r.err).toContain(`Batch ${micro} with ${steps} gradient-accumulation steps keeps the effective batch at ${micro * steps}`);
  });

  it('the command wins over the block: a batch that fits on the line is let through', async () => {
    const mb = sizeJob(await template('phi3-mini'), { batchSize: 1, gpu: 'A10G', seqLen: 2048 }).maxBatch!.A10G!;
    const payload = JSON.stringify({ tool_name: 'Bash', cwd: BOT_REPO, tool_input: { command: `python train.py --model phi3-mini.neurarch.json --batch_size ${mb}` } });
    expect(await run(payload)).toEqual({ code: 0, err: '', out: '' });
  });

  it('merges a model entry\'s job over the top-level one, key by key, and picks the entry by its path', () => {
    const cfg = parseYamlSubset([
      'job:',
      '  gpu: T4',
      '  batch: 8',
      '  owner: ml-platform        # not a job key: ignored',
      'models:',
      '  - path: models/a.py:A',
      '    input: 1,3,32,32',
      '    job:',
      '      batch: 64',
      '  - path: models/b.py:B',
      '    input: 1,3,32,32',
      '    job:',
      '      gpu: "A100 40GB"',
      '',
    ].join('\n'));
    expect(pickJob(cfg, 'models/b.py', '/repo')).toEqual({ gpu: 'A100 40GB', batch: 8, owner: 'ml-platform' });
    expect(pickJob(cfg, 'models/a.py', '/repo')).toEqual({ gpu: 'T4', batch: 64, owner: 'ml-platform' });
    // Two entries with a job and no match: the top level alone, not a guess.
    expect(pickJob(cfg, 'models/c.py', '/repo')).toEqual({ gpu: 'T4', batch: 8, owner: 'ml-platform' });
  });

  it('resolves like the bot: unknown keys ignored, JSON with dotted keys, bad pointers and non-numbers dropped and named', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'nx-hook-ref-'));
    try {
      await writeFile(join(dir, 'train.json'), JSON.stringify({ trainer: { batch_size: 16, accum: 2 } }));
      const r = await resolveJob({
        gpu: 'L4',
        batch: 'train.json:trainer.batch_size',
        grad_accum: 'train.json:trainer.accum',
        seq_len: 'train.json:trainer.seq_len',
        epochs: 'three',
        attention: 'eager',
        owner: 'ml-platform',
        model: 'hf:Qwen/Qwen2.5-0.5B',
      }, dir);
      expect(r.job).toEqual({ gpu: 'L4', batch: 16, grad_accum: 2, attention: 'eager', model: 'hf:Qwen/Qwen2.5-0.5B' });
      expect(r.sources).toEqual({ batch: 'train.json:trainer.batch_size', grad_accum: 'train.json:trainer.accum' });
      expect(r.problems).toHaveLength(2);
      expect(r.problems[0]).toMatch(/^job\.seq_len reads train\.json:trainer\.seq_len/);
      expect(r.problems[1]).toMatch(/^job\.epochs is "three", not a positive whole number/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('a pointer it cannot read leaves the key unset, so the hook fails open rather than sizing a guess', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'nx-hook-ref-'));
    try {
      await copyFile(new URL('./test/templates/phi3-mini.neurarch.json', import.meta.url), join(dir, 'phi3-mini.neurarch.json'));
      await writeFile(join(dir, '.neurarch.yml'), 'job:\n  model: phi3-mini.neurarch.json\n  gpu: T4\n  batch: configs/missing.yaml:batch_size\n');
      const payload = JSON.stringify({ tool_name: 'Bash', cwd: dir, tool_input: { command: 'accelerate launch train.py' } });
      expect(await run(payload)).toEqual({ code: 0, err: '', out: '' });
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('reads --gradient_accumulation_steps off the command line', () => {
    expect(parseLaunch('python train.py --gradient_accumulation_steps 8')?.gradAccum).toBe(8);
  });

  it('splits an effective batch the way the bot does', () => {
    expect(accumulationSplit(128, 12)).toEqual([8, 16]);
    expect(accumulationSplit(32, 40)).toEqual([32, 1]);
    expect(accumulationSplit(97, 10)).toEqual([10, 10]);
  });
});
