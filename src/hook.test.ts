import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { mkdtemp, writeFile, readFile, rm, copyFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runHookCommand, parseLaunch, shellWords, findJobBlock } from './hook.js';
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
