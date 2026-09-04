/**
 * The property that matters here is not "does it parse YAML". It is that the
 * policy the `plan` tool sends is the policy the pull request will be judged
 * by. A parser that reads three of four house rules and says nothing produces
 * a plan that passes a rule the repository fails, which is worse than having
 * no policy at all, so anything unreadable throws with the line number.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseYamlSubset, resolvePolicy, readPolicyFor, POLICY_KEYS } from './policyFile.js';

let dir: string;
beforeEach(async () => { dir = await mkdtemp(join(tmpdir(), 'nx-policy-')); });
afterEach(async () => { await rm(dir, { recursive: true, force: true }); });

/** The config from neurarch-ai/model-ci-example, verbatim in shape. */
const REAL_CONFIG = `# Models neurarch-bot plans on every pull request that touches them.

policy:
  forbid_types: [softmax]   # a Softmax head in front of CrossEntropyLoss is the classic one

models:
  - path: models/tiny_gpt.py:TinyGPT
    input: "1,32:long"
  - path: models/mnist_net.py:MnistNet
    input: 1,1,28,28          # one greyscale 28x28 channel

  - path: models/small_cnn.py:SmallCNN
    input: 1,3,32,32
    policy:
      max_params: 500K      # this classifier's own cap
`;

describe('parseYamlSubset', () => {
  it('reads the real .neurarch.yml: comments, flow sequences, quoted and bare scalars, nested policy', () => {
    const cfg = parseYamlSubset(REAL_CONFIG) as any;
    expect(cfg.policy).toEqual({ forbid_types: ['softmax'] });
    expect(cfg.models).toHaveLength(3);
    expect(cfg.models[0]).toEqual({ path: 'models/tiny_gpt.py:TinyGPT', input: '1,32:long' });
    // `1,1,28,28` must stay the string the endpoint expects, not become a number.
    expect(cfg.models[1].input).toBe('1,1,28,28');
    expect(cfg.models[2].policy).toEqual({ max_params: '500K' });
  });

  it('keeps a size suffix as text and a plain number as a number', () => {
    const cfg = parseYamlSubset('policy:\n  max_params: 200M\n  max_layers: 40\n  max_cost_usd: 0.5\n  must_fit: "A100-40GB"\n') as any;
    expect(cfg.policy).toEqual({ max_params: '200M', max_layers: 40, max_cost_usd: 0.5, must_fit: 'A100-40GB' });
  });

  it('reads a block sequence as well as a flow one', () => {
    const cfg = parseYamlSubset('policy:\n  forbid_types:\n    - softmax\n    - dropout\n') as any;
    expect(cfg.policy.forbid_types).toEqual(['softmax', 'dropout']);
  });

  it('leaves a # inside quotes alone', () => {
    const cfg = parseYamlSubset('policy:\n  must_fit: "T4 # 16GB"\n') as any;
    expect(cfg.policy.must_fit).toBe('T4 # 16GB');
  });

  it('refuses what it cannot read, by name and line, instead of dropping it', () => {
    expect(() => parseYamlSubset('policy:\n  note: |\n    a block scalar\n')).toThrow(/line 2: block scalars/);
    expect(() => parseYamlSubset('policy: &anchor\n  max_layers: 4\n')).toThrow(/anchors/);
    expect(() => parseYamlSubset('policy: {max_layers: 4}\n')).toThrow(/flow mappings/);
    expect(() => parseYamlSubset('---\npolicy:\n  max_layers: 4\n')).toThrow(/multi-document/);
    expect(() => parseYamlSubset('policy:\n\tmax_layers: 4\n')).toThrow(/tabs/);
    expect(() => parseYamlSubset('policy:\n  forbid_types: [softmax\n')).toThrow(/close on the same line/);
    expect(() => parseYamlSubset('just a sentence\n')).toThrow(/expected "key: value"/);
  });

  it('is empty for an empty or comment-only file rather than throwing', () => {
    expect(parseYamlSubset('')).toEqual({});
    expect(parseYamlSubset('# nothing here\n\n')).toEqual({});
  });
});

describe('resolvePolicy', () => {
  const cfg = parseYamlSubset(REAL_CONFIG);
  const configPath = '/repo/.neurarch.yml';

  it('merges a model\'s own lines over the house rules, key by key', () => {
    const r = resolvePolicy(cfg, configPath, resolve('/repo/models/small_cnn.py'));
    // The tighter cap applies AND softmax is still forbidden: an entry that
    // tightens one line must not silently drop the rest.
    expect(r.policy).toEqual({ forbid_types: ['softmax'], max_params: '500K' });
    expect(r.matchedModel).toBe('models/small_cnn.py:SmallCNN');
    expect(r.note).toContain('small_cnn.py:SmallCNN');
  });

  it('applies the house rules to a model the config does not list', () => {
    const r = resolvePolicy(cfg, configPath, resolve('/repo/models/unlisted.py'));
    expect(r.policy).toEqual({ forbid_types: ['softmax'] });
    expect(r.matchedModel).toBeNull();
  });

  it('sends only the keys the endpoint accepts, and names the ones it dropped', () => {
    const odd = parseYamlSubset('policy:\n  max_layers: 40\n  maxparams: 10\n  notes:\n    - one\n');
    const r = resolvePolicy(odd, configPath, resolve('/repo/m.py'));
    expect(r.policy).toEqual({ max_layers: 40 });
    expect(r.ignoredKeys.sort()).toEqual(['maxparams', 'notes']);
    for (const k of Object.keys(r.policy!)) expect(POLICY_KEYS).toContain(k as never);
  });

  it('reports no policy rather than an empty one when the file declares none', () => {
    const r = resolvePolicy(parseYamlSubset('models:\n  - path: a.py:A\n    input: 1,3,8,8\n'), configPath, resolve('/repo/a.py'));
    expect(r.policy).toBeNull();
    expect(r.note).toContain('declares no policy lines');
  });
});

describe('readPolicyFor', () => {
  it('walks up from the model to the repository root and says what it found', async () => {
    await mkdir(join(dir, 'models'), { recursive: true });
    await writeFile(join(dir, '.neurarch.yml'), REAL_CONFIG);
    const model = join(dir, 'models', 'small_cnn.py');
    await writeFile(model, '# model\n');
    const r = await readPolicyFor(model);
    expect(r.path).toBe(join(dir, '.neurarch.yml'));
    expect(r.policy).toEqual({ forbid_types: ['softmax'], max_params: '500K' });
    expect(r.note).toContain(join(dir, '.neurarch.yml'));
  });

  it('finds one sitting next to the model too', async () => {
    await writeFile(join(dir, '.neurarch.yaml'), 'policy:\n  max_layers: 12\n');
    await writeFile(join(dir, 'm.neurarch.json'), '{}');
    const r = await readPolicyFor(join(dir, 'm.neurarch.json'));
    expect(r.path).toBe(join(dir, '.neurarch.yaml'));
    expect(r.policy).toEqual({ max_layers: 12 });
  });

  it('says so plainly when there is none, and for a model with no path at all', async () => {
    await writeFile(join(dir, 'm.neurarch.json'), '{}');
    for (const path of [join(dir, 'm.neurarch.json'), '']) {
      const r = await readPolicyFor(path);
      expect(r.path).toBeNull();
      expect(r.policy).toBeNull();
      expect(r.note).toMatch(/No \.neurarch\.yml found/);
    }
  });

  it('errors on a config it cannot read rather than planning without the house rules', async () => {
    await writeFile(join(dir, '.neurarch.yml'), 'policy:\n  note: |\n    unreadable\n');
    await writeFile(join(dir, 'm.neurarch.json'), '{}');
    await expect(readPolicyFor(join(dir, 'm.neurarch.json'))).rejects.toThrow(/could not be read.*block scalars/s);
  });
});
