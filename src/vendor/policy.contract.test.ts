/**
 * The policy bundle is a copy of the Neurarch CLI's policy.bundle.mjs, and
 * check_pins reports its findings as the same ones `npx neurarch pin-check`
 * prints. So this pins behaviour, not only exports: the three rules on the
 * spellings the CLI's own tests use, and the Bandit-compatible exemptions
 * (a commit sha, a local path, `# nosec`).
 */
import { describe, it, expect } from 'vitest';
import * as policy from './policy.bundle.mjs';

describe('vendored policy contract', () => {
  it('exports the pin-check functions this server declares', () => {
    for (const fn of ['scanPins', 'pinFileKind', 'fromPinCheck', 'isPinnedRevision', 'pinSpelling'] as const) {
      expect(typeof policy[fn], fn).toBe('function');
    }
    expect(Object.keys(policy.PIN_RULE_META).sort()).toEqual(['pin/hf-floating-revision', 'pin/hf-unpinned', 'pin/hf-unpinned-dynamic']);
  });

  it('flags an unpinned, a floating and a dynamic download and leaves a pinned one alone', () => {
    const src = [
      'from transformers import AutoModel',
      'a = AutoModel.from_pretrained("Qwen/Qwen3-8B")',
      'b = AutoModel.from_pretrained("Qwen/Qwen3-8B", revision="main")',
      'c = AutoModel.from_pretrained(name)',
      'd = AutoModel.from_pretrained("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")',
      'e = AutoModel.from_pretrained("./local/ckpt")',
      'f = AutoModel.from_pretrained("Qwen/Qwen3-8B")  # nosec B615',
    ].join('\n');
    const hits = policy.scanPins('train.py', src);
    expect(hits.map(h => [h.line, h.rule])).toEqual([
      [2, 'pin/hf-unpinned'],
      [3, 'pin/hf-floating-revision'],
      [4, 'pin/hf-unpinned-dynamic'],
    ]);
    const findings = policy.fromPinCheck(hits);
    expect(findings.map(f => f.severity)).toEqual(['warn', 'warn', 'info']);
    expect(findings[0].fix).toContain('revision="<sha>"');
  });

  it('reads a vllm serve line and a YAML model key', () => {
    expect(policy.scanPins('serve.sh', 'vllm serve Qwen/Qwen3-8B --tensor-parallel-size 4\n').map(h => h.rule)).toEqual(['pin/hf-unpinned']);
    expect(policy.scanPins('cfg.yaml', 'base_model: meta-llama/Llama-3.1-8B\n').map(h => h.rule)).toEqual(['pin/hf-unpinned']);
    expect(policy.pinFileKind('README.md')).toBeNull();
  });
});
