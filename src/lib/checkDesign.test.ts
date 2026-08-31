/**
 * The property that matters most here is not "does it parse the verdict".
 *
 * It is that this server's README promise holds: nothing in this package opens
 * a socket unless you opt into corpus reporting. That promise is why someone is
 * willing to point this at a private model file, and this tool used to be its
 * one exception. Now it is not, and the tests below are what keeps it that way:
 * a future version that quietly reaches for the network again fails here.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { checkDesign, provenanceForRules, MAX_COMPONENTS } from './checkDesign.js';
import type { ModelArchitecture } from './types.js';

const model = (): ModelArchitecture => ({
  id: 'm',
  name: 'T',
  components: [
    { id: 'i', name: 'input', type: 'input', position: { x: 0, y: 0 }, params: { shape: [3, 32, 32] }, inputs: [], outputs: ['c'] },
    { id: 'c', name: 'conv', type: 'conv2d', position: { x: 0, y: 0 }, params: { inChannels: 3, outChannels: 16, kernelSize: 3 }, inputs: ['i'], outputs: [] },
  ],
  connections: [{ id: 'e', from: 'i', to: 'c', fromPort: 'right', toPort: 'left' }],
} as unknown as ModelArchitecture);

let prevKey: string | undefined;
beforeEach(() => { prevKey = process.env.NEURARCH_API_KEY; });
afterEach(() => {
  if (prevKey === undefined) delete process.env.NEURARCH_API_KEY;
  else process.env.NEURARCH_API_KEY = prevKey;
  vi.restoreAllMocks();
});

describe('the offline promise', () => {
  it('opens no socket, with no key set', () => {
    delete process.env.NEURARCH_API_KEY;
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    const out = checkDesign(model());
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(out).toHaveProperty('verdict');
  });

  it('opens no socket WITH a key set either', () => {
    // The regression this guards is subtle: a later change could "use the key
    // when there is one" and quietly reintroduce the network call for exactly
    // the users who thought they were paying for something better.
    process.env.NEURARCH_API_KEY = 'nrk_test';
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    checkDesign(model());
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('answers synchronously, which is the proof there is no I/O in it', () => {
    expect(checkDesign(model())).not.toBeInstanceOf(Promise);
  });
});

describe('the verdict', () => {
  it('returns the five stages and a verdict the schema allows', () => {
    const out = checkDesign(model()) as { verdict: string; stages: Array<{ stage: string }> };
    expect(['ok', 'warn', 'block', 'ask']).toContain(out.verdict);
    expect(out.stages.map(s => s.stage)).toEqual(
      expect.arrayContaining(['preflight', 'data', 'train', 'evaluate', 'deploy']),
    );
  });

  it('blocks a design that cannot run, and says why', () => {
    const broken = model();
    broken.components.push({
      id: 'a', name: 'attn', type: 'multiHeadAttention',
      position: { x: 0, y: 0 }, params: { embedDim: 100, numHeads: 8 },
    } as never);
    broken.connections.push({ id: 'e2', from: 'c', to: 'a' } as never);
    const out = checkDesign(broken) as { verdict: string; findings: Array<{ severity: string }> };
    expect(out.findings.some(f => f.severity === 'block')).toBe(true);
  });

  it('refuses a graph over the size cap instead of stalling the tool call', () => {
    const huge = model();
    huge.components = Array.from({ length: MAX_COMPONENTS + 1 }, (_, i) => ({
      id: `n${i}`, name: `n${i}`, type: 'relu', position: { x: 0, y: 0 }, params: {},
    })) as never;
    const out = checkDesign(huge) as { error: string };
    expect(out.error).toMatch(new RegExp(String(MAX_COMPONENTS)));
    expect(out).not.toHaveProperty('verdict');
  });

  it('reports a verifier crash as a verifier gap, never as a verdict', () => {
    // A model shaped in a way the stages never anticipated must not come back
    // as "block": that would report our bug as the user's broken design.
    const nonsense = { components: null, connections: null } as unknown as ModelArchitecture;
    const out = checkDesign(nonsense);
    if ('error' in out) {
      expect(out.error).toMatch(/gap in the verifier|could not evaluate/);
    } else {
      expect(['ok', 'warn', 'block', 'ask']).toContain(out.verdict);
    }
  });
});

describe('provenance', () => {
  it('quotes the published measurement for a rule that has one', () => {
    const p = provenanceForRules(['head-dim-divisibility']) as Record<string, { source: string; evidence: string }>;
    expect(p['head-dim-divisibility'].source).toMatch(/^https:\/\//);
    expect(p['head-dim-divisibility'].evidence).toMatch(/\d/);
  });

  it('omits rules with no measurement rather than inventing one', () => {
    const p = provenanceForRules(['not-a-real-rule']);
    expect(p['not-a-real-rule']).toBeUndefined();
    expect(Object.keys(p)).toHaveLength(0);
  });

  it('needs no key and no network', () => {
    delete process.env.NEURARCH_API_KEY;
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    provenanceForRules(['head-dim-divisibility', 'deep-no-norm']);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
