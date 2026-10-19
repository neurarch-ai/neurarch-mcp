import { describe, it, expect } from 'vitest';
import { TOOLS } from './tools.js';
import { loadZooModel } from './sources.js';

const call = async (name: string, args: Record<string, unknown>, model: Awaited<ReturnType<typeof loadZooModel>>) =>
  (await TOOLS.find(t => t.name === name)!.handler(args, model, { modelPath: '', writeEnabled: false } as never)) as Record<string, any>;

describe('response_format', () => {
  it('lint_model concise keeps every block and warn, counts the info it left out, and detailed has them all', async () => {
    const m = await loadZooModel('qwen2.5-7b');
    const c = await call('lint_model', {}, m);
    const d = await call('lint_model', { response_format: 'detailed' }, m);
    expect(d.findings.length).toBe(c.counts.block + c.counts.warn + c.counts.info);
    expect(c.findings.every((f: any) => f.severity !== 'info')).toBe(true);
    expect(c.omitted.count).toBe(c.counts.info);
    expect(Object.values(c.infoByRule as Record<string, number>).reduce((a, b) => a + b, 0)).toBe(c.counts.info);
    expect(c.omitted.how).toMatch(/response_format: "detailed"/);
    expect(d.omitted).toBeUndefined();
    expect(JSON.stringify(c).length).toBeLessThan(JSON.stringify(d).length / 3);
  });

  it('find_layers concise drops per-row fields, says so, and keeps paging', async () => {
    const m = await loadZooModel('qwen2.5-7b');
    const c = await call('find_layers', {}, m);
    const d = await call('find_layers', { response_format: 'detailed' }, m);
    expect(c.layers.length).toBe(d.layers.length);
    expect(c.nextCursor).toBe(d.nextCursor);
    expect(c.layers[0]).not.toHaveProperty('paramCountFormatted');
    expect(d.layers[0]).toHaveProperty('paramCountFormatted');
    expect(c.fields).toMatch(/detailed/);
  });

  it('layer_impact concise caps each direction at 25 and counts the rest; totals are unchanged', async () => {
    const m = await loadZooModel('qwen2.5-7b');
    const target = m.components.find(c => c.type === 'embedding')!.name;
    const c = await call('layer_impact', { names: [target] }, m);
    const d = await call('layer_impact', { names: [target], response_format: 'detailed' }, m);
    expect(c.downstream.length).toBe(Math.min(25, d.downstream.length));
    expect(c.totals).toEqual(d.totals);
    if (d.downstream.length > 25) expect(c.omitted.count).toBe(d.downstream.length - 25 + Math.max(0, d.upstream.length - 25));
  });

  it('describe_architecture concise keeps the top 3 hotspots; check_design detailed equals verbose', async () => {
    const m = await loadZooModel('resnet-50');
    const c = await call('describe_architecture', {}, m);
    expect(c.paramHotspots.length).toBeLessThanOrEqual(3);
    expect(c.omitted?.count ?? 0).toBeGreaterThanOrEqual(0);
    const v = await call('check_design', { verbose: true }, m);
    const d = await call('check_design', { response_format: 'detailed' }, m);
    expect(d).toEqual(v);
  });
});
