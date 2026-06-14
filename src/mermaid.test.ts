import { describe, it, expect } from 'vitest';
import { renderMermaid } from './mermaid.js';
import { makeModel } from './test/fixtures.js';

describe('renderMermaid', () => {
  it('emits a top-down flowchart header', () => {
    expect(renderMermaid(makeModel())).toMatch(/^flowchart TD/);
  });

  it('renders a node per component with name + type', () => {
    const out = renderMermaid(makeModel());
    expect(out).toContain('block_0');
    expect(out).toContain('(transformerBlock)');
  });

  it('renders edges, including labelled ones', () => {
    const out = renderMermaid(makeModel());
    expect(out).toMatch(/N\d+ --> N\d+/);
    expect(out).toContain('|residual|');
  });

  it('escapes double quotes in labels', () => {
    const m = makeModel();
    m.components[0].name = 'in"jected';
    expect(renderMermaid(m)).toContain('in\\"jected');
  });

  it('wraps grouped layers in a Mermaid subgraph', () => {
    const out = renderMermaid(makeModel());
    expect(out).toContain('subgraph G0["encoder"]');
    expect(out).toContain('end');
    // both block members are declared exactly once (inside the subgraph)
    const blockNodeDecls = out.match(/N\d+\["block_\d/g) ?? [];
    expect(blockNodeDecls).toHaveLength(2);
  });

  it('renders no subgraph when the model has no groups', () => {
    const m = makeModel();
    delete m.groups;
    expect(renderMermaid(m)).not.toContain('subgraph');
  });

  it('skips edges with dangling endpoints', () => {
    const m = makeModel();
    m.connections.push({ id: 'x', from: 'ghost', to: 'in', fromPort: 'bottom', toPort: 'top' });
    // should not throw and should not emit a NaN node reference
    expect(() => renderMermaid(m)).not.toThrow();
    expect(renderMermaid(m)).not.toContain('Nundefined');
  });
});
