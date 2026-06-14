import { describe, it, expect } from 'vitest';
import {
  addLayer, modifyLayer, addConnection, deleteLayer, deleteConnection,
} from './writeOps.js';
import { makeModel } from '../test/fixtures.js';

describe('addLayer', () => {
  it('appends a layer and auto-wires it after an existing one', () => {
    const m = makeModel();
    const before = m.components.length;
    const res = addLayer(m, { type: 'dropout', name: 'drop_0', after: 'block_1', params: { p: 0.1 } });
    expect(m.components).toHaveLength(before + 1);
    expect(res.connectedTo).toBe('block_1');
    expect(res.connectionId).toBeDefined();
    const created = m.components.find(c => c.id === res.id)!;
    const parent = m.components.find(c => c.name === 'block_1')!;
    expect(created.inputs).toContain(parent.id);
    expect(parent.outputs).toContain(created.id);
    expect(m.connections.some(c => c.from === parent.id && c.to === res.id)).toBe(true);
  });

  it('adds a free-floating layer when no "after" is given', () => {
    const m = makeModel();
    const res = addLayer(m, { type: 'relu', name: 'act' });
    expect(res.connectedTo).toBeUndefined();
    expect(res.connectionId).toBeUndefined();
  });

  it('throws on duplicate name', () => {
    const m = makeModel();
    expect(() => addLayer(m, { type: 'relu', name: 'block_0' })).toThrow(/already exists/);
  });

  it('throws when "after" cannot be resolved', () => {
    const m = makeModel();
    expect(() => addLayer(m, { type: 'relu', name: 'x', after: 'nope_xyz' })).toThrow(/cannot find/);
  });
});

describe('modifyLayer', () => {
  it('shallow-merges params and invalidates cached shapes', () => {
    const m = makeModel();
    const res = modifyLayer(m, { name: 'lm_head', params: { vocabSize: 50000 } });
    expect(res.invalidatedShapes).toBe(true);
    expect(res.after.params.vocabSize).toBe(50000);
    expect(res.after.params.embedDim).toBe(256); // preserved by shallow merge
    const comp = m.components.find(c => c.name === 'lm_head')!;
    expect(comp.outputShape).toBeUndefined();
  });

  it('renames a layer and returns a before/after diff', () => {
    const m = makeModel();
    const res = modifyLayer(m, { name: 'block_0', rename: 'encoder_block_0' });
    expect(res.before.name).toBe('block_0');
    expect(res.after.name).toBe('encoder_block_0');
    expect(res.invalidatedShapes).toBe(false);
  });

  it('throws when renaming onto an existing name', () => {
    const m = makeModel();
    expect(() => modifyLayer(m, { name: 'block_0', rename: 'block_1' })).toThrow(/already in use/);
  });

  it('throws on unknown layer', () => {
    const m = makeModel();
    expect(() => modifyLayer(m, { name: 'ghost_xyz' })).toThrow(/cannot find/);
  });
});

describe('addConnection', () => {
  it('wires two existing layers and updates inputs/outputs', () => {
    const m = makeModel();
    const res = addConnection(m, { from: 'embed', to: 'final_norm', label: 'skip' });
    expect(res.from).toBe('embed');
    expect(res.to).toBe('final_norm');
    const conn = m.connections.find(c => c.id === res.id)!;
    expect(conn.label).toBe('skip');
  });

  it('rejects self-loops', () => {
    const m = makeModel();
    expect(() => addConnection(m, { from: 'block_0', to: 'block_0' })).toThrow(/itself/);
  });

  it('rejects duplicate edges', () => {
    const m = makeModel();
    expect(() => addConnection(m, { from: 'embed', to: 'block_0' })).toThrow(/already exists/);
  });
});

describe('deleteLayer', () => {
  it('removes a layer and every edge touching it', () => {
    const m = makeModel();
    const res = deleteLayer(m, { name: 'block_1' });
    expect(m.components.some(c => c.name === 'block_1')).toBe(false);
    expect(res.removedConnections).toBe(2); // blk0->blk1 and blk1->norm
    expect(res.invalidatedDownstream).toContain('final_norm');
    expect(res.removedFromGroups).toContain('encoder');
    // upstream parent no longer lists the deleted id
    const blk0 = m.components.find(c => c.name === 'block_0')!;
    expect(blk0.outputs).not.toContain(res.id);
  });

  it('throws on unknown layer', () => {
    const m = makeModel();
    expect(() => deleteLayer(m, { name: 'ghost_xyz' })).toThrow(/cannot find/);
  });
});

describe('deleteConnection', () => {
  it('removes a single directed edge and invalidates the target shape', () => {
    const m = makeModel();
    const res = deleteConnection(m, { from: 'block_0', to: 'final_norm' });
    expect(res.from).toBe('block_0');
    expect(m.connections.some(c => c.from === 'blk0' && c.to === 'norm')).toBe(false);
    const norm = m.components.find(c => c.name === 'final_norm')!;
    expect(norm.outputShape).toBeUndefined();
  });

  it('throws when the edge does not exist', () => {
    const m = makeModel();
    expect(() => deleteConnection(m, { from: 'lm_head', to: 'embed' })).toThrow(/no edge/);
  });
});
