import { writeFile } from 'node:fs/promises';
import type { ToolDef } from './tools.js';
import { sourceKindFor } from './loader.js';
import {
  addLayer,
  modifyLayer,
  addConnection,
  deleteLayer,
  deleteConnection,
} from './lib/writeOps.js';

const addLayerTool: ToolDef = {
  name: 'add_layer',
  description:
    'Insert a new layer into the model. Provide the layer type, a unique name, optionally an existing layer to auto-connect from, and any layer-specific params. Returns the new layer id. Call save_model afterwards to persist to disk. WARNING: this mutates the model. Call layer_impact first when inserting into the middle of an existing path so you can warn the user about cascading shape changes.',
  inputSchema: {
    type: 'object',
    properties: {
      type: {
        type: 'string',
        description:
          'Layer type, e.g. "linear", "conv2d", "layerNorm", "multiHeadAttention".',
      },
      name: { type: 'string', description: 'Unique layer name.' },
      after: {
        type: 'string',
        description:
          'Existing layer (name or id) to wire the new one after.',
      },
      params: {
        type: 'object',
        description:
          'Hyperparameters by type, e.g. linear {in_features, out_features}, dropout {p}.',
      },
      scope: {
        type: 'string',
        description: 'Dotted module path, e.g. "encoder.layer.3".',
      },
    },
    required: ['type', 'name'],
    additionalProperties: false,
  },
  // Adds; never overwrites. Not idempotent: calling it twice with the same
  // name fails on the duplicate, and with a different name adds a second layer.
  annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
  handler: (args, model) => addLayer(model, args),
};

const modifyLayerTool: ToolDef = {
  name: 'modify_layer',
  description:
    'Modify an existing layer: shallow-merge new params, rename, or change scope. Returns a before/after diff. Cached shapes on this layer are invalidated when params change, so the user should reopen the model in Neurarch to recompute shape contracts. Call save_model afterwards. WARNING: shape-changing edits propagate downstream — call layer_impact first.',
  inputSchema: {
    type: 'object',
    properties: {
      name:   { type: 'string', description: 'Layer name or id to modify.' },
      params: { type: 'object', description: 'Shallow-merged into params.' },
      rename: { type: 'string', description: 'New unique name.' },
      scope:  { type: 'string', description: 'New dotted module path.' },
    },
    required: ['name'],
    additionalProperties: false,
  },
  // Overwrites existing params, so destructive in the sense the spec means:
  // the previous value is gone. Idempotent because re-applying the same patch
  // lands on the same state.
  annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
  handler: (args, model) => modifyLayer(model, args),
};

const addConnectionTool: ToolDef = {
  name: 'add_connection',
  description:
    'Wire two existing layers: a directed edge from "from" to "to". Fails on self-loops or duplicate edges. The target layer\'s cached shape is invalidated. Call save_model afterwards to persist.',
  inputSchema: {
    type: 'object',
    properties: {
      from:  { type: 'string', description: 'Source layer name or id.' },
      to:    { type: 'string', description: 'Target layer name or id.' },
      label: { type: 'string', description: 'Optional edge label.' },
    },
    required: ['from', 'to'],
    additionalProperties: false,
  },
  annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
  handler: (args, model) => addConnection(model, args),
};

const deleteLayerTool: ToolDef = {
  name: 'delete_layer',
  description:
    'Remove a layer and every connection touching it. Cached shapes on former downstream layers are invalidated and the layer is removed from any groups it belonged to. Returns the list of layers whose shape contracts must be recomputed and the groups affected. Call save_model afterwards. WARNING: this is destructive. Call layer_impact first if the layer sits in the middle of an active path.',
  inputSchema: {
    type: 'object',
    properties: {
      name: { type: 'string', description: 'Layer name or id to delete.' },
    },
    required: ['name'],
    additionalProperties: false,
  },
  // The most destructive tool here: it takes every edge touching the layer
  // with it. Idempotent only in the weak sense that a second call finds
  // nothing left to remove.
  annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
  handler: (args, model) => deleteLayer(model, args),
};

const deleteConnectionTool: ToolDef = {
  name: 'delete_connection',
  description:
    'Remove a single directed edge between two existing layers. The target layer\'s cached shape is invalidated. Fails when no such edge exists. Call save_model afterwards.',
  inputSchema: {
    type: 'object',
    properties: {
      from: { type: 'string', description: 'Source layer name or id.' },
      to:   { type: 'string', description: 'Target layer name or id.' },
    },
    required: ['from', 'to'],
    additionalProperties: false,
  },
  annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
  handler: (args, model) => deleteConnection(model, args),
};

const saveModelTool: ToolDef = {
  name: 'save_model',
  description:
    'Persist the current in-memory model to disk. Writes to the file passed on the command line by default; pass "path" to write elsewhere (will overwrite). Call this AFTER add_layer / modify_layer / add_connection — those tools only mutate memory. Returns the written path and byte count.',
  inputSchema: {
    type: 'object',
    properties: {
      path: {
        type: 'string',
        description: 'Default: the startup file.',
      },
    },
    additionalProperties: false,
  },
  // Overwrites a file on disk, which is the only irreversible thing this
  // server does. A client that confirms exactly one tool should confirm this
  // one.
  annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
  handler: async (args: { path?: string }, model, ctx) => {
    const target = args.path ?? ctx.modelPath;
    if (!target) throw new Error('save_model: no path available.');
    // The model serializes as JSON. Writing it to a .py would destroy the
    // source file and leave something Python cannot import.
    if (sourceKindFor(target) === 'pytorch-source') {
      throw new Error(
        `save_model: refusing to write JSON over ${target}. This server does not generate Python; `
        + 'pass a .neurarch.json path instead.',
      );
    }
    const json = JSON.stringify(model, null, 2);
    await writeFile(target, json, 'utf-8');
    return { ok: true, written: target, sizeBytes: Buffer.byteLength(json, 'utf-8') };
  },
};

export const WRITE_TOOLS: ToolDef[] = [
  addLayerTool,
  modifyLayerTool,
  addConnectionTool,
  deleteLayerTool,
  deleteConnectionTool,
  saveModelTool,
];
