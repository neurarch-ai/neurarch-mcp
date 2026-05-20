import { writeFile } from 'node:fs/promises';
import type { ToolDef } from './tools.js';
import { addLayer, modifyLayer, addConnection } from './lib/writeOps.js';

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
          'Layer type. Examples: "linear", "conv2d", "layerNorm", "multiHeadAttention", "dropout", "embedding". The full list lives in the ComponentType union in the Neurarch schema.',
      },
      name: { type: 'string', description: 'Unique layer name. Fails if duplicate.' },
      after: {
        type: 'string',
        description:
          'Optional. Name or id of an existing layer. When provided, the new layer is auto-connected as a downstream of it (single edge, from bottom to top).',
      },
      params: {
        type: 'object',
        description:
          'Layer hyperparameters, shape varies by type. For example linear takes {in_features: 768, out_features: 256}; conv2d takes {in_channels, out_channels, kernel_size, stride, padding}; dropout takes {p: 0.1}.',
      },
      scope: {
        type: 'string',
        description: 'Optional dotted module path, e.g. "encoder.layer.3". Used by the Scope Folder panel in the UI.',
      },
    },
    required: ['type', 'name'],
    additionalProperties: false,
  },
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
      params: { type: 'object', description: 'Partial params object, shallow-merged with existing params.' },
      rename: { type: 'string', description: 'Optional new name. Fails if the new name is already in use elsewhere.' },
      scope:  { type: 'string', description: 'Optional new scope (dotted module path).' },
    },
    required: ['name'],
    additionalProperties: false,
  },
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
  handler: (args, model) => addConnection(model, args),
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
        description: 'Optional override path. Defaults to the file loaded at startup.',
      },
    },
    additionalProperties: false,
  },
  handler: async (args: { path?: string }, model, ctx) => {
    const target = args.path ?? ctx.modelPath;
    if (!target) throw new Error('save_model: no path available.');
    const json = JSON.stringify(model, null, 2);
    await writeFile(target, json, 'utf-8');
    return { ok: true, written: target, sizeBytes: Buffer.byteLength(json, 'utf-8') };
  },
};

export const WRITE_TOOLS: ToolDef[] = [
  addLayerTool,
  modifyLayerTool,
  addConnectionTool,
  saveModelTool,
];
