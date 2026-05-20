import { randomUUID } from 'node:crypto';
import type {
  ComponentConnection,
  ComponentType,
  MLComponent,
  ModelArchitecture,
} from './types.js';
import { resolveTargets } from './modelImpact.js';

export interface AddLayerInput {
  type: string;
  name: string;
  after?: string;
  params?: Record<string, unknown>;
  scope?: string;
}

export interface AddLayerResult {
  id: string;
  name: string;
  type: string;
  connectedTo?: string;
  connectionId?: string;
}

export function addLayer(model: ModelArchitecture, input: AddLayerInput): AddLayerResult {
  const { type, name, after, params, scope } = input;
  if (!type || typeof type !== 'string') throw new Error('add_layer: "type" is required.');
  if (!name || typeof name !== 'string') throw new Error('add_layer: "name" is required.');
  if (model.components.some(c => c.name === name)) {
    throw new Error(`add_layer: name "${name}" already exists. Pick a unique name.`);
  }

  let position = { x: 0, y: 0 };
  let afterComp: MLComponent | undefined;
  if (after) {
    const { resolved, unresolved } = resolveTargets(model, [after]);
    if (unresolved.length || !resolved[0]) {
      throw new Error(`add_layer: cannot find "after" target "${after}".`);
    }
    afterComp = resolved[0];
    position = { x: afterComp.position.x, y: afterComp.position.y + 100 };
  }

  const id = randomUUID();
  const comp: MLComponent = {
    id,
    type: type as ComponentType,
    name,
    position,
    params: (params as Record<string, unknown>) ?? {},
    inputs: [],
    outputs: [],
    ...(scope ? { scope } : {}),
  };
  model.components.push(comp);

  const result: AddLayerResult = { id, name, type };
  if (afterComp) {
    const connId = randomUUID();
    model.connections.push({
      id: connId,
      from: afterComp.id,
      to: id,
      fromPort: 'bottom',
      toPort: 'top',
    });
    if (!afterComp.outputs.includes(id)) afterComp.outputs.push(id);
    if (!comp.inputs.includes(afterComp.id)) comp.inputs.push(afterComp.id);
    result.connectedTo = afterComp.name;
    result.connectionId = connId;
  }
  return result;
}

export interface ModifyLayerInput {
  name: string;
  params?: Record<string, unknown>;
  rename?: string;
  scope?: string;
}

export interface ModifyLayerResult {
  id: string;
  before: { name: string; params: Record<string, unknown>; scope?: string };
  after: { name: string; params: Record<string, unknown>; scope?: string };
  invalidatedShapes: boolean;
}

export function modifyLayer(model: ModelArchitecture, input: ModifyLayerInput): ModifyLayerResult {
  const { name, params, rename, scope } = input;
  if (!name) throw new Error('modify_layer: "name" is required.');

  const { resolved, unresolved } = resolveTargets(model, [name]);
  if (unresolved.length || !resolved[0]) {
    throw new Error(`modify_layer: cannot find layer "${name}".`);
  }
  const comp = resolved[0];

  if (rename && rename !== comp.name) {
    if (model.components.some(c => c.name === rename && c.id !== comp.id)) {
      throw new Error(`modify_layer: name "${rename}" already in use.`);
    }
  }

  const before = {
    name: comp.name,
    params: { ...comp.params },
    ...(comp.scope !== undefined ? { scope: comp.scope } : {}),
  };

  let shapeAffected = false;
  if (params) {
    comp.params = { ...comp.params, ...params };
    shapeAffected = true;
  }
  if (rename) comp.name = rename;
  if (scope !== undefined) comp.scope = scope;

  if (shapeAffected) {
    delete comp.inputShape;
    delete comp.outputShape;
  }

  return {
    id: comp.id,
    before,
    after: {
      name: comp.name,
      params: { ...comp.params },
      ...(comp.scope !== undefined ? { scope: comp.scope } : {}),
    },
    invalidatedShapes: shapeAffected,
  };
}

export interface AddConnectionInput {
  from: string;
  to: string;
  label?: string;
}

export interface AddConnectionResult {
  id: string;
  from: string;
  to: string;
}

export function addConnection(
  model: ModelArchitecture,
  input: AddConnectionInput,
): AddConnectionResult {
  const { from, to, label } = input;
  if (!from || !to) throw new Error('add_connection: "from" and "to" are required.');

  const { resolved: fromList, unresolved: fromUnresolved } = resolveTargets(model, [from]);
  if (fromUnresolved.length || !fromList[0]) {
    throw new Error(`add_connection: cannot find "from" layer "${from}".`);
  }
  const { resolved: toList, unresolved: toUnresolved } = resolveTargets(model, [to]);
  if (toUnresolved.length || !toList[0]) {
    throw new Error(`add_connection: cannot find "to" layer "${to}".`);
  }
  const fromComp = fromList[0];
  const toComp = toList[0];

  if (fromComp.id === toComp.id) {
    throw new Error('add_connection: cannot connect a layer to itself.');
  }
  if (model.connections.some(c => c.from === fromComp.id && c.to === toComp.id)) {
    throw new Error(
      `add_connection: edge from "${fromComp.name}" to "${toComp.name}" already exists.`,
    );
  }

  const id = randomUUID();
  const connection: ComponentConnection = {
    id,
    from: fromComp.id,
    to: toComp.id,
    fromPort: 'bottom',
    toPort: 'top',
    ...(label ? { label } : {}),
  };
  model.connections.push(connection);

  if (!fromComp.outputs.includes(toComp.id)) fromComp.outputs.push(toComp.id);
  if (!toComp.inputs.includes(fromComp.id)) toComp.inputs.push(fromComp.id);

  delete toComp.inputShape;
  delete toComp.outputShape;

  return { id, from: fromComp.name, to: toComp.name };
}
