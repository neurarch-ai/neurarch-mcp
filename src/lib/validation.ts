/**
 * Lightweight model validator.
 *
 * Runs structural invariants that the Neurarch app's guardrail pipeline also
 * enforces, but in a form an MCP-aware agent can call before recommending a
 * destructive edit:
 *   - cycles in the connection graph
 *   - dangling connection endpoints (from/to point to a nonexistent component)
 *   - duplicate component ids or names
 *   - orphan components (no upstream and not an `input`, or no downstream and not an `output`)
 *
 * Cheap, pure, sub-millisecond on typical models.
 */
import type { MLComponent, ModelArchitecture } from './types.js';

export type ValidationSeverity = 'error' | 'warn';

export interface ValidationFinding {
  rule: 'cycle' | 'dangling-connection' | 'duplicate-id' | 'duplicate-name' | 'orphan';
  severity: ValidationSeverity;
  message: string;
  componentIds?: string[];
  componentNames?: string[];
}

export interface ValidationReport {
  ok: boolean;
  findings: ValidationFinding[];
  totals: {
    errors: number;
    warnings: number;
  };
}

export function validateModel(model: ModelArchitecture): ValidationReport {
  const findings: ValidationFinding[] = [];
  const compById = new Map<string, MLComponent>();

  for (const c of model.components) {
    if (compById.has(c.id)) {
      findings.push({
        rule: 'duplicate-id',
        severity: 'error',
        message: `Two components share id "${c.id}".`,
        componentIds: [c.id],
        componentNames: [c.name],
      });
    } else {
      compById.set(c.id, c);
    }
  }

  const seenNames = new Map<string, string[]>();
  for (const c of model.components) {
    const arr = seenNames.get(c.name) ?? [];
    arr.push(c.id);
    seenNames.set(c.name, arr);
  }
  for (const [name, ids] of seenNames) {
    if (ids.length > 1) {
      findings.push({
        rule: 'duplicate-name',
        severity: 'warn',
        message: `${ids.length} components share name "${name}". Lookups by name are ambiguous.`,
        componentIds: ids,
        componentNames: [name],
      });
    }
  }

  for (const conn of model.connections) {
    const missing: string[] = [];
    if (!compById.has(conn.from)) missing.push(`from=${conn.from}`);
    if (!compById.has(conn.to)) missing.push(`to=${conn.to}`);
    if (missing.length) {
      findings.push({
        rule: 'dangling-connection',
        severity: 'error',
        message: `Connection ${conn.id} references missing component(s): ${missing.join(', ')}.`,
      });
    }
  }

  // Cycle detection via iterative DFS with grey/black coloring.
  const adj = new Map<string, string[]>();
  for (const c of model.components) adj.set(c.id, []);
  for (const e of model.connections) {
    if (!adj.has(e.from) || !adj.has(e.to)) continue;
    adj.get(e.from)!.push(e.to);
  }
  const WHITE = 0, GREY = 1, BLACK = 2;
  const color = new Map<string, number>();
  for (const id of adj.keys()) color.set(id, WHITE);

  const cyclesReported = new Set<string>();
  for (const start of adj.keys()) {
    if (color.get(start) !== WHITE) continue;
    const stack: Array<{ id: string; itIdx: number; path: string[] }> = [
      { id: start, itIdx: 0, path: [start] },
    ];
    color.set(start, GREY);
    while (stack.length) {
      const frame = stack[stack.length - 1];
      const neighbors = adj.get(frame.id) ?? [];
      if (frame.itIdx >= neighbors.length) {
        color.set(frame.id, BLACK);
        stack.pop();
        continue;
      }
      const next = neighbors[frame.itIdx++];
      const c = color.get(next) ?? WHITE;
      if (c === GREY) {
        const idx = frame.path.indexOf(next);
        const loop = idx >= 0 ? frame.path.slice(idx) : [next];
        loop.push(next);
        const key = [...loop].sort().join('|');
        if (!cyclesReported.has(key)) {
          cyclesReported.add(key);
          const names = loop.map(id => compById.get(id)?.name ?? id);
          findings.push({
            rule: 'cycle',
            severity: 'error',
            message: `Cycle: ${names.join(' -> ')}.`,
            componentIds: loop,
            componentNames: names,
          });
        }
      } else if (c === WHITE) {
        color.set(next, GREY);
        stack.push({ id: next, itIdx: 0, path: [...frame.path, next] });
      }
    }
  }

  const hasUpstream = new Set<string>();
  const hasDownstream = new Set<string>();
  for (const e of model.connections) {
    if (compById.has(e.to))   hasUpstream.add(e.to);
    if (compById.has(e.from)) hasDownstream.add(e.from);
  }
  if (model.components.length > 1) {
    for (const c of model.components) {
      const isInput  = c.type === 'input';
      const isOutput = c.type === 'output';
      const isSticky = c.type === 'stickyNote';
      if (isSticky) continue;
      const upstream   = hasUpstream.has(c.id);
      const downstream = hasDownstream.has(c.id);
      if (!upstream && !downstream) {
        findings.push({
          rule: 'orphan',
          severity: 'warn',
          message: `"${c.name}" has no connections.`,
          componentIds: [c.id],
          componentNames: [c.name],
        });
      } else if (!upstream && !isInput) {
        findings.push({
          rule: 'orphan',
          severity: 'warn',
          message: `"${c.name}" has no upstream and is not an input layer.`,
          componentIds: [c.id],
          componentNames: [c.name],
        });
      } else if (!downstream && !isOutput) {
        findings.push({
          rule: 'orphan',
          severity: 'warn',
          message: `"${c.name}" has no downstream and is not an output layer.`,
          componentIds: [c.id],
          componentNames: [c.name],
        });
      }
    }
  }

  const errors = findings.filter(f => f.severity === 'error').length;
  const warnings = findings.filter(f => f.severity === 'warn').length;
  return {
    ok: errors === 0,
    findings,
    totals: { errors, warnings },
  };
}
