import type { MLComponent, ModelArchitecture } from './lib/types.js';

/**
 * Render a ModelArchitecture as a Mermaid flowchart (top-down).
 * Output is plain text — the consumer (e.g., an agent rendering markdown)
 * pastes it inside ```mermaid``` fences.
 *
 * Node IDs are sanitized for Mermaid syntax: only [A-Za-z0-9_] is safe, so we
 * use the array index. The component name is preserved as the label.
 */
export function renderMermaid(model: ModelArchitecture): string {
  const idToIdx = new Map<string, number>();
  model.components.forEach((c, i) => idToIdx.set(c.id, i));

  const lines: string[] = ['flowchart TD'];

  // Declare each node once. Grouped layers are wrapped in a Mermaid `subgraph`
  // so the diagram mirrors the model's blocks; ungrouped layers sit at the top
  // level. Cross-subgraph edges are emitted normally afterwards.
  const declared = new Set<string>();
  const declareNode = (c: typeof model.components[number], i: number, indent: string) => {
    lines.push(`${indent}N${i}["${escapeLabel(formatNodeLabel(c))}"]`);
    declared.add(c.id);
  };

  const groups = model.groups ?? [];
  groups.forEach((g, gi) => {
    const members = g.componentIds
      .map(id => idToIdx.get(id))
      .filter((i): i is number => i !== undefined && !declared.has(model.components[i].id));
    if (!members.length) return;
    lines.push(`  subgraph G${gi}["${escapeLabel(g.name)}"]`);
    for (const i of members) declareNode(model.components[i], i, '    ');
    lines.push('  end');
  });

  for (const [i, c] of model.components.entries()) {
    if (declared.has(c.id)) continue;
    declareNode(c, i, '  ');
  }

  for (const conn of model.connections) {
    const a = idToIdx.get(conn.from);
    const b = idToIdx.get(conn.to);
    if (a === undefined || b === undefined) continue;
    if (conn.label) lines.push(`  N${a} -->|${escapeLabel(conn.label)}| N${b}`);
    else            lines.push(`  N${a} --> N${b}`);
  }

  return lines.join('\n');
}

function formatNodeLabel(c: MLComponent): string {
  const shape = Array.isArray(c.outputShape) && c.outputShape.length
    ? `\\n${c.outputShape.join('×')}`
    : '';
  return `${c.name}\\n(${c.type})${shape}`;
}

function escapeLabel(s: string): string {
  return s.replace(/"/g, '\\"');
}
