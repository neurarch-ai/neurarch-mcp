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

  for (const [i, c] of model.components.entries()) {
    lines.push(`  N${i}["${escapeLabel(formatNodeLabel(c))}"]`);
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
