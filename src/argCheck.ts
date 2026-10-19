/**
 * Argument checking at the dispatch boundary, against the schema the client
 * was shown. Deliberately small (required, unknown, primitive type, enum,
 * integer bounds): the point is an error an agent can act on in one retry,
 * naming the argument and what it accepts, not a full JSON Schema engine.
 */
type Prop = { type?: string | string[]; enum?: unknown[]; minimum?: number; maximum?: number; description?: string };

function typeOf(v: unknown): string {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  if (typeof v === 'number') return Number.isInteger(v) ? 'integer' : 'number';
  return typeof v;
}

function matches(v: unknown, t: string): boolean {
  const actual = typeOf(v);
  if (t === 'number') return actual === 'number' || actual === 'integer';
  return actual === t;
}

export function validateArgs(tool: string, schema: Record<string, unknown>, raw: unknown): string | null {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    return `${tool}: arguments must be an object of named arguments.`;
  }
  const args = raw as Record<string, unknown>;
  const props = (schema.properties ?? {}) as Record<string, Prop>;
  const names = Object.keys(props);
  const accepted = names.length ? `Accepted: ${names.join(', ')}.` : 'It takes no arguments.';
  const problems: string[] = [];

  if (schema.additionalProperties === false) {
    const unknown = Object.keys(args).filter(k => !(k in props));
    if (unknown.length) problems.push(`unknown argument${unknown.length > 1 ? 's' : ''} ${unknown.map(k => `"${k}"`).join(', ')}`);
  }
  for (const r of (schema.required as string[] | undefined) ?? []) {
    if (args[r] === undefined || args[r] === null || args[r] === '') {
      const d = props[r]?.description;
      problems.push(`missing required "${r}"${d ? ` (${d.replace(/\.$/, '')})` : ''}`);
    }
  }
  for (const [k, v] of Object.entries(args)) {
    const p = props[k];
    if (!p || v === undefined) continue;
    const types = p.type === undefined ? [] : Array.isArray(p.type) ? p.type : [p.type];
    if (types.length && !types.some(t => matches(v, t))) {
      problems.push(`"${k}" must be ${types.join(' or ')}, got ${typeOf(v)}`);
      continue;
    }
    if (p.enum && !p.enum.includes(v)) problems.push(`"${k}" must be one of ${p.enum.map(e => JSON.stringify(e)).join(', ')}`);
    if (typeof v === 'number') {
      if (p.minimum !== undefined && v < p.minimum) problems.push(`"${k}" must be >= ${p.minimum}`);
      if (p.maximum !== undefined && v > p.maximum) problems.push(`"${k}" must be <= ${p.maximum}`);
    }
  }
  if (!problems.length) return null;
  return `${tool}: ${problems.join('; ')}. ${accepted} Full contract: neurarch://docs/${tool}.`;
}
