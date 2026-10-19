/**
 * One way to page a long list, for every tool that returns one.
 *
 * Before 0.20 an unfiltered list_architectures answered with 41 KB and
 * find_layers on a 7B model with 24 KB, all of it landing in the agent's
 * context whether it was wanted or not. A page is `limit` items plus an opaque
 * `nextCursor` (null on the last page), the same shape MCP uses for its own
 * list methods, so an agent that has paged a resource list already knows it.
 *
 * The cursor is the offset, base64-encoded so nobody is tempted to do
 * arithmetic on it; the list underneath is deterministic for a given model and
 * arguments, which is what makes an offset a valid position.
 */
export const PAGE_PARAMS = {
  limit: { type: 'integer', description: 'Page size.' },
  cursor: { type: 'string', description: 'Previous nextCursor.' },
} as const;

export interface Page<T> {
  items: T[];
  total: number;
  nextCursor: string | null;
}

export function encodeCursor(offset: number): string {
  return Buffer.from(`o:${offset}`, 'utf-8').toString('base64url');
}

export function decodeCursor(cursor: string): number {
  const raw = Buffer.from(cursor, 'base64url').toString('utf-8');
  const m = /^o:(\d+)$/.exec(raw);
  if (!m) throw new Error(`Invalid cursor "${cursor}": pass the nextCursor value from the previous page unchanged, or omit cursor for the first page.`);
  return Number(m[1]);
}

export function pageOf<T>(all: T[], args: { limit?: number; cursor?: string }, defaultLimit: number): Page<T> {
  const limit = typeof args.limit === 'number' && args.limit > 0 ? Math.min(500, Math.floor(args.limit)) : defaultLimit;
  const offset = args.cursor ? decodeCursor(args.cursor) : 0;
  const items = all.slice(offset, offset + limit);
  const next = offset + limit;
  return { items, total: all.length, nextCursor: next < all.length ? encodeCursor(next) : null };
}
