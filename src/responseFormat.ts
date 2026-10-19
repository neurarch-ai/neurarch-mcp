/**
 * `response_format: "concise" | "detailed"` for the tools whose answer can run
 * to kilobytes (lint_model on a 7B model was 10.6 KB, find_layers 10.5 KB).
 *
 * Concise is the default, and it never hides what it left out: every concise
 * result that dropped anything carries `omitted`, saying how much and the one
 * argument that brings it back. A count of what is missing is the difference
 * between a summary and a silent truncation, and an agent that sees "29 info
 * findings omitted" can decide whether it needs them.
 */
export type ResponseFormat = 'concise' | 'detailed';

export const RESPONSE_FORMAT_PARAM = {
  response_format: { type: 'string', enum: ['concise', 'detailed'], description: 'Default concise.' },
} as const;

export const isDetailed = (args: { response_format?: string } | undefined): boolean => args?.response_format === 'detailed';

export interface Omitted {
  count: number;
  what: string;
  how: string;
}

export function omitted(count: number, what: string): { omitted?: Omitted } {
  return count > 0 ? { omitted: { count, what, how: 'Pass response_format: "detailed" for all of it.' } } : {};
}
