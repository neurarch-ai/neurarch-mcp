// Types for the slice of policy.bundle.mjs this server calls. The bundle is a
// byte-for-byte copy of neurarch-ai/neurarch packages/neurarch-cli/policy.bundle.mjs
// (`npm run build:policy-cli` there, from src/utils/policy/index.ts); the
// pin-check sources are src/utils/policy/pinCheck.ts. Not edited here.
export type Severity = 'block' | 'warn' | 'info';
export type PinRule = 'pin/hf-unpinned' | 'pin/hf-floating-revision' | 'pin/hf-unpinned-dynamic';
export type PinFileKind = 'python' | 'notebook' | 'shell' | 'yaml';
export type PinSyntax = 'python' | 'shell' | 'yaml';

export interface PinHit {
  rule: PinRule;
  path: string;
  line: number;
  api: string;
  repo?: string;
  repoType: 'model' | 'dataset';
  revision?: string;
  syntax: PinSyntax;
}

export interface PolicyFinding {
  checker: string;
  rule: string;
  severity: Severity;
  title: string;
  detail?: string;
  fix?: string;
  model?: string;
  path?: string;
  line?: number;
}

export const PIN_RULE_META: Record<PinRule, { code: string; severity: Severity; bandit: string; summary: string }>;
export const RENOVATE_PRESET_URL: string;
export const CHANGE_FEEDS_URL: string;
export function pinFileKind(path: string): PinFileKind | null;
export function scanPins(path: string, text: string): PinHit[];
export function isPinnedRevision(rev: string): boolean;
export function pinSpelling(syntax: PinSyntax, sha: string): string;
export function fromPinCheck(hits: PinHit[], opts?: { shas?: Record<string, string>; resolvedOn?: string }): PolicyFinding[];
