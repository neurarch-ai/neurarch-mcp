import { describe, it, expect } from 'vitest';
import { compileUserRegExp, tryCompileUserRegExp, MAX_PATTERN_LENGTH } from './regexGuard.js';

describe('compileUserRegExp', () => {
  it('compiles ordinary layer-name patterns', () => {
    expect(compileUserRegExp('^encoder\\.\\d+').test('encoder.3')).toBe(true);
    expect(compileUserRegExp('attn|mlp').test('mlp')).toBe(true);
  });

  it('rejects an over-long source', () => {
    const long = 'a'.repeat(MAX_PATTERN_LENGTH + 1);
    expect(() => compileUserRegExp(long)).toThrow(/too long/);
  });

  it('rejects a catastrophic nested-quantifier shape', () => {
    expect(() => compileUserRegExp('(a+)+')).toThrow(/catastrophic/);
    expect(() => compileUserRegExp('(x*)*')).toThrow(/catastrophic/);
  });

  it('rejects a syntactically invalid pattern', () => {
    expect(() => compileUserRegExp('(')).toThrow();
  });
});

describe('tryCompileUserRegExp', () => {
  it('returns a RegExp on success and null on failure', () => {
    expect(tryCompileUserRegExp('block_\\d')).toBeInstanceOf(RegExp);
    expect(tryCompileUserRegExp('(a+)+')).toBeNull();
    expect(tryCompileUserRegExp('(')).toBeNull();
  });
});
