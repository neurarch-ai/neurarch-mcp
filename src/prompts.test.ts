import { describe, it, expect } from 'vitest';
import { PROMPTS, renderPrompt } from './prompts.js';
import { TOOLS } from './tools.js';
import { WRITE_TOOLS } from './writeTools.js';

const KNOWN = new Set([...TOOLS, ...WRITE_TOOLS].map(t => t.name).concat(['load_hf_model']));

describe('prompts', () => {
  it('have unique names and render without arguments where none are required', () => {
    expect(new Set(PROMPTS.map(p => p.name)).size).toBe(PROMPTS.length);
    for (const p of PROMPTS) {
      const args: Record<string, string> = {};
      for (const a of p.arguments ?? []) if (a.required) args[a.name] = 'x';
      expect(renderPrompt(p, args, { writeEnabled: false }).length).toBeGreaterThan(200);
    }
  });

  it('only ever name tools that exist', () => {
    for (const p of PROMPTS) {
      const text = renderPrompt(p, { target: 't', rule: 'r', architecture: 'a', focus: 'f' }, { writeEnabled: true });
      for (const m of text.matchAll(/\b([a-z]+(?:_[a-z]+)+)\b/g)) {
        const name = m[1];
        // Only tokens that look like our tool names; other snake_case is prose (head_dim, save_to...).
        if (['head_dim', 'save_to', 'model_path', 'include_current', 'model_source', 'input_ids', 'num_layers', 'n_layers'].includes(name)) continue;
        if (KNOWN.has(name) || !/^(get|list|find|describe|validate|lint|check|rank|export|load|layer|param|flops|mermaid|diff|compare|add|modify|delete|save)_/.test(name)) continue;
        expect(KNOWN.has(name), `prompt ${p.name} names unknown tool ${name}`).toBe(true);
      }
    }
  });

  it('shrink_for_target changes its recipe with --write', () => {
    const p = PROMPTS.find(x => x.name === 'shrink_for_target')!;
    expect(renderPrompt(p, { target: 'T4' }, { writeEnabled: true })).toMatch(/write tools/);
    expect(renderPrompt(p, { target: 'T4' }, { writeEnabled: false })).toMatch(/read-only/);
  });
});
