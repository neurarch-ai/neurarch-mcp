/**
 * Every rule id a tool in this server can report, in one table, assembled
 * from the vendored bundles rather than typed here: the lint advisor and
 * shape rules (engine), the edit rules (plan) and the pin rules (policy).
 *
 * Served as neurarch://rules/catalogue and used for prompt completions. It
 * answers the question an agent has when a finding names a rule it has not
 * seen: is this one of yours, where does it come from, and is there a
 * measurement behind it. `provenance` is present only where the verifier's own
 * table has one, so a rule without it is a convention, and reads as one.
 */
import { ADVISOR_RULE_IDS, SHAPE_RULE_IDS } from './vendor/engine.bundle.mjs';
import { EDIT_RULE_IDS, EDIT_RULE_SEEN_IN } from './vendor/plan.bundle.mjs';
import { RULE_PROVENANCE } from './vendor/verifier.bundle.mjs';
import { PIN_RULE_META } from './vendor/policy.bundle.mjs';

export interface CatalogueRule {
  id: string;
  source: 'lint' | 'shape' | 'edit' | 'pin';
  reportedBy: string[];
  provenance?: unknown;
  seenIn?: string[];
  summary?: string;
  code?: string;
}

export function ruleCatalogue(): { count: number; rules: CatalogueRule[] } {
  const rules: CatalogueRule[] = [];
  const seen = new Set<string>();
  const add = (r: CatalogueRule) => {
    if (seen.has(r.id)) return;
    seen.add(r.id);
    const prov = (RULE_PROVENANCE as Record<string, unknown>)[r.id];
    rules.push(prov ? { ...r, provenance: prov } : r);
  };
  for (const id of SHAPE_RULE_IDS) add({ id, source: 'shape', reportedBy: ['lint_model', 'check_design', 'plan'] });
  for (const id of ADVISOR_RULE_IDS) add({ id, source: 'lint', reportedBy: ['lint_model', 'check_design', 'plan'] });
  for (const id of EDIT_RULE_IDS) add({ id, source: 'edit', reportedBy: ['check_edit'], seenIn: EDIT_RULE_SEEN_IN[id] });
  for (const [id, m] of Object.entries(PIN_RULE_META)) add({ id, source: 'pin', reportedBy: ['check_pins'], summary: m.summary, code: m.code });
  return { count: rules.length, rules };
}
