// Re-scores saved bench/examples.ts reports after a dataset's answer keys
// (mustInclude, mustNotMatch) change, without asking again. Retrieval and
// citation do not depend on the keys and are kept; pass is recomputed as
// examples.ts scores an answerable case. Reports are rewritten in place.
//   bun rescore.ts <dataset.json> <report.json>...
import { readFileSync, writeFileSync } from "node:fs";

const [datasetPath, ...reports] = process.argv.slice(2);
type Case = { id: string; source?: unknown; mustInclude?: string[][]; mustNotMatch?: string[] };
const cases = new Map<string, Case>(
  (JSON.parse(readFileSync(datasetPath, "utf8")).cases as Case[]).map((c) => [c.id, c]),
);

for (const path of reports) {
  const report = JSON.parse(readFileSync(path, "utf8"));
  for (const result of report.cases) {
    const spec = cases.get(result.id);
    if (!spec?.source || result.error) continue;
    const text = result.answer.toLowerCase();
    const missing = (spec.mustInclude ?? [])
      .filter((group) => !group.some((alternative) => text.includes(alternative.toLowerCase())))
      .map((group) => group.join("|"));
    const forbidden = (spec.mustNotMatch ?? []).filter((pattern) => new RegExp(pattern, "i").test(result.answer));
    const pass = result.retrieved && result.cited && missing.length === 0 && forbidden.length === 0;
    if (pass !== result.pass) console.log(`${path}: ${result.id} ${result.pass} -> ${pass}`);
    Object.assign(result, { pass, missing, forbidden });
  }
  writeFileSync(path, JSON.stringify(report, null, 2));
}
