// Summarizes eval.ts output: recall at each stage per configuration, per
// category, and the questions where configurations disagree.
//   bun summarize.ts results.json [baseline] [candidate]
import { readFileSync } from "node:fs";

const [file, baseline = "nemotron", candidate = "eg2-bf16-280-search"] = process.argv.slice(2);
const { configs, results } = JSON.parse(readFileSync(file, "utf8"));
const within = (rank: number | null, k: number) => rank !== null && rank <= k;
const pct = (n: number, d: number) => `${n}/${d}`;

const rows = configs.map((name: string) => {
  const e = results.map((r: any) => r.configs[name]);
  const n = e.length;
  const count = (stage: string, k: number) => e.filter((x: any) => within(x[stage], k)).length;
  const mrr = e.reduce((s: number, x: any) => s + (x.dense ? 1 / x.dense : 0), 0) / n;
  const reranked = e.filter((x: any) => x.final !== undefined);
  return {
    config: name,
    "dense@1": pct(count("dense", 1), n), "dense@6": pct(count("dense", 6), n), "dense@10": pct(count("dense", 10), n),
    "dense@50": pct(count("dense", 50), n), "dense MRR": mrr.toFixed(3),
    "rrf@6": pct(count("fused", 6), n), "rrf@50": pct(count("fused", 50), n),
    "final@1": reranked.length ? pct(reranked.filter((x: any) => within(x.final, 1)).length, n) : "",
    "final@6": reranked.length ? pct(reranked.filter((x: any) => within(x.final, 6)).length, n) : "",
    "rerank s": reranked.length ? (reranked.reduce((s: number, x: any) => s + x.rerank_s, 0) / reranked.length).toFixed(2) : "",
  };
});
console.table(rows);
console.log("BM25 alone @50:", results.filter((r: any) => within(r.bm25, 50)).length, "/", results.length);

const groups = new Map<string, any[]>();
for (const r of results) {
  const key = r.category.replace(/-(zh|en)$/, "");
  groups.set(key, [...(groups.get(key) ?? []), r]);
}
console.log(`\nper category (${baseline} vs ${candidate}): dense@6 | rrf@6 | final@6`);
for (const [key, rs] of groups) {
  const fmt = (name: string) => {
    const e = rs.map((r) => r.configs[name]);
    const c = (stage: string) => e.filter((x: any) => within(x[stage], 6)).length;
    return `${c("dense")} | ${c("fused")} | ${e[0].final !== undefined ? c("final") : "-"}`;
  };
  console.log(`  ${key.padEnd(18)} n=${rs.length}  ${fmt(baseline).padEnd(14)} vs  ${fmt(candidate)}`);
}

console.log(`\nquestions that differ (dense / rrf / final rank):`);
for (const r of results) {
  const a = r.configs[baseline], b = r.configs[candidate];
  const key = (e: any) => [e.dense, e.fused, e.final].join("/");
  if (key(a) !== key(b)) console.log(`  ${r.id.padEnd(24)} ${baseline}=${key(a).padEnd(12)} ${candidate}=${key(b)}`);
}
