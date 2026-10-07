// Compares run.sh's configurations against `current`: concurrency per level,
// rerank time, memory, and each answer dataset per case.
//   bun compare.ts <results dir> dense-6 dense-8 dense-rerank-16 current
import { existsSync, readFileSync } from "node:fs";

const [dir, ...labels] = process.argv.slice(2);
const strip = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : NaN;
};
const f = (x: number, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : "—");
const read = (name: string) => (existsSync(`${dir}/${name}`) ? readFileSync(`${dir}/${name}`, "utf8") : undefined);

type Result = { id: string; correct: boolean; error?: string; first_token_s?: number; total_s: number };
type Burst = { level: number; makespan_s: number; results: Result[] };

console.log("## concurrency\n| config | N | mean makespan | answers/min | mean first token | max first token | correct | errors |");
console.log("| --- | --- | --- | --- | --- | --- | --- | --- |");
for (const label of labels) {
  const text = read(`concurrency-${label}.json`);
  if (!text) continue;
  const bursts = (JSON.parse(text) as { bursts: Burst[] }).bursts;
  for (const level of [...new Set(bursts.map((b) => b.level))]) {
    const rows = bursts.filter((b) => b.level === level);
    const results = rows.flatMap((b) => b.results);
    const firsts = results.flatMap((r) => (r.first_token_s === undefined ? [] : [r.first_token_s]));
    const makespan = mean(rows.map((b) => b.makespan_s));
    console.log(`| ${label} | ${level} | ${f(makespan)}s | ${f((level / makespan) * 60, 2)} | ${f(mean(firsts))}s | ` +
      `${f(Math.max(...firsts))}s | ${results.filter((r) => r.correct).length}/${results.length} | ` +
      `${results.filter((r) => r.error).length} |`);
  }
}

// The first 21 retrievals are the concurrency run (warm-up + 20), the rest the
// sequential datasets. Footprints are top's MEM column (phys_footprint).
const mib = (raw: string | undefined) => {
  const m = /^([\d.]+)([KMG])/.exec(raw ?? "");
  return m ? Number(m[1]) * { K: 1 / 1024, M: 1, G: 1024 }[m[2] as "K" | "M" | "G"] : NaN;
};
console.log("\n## rerank and memory\n| config | rerank mean, concurrent | rerank max, concurrent | rerank mean, sequential | " +
  "candidates | inference footprint max | inference footprint median | swap max | free min | pressure max |");
console.log("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
for (const label of labels) {
  const app = read(`app-${label}.log`);
  const memory = read(`memory-${label}.log`);
  if (!app) continue;
  const reranks = strip(app).split("\n")
    .filter((l) => l.includes("retrieval stage completed") && l.includes('stage="rerank"'));
  const ms = reranks.map((l) => Number(/latency_ms=(\d+)/.exec(l)?.[1]));
  const candidates = reranks.map((l) => Number(/candidate_count=(\d+)/.exec(l)?.[1] ?? NaN));
  const samples = (memory ?? "").trim().split("\n").filter(Boolean).map((l) => ({
    swap: Number(/swap_used=([\d.]+)M/.exec(l)?.[1] ?? NaN) / 1024,
    free: Number(/free=([\d.]+)GiB/.exec(l)?.[1] ?? NaN),
    pressure: Number(/pressure=(\d+)/.exec(l)?.[1] ?? NaN),
    inference: mib(/inference=(\S+)/.exec(l)?.[1]),
  }));
  const inference = samples.map((s) => s.inference).filter(Number.isFinite);
  console.log(`| ${label} | ${f(mean(ms.slice(0, 21)) / 1000, 2)}s | ${f(Math.max(...ms.slice(0, 21)) / 1000, 2)}s | ` +
    `${f(mean(ms.slice(21)) / 1000, 2)}s | ${f(mean(candidates), 0)} | ${f(Math.max(...inference) / 1024, 2)} GiB | ` +
    `${f(median(inference) / 1024, 2)} GiB | ${f(Math.max(...samples.map((s) => s.swap)), 2)} GiB | ` +
    `${f(Math.min(...samples.map((s) => s.free)), 2)} GiB | ${Math.max(...samples.map((s) => s.pressure))} |`);
}

type Case = {
  id: string; pass: boolean; error?: string; first_token_s: number; total_s: number;
  missing?: string[]; cited?: boolean; retrieved?: boolean; sources?: string[];
};
const describe = (c: Case | undefined) => !c ? "—" : c.pass ? "ok" : c.error ? "ERR" :
  `X${c.retrieved === false ? " not-retrieved" : ""}${c.cited === false ? " not-cited" : ""}` +
  `${c.missing?.length ? ` missing:${c.missing.join(",")}` : ""}`;
// "source" when the expected source was retrieved in one and not the other,
// "answer" when both runs had it.
const sourceSet = (c: Case) => new Set((c.sources ?? []).map((s) => s.replace(/^S\d+:/, "")));
for (const dataset of ["examples", "realistic", "identifiers"]) {
  const runs = new Map<string, Case[]>();
  for (const label of labels) {
    const text = read(`${dataset}-${label}.json`);
    if (text) runs.set(label, (JSON.parse(text) as { cases: Case[] }).cases);
  }
  if (!runs.size) continue;
  console.log(`\n## ${dataset}\n| config | passed | expected source retrieved | first token median | first token mean | total mean | errors |`);
  console.log("| --- | --- | --- | --- | --- | --- | --- |");
  for (const [label, cases] of runs) {
    const answerable = cases.filter((c) => c.retrieved !== undefined);
    console.log(`| ${label} | ${cases.filter((c) => c.pass).length}/${cases.length} | ` +
      `${answerable.filter((c) => c.retrieved).length}/${answerable.length} | ${f(median(cases.map((c) => c.first_token_s)))}s | ` +
      `${f(mean(cases.map((c) => c.first_token_s)))}s | ${f(mean(cases.map((c) => c.total_s)))}s | ${cases.filter((c) => c.error).length} |`);
  }
  const base = runs.get("current");
  if (!base) continue;
  console.log(`\n| case | ${[...runs.keys()].join(" | ")} |\n| --- |${[...runs.keys()].map(() => " --- |").join("")}`);
  for (const { id } of base) {
    const cells = [...runs.entries()].map(([label, cases]) => [label, cases.find((c) => c.id === id)] as const);
    const now = base.find((c) => c.id === id)!;
    if (cells.every(([, c]) => c?.pass === now.pass)) continue;
    console.log(`| ${id} | ${cells.map(([label, c]) => {
      if (!c || label === "current" || c.pass === now.pass) return describe(c);
      const kind = c.retrieved !== now.retrieved ? "source" : "answer";
      const same = [...sourceSet(c)].filter((s) => sourceSet(now).has(s)).length;
      return `${describe(c)} (${kind}; ${same}/${sourceSet(now).size} sources shared)`;
    }).join(" | ")} |`);
  }
}
