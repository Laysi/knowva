// Concurrent questions through the running application: fires N single-turn
// questions at the same moment and records each one's first-token and total
// time, for every N in --levels, --rounds bursts each. Every question in a run
// is distinct, so no burst hits llama-server's prompt cache; restart Ollama
// before repeating a run, or the second one measures cache hits.
//
//   bun bench/concurrency.ts --levels 1,2,3,4 --rounds 2 \
//     --out ~/knowva-inference/bench-results/concurrency.json
//
// An answer counts as correct when its expected source was retrieved, it has
// every mustInclude group of its case in examples.json or realistic.json and
// matches none of its mustNotMatch patterns; citation is not checked here,
// examples.ts covers that.

import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({
  options: {
    url: { type: "string", default: "http://127.0.0.1:8080" },
    levels: { type: "string", default: "1,2,3,4" },
    rounds: { type: "string", default: "2" },
    label: { type: "string" },
    out: { type: "string" },
  },
});

type Case = {
  id: string;
  turns: string[];
  source?: { filename: string; pages?: (number | null)[] };
  mustInclude?: string[][];
  mustNotMatch?: string[];
};
type Source = { sourceId: string; filename: string; pageNumber: number | null; unitId: number };

const cases = new Map<string, Case>();
for (const name of ["examples.json", "realistic.json"]) {
  const dataset = JSON.parse(readFileSync(new URL(`./${name}`, import.meta.url), "utf8")) as { cases: Case[] };
  for (const spec of dataset.cases) cases.set(spec.id, spec);
}

// Single-turn questions that passed in docs/measurements/README.md §8.11,
// mixing text pages, pages sent as images, image uploads, Office files, mail
// and Markdown. Bursts take them in order, so every run asks the same
// questions together. Each level therefore gets different questions, and
// reusing them would hit the prompt cache: compare runs at the same level,
// not levels within one run.
const WARMUP = "pdf-p3-funding";
const POOL = [
  "pdf-p1-authors", "osha-p9-circular-saw",
  "pdf-p7-kit", "md-port",
  "ey-p4-strategies", "docx-prod",
  "pdf-p10-il050", "jpg-brand", "osha-p12-air-hose",
  "pptx-robot-accidents", "nasa-crack-cause", "eml-report",
  "pdf-p4-goals", "ey-p3-technologies", "docx-wire-rope", "md-license",
  "pdf-p7-figure", "osha-p16-jack", "pptx-guard-types", "eml-subject",
];

const levels = args.levels!.split(",").map(Number);
const rounds = Number(args.rounds);
if (levels.some((n) => !Number.isInteger(n) || n < 1) || !Number.isInteger(rounds) || rounds < 1) {
  throw new Error("--levels and --rounds take positive integers");
}
const needed = levels.reduce((sum, n) => sum + n, 0) * rounds;
if (needed > POOL.length) throw new Error(`the levels need ${needed} distinct questions, the pool has ${POOL.length}`);
for (const id of [WARMUP, ...POOL]) {
  const spec = cases.get(id);
  if (!spec || spec.turns.length !== 1 || !spec.source) throw new Error(`${id} is not a single-turn answerable case`);
}

async function ask(spec: Case) {
  const began = performance.now();
  let first: number | undefined;
  let answer = "";
  let sources: Source[] = [];
  let error: string | undefined;
  let done = false;
  try {
    const response = await fetch(`${args.url}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: spec.turns[0] }], previousSources: [] }),
      signal: AbortSignal.timeout(600_000),
    });
    if (!response.ok || !response.body) throw new Error(`chat HTTP ${response.status}`);
    let buffer = "";
    const decoder = new TextDecoder();
    for await (const chunk of response.body) {
      buffer += decoder.decode(chunk, { stream: true });
      let newline: number;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line.startsWith("data: ")) continue;
        const event = JSON.parse(line.slice(6));
        if (event.type === "text-delta") {
          first ??= performance.now();
          answer += event.text;
        } else if (event.type === "sources") sources = event.sources;
        else if (event.type === "error") error = event.message;
        else if (event.type === "done") done = true;
      }
    }
    // A stream cut off before `done` is not a finished answer.
    if (!error && !done) error = "stream ended without done";
  } catch (caught) {
    error = String(caught);
  }
  const finished = performance.now();
  const text = answer.toLowerCase();
  const retrieved = sources.some((s) =>
    s.filename === spec.source!.filename &&
    (spec.source!.pages === undefined || spec.source!.pages.includes(s.pageNumber ?? null)));
  const missing = (spec.mustInclude ?? [])
    .filter((group) => !group.some((alternative) => text.includes(alternative.toLowerCase())))
    .map((group) => group.join("|"));
  const forbidden = (spec.mustNotMatch ?? []).filter((pattern) => new RegExp(pattern, "i").test(answer));
  return {
    id: spec.id,
    correct: !error && answer.length > 0 && retrieved && missing.length === 0 && forbidden.length === 0,
    error,
    missing,
    forbidden,
    first_token_s: first === undefined ? undefined : +((first - began) / 1000).toFixed(1),
    total_s: +((finished - began) / 1000).toFixed(1),
    answer_chars: answer.length,
    answer,
  };
}

const warmup = await ask(cases.get(WARMUP)!);
const brief = (result: Awaited<ReturnType<typeof ask>>) => ({ ...result, answer: result.answer.replace(/\s+/g, " ").slice(0, 160) });
console.log(JSON.stringify({ warmup: brief(warmup) }));
// Measuring after a failed warm-up would time a cold model load.
if (warmup.error) throw new Error(`warm-up failed: ${warmup.error}`);

type Burst = { level: number; round: number; makespan_s: number; results: Awaited<ReturnType<typeof ask>>[] };
const bursts: Burst[] = [];
let next = 0;
for (const level of levels) {
  for (let round = 1; round <= rounds; round++) {
    const specs = POOL.slice(next, next + level).map((id) => cases.get(id)!);
    next += level;
    const began = performance.now();
    const results = await Promise.all(specs.map(ask));
    const burst = { level, round, makespan_s: +((performance.now() - began) / 1000).toFixed(1), results };
    bursts.push(burst);
    console.log(JSON.stringify({ ...burst, results: results.map(brief) }));
    await Bun.sleep(5000);
  }
}

const mean = (values: number[]) => values.reduce((sum, v) => sum + v, 0) / values.length;
console.log(`\n${args.label ?? ""}\n| N | mean first token | max first token | mean total | mean makespan | answers/min | correct | errors |`);
console.log("| --- | --- | --- | --- | --- | --- | --- | --- |");
for (const level of levels) {
  const rows = bursts.filter((b) => b.level === level);
  const results = rows.flatMap((b) => b.results);
  const firsts = results.flatMap((r) => (r.first_token_s === undefined ? [] : [r.first_token_s]));
  const makespan = mean(rows.map((b) => b.makespan_s));
  const seconds = (value: number | undefined) => (value === undefined ? "—" : `${value.toFixed(1)}s`);
  console.log(`| ${level} | ${seconds(firsts.length ? mean(firsts) : undefined)} | ${seconds(firsts.length ? Math.max(...firsts) : undefined)} | ` +
    `${mean(results.map((r) => r.total_s)).toFixed(1)}s | ${makespan.toFixed(1)}s | ${((level / makespan) * 60).toFixed(2)} | ` +
    `${results.filter((r) => r.correct).length}/${results.length} | ${results.filter((r) => r.error).length} |`);
}
if (args.out) writeFileSync(args.out, JSON.stringify({ label: args.label, warmup, bursts }, null, 2));
process.exitCode = bursts.some((b) => b.results.some((r) => r.error)) ? 1 : 0;
