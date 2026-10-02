// Measures the two Nemotron VL routes the way Knowva calls them: page
// embeddings at ingestion (image+text, one page per request) and one query's
// rerank pass (50 candidates in batches of 8, image+text passages).
//
//   bun bench/retrieval.ts --pages ~/knowva-inference/bench-pages-png
//
// Pages are p-NN.png + p-NN.txt pairs at the backend's PDF_RENDER_DPI.

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({
  options: {
    pages: { type: "string" },
    url: { type: "string", default: "http://127.0.0.1:9500" },
    embed: { type: "string", default: "8" },
    candidates: { type: "string", default: "50" },
    batch: { type: "string", default: "8" },
    query: { type: "string", default: "JADE 在 IoT4Industry 裡是怎麼部署的？" },
  },
});
if (!args.pages) throw new Error("--pages is required");

const pages = readdirSync(args.pages)
  .map((name) => /^p-(\d+)\.png$/.exec(name)?.[1])
  .filter((n): n is string => n !== undefined)
  .sort()
  .map((n) => ({
    page: Number(n),
    text: readFileSync(join(args.pages!, `p-${n}.txt`), "utf8").trim(),
    image: `data:image/png;base64,${readFileSync(join(args.pages!, `p-${n}.png`)).toString("base64")}`,
  }));

async function post(route: string, body: unknown) {
  const began = performance.now();
  const response = await fetch(`${args.url}${route}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${route} HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
  const json = await response.json();
  return { json, seconds: (performance.now() - began) / 1000 };
}

const embed = (input: string[], input_type: string, modality: string) =>
  post("/v1/embeddings", {
    model: "nvidia/llama-nemotron-embed-vl-1b-v2",
    input,
    encoding_format: "float",
    embedding_type: "float",
    input_type,
    modality: [modality],
    truncate: "NONE",
  });

const round = (n: number) => +n.toFixed(2);

// Query embeds: the first includes loading the model.
const queryTimes = [];
for (let i = 0; i < 3; i++) {
  const { json, seconds } = await embed([args.query!], "query", "text");
  if (json.data[0].embedding.length !== 2048) throw new Error("embedding is not 2048-dimensional");
  queryTimes.push(round(seconds));
}
console.log(JSON.stringify({ query_embed_s: queryTimes }));

const pageTimes = [];
for (const page of pages.slice(0, Number(args.embed))) {
  const input = page.text ? `${page.text}\n<img src="${page.image}"/>` : page.image;
  const { seconds } = await embed([input], "passage", page.text ? "text_image" : "image");
  pageTimes.push(round(seconds));
}
const meanPage = pageTimes.reduce((a, b) => a + b, 0) / pageTimes.length;
console.log(JSON.stringify({ page_embed_s: pageTimes, mean: round(meanPage) }));

const size = Number(args.batch);
async function rerank(candidates: typeof pages) {
  const scores: number[] = [];
  const batchTimes = [];
  for (let start = 0; start < candidates.length; start += size) {
    const slice = candidates.slice(start, start + size);
    const { json, seconds } = await post("/v1/ranking", {
      model: "nvidia/llama-nemotron-rerank-vl-1b-v2",
      query: { text: args.query },
      passages: slice.map((p) => ({ text: p.text || undefined, image: p.image })),
      truncate: "NONE",
    });
    scores.push(...json.rankings.map((r: { logit: number }) => r.logit));
    batchTimes.push(round(seconds));
  }
  return { scores, batchTimes };
}

// Every distinct page once, to see whether the pages that answer the query
// come out on top under the current settings.
const quality = await rerank(pages);
const top = quality.scores
  .map((score, i) => ({ page: pages[i].page, score: round(score) }))
  .sort((a, b) => b.score - a.score)
  .slice(0, 6);
console.log(JSON.stringify({ top_pages: top }));

// One query's rerank pass: the backend's candidate count, cycling through
// the pages as a corpus of many similar pages would.
const candidates = Array.from({ length: Number(args.candidates) }, (_, i) => pages[i % pages.length]);
const rerankBegan = performance.now();
const pass = await rerank(candidates);
const rerankTotal = (performance.now() - rerankBegan) / 1000;
console.log(JSON.stringify({ rerank_batches_s: pass.batchTimes, rerank_total_s: round(rerankTotal) }));

const health = await fetch(`${args.url}/health`).then((r) => r.json());
console.log(JSON.stringify({ health }));
