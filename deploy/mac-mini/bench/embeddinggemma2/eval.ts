// Replays Knowva's retrieval (dense top-50 + BM25 top-50 -> RRF k=60 -> top-50
// reranked in single-modality batches of 8 -> top-6) for every question, with
// the dense stage swapped between embedding models, and scores where the gold
// source lands at each stage.
//
//   bun eval.ts --db postgres://km:km@127.0.0.1:5433/knowva_eg2 \
//     --rerank http://127.0.0.1:19501 --files files --out results.json
//
// Run from a directory holding export-corpus.sh's corpus.jsonl and
// queries.json and run_embeds.sh's out/*.json. --db is the instance the corpus
// was exported from (BM25 runs there, with search.rs's SQL), --files its
// FILES_DIR, --rerank the inference service. Rerank results are cached in
// rerank-cache.json by question and candidate list.
import { SQL } from "bun";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({
  options: {
    db: { type: "string" },
    rerank: { type: "string" },
    files: { type: "string", default: "files" },
    out: { type: "string", default: "results.json" },
    "rerank-configs": { type: "string", default: "nemotron,eg2-bf16-280-search" },
  },
});

type Unit = {
  id: number; filename: string; kind: string; unit_index: number; page: number | null;
  content: string | null; image_path: string | null; uses_image: boolean; document_id: string;
};
// pages: page numbers, null meaning a text unit; absent means any unit of the file.
type Question = { id: string; set: string; category: string; query: string; gold: { filename: string; pages?: (number | null)[] } };

const units: (Unit & { nemotron: string })[] = readFileSync("corpus.jsonl", "utf8").trim().split("\n").map((l) => JSON.parse(l));
const byId = new Map(units.map((u) => [u.id, u]));
const questions: Question[] = JSON.parse(readFileSync("queries.json", "utf8"));

const normalize = (v: number[]) => {
  const norm = Math.hypot(...v);
  return v.map((x) => x / norm);
};
// Vectors are stored and queried as pgvector halfvec, so both sides are fp16;
// without this the dense order past the top 10 differs from the database's.
const half = (v: number[]) => Array.from(new Float16Array(v));
const truncate = (v: number[], dims?: number) => normalize(half(normalize(dims ? v.slice(0, dims) : v)));
const dot = (a: number[], b: number[]) => a.reduce((sum, x, i) => sum + x * b[i], 0);

function loadVectors(file: string) {
  const json = JSON.parse(readFileSync(join("out", `${file}.json`), "utf8"));
  return { units: json.units as Record<string, number[]>, queries: json.queries as Record<string, Record<string, number[]>> };
}

// Each dense configuration: unit vectors, a query vector per question.
type Dense = { name: string; unit: (id: number) => number[]; query: (q: Question) => number[] };
const nemotron = loadVectors("nemotron");
const dbVectors = new Map(units.map((u) => [u.id, normalize(JSON.parse(u.nemotron))]));
const configs: Dense[] = [
  { name: "nemotron", unit: (id) => dbVectors.get(id)!, query: (q) => normalize(half(nemotron.queries[q.id].query)) },
];
for (const [file, prompts, dims] of [
  ["eg2-bf16-280", ["search", "qa"], [undefined, 256]],
  ["eg2-bf16-1120", ["search"], [undefined]],
  ["eg2-mxfp8-280", ["search"], [undefined]],
] as const) {
  const vectors = loadVectors(file);
  for (const prompt of prompts)
    for (const d of dims) {
      const cache = new Map<number, number[]>();
      configs.push({
        name: `${file}-${prompt}${d ? `-${d}d` : ""}`,
        unit: (id) => cache.get(id) ?? (cache.set(id, truncate(vectors.units[id], d)), cache.get(id)!),
        query: (q) => truncate(vectors.queries[q.id][prompt], d),
      });
    }
}

const isGold = (q: Question, u: Unit) =>
  u.filename === q.gold.filename && (!q.gold.pages || q.gold.pages.includes(u.page));

type Ranked = { id: number; rank: number; score: number };

function denseRanking(config: Dense, q: Question): Ranked[] {
  const query = config.query(q);
  return units
    .map((u) => ({ id: u.id, score: dot(query, config.unit(u.id)) }))
    .sort((a, b) => b.score - a.score || a.id - b.id)
    .map((r, i) => ({ ...r, rank: i + 1 }));
}

const db = new SQL(args.db!);
// search.rs bm25_candidates_from_pool, verbatim.
const BM25_SQL = `WITH query_terms AS MATERIALIZED (
  SELECT DISTINCT token AS term FROM unnest(knowva_bm25_tokenize($1)) AS token WHERE token <> ''
), query_token_array AS (
  SELECT COALESCE(array_agg(term ORDER BY term), ARRAY[]::text[]) AS terms FROM query_terms
), candidate_units AS MATERIALIZED (
  SELECT u.id AS unit_id, u.document_id, u.bm25_tokens, u.bm25_length
  FROM document_units AS u CROSS JOIN query_token_array AS query
  WHERE u.retrieval_ready AND u.content IS NOT NULL AND btrim(u.content) <> '' AND u.bm25_tokens && query.terms
), scored AS (
  SELECT candidate.unit_id,
    sum(ln(1.0 + ((greatest(corpus.document_count - term_stats.document_frequency, 0)::double precision + 0.5)
      / (term_stats.document_frequency::double precision + 0.5)))
      * ((frequency.tf * ($3::double precision + 1.0))
        / (frequency.tf + $3::double precision * (1.0 - $4::double precision
          + $4::double precision * candidate.bm25_length::double precision
            / greatest(corpus.total_length::double precision / corpus.document_count::double precision, 1.0)))))::double precision AS score
  FROM candidate_units AS candidate
  JOIN documents AS document ON document.id = candidate.document_id
  CROSS JOIN knowva_bm25_corpus_stats AS corpus
  JOIN query_terms AS query ON query.term = ANY(candidate.bm25_tokens)
  JOIN knowva_bm25_term_stats AS term_stats ON term_stats.term = query.term
  CROSS JOIN LATERAL (SELECT cardinality(array_positions(candidate.bm25_tokens, query.term))::double precision AS tf) AS frequency
  WHERE corpus.singleton AND document.status = 'ready' AND corpus.document_count > 0 AND frequency.tf > 0
  GROUP BY candidate.unit_id
)
SELECT unit_id, score FROM scored ORDER BY score DESC, unit_id LIMIT $2`;

async function bm25(q: Question): Promise<Ranked[]> {
  const rows = await db.unsafe(BM25_SQL, [q.query, 50, 1.2, 0.75]);
  return rows.map((r: any, i: number) => ({ id: Number(r.unit_id), score: Number(r.score), rank: i + 1 }));
}

// search.rs reciprocal_rank_fusion.
function rrf(dense: Ranked[], lexical: Ranked[]) {
  const acc = new Map<number, { id: number; score: number; best: number }>();
  for (const ranking of [dense, lexical])
    for (const r of ranking) {
      const a = acc.get(r.id) ?? { id: r.id, score: 0, best: Number.MAX_SAFE_INTEGER };
      a.score += 1 / (60 + r.rank);
      a.best = Math.min(a.best, r.rank);
      acc.set(r.id, a);
    }
  return [...acc.values()].sort((a, b) => b.score - a.score || a.best - b.best || a.id - b.id).slice(0, 50).map((a) => a.id);
}

const imageCache = new Map<number, string>();
function passage(u: Unit) {
  const text = u.content?.trim();
  if (!u.uses_image) return { text: text! };
  let image = imageCache.get(u.id);
  if (!image) {
    const bytes = readFileSync(join(args.files!, u.image_path!));
    const mime = bytes[0] === 0x89 ? "image/png" : "image/jpeg";
    image = `data:${mime};base64,${bytes.toString("base64")}`;
    imageCache.set(u.id, image);
  }
  return text ? { text, image } : { image };
}

// search.rs rerank_candidates_inner: modality groups in first-seen order, chunks of 8.
async function rerank(query: string, fused: number[]) {
  const groups: [boolean, number[]][] = [];
  fused.forEach((id, index) => {
    const uses = byId.get(id)!.uses_image;
    const group = groups.find(([g]) => g === uses);
    group ? group[1].push(index) : groups.push([uses, [index]]);
  });
  const scores: number[] = new Array(fused.length);
  let seconds = 0;
  for (const [, indices] of groups)
    for (let i = 0; i < indices.length; i += 8) {
      const batch = indices.slice(i, i + 8);
      const body = JSON.stringify({
        model: "nvidia/llama-nemotron-rerank-vl-1b-v2",
        query: { text: query },
        passages: batch.map((index) => passage(byId.get(fused[index])!)),
        truncate: "NONE",
      });
      // Three attempts like the backend; the SSH tunnel to the Mac drops now and then.
      for (let attempt = 1; ; attempt++) {
        const began = performance.now();
        try {
          const response = await fetch(`${args.rerank}/v1/ranking`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body,
          });
          if (!response.ok) throw new Error(`rerank HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`);
          const rankings = (await response.json()).rankings;
          seconds += (performance.now() - began) / 1000;
          for (const r of rankings) scores[batch[r.index]] = r.logit;
          break;
        } catch (error) {
          if (attempt === 3) throw error;
          console.error(`rerank attempt ${attempt} failed: ${error}; retrying`);
          await Bun.sleep(5000);
        }
      }
    }
  const order = fused.map((id, index) => ({ id, index, score: scores[index] }))
    .sort((a, b) => b.score - a.score || a.index - b.index);
  return { top: order.slice(0, 6).map((o) => o.id), seconds };
}

const firstGold = (q: Question, ids: number[]) => {
  const i = ids.findIndex((id) => isGold(q, byId.get(id)!));
  return i < 0 ? null : i + 1;
};

const rerankConfigs = new Set(args["rerank-configs"]!.split(","));
const cachePath = "rerank-cache.json";
const rerankCache: Record<string, { top: number[]; seconds: number }> = existsSync(cachePath) ? JSON.parse(readFileSync(cachePath, "utf8")) : {};

const results: any[] = [];
for (const q of questions) {
  const lexical = await bm25(q);
  const row: any = { id: q.id, set: q.set, category: q.category, query: q.query, bm25: firstGold(q, lexical.map((r) => r.id)), configs: {} };
  for (const config of configs) {
    const dense = denseRanking(config, q);
    const fused = rrf(dense.slice(0, 50), lexical);
    const entry: any = { dense: firstGold(q, dense.map((r) => r.id)), fused: firstGold(q, fused), fusedIds: fused };
    if (rerankConfigs.has(config.name)) {
      // Rerank output depends only on the query and the passages sent, so the
      // key fingerprints exactly those (image keys are unique per ingestion).
      // Delete the cache when the reranker itself changes.
      const sent = fused.map((id) => {
        const u = byId.get(id)!;
        return [id, u.uses_image ? u.image_path : null, u.content];
      });
      const key = Bun.hash(JSON.stringify([q.query, sent])).toString(16);
      rerankCache[key] ??= await rerank(q.query, fused);
      writeFileSync(cachePath, JSON.stringify(rerankCache));
      entry.top6 = rerankCache[key].top;
      entry.final = firstGold(q, entry.top6);
      entry.rerank_s = +rerankCache[key].seconds.toFixed(2);
    }
    row.configs[config.name] = entry;
  }
  results.push(row);
  console.log(q.id, Object.entries(row.configs).map(([n, e]: any) => `${n}:${e.dense}/${e.fused}${e.final !== undefined ? `/${e.final}` : ""}`).join(" "));
}
writeFileSync(args.out!, JSON.stringify({ configs: configs.map((c) => c.name), results }, null, 1));
await db.close();
