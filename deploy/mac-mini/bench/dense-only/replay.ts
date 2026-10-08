// Replays where #60's retrieval configurations put each question's expected
// source in Gemma's context, against a running Knowva database and the
// inference service, with search.rs's own dense and BM25 SQL:
//
//   current          dense 50 + BM25 50 -> RRF (k=60) 50 -> rerank -> top 6
//   dense-6, dense-8 the dense ranking's first 6 or 8, no BM25 and no rerank
//   dense-rerank-N   dense top N -> rerank -> top 6, for each N in --ns
//
//   bun replay.ts --db postgres://knowva:<password>@127.0.0.1:5434/knowva \
//     --files <FILES_DIR> --service http://127.0.0.1:9500 \
//     --datasets examples.json,realistic.json,identifiers.json --out replay.json
//
// Questions are the answerable cases of each dataset, queried with their last
// turn as the backend does. Query vectors come from the service's embeddings
// route exactly as the backend requests them, and rerank calls follow
// rerank_candidates_inner: modality groups in first-seen order, batches of 8.
// Both are cached next to --out by query and by passages sent, so a rerun only
// calls the service for what changed; delete the caches when a model changes.
import { SQL } from "bun";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({
  options: {
    db: { type: "string" },
    files: { type: "string" },
    service: { type: "string", default: "http://127.0.0.1:9500" },
    datasets: { type: "string" },
    ns: { type: "string", default: "8,10,12,16,20,24" },
    out: { type: "string", default: "replay.json" },
  },
});
if (!args.db || !args.files || !args.datasets) throw new Error("--db, --files and --datasets are required");

const EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2";
const RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2";
const TOP_N = 6;
const BATCH = 8;

type Gold = { filename: string; pages?: (number | null)[] };
type Question = { id: string; set: string; category: string; query: string; gold: Gold };
type Unit = {
  id: number; filename: string; kind: string; page: number | null;
  content: string | null; image_path: string | null; uses_image: boolean;
};

const questions: Question[] = [];
for (const path of args.datasets.split(",")) {
  const set = basename(path, ".json");
  for (const c of JSON.parse(readFileSync(path, "utf8")).cases)
    if (c.source) questions.push({ id: c.id, set, category: c.category, query: c.turns.at(-1), gold: c.source });
}

const db = new SQL(args.db);
// uses_image mirrors RetrievalCandidate::uses_page_image.
const units = new Map<number, Unit>(
  (await db.unsafe(`SELECT u.id::int AS id, d.filename, u.kind, u.page_number AS page, u.content, u.image_path,
      (u.kind = 'page' AND (u.needs_vision OR u.content IS NULL OR btrim(u.content) = '')) AS uses_image
    FROM document_units u JOIN documents d ON d.id = u.document_id
    WHERE u.retrieval_ready AND d.status = 'ready'`)).map((u: Unit) => [u.id, u]),
);
console.log(`${units.size} units, ${questions.length} questions`);

const isGold = (q: Question, id: number) => {
  const u = units.get(id)!;
  return u.filename === q.gold.filename && (!q.gold.pages || q.gold.pages.includes(u.page));
};
const firstGold = (q: Question, ids: number[]) => {
  const i = ids.findIndex((id) => isGold(q, id));
  return i < 0 ? null : i + 1;
};

const cacheDir = dirname(args.out!);
function cache<T>(name: string) {
  const path = join(cacheDir, name);
  const data: Record<string, T> = existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : {};
  return { data, save: () => writeFileSync(path, JSON.stringify(data)) };
}
const vectors = cache<number[]>("replay-query-vectors.json");
const reranks = cache<{ scores: number[]; seconds: number }>("replay-rerank-cache.json");

async function post(route: string, body: unknown) {
  for (let attempt = 1; ; attempt++) {
    try {
      const response = await fetch(`${args.service}${route}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(300_000),
      });
      if (!response.ok) throw new Error(`${route} HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`);
      return await response.json();
    } catch (error) {
      if (attempt === 3) throw error;
      console.error(`${route} attempt ${attempt} failed: ${error}; retrying`);
      await Bun.sleep(5000);
    }
  }
}

// CloudModelClient::embed_text with input_type "query".
async function embedQuery(query: string) {
  vectors.data[query] ??= (await post("/v1/embeddings", {
    model: EMBED_MODEL, input: [query], encoding_format: "float", embedding_type: "float",
    input_type: "query", modality: ["text"], truncate: "NONE",
  })).data[0].embedding;
  vectors.save();
  return vectors.data[query];
}

// search.rs dense_candidates_from_pool; the vector is cast to halfvec as to_halfvec does.
const DENSE_SQL = `WITH nearest AS MATERIALIZED (
  SELECT u.id AS unit_id, u.document_id, (u.embedding <=> $1::halfvec)::double precision AS distance
  FROM document_units AS u WHERE u.retrieval_ready
  ORDER BY u.embedding <=> $1::halfvec LIMIT $2
)
SELECT nearest.unit_id FROM nearest JOIN documents AS d ON d.id = nearest.document_id
WHERE d.status = 'ready' ORDER BY nearest.distance, nearest.unit_id LIMIT $2`;

// search.rs bm25_candidates_from_pool.
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
SELECT unit_id FROM scored ORDER BY score DESC, unit_id LIMIT $2`;

const ids = (rows: { unit_id: number | bigint }[]) => rows.map((r) => Number(r.unit_id));

// search.rs reciprocal_rank_fusion: ties by best contributing rank, then unit id.
function rrf(dense: number[], lexical: number[]) {
  const acc = new Map<number, { id: number; score: number; best: number }>();
  for (const ranking of [dense, lexical])
    ranking.forEach((id, index) => {
      const a = acc.get(id) ?? { id, score: 0, best: Number.MAX_SAFE_INTEGER };
      a.score += 1 / (60 + index + 1);
      a.best = Math.min(a.best, index + 1);
      acc.set(id, a);
    });
  return [...acc.values()].sort((a, b) => b.score - a.score || a.best - b.best || a.id - b.id).slice(0, 50).map((a) => a.id);
}

const images = new Map<number, string>();
function passage(u: Unit) {
  const text = u.content?.trim() || undefined;
  if (!u.uses_image) return { text };
  let image = images.get(u.id);
  if (!image) {
    const bytes = readFileSync(join(args.files!, u.image_path!));
    image = `data:${bytes[0] === 0x89 ? "image/png" : "image/jpeg"};base64,${bytes.toString("base64")}`;
    images.set(u.id, image);
  }
  return text ? { text, image } : { image };
}

// search.rs rerank_candidates_inner: the top 6 by logit, ties in pool order.
async function rerank(query: string, pool: number[]) {
  // Keyed by what is sent, image bytes included.
  const key = Bun.hash(JSON.stringify([query, pool.map((id) => {
    const u = units.get(id)!;
    return [id, u.uses_image ? Bun.hash(passage(u).image!).toString(16) : null, u.content];
  })])).toString(16);
  if (!reranks.data[key]) {
    const groups: [boolean, number[]][] = [];
    pool.forEach((id, index) => {
      const uses = units.get(id)!.uses_image;
      const group = groups.find(([g]) => g === uses);
      group ? group[1].push(index) : groups.push([uses, [index]]);
    });
    const scores: number[] = new Array(pool.length);
    let seconds = 0;
    for (const [, indices] of groups)
      for (let i = 0; i < indices.length; i += BATCH) {
        const batch = indices.slice(i, i + BATCH);
        const began = performance.now();
        const { rankings } = await post("/v1/ranking", {
          model: RERANK_MODEL, query: { text: query },
          passages: batch.map((index) => passage(units.get(pool[index])!)), truncate: "NONE",
        });
        seconds += (performance.now() - began) / 1000;
        for (const r of rankings) scores[batch[r.index]] = r.logit;
      }
    reranks.data[key] = { scores, seconds: +seconds.toFixed(3) };
    reranks.save();
  }
  const { scores, seconds } = reranks.data[key];
  const top = pool.map((id, index) => ({ id, index, score: scores[index] }))
    .sort((a, b) => b.score - a.score || a.index - b.index).slice(0, TOP_N).map((o) => o.id);
  return { top, seconds };
}

const ns = args.ns!.split(",").map(Number);
const configNames = ["current", "dense-6", "dense-8", ...ns.map((n) => `dense-rerank-${n}`)];
const results: any[] = [];
for (const q of questions) {
  const vector = `[${(await embedQuery(q.query)).join(",")}]`;
  const dense = ids(await db.unsafe(DENSE_SQL, [vector, 50]));
  const lexical = ids(await db.unsafe(BM25_SQL, [q.query, 50, 1.2, 0.75]));
  const fused = rrf(dense, lexical);
  const row: any = {
    id: q.id, set: q.set, category: q.category, query: q.query,
    dense: firstGold(q, dense), bm25: firstGold(q, lexical), rrf: firstGold(q, fused), configs: {},
  };
  const current = await rerank(q.query, fused);
  row.configs.current = { context: current.top, rerank_s: current.seconds };
  row.configs["dense-6"] = { context: dense.slice(0, 6) };
  row.configs["dense-8"] = { context: dense.slice(0, 8) };
  for (const n of ns) {
    const reranked = await rerank(q.query, dense.slice(0, n));
    row.configs[`dense-rerank-${n}`] = { context: reranked.top, rerank_s: reranked.seconds };
  }
  for (const entry of Object.values(row.configs) as any[]) entry.gold = firstGold(q, entry.context);
  results.push(row);
  console.log(q.id, `dense:${row.dense} bm25:${row.bm25} rrf:${row.rrf}`,
    configNames.map((name) => `${name}:${row.configs[name].gold ?? "-"}`).join(" "));
}
await db.close();

// Questions whose expected source reaches the context, per configuration and dataset.
const sets = [...new Set(questions.map((q) => q.set))];
const summary = configNames.map((name) => {
  const hits = Object.fromEntries(sets.map((set) => {
    const rows = results.filter((r) => r.set === set);
    return [set, `${rows.filter((r) => r.configs[name].gold !== null).length}/${rows.length}`];
  }));
  const timed = results.map((r) => r.configs[name].rerank_s).filter((s) => s !== undefined);
  return {
    config: name, ...hits,
    total: `${results.filter((r) => r.configs[name].gold !== null).length}/${results.length}`,
    rerank_mean_s: timed.length ? +(timed.reduce((a, b) => a + b, 0) / timed.length).toFixed(2) : null,
  };
});
console.table(summary);
writeFileSync(args.out!, JSON.stringify({ configs: configNames, summary, results }, null, 1));
