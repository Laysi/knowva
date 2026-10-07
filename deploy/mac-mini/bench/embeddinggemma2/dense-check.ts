// Runs search.rs dense_candidates_from_pool with the Nemotron query vectors and
// checks that it returns exactly eval.ts's exact-cosine top-50, in order.
//   bun dense-check.ts postgres://km:km@127.0.0.1:5433/knowva_eg2
import { SQL } from "bun";
import { readFileSync } from "node:fs";

const db = new SQL(process.argv[2]);
const normalize = (v: number[]) => {
  const norm = Math.hypot(...v);
  return v.map((x) => x / norm);
};
const units = readFileSync("corpus.jsonl", "utf8").trim().split("\n").map((line) => JSON.parse(line))
  .map((u) => ({ id: u.id as number, vector: normalize(JSON.parse(u.nemotron)) }));
const queries: Record<string, { query: number[] }> = JSON.parse(readFileSync("out/nemotron.json", "utf8")).queries;

const DENSE = `WITH nearest AS MATERIALIZED (
  SELECT u.id AS unit_id, u.document_id, (u.embedding <=> $1::halfvec)::double precision AS distance
  FROM document_units AS u WHERE u.retrieval_ready ORDER BY u.embedding <=> $1::halfvec LIMIT $2)
SELECT nearest.unit_id FROM nearest JOIN documents AS d ON d.id = nearest.document_id
WHERE d.status = 'ready' ORDER BY nearest.distance, nearest.unit_id LIMIT $2`;
const literal = (v: number[]) => `[${v.join(",")}]`;

const first = Object.values(queries)[0].query;
const plan = await db.unsafe(`EXPLAIN ${DENSE}`, [literal(first), 50]);
const scans = plan.map((r: any) => r["QUERY PLAN"].trim()).filter((l: string) => /Scan/.test(l)).map((l: string) => l.split("  (")[0]);
console.log("plan:", scans.join("; "));

let identical = 0;
for (const { query } of Object.values(queries)) {
  // The backend binds the query as halfvec.
  const q = normalize(Array.from(new Float16Array(query)));
  const exact = units.map((u) => ({ id: u.id, score: u.vector.reduce((sum, x, i) => sum + x * q[i], 0) }))
    .sort((a, b) => b.score - a.score || a.id - b.id).slice(0, 50).map((u) => u.id);
  const sql = (await db.unsafe(DENSE, [literal(query), 50])).map((r: any) => Number(r.unit_id));
  if (sql.join() === exact.join()) identical++;
}
console.log(`top-50 identical in order: ${identical}/${Object.keys(queries).length}`);
await db.close();
