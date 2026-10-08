// Asks the real backend a few questions and compares the sources it returns
// with eval.ts's Nemotron replay of the same retrieval.
//   bun backend-check.ts http://127.0.0.1:18081 results.json id1 id2 ...
import { readFileSync } from "node:fs";

const [base, file, ...ids] = process.argv.slice(2);
const { results } = JSON.parse(readFileSync(file, "utf8"));
for (const id of ids) {
  const row = results.find((r: any) => r.id === id);
  const response = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: row.query }] }),
  });
  let sources: number[] | undefined;
  let buffer = "";
  for await (const chunk of response.body!) {
    buffer += new TextDecoder().decode(chunk);
    for (const line of buffer.split("\n").slice(0, -1)) {
      if (!line.startsWith("data:")) continue;
      const event = JSON.parse(line.slice(5));
      if (event.type === "sources") sources = event.sources.map((s: any) => s.unitId ?? s.unit_id);
    }
    buffer = buffer.slice(buffer.lastIndexOf("\n") + 1);
  }
  const replay = row.configs.nemotron.top6;
  console.log(JSON.stringify({ id, backend: sources, replay, identical: JSON.stringify(sources) === JSON.stringify(replay) }));
}
