// End to end through the running application: upload one document, wait for
// it to become ready, then hold a three-turn conversation that echoes
// `previousSources` the way the frontend does (docs/measurements/three_turn.py).
//
//   bun bench/e2e.ts --file ~/knowva-inference/iiot_industry_report.pdf

import { basename } from "node:path";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({
  options: {
    file: { type: "string" },
    url: { type: "string", default: "http://127.0.0.1:8080" },
    // Converse over the already-uploaded copy instead of uploading again, so
    // repeated runs do not retrieve duplicate pages.
    reuse: { type: "boolean", default: false },
  },
});
if (!args.file) throw new Error("--file is required");

type Source = { unitId: number; pageNumber?: number };

const existing = args.reuse
  ? ((await (await fetch(`${args.url}/api/documents`)).json()) as { filename: string; status: string }[])
      .find((d) => d.filename === basename(args.file!) && d.status === "ready")
  : undefined;
if (args.reuse && !existing) throw new Error("--reuse: no ready document with that filename");

if (!existing) await ingest(args.file);

async function ingest(file: string) {
  const uploadBegan = performance.now();
  const form = new FormData();
  form.append("file", Bun.file(file), basename(file));
  const upload = await fetch(`${args.url}/api/documents`, { method: "POST", body: form });
  if (!upload.ok) throw new Error(`upload HTTP ${upload.status}: ${await upload.text()}`);
  const { id } = (await upload.json()) as { id: string };

  // Gemma also writes the document summary while it processes. Waiting for the
  // summary as well as `ready` keeps it from competing with the first turn,
  // so the turns below measure the conversation alone.
  let readyAt: number | undefined;
  for (;;) {
    const documents = (await (await fetch(`${args.url}/api/documents`)).json()) as {
      id: string; status: string; processingStage: string; processingError: unknown;
      summary: string | null; pageCount: number;
    }[];
    const document = documents.find((d) => d.id === id);
    if (!document) throw new Error("uploaded document is not listed");
    if (document.processingError) throw new Error(`ingestion failed: ${JSON.stringify(document)}`);
    if (document.status === "ready" && readyAt === undefined) {
      readyAt = performance.now();
      console.log(JSON.stringify({ pages: document.pageCount, ingest_ready_s: +((readyAt - uploadBegan) / 1000).toFixed(1) }));
    }
    if (readyAt !== undefined && (document.summary || performance.now() - readyAt > 300_000)) {
      console.log(JSON.stringify({ summary_after_ready_s: +((performance.now() - readyAt) / 1000).toFixed(1), summary: document.summary?.slice(0, 120) }));
      break;
    }
    await Bun.sleep(1000);
  }
}

async function turn(messages: { role: string; content: string }[], previousSources: Source[]) {
  const began = performance.now();
  const response = await fetch(`${args.url}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ messages, previousSources }),
  });
  if (!response.ok || !response.body) throw new Error(`chat HTTP ${response.status}`);
  let first: number | undefined;
  let answer = "";
  let sources: Source[] = [];
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
      else if (event.type === "error") throw new Error(`chat error: ${event.message}`);
    }
  }
  const finished = performance.now();
  return {
    answer,
    sources,
    first_token_s: +(((first ?? finished) - began) / 1000).toFixed(1),
    total_s: +((finished - began) / 1000).toFixed(1),
  };
}

const questions = [
  "JADE 在 IoT4Industry 裡是怎麼部署的？",
  "那 Contract-Net 協定跑起來的流程是什麼？",
  "報告最後提到哪些替代的 IoT 平台或 agent framework？",
];
const history: { role: string; content: string }[] = [];
let previous: Source[] = [];
for (const [index, question] of questions.entries()) {
  history.push({ role: "user", content: question });
  const result = await turn(history, previous);
  let stablePrefix = 0;
  for (const [i, source] of result.sources.entries()) {
    if (previous[i]?.unitId !== source.unitId) break;
    stablePrefix++;
  }
  console.log(JSON.stringify({
    turn: index + 1,
    first_token_s: result.first_token_s,
    total_s: result.total_s,
    pages: result.sources.map((s) => s.pageNumber),
    stable_prefix: stablePrefix,
    answer: result.answer.replace(/\s+/g, " ").slice(0, 220),
  }));
  previous = result.sources;
  history.push({ role: "assistant", content: result.answer });
}
