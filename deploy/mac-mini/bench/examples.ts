// Acceptance run over example_file/ through the running application: uploads
// every example (or reuses what is there), checks how each one was ingested,
// then asks the questions in examples.json. An answer passes when the expected
// source was retrieved and cited and the text has every required phrase and
// none of the forbidden ones.
//
//   bun bench/examples.ts --files ~/knowva-inference/example_file --replace \
//     --db-container knowva-postgres-1 --out ~/knowva-inference/examples.json
//   bun bench/examples.ts --only '^pdf-'      # questions only, over what is uploaded
//   bun bench/examples.ts --dataset bench/realistic.json --fetch \
//     --files ~/knowva-inference/realistic --db-container knowva-postgres-1
//
// --replace deletes earlier uploads that share an example's filename once the
// new upload is ready, so retrieval never sees two copies; a rejected or failed
// upload leaves them in place. --db-container reads which pages were sent as
// images (needs_vision), which the API does not expose. --fetch downloads files
// the dataset gives a url for when they are not in --files yet. Every file with
// a sha256 is checked before it is uploaded, and the checked bytes are what
// gets uploaded.

import {
  closeSync, constants, existsSync, mkdirSync, openSync, readFileSync, realpathSync, renameSync, rmSync,
  writeFileSync,
} from "node:fs";
import { basename, resolve, sep } from "node:path";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({
  options: {
    url: { type: "string", default: "http://127.0.0.1:8080" },
    dataset: { type: "string", default: new URL("./examples.json", import.meta.url).pathname },
    files: { type: "string" },
    replace: { type: "boolean", default: false },
    fetch: { type: "boolean", default: false },
    "db-container": { type: "string" },
    "db-user": { type: "string", default: "knowva" },
    "db-name": { type: "string", default: "knowva" },
    only: { type: "string" },
    out: { type: "string" },
  },
});

type FileSpec = {
  filename: string; pages?: number; visionPages?: number[]; rejectStatus?: number;
  url?: string; sha256?: string;
};
type Case = {
  id: string;
  category: string;
  turns: string[];
  source?: { filename: string; pages?: (number | null)[] };
  mustInclude?: string[][];
  mustNotMatch?: string[];
  notFound?: boolean;
  catalog?: boolean;
};
type Source = { sourceId: string; filename: string; pageNumber: number | null; unitId: number };
type DocumentRow = {
  id: string; filename: string; status: string; processingError: unknown;
  summary: string | null; pageCount: number;
};

const dataset = JSON.parse(readFileSync(args.dataset!, "utf8")) as { files: FileSpec[]; cases: Case[] };

// Phrases the answering prompt leads Gemma to use when the sources do not
// contain the answer (backend/src/rag_context.rs RAG_SYSTEM_PROMPT).
const NOT_FOUND_MARKERS = [
  "不足", "沒有提供", "未提供", "未提及", "沒有提及", "沒有提到", "未提到", "沒有相關", "找不到",
  "無法回答", "無法從", "無法確定", "並未", "沒有記載", "未記載", "沒有說明", "未說明", "沒有包含", "未包含",
];

async function documents(): Promise<DocumentRow[]> {
  const response = await fetch(`${args.url}/api/documents`, { signal: AbortSignal.timeout(30_000) });
  if (!response.ok) throw new Error(`list documents: HTTP ${response.status}`);
  return (await response.json()) as DocumentRow[];
}

function visionPages(documentId: string): number[] {
  if (!/^[0-9a-f-]{36}$/.test(documentId)) throw new Error(`unexpected document id ${documentId}`);
  const sql = `SELECT page_number FROM page_artifacts WHERE document_id = '${documentId}' AND needs_vision ORDER BY 1`;
  const result = Bun.spawnSync([
    "docker", "exec", args["db-container"]!, "psql", "-U", args["db-user"]!, "-d", args["db-name"]!, "-Atc", sql,
  ]);
  if (result.exitCode !== 0) throw new Error(`psql failed: ${result.stderr.toString().trim()}`);
  return result.stdout.toString().split("\n").filter(Boolean).map(Number);
}

// Some publishers (osha.gov) answer 403 to requests without a browser user agent.
const BROWSER_USER_AGENT =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36";

const sha256 = (bytes: Uint8Array) => new Bun.CryptoHasher("sha256").update(bytes).digest("hex");

// Dataset filenames are plain names inside --files, never paths, and an
// existing file must not lead outside --files through a symlink.
function filePath(filename: string): string {
  const root = resolve(args.files!);
  const path = resolve(root, filename);
  if (filename !== basename(filename) || filename.includes("\\") || filename === ".." || !path.startsWith(root + sep)) {
    throw new Error(`${JSON.stringify(filename)}: filenames must be plain names inside --files`);
  }
  if (existsSync(path) && !realpathSync(path).startsWith(realpathSync(root) + sep)) {
    throw new Error(`${JSON.stringify(filename)}: resolves outside --files`);
  }
  return path;
}

function verify(spec: FileSpec, bytes: Uint8Array) {
  if (spec.sha256 && sha256(bytes) !== spec.sha256) {
    throw new Error(`${spec.filename}: sha256 ${sha256(bytes)} does not match the dataset`);
  }
}

async function fetchFile(spec: FileSpec) {
  const path = filePath(spec.filename);
  if (existsSync(path)) return;
  if (!spec.url) throw new Error(`${spec.filename}: not in ${args.files} and the dataset has no url`);
  const response = await fetch(spec.url, {
    headers: { "user-agent": BROWSER_USER_AGENT },
    signal: AbortSignal.timeout(300_000),
  });
  if (!response.ok) throw new Error(`${spec.filename}: download HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  verify(spec, bytes);
  // An interrupted download never leaves a partial file under the real name,
  // and the temporary file is created fresh rather than through a symlink.
  const part = `${path}.part`;
  rmSync(part, { force: true });
  const fd = openSync(part, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o644);
  try {
    writeFileSync(fd, bytes);
  } finally {
    closeSync(fd);
  }
  renameSync(part, path);
}

const sameList = (left: number[], right: number[]) =>
  left.length === right.length && left.every((value, index) => value === right[index]);

async function ingest(spec: FileSpec) {
  const bytes = readFileSync(filePath(spec.filename));
  verify(spec, bytes);
  const earlier = args.replace
    ? (await documents()).filter((d) => d.filename === spec.filename).map((d) => d.id)
    : [];
  const began = performance.now();
  const form = new FormData();
  form.append("file", new File([bytes], spec.filename));
  const upload = await fetch(`${args.url}/api/documents`, {
    method: "POST",
    body: form,
    signal: AbortSignal.timeout(300_000),
  });
  if (spec.rejectStatus !== undefined) {
    const body = (await upload.text()).slice(0, 120);
    return { filename: spec.filename, pass: upload.status === spec.rejectStatus, status: upload.status, body };
  }
  if (!upload.ok) {
    return { filename: spec.filename, pass: false, status: upload.status, body: (await upload.text()).slice(0, 200) };
  }
  const { id } = (await upload.json()) as { id: string };

  // Wait for the summary as well as `ready`, so the next upload and the
  // questions do not compete with Gemma writing it.
  let readyAt: number | undefined;
  let document: DocumentRow | undefined;
  for (;;) {
    document = (await documents()).find((d) => d.id === id);
    if (!document) throw new Error(`${spec.filename}: uploaded document is not listed`);
    if (document.processingError || document.status === "failed") break;
    if (document.status === "ready") readyAt ??= performance.now();
    if (readyAt !== undefined && (document.summary || performance.now() - readyAt > 300_000)) break;
    if (performance.now() - began > 900_000) break;
    await Bun.sleep(1000);
  }
  document = document!;
  if (document.status === "ready") {
    for (const old of earlier) {
      const removed = await fetch(`${args.url}/api/documents/${old}`, {
        method: "DELETE",
        signal: AbortSignal.timeout(30_000),
      });
      if (removed.status !== 204) throw new Error(`delete ${old}: HTTP ${removed.status}`);
    }
  }
  const vision = args["db-container"] && document.status === "ready" ? visionPages(id) : undefined;
  const checks = {
    ready: document.status === "ready",
    pages: spec.pages === undefined || document.pageCount === spec.pages,
    vision: vision === undefined || spec.visionPages === undefined || sameList(vision, spec.visionPages),
  };
  return {
    filename: spec.filename,
    pass: Object.values(checks).every(Boolean),
    checks,
    status: document.status,
    error: document.processingError ?? undefined,
    pages: document.pageCount,
    vision_pages: vision,
    ingest_ready_s: readyAt === undefined ? undefined : +((readyAt - began) / 1000).toFixed(1),
    summary: document.summary?.slice(0, 80),
  };
}

async function turn(messages: { role: string; content: string }[], previousSources: Source[]) {
  const began = performance.now();
  const response = await fetch(`${args.url}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ messages, previousSources }),
    signal: AbortSignal.timeout(600_000),
  });
  if (!response.ok || !response.body) throw new Error(`chat HTTP ${response.status}`);
  let first: number | undefined;
  let answer = "";
  let sources: Source[] = [];
  let error: string | undefined;
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
    }
  }
  const finished = performance.now();
  return {
    answer, sources, error,
    first_token_s: +(((first ?? finished) - began) / 1000).toFixed(1),
    total_s: +((finished - began) / 1000).toFixed(1),
  };
}

function citedIds(answer: string): Set<string> {
  const ids = new Set<string>();
  for (const [, inside] of answer.matchAll(/\[([^\]]{1,40})\]/g)) {
    for (const part of inside.split(/[,，、\s]+/)) if (/^S\d+$/.test(part)) ids.add(part);
  }
  return ids;
}

function score(spec: Case, answer: string, sources: Source[]) {
  const text = answer.toLowerCase();
  const forbidden = (spec.mustNotMatch ?? []).filter((pattern) => new RegExp(pattern, "i").test(answer));
  if (spec.catalog) {
    const expected = dataset.files.filter((f) => f.rejectStatus === undefined).map((f) => f.filename);
    const missing = expected.filter((filename) => !sources.some((s) => s.filename === filename));
    return { pass: missing.length === 0, missing };
  }
  if (spec.notFound) {
    const refused = NOT_FOUND_MARKERS.some((marker) => answer.includes(marker));
    return { pass: refused && forbidden.length === 0, refused, forbidden };
  }
  const matching = sources.filter((s) =>
    s.filename === spec.source!.filename &&
    (spec.source!.pages === undefined || spec.source!.pages.includes(s.pageNumber ?? null)));
  const cited = citedIds(answer);
  const missing = (spec.mustInclude ?? [])
    .filter((group) => !group.some((alternative) => text.includes(alternative.toLowerCase())))
    .map((group) => group.join("|"));
  const retrieved = matching.length > 0;
  const citedExpected = matching.some((s) => cited.has(s.sourceId));
  return {
    pass: retrieved && citedExpected && missing.length === 0 && forbidden.length === 0,
    retrieved,
    cited: citedExpected,
    missing,
    forbidden,
  };
}

const report: { files: unknown[]; cases: unknown[] } = { files: [], cases: [] };

if (args.files) {
  if (args.fetch) {
    mkdirSync(args.files, { recursive: true });
    for (const spec of dataset.files) await fetchFile(spec);
  }
  for (const spec of dataset.files) {
    const result = await ingest(spec);
    report.files.push(result);
    console.log(JSON.stringify({ file: result }));
  }
}

const only = args.only ? new RegExp(args.only) : undefined;
for (const spec of dataset.cases.filter((c) => !only || only.test(c.id))) {
  const history: { role: string; content: string }[] = [];
  let previous: Source[] = [];
  let last: Awaited<ReturnType<typeof turn>> | undefined;
  for (const question of spec.turns) {
    history.push({ role: "user", content: question });
    last = await turn(history, previous);
    history.push({ role: "assistant", content: last.answer });
    previous = last.sources;
  }
  const result = {
    id: spec.id,
    category: spec.category,
    ...(last!.error ? { pass: false, error: last!.error } : score(spec, last!.answer, last!.sources)),
    first_token_s: last!.first_token_s,
    total_s: last!.total_s,
    sources: last!.sources.map((s) => `${s.sourceId}:${s.filename}${s.pageNumber === null ? "" : `#${s.pageNumber}`}`),
    answer: last!.answer,
  };
  report.cases.push(result);
  console.log(JSON.stringify({ case: { ...result, answer: result.answer.replace(/\s+/g, " ").slice(0, 200) } }));
}

// Summary per category, then the failures.
const byCategory = new Map<string, { total: number; passed: number; cited: number; answerable: number }>();
for (const result of report.cases as { category: string; pass: boolean; cited?: boolean; retrieved?: boolean }[]) {
  const row = byCategory.get(result.category) ?? { total: 0, passed: 0, cited: 0, answerable: 0 };
  row.total++;
  if (result.pass) row.passed++;
  if (result.retrieved !== undefined) row.answerable++;
  if (result.cited) row.cited++;
  byCategory.set(result.category, row);
}
console.log("\n| category | passed | cited expected source |\n| --- | --- | --- |");
for (const [category, row] of byCategory) {
  console.log(`| ${category} | ${row.passed}/${row.total} | ${row.answerable ? `${row.cited}/${row.answerable}` : "—"} |`);
}
const filesPassed = (report.files as { pass: boolean }[]).filter((f) => f.pass).length;
const casesPassed = (report.cases as { pass: boolean }[]).filter((c) => c.pass).length;
console.log(`\nfiles ${filesPassed}/${report.files.length}, cases ${casesPassed}/${report.cases.length}`);
if (args.out) writeFileSync(args.out, JSON.stringify(report, null, 2));
process.exitCode = filesPassed === report.files.length && casesPassed === report.cases.length ? 0 : 1;
