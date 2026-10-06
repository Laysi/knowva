// Uploads copies of the given files one after another under a name prefix
// that is unique to this run, waits for each to be ready and summarized, then
// deletes it, so the knowledge base ends as it started.
//   bun stress-upload.ts [--prefix P] <file>...
//   bun stress-upload.ts --cleanup --prefix P    # delete what a run with P left
// --cleanup only takes an explicit prefix, so it never deletes documents that
// another run or a person uploaded.
import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { parseArgs } from "node:util";

const URL_BASE = "http://127.0.0.1:8080";
type Row = { id: string; filename: string; status: string; summary?: string | null; processingError?: string | null; pageCount?: number };

const { values: options, positionals: files } = parseArgs({
  allowPositionals: true,
  options: { prefix: { type: "string" }, cleanup: { type: "boolean", default: false } },
});

async function documents(): Promise<Row[]> {
  const response = await fetch(`${URL_BASE}/api/documents`, { signal: AbortSignal.timeout(30_000) });
  if (!response.ok) throw new Error(`list documents: HTTP ${response.status}`);
  return (await response.json()) as Row[];
}

async function remove(id: string) {
  const response = await fetch(`${URL_BASE}/api/documents/${id}`, { method: "DELETE", signal: AbortSignal.timeout(60_000) });
  if (response.status !== 204) throw new Error(`delete ${id}: HTTP ${response.status}`);
}

if (options.cleanup) {
  if (!options.prefix) {
    console.error("--cleanup needs the --prefix of the run to clean up");
    process.exit(2);
  }
  const leftovers = (await documents()).filter((d) => d.filename.startsWith(options.prefix!));
  for (const d of leftovers) await remove(d.id);
  console.log(JSON.stringify({ cleanup: leftovers.map((d) => d.filename) }));
  process.exit(0);
}

const prefix = options.prefix ?? `stress-${Date.now()}-${process.pid}-`;
const clock = (since: number) => +((performance.now() - since) / 1000).toFixed(1);
for (const path of files) {
  const name = prefix + basename(path);
  const began = performance.now();
  const form = new FormData();
  form.append("file", new File([readFileSync(path)], name));
  const upload = await fetch(`${URL_BASE}/api/documents`, { method: "POST", body: form, signal: AbortSignal.timeout(300_000) });
  if (!upload.ok) {
    console.log(JSON.stringify({ upload: name, status: upload.status, body: (await upload.text()).slice(0, 200) }));
    process.exitCode = 1;
    continue;
  }
  const { id } = (await upload.json()) as { id: string };
  const accepted = clock(began);
  let ready: number | undefined;
  let row: Row | undefined;
  let deleted = false;
  try {
    for (;;) {
      row = (await documents()).find((d) => d.id === id);
      if (!row || row.processingError || row.status === "failed") break;
      if (row.status === "ready") ready ??= clock(began);
      if (ready !== undefined && row.summary) break;
      if (clock(began) > 900) break;
      await Bun.sleep(1000);
    }
  } finally {
    // Also when polling fails, so no copy is left behind.
    try {
      await remove(id);
      deleted = true;
    } catch (error) {
      console.error(String(error));
      process.exitCode = 1;
    }
  }
  const summarized = row?.summary ? clock(began) : undefined;
  console.log(JSON.stringify({
    upload: name, status: row?.status, error: row?.processingError ?? undefined, pages: row?.pageCount,
    accepted_s: accepted, ready_s: ready, summary_s: summarized, deleted,
  }));
  if (row?.status !== "ready" || !summarized) process.exitCode = 1;
}
