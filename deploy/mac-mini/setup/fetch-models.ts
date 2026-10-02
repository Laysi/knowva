// Downloads the two Nemotron VL snapshots at pinned revisions and verifies
// every file: LFS weights against their SHA-256, everything else (including
// the remote code that trust_remote_code executes) against its git blob id.
// Files already present and intact are left alone, so re-running only checks.
//
//   bun setup/fetch-models.ts [dest]   # dest defaults to ~/knowva-inference/models

import { createHash } from "node:crypto";
import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// The rerank revision is the last one before the 2026-10-01 chat-template
// change; it is the code the M1 measurements in docs/measurements ran.
const MODELS = [
  { repo: "nvidia/llama-nemotron-embed-vl-1b-v2", rev: "582e3bf72aee355e3c59ed89de53543c5b0657ee" },
  { repo: "nvidia/llama-nemotron-rerank-vl-1b-v2", rev: "0c733be3b6451289f4288c0b170376d1fbeda55a" },
];

type Entry = { type: string; path: string; oid: string; size: number; lfs?: { oid: string } };

const root = process.argv[2] ?? join(homedir(), "knowva-inference", "models");

async function intact(path: string, entry: Entry) {
  const file = Bun.file(path);
  if (!(await file.exists()) || file.size !== entry.size) return false;
  // Streamed: the embed weights alone are 3 GB.
  const hash = entry.lfs
    ? createHash("sha256")
    : createHash("sha1").update(`blob ${file.size}\0`);
  for await (const chunk of file.stream()) hash.update(chunk);
  return hash.digest("hex") === (entry.lfs?.oid ?? entry.oid);
}

for (const { repo, rev } of MODELS) {
  const dest = join(root, repo.split("/")[1]);
  const tree = await fetch(`https://huggingface.co/api/models/${repo}/tree/${rev}?recursive=true`);
  if (!tree.ok) throw new Error(`${repo}: tree listing failed with HTTP ${tree.status}`);
  const files = ((await tree.json()) as Entry[]).filter((entry) => entry.type === "file");

  for (const entry of files) {
    const path = join(dest, entry.path);
    if (await intact(path, entry)) continue;
    mkdirSync(dirname(path), { recursive: true });
    const response = await fetch(`https://huggingface.co/${repo}/resolve/${rev}/${entry.path}`);
    if (!response.ok) throw new Error(`${repo}/${entry.path}: HTTP ${response.status}`);
    const partial = `${path}.partial`;
    await Bun.write(partial, response);
    if (!(await intact(partial, entry))) throw new Error(`${repo}/${entry.path}: checksum mismatch`);
    renameSync(partial, path);
    console.log(`fetched ${repo}/${entry.path}`);
  }
  writeFileSync(join(dest, "REVISION"), `${rev}\n`);
  console.log(`verified ${repo}@${rev} (${files.length} files)`);
}
