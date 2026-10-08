#!/usr/bin/env bash
# Exports a Knowva instance's retrieval units (content, page image key,
# needs_vision, stored Nemotron vector), the page images that are embedded as
# images, and the answerable questions of bench/examples.json and
# bench/realistic.json with their expected sources.
#
#   export-corpus.sh <postgres-container> <db-user> <db-name> <files-dir> <out-dir>
#
# <files-dir> is the instance's FILES_DIR (local storage). Writes
# corpus.jsonl, queries.json and corpus-images.tar to <out-dir>.
set -euo pipefail
container=$1 user=$2 db=$3 files=$4 out=$5
bench=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"

psql() { podman exec "$container" psql -U "$user" -d "$db" -At -c "$1"; }

# uses_image mirrors page_vision::page_uses_image.
psql "SELECT json_build_object(
    'id', u.id, 'document_id', u.document_id, 'filename', d.filename, 'kind', u.kind,
    'unit_index', u.unit_index, 'page', u.page_number, 'content', u.content,
    'image_path', u.image_path, 'needs_vision', u.needs_vision,
    'uses_image', (u.needs_vision OR u.content IS NULL OR btrim(u.content) = ''),
    'nemotron', u.embedding::text)
  FROM document_units u JOIN documents d ON d.id = u.document_id
  WHERE u.retrieval_ready AND d.status = 'ready' ORDER BY u.id" > "$out/corpus.jsonl"

psql "SELECT image_path FROM document_units
  WHERE image_path IS NOT NULL AND (needs_vision OR content IS NULL OR btrim(content) = '')" \
  | tar cf "$out/corpus-images.tar" -C "$files" -T -

# The backend retrieves with the last user message only.
bun -e '
const [bench, out] = process.argv.slice(1);
const questions = [];
for (const set of ["examples", "realistic"])
  for (const c of (await Bun.file(`${bench}/${set}.json`).json()).cases)
    if (c.source) questions.push({ id: c.id, set, category: c.category, query: c.turns.at(-1), gold: c.source });
await Bun.write(`${out}/queries.json`, JSON.stringify(questions, null, 1));
console.log(`${questions.length} questions`);
' "$bench" "$out"
echo "$(wc -l < "$out/corpus.jsonl") units, $(tar tf "$out/corpus-images.tar" | wc -l) images"
