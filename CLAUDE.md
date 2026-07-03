# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Demo of a general-purpose, domain-neutral AI knowledge-management system. `PLAN.md` (zh-TW) is the spec. **Demo rule: always choose the simplest implementation** — no auth, no job queue (plain `tokio::spawn`), vector-only retrieval, chat history lives in frontend state only.

## Dev commands

- DB (required first): `docker compose up -d` — pgvector/pgvector:pg17 on :5432, user/pw/db = km/km/km
- Backend: `cd backend && cargo run` — **must run from `backend/`** (FILES_DIR `./data/files` and `../frontend/dist` are cwd-relative; root `.env` is found by dotenvy's ancestor search). Listens on :8080, runs sqlx migrations automatically.
- Frontend dev: `cd frontend && npm run dev` — :5173, proxies `/api` to :8080
- Prod-ish: `cd frontend && npm run build`, then the backend serves `frontend/dist` at :8080
- Checks: `cd backend && cargo build` and `cd frontend && npm run build` (tsc). No test suite (demo).
- To stop the backend use `pkill -x km-server` — never `pkill -f` (the pattern matches your own shell's command line).

## Architecture

- `backend/` — single axum crate `km-server`.
  - Routes: `GET /api/health`, `GET|POST /api/documents`, `DELETE /api/documents/{id}`, `POST /api/chat` (SSE).
  - SSE protocol (frontend depends on exact shapes): `data: {"type":"text-delta","text":...}` per delta, one `{"type":"sources","sources":[{documentId,filename}]}` before `{"type":"done"}`; failures emit `{"type":"error","message":...}` instead of panicking.
  - `bedrock.rs` — Nova 2 Lite via `converse_stream` (chat) / `converse_once` (summaries); Titan Text Embeddings V2 via `invoke_model` (1024-dim).
  - `ingest.rs` — upload → save to `backend/data/files/{uuid}.{ext}` → spawned pipeline: markitdown subprocess for non-md/txt (converted `.md` written alongside) → heading-aware chunking (merge ≤~2000 chars, hard max 3000, overlap 200) → embed → insert `chunks` → Nova summary/tags (best-effort JSON) → status `processing|ready|failed` on `documents`.
  - `search.rs` — RAG: embed query, pgvector cosine top-8 over `status='ready'`, builds system prompt with numbered sources; empty retrieval falls back to plain chat.
- `frontend/` — Vite + React TS + `@assistant-ui/react`, **no Tailwind** (plain CSS). `useLocalRuntime` + custom SSE adapter in `src/BackendAdapter.ts` — yields **cumulative** text (never deltas), appends a 資料來源 markdown block from the `sources` event. `Thread.tsx` is hand-built from assistant-ui primitives; `Documents.tsx` polls every 3s while any doc is processing. UI language: Traditional Chinese.
- Schema: two tables (`documents`, `chunks` with `vector(1024)` + HNSW), see `backend/migrations/`.

## Gotchas

- AWS credentials come from an `aws login` session → `aws-config` requires the `credentials-login` cargo feature (already enabled; do not remove). Region/model come from `.env`: us-east-2, `us.amazon.nova-2-lite-v1:0` (verified working), `amazon.titan-embed-text-v2:0`.
- The login session expires after a few hours. Symptom: chat SSE returns `{"type":"error","message":"dispatch failure"}` and the server log shows `The refresh token has expired`. Fix: the user re-runs `aws login` (interactive; no server restart needed — credentials are re-resolved per request).
- `markitdown` must be on PATH or at `~/.local/bin/markitdown` (`pipx install "markitdown[all]"`); md/txt files bypass it. Scanned PDFs are unsupported by design.
- Everything is on-prem except Bedrock calls; only egress needed is `bedrock-runtime.<region>.amazonaws.com`.

## Frontend rewrite policy

For frontend visual quality problems, prefer ui-rewriter directly.

Use ui-rewriter when:
- The UI looks ugly, cheap, inconsistent, or template-like.
- The task requires product polish, visual hierarchy, spacing, typography, or layout judgment.
- The goal is to make the demo feel like a real B2B SaaS product.

Do not route frontend visual redesign through fast-worker first.
fast-worker may only handle repetitive follow-up edits after ui-rewriter has established the visual direction.

The main Fable session should review the final result and request one focused revision if needed.