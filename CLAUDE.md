# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Demo of a general-purpose, domain-neutral AI knowledge-management system. `PLAN.md` (zh-TW) is the spec. `docs/` (zh-TW) is the product cold-start plan — `mvp-now.md` (current work), `future-plan.md` (fully-on-prem roadmap), `deployment.md` (all-container deployment, applies from MVP), with `README.md` as overview; the product is a single-enterprise on-prem system, not SaaS, target hardware is NVIDIA + Linux only (no macOS), and cloud dev/testing must use the same models as delivery (Gemma family for the LLM; `llama-nemotron-embed-vl-1b-v2` / `llama-nemotron-rerank-vl-1b-v2` for retrieval — plan since 2026-07-10 is page-image visual retrieval, no transcription on the critical path). **Demo rule: always choose the simplest implementation** — no auth, no job queue (plain `tokio::spawn`), vector-only retrieval, chat history lives in frontend state only.

## Dev commands

- DB (required first): `docker compose up -d` — pgvector/pgvector:pg17 on :5432, user/pw/db = km/km/km
- Backend: `cd backend && cargo run` — **must run from `backend/`** (FILES_DIR `./data/files` and `../frontend/dist` are cwd-relative; root `.env` is found by dotenvy's ancestor search). Listens on :8080, runs sqlx migrations automatically.
- Frontend dev: `cd frontend && npm run dev` — :5173, proxies `/api` to :8080
- Prod-ish: `cd frontend && npm run build`, then the backend serves `frontend/dist` at :8080
- Checks: `cd backend && cargo build` and `cd frontend && npm run build` (tsc). No test suite (demo).
- To stop the backend use `pkill -x km-server` — never `pkill -f` (the pattern matches your own shell's command line).
- Model provider is chosen at startup via `MODEL_PROVIDER=bedrock|nvidia` in `.env` (default `bedrock`); restart the backend to switch. `nvidia` mode requires `NVIDIA_API_KEY` (build.nvidia.com) plus `NVIDIA_CHAT_MODEL` / `NVIDIA_EMBED_MODEL` / `NVIDIA_BASE_URL` (all default in `.env.example`).

## Architecture

- `backend/` — single axum crate `km-server`.
  - Routes: `GET /api/health`, `GET|POST /api/documents`, `DELETE /api/documents/{id}`, `POST /api/chat` (SSE).
  - SSE protocol (frontend depends on exact shapes): `data: {"type":"text-delta","text":...}` per delta, one `{"type":"sources","sources":[{documentId,filename}]}` before `{"type":"done"}`; failures emit `{"type":"error","message":...}` instead of panicking.
  - `llm.rs` — provider-neutral dispatch enum `Llm { Bedrock, Nvidia }`, chosen once at startup from `MODEL_PROVIDER`; holds `SYSTEM_PROMPT`/`ChatMessage`/`SseEvent` and delegates `stream_chat(_with_system)`/`converse_once`/`embed_query`/`embed_passage` to the matching module.
  - `bedrock.rs` — Nova 2 Lite via `converse_stream` (chat) / `converse_once` (summaries); Titan Text Embeddings V2 via `invoke_model` (1024-dim). Unchanged role, just imports shared types from `llm.rs`.
  - `nvidia.rs` — OpenAI-compatible REST client (reqwest) for NVIDIA Build (build.nvidia.com/NIM): SSE-parsed streaming `/chat/completions`, non-streaming variant for summaries, `/embeddings`. Strips inline `<think>...</think>` reasoning out of the delta stream via a small stateful filter (tag can split across chunk boundaries).
  - `ingest.rs` — upload → save to `backend/data/files/{uuid}.{ext}` → spawned pipeline: VLM transcription for non-md/txt (file bytes sent to the chat model via `Llm::transcribe_file`, ≤4MB, converted `.md` written alongside) → heading-aware chunking (merge ≤~2000 chars, hard max 3000, overlap 200) → embed → insert `chunks` → Nova summary/tags (best-effort JSON) → status `processing|ready|failed` on `documents`.
  - `search.rs` — RAG: embed query, pgvector cosine top-8 over `status='ready'`, builds system prompt with numbered sources; empty retrieval falls back to plain chat.
- `frontend/` — Vite + React TS + `@assistant-ui/react`, **no Tailwind** (plain CSS). `useLocalRuntime` + custom SSE adapter in `src/BackendAdapter.ts` — yields **cumulative** text (never deltas), appends a 資料來源 markdown block from the `sources` event. `Thread.tsx` is hand-built from assistant-ui primitives; `Documents.tsx` polls every 3s while any doc is processing. UI language: Traditional Chinese.
- Schema: two tables (`documents`, `chunks` with `vector(1024)` + HNSW), see `backend/migrations/`.

## Gotchas

- Bedrock auth uses an API key (bearer token). The SDK only recognizes the fixed env var name `AWS_BEARER_TOKEN_BEDROCK`. Per user decision (2026-07-04) the "never write the token into `.env`" rule is rescinded: `.env` now holds the Bedrock bearer token and `NVIDIA_API_KEY` directly (both are secrets — don't commit `.env`). Region/model come from `.env`: us-east-2, `us.amazon.nova-2-lite-v1:0` (verified working), `amazon.titan-embed-text-v2:0`.
- Document parsing is VLM-direct (markitdown was removed 2026-07-10 for poor quality): non-md/txt files go to the multimodal chat model as-is. Bedrock accepts PDF/DOC(X)/XLS(X)/CSV/HTML via Converse document blocks and PNG/JPG/GIF/WebP via image blocks; **PPTX is not supported** (Converse has no PPT document format — users must export to PDF). The nvidia provider only supports images for binary uploads (OpenAI-compatible chat has no document input) and requires a vision-capable `NVIDIA_CHAT_MODEL`. Transcription cap: 4MB per file (`MAX_TRANSCRIBE_BYTES`, under Bedrock's 4.5MB document limit). Scanned-PDF quality depends on the model's visual document understanding — unverified.
- Egress depends on `MODEL_PROVIDER`: `bedrock-runtime.<region>.amazonaws.com` for bedrock, `integrate.api.nvidia.com` for nvidia. Everything else is on-prem.
- Embedding spaces are incompatible between providers (Titan vs NVIDIA's embed model produce unrelated vector spaces) — after switching `MODEL_PROVIDER`, delete and re-upload existing documents, or retrieval silently returns garbage.
- `NVIDIA_EMBED_MODEL` must yield 1024-dim vectors to match the DB's `vector(1024)` schema; `nvidia.rs` fails fast if not. Default `nvidia/llama-nemotron-embed-1b-v2` (natively 2048) needs `NVIDIA_EMBED_DIMENSIONS=1024` (Matryoshka); natively-1024 models like `nvidia/nv-embedqa-e5-v5` reject the `dimensions` field — leave `NVIDIA_EMBED_DIMENSIONS` empty for those. Verified live 2026-07-04: `baai/bge-m3` is listed by `/v1/models` but not invocable (generic 4xx), `nvidia/llama-3.2-nv-embedqa-1b-v2` is EOL (410), e5-v5 works but truncates input at 512 tokens (too short for 2000–3000-char chunks).

