# CLAUDE.md

This file provides repository guidance for coding agents.

## Project

Demo of a general-purpose AI knowledge-management system. For the active MVP, all inference uses cloud-hosted APIs and the model families are fixed: Gemma 4 for chat/multimodal reading, `llama-nemotron-embed-vl-1b-v2` for embeddings, and `llama-nemotron-rerank-vl-1b-v2` for reranking. Do not add alternative or local inference paths.

The specification priority is the active GitHub issue, then tracker issue #1, then `docs/mvp-now.md`. Keep implementations simple: no auth, no job queue, and chat history remains in frontend state.

## Development commands

- Install pinned toolchains: `proto install`
- Full stack: configure `.env`, then `docker compose up -d --build`
- Database only: `docker compose up -d postgres`
- Backend: `cd backend && proto exec rust -- cargo run` (run from `backend/` because data and frontend paths are working-directory relative)
- Frontend: `cd frontend && proto run npm -- run dev`
- Backend checks: run `proto exec rust -- cargo fmt --check`, `proto exec rust -- cargo clippy --all-targets -- -D warnings`, and `proto exec rust -- cargo test` from `backend/`
- Frontend checks: run `proto run npm -- ci`, `proto run npm -- run lint`, and `proto run npm -- run build` from `frontend/`
- Native conversion smoke test: `docker build --target native-test -t ai-km-demo:native-test .`
- Stop the backend with `pkill -x km-server`; never use `pkill -f`.

The backend requires `NVIDIA_API_KEY` and a confirmed `GEMMA_4_MODEL_ID`. Shared and per-capability cloud URLs, the two locked Nemotron model IDs, and request timeout are documented in `.env.example`. Restart after changing configuration.

## Architecture

- `backend/` is one axum crate.
  - Routes: `GET /api/health`, `GET|POST /api/documents`, `DELETE /api/documents/{id}`, and `POST /api/chat`.
  - `cloud_models.rs` owns fail-fast configuration, typed chat/embed/rerank payloads, the unified safe error type, and OpenAI-compatible chat SSE parsing.
  - SSE events remain `text-delta`, optional `sources`, then `done`; failures emit `error`. When sources exist they must precede `done`.
  - `ingest.rs` owns upload, atomic artifact publishing, staging persistence, and delete/retry cleanup. New documents remain `processing` after artifacts are ready until the later embedding issue promotes them into searchable units.
  - `page_artifacts.rs` uses Pdfium for PDF rasterization/text-layer extraction, normalizes images, and runs isolated headless LibreOffice conversion for DOCX/PPTX/XLSX. This stage never calls a model or fabricates embeddings.
  - `search.rs` contains the current dense retrieval path. Hybrid retrieval and reranking integration are later MVP work.
- `frontend/` is Vite + React TypeScript + assistant-ui with plain CSS.
- SQLx migrations run automatically at backend startup.

## Cloud model safety

- Parse model configuration before connecting to the database so missing secrets/model IDs fail before traffic is accepted.
- Logs may include endpoint and model ID, never the API key or request Authorization header.
- Model HTTP errors use a maximum 500-character sanitized snippet and expose HTTP status, timeout, and retryability through one error type.
- The Gemma model ID has no repository default. Embed and rerank settings may only vendor-qualify their locked family names.
- Do not send an embedding dimension-reduction parameter; the fixed Embed VL response is validated as 2048 dimensions.

## Document conversion

- Pdfium is a runtime shared-library dependency for PDF/Office page rendering; `PDFIUM_LIBRARY_PATH` may point to the library or its directory.
- LibreOffice is only required for DOCX/PPTX/XLSX and is selected with `LIBREOFFICE_BIN`.
- The runtime image pins Pdfium Chromium 7763 by architecture and includes LibreOffice plus Noto CJK fonts; it runs as the non-root `km` user.
- The default page render setting is 180 DPI and conversion timeout is 120 seconds; see `.env.example`.
- The real PDF/Office fixture smoke test is explicitly ignored on machines without both native dependencies. The `native-test` image target provides a reproducible opt-in run; its command and expected page counts are documented in `README.md` and `backend/tests/fixtures/README.md`.

## Working tree safety

Preserve unrelated user changes. Use focused patches, run the backend checks above, and do not push or create a PR unless explicitly requested.
