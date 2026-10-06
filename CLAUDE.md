# CLAUDE.md

This file provides repository guidance for coding agents.

## Project

General-purpose AI knowledge-management service built for enterprise on-premise deployment. Production inference runs on an on-premise service reached over configured HTTP endpoints; during development and testing the same endpoints may point at a hosted OpenAI-compatible service. The model families are fixed: Gemma 4 for chat/multimodal reading, `llama-nemotron-embed-vl-1b-v2` for embeddings, and `llama-nemotron-rerank-vl-1b-v2` for reranking. Do not add in-process inference or provider-specific SDK paths; a different service is a URL change.

The specification priority is the active GitHub issue, then tracker issue #1, then `docs/mvp-now.md`. Keep implementations simple: no auth, no job queue, and chat history remains in frontend state.

## Development commands

- Install pinned toolchains: `proto install`
- Full stack: configure `.env`, then `docker compose up -d --build`
- Database only: `docker compose up -d postgres`
- Backend: `cd backend && proto exec rust -- cargo run` (run from `backend/` because data and frontend paths are working-directory relative)
- Frontend: `cd frontend && proto run npm -- run dev` (starts the default theme; `--workspace @knowva/theme-default` names it explicitly)
- Backend checks: run `proto exec rust -- cargo fmt --check`, `proto exec rust -- cargo clippy --all-targets -- -D warnings`, and `proto exec rust -- cargo test` from `backend/`
- Frontend checks: run `proto run npm -- ci`, `proto run npm -- run lint`, `proto run npm -- run test`, and `proto run npm -- run build` from `frontend/`; each runs every workspace
- Native conversion smoke test: `docker build --target native-test -t knowva:native-test .`
- Stop the backend with `pkill -x knowva-server`; never use `pkill -f`.

The backend requires a confirmed `GEMMA_4_MODEL_ID` and `MODEL_BASE_URL` (unless all three per-capability `MODEL_*_URL` routes are set). `MODEL_API_TOKEN` is optional: an unauthenticated on-premise service is addressed without any authorization header. The per-capability routes, the two locked Nemotron model IDs, and request timeout are documented in `.env.example`. Restart after changing configuration.

## Architecture

- `backend/` is one axum crate.
  - Routes: `GET /api/health`, `GET|POST /api/documents`, `DELETE /api/documents/{id}`, and `POST /api/chat`.
  - `cloud_models.rs` owns fail-fast configuration, typed chat/embed/rerank payloads, the unified safe error type, and OpenAI-compatible chat SSE parsing.
  - SSE events remain `text-delta`, optional `sources`, then `done`; failures emit `error`. When sources exist they must precede `done`.
  - `ingest.rs` owns upload, atomic artifact publishing, bounded-concurrency Embed VL promotion, transactional/idempotent `document_units` replacement, and delete/retry cleanup. A document becomes `ready` only after every staged artifact has one validated embedding and the counts match.
  - `storage.rs` owns init-time locked file storage (`STORAGE_BACKEND=local|s3` through `object_store`, S3 endpoint/region/credentials via generic `STORAGE_*` applied over `AmazonS3Builder::from_env` so platform IAM credentials still work): relative `{document_id}/...` keys in DB rows and manifests, document-scoped key validation, size-bounded reads, prefix deletion, and the `storage_settings` lock from migration `0006_storage_settings.sql`. Conversion runs in a per-run local scratch directory; uploads go to a fresh `artifacts-{run_id}/` prefix and the DB commit is the only publication switch.
  - `page_artifacts.rs` uses Pdfium for PDF rasterization/text-layer extraction, normalizes images, and runs isolated headless LibreOffice conversion for DOCX/PPTX/XLSX. Artifact conversion itself never calls a model or fabricates embeddings; ingestion embeds the published artifacts afterward.
  - `page_vision.rs` decides once per rendered page, from Pdfium page objects and without any model, whether the page needs vision (`needs_vision`, migration `0007_page_vision.sql`). Text-only pages are embedded, reranked, summarized, and answered from their text layer and never read their image; every stage goes through `page_uses_image`. Thresholds are constants there, uncertain pages count as vision, and image uploads always do. Images and shapes drawn with identical content and position on most of a document's pages, and confirmed to look the same in a text-masked 72 dpi rendering, are page furniture and do not count, so the decision is made after every page is observed.
  - `search.rs` owns injectable dense/BM25 candidate stages, deterministic RRF, safe artifact-to-passage preparation, VL reranking batched per modality (image vs text-only), strict index/score mapping, and transient-only RRF fallback. Dense and lexical paths default to top-50 over ready-only partial HNSW/GIN indexes; BM25 corpus statistics are transactionally maintained by migration `0004_hybrid_retrieval.sql`. Rerank output defaults to six candidates and is configurable from three to eight.
- `frontend/` is an npm workspaces root (Vite + React TypeScript + assistant-ui with plain CSS); the only lockfile is `frontend/package-lock.json`.
  - `packages/core` (`@knowva/core`) owns `BackendAdapter`, `DocumentModel`, `useDocuments` (list, polling, upload, delete) and the safe error messages, with no CSS or layout. `@knowva/core/testing` exports `describeThemeContract`, which every theme's vitest runs against the theme's own components and `:root` colours; `@knowva/core/vite` exports `offlineBundle`, which fails a build whose HTML, SVG or CSS loads anything from another host.
  - `themes/default` (`@knowva/theme-default`) is the current interface. It builds into the flat `frontend/dist/` that the backend and the Docker image serve.
- SQLx migrations run automatically at backend startup.

## Cloud model safety

- Model traffic goes to the configured inference service (on-premise in production, optionally hosted during development) over the OpenAI-compatible chat/embeddings routes and the NVIDIA ranking route; there are no hosted-provider endpoint defaults. Endpoints are configuration, so no URL is hardcoded and no hosted gateway is required.
- `MODEL_API_TOKEN` is optional and sent as a standard `Authorization: Bearer` header; when it is empty no authorization header is sent at all.
- Parse model configuration before connecting to the database so missing settings/model IDs fail before traffic is accepted.
- Logs may include endpoint and model ID, never the API token or request auth headers.
- Model HTTP errors use a maximum 500-character sanitized snippet and expose HTTP status, timeout, and retryability through one error type.
- The Gemma model ID has no repository default. Embed and rerank settings may only vendor-qualify their locked family names.
- Do not send an embedding dimension-reduction parameter; the fixed Embed VL response is validated as 2048 dimensions.
- Embed query input is text-only. Passage input may be text (MD/TXT chunks and `needs_vision = false` pages), image-only, or same-page image+text. Images are sent as bounded PNG/JPEG data URLs, never as local artifact paths; logs/errors must redact data URLs and long base64 strings.
- Embedding retries are finite exponential backoff with jitter for 408, 429, 5xx, transport errors, and timeouts. Other 4xx responses fail immediately.

## Document conversion

- Pdfium is a runtime shared-library dependency for PDF/Office page rendering; `PDFIUM_LIBRARY_PATH` may point to the library or its directory.
- LibreOffice is only required for DOCX/PPTX/XLSX and is selected with `LIBREOFFICE_BIN`.
- The runtime image pins Pdfium Chromium 7763 by architecture and includes LibreOffice plus Noto CJK fonts; it runs as the non-root `knowva` user.
- The default page render setting is 180 DPI and conversion timeout is 120 seconds; see `.env.example`.
- The real PDF/Office fixture smoke test is explicitly ignored on machines without both native dependencies. It also verifies native PDF image+text and scanned PDF image-only requests against a mock endpoint, and a second ignored test checks per-page `needs_vision` decisions on generated fixtures and `example_file/iiot_industry_report.pdf` (only page 7 needs vision). The `native-test` image target provides a reproducible opt-in run; its command and expected page counts are documented in `README.md` and `backend/tests/fixtures/README.md`.
- A separate ignored real-cloud Embed VL smoke covers all four input shapes and must only be run explicitly with configured credentials; normal tests and CI use deterministic mock endpoints.

## Working tree safety

Preserve unrelated user changes. Use focused patches, run the backend checks above, and do not push or create a PR unless explicitly requested.

## Project management (GitHub issues)

Keeping the tracking issue in sync is part of done: when issue-tracked work gets blocked, lands a PR/commit, or completes, update the issue in the same session. Issue operations run through the org-distributed `github-loop-engineering:github-loop-engineering-skill`; the rules below are this repo's policy and take precedence over that skill, including its board steps and work-item format.

- No Projects board is linked to this repo. State is open/closed plus comments, and tracker #1's checklist is the progress overview: a new `[MVP-NN]` issue gets a `- [ ] #N` line there, and the edited #1 body is read back. Do not attach issues to any board.
- Agent-discovered follow-ups or tech debt are proposals; create an issue only after the user approves. Do not assign anyone unless asked. Reuse existing labels only; ask before adding a new one.
- Issue titles and bodies are zh-TW and cold-start readable, following the existing format (#8 for `[MVP-NN]` work, #15 for small issues). Acceptance criteria must be independently verifiable and map to the check commands above.
- Re-read after every write: a 200 response only proves GitHub accepted the request. Close an issue only with a completion comment carrying validation evidence, and tick its #1 checklist entry when it has one.
- Before handing back or committing a nontrivial change, run the org-distributed `codex-review:codex-review` skill. It is review-only; never auto-apply its findings.

## Agent tooling

Skills, MCP servers (GitHub, AWS, Cloudflare) and the codex-review commit gate come from org-distributed Claude Code plugins, so the repo keeps no `.claude/skills`, `.claude/agents` or `.mcp.json` of its own. The GitHub MCP server authenticates with the member's own `GITHUB_PERSONAL_ACCESS_TOKEN` environment variable.