# CLAUDE.md

This file provides repository guidance for coding agents.

## Project

Demo of a general-purpose AI knowledge-management system. For the active MVP, all inference runs on an on-premise service reached over configured HTTP endpoints, and the model families are fixed: Gemma 4 for chat/multimodal reading, `llama-nemotron-embed-vl-1b-v2` for embeddings, and `llama-nemotron-rerank-vl-1b-v2` for reranking. Do not add in-process inference or provider-specific SDK paths; a different service is a URL change.

The specification priority is the active GitHub issue, then tracker issue #1, then `docs/mvp-now.md`. Keep implementations simple: no auth, no job queue, and chat history remains in frontend state.

## Development commands

- Install pinned toolchains: `proto install`
- Full stack: configure `.env`, then `docker compose up -d --build`
- Database only: `docker compose up -d postgres`
- Backend: `cd backend && proto exec rust -- cargo run` (run from `backend/` because data and frontend paths are working-directory relative)
- Frontend: `cd frontend && proto run npm -- run dev`
- Backend checks: run `proto exec rust -- cargo fmt --check`, `proto exec rust -- cargo clippy --all-targets -- -D warnings`, and `proto exec rust -- cargo test` from `backend/`
- Frontend checks: run `proto run npm -- ci`, `proto run npm -- run lint`, and `proto run npm -- run build` from `frontend/`
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
  - `search.rs` owns injectable dense/BM25 candidate stages, deterministic RRF, safe artifact-to-passage preparation, batched VL reranking, strict index/score mapping, and transient-only RRF fallback. Dense and lexical paths default to top-50 over ready-only partial HNSW/GIN indexes; BM25 corpus statistics are transactionally maintained by migration `0004_hybrid_retrieval.sql`. Rerank output defaults to six candidates and is configurable from three to eight.
- `frontend/` is Vite + React TypeScript + assistant-ui with plain CSS.
- SQLx migrations run automatically at backend startup.

## Cloud model safety

- Model traffic goes to a self-hosted inference service over the OpenAI-compatible chat/embeddings routes and the NVIDIA ranking route; there are no hosted-provider endpoint defaults. Endpoints are configuration, so no URL is hardcoded and no hosted gateway is required.
- `MODEL_API_TOKEN` is optional and sent as a standard `Authorization: Bearer` header; when it is empty no authorization header is sent at all.
- Parse model configuration before connecting to the database so missing settings/model IDs fail before traffic is accepted.
- Logs may include endpoint and model ID, never the API token or request auth headers.
- Model HTTP errors use a maximum 500-character sanitized snippet and expose HTTP status, timeout, and retryability through one error type.
- The Gemma model ID has no repository default. Embed and rerank settings may only vendor-qualify their locked family names.
- Do not send an embedding dimension-reduction parameter; the fixed Embed VL response is validated as 2048 dimensions.
- Embed query input is text-only. Passage input may be text, image-only, or same-page image+text. Images are sent as bounded PNG/JPEG data URLs, never as local artifact paths; logs/errors must redact data URLs and long base64 strings.
- Embedding retries are finite exponential backoff with jitter for 408, 429, 5xx, transport errors, and timeouts. Other 4xx responses fail immediately.

## Document conversion

- Pdfium is a runtime shared-library dependency for PDF/Office page rendering; `PDFIUM_LIBRARY_PATH` may point to the library or its directory.
- LibreOffice is only required for DOCX/PPTX/XLSX and is selected with `LIBREOFFICE_BIN`.
- The runtime image pins Pdfium Chromium 7763 by architecture and includes LibreOffice plus Noto CJK fonts; it runs as the non-root `km` user.
- The default page render setting is 180 DPI and conversion timeout is 120 seconds; see `.env.example`.
- The real PDF/Office fixture smoke test is explicitly ignored on machines without both native dependencies. It also verifies native PDF image+text and scanned PDF image-only requests against a mock endpoint. The `native-test` image target provides a reproducible opt-in run; its command and expected page counts are documented in `README.md` and `backend/tests/fixtures/README.md`.
- A separate ignored real-cloud Embed VL smoke covers all four input shapes and must only be run explicitly with configured credentials; normal tests and CI use deterministic mock endpoints.

## Working tree safety

Preserve unrelated user changes. Use focused patches, run the backend checks above, and do not push or create a PR unless explicitly requested.

## Project management (GitHub issues)

Keeping the tracking issue in sync is part of done: when issue-tracked work gets blocked, lands a PR/commit, or completes, update the issue in the same session. Operational details (MCP payloads, issue format, gh fallback) live in `.claude/skills/project-board/SKILL.md`, executed through `github-loop-engineering-skill`; this section only sets policy.

- No Projects board is linked to this repo. State is open/closed plus comments, and tracker #1's checklist is the progress overview. Do not attach issues to any board.
- Agent-discovered follow-ups or tech debt are proposals; create an issue only after the user approves. Do not assign anyone unless asked. Reuse existing labels only; ask before adding a new one.
- Issue titles and bodies are zh-TW and cold-start readable, following the existing format (#8 for `[MVP-NN]` work, #15 for small issues). Acceptance criteria must be independently verifiable and map to the check commands above.
- Re-read after every write: a 200 response only proves GitHub accepted the request. Close an issue only with a completion comment carrying validation evidence, and tick its #1 checklist entry when it has one.
- Before handing back or committing a nontrivial change, run the `codex-review` skill. It is review-only; never auto-apply its findings.

## Agent routing

Use `deep-reasoner` for architecture, unclear bugs, security, database/API/deployment decisions, design review, and high-risk tradeoffs. It is read-only and should return a decision memo plus implementation plan.
Use `senior-engineer` for difficult implementation that requires strong engineering judgment and direct code edits, especially Rust, backend logic, APIs, async/concurrency, database integration, state management, complex refactors, or bugs that are too hard for mechanical patching.
Use `fast-worker` for clear, scoped implementation: small code edits, refactors, file creation, tests, formatting, migrations, and applying an existing plan. Do not use it for architecture or hard engineering judgment.
Use `ui-rewriter` when the frontend looks ugly, cheap, inconsistent, template-like, or not credible as a B2B SaaS product. It may directly edit frontend files, but should not touch backend architecture.
For risky architecture: `deep-reasoner` first, then `senior-engineer`, `fast-worker`, or `ui-rewriter`.
For hard implementation: `senior-engineer`.
For obvious scoped bugs or mechanical edits: `fast-worker`.
For frontend visual polish/redesign: `ui-rewriter`.
Do not let multiple agents independently redesign the same area.
Do not ask `fast-worker` to invent architecture.
Do not ask `deep-reasoner` to edit files.
Do not use Opus agents for routine mechanical edits.