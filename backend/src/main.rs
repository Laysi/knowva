pub mod cloud_models;
mod document_summary;
mod embedding;
pub mod evaluation;
mod ingest;
mod page_artifacts;
mod page_vision;
mod profile;
mod rag_context;
mod search;
mod storage;

#[cfg(test)]
mod database_tests;
#[cfg(test)]
mod test_fixtures;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderValue, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use cloud_models::{ChatMessage, CloudModelClient, CloudModelConfig, SseEvent};
use futures::StreamExt;
use profile::{Profile, PromptKind};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::compression::CompressionLayer;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::set_header::SetResponseHeaderLayer;

pub(crate) const DEFAULT_DATABASE_URL: &str = "postgres://knowva:knowva@localhost:5432/knowva";
pub(crate) const DEFAULT_FILES_DIR: &str = "./data/files";
/// Holds one build per theme, `<theme>/index.html` and `<theme>/assets/`.
const FRONTEND_DIST_DIR: &str = "../frontend/dist";

#[derive(Clone)]
pub struct AppState {
    pool: PgPool,
    models: CloudModelClient,
    storage: storage::Storage,
    artifact_config: page_artifacts::ArtifactConfig,
    document_locks: ingest::DocumentLocks,
    profile: Arc<Profile>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("eval") {
        return evaluation::run_cli(std::env::args().skip(2)).await;
    }

    dotenvy::dotenv().ok();

    tracing_subscriber::fmt::init();

    let database_url =
        std::env::var("DATABASE_URL").unwrap_or_else(|_| DEFAULT_DATABASE_URL.to_string());
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let model_config = CloudModelConfig::from_env()?;
    let artifact_config = page_artifacts::ArtifactConfig::from_env()?;
    let storage_config = storage::StorageConfig::from_env()?;
    // The profile and its theme build are checked here, before the database, so a missing
    // or malformed profile never accepts traffic.
    let profile = Arc::new(Profile::from_env()?);
    let frontend_dir = profile.theme_dist_dir(Path::new(FRONTEND_DIST_DIR))?;
    tracing::info!(
        profile = profile.id(),
        theme = profile.theme(),
        industry_section_chars = profile.industry_section_chars(),
        answer_prompt_sha256 = %profile.system_prompt_sha256(PromptKind::Answer),
        "industry profile loaded"
    );
    tracing::info!(
        chat_model = model_config.chat_model(),
        embed_model = model_config.embed_model(),
        rerank_model = model_config.rerank_model(),
        chat_endpoint = %model_config.chat_endpoint(),
        embed_endpoint = %model_config.embed_endpoint(),
        rerank_endpoint = %model_config.rerank_endpoint(),
        request_timeout_secs = model_config.request_timeout().as_secs(),
        rerank_candidate_top_k = model_config.rerank_candidate_top_k(),
        rerank_top_n = model_config.rerank_top_n(),
        rerank_batch_size = model_config.rerank_batch_size(),
        "cloud model client configured"
    );
    tracing::info!(
        render_dpi = artifact_config.render_dpi,
        conversion_timeout_secs = artifact_config.conversion_timeout.as_secs(),
        libreoffice_binary = %artifact_config.libreoffice_binary.display(),
        "document artifact converter configured"
    );
    let models = CloudModelClient::new(model_config)?;
    let storage = storage_config.open().await?;

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    sqlx::migrate!().run(&pool).await?;
    storage::ensure_initialization_lock(&pool, &storage_config).await?;

    let state = Arc::new(AppState {
        pool,
        models,
        storage,
        artifact_config,
        document_locks: ingest::DocumentLocks::default(),
        profile,
    });

    // No CORS layer: the frontend is served from this process, so the browser
    // only ever makes same-origin requests -- in development too, where Vite
    // proxies /api rather than letting the page call the backend directly. A
    // permissive layer would just let any other site reach an API that has no
    // authentication in front of it.
    let app = api_router(state);

    let app = match frontend_dir {
        Some(theme_dir) => {
            tracing::info!(frontend = %theme_dir.display(), "serving the theme build");
            with_frontend(app, &theme_dir)
        }
        None => {
            tracing::warn!("{FRONTEND_DIST_DIR} not found; serving the API without a frontend");
            app
        }
    };

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn api_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/profile", get(profile_ui))
        .route(
            "/api/documents",
            get(list_documents).post(ingest::upload_document),
        )
        .route("/api/documents/{id}", delete(ingest::delete_document))
        .route("/api/chat", post(chat))
        .layer(DefaultBodyLimit::max(ingest::MULTIPART_BODY_LIMIT_BYTES))
        .with_state(state)
}

/// Serves one theme's build behind the API: `assets/` and, for every other path, the
/// single-page shell.
fn with_frontend(app: Router, theme_dir: &Path) -> Router {
    // Compression wraps the static files only. /api/chat is an SSE stream, and
    // a compressor in front of it can hold token chunks back.
    //
    // Vite writes content-hashed filenames under assets/, so those never
    // change meaning and can be cached indefinitely. index.html names them
    // and must be revalidated, or a deploy would not be picked up.
    let assets = Router::new()
        .fallback_service(ServeDir::new(theme_dir.join("assets")))
        .layer(SetResponseHeaderLayer::overriding(
            header::CACHE_CONTROL,
            HeaderValue::from_static("public, max-age=31536000, immutable"),
        ))
        .layer(CompressionLayer::new());
    let shell = Router::new()
        .fallback_service(
            ServeDir::new(theme_dir)
                .not_found_service(ServeFile::new(theme_dir.join("index.html"))),
        )
        .layer(SetResponseHeaderLayer::overriding(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        ))
        .layer(CompressionLayer::new());
    app.nest_service("/assets", assets).fallback_service(shell)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

/// Interface defaults of the deployed profile. Prompt text and catalog keywords stay on the
/// server: `ProfileUi` has no field for either.
async fn profile_ui(State(state): State<Arc<AppState>>) -> Json<profile::ProfileUi> {
    Json(state.profile.ui().clone())
}

#[derive(Debug, sqlx::FromRow)]
struct DocumentDbRow {
    id: uuid::Uuid,
    filename: String,
    status: String,
    artifact_status: String,
    summary: Option<String>,
    tags: Option<Vec<String>>,
    created_at: chrono::DateTime<chrono::Utc>,
    page_count: i64,
    unit_count: i64,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
struct DocumentProcessingError {
    code: &'static str,
    message: &'static str,
}

#[derive(Debug, Serialize)]
struct DocumentResponse {
    id: uuid::Uuid,
    filename: String,
    status: String,
    #[serde(rename = "processingStage")]
    processing_stage: &'static str,
    #[serde(rename = "processingError")]
    processing_error: Option<DocumentProcessingError>,
    summary: Option<String>,
    tags: Option<Vec<String>>,
    #[serde(rename = "createdAt")]
    canonical_created_at: chrono::DateTime<chrono::Utc>,
    #[serde(rename = "pageCount")]
    canonical_page_count: i64,
    #[serde(rename = "unitCount")]
    canonical_unit_count: i64,
    // Deprecated compatibility aliases retained for existing MVP clients.
    artifact_status: String,
    page_count: i64,
    unit_count: i64,
    chunk_count: i64,
    created_at: chrono::DateTime<chrono::Utc>,
    // This compatibility field is a fixed safe message, never the raw DB diagnostic.
    error: Option<&'static str>,
}

impl From<DocumentDbRow> for DocumentResponse {
    fn from(row: DocumentDbRow) -> Self {
        let processing_stage = document_processing_stage(&row.status, &row.artifact_status);
        let processing_error = document_processing_error(&row.status, &row.artifact_status);
        Self {
            id: row.id,
            filename: row.filename,
            status: row.status,
            processing_stage,
            processing_error,
            summary: row.summary,
            tags: row.tags,
            canonical_created_at: row.created_at,
            canonical_page_count: row.page_count,
            canonical_unit_count: row.unit_count,
            artifact_status: row.artifact_status,
            page_count: row.page_count,
            unit_count: row.unit_count,
            chunk_count: row.unit_count,
            created_at: row.created_at,
            error: processing_error.map(|error| error.message),
        }
    }
}

fn document_processing_stage(status: &str, artifact_status: &str) -> &'static str {
    match (status, artifact_status) {
        ("ready", _) => "ready",
        ("failed", _) => "failed",
        ("processing", "pending") => "queued",
        ("processing", "ready") => "indexing",
        ("processing", _) => "extracting",
        _ => "failed",
    }
}

fn document_processing_error(
    status: &str,
    artifact_status: &str,
) -> Option<DocumentProcessingError> {
    if status != "failed" {
        return None;
    }
    Some(match artifact_status {
        "failed" => DocumentProcessingError {
            code: "artifact_processing_failed",
            message: "文件解析失敗，請確認檔案格式後重試。",
        },
        "ready" => DocumentProcessingError {
            code: "indexing_failed",
            message: "文件索引建立失敗，請稍後重試。",
        },
        _ => DocumentProcessingError {
            code: "document_processing_failed",
            message: "文件處理失敗，請稍後重試。",
        },
    })
}

async fn list_documents(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<DocumentResponse>>, (axum::http::StatusCode, String)> {
    let rows = sqlx::query_as::<_, DocumentDbRow>(
        "WITH unit_counts AS ( \
           SELECT document_id, COUNT(*) AS unit_count, \
                  COUNT(*) FILTER (WHERE kind = 'page') AS page_count \
           FROM document_units GROUP BY document_id \
         ), artifact_counts AS ( \
           SELECT d.id AS document_id, COUNT(DISTINCT p.id) AS page_count, \
                  COUNT(DISTINCT p.id) + COUNT(DISTINCT t.id) AS unit_count \
           FROM documents d \
           LEFT JOIN page_artifacts p ON p.document_id = d.id \
           LEFT JOIN text_artifacts t ON t.document_id = d.id \
           GROUP BY d.id \
         ) \
         SELECT d.id, d.filename, d.status, d.artifact_status, d.summary, d.tags, \
                d.created_at, \
                GREATEST(COALESCE(u.page_count, 0), COALESCE(a.page_count, 0)) AS page_count, \
                GREATEST(COALESCE(u.unit_count, 0), COALESCE(a.unit_count, 0)) AS unit_count \
         FROM documents d \
         LEFT JOIN unit_counts u ON u.document_id = d.id \
         LEFT JOIN artifact_counts a ON a.document_id = d.id \
         ORDER BY d.created_at DESC",
    )
    .fetch_all(&state.pool)
    .await
    .map_err(|error| {
        tracing::error!(
            stage = "document_list",
            "document list query failed: {error}"
        );
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "無法載入文件列表，請稍後再試。".to_string(),
        )
    })?;

    Ok(Json(rows.into_iter().map(DocumentResponse::from).collect()))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    /// Sources the previous answer cited, echoed back by the client in the order they were
    /// laid out. `/api/chat` is stateless and the history carries text only, so this is the
    /// only way the next context can keep those blocks where the prompt cache expects them.
    /// Unknown fields are ignored, so a client may echo the `sources` event unchanged.
    #[serde(default)]
    previous_sources: Vec<PreviousSource>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PreviousSource {
    unit_id: i64,
}

async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<SseEvent>(64);
    tokio::spawn(run_chat(state, req, tx));

    let stream = ReceiverStream::new(rx).map(|event| {
        let data = serde_json::to_string(&event).unwrap_or_else(|_| {
            r#"{"type":"error","message":"serialization failure"}"#.to_string()
        });
        Ok(Event::default().data(data))
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn run_chat(state: Arc<AppState>, req: ChatRequest, tx: tokio::sync::mpsc::Sender<SseEvent>) {
    // A retrieval context never lays out more than this many blocks, so anything past the cap
    // could not name a block of the previous context even if a client sent it.
    let previous_units = req
        .previous_sources
        .iter()
        .take(rag_context::MAX_RAG_SOURCES)
        .map(|source| source.unit_id)
        .collect::<Vec<_>>();
    let last_user_text = req
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| message.content.clone());

    let catalog_query = last_user_text
        .as_deref()
        .is_some_and(|text| is_catalog_query(text, state.profile.catalog_keywords()));
    let candidates = match last_user_text {
        Some(text) if !text.trim().is_empty() => {
            let retrieval = if catalog_query {
                search::catalog_candidates(&state.pool)
                    .await
                    .map_err(|error| {
                        search::RetrieveError::Candidates(search::CandidateRetrievalError::Bm25(
                            error,
                        ))
                    })
            } else {
                tokio::select! {
                    biased;
                    _ = tx.closed() => return,
                    result = search::retrieve(&state, &text) => result,
                }
            };
            match retrieval {
                Ok(candidates) => candidates,
                Err(error) => {
                    let error_kind = match &error {
                        search::RetrieveError::Candidates(_) => "candidate_retrieval",
                        search::RetrieveError::Rerank(_) => "rerank",
                    };
                    tracing::error!(
                        stage = "retrieval",
                        error_kind,
                        "knowledge-base retrieval failed"
                    );
                    let _ = tx
                        .send(SseEvent::Error {
                            message: error.client_message(),
                        })
                        .await;
                    return;
                }
            }
        }
        _ => Vec::new(),
    };

    // Complete artifact validation, image conversion, and payload trimming before the
    // provider request is opened. Structured metadata remains separate from model content.
    let context = if catalog_query {
        rag_context::build_catalog_context(&candidates)
    } else {
        tokio::select! {
            biased;
            _ = tx.closed() => return,
            result = rag_context::build_rag_context(
                &candidates,
                &state.storage,
                &previous_units,
            ) => result,
        }
    };
    let context = match context {
        Ok(context) => context,
        Err(error) => {
            tracing::error!(
                stage = "rag_context",
                error_kind = error.safe_kind(),
                "RAG context preflight failed"
            );
            let _ = tx
                .send(SseEvent::Error {
                    message: error.client_message(),
                })
                .await;
            return;
        }
    };
    let prompt_kind = if catalog_query {
        PromptKind::Catalog
    } else if context.sources.is_empty() {
        PromptKind::NoSources
    } else {
        PromptKind::Answer
    };
    let system_prompt = state.profile.system_prompt(prompt_kind).to_owned();
    let models = state.models.clone();
    let context = match if catalog_query {
        models.fit_catalog_context_to_request(&system_prompt, context, &req.messages)
    } else {
        models.fit_rag_context_to_request(&system_prompt, context, &req.messages)
    } {
        Ok(context) => context,
        Err(error) => {
            tracing::error!(
                stage = "rag_request_preflight",
                error_kind = error.safe_kind(),
                "Gemma 4 request payload preflight failed"
            );
            let _ = tx
                .send(SseEvent::Error {
                    message: error.chat_client_message(),
                })
                .await;
            return;
        }
    };
    tracing::info!(
        stage = "rag_context",
        source_count = context.sources.len(),
        image_count = context.image_count(),
        catalog = catalog_query,
        "RAG context prepared"
    );
    let (content, sources) = context.into_parts();

    let (inner_tx, mut inner_rx) = tokio::sync::mpsc::channel::<SseEvent>(64);
    let stream_task = tokio::spawn(async move {
        models
            .stream_chat_with_context(system_prompt, content, req.messages, inner_tx)
            .await;
    });
    let abort_handle = stream_task.abort_handle();
    let client_connected = forward_rag_events(&mut inner_rx, &tx, &sources, &abort_handle).await;
    drop(inner_rx);
    if client_connected {
        let _ = stream_task.await;
    }
}

/// A whole-knowledge-base question names a scope term and an intent term from the profile's
/// catalog keywords, which the profile stores lowercased.
fn is_catalog_query(query: &str, keywords: &profile::CatalogKeywords) -> bool {
    let query = query.to_lowercase();
    keywords
        .scope()
        .iter()
        .any(|term| query.contains(term.as_str()))
        && keywords
            .intent()
            .iter()
            .any(|term| query.contains(term.as_str()))
}

async fn forward_rag_events(
    inner_rx: &mut tokio::sync::mpsc::Receiver<SseEvent>,
    tx: &tokio::sync::mpsc::Sender<SseEvent>,
    sources: &[rag_context::SourceRef],
    upstream_abort: &tokio::task::AbortHandle,
) -> bool {
    loop {
        let event = tokio::select! {
            biased;
            _ = tx.closed() => {
                upstream_abort.abort();
                return false;
            }
            event = inner_rx.recv() => event,
        };
        match event {
            Some(SseEvent::TextDelta { text }) => {
                if tx.send(SseEvent::TextDelta { text }).await.is_err() {
                    upstream_abort.abort();
                    return false;
                }
            }
            Some(SseEvent::Done) => {
                if tx
                    .send(SseEvent::Sources {
                        sources: sources.to_vec(),
                    })
                    .await
                    .is_err()
                {
                    upstream_abort.abort();
                    return false;
                }
                if tx.send(SseEvent::Done).await.is_err() {
                    upstream_abort.abort();
                    return false;
                }
                return true;
            }
            Some(error @ SseEvent::Error { .. }) => {
                if tx.send(error).await.is_err() {
                    upstream_abort.abort();
                    return false;
                }
                return true;
            }
            Some(SseEvent::Sources { .. }) => {
                upstream_abort.abort();
                let _ = tx
                    .send(SseEvent::Error {
                        message: "雲端回答串流狀態無效（錯誤代碼：chat_stream_protocol）"
                            .to_string(),
                    })
                    .await;
                return true;
            }
            None => {
                upstream_abort.abort();
                let _ = tx
                    .send(SseEvent::Error {
                        message: "雲端回答串流意外中斷（錯誤代碼：chat_stream_closed）".to_string(),
                    })
                    .await;
                return true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::test_support::{TempDir, valid_profile_json, write_profile};

    fn pending_upstream() -> tokio::task::JoinHandle<()> {
        tokio::spawn(std::future::pending())
    }

    #[test]
    fn chat_request_reads_echoed_sources_and_stays_valid_without_them() {
        // A client echoes the `sources` event unchanged, so the request must accept the
        // whole source object and read only the unit it names.
        let echoed: ChatRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"追問"}],
                "previousSources":[
                  {"sourceId":"S1","documentId":"00000000-0000-0000-0000-000000000001",
                   "unitId":103,"unitIndex":2,"filename":"manual.pdf","pageNumber":3,
                   "kind":"page"},
                  {"sourceId":"S2","documentId":"00000000-0000-0000-0000-000000000001",
                   "unitId":106,"unitIndex":5,"filename":"manual.pdf","pageNumber":6,
                   "kind":"page"}]}"#,
        )
        .expect("echoed sources are accepted");
        assert_eq!(
            echoed
                .previous_sources
                .iter()
                .map(|source| source.unit_id)
                .collect::<Vec<_>>(),
            vec![103, 106]
        );

        let first_question: ChatRequest =
            serde_json::from_str(r#"{"messages":[{"role":"user","content":"第一個問題"}]}"#)
                .expect("the field stays optional");
        assert!(first_question.previous_sources.is_empty());
    }

    #[test]
    fn document_api_maps_stages_and_only_serializes_safe_errors() {
        for (status, artifact_status, expected) in [
            ("processing", "pending", "queued"),
            ("processing", "processing", "extracting"),
            ("processing", "ready", "indexing"),
            ("ready", "ready", "ready"),
            ("failed", "failed", "failed"),
        ] {
            assert_eq!(document_processing_stage(status, artifact_status), expected);
        }

        let created_at = chrono::Utc::now();
        let response = DocumentResponse::from(DocumentDbRow {
            id: uuid::Uuid::new_v4(),
            filename: "fixture.pdf".to_string(),
            status: "failed".to_string(),
            artifact_status: "ready".to_string(),
            summary: None,
            tags: None,
            created_at,
            page_count: 2,
            unit_count: 3,
        });
        let json = serde_json::to_value(response).expect("serialize document response");
        assert_eq!(json["processingStage"], "failed");
        assert_eq!(json["processingError"]["code"], "indexing_failed");
        assert_eq!(json["error"], json["processingError"]["message"]);
        assert_eq!(json["pageCount"], 2);
        assert_eq!(json["unitCount"], 3);
        assert_eq!(json["page_count"], json["pageCount"]);
        assert_eq!(json["unit_count"], json["unitCount"]);
        assert_eq!(json["chunk_count"], json["unitCount"]);
        assert_eq!(json["created_at"], json["createdAt"]);
        assert!(!json.to_string().contains("provider-secret"));
    }

    #[tokio::test]
    async fn rag_sse_keeps_text_sources_done_order() {
        let (inner_tx, mut inner_rx) = tokio::sync::mpsc::channel(4);
        let (outer_tx, mut outer_rx) = tokio::sync::mpsc::channel(4);
        let document_id = uuid::Uuid::new_v4();
        let sources = vec![rag_context::SourceRef {
            source_id: "S1".to_string(),
            document_id,
            unit_id: 17,
            unit_index: 0,
            filename: "source.pdf".to_string(),
            page_number: Some(1),
            kind: "page".to_string(),
        }];
        let upstream = pending_upstream();
        let abort_handle = upstream.abort_handle();

        inner_tx
            .send(SseEvent::TextDelta {
                text: "answer".to_string(),
            })
            .await
            .expect("send text");
        inner_tx.send(SseEvent::Done).await.expect("send done");
        drop(inner_tx);

        assert!(forward_rag_events(&mut inner_rx, &outer_tx, &sources, &abort_handle).await);
        upstream.abort();
        drop(outer_tx);

        assert!(matches!(
            outer_rx.recv().await,
            Some(SseEvent::TextDelta { text }) if text == "answer"
        ));
        assert!(matches!(
            outer_rx.recv().await,
            Some(SseEvent::Sources { sources })
                if sources.len() == 1 && sources[0].document_id == document_id
        ));
        assert!(matches!(outer_rx.recv().await, Some(SseEvent::Done)));
    }

    #[tokio::test]
    async fn rag_sse_zero_text_still_emits_one_sources_then_done() {
        let (inner_tx, mut inner_rx) = tokio::sync::mpsc::channel(2);
        let (outer_tx, mut outer_rx) = tokio::sync::mpsc::channel(2);
        let upstream = pending_upstream();
        let abort_handle = upstream.abort_handle();
        inner_tx.send(SseEvent::Done).await.expect("send done");
        drop(inner_tx);

        assert!(forward_rag_events(&mut inner_rx, &outer_tx, &[], &abort_handle).await);
        upstream.abort();
        drop(outer_tx);
        assert!(matches!(
            outer_rx.recv().await,
            Some(SseEvent::Sources { sources }) if sources.is_empty()
        ));
        assert!(matches!(outer_rx.recv().await, Some(SseEvent::Done)));
        assert!(outer_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn rag_sse_stops_when_client_disconnects() {
        let (inner_tx, mut inner_rx) = tokio::sync::mpsc::channel(1);
        let (outer_tx, outer_rx) = tokio::sync::mpsc::channel(1);
        drop(outer_rx);
        inner_tx
            .send(SseEvent::TextDelta {
                text: "answer".to_string(),
            })
            .await
            .expect("send text");
        let upstream = pending_upstream();
        let abort_handle = upstream.abort_handle();

        let connected = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            forward_rag_events(&mut inner_rx, &outer_tx, &[], &abort_handle),
        )
        .await
        .expect("disconnect must not block the event pump");
        assert!(!connected);
        assert!(
            upstream
                .await
                .expect_err("upstream must be aborted")
                .is_cancelled()
        );
        drop(inner_rx);
        assert!(inner_tx.send(SseEvent::Done).await.is_err());
    }

    #[tokio::test]
    async fn rag_sse_error_after_text_has_no_fake_sources_or_done() {
        let (inner_tx, mut inner_rx) = tokio::sync::mpsc::channel(4);
        let (outer_tx, mut outer_rx) = tokio::sync::mpsc::channel(4);
        let upstream = pending_upstream();
        let abort_handle = upstream.abort_handle();
        inner_tx
            .send(SseEvent::TextDelta {
                text: "partial".to_string(),
            })
            .await
            .expect("send text");
        inner_tx
            .send(SseEvent::Error {
                message: "安全錯誤".to_string(),
            })
            .await
            .expect("send error");
        inner_tx
            .send(SseEvent::Done)
            .await
            .expect("queue fake done");

        assert!(forward_rag_events(&mut inner_rx, &outer_tx, &[], &abort_handle).await);
        upstream.abort();
        drop(outer_tx);
        assert!(matches!(
            outer_rx.recv().await,
            Some(SseEvent::TextDelta { text }) if text == "partial"
        ));
        assert!(matches!(
            outer_rx.recv().await,
            Some(SseEvent::Error { message }) if message == "安全錯誤"
        ));
        assert!(outer_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn unexpected_inner_close_emits_one_safe_error_only() {
        let (inner_tx, mut inner_rx) = tokio::sync::mpsc::channel(1);
        let (outer_tx, mut outer_rx) = tokio::sync::mpsc::channel(2);
        let upstream = pending_upstream();
        let abort_handle = upstream.abort_handle();
        drop(inner_tx);

        assert!(forward_rag_events(&mut inner_rx, &outer_tx, &[], &abort_handle).await);
        assert!(
            upstream
                .await
                .expect_err("upstream must be aborted")
                .is_cancelled()
        );
        drop(outer_tx);
        assert!(matches!(
            outer_rx.recv().await,
            Some(SseEvent::Error { message }) if message.contains("chat_stream_closed")
        ));
        assert!(outer_rx.recv().await.is_none());
    }

    #[test]
    fn catalog_query_detection_is_limited_to_whole_kb_intent() {
        let profile = Profile::for_test();
        let keywords = profile.catalog_keywords();
        assert!(is_catalog_query("知識庫目前涵蓋哪些主題？", keywords));
        assert!(is_catalog_query("專案有哪些文件？", keywords));
        assert!(!is_catalog_query("維護報告涵蓋哪些主題？", keywords));
        assert!(!is_catalog_query("感測器告警怎麼處理？", keywords));
    }

    #[test]
    fn every_repository_profile_marks_exactly_its_catalog_suggestions() {
        let profiles_dir = profile::repository_profiles_dir();
        let mut visited = Vec::new();
        for entry in std::fs::read_dir(&profiles_dir).expect("read the repository profiles") {
            let entry = entry.expect("read a profiles entry");
            if !entry.file_type().expect("profiles entry type").is_dir() {
                continue;
            }
            let id = entry.file_name().into_string().expect("UTF-8 profile id");
            let profile = Profile::load(&profiles_dir, &id)
                .unwrap_or_else(|error| panic!("profile {id:?} must load: {error:#}"));
            for suggestion in profile.suggestions() {
                // A suggestion marked catalog must reach the overview path; an unmarked one
                // must not, or the marking would misdescribe what the question does.
                assert_eq!(
                    is_catalog_query(&suggestion.text, profile.catalog_keywords()),
                    suggestion.catalog,
                    "profile {id:?}: suggestion {:?} is marked catalog={}",
                    suggestion.text,
                    suggestion.catalog
                );
            }
            visited.push(id);
        }
        assert!(
            visited.iter().any(|id| id == profile::DEFAULT_PROFILE_ID),
            "profiles/default must exist: {visited:?}"
        );
    }

    fn offline_state(profile: Arc<Profile>) -> Arc<AppState> {
        Arc::new(AppState {
            // Nothing these tests request reaches the database; this pool never connects.
            pool: PgPoolOptions::new()
                .connect_lazy("postgres://unused@127.0.0.1:9/unused")
                .expect("lazy pool"),
            models: CloudModelClient::for_test(),
            storage: storage::Storage::memory(),
            artifact_config: page_artifacts::ArtifactConfig::for_test(),
            document_locks: ingest::DocumentLocks::default(),
            profile,
        })
    }

    async fn get(app: &Router, uri: &str) -> (axum::http::StatusCode, Option<String>, String) {
        use tower::ServiceExt;

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::get(uri)
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("route the request");
        let status = response.status();
        let cache_control = response
            .headers()
            .get(header::CACHE_CONTROL)
            .map(|value| value.to_str().expect("ASCII header").to_owned());
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read the body");
        (
            status,
            cache_control,
            String::from_utf8(body.to_vec()).expect("UTF-8 body"),
        )
    }

    #[tokio::test]
    async fn profile_route_serves_the_default_interface_fields() {
        let app = api_router(offline_state(Profile::for_test()));
        let (status, _, body) = get(&app, "/api/profile").await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let json: serde_json::Value = serde_json::from_str(&body).expect("JSON body");
        assert_eq!(
            json,
            serde_json::json!({
                "id": "default",
                "theme": "default",
                "brand": { "name": "Knowva 知星", "tagline": "點亮你的知識星系" },
                "pages": {
                    "chat": { "title": "知識問答" },
                    "documents": { "title": "文件管理" }
                },
                "suggestions": [
                    { "text": "知識庫裡有哪些主題？", "catalog": true },
                    { "text": "整理知識庫的重點", "catalog": false },
                    { "text": "新人該先看哪幾份文件？", "catalog": false }
                ],
                "composerPlaceholder": "問問你的文件…"
            })
        );
    }

    #[tokio::test]
    async fn profile_route_never_returns_prompt_text_or_catalog_keywords() {
        let root = TempDir::new("route");
        let industry = "產業段標記：你是精密機械廠的維修顧問。";
        let mut json = valid_profile_json("default");
        json["catalogKeywords"] = serde_json::json!({
            "scope": ["範圍標記詞甲", "範圍標記詞乙"],
            "intent": ["意圖標記詞丙"]
        });
        write_profile(&root.0, "machining", &json, industry);
        let profile = Arc::new(Profile::load(&root.0, "machining").expect("load profile"));
        assert!(
            profile
                .system_prompt(PromptKind::Answer)
                .ends_with(industry)
        );

        let (status, _, body) = get(&api_router(offline_state(profile)), "/api/profile").await;
        assert_eq!(status, axum::http::StatusCode::OK);
        for secret in [
            industry,
            "產業段標記",
            rag_context::INDUSTRY_SECTION_HEADING,
            rag_context::RAG_SYSTEM_PROMPT,
            "範圍標記詞甲",
            "範圍標記詞乙",
            "意圖標記詞丙",
            "catalogKeywords",
            "prompt",
        ] {
            assert!(!body.contains(secret), "response leaks {secret:?}: {body}");
        }
        let json: serde_json::Value = serde_json::from_str(&body).expect("JSON body");
        let keys = json
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        assert_eq!(
            keys,
            [
                "brand",
                "composerPlaceholder",
                "id",
                "pages",
                "suggestions",
                "theme"
            ]
        );
    }

    #[tokio::test]
    async fn frontend_follows_the_theme_of_the_selected_profile() {
        let root = TempDir::new("themes");
        let profiles_dir = root.0.join("profiles");
        let dist = root.0.join("dist");
        for theme in ["alpha", "beta"] {
            write_profile(
                &profiles_dir,
                &format!("{theme}-industry"),
                &valid_profile_json(theme),
                "",
            );
            std::fs::create_dir_all(dist.join(theme).join("assets")).expect("create theme");
            std::fs::write(
                dist.join(theme).join("index.html"),
                format!("<!doctype html><title>{theme}</title>"),
            )
            .expect("write index.html");
            std::fs::write(
                dist.join(theme).join("assets/index-c0ffee.js"),
                format!("console.log({theme:?});"),
            )
            .expect("write asset");
        }

        for theme in ["alpha", "beta"] {
            let profile = Profile::from_lookup(|name| match name {
                "KNOWVA_PROFILE" => Some(format!("{theme}-industry")),
                "PROFILES_DIR" => Some(profiles_dir.display().to_string()),
                _ => None,
            })
            .expect("load profile");
            let theme_dir = profile
                .theme_dist_dir(&dist)
                .expect("the theme is built")
                .expect("dist exists");
            let app = with_frontend(api_router(offline_state(Arc::new(profile))), &theme_dir);
            let index = format!("<!doctype html><title>{theme}</title>");

            for path in ["/", "/index.html"] {
                let (status, cache_control, body) = get(&app, path).await;
                assert_eq!(status, axum::http::StatusCode::OK, "{theme} {path}");
                assert_eq!(body, index, "{theme} {path}");
                assert_eq!(cache_control.as_deref(), Some("no-cache"), "{theme} {path}");
            }
            // Any other path still gets this theme's shell (with ServeDir's not-found status,
            // as before themes existed).
            let (_, cache_control, body) = get(&app, "/documents").await;
            assert_eq!(body, index, "{theme} shell fallback");
            assert_eq!(cache_control.as_deref(), Some("no-cache"));

            let (status, cache_control, body) = get(&app, "/assets/index-c0ffee.js").await;
            assert_eq!(status, axum::http::StatusCode::OK, "{theme} asset");
            assert_eq!(body, format!("console.log({theme:?});"));
            assert_eq!(
                cache_control.as_deref(),
                Some("public, max-age=31536000, immutable")
            );
            let (status, _, _) = get(&app, "/assets/missing.js").await;
            assert_eq!(status, axum::http::StatusCode::NOT_FOUND);

            let (_, _, body) = get(&app, "/api/profile").await;
            let json: serde_json::Value = serde_json::from_str(&body).expect("JSON body");
            assert_eq!(json["theme"], theme);
        }
    }
}
