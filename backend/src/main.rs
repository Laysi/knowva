pub mod cloud_models;
mod document_summary;
mod embedding;
pub mod evaluation;
mod ingest;
mod page_artifacts;
mod page_vision;
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

#[derive(Clone)]
pub struct AppState {
    pool: PgPool,
    models: CloudModelClient,
    storage: storage::Storage,
    artifact_config: page_artifacts::ArtifactConfig,
    document_locks: ingest::DocumentLocks,
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
    tracing::info!(
        chat_model = model_config.chat_model(),
        embed_model = model_config.embed_model(),
        rerank_model = model_config.rerank_model(),
        chat_endpoint = %model_config.chat_endpoint(),
        embed_endpoint = %model_config.embed_endpoint(),
        rerank_endpoint = %model_config.rerank_endpoint(),
        request_timeout_secs = model_config.request_timeout().as_secs(),
        retrieval_mode = model_config.retrieval_mode().as_str(),
        rerank_enabled = model_config.rerank_enabled(),
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
    });

    // No CORS layer: the frontend is served from this process, so the browser
    // only ever makes same-origin requests -- in development too, where Vite
    // proxies /api rather than letting the page call the backend directly. A
    // permissive layer would just let any other site reach an API that has no
    // authentication in front of it.
    let app = api_router(state);

    // Compression wraps the static files only. /api/chat is an SSE stream, and
    // a compressor in front of it can hold token chunks back.
    let dist_dir = "../frontend/dist";
    let app = if Path::new(dist_dir).exists() {
        // Vite writes content-hashed filenames under assets/, so those never
        // change meaning and can be cached indefinitely. index.html names them
        // and must be revalidated, or a deploy would not be picked up.
        let assets = Router::new()
            .fallback_service(ServeDir::new(format!("{dist_dir}/assets")))
            .layer(SetResponseHeaderLayer::overriding(
                header::CACHE_CONTROL,
                HeaderValue::from_static("public, max-age=31536000, immutable"),
            ))
            .layer(CompressionLayer::new());
        let index_html = format!("{dist_dir}/index.html");
        let shell = Router::new()
            .fallback_service(ServeDir::new(dist_dir).not_found_service(ServeFile::new(index_html)))
            .layer(SetResponseHeaderLayer::overriding(
                header::CACHE_CONTROL,
                HeaderValue::from_static("no-cache"),
            ))
            .layer(CompressionLayer::new());
        app.nest_service("/assets", assets).fallback_service(shell)
    } else {
        tracing::warn!("{dist_dir} not found; serving the API without a frontend");
        app
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
        .route(
            "/api/documents",
            get(list_documents).post(ingest::upload_document),
        )
        .route("/api/documents/{id}", delete(ingest::delete_document))
        .route("/api/chat", post(chat))
        .layer(DefaultBodyLimit::max(ingest::MULTIPART_BODY_LIMIT_BYTES))
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
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

    let catalog_query = last_user_text.as_deref().is_some_and(is_catalog_query);
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
    let system_prompt = if catalog_query {
        rag_context::RAG_CATALOG_SYSTEM_PROMPT
    } else if context.sources.is_empty() {
        rag_context::RAG_NO_SOURCES_SYSTEM_PROMPT
    } else {
        rag_context::RAG_SYSTEM_PROMPT
    };
    let models = state.models.clone();
    let context = match if catalog_query {
        models.fit_catalog_context_to_request(system_prompt, context, &req.messages)
    } else {
        models.fit_rag_context_to_request(system_prompt, context, &req.messages)
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
            .stream_chat_with_context(system_prompt.to_string(), content, req.messages, inner_tx)
            .await;
    });
    let abort_handle = stream_task.abort_handle();
    let client_connected = forward_rag_events(&mut inner_rx, &tx, &sources, &abort_handle).await;
    drop(inner_rx);
    if client_connected {
        let _ = stream_task.await;
    }
}

fn is_catalog_query(query: &str) -> bool {
    let query = query.to_lowercase();
    let scope = ["知識庫", "資料庫", "專案", "全庫", "所有文件", "整個"];
    let intent = ["主題", "文件", "內容", "涵蓋", "有哪些", "什麼資料"];
    scope.iter().any(|term| query.contains(term)) && intent.iter().any(|term| query.contains(term))
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
        assert!(is_catalog_query("知識庫目前涵蓋哪些主題？"));
        assert!(is_catalog_query("專案有哪些文件？"));
        assert!(!is_catalog_query("維護報告涵蓋哪些主題？"));
        assert!(!is_catalog_query("感測器告警怎麼處理？"));
    }
}
