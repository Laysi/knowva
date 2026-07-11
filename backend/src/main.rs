pub mod cloud_models;
mod embedding;
mod ingest;
mod page_artifacts;
mod search;

#[cfg(test)]
mod database_tests;
#[cfg(test)]
mod test_fixtures;

use axum::extract::{DefaultBodyLimit, State};
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
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};

#[derive(Clone)]
pub struct AppState {
    pool: PgPool,
    models: CloudModelClient,
    files_dir: String,
    artifact_config: page_artifacts::ArtifactConfig,
    document_locks: ingest::DocumentLocks,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt::init();

    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://km:km@localhost:5432/km".to_string());
    let files_dir = std::env::var("FILES_DIR").unwrap_or_else(|_| "./data/files".to_string());
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let model_config = CloudModelConfig::from_env()?;
    let artifact_config = page_artifacts::ArtifactConfig::from_env()?;
    tracing::info!(
        chat_model = model_config.chat_model(),
        embed_model = model_config.embed_model(),
        rerank_model = model_config.rerank_model(),
        chat_endpoint = %model_config.chat_endpoint(),
        embed_endpoint = %model_config.embed_endpoint(),
        rerank_endpoint = %model_config.rerank_endpoint(),
        request_timeout_secs = model_config.request_timeout().as_secs(),
        "cloud model client configured"
    );
    tracing::info!(
        render_dpi = artifact_config.render_dpi,
        conversion_timeout_secs = artifact_config.conversion_timeout.as_secs(),
        libreoffice_binary = %artifact_config.libreoffice_binary.display(),
        "document artifact converter configured"
    );
    let models = CloudModelClient::new(model_config)?;

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    sqlx::migrate!().run(&pool).await?;

    let state = Arc::new(AppState {
        pool,
        models,
        files_dir,
        artifact_config,
        document_locks: ingest::DocumentLocks::default(),
    });

    let api_routes = api_router(state);

    let app = api_routes.layer(CorsLayer::permissive());

    let dist_dir = "../frontend/dist";
    let app = if Path::new(dist_dir).exists() {
        let index_html = format!("{dist_dir}/index.html");
        let serve_dir = ServeDir::new(dist_dir).not_found_service(ServeFile::new(index_html));
        app.fallback_service(serve_dir)
    } else {
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

#[derive(Debug, Serialize, sqlx::FromRow)]
struct DocumentRow {
    id: uuid::Uuid,
    filename: String,
    status: String,
    artifact_status: String,
    summary: Option<String>,
    tags: Option<Vec<String>>,
    error: Option<String>,
    created_at: chrono::DateTime<chrono::Utc>,
    page_count: i64,
    unit_count: i64,
    // Kept during the MVP API transition for clients that still display chunk_count.
    chunk_count: i64,
}

async fn list_documents(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<DocumentRow>>, (axum::http::StatusCode, String)> {
    let rows = sqlx::query_as::<_, DocumentRow>(
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
                d.error, d.created_at, \
                GREATEST(COALESCE(u.page_count, 0), COALESCE(a.page_count, 0)) AS page_count, \
                GREATEST(COALESCE(u.unit_count, 0), COALESCE(a.unit_count, 0)) AS unit_count, \
                GREATEST(COALESCE(u.unit_count, 0), COALESCE(a.unit_count, 0)) AS chunk_count \
         FROM documents d \
         LEFT JOIN unit_counts u ON u.document_id = d.id \
         LEFT JOIN artifact_counts a ON a.document_id = d.id \
         ORDER BY d.created_at DESC",
    )
    .fetch_all(&state.pool)
    .await
    .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(rows))
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
}

async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<SseEvent>(64);

    let models = state.models.clone();

    let last_user_text = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone());

    tokio::spawn(async move {
        let retrieval = match &last_user_text {
            Some(text) if !text.trim().is_empty() => search::retrieve(&state, text)
                .await
                .map_err(|e| tracing::warn!("retrieval failed, falling back to plain chat: {e}"))
                .ok(),
            _ => None,
        };

        match retrieval {
            Some(chunks) if !chunks.is_empty() => {
                let system_prompt = search::build_rag_system_prompt(&chunks);
                let sources = search::dedupe_sources(&chunks);

                let (inner_tx, mut inner_rx) = tokio::sync::mpsc::channel::<SseEvent>(64);
                let models2 = models.clone();
                let stream_task = tokio::spawn(async move {
                    models2
                        .stream_chat_with_system(system_prompt, req.messages, inner_tx)
                        .await
                });

                forward_rag_events(&mut inner_rx, &tx, &sources).await;
                // Dropping the receiver first is load-bearing: if we broke out early
                // (client disconnected), the pump blocks on a full channel forever and
                // stream_task.await would never resolve.
                drop(inner_rx);
                let _ = stream_task.await;
            }
            _ => {
                models.stream_chat(req.messages, tx).await;
            }
        }
    });

    let stream = ReceiverStream::new(rx).map(|event| {
        let data = serde_json::to_string(&event).unwrap_or_else(|_| {
            r#"{"type":"error","message":"serialization failure"}"#.to_string()
        });
        Ok(Event::default().data(data))
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn forward_rag_events(
    inner_rx: &mut tokio::sync::mpsc::Receiver<SseEvent>,
    tx: &tokio::sync::mpsc::Sender<SseEvent>,
    sources: &[search::SourceRef],
) {
    while let Some(event) = inner_rx.recv().await {
        match event {
            SseEvent::Done => {
                if tx
                    .send(SseEvent::Sources {
                        sources: sources.to_vec(),
                    })
                    .await
                    .is_err()
                {
                    break;
                }
                let _ = tx.send(SseEvent::Done).await;
                break;
            }
            other => {
                if tx.send(other).await.is_err() {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rag_sse_keeps_text_sources_done_order() {
        let (inner_tx, mut inner_rx) = tokio::sync::mpsc::channel(4);
        let (outer_tx, mut outer_rx) = tokio::sync::mpsc::channel(4);
        let document_id = uuid::Uuid::new_v4();
        let sources = vec![search::SourceRef {
            document_id,
            filename: "source.pdf".to_string(),
            page_number: Some(1),
        }];

        inner_tx
            .send(SseEvent::TextDelta {
                text: "answer".to_string(),
            })
            .await
            .expect("send text");
        inner_tx.send(SseEvent::Done).await.expect("send done");
        drop(inner_tx);

        forward_rag_events(&mut inner_rx, &outer_tx, &sources).await;
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

        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            forward_rag_events(&mut inner_rx, &outer_tx, &[]),
        )
        .await
        .expect("disconnect must not block the event pump");
        drop(inner_rx);
        assert!(inner_tx.send(SseEvent::Done).await.is_err());
    }
}
