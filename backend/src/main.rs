mod bedrock;
mod ingest;
mod llm;
mod nvidia;
mod search;

use axum::extract::{DefaultBodyLimit, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures::StreamExt;
use llm::{ChatMessage, Llm, SseEvent};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
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
    llm: Llm,
    files_dir: String,
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

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    sqlx::migrate!().run(&pool).await?;

    let provider = std::env::var("MODEL_PROVIDER").unwrap_or_else(|_| "bedrock".to_string());
    let llm = match provider.as_str() {
        "bedrock" => {
            let chat_model = std::env::var("BEDROCK_CHAT_MODEL_ID")
                .unwrap_or_else(|_| "us.amazon.nova-2-lite-v1:0".to_string());
            let embed_model = std::env::var("BEDROCK_EMBED_MODEL_ID")
                .unwrap_or_else(|_| "amazon.titan-embed-text-v2:0".to_string());
            let aws_config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
            let client = aws_sdk_bedrockruntime::Client::new(&aws_config);
            tracing::info!(
                "model provider: bedrock (chat={chat_model}, embed={embed_model})"
            );
            Llm::Bedrock {
                client,
                chat_model,
                embed_model,
            }
        }
        "nvidia" => {
            let api_key = std::env::var("NVIDIA_API_KEY").unwrap_or_default();
            if api_key.trim().is_empty() {
                anyhow::bail!(
                    "MODEL_PROVIDER=nvidia requires NVIDIA_API_KEY to be set in .env (get one at build.nvidia.com)"
                );
            }
            let chat_model = std::env::var("NVIDIA_CHAT_MODEL")
                .unwrap_or_else(|_| "nvidia/nemotron-3-super-120b-a12b".to_string());
            let embed_model = std::env::var("NVIDIA_EMBED_MODEL")
                .unwrap_or_else(|_| "nvidia/llama-nemotron-embed-1b-v2".to_string());
            // Empty = don't send `dimensions` (natively-1024 models reject the field).
            let embed_dimensions = match std::env::var("NVIDIA_EMBED_DIMENSIONS") {
                Ok(s) if s.trim().is_empty() => None,
                Ok(s) => Some(s.trim().parse::<u32>().map_err(|_| {
                    anyhow::anyhow!("NVIDIA_EMBED_DIMENSIONS must be a number or empty, got {s:?}")
                })?),
                Err(_) => Some(1024),
            };
            let base_url = std::env::var("NVIDIA_BASE_URL")
                .unwrap_or_else(|_| "https://integrate.api.nvidia.com/v1".to_string());
            tracing::info!(
                "model provider: nvidia (chat={chat_model}, embed={embed_model}, dims={embed_dimensions:?}, base_url={base_url})"
            );
            Llm::Nvidia(nvidia::NvidiaClient::new(
                api_key,
                base_url,
                chat_model,
                embed_model,
                embed_dimensions,
            ))
        }
        other => {
            anyhow::bail!(
                "invalid MODEL_PROVIDER={other:?}; valid values are \"bedrock\" or \"nvidia\""
            );
        }
    };

    let state = Arc::new(AppState {
        pool,
        llm,
        files_dir,
    });

    let api_routes = Router::new()
        .route("/api/health", get(health))
        .route(
            "/api/documents",
            get(list_documents).post(ingest::upload_document),
        )
        .route("/api/documents/{id}", delete(ingest::delete_document))
        .route("/api/chat", post(chat))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .with_state(state);

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

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

#[derive(Debug, Serialize, sqlx::FromRow)]
struct DocumentRow {
    id: uuid::Uuid,
    filename: String,
    status: String,
    summary: Option<String>,
    tags: Option<Vec<String>>,
    error: Option<String>,
    created_at: chrono::DateTime<chrono::Utc>,
    chunk_count: i64,
}

async fn list_documents(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<DocumentRow>>, (axum::http::StatusCode, String)> {
    let rows = sqlx::query_as::<_, DocumentRow>(
        "SELECT d.id, d.filename, d.status, d.summary, d.tags, d.error, d.created_at, \
         (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count \
         FROM documents d ORDER BY d.created_at DESC",
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

    let llm = state.llm.clone();

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
                let llm2 = llm.clone();
                let stream_task = tokio::spawn(async move {
                    llm2.stream_chat_with_system(system_prompt, req.messages, inner_tx)
                        .await
                });

                while let Some(event) = inner_rx.recv().await {
                    match event {
                        SseEvent::Done => {
                            let _ = tx.send(SseEvent::Sources { sources: sources.clone() }).await;
                            let _ = tx.send(SseEvent::Done).await;
                        }
                        other => {
                            if tx.send(other).await.is_err() {
                                break;
                            }
                        }
                    }
                }
                // Dropping the receiver first is load-bearing: if we broke out early
                // (client disconnected), the pump blocks on a full channel forever and
                // stream_task.await would never resolve.
                drop(inner_rx);
                let _ = stream_task.await;
            }
            _ => {
                llm.stream_chat(req.messages, tx).await;
            }
        }
    });

    let stream = ReceiverStream::new(rx).map(|event| {
        let data = serde_json::to_string(&event)
            .unwrap_or_else(|_| r#"{"type":"error","message":"serialization failure"}"#.to_string());
        Ok(Event::default().data(data))
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}
