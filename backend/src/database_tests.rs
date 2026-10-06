use super::*;
use crate::storage::{Storage, StorageConfig, ensure_initialization_lock};
use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use pgvector::HalfVector;
use serde_json::Value;
use sqlx::{PgPool, Row};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::ServiceExt;
use uuid::Uuid;

#[derive(Clone)]
struct EmbeddingMockState {
    requests: Arc<Mutex<Vec<Value>>>,
    fail: Arc<AtomicBool>,
    summary_requests: Arc<Mutex<Vec<Value>>>,
    summary_status: Arc<AtomicU16>,
    summary_delay_ms: Arc<AtomicU64>,
    summary_response: Arc<Mutex<String>>,
    rerank_requests: Arc<Mutex<Vec<Value>>>,
}

async fn embedding_mock(
    State(state): State<EmbeddingMockState>,
    Json(body): Json<Value>,
) -> Response {
    state
        .requests
        .lock()
        .expect("embedding request lock")
        .push(body);
    if state.fail.load(Ordering::SeqCst) {
        return (StatusCode::BAD_REQUEST, "deterministic embedding failure").into_response();
    }

    Json(serde_json::json!({
        "data": [{
            "index": 0,
            "embedding": vec![0.25_f32; crate::embedding::EMBEDDING_DIMENSIONS]
        }]
    }))
    .into_response()
}

async fn summary_mock(
    State(state): State<EmbeddingMockState>,
    Json(body): Json<Value>,
) -> Response {
    let streamed_answer = body["stream"] == true;
    state
        .summary_requests
        .lock()
        .expect("summary request lock")
        .push(body);
    if streamed_answer {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"藍色頁面 [S1]\"}}]}\n\n\
                   data: [DONE]\n\n";
        return (
            [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
            sse,
        )
            .into_response();
    }
    let delay_ms = state.summary_delay_ms.load(Ordering::SeqCst);
    if delay_ms > 0 {
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    }
    let status = StatusCode::from_u16(state.summary_status.load(Ordering::SeqCst))
        .expect("valid summary mock status");
    if !status.is_success() {
        return (status, "provider-secret-summary-error").into_response();
    }
    let content = state
        .summary_response
        .lock()
        .expect("summary response lock")
        .clone();
    Json(serde_json::json!({
        "choices": [{ "message": { "content": content } }]
    }))
    .into_response()
}

async fn rerank_mock(State(state): State<EmbeddingMockState>, Json(body): Json<Value>) -> Response {
    let passage_count = body["passages"].as_array().map(Vec::len).unwrap_or(0);
    state
        .rerank_requests
        .lock()
        .expect("rerank request lock")
        .push(body);
    Json(serde_json::json!({
        "rankings": (0..passage_count)
            .map(|index| serde_json::json!({ "index": index, "logit": 1.0 - index as f64 }))
            .collect::<Vec<_>>()
    }))
    .into_response()
}

fn test_state(pool: &PgPool, models: CloudModelClient, storage: Storage) -> Arc<AppState> {
    Arc::new(AppState {
        pool: pool.clone(),
        models,
        storage,
        artifact_config: crate::page_artifacts::ArtifactConfig::for_test(),
        document_locks: crate::ingest::DocumentLocks::default(),
        profile: crate::profile::Profile::for_test(),
    })
}

fn png_bytes(width: u32, height: u32, pixel: [u8; 4]) -> Vec<u8> {
    let mut bytes = Vec::new();
    image::RgbaImage::from_pixel(width, height, image::Rgba(pixel))
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("encode PNG fixture");
    bytes
}

/// Stores an original under its relative key and registers the document row,
/// mirroring what upload does before background processing starts.
async fn store_original(
    pool: &PgPool,
    storage: &Storage,
    filename: &str,
    bytes: Vec<u8>,
) -> anyhow::Result<(Uuid, String)> {
    let document_id = Uuid::new_v4();
    let extension = std::path::Path::new(filename)
        .extension()
        .and_then(|extension| extension.to_str())
        .expect("fixture filename has an extension");
    let key = format!("{document_id}/original.{extension}");
    storage.put(&key, bytes).await?;
    sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, artifact_status) \
         VALUES ($1, $2, $3, 'application/octet-stream', 'processing', 'pending')",
    )
    .bind(document_id)
    .bind(filename)
    .bind(&key)
    .execute(pool)
    .await?;
    Ok((document_id, key))
}

async fn spawn_embedding_mock(
    fail: bool,
) -> (
    CloudModelClient,
    EmbeddingMockState,
    tokio::task::JoinHandle<()>,
) {
    spawn_embedding_mock_with_timeout(fail, Duration::from_secs(1)).await
}

async fn spawn_embedding_mock_with_timeout(
    fail: bool,
    timeout: Duration,
) -> (
    CloudModelClient,
    EmbeddingMockState,
    tokio::task::JoinHandle<()>,
) {
    let state = EmbeddingMockState {
        requests: Arc::new(Mutex::new(Vec::new())),
        fail: Arc::new(AtomicBool::new(fail)),
        summary_requests: Arc::new(Mutex::new(Vec::new())),
        summary_status: Arc::new(AtomicU16::new(StatusCode::OK.as_u16())),
        summary_delay_ms: Arc::new(AtomicU64::new(0)),
        summary_response: Arc::new(Mutex::new(
            r#"{"summary":"測試文件摘要","tags":["測試","文件","知識庫"]}"#.to_string(),
        )),
        rerank_requests: Arc::new(Mutex::new(Vec::new())),
    };
    let app = Router::new()
        .route("/embeddings", post(embedding_mock))
        .route("/chat/completions", post(summary_mock))
        .route("/ranking", post(rerank_mock))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind embedding mock");
    let address = listener.local_addr().expect("embedding mock address");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve embedding mock");
    });
    let embed_endpoint = format!("http://{address}/embeddings");
    let chat_endpoint = format!("http://{address}/chat/completions");
    let client = CloudModelClient::for_test_endpoints(&chat_endpoint, &embed_endpoint, timeout)
        .with_test_rerank_endpoint(&format!("http://{address}/ranking"));
    (client, state, handle)
}

fn vector_with(values: &[(usize, f32)]) -> HalfVector {
    let mut vector = vec![0.0; crate::embedding::EMBEDDING_DIMENSIONS];
    for (index, value) in values {
        vector[*index] = *value;
    }
    HalfVector::from_f32_slice(&vector)
}

async fn insert_document(pool: &PgPool, status: &str) -> sqlx::Result<Uuid> {
    let id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO documents (id, filename, storage_path, mime_type, status) \
         VALUES ($1, 'test.md', $2, 'text/markdown', $3)",
    )
    .bind(id)
    .bind(format!("/tmp/{id}.md"))
    .bind(status)
    .execute(pool)
    .await?;
    Ok(id)
}

async fn insert_text_unit(
    pool: &PgPool,
    document_id: Uuid,
    unit_index: i32,
    content: &str,
    embedding: &HalfVector,
) -> sqlx::Result<i64> {
    sqlx::query_scalar(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, content, needs_vision, embedding) \
         VALUES ($1, 'text', $2, $3, false, $4) RETURNING id",
    )
    .bind(document_id)
    .bind(unit_index)
    .bind(content)
    .bind(embedding)
    .fetch_one(pool)
    .await
}

async fn insert_page_unit(
    pool: &PgPool,
    document_id: Uuid,
    unit_index: i32,
    page_number: i32,
    content: Option<&str>,
    embedding: &HalfVector,
) -> sqlx::Result<i64> {
    sqlx::query_scalar(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, page_number, content, image_path, needs_vision, \
          embedding) \
         VALUES ($1, 'page', $2, $3, $4, $5, true, $6) RETURNING id",
    )
    .bind(document_id)
    .bind(unit_index)
    .bind(page_number)
    .bind(content)
    .bind(format!("/pages/{document_id}/{page_number:04}.png"))
    .bind(embedding)
    .fetch_one(pool)
    .await
}

#[sqlx::test]
async fn fresh_database_migrates_to_halfvec(pool: PgPool) -> anyhow::Result<()> {
    let embedding_type: String = sqlx::query_scalar(
        "SELECT format_type(a.atttypid, a.atttypmod) \
         FROM pg_attribute a \
         WHERE a.attrelid = 'document_units'::regclass \
           AND a.attname = 'embedding'",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(embedding_type, "halfvec(2048)");
    let embedding_is_not_null: bool = sqlx::query_scalar(
        "SELECT attnotnull FROM pg_attribute \
         WHERE attrelid = 'document_units'::regclass AND attname = 'embedding'",
    )
    .fetch_one(&pool)
    .await?;
    assert!(embedding_is_not_null);

    let index_definition: String = sqlx::query_scalar(
        "SELECT indexdef FROM pg_indexes \
         WHERE schemaname = current_schema() \
           AND indexname = 'document_units_embedding_hnsw'",
    )
    .fetch_one(&pool)
    .await?;
    assert!(index_definition.contains("USING hnsw"));
    assert!(index_definition.contains("halfvec_cosine_ops"));
    assert!(index_definition.contains("WHERE retrieval_ready"));

    let bm25_index_definition: String = sqlx::query_scalar(
        "SELECT indexdef FROM pg_indexes \
         WHERE schemaname = current_schema() \
           AND indexname = 'document_units_bm25_tokens_gin'",
    )
    .fetch_one(&pool)
    .await?;
    assert!(bm25_index_definition.contains("USING gin"));
    assert!(bm25_index_definition.contains("bm25_tokens"));
    assert!(bm25_index_definition.contains("retrieval_ready"));

    let chunks_table: Option<String> =
        sqlx::query_scalar("SELECT to_regclass('public.chunks')::text")
            .fetch_one(&pool)
            .await?;
    assert!(chunks_table.is_none());
    let page_artifacts_table: Option<String> =
        sqlx::query_scalar("SELECT to_regclass('public.page_artifacts')::text")
            .fetch_one(&pool)
            .await?;
    assert_eq!(page_artifacts_table.as_deref(), Some("page_artifacts"));
    Ok(())
}

#[sqlx::test(migrations = false)]
async fn migration_invalidates_legacy_vectors_but_preserves_documents(
    pool: PgPool,
) -> anyhow::Result<()> {
    sqlx::raw_sql(include_str!("../migrations/0001_init.sql"))
        .execute(&pool)
        .await?;

    let id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, summary, tags) \
         VALUES ($1, 'legacy.pdf', '/files/legacy.pdf', 'application/pdf', \
                 'ready', 'keep me', ARRAY['legacy'])",
    )
    .bind(id)
    .execute(&pool)
    .await?;

    let legacy_embedding = format!("[{}]", vec!["0.25"; 1024].join(","));
    sqlx::query(
        "INSERT INTO chunks (document_id, chunk_index, content, embedding) \
         VALUES ($1, 0, 'legacy content', $2::vector)",
    )
    .bind(id)
    .bind(legacy_embedding)
    .execute(&pool)
    .await?;

    sqlx::raw_sql(include_str!("../migrations/0002_page_units_halfvec.sql"))
        .execute(&pool)
        .await?;

    let row = sqlx::query(
        "SELECT filename, storage_path, status, error, summary, tags \
         FROM documents WHERE id = $1",
    )
    .bind(id)
    .fetch_one(&pool)
    .await?;
    assert_eq!(row.get::<String, _>("filename"), "legacy.pdf");
    assert_eq!(row.get::<String, _>("storage_path"), "/files/legacy.pdf");
    assert_eq!(row.get::<String, _>("status"), "failed");
    assert!(
        row.get::<Option<String>, _>("error")
            .unwrap()
            .contains("Re-ingestion required")
    );
    assert_eq!(
        row.get::<Option<String>, _>("summary").as_deref(),
        Some("keep me")
    );
    assert_eq!(
        row.get::<Option<Vec<String>>, _>("tags").unwrap(),
        vec!["legacy"]
    );

    let legacy_table: Option<String> =
        sqlx::query_scalar("SELECT to_regclass('public.chunks')::text")
            .fetch_one(&pool)
            .await?;
    assert!(legacy_table.is_none());
    let unit_count: i64 = sqlx::query_scalar("SELECT count(*) FROM document_units")
        .fetch_one(&pool)
        .await?;
    assert_eq!(unit_count, 0);
    Ok(())
}

#[sqlx::test]
async fn page_vision_columns_have_no_default_and_require_text_when_text_only(
    pool: PgPool,
) -> anyhow::Result<()> {
    for table in ["page_artifacts", "document_units"] {
        let (default, nullable): (Option<String>, String) = sqlx::query_as(
            "SELECT column_default, is_nullable FROM information_schema.columns \
             WHERE table_schema = current_schema() AND table_name = $1 \
               AND column_name = 'needs_vision'",
        )
        .bind(table)
        .fetch_one(&pool)
        .await?;
        assert_eq!(
            default, None,
            "{table}.needs_vision must be stated on every write"
        );
        assert_eq!(nullable, "NO");
    }

    let document_id = insert_document(&pool, "processing").await?;
    sqlx::query(
        "INSERT INTO document_artifact_sets \
         (document_id, source_kind, artifact_dir, manifest_path) \
         VALUES ($1, 'pdf', '/files/run', '/files/run/manifest.json')",
    )
    .bind(document_id)
    .execute(&pool)
    .await?;
    for (statement, accepted) in [
        (
            "INSERT INTO page_artifacts (document_id, page_number, image_path, width, height) \
             VALUES ($1, 1, '/p/1.png', 1, 1)",
            false,
        ),
        (
            "INSERT INTO page_artifacts \
             (document_id, page_number, image_path, width, height, needs_vision) \
             VALUES ($1, 1, '/p/1.png', 1, 1, false)",
            false,
        ),
        (
            "INSERT INTO page_artifacts \
             (document_id, page_number, image_path, width, height, extracted_text, needs_vision) \
             VALUES ($1, 1, '/p/1.png', 1, 1, '  ', false)",
            false,
        ),
        (
            "INSERT INTO page_artifacts \
             (document_id, page_number, image_path, width, height, extracted_text, needs_vision) \
             VALUES ($1, 1, '/p/1.png', 1, 1, 'table text', false)",
            true,
        ),
        (
            "INSERT INTO page_artifacts \
             (document_id, page_number, image_path, width, height, needs_vision) \
             VALUES ($1, 2, '/p/2.png', 1, 1, true)",
            true,
        ),
    ] {
        let result = sqlx::query(statement)
            .bind(document_id)
            .execute(&pool)
            .await;
        assert_eq!(result.is_ok(), accepted, "{statement}: {result:?}");
    }

    let embedding = vector_with(&[(0, 1.0)]);
    for (statement, accepted) in [
        (
            "INSERT INTO document_units (document_id, kind, unit_index, content, embedding) \
             VALUES ($1, 'text', 0, 'chunk', $2)",
            false,
        ),
        (
            "INSERT INTO document_units \
             (document_id, kind, unit_index, content, needs_vision, embedding) \
             VALUES ($1, 'text', 0, 'chunk', true, $2)",
            false,
        ),
        (
            "INSERT INTO document_units \
             (document_id, kind, unit_index, page_number, image_path, needs_vision, embedding) \
             VALUES ($1, 'page', 0, 1, '/p/1.png', false, $2)",
            false,
        ),
        (
            "INSERT INTO document_units \
             (document_id, kind, unit_index, page_number, content, image_path, needs_vision, \
              embedding) \
             VALUES ($1, 'page', 0, 1, ' ', '/p/1.png', false, $2)",
            false,
        ),
        (
            "INSERT INTO document_units \
             (document_id, kind, unit_index, page_number, content, image_path, needs_vision, \
              embedding) \
             VALUES ($1, 'page', 0, 1, 'table text', '/p/1.png', false, $2)",
            true,
        ),
        (
            "INSERT INTO document_units \
             (document_id, kind, unit_index, page_number, image_path, needs_vision, embedding) \
             VALUES ($1, 'page', 1, 2, '/p/2.png', true, $2)",
            true,
        ),
    ] {
        let result = sqlx::query(statement)
            .bind(document_id)
            .bind(&embedding)
            .execute(&pool)
            .await;
        assert_eq!(result.is_ok(), accepted, "{statement}: {result:?}");
    }
    Ok(())
}

#[sqlx::test(migrations = false)]
async fn page_vision_migration_keeps_existing_pages_on_the_image_path(
    pool: PgPool,
) -> anyhow::Result<()> {
    for migration in [
        include_str!("../migrations/0001_init.sql"),
        include_str!("../migrations/0002_page_units_halfvec.sql"),
        include_str!("../migrations/0003_ingest_artifacts.sql"),
        include_str!("../migrations/0004_hybrid_retrieval.sql"),
        include_str!("../migrations/0005_allow_binary_artifacts.sql"),
        include_str!("../migrations/0006_storage_settings.sql"),
    ] {
        sqlx::raw_sql(migration).execute(&pool).await?;
    }

    let document_id = insert_document(&pool, "ready").await?;
    sqlx::query(
        "INSERT INTO document_artifact_sets \
         (document_id, source_kind, artifact_dir, manifest_path) \
         VALUES ($1, 'pdf', '/files/run', '/files/run/manifest.json')",
    )
    .bind(document_id)
    .execute(&pool)
    .await?;
    sqlx::query(
        "INSERT INTO page_artifacts \
         (document_id, page_number, image_path, width, height, extracted_text) \
         VALUES ($1, 1, '/files/run/pages/0001.png', 10, 10, 'legacy page text')",
    )
    .bind(document_id)
    .execute(&pool)
    .await?;
    let embedding = vector_with(&[(0, 1.0)]);
    sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, page_number, content, image_path, embedding) \
         VALUES ($1, 'page', 0, 1, 'legacy page text', '/files/run/pages/0001.png', $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await?;
    sqlx::query(
        "INSERT INTO document_units (document_id, kind, unit_index, content, embedding) \
         VALUES ($1, 'text', 0, 'legacy chunk', $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await?;
    let stats_before: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM knowva_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;

    sqlx::raw_sql(include_str!("../migrations/0007_page_vision.sql"))
        .execute(&pool)
        .await?;

    let page_artifact_vision: bool =
        sqlx::query_scalar("SELECT needs_vision FROM page_artifacts WHERE document_id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert!(page_artifact_vision);
    let units: Vec<(String, bool, bool)> = sqlx::query_as(
        "SELECT kind, needs_vision, retrieval_ready FROM document_units \
         WHERE document_id = $1 ORDER BY kind",
    )
    .bind(document_id)
    .fetch_all(&pool)
    .await?;
    assert_eq!(
        units,
        vec![
            ("page".to_string(), true, true),
            ("text".to_string(), false, true),
        ]
    );
    let stats_after: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM knowva_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(stats_after, stats_before);
    Ok(())
}

#[sqlx::test]
async fn unit_constraints_dimensions_and_cascade_are_enforced(pool: PgPool) -> anyhow::Result<()> {
    let document_id = insert_document(&pool, "processing").await?;
    let embedding = vector_with(&[(0, 1.0)]);

    sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, page_number, image_path, needs_vision, embedding) \
         VALUES ($1, 'page', 0, 1, '/pages/1.png', true, $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await?;
    insert_text_unit(&pool, document_id, 0, "raw text", &embedding).await?;

    let page_zero = sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, page_number, image_path, needs_vision, embedding) \
         VALUES ($1, 'page', 1, 0, '/pages/0.png', true, $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await;
    assert!(page_zero.is_err());

    let page_without_number = sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, image_path, needs_vision, embedding) \
         VALUES ($1, 'page', 1, '/pages/missing-number.png', true, $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await;
    assert!(page_without_number.is_err());

    let page_without_image = sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, page_number, needs_vision, embedding) \
         VALUES ($1, 'page', 1, 2, true, $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await;
    assert!(page_without_image.is_err());

    let text_without_content = sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, needs_vision, embedding) \
         VALUES ($1, 'text', 1, false, $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await;
    assert!(text_without_content.is_err());

    let duplicate_page_number = sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, page_number, image_path, needs_vision, embedding) \
         VALUES ($1, 'page', 1, 1, '/pages/duplicate.png', true, $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await;
    assert!(duplicate_page_number.is_err());

    let duplicate_index = insert_text_unit(&pool, document_id, 0, "duplicate", &embedding).await;
    assert!(duplicate_index.is_err());

    let wrong_dimensions = HalfVector::from_f32_slice(&vec![0.0; 1024]);
    let dimension_error = sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, content, needs_vision, embedding) \
         VALUES ($1, 'text', 2, 'wrong dimension', false, $2)",
    )
    .bind(document_id)
    .bind(wrong_dimensions)
    .execute(&pool)
    .await
    .unwrap_err()
    .to_string();
    assert!(dimension_error.contains("2048"));
    assert!(dimension_error.contains("1024"));

    sqlx::query("DELETE FROM documents WHERE id = $1")
        .bind(document_id)
        .execute(&pool)
        .await?;
    let remaining: i64 =
        sqlx::query_scalar("SELECT count(*) FROM document_units WHERE document_id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(remaining, 0);
    Ok(())
}

#[sqlx::test]
async fn artifact_staging_constraints_are_enforced_without_weakening_halfvec(
    pool: PgPool,
) -> anyhow::Result<()> {
    let document_id = insert_document(&pool, "processing").await?;
    sqlx::query(
        "INSERT INTO document_artifact_sets \
         (document_id, source_kind, artifact_dir, manifest_path) \
         VALUES ($1, 'pdf', '/files/run', '/files/run/manifest.json')",
    )
    .bind(document_id)
    .execute(&pool)
    .await?;
    sqlx::query(
        "INSERT INTO page_artifacts \
         (document_id, page_number, image_path, width, height, needs_vision) \
         VALUES ($1, 1, '/files/run/pages/0001.png', 100, 200, true)",
    )
    .bind(document_id)
    .execute(&pool)
    .await?;

    for statement in [
        "INSERT INTO page_artifacts (document_id, page_number, image_path, width, height, needs_vision) VALUES ($1, 0, '/bad.png', 1, 1, true)",
        "INSERT INTO page_artifacts (document_id, page_number, image_path, width, height, needs_vision) VALUES ($1, 2, '', 1, 1, true)",
        "INSERT INTO page_artifacts (document_id, page_number, image_path, width, height, needs_vision) VALUES ($1, 2, '/bad.png', 0, 1, true)",
        "INSERT INTO page_artifacts (document_id, page_number, image_path, width, height, needs_vision) VALUES ($1, 1, '/duplicate.png', 1, 1, true)",
    ] {
        assert!(
            sqlx::query(statement)
                .bind(document_id)
                .execute(&pool)
                .await
                .is_err()
        );
    }

    let office_document = insert_document(&pool, "processing").await?;
    let office_without_pdf = sqlx::query(
        "INSERT INTO document_artifact_sets \
         (document_id, source_kind, artifact_dir, manifest_path) \
         VALUES ($1, 'office', '/files/office', '/files/office/manifest.json')",
    )
    .bind(office_document)
    .execute(&pool)
    .await;
    assert!(office_without_pdf.is_err());

    let unit_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM document_units WHERE document_id IN ($1, $2)")
            .bind(document_id)
            .bind(office_document)
            .fetch_one(&pool)
            .await?;
    assert_eq!(unit_count, 0);
    Ok(())
}

#[sqlx::test]
async fn halfvec_cosine_query_orders_results_and_uses_hnsw(pool: PgPool) -> anyhow::Result<()> {
    let document_id = insert_document(&pool, "ready").await?;
    let closest = vector_with(&[(0, 1.0)]);
    let nearby = vector_with(&[(0, 0.9), (1, 0.1)]);
    let orthogonal = vector_with(&[(1, 1.0)]);

    let closest_id = insert_text_unit(&pool, document_id, 0, "closest", &closest).await?;
    let nearby_id = insert_text_unit(&pool, document_id, 1, "nearby", &nearby).await?;
    let orthogonal_id = insert_text_unit(&pool, document_id, 2, "orthogonal", &orthogonal).await?;

    let query = vector_with(&[(0, 1.0)]);
    let rows: Vec<(i64, HalfVector)> = sqlx::query_as(
        "SELECT id, embedding FROM document_units \
         WHERE retrieval_ready \
         ORDER BY embedding <=> $1, id LIMIT 3",
    )
    .bind(&query)
    .fetch_all(&pool)
    .await?;
    assert_eq!(
        rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![closest_id, nearby_id, orthogonal_id]
    );
    assert_eq!(
        rows[0].1.as_slice().len(),
        crate::embedding::EMBEDDING_DIMENSIONS
    );

    let mut connection = pool.acquire().await?;
    sqlx::query("SET enable_seqscan = off")
        .execute(&mut *connection)
        .await?;
    let plan: Vec<String> = sqlx::query_scalar(
        "EXPLAIN (COSTS OFF) \
         SELECT id FROM document_units \
         WHERE retrieval_ready \
         ORDER BY embedding <=> $1, id LIMIT 2",
    )
    .bind(&query)
    .fetch_all(&mut *connection)
    .await?;
    assert!(
        plan.iter()
            .any(|line| line.contains("document_units_embedding_hnsw")),
        "expected HNSW index in query plan: {plan:?}"
    );
    Ok(())
}

#[sqlx::test]
async fn bm25_tokenizer_normalizes_traditional_chinese_english_numbers_and_product_codes(
    pool: PgPool,
) -> anyhow::Result<()> {
    let tokens: Vec<String> = sqlx::query_scalar(
        "SELECT knowva_bm25_tokenize('知識，管理！PostgreSQL／ＢＭ２５ 版本１２３ ＮＸ－２０４８－Ｖ２')",
    )
    .fetch_one(&pool)
    .await?;

    for expected in [
        "知識",
        "管理",
        "postgresql",
        "bm25",
        "版本",
        "123",
        "nx-2048-v2",
        "nx2048v2",
        "nx",
        "2048",
        "v2",
    ] {
        assert!(
            tokens.iter().any(|token| token == expected),
            "missing {expected:?} from {tokens:?}"
        );
    }

    let full_width: Vec<String> =
        sqlx::query_scalar("SELECT knowva_bm25_tokenize('ＮＸ－２０４８－Ｖ２')")
            .fetch_one(&pool)
            .await?;
    let half_width: Vec<String> = sqlx::query_scalar("SELECT knowva_bm25_tokenize('nx-2048-v2')")
        .fetch_one(&pool)
        .await?;
    assert_eq!(full_width, half_width);

    let punctuation_variant: Vec<String> =
        sqlx::query_scalar("SELECT knowva_bm25_tokenize('知識。管理 PostgreSQL, BM25; 123')")
            .fetch_one(&pool)
            .await?;
    for basic_term in ["知識", "管理", "postgresql", "bm25", "123"] {
        assert!(
            punctuation_variant.iter().any(|token| token == basic_term),
            "punctuation removed basic term {basic_term:?}: {punctuation_variant:?}"
        );
    }
    Ok(())
}

#[sqlx::test]
async fn bm25_ready_corpus_stats_follow_status_content_updates_and_cascade_delete(
    pool: PgPool,
) -> anyhow::Result<()> {
    let first_document = insert_document(&pool, "processing").await?;
    let embedding = vector_with(&[(0, 1.0)]);
    let first_unit = insert_text_unit(
        &pool,
        first_document,
        0,
        "alpha alpha product-100",
        &embedding,
    )
    .await?;

    let initial_stats: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM knowva_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(initial_stats, (0, 0));

    sqlx::query("UPDATE documents SET status = 'ready' WHERE id = $1")
        .bind(first_document)
        .execute(&pool)
        .await?;
    let first_length: i32 =
        sqlx::query_scalar("SELECT bm25_length FROM document_units WHERE id = $1")
            .bind(first_unit)
            .fetch_one(&pool)
            .await?;
    let ready_stats: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM knowva_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(ready_stats, (1, i64::from(first_length)));
    let alpha_df: i64 = sqlx::query_scalar(
        "SELECT document_frequency FROM knowva_bm25_term_stats WHERE term = 'alpha'",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(alpha_df, 1, "repeated terms count once toward DF");

    sqlx::query("UPDATE document_units SET content = 'beta beta' WHERE id = $1")
        .bind(first_unit)
        .execute(&pool)
        .await?;
    let changed_length: i32 =
        sqlx::query_scalar("SELECT bm25_length FROM document_units WHERE id = $1")
            .bind(first_unit)
            .fetch_one(&pool)
            .await?;
    let changed_stats: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM knowva_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(changed_stats, (1, i64::from(changed_length)));
    let alpha_df: Option<i64> = sqlx::query_scalar(
        "SELECT document_frequency FROM knowva_bm25_term_stats WHERE term = 'alpha'",
    )
    .fetch_optional(&pool)
    .await?;
    assert!(alpha_df.is_none());
    let beta_df: i64 = sqlx::query_scalar(
        "SELECT document_frequency FROM knowva_bm25_term_stats WHERE term = 'beta'",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(beta_df, 1);

    let second_document = insert_document(&pool, "ready").await?;
    insert_text_unit(&pool, second_document, 0, "beta gamma", &embedding).await?;
    let two_document_stats: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM knowva_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(two_document_stats.0, 2);
    let beta_df: i64 = sqlx::query_scalar(
        "SELECT document_frequency FROM knowva_bm25_term_stats WHERE term = 'beta'",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(beta_df, 2);

    sqlx::query("UPDATE documents SET status = 'failed' WHERE id = $1")
        .bind(first_document)
        .execute(&pool)
        .await?;
    let failed_stats: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM knowva_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(failed_stats.0, 1);
    let beta_df: i64 = sqlx::query_scalar(
        "SELECT document_frequency FROM knowva_bm25_term_stats WHERE term = 'beta'",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(beta_df, 1);

    sqlx::query("UPDATE documents SET status = 'ready' WHERE id = $1")
        .bind(first_document)
        .execute(&pool)
        .await?;
    let restored_count: i64 =
        sqlx::query_scalar("SELECT document_count FROM knowva_bm25_corpus_stats WHERE singleton")
            .fetch_one(&pool)
            .await?;
    assert_eq!(restored_count, 2);

    let processing_document = insert_document(&pool, "processing").await?;
    sqlx::query("UPDATE document_units SET document_id = $1 WHERE id = $2")
        .bind(processing_document)
        .bind(first_unit)
        .execute(&pool)
        .await?;
    let moved_ready: bool =
        sqlx::query_scalar("SELECT retrieval_ready FROM document_units WHERE id = $1")
            .bind(first_unit)
            .fetch_one(&pool)
            .await?;
    assert!(!moved_ready);
    let moved_count: i64 =
        sqlx::query_scalar("SELECT document_count FROM knowva_bm25_corpus_stats WHERE singleton")
            .fetch_one(&pool)
            .await?;
    assert_eq!(moved_count, 1);

    sqlx::query("UPDATE document_units SET retrieval_ready = true WHERE id = $1")
        .bind(first_unit)
        .execute(&pool)
        .await?;
    let canonical_ready: bool =
        sqlx::query_scalar("SELECT retrieval_ready FROM document_units WHERE id = $1")
            .bind(first_unit)
            .fetch_one(&pool)
            .await?;
    assert!(
        !canonical_ready,
        "direct flag updates must follow canonical document status"
    );
    let canonical_count: i64 =
        sqlx::query_scalar("SELECT document_count FROM knowva_bm25_corpus_stats WHERE singleton")
            .fetch_one(&pool)
            .await?;
    assert_eq!(canonical_count, 1);

    sqlx::query("UPDATE document_units SET document_id = $1 WHERE id = $2")
        .bind(first_document)
        .bind(first_unit)
        .execute(&pool)
        .await?;
    let moved_back_count: i64 =
        sqlx::query_scalar("SELECT document_count FROM knowva_bm25_corpus_stats WHERE singleton")
            .fetch_one(&pool)
            .await?;
    assert_eq!(moved_back_count, 2);

    sqlx::query("DELETE FROM documents WHERE id = $1")
        .bind(second_document)
        .execute(&pool)
        .await?;
    let after_first_delete: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM knowva_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(after_first_delete, (1, i64::from(changed_length)));

    sqlx::query("DELETE FROM documents WHERE id = $1")
        .bind(first_document)
        .execute(&pool)
        .await?;
    let final_stats: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM knowva_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(final_stats, (0, 0));
    let term_count: i64 = sqlx::query_scalar("SELECT count(*) FROM knowva_bm25_term_stats")
        .fetch_one(&pool)
        .await?;
    assert_eq!(term_count, 0);
    Ok(())
}

#[sqlx::test]
async fn dense_and_bm25_return_complete_ready_only_metadata_and_keep_scan_pages(
    pool: PgPool,
) -> anyhow::Result<()> {
    let query_vector = vector_with(&[(0, 1.0)]);
    let orthogonal_vector = vector_with(&[(1, 1.0)]);

    let scan_document = insert_document(&pool, "ready").await?;
    sqlx::query("UPDATE documents SET filename = 'scan.pdf' WHERE id = $1")
        .bind(scan_document)
        .execute(&pool)
        .await?;
    let scan_unit = insert_page_unit(&pool, scan_document, 0, 1, None, &query_vector).await?;

    let product_document = insert_document(&pool, "ready").await?;
    sqlx::query("UPDATE documents SET filename = 'products.md' WHERE id = $1")
        .bind(product_document)
        .execute(&pool)
        .await?;
    let product_unit = insert_text_unit(
        &pool,
        product_document,
        0,
        "精確產品代號 NX-2048-V2，支援 PostgreSQL 17。",
        &orthogonal_vector,
    )
    .await?;

    let duplicate_document = insert_document(&pool, "ready").await?;
    sqlx::query("UPDATE documents SET filename = 'duplicate.pdf' WHERE id = $1")
        .bind(duplicate_document)
        .execute(&pool)
        .await?;
    let duplicate_one = insert_page_unit(
        &pool,
        duplicate_document,
        0,
        1,
        Some("重複內容：人工智慧知識中樞"),
        &orthogonal_vector,
    )
    .await?;
    let duplicate_two = insert_page_unit(
        &pool,
        duplicate_document,
        1,
        2,
        Some("重複內容：人工智慧知識中樞"),
        &orthogonal_vector,
    )
    .await?;

    for status in ["processing", "failed"] {
        let inactive_document = insert_document(&pool, status).await?;
        insert_text_unit(
            &pool,
            inactive_document,
            0,
            "NX-2048-V2 NX-2048-V2 NX-2048-V2",
            &query_vector,
        )
        .await?;
    }

    let dense = crate::search::dense_candidates_from_pool(&pool, &query_vector, 50).await?;
    assert_eq!(dense[0].metadata.unit_id, scan_unit);
    assert_eq!(dense[0].rank, 1);
    assert_eq!(dense[0].metadata.document_id, scan_document);
    assert_eq!(dense[0].metadata.filename, "scan.pdf");
    assert_eq!(dense[0].metadata.kind, "page");
    assert_eq!(dense[0].metadata.page_number, Some(1));
    assert!(dense[0].metadata.image_path.is_some());
    assert!(dense[0].metadata.content.is_none());
    assert_eq!(dense.len(), 4, "only four ready units are dense candidates");

    let product =
        crate::search::bm25_candidates_from_pool(&pool, "ｎｘ－２０４８－ｖ２", 50).await?;
    assert_eq!(product.len(), 1);
    assert_eq!(product[0].metadata.unit_id, product_unit);
    assert_eq!(product[0].rank, 1);
    assert!(product[0].score > 0.0);
    assert_eq!(product[0].metadata.filename, "products.md");
    assert_eq!(product[0].metadata.kind, "text");
    assert_eq!(product[0].metadata.page_number, None);
    assert_eq!(product[0].metadata.image_path, None);
    assert!(
        product[0]
            .metadata
            .content
            .as_deref()
            .is_some_and(|content| content.contains("NX-2048-V2"))
    );

    let duplicate = crate::search::bm25_candidates_from_pool(&pool, "重複，內容", 50).await?;
    assert_eq!(
        duplicate
            .iter()
            .map(|candidate| candidate.metadata.unit_id)
            .collect::<Vec<_>>(),
        vec![
            duplicate_one.min(duplicate_two),
            duplicate_one.max(duplicate_two)
        ]
    );
    assert!(
        duplicate
            .iter()
            .all(|candidate| candidate.metadata.image_path.is_some())
    );
    assert!(
        duplicate
            .iter()
            .all(|candidate| candidate.metadata.page_number.is_some())
    );
    assert!(
        product
            .iter()
            .chain(&duplicate)
            .all(|candidate| candidate.metadata.unit_id != scan_unit),
        "image-only scan page must not be fabricated into lexical results"
    );
    Ok(())
}

#[sqlx::test]
async fn lexical_scores_match_okapi_bm25_formula_and_do_not_pad_top_k(
    pool: PgPool,
) -> anyhow::Result<()> {
    let embedding = vector_with(&[(0, 1.0)]);
    let first_document = insert_document(&pool, "ready").await?;
    let first_id = insert_text_unit(&pool, first_document, 0, "alpha alpha", &embedding).await?;
    let second_document = insert_document(&pool, "ready").await?;
    let second_id =
        insert_text_unit(&pool, second_document, 0, "alpha beta beta", &embedding).await?;
    let third_document = insert_document(&pool, "ready").await?;
    insert_text_unit(&pool, third_document, 0, "gamma", &embedding).await?;

    let results = crate::search::bm25_candidates_from_pool(&pool, "alpha", 50).await?;
    assert_eq!(results.len(), 2, "top_k is not padded with placeholders");
    assert_eq!(results[0].metadata.unit_id, first_id);
    assert_eq!(results[1].metadata.unit_id, second_id);

    let idf = (1.0_f64 + ((3.0_f64 - 2.0 + 0.5) / (2.0 + 0.5))).ln();
    let expected_first = idf * ((2.0 * (1.2 + 1.0)) / (2.0 + 1.2 * (1.0 - 0.75 + 0.75)));
    let expected_second =
        idf * ((1.0 * (1.2 + 1.0)) / (1.0 + 1.2 * (1.0 - 0.75 + 0.75 * (3.0 / 2.0))));
    assert!((results[0].score - expected_first).abs() < 1e-12);
    assert!((results[1].score - expected_second).abs() < 1e-12);
    Ok(())
}

#[sqlx::test]
async fn production_candidate_predicates_use_hnsw_and_gin_indexes(
    pool: PgPool,
) -> anyhow::Result<()> {
    let document_id = insert_document(&pool, "ready").await?;
    let query_vector = vector_with(&[(0, 1.0)]);
    insert_text_unit(
        &pool,
        document_id,
        0,
        "NX-2048-V2 deterministic fixture",
        &query_vector,
    )
    .await?;

    let mut connection = pool.acquire().await?;
    sqlx::query("SET enable_seqscan = off")
        .execute(&mut *connection)
        .await?;
    let dense_plan: Vec<String> = sqlx::query_scalar(
        "EXPLAIN (COSTS OFF) \
         WITH nearest AS MATERIALIZED ( \
           SELECT u.id, u.document_id, (u.embedding <=> $1) AS distance \
           FROM document_units AS u \
           WHERE u.retrieval_ready \
           ORDER BY u.embedding <=> $1 \
           LIMIT 50 \
         ) \
         SELECT nearest.id \
         FROM nearest \
         JOIN documents AS d ON d.id = nearest.document_id \
         WHERE d.status = 'ready' \
         ORDER BY nearest.distance, nearest.id \
         LIMIT 50",
    )
    .bind(&query_vector)
    .fetch_all(&mut *connection)
    .await?;
    assert!(
        dense_plan
            .iter()
            .any(|line| line.contains("document_units_embedding_hnsw")),
        "production dense query must use HNSW: {dense_plan:?}"
    );

    let lexical_plan: Vec<String> = sqlx::query_scalar(
        "EXPLAIN (COSTS OFF) \
         WITH query_terms AS MATERIALIZED ( \
           SELECT DISTINCT token AS term \
           FROM unnest(knowva_bm25_tokenize($1)) AS token \
           WHERE token <> '' \
         ), query_token_array AS ( \
           SELECT COALESCE(array_agg(term ORDER BY term), ARRAY[]::text[]) AS terms \
           FROM query_terms \
         ), lexical AS MATERIALIZED ( \
           SELECT u.id, u.document_id \
           FROM document_units AS u \
           CROSS JOIN query_token_array AS query \
           WHERE u.retrieval_ready \
             AND u.content IS NOT NULL \
             AND btrim(u.content) <> '' \
             AND u.bm25_tokens && query.terms \
         ) \
         SELECT lexical.id \
         FROM lexical \
         JOIN documents AS d ON d.id = lexical.document_id \
         WHERE d.status = 'ready'",
    )
    .bind("ＮＸ－２０４８－Ｖ２")
    .fetch_all(&mut *connection)
    .await?;
    assert!(
        lexical_plan
            .iter()
            .any(|line| line.contains("document_units_bm25_tokens_gin")),
        "production lexical candidate predicate must use GIN: {lexical_plan:?}"
    );
    Ok(())
}

#[sqlx::test]
async fn document_list_and_delete_api_use_the_new_schema(pool: PgPool) -> anyhow::Result<()> {
    let document_id = Uuid::new_v4();
    let files_dir = std::env::temp_dir().join(format!("knowva-server-{document_id}"));
    let storage = Storage::local(&files_dir)?;
    let storage_key = format!("{document_id}/original.pdf");
    let artifact_dir = format!("{document_id}/artifacts-fixture");
    let manifest_path = format!("{artifact_dir}/manifest.json");
    let page_path = format!("{artifact_dir}/pages/0001.png");
    // A run uploaded by an interrupted process is still below the prefix.
    let stale_run_object = format!("{document_id}/artifacts-stale/pages/0001.png");
    for (key, bytes) in [
        (&storage_key, b"test document".to_vec()),
        (&manifest_path, b"{}".to_vec()),
        (&page_path, b"fixture".to_vec()),
        (&stale_run_object, b"partial".to_vec()),
    ] {
        storage.put(key, bytes).await?;
    }

    sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, artifact_status, error) \
         VALUES ($1, 'api.pdf', $2, 'application/pdf', 'processing', 'ready', \
                 'provider-secret Authorization: Bearer should-never-leak')",
    )
    .bind(document_id)
    .bind(&storage_key)
    .execute(&pool)
    .await?;
    sqlx::query(
        "INSERT INTO document_artifact_sets \
         (document_id, source_kind, artifact_dir, manifest_path) \
         VALUES ($1, 'pdf', $2, $3)",
    )
    .bind(document_id)
    .bind(&artifact_dir)
    .bind(&manifest_path)
    .execute(&pool)
    .await?;
    sqlx::query(
        "INSERT INTO page_artifacts \
         (document_id, page_number, image_path, width, height, extracted_text, needs_vision) \
         VALUES ($1, 1, $2, 640, 480, 'native text', true)",
    )
    .bind(document_id)
    .bind(&page_path)
    .execute(&pool)
    .await?;

    let embedding = vector_with(&[(0, 1.0)]);
    sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, page_number, image_path, needs_vision, embedding) \
         VALUES ($1, 'page', 0, 1, $2, true, $3)",
    )
    .bind(document_id)
    .bind(&page_path)
    .bind(&embedding)
    .execute(&pool)
    .await?;
    insert_text_unit(&pool, document_id, 0, "api text", &embedding).await?;

    let state = test_state(&pool, CloudModelClient::for_test(), storage.clone());
    let app = api_router(state.clone());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/documents")
                .body(Body::empty())?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await?;
    let documents: Value = serde_json::from_slice(&body)?;
    let document = &documents.as_array().unwrap()[0];
    assert_eq!(document["id"], document_id.to_string());
    assert_eq!(document["status"], "processing");
    assert_eq!(document["processingStage"], "indexing");
    assert!(document["processingError"].is_null());
    assert!(document["error"].is_null());
    assert_eq!(document["artifact_status"], "ready");
    assert_eq!(document["pageCount"], 1);
    assert_eq!(document["unitCount"], 2);
    assert_eq!(document["page_count"], 1);
    assert_eq!(document["unit_count"], 2);
    assert_eq!(document["chunk_count"], 2);
    assert_eq!(document["createdAt"], document["created_at"]);
    assert!(!String::from_utf8_lossy(&body).contains("provider-secret"));

    let processing_guard = state.document_locks.lock(document_id).await;
    let request = Request::builder()
        .method("DELETE")
        .uri(format!("/api/documents/{document_id}"))
        .body(Body::empty())?;
    let mut delete = Box::pin(app.oneshot(request));
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(25), &mut delete)
            .await
            .is_err(),
        "delete must wait for the in-flight conversion lock"
    );
    drop(processing_guard);
    let response = tokio::time::timeout(std::time::Duration::from_secs(1), &mut delete).await??;
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    let document_count: i64 = sqlx::query_scalar("SELECT count(*) FROM documents WHERE id = $1")
        .bind(document_id)
        .fetch_one(&pool)
        .await?;
    let unit_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM document_units WHERE document_id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    let artifact_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM page_artifacts WHERE document_id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(document_count, 0);
    assert_eq!(unit_count, 0);
    assert_eq!(artifact_count, 0);
    assert!(storage.list(&document_id.to_string()).await?.is_empty());
    assert!(
        !files_dir.join(document_id.to_string()).exists(),
        "no empty document directory may remain"
    );

    tokio::fs::remove_dir_all(files_dir).await?;
    Ok(())
}

#[sqlx::test]
async fn image_embedding_publish_is_idempotent_and_never_duplicates_units(
    pool: PgPool,
) -> anyhow::Result<()> {
    let files_dir = std::env::temp_dir().join(format!("knowva-artifact-retry-{}", Uuid::new_v4()));
    let storage = Storage::local(&files_dir)?;
    let (document_id, storage_key) = store_original(
        &pool,
        &storage,
        "retry.png",
        png_bytes(11, 7, [20, 40, 60, 255]),
    )
    .await?;

    let (models, mock, server) = spawn_embedding_mock(false).await;
    let state = test_state(&pool, models, storage.clone());
    crate::ingest::process_document(
        state.clone(),
        document_id,
        storage_key.clone(),
        crate::page_artifacts::SourceKind::Image,
    )
    .await;

    let first_artifact_dir: String = sqlx::query_scalar(
        "SELECT artifact_dir FROM document_artifact_sets WHERE document_id = $1",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;
    let state_row: (String, String) =
        sqlx::query_as("SELECT status, artifact_status FROM documents WHERE id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(state_row, ("ready".to_string(), "ready".to_string()));
    assert!(first_artifact_dir.starts_with(&format!("{document_id}/artifacts-")));
    let first_objects = storage.list(&first_artifact_dir).await?;
    let manifest_key = format!("{first_artifact_dir}/manifest.json");
    assert!(first_objects.contains(&manifest_key), "{first_objects:?}");
    assert!(first_objects.contains(&format!("{first_artifact_dir}/pages/0001.png")));
    // The manifest records relative keys, never absolute local paths.
    let manifest: crate::page_artifacts::ArtifactManifest =
        serde_json::from_slice(&storage.get(document_id, &manifest_key, usize::MAX).await?)?;
    assert_eq!(manifest.original_path, storage_key);
    assert_eq!(
        manifest.pages[0].image_path,
        format!("{first_artifact_dir}/pages/0001.png")
    );

    // A run uploaded by a process that died before its publishing commit.
    let interrupted = format!("{document_id}/artifacts-interrupted/pages/0001.png");
    storage.put(&interrupted, b"partial".to_vec()).await?;
    crate::ingest::process_document(
        state,
        document_id,
        storage_key.clone(),
        crate::page_artifacts::SourceKind::Image,
    )
    .await;

    let second_artifact_dir: String = sqlx::query_scalar(
        "SELECT artifact_dir FROM document_artifact_sets WHERE document_id = $1",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;
    assert_ne!(first_artifact_dir, second_artifact_dir);
    let objects = storage.list(&document_id.to_string()).await?;
    assert!(
        objects
            .iter()
            .all(|key| key == &storage_key || key.starts_with(&format!("{second_artifact_dir}/"))),
        "superseded and interrupted runs must be removed: {objects:?}"
    );
    assert!(!files_dir.join(&first_artifact_dir).exists());
    let page_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM page_artifacts WHERE document_id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    let unit_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM document_units WHERE document_id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(page_count, 1);
    assert_eq!(unit_count, 1, "rerun must replace instead of duplicate");
    let unit: (String, i32, i32, String, Option<String>, HalfVector) = sqlx::query_as(
        "SELECT kind, unit_index, page_number, image_path, content, embedding \
         FROM document_units WHERE document_id = $1",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;
    assert_eq!(unit.0, "page");
    assert_eq!((unit.1, unit.2), (0, 1));
    assert_eq!(unit.3, format!("{second_artifact_dir}/pages/0001.png"));
    assert!(unit.4.is_none());
    assert_eq!(
        unit.5.as_slice().len(),
        crate::embedding::EMBEDDING_DIMENSIONS
    );
    assert_eq!(mock.requests.lock().expect("request lock").len(), 2);

    server.abort();
    tokio::fs::remove_dir_all(files_dir).await?;
    Ok(())
}

#[sqlx::test]
async fn embedding_failure_keeps_artifacts_ready_but_document_not_ready_and_retry_is_atomic(
    pool: PgPool,
) -> anyhow::Result<()> {
    let storage = Storage::memory();
    let (document_id, storage_key) = store_original(
        &pool,
        &storage,
        "failure.png",
        png_bytes(12, 8, [80, 40, 20, 255]),
    )
    .await?;

    let (models, mock, server) = spawn_embedding_mock(true).await;
    let state = test_state(&pool, models, storage);
    crate::ingest::process_document(
        state.clone(),
        document_id,
        storage_key.clone(),
        crate::page_artifacts::SourceKind::Image,
    )
    .await;

    let failed: (String, String, Option<String>) =
        sqlx::query_as("SELECT status, artifact_status, error FROM documents WHERE id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(failed.0, "failed");
    assert_eq!(
        failed.1, "ready",
        "completed artifacts must remain retryable"
    );
    let error = failed.2.expect("embedding failure is recorded");
    assert!(error.contains("embedding-request"));
    assert!(error.contains("unit=0"));
    assert!(error.contains("page=1"));
    assert!(!error.contains(&storage_key));
    assert!(!error.contains("base64"));
    let failed_counts: (i64, i64) = sqlx::query_as(
        "SELECT \
           (SELECT count(*) FROM page_artifacts WHERE document_id = $1), \
           (SELECT count(*) FROM document_units WHERE document_id = $1)",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;
    assert_eq!(failed_counts, (1, 0));
    assert_eq!(mock.requests.lock().expect("request lock").len(), 1);

    mock.fail.store(false, Ordering::SeqCst);
    crate::ingest::process_document(
        state.clone(),
        document_id,
        storage_key.clone(),
        crate::page_artifacts::SourceKind::Image,
    )
    .await;

    let ready: (String, String, Option<String>) =
        sqlx::query_as("SELECT status, artifact_status, error FROM documents WHERE id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(ready, ("ready".to_string(), "ready".to_string(), None));
    let ready_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM document_units WHERE document_id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(ready_count, 1);
    assert_eq!(mock.requests.lock().expect("request lock").len(), 2);

    // A failed re-run of a ready document keeps the run behind its current units,
    // because superseded runs are removed only after the new units are promoted.
    mock.fail.store(true, Ordering::SeqCst);
    crate::ingest::process_document(
        state.clone(),
        document_id,
        storage_key,
        crate::page_artifacts::SourceKind::Image,
    )
    .await;
    let unit_keys: Vec<String> = sqlx::query_scalar(
        "SELECT image_path FROM document_units \
         WHERE document_id = $1 AND image_path IS NOT NULL",
    )
    .bind(document_id)
    .fetch_all(&pool)
    .await?;
    assert_eq!(unit_keys.len(), 1);
    for key in unit_keys {
        state
            .storage
            .get(document_id, &key, usize::MAX)
            .await
            .expect("unit image must survive a failed re-run");
    }

    server.abort();
    Ok(())
}

#[sqlx::test]
async fn summary_success_and_permanent_failures_publish_ready_without_reembedding(
    pool: PgPool,
) -> anyhow::Result<()> {
    async fn process_text_fixture(
        pool: &PgPool,
        state: Arc<AppState>,
        filename: &str,
    ) -> anyhow::Result<Uuid> {
        let (document_id, storage_key) = store_original(
            pool,
            &state.storage,
            filename,
            "# 測試文件\n\n這是供摘要與索引測試使用的繁體中文內容。"
                .as_bytes()
                .to_vec(),
        )
        .await?;
        crate::ingest::process_document(
            state,
            document_id,
            storage_key,
            crate::page_artifacts::SourceKind::Text,
        )
        .await;
        Ok(document_id)
    }

    async fn metadata_row(
        pool: &PgPool,
        document_id: Uuid,
    ) -> anyhow::Result<(
        String,
        String,
        Option<String>,
        Option<Vec<String>>,
        Option<String>,
        i64,
    )> {
        Ok(sqlx::query_as(
            "SELECT d.status, d.artifact_status, d.summary, d.tags, d.error, \
                    (SELECT count(*) FROM document_units u WHERE u.document_id = d.id) \
             FROM documents d WHERE d.id = $1",
        )
        .bind(document_id)
        .fetch_one(pool)
        .await?)
    }

    let (models, mock, server) =
        spawn_embedding_mock_with_timeout(false, Duration::from_millis(20)).await;
    let state = test_state(&pool, models, Storage::memory());

    let success = process_text_fixture(&pool, state.clone(), "success.md").await?;
    let success_row = metadata_row(&pool, success).await?;
    assert_eq!(success_row.0, "ready");
    assert_eq!(success_row.1, "ready");
    assert_eq!(success_row.2.as_deref(), Some("測試文件摘要"));
    assert_eq!(
        success_row.3,
        Some(vec![
            "測試".to_string(),
            "文件".to_string(),
            "知識庫".to_string()
        ])
    );
    assert!(success_row.4.is_none());
    assert_eq!(success_row.5, 1);

    mock.summary_status
        .store(StatusCode::TOO_MANY_REQUESTS.as_u16(), Ordering::SeqCst);
    let rate_limited = process_text_fixture(&pool, state.clone(), "rate-limit.md").await?;
    let rate_limited_row = metadata_row(&pool, rate_limited).await?;
    assert_eq!(
        rate_limited_row,
        (
            "ready".to_string(),
            "ready".to_string(),
            None,
            None,
            None,
            1
        )
    );

    mock.summary_status
        .store(StatusCode::OK.as_u16(), Ordering::SeqCst);
    *mock.summary_response.lock().expect("summary response lock") = "not-json".to_string();
    let malformed = process_text_fixture(&pool, state.clone(), "malformed.md").await?;
    let malformed_row = metadata_row(&pool, malformed).await?;
    assert_eq!(malformed_row.0, "ready");
    assert_eq!(malformed_row.1, "ready");
    assert!(malformed_row.2.is_none());
    assert!(malformed_row.3.is_none());
    assert!(malformed_row.4.is_none());
    assert_eq!(malformed_row.5, 1);

    *mock.summary_response.lock().expect("summary response lock") =
        r#"{"summary":"不應抵達","tags":["一","二","三"]}"#.to_string();
    mock.summary_delay_ms.store(100, Ordering::SeqCst);
    let timed_out = process_text_fixture(&pool, state, "timeout.md").await?;
    let timed_out_row = metadata_row(&pool, timed_out).await?;
    assert_eq!(timed_out_row.0, "ready");
    assert_eq!(timed_out_row.1, "ready");
    assert!(timed_out_row.2.is_none());
    assert!(timed_out_row.3.is_none());
    assert!(timed_out_row.4.is_none());
    assert_eq!(timed_out_row.5, 1);

    assert_eq!(
        mock.requests.lock().expect("embedding request lock").len(),
        4,
        "each document must be embedded exactly once"
    );
    assert_eq!(
        mock.summary_requests
            .lock()
            .expect("summary request lock")
            .len(),
        6,
        "success/malformed use one request; 429/timeout use two bounded attempts"
    );

    server.abort();
    Ok(())
}

#[sqlx::test]
async fn page_image_and_text_are_promoted_to_one_matching_halfvec_unit(
    pool: PgPool,
) -> anyhow::Result<()> {
    let storage = Storage::memory();
    let document_id = Uuid::new_v4();
    let artifact_dir = format!("{document_id}/artifacts-fixture");
    let page_path = format!("{artifact_dir}/pages/0001.png");
    storage
        .put(&page_path, png_bytes(16, 9, [10, 30, 50, 255]))
        .await?;
    let original_key = format!("{document_id}/original.pdf");
    storage.put(&original_key, b"fixture".to_vec()).await?;
    sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, artifact_status) \
         VALUES ($1, 'mixed.pdf', $2, 'application/pdf', 'processing', 'ready')",
    )
    .bind(document_id)
    .bind(&original_key)
    .execute(&pool)
    .await?;
    sqlx::query(
        "INSERT INTO document_artifact_sets \
         (document_id, source_kind, artifact_dir, manifest_path) \
         VALUES ($1, 'pdf', $2, $3)",
    )
    .bind(document_id)
    .bind(&artifact_dir)
    .bind(format!("{artifact_dir}/manifest.json"))
    .execute(&pool)
    .await?;
    let page = crate::page_artifacts::PageArtifact {
        document_id,
        page_number: 1,
        image_path: page_path.clone(),
        width: 16,
        height: 9,
        extracted_text: Some("native same-page text".to_string()),
        needs_vision: true,
    };
    sqlx::query(
        "INSERT INTO page_artifacts \
         (document_id, page_number, image_path, width, height, extracted_text, needs_vision) \
         VALUES ($1, $2, $3, $4, $5, $6, $7)",
    )
    .bind(document_id)
    .bind(page.page_number)
    .bind(&page.image_path)
    .bind(page.width)
    .bind(page.height)
    .bind(&page.extracted_text)
    .bind(page.needs_vision)
    .execute(&pool)
    .await?;

    let (models, mock, server) = spawn_embedding_mock(false).await;
    let state = test_state(&pool, models, storage);
    crate::ingest::embed_published_artifacts_for_test(&state, document_id, &[page], &[])
        .await
        .map_err(anyhow::Error::msg)?;

    let status: (String, String) =
        sqlx::query_as("SELECT status, artifact_status FROM documents WHERE id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(status, ("ready".to_string(), "ready".to_string()));
    let unit: (String, i32, i32, Option<String>, String, HalfVector) = sqlx::query_as(
        "SELECT kind, unit_index, page_number, content, image_path, embedding \
         FROM document_units WHERE document_id = $1",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;
    assert_eq!(unit.0, "page");
    assert_eq!((unit.1, unit.2), (0, 1));
    assert_eq!(unit.3.as_deref(), Some("native same-page text"));
    assert_eq!(unit.4, page_path);
    assert_eq!(
        unit.5.as_slice().len(),
        crate::embedding::EMBEDDING_DIMENSIONS
    );
    assert!(
        unit.5
            .as_slice()
            .iter()
            .all(|value| value.to_f32().is_finite())
    );

    {
        let requests = mock.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["input_type"], "passage");
        assert_eq!(requests[0]["modality"], serde_json::json!(["text_image"]));
        let input = requests[0]["input"][0]
            .as_str()
            .expect("mixed input string");
        assert!(input.starts_with("native same-page text\n<img src=\"data:image/png;base64,"));
        assert!(!input.contains(&page_path));
    }

    server.abort();
    Ok(())
}

#[sqlx::test]
async fn text_only_page_is_promoted_from_text_without_reading_its_image(
    pool: PgPool,
) -> anyhow::Result<()> {
    // The page image is deliberately absent: a text-only page must not read it.
    let storage = Storage::memory();
    let document_id = Uuid::new_v4();
    let artifact_dir = format!("{document_id}/artifacts-fixture");
    let page_path = format!("{artifact_dir}/pages/0001.png");
    let original_key = format!("{document_id}/original.pdf");
    storage.put(&original_key, b"fixture".to_vec()).await?;
    sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, artifact_status) \
         VALUES ($1, 'tables.pdf', $2, 'application/pdf', 'processing', 'ready')",
    )
    .bind(document_id)
    .bind(&original_key)
    .execute(&pool)
    .await?;
    sqlx::query(
        "INSERT INTO document_artifact_sets \
         (document_id, source_kind, artifact_dir, manifest_path) \
         VALUES ($1, 'pdf', $2, $3)",
    )
    .bind(document_id)
    .bind(&artifact_dir)
    .bind(format!("{artifact_dir}/manifest.json"))
    .execute(&pool)
    .await?;
    let page = crate::page_artifacts::PageArtifact {
        document_id,
        page_number: 1,
        image_path: page_path.clone(),
        width: 16,
        height: 9,
        extracted_text: Some("IL050 deployment table".to_string()),
        needs_vision: false,
    };
    sqlx::query(
        "INSERT INTO page_artifacts \
         (document_id, page_number, image_path, width, height, extracted_text, needs_vision) \
         VALUES ($1, $2, $3, $4, $5, $6, $7)",
    )
    .bind(document_id)
    .bind(page.page_number)
    .bind(&page.image_path)
    .bind(page.width)
    .bind(page.height)
    .bind(&page.extracted_text)
    .bind(page.needs_vision)
    .execute(&pool)
    .await?;

    let (models, mock, server) = spawn_embedding_mock(false).await;
    let state = test_state(&pool, models, storage);
    crate::ingest::embed_published_artifacts_for_test(&state, document_id, &[page], &[])
        .await
        .map_err(anyhow::Error::msg)?;

    let status: String = sqlx::query_scalar("SELECT status FROM documents WHERE id = $1")
        .bind(document_id)
        .fetch_one(&pool)
        .await?;
    assert_eq!(status, "ready");
    let unit: (String, i32, Option<String>, String, bool) = sqlx::query_as(
        "SELECT kind, page_number, content, image_path, needs_vision \
         FROM document_units WHERE document_id = $1",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;
    assert_eq!(
        unit,
        (
            "page".to_string(),
            1,
            Some("IL050 deployment table".to_string()),
            page_path,
            false
        )
    );
    {
        let requests = mock.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["input_type"], "passage");
        assert_eq!(requests[0]["modality"], serde_json::json!(["text"]));
        assert_eq!(
            requests[0]["input"],
            serde_json::json!(["IL050 deployment table"])
        );
    }

    server.abort();
    Ok(())
}

#[sqlx::test]
async fn failed_conversion_can_retry_without_partial_artifacts(pool: PgPool) -> anyhow::Result<()> {
    let storage = Storage::memory();
    let (document_id, storage_key) = store_original(
        &pool,
        &storage,
        "retry-after-failure.png",
        b"not an image".to_vec(),
    )
    .await?;

    let (models, mock, server) = spawn_embedding_mock(false).await;
    let state = test_state(&pool, models, storage.clone());
    crate::ingest::process_document(
        state.clone(),
        document_id,
        storage_key.clone(),
        crate::page_artifacts::SourceKind::Image,
    )
    .await;

    let failed_statuses: (String, String) =
        sqlx::query_as("SELECT status, artifact_status FROM documents WHERE id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(
        failed_statuses,
        ("failed".to_string(), "failed".to_string())
    );
    let failed_artifact_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM document_artifact_sets WHERE document_id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(failed_artifact_count, 0);
    assert_eq!(
        storage.list(&document_id.to_string()).await?,
        vec![storage_key.clone()],
        "failed run left partial artifacts"
    );

    storage
        .put(&storage_key, png_bytes(9, 5, [30, 60, 90, 255]))
        .await?;
    crate::ingest::process_document(
        state,
        document_id,
        storage_key,
        crate::page_artifacts::SourceKind::Image,
    )
    .await;

    let retry_statuses: (String, String) =
        sqlx::query_as("SELECT status, artifact_status FROM documents WHERE id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(retry_statuses, ("ready".to_string(), "ready".to_string()));
    let counts: (i64, i64, i64) = sqlx::query_as(
        "SELECT \
           (SELECT count(*) FROM document_artifact_sets WHERE document_id = $1), \
           (SELECT count(*) FROM page_artifacts WHERE document_id = $1), \
           (SELECT count(*) FROM document_units WHERE document_id = $1)",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;
    assert_eq!(counts, (1, 1, 1));
    assert_eq!(mock.requests.lock().expect("request lock").len(), 1);

    server.abort();
    Ok(())
}

#[sqlx::test]
async fn partial_artifact_upload_is_removed_and_never_referenced(
    pool: PgPool,
) -> anyhow::Result<()> {
    // Pages upload before the manifest, so failing the manifest leaves a
    // partial run in storage that must be removed before anything references it.
    let storage = Storage::memory().failing_puts_containing("/manifest.json");
    let (document_id, storage_key) = store_original(
        &pool,
        &storage,
        "partial.png",
        png_bytes(9, 5, [1, 2, 3, 255]),
    )
    .await?;
    let (models, mock, server) = spawn_embedding_mock(false).await;
    crate::ingest::process_document(
        test_state(&pool, models, storage.clone()),
        document_id,
        storage_key.clone(),
        crate::page_artifacts::SourceKind::Image,
    )
    .await;

    let (status, artifact_status, error): (String, String, Option<String>) =
        sqlx::query_as("SELECT status, artifact_status, error FROM documents WHERE id = $1")
            .bind(document_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(
        (status.as_str(), artifact_status.as_str()),
        ("failed", "failed")
    );
    assert!(
        error
            .as_deref()
            .is_some_and(|error| error.starts_with("artifact-publish")),
        "{error:?}"
    );
    let artifact_rows: i64 = sqlx::query_scalar(
        "SELECT (SELECT count(*) FROM document_artifact_sets WHERE document_id = $1) \
              + (SELECT count(*) FROM page_artifacts WHERE document_id = $1)",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;
    assert_eq!(artifact_rows, 0);
    assert_eq!(
        storage.list(&document_id.to_string()).await?,
        vec![storage_key]
    );
    assert!(mock.requests.lock().expect("request lock").is_empty());

    server.abort();
    Ok(())
}

#[sqlx::test]
async fn failed_publication_commit_keeps_the_previous_artifact_set_complete(
    pool: PgPool,
) -> anyhow::Result<()> {
    let storage = Storage::memory();
    let (document_id, storage_key) = store_original(
        &pool,
        &storage,
        "commit.png",
        png_bytes(10, 6, [5, 6, 7, 255]),
    )
    .await?;
    let (models, _mock, server) = spawn_embedding_mock(false).await;
    let state = test_state(&pool, models, storage.clone());
    crate::ingest::process_document(
        state.clone(),
        document_id,
        storage_key.clone(),
        crate::page_artifacts::SourceKind::Image,
    )
    .await;
    let published: String = sqlx::query_scalar(
        "SELECT artifact_dir FROM document_artifact_sets WHERE document_id = $1",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;

    // A deferred constraint trigger makes the next publication fail inside COMMIT.
    sqlx::raw_sql(
        "CREATE FUNCTION knowva_test_fail_commit() RETURNS trigger LANGUAGE plpgsql AS \
         $$ BEGIN RAISE EXCEPTION 'injected commit failure'; END $$; \
         CREATE CONSTRAINT TRIGGER knowva_test_fail_commit \
         AFTER INSERT ON document_artifact_sets \
         DEFERRABLE INITIALLY DEFERRED \
         FOR EACH ROW EXECUTE FUNCTION knowva_test_fail_commit();",
    )
    .execute(&pool)
    .await?;
    crate::ingest::process_document(
        state,
        document_id,
        storage_key.clone(),
        crate::page_artifacts::SourceKind::Image,
    )
    .await;

    let error: Option<String> = sqlx::query_scalar("SELECT error FROM documents WHERE id = $1")
        .bind(document_id)
        .fetch_one(&pool)
        .await?;
    assert!(
        error
            .as_deref()
            .is_some_and(|error| error.contains("injected commit failure")),
        "{error:?}"
    );
    let artifact_dir: String = sqlx::query_scalar(
        "SELECT artifact_dir FROM document_artifact_sets WHERE document_id = $1",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;
    assert_eq!(
        artifact_dir, published,
        "the rolled-back run must not replace it"
    );
    let referenced: Vec<String> = sqlx::query_scalar(
        "SELECT manifest_path FROM document_artifact_sets WHERE document_id = $1 \
         UNION ALL SELECT image_path FROM page_artifacts WHERE document_id = $1 \
         UNION ALL SELECT image_path FROM document_units \
                   WHERE document_id = $1 AND image_path IS NOT NULL",
    )
    .bind(document_id)
    .fetch_all(&pool)
    .await?;
    let stored = storage.list(&document_id.to_string()).await?;
    assert_eq!(referenced.len(), 3);
    for key in &referenced {
        assert!(
            stored.contains(key),
            "DB references a missing object: {key}"
        );
    }
    assert!(
        stored
            .iter()
            .all(|key| key == &storage_key || key.starts_with(&format!("{published}/"))),
        "the failed run must leave no objects: {stored:?}"
    );

    server.abort();
    Ok(())
}

#[cfg(unix)]
#[sqlx::test]
async fn delete_keeps_a_retry_handle_when_storage_cleanup_fails(
    pool: PgPool,
) -> anyhow::Result<()> {
    let files_dir = std::env::temp_dir().join(format!("knowva-delete-retry-{}", Uuid::new_v4()));
    let storage = Storage::local(&files_dir)?;
    let (document_id, _) =
        store_original(&pool, &storage, "cleanup.pdf", b"fixture".to_vec()).await?;
    // A regular file at the document directory path makes the local prefix
    // delete fail deterministically without relying on platform permissions.
    let document_dir = files_dir.join(document_id.to_string());
    std::fs::remove_dir_all(&document_dir)?;
    std::fs::write(&document_dir, b"not a directory")?;

    let app = api_router(test_state(&pool, CloudModelClient::for_test(), storage));
    let request = Request::builder()
        .method("DELETE")
        .uri(format!("/api/documents/{document_id}"))
        .body(Body::empty())?;
    let response = app.clone().oneshot(request).await?;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = to_bytes(response.into_body(), usize::MAX).await?;
    assert!(!String::from_utf8_lossy(&body).contains(files_dir.to_string_lossy().as_ref()));

    let document_count: i64 = sqlx::query_scalar("SELECT count(*) FROM documents WHERE id = $1")
        .bind(document_id)
        .fetch_one(&pool)
        .await?;
    assert_eq!(document_count, 1, "failed cleanup must remain retryable");

    let status: String = sqlx::query_scalar("SELECT status FROM documents WHERE id = $1")
        .bind(document_id)
        .fetch_one(&pool)
        .await?;
    assert_eq!(
        status, "failed",
        "a half-deleted document must leave retrieval"
    );

    std::fs::remove_file(&document_dir)?;
    let retry = Request::builder()
        .method("DELETE")
        .uri(format!("/api/documents/{document_id}"))
        .body(Body::empty())?;
    let response = app.oneshot(retry).await?;
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    let document_count: i64 = sqlx::query_scalar("SELECT count(*) FROM documents WHERE id = $1")
        .bind(document_id)
        .fetch_one(&pool)
        .await?;
    assert_eq!(document_count, 0);
    assert!(!document_dir.exists());

    tokio::fs::remove_dir_all(files_dir).await?;
    Ok(())
}

#[sqlx::test]
async fn markdown_and_text_fixtures_keep_heading_aware_staging(pool: PgPool) -> anyhow::Result<()> {
    let root = std::env::temp_dir().join(format!("knowva-text-fixtures-{}", Uuid::new_v4()));
    let fixtures = crate::test_fixtures::write_all(&root)?;
    let (models, mock, server) = spawn_embedding_mock(false).await;
    let state = test_state(&pool, models, Storage::memory());

    for (filename, fixture) in [
        ("headings.md", fixtures.markdown),
        ("plain.txt", fixtures.text),
        ("hybrid-retrieval.md", fixtures.hybrid_retrieval),
    ] {
        let (document_id, storage_key) = store_original(
            &pool,
            &state.storage,
            filename,
            tokio::fs::read(fixture).await?,
        )
        .await?;

        crate::ingest::process_document(
            state.clone(),
            document_id,
            storage_key,
            crate::page_artifacts::SourceKind::Text,
        )
        .await;
        let artifact_count: i64 =
            sqlx::query_scalar("SELECT count(*) FROM text_artifacts WHERE document_id = $1")
                .bind(document_id)
                .fetch_one(&pool)
                .await?;
        let page_count: i64 =
            sqlx::query_scalar("SELECT count(*) FROM page_artifacts WHERE document_id = $1")
                .bind(document_id)
                .fetch_one(&pool)
                .await?;
        let unit_count: i64 =
            sqlx::query_scalar("SELECT count(*) FROM document_units WHERE document_id = $1")
                .bind(document_id)
                .fetch_one(&pool)
                .await?;
        let statuses: (String, String) =
            sqlx::query_as("SELECT status, artifact_status FROM documents WHERE id = $1")
                .bind(document_id)
                .fetch_one(&pool)
                .await?;
        assert!(artifact_count >= 1);
        assert_eq!(page_count, 0);
        assert_eq!(unit_count, artifact_count);
        assert_eq!(statuses, ("ready".to_string(), "ready".to_string()));
    }

    {
        let requests = mock.requests.lock().expect("request lock");
        assert!(requests.len() >= 2);
        assert!(requests.iter().all(|request| {
            request["input_type"] == "passage"
                && request["modality"] == serde_json::json!(["text"])
                && request["input"][0]
                    .as_str()
                    .is_some_and(|input| !input.trim().is_empty())
        }));
    }

    server.abort();
    tokio::fs::remove_dir_all(root).await?;
    Ok(())
}

/// Upload -> ready -> answer (dense retrieval, VL rerank and RAG all read the
/// page image through storage) -> delete, for any storage backend.
async fn upload_answer_delete_flow(pool: &PgPool, storage: Storage) -> anyhow::Result<()> {
    let (models, mock, server) = spawn_embedding_mock(false).await;
    let app = api_router(test_state(pool, models, storage.clone()));

    let boundary = "knowva-storage-flow";
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; \
         filename=\"blue-page.png\"\r\nContent-Type: image/png\r\n\r\n"
    )
    .into_bytes();
    body.extend(png_bytes(24, 16, [40, 80, 160, 255]));
    body.extend(format!("\r\n--{boundary}--\r\n").into_bytes());
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/documents")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let uploaded: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await?)?;
    let document_id: Uuid = uploaded["id"].as_str().expect("upload id").parse()?;

    // Processing runs in the background; wait for its atomic ready publication.
    let mut status = String::from("processing");
    for _ in 0..400 {
        status = sqlx::query_scalar("SELECT status FROM documents WHERE id = $1")
            .bind(document_id)
            .fetch_one(pool)
            .await?;
        if status != "processing" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(status, "ready");
    let (storage_path, image_path): (String, String) = sqlx::query_as(
        "SELECT d.storage_path, u.image_path \
         FROM documents d JOIN document_units u ON u.document_id = d.id \
         WHERE d.id = $1",
    )
    .bind(document_id)
    .fetch_one(pool)
    .await?;
    assert_eq!(storage_path, format!("{document_id}/original.png"));
    assert!(
        image_path.starts_with(&format!("{document_id}/artifacts-"))
            && image_path.ends_with("/pages/0001.png"),
        "{image_path}"
    );
    let stored = storage.list(&document_id.to_string()).await?;
    assert!(
        stored.contains(&storage_path) && stored.contains(&image_path),
        "{stored:?}"
    );

    let question = serde_json::json!({
        "messages": [{ "role": "user", "content": "這一頁是什麼顏色？" }]
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat")
                .header("content-type", "application/json")
                .body(Body::from(question.to_string()))?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let body = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await?.to_vec())?;
    let events = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    let kinds = events
        .iter()
        .map(|event| event["type"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    assert_eq!(kinds, ["text-delta", "sources", "done"], "{body}");
    assert_eq!(
        events[1]["sources"][0]["documentId"],
        document_id.to_string()
    );
    assert_eq!(events[1]["sources"][0]["pageNumber"], 1);
    {
        let reranks = mock.rerank_requests.lock().expect("rerank request lock");
        assert_eq!(reranks.len(), 1);
        assert!(
            reranks[0]["passages"][0]["image"]
                .as_str()
                .is_some_and(|image| image.starts_with("data:image/png;base64,"))
        );
        let chats = mock.summary_requests.lock().expect("chat request lock");
        let answer_request = chats
            .iter()
            .find(|request| request["stream"] == true)
            .expect("streamed answer request");
        // The default profile has no industry section, so /api/chat sends the core rules
        // exactly as written.
        assert_eq!(answer_request["messages"][0]["role"], "system");
        assert_eq!(
            answer_request["messages"][0]["content"],
            crate::rag_context::RAG_SYSTEM_PROMPT
        );
        let answer = answer_request.to_string();
        assert!(answer.contains("data:image/jpeg;base64,"));
        assert!(
            !answer.contains(&image_path),
            "keys must never reach the model"
        );
    }

    let response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/documents/{document_id}"))
                .body(Body::empty())?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert!(storage.list(&document_id.to_string()).await?.is_empty());
    let remaining: i64 = sqlx::query_scalar("SELECT count(*) FROM documents WHERE id = $1")
        .bind(document_id)
        .fetch_one(pool)
        .await?;
    assert_eq!(remaining, 0);

    server.abort();
    Ok(())
}

#[sqlx::test]
async fn upload_answer_delete_flow_with_local_storage(pool: PgPool) -> anyhow::Result<()> {
    let files_dir = std::env::temp_dir().join(format!("knowva-flow-local-{}", Uuid::new_v4()));
    upload_answer_delete_flow(&pool, Storage::local(&files_dir)?).await?;
    assert_eq!(
        std::fs::read_dir(&files_dir)?.count(),
        0,
        "delete must not leave an empty document directory"
    );
    std::fs::remove_dir_all(files_dir)?;
    Ok(())
}

#[sqlx::test]
async fn upload_answer_delete_flow_with_in_memory_storage(pool: PgPool) -> anyhow::Result<()> {
    upload_answer_delete_flow(&pool, Storage::memory()).await
}

/// Opt-in smoke against a real S3-compatible endpoint; the command is in README.
#[sqlx::test]
#[ignore = "requires MinIO/S3 via STORAGE_BACKEND=s3 and STORAGE_*"]
async fn minio_smoke_upload_answer_delete(pool: PgPool) -> anyhow::Result<()> {
    let config = StorageConfig::from_env()?;
    anyhow::ensure!(
        matches!(config, StorageConfig::S3 { .. }),
        "set STORAGE_BACKEND=s3 for the MinIO smoke test"
    );
    let storage = config.open().await?;
    ensure_initialization_lock(&pool, &config).await?;
    upload_answer_delete_flow(&pool, storage).await
}

#[sqlx::test]
async fn first_start_records_the_backend_and_only_files_dir_may_change(
    pool: PgPool,
) -> anyhow::Result<()> {
    ensure_initialization_lock(
        &pool,
        &StorageConfig::Local {
            files_dir: "/srv/knowva/files".into(),
        },
    )
    .await?;
    let recorded: (String, Option<String>, Option<String>) =
        sqlx::query_as("SELECT backend, s3_bucket, s3_prefix FROM storage_settings")
            .fetch_one(&pool)
            .await?;
    assert_eq!(recorded, ("local".to_owned(), None, None));

    // FILES_DIR is a mount point, not part of the stored keys.
    ensure_initialization_lock(
        &pool,
        &StorageConfig::Local {
            files_dir: "/mnt/other-volume".into(),
        },
    )
    .await?;

    let error = ensure_initialization_lock(
        &pool,
        &StorageConfig::S3 {
            bucket: "knowva-files".to_owned(),
            prefix: None,
        },
    )
    .await
    .expect_err("backend change must fail")
    .to_string();
    assert!(error.contains("儲存後端已於初始化時鎖定，變更需重新初始化 DB 與儲存空間"));
    assert!(error.contains("已鎖定：backend=local"), "{error}");
    assert!(
        error.contains("目前設定：backend=s3, bucket=knowva-files"),
        "{error}"
    );
    let rows: i64 = sqlx::query_scalar("SELECT count(*) FROM storage_settings")
        .fetch_one(&pool)
        .await?;
    assert_eq!(rows, 1);
    Ok(())
}

#[sqlx::test]
async fn s3_lock_covers_bucket_and_prefix_but_not_the_endpoint(pool: PgPool) -> anyhow::Result<()> {
    let s3 = |bucket: &str, prefix: Option<&str>| StorageConfig::S3 {
        bucket: bucket.to_owned(),
        prefix: prefix.map(str::to_owned),
    };
    ensure_initialization_lock(&pool, &s3("knowva-files", Some("tenant-a"))).await?;
    // The endpoint is a client setting outside the lock, so moving endpoints
    // restarts with the identical recorded identity.
    ensure_initialization_lock(&pool, &s3("knowva-files", Some("tenant-a"))).await?;
    for changed in [
        s3("knowva-other", Some("tenant-a")),
        s3("knowva-files", Some("tenant-b")),
        s3("knowva-files", None),
        StorageConfig::Local {
            files_dir: "./data/files".into(),
        },
    ] {
        let error = ensure_initialization_lock(&pool, &changed)
            .await
            .expect_err("changed storage identity must fail");
        assert!(
            error
                .to_string()
                .contains("儲存後端已於初始化時鎖定，變更需重新初始化 DB 與儲存空間")
        );
    }
    Ok(())
}

#[sqlx::test]
async fn unlocked_database_with_documents_requires_reinitialization(
    pool: PgPool,
) -> anyhow::Result<()> {
    insert_document(&pool, "ready").await?;
    let error = ensure_initialization_lock(
        &pool,
        &StorageConfig::Local {
            files_dir: "./data/files".into(),
        },
    )
    .await
    .expect_err("pre-upgrade data must not be adopted");
    assert!(error.to_string().contains("需清空 DB 與檔案後重新初始化"));
    let rows: i64 = sqlx::query_scalar("SELECT count(*) FROM storage_settings")
        .fetch_one(&pool)
        .await?;
    assert_eq!(rows, 0);
    Ok(())
}

// Conversation history (issue #52).

struct ConversationReply {
    status: StatusCode,
    set_cookie: Option<String>,
    body: Value,
}

impl ConversationReply {
    /// The `knowva_client` token this reply issued.
    fn issued_token(&self) -> String {
        self.set_cookie
            .as_deref()
            .and_then(|cookie| cookie.strip_prefix("knowva_client="))
            .and_then(|rest| rest.split(';').next())
            .expect("the reply issues a client cookie")
            .to_owned()
    }
}

/// One conversation API call as the browser holding `token` would make it.
async fn conversation_call(
    app: &Router,
    method: &str,
    uri: &str,
    token: Option<&str>,
    body: Option<Value>,
) -> anyhow::Result<ConversationReply> {
    let mut request = Request::builder().method(method).uri(uri);
    if let Some(token) = token {
        request = request.header("cookie", format!("knowva_client={token}"));
    }
    let request = match body {
        Some(body) => request
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body)?))?,
        None => request.body(Body::empty())?,
    };
    let response = app.clone().oneshot(request).await?;
    let status = response.status();
    let set_cookie = response
        .headers()
        .get("set-cookie")
        .map(|value| value.to_str().map(str::to_owned))
        .transpose()?;
    let bytes = to_bytes(response.into_body(), usize::MAX).await?;
    let body = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
    };
    Ok(ConversationReply {
        status,
        set_cookie,
        body,
    })
}

async fn create_conversation_for(app: &Router, token: &str) -> anyhow::Result<String> {
    let reply = conversation_call(app, "POST", "/api/conversations", Some(token), None).await?;
    assert_eq!(reply.status, StatusCode::CREATED, "{:?}", reply.body);
    Ok(reply.body["conversation"]["id"]
        .as_str()
        .expect("conversation id")
        .to_owned())
}

fn user_message(id: &str, text: &str) -> Value {
    serde_json::json!({
        "id": id,
        "role": "user",
        "createdAt": "2026-10-06T08:00:00.000Z",
        "content": [{ "type": "text", "text": text }],
        "attachments": [],
        "metadata": { "custom": {} }
    })
}

fn assistant_message(id: &str, text: &str, sources: Value) -> Value {
    serde_json::json!({
        "id": id,
        "role": "assistant",
        "createdAt": "2026-10-06T08:00:05.000Z",
        "content": [
            { "type": "text", "text": text },
            { "type": "data", "name": "sources", "data": sources }
        ],
        "status": { "type": "complete", "reason": "unknown" },
        "metadata": {
            "unstable_state": null,
            "unstable_annotations": [],
            "unstable_data": [],
            "steps": [],
            "custom": {}
        }
    })
}

async fn save_conversation_message(
    app: &Router,
    token: &str,
    conversation_id: &str,
    parent_id: Option<&str>,
    message: Value,
) -> anyhow::Result<ConversationReply> {
    conversation_call(
        app,
        "POST",
        &format!("/api/conversations/{conversation_id}/messages"),
        Some(token),
        Some(serde_json::json!({ "parentId": parent_id, "message": message })),
    )
    .await
}

fn conversation_ids(reply: &ConversationReply) -> Vec<String> {
    reply
        .body
        .as_array()
        .expect("conversation list")
        .iter()
        .map(|conversation| conversation["id"].as_str().expect("id").to_owned())
        .collect()
}

#[sqlx::test]
async fn conversations_keep_every_message_part_in_save_order(pool: PgPool) -> anyhow::Result<()> {
    let app = api_router(test_state(
        &pool,
        CloudModelClient::for_test(),
        Storage::memory(),
    ));
    let token = crate::conversations::test_support::new_token();
    let id = create_conversation_for(&app, &token).await?;
    let sources = serde_json::json!([
        { "sourceId": "S1", "documentId": "00000000-0000-0000-0000-000000000001",
          "unitId": 137, "unitIndex": 36, "filename": "設備保養手冊.pdf",
          "pageNumber": 37, "kind": "page" },
        { "sourceId": "S2", "documentId": "00000000-0000-0000-0000-000000000002",
          "unitId": 202, "filename": "維運交接筆記.md", "pageNumber": null, "kind": "text" }
    ]);
    let question = user_message("q1", "馬達多久保養一次？");
    let answer = assistant_message("a1", "每三個月保養一次。", sources);
    let mut failed = assistant_message("a2", "請先", serde_json::json!([]));
    failed["content"] = serde_json::json!([{ "type": "text", "text": "請先" }]);
    failed["status"] = serde_json::json!({
        "type": "incomplete", "reason": "error", "error": "回覆傳輸中斷，請重新送出問題。"
    });
    let follow_up = user_message("q2", "交接要注意什麼？");
    for (parent_id, message) in [
        (None, &question),
        (Some("q1"), &answer),
        (Some("a1"), &follow_up),
        (Some("q2"), &failed),
    ] {
        let reply =
            save_conversation_message(&app, &token, &id, parent_id, message.clone()).await?;
        assert_eq!(reply.status, StatusCode::OK, "{:?}", reply.body);
    }

    let reply = conversation_call(
        &app,
        "GET",
        &format!("/api/conversations/{id}/messages"),
        Some(&token),
        None,
    )
    .await?;
    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(
        reply.body,
        serde_json::json!([
            { "parentId": null, "message": question },
            { "parentId": "q1", "message": answer },
            { "parentId": "a1", "message": follow_up },
            { "parentId": "q2", "message": failed }
        ])
    );
    assert_eq!(reply.set_cookie, None, "a valid cookie is never replaced");
    Ok(())
}

#[sqlx::test]
async fn conversations_of_another_browser_answer_404(pool: PgPool) -> anyhow::Result<()> {
    let app = api_router(test_state(
        &pool,
        CloudModelClient::for_test(),
        Storage::memory(),
    ));
    let alice = crate::conversations::test_support::new_token();
    let bob = crate::conversations::test_support::new_token();
    let alices = create_conversation_for(&app, &alice).await?;
    let saved =
        save_conversation_message(&app, &alice, &alices, None, user_message("q1", "甲的問題"))
            .await?;
    assert_eq!(saved.status, StatusCode::OK);
    let bobs = create_conversation_for(&app, &bob).await?;

    let listed = conversation_call(&app, "GET", "/api/conversations", Some(&bob), None).await?;
    assert_eq!(conversation_ids(&listed), std::slice::from_ref(&bobs));
    for (method, uri, body) in [
        ("GET", format!("/api/conversations/{alices}"), None),
        ("GET", format!("/api/conversations/{alices}/messages"), None),
        (
            "POST",
            format!("/api/conversations/{alices}/messages"),
            Some(
                serde_json::json!({ "parentId": "q1", "message": user_message("q2", "乙的問題") }),
            ),
        ),
        ("DELETE", format!("/api/conversations/{alices}"), None),
    ] {
        let reply = conversation_call(&app, method, &uri, Some(&bob), body).await?;
        assert_eq!(reply.status, StatusCode::NOT_FOUND, "{method} {uri}");
        assert_eq!(reply.body, Value::String("找不到這串對話。".to_owned()));
    }
    let missing = conversation_call(
        &app,
        "GET",
        &format!("/api/conversations/{}", Uuid::new_v4()),
        Some(&alice),
        None,
    )
    .await?;
    assert_eq!(missing.status, StatusCode::NOT_FOUND);
    assert_eq!(missing.body, Value::String("找不到這串對話。".to_owned()));
    let malformed = conversation_call(
        &app,
        "GET",
        "/api/conversations/not-a-uuid",
        Some(&alice),
        None,
    )
    .await?;
    assert_eq!(malformed.status, StatusCode::NOT_FOUND);

    let cleared = conversation_call(&app, "DELETE", "/api/conversations", Some(&bob), None).await?;
    assert_eq!(cleared.status, StatusCode::NO_CONTENT);
    let alices_list =
        conversation_call(&app, "GET", "/api/conversations", Some(&alice), None).await?;
    assert_eq!(
        conversation_ids(&alices_list),
        std::slice::from_ref(&alices)
    );
    let messages: i64 = sqlx::query_scalar("SELECT count(*) FROM conversation_messages")
        .fetch_one(&pool)
        .await?;
    assert_eq!(
        messages, 1,
        "clearing one browser keeps the other's messages"
    );

    let deleted = conversation_call(
        &app,
        "DELETE",
        &format!("/api/conversations/{alices}"),
        Some(&alice),
        None,
    )
    .await?;
    assert_eq!(deleted.status, StatusCode::NO_CONTENT);
    let remaining: i64 = sqlx::query_scalar("SELECT count(*) FROM conversations")
        .fetch_one(&pool)
        .await?;
    let remaining_messages: i64 = sqlx::query_scalar("SELECT count(*) FROM conversation_messages")
        .fetch_one(&pool)
        .await?;
    assert_eq!((remaining, remaining_messages), (0, 0));
    Ok(())
}

/// Collects everything the fmt subscriber writes while it is the default.
#[derive(Clone, Default)]
struct CapturedLog(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for CapturedLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("log lock").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
    type Writer = CapturedLog;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[sqlx::test]
async fn the_client_token_never_reaches_the_log(pool: PgPool) -> anyhow::Result<()> {
    let log = CapturedLog::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(log.clone())
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .finish();
    let _default = tracing::subscriber::set_default(subscriber);
    let app = api_router(test_state(
        &pool,
        CloudModelClient::for_test(),
        Storage::memory(),
    ));

    let created = conversation_call(&app, "POST", "/api/conversations", None, None).await?;
    assert_eq!(created.status, StatusCode::CREATED);
    let token = created.issued_token();
    let id = created.body["conversation"]["id"]
        .as_str()
        .expect("id")
        .to_owned();
    for message in [
        user_message("q1", "權杖不能外流嗎？"),
        // Invalid, oversized and misplaced saves go through the error paths too.
        serde_json::json!({ "id": "bad", "role": "system", "content": [] }),
    ] {
        save_conversation_message(&app, &token, &id, None, message).await?;
    }
    save_conversation_message(&app, &token, &id, Some("missing"), user_message("q9", "x")).await?;
    conversation_call(&app, "GET", "/api/conversations", Some(&token), None).await?;
    conversation_call(
        &app,
        "GET",
        &format!("/api/conversations/{}", Uuid::new_v4()),
        Some(&token),
        None,
    )
    .await?;
    // A dropped connection mid-request still must not log the token.
    pool.close().await;
    let unavailable =
        conversation_call(&app, "GET", "/api/conversations", Some(&token), None).await?;
    assert_eq!(unavailable.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        unavailable.body,
        Value::String("對話紀錄暫時無法使用，請稍後再試。".to_owned())
    );

    let log = String::from_utf8(log.0.lock().expect("log lock").clone())?;
    assert!(log.contains("issued an anonymous client cookie"), "{log}");
    assert!(log.contains("conversation query failed"), "{log}");
    assert!(!log.contains(&token), "the log holds the client token");

    Ok(())
}

#[sqlx::test]
async fn conversations_store_only_the_token_hash_as_an_anonymous_owner(
    pool: PgPool,
) -> anyhow::Result<()> {
    let app = api_router(test_state(
        &pool,
        CloudModelClient::for_test(),
        Storage::memory(),
    ));
    let created = conversation_call(&app, "POST", "/api/conversations", None, None).await?;
    let token = created.issued_token();
    let id = created.body["conversation"]["id"]
        .as_str()
        .expect("id")
        .to_owned();
    save_conversation_message(&app, &token, &id, None, user_message("q1", "第一個問題")).await?;
    create_conversation_for(&app, &crate::conversations::test_support::new_token()).await?;

    let owners: Vec<(String, String)> =
        sqlx::query_as("SELECT owner_kind, owner_id FROM conversations ORDER BY created_at")
            .fetch_all(&pool)
            .await?;
    assert_eq!(owners.len(), 2);
    assert_eq!(
        owners[0].1,
        crate::conversations::test_support::hash(&token)
    );
    assert!(owners.iter().all(|(kind, _)| kind == "anonymous"));
    let rows_with_token: i64 = sqlx::query_scalar(
        "SELECT (SELECT count(*) FROM conversations c WHERE c::text LIKE '%' || $1 || '%') \
              + (SELECT count(*) FROM conversation_messages m WHERE m::text LIKE '%' || $1 || '%')",
    )
    .bind(&token)
    .fetch_one(&pool)
    .await?;
    assert_eq!(rows_with_token, 0);

    // The schema refuses an unknown owner kind and an anonymous owner that is not a hash.
    for (kind, owner_id) in [
        ("robot", crate::conversations::test_support::hash(&token)),
        ("anonymous", token.clone()),
    ] {
        let insert =
            sqlx::query("INSERT INTO conversations (owner_kind, owner_id) VALUES ($1, $2)")
                .bind(kind)
                .bind(&owner_id)
                .execute(&pool)
                .await;
        assert!(insert.is_err(), "{kind} {owner_id} must be rejected");
    }
    sqlx::query("INSERT INTO conversations (owner_kind, owner_id) VALUES ('user', 'account-42')")
        .execute(&pool)
        .await?;
    Ok(())
}

#[sqlx::test]
async fn the_51st_conversation_deletes_the_least_recently_updated(
    pool: PgPool,
) -> anyhow::Result<()> {
    let app = api_router(test_state(
        &pool,
        CloudModelClient::for_test(),
        Storage::memory(),
    ));
    let token = crate::conversations::test_support::new_token();
    let other = crate::conversations::test_support::new_token();
    let others = create_conversation_for(&app, &other).await?;
    let mut ids = Vec::new();
    for _ in 0..crate::conversations::MAX_CONVERSATIONS {
        ids.push(create_conversation_for(&app, &token).await?);
    }
    // The second one gets a message so its messages must go with it; the first one is
    // updated last, so the second is now the stalest.
    save_conversation_message(&app, &token, &ids[1], None, user_message("q1", "舊的問題")).await?;
    save_conversation_message(
        &app,
        &token,
        &ids[0],
        None,
        user_message("q1", "最近的問題"),
    )
    .await?;
    let stalest = ids[2].clone();

    let created = conversation_call(&app, "POST", "/api/conversations", Some(&token), None).await?;
    assert_eq!(created.status, StatusCode::CREATED);
    assert_eq!(created.body["removedIds"], serde_json::json!([stalest]));
    let newest = created.body["conversation"]["id"]
        .as_str()
        .expect("id")
        .to_owned();

    let listed = conversation_call(&app, "GET", "/api/conversations", Some(&token), None).await?;
    let listed = conversation_ids(&listed);
    assert_eq!(listed.len(), 50);
    assert_eq!(listed[0], newest);
    assert_eq!(listed[1], ids[0]);
    assert!(!listed.contains(&stalest));
    assert!(listed.contains(&ids[1]));
    let others_list =
        conversation_call(&app, "GET", "/api/conversations", Some(&other), None).await?;
    assert_eq!(conversation_ids(&others_list), [others]);

    // Once more: now the third-created one goes, taking its messages with it.
    save_conversation_message(&app, &token, &ids[3], None, user_message("q1", "x")).await?;
    let created = conversation_call(&app, "POST", "/api/conversations", Some(&token), None).await?;
    assert_eq!(created.body["removedIds"], serde_json::json!([ids[4]]));
    let total: i64 = sqlx::query_scalar("SELECT count(*) FROM conversations WHERE owner_id = $1")
        .bind(crate::conversations::test_support::hash(&token))
        .fetch_one(&pool)
        .await?;
    assert_eq!(total, 50);
    Ok(())
}

#[sqlx::test]
async fn conversation_messages_are_capped_checked_and_idempotent(
    pool: PgPool,
) -> anyhow::Result<()> {
    let app = api_router(test_state(
        &pool,
        CloudModelClient::for_test(),
        Storage::memory(),
    ));
    let token = crate::conversations::test_support::new_token();
    let id = create_conversation_for(&app, &token).await?;

    let limit = crate::conversations::MAX_MESSAGES_PER_CONVERSATION as usize;
    let mut parent: Option<String> = None;
    for index in 0..limit {
        let message_id = format!("m{index}");
        let message = if index % 2 == 0 {
            user_message(&message_id, &format!("問題 {index}"))
        } else {
            assistant_message(&message_id, &format!("回答 {index}"), serde_json::json!([]))
        };
        let reply =
            save_conversation_message(&app, &token, &id, parent.as_deref(), message).await?;
        assert_eq!(
            reply.status,
            StatusCode::OK,
            "message {index}: {:?}",
            reply.body
        );
        parent = Some(message_id);
    }

    let over = save_conversation_message(
        &app,
        &token,
        &id,
        parent.as_deref(),
        user_message("m-over", "第 201 則"),
    )
    .await?;
    assert_eq!(over.status, StatusCode::CONFLICT);
    assert_eq!(
        over.body,
        Value::String("這串對話已達 200 則訊息，無法再保存。".to_owned())
    );

    // Resending a saved id replaces it in place, even at the limit.
    let resent = save_conversation_message(
        &app,
        &token,
        &id,
        None,
        user_message("m0", "改寫過的第一個問題"),
    )
    .await?;
    assert_eq!(resent.status, StatusCode::OK);
    let stored: Vec<(String, i32, Value)> = sqlx::query_as(
        "SELECT message_id, position, message FROM conversation_messages \
         WHERE conversation_id = $1::uuid ORDER BY position",
    )
    .bind(&id)
    .fetch_all(&pool)
    .await?;
    assert_eq!(stored.len(), limit);
    assert_eq!(stored[0].0, "m0");
    assert_eq!(stored[0].1, 0);
    assert_eq!(stored[0].2["content"][0]["text"], "改寫過的第一個問題");
    assert!(
        stored
            .iter()
            .enumerate()
            .all(|(index, row)| row.1 == index as i32)
    );

    let fresh = create_conversation_for(&app, &token).await?;
    let missing_parent = save_conversation_message(
        &app,
        &token,
        &fresh,
        Some("nowhere"),
        user_message("q1", "x"),
    )
    .await?;
    assert_eq!(missing_parent.status, StatusCode::CONFLICT);
    let invalid = save_conversation_message(
        &app,
        &token,
        &fresh,
        None,
        serde_json::json!({ "id": "q1", "role": "tool", "content": [] }),
    )
    .await?;
    assert_eq!(invalid.status, StatusCode::BAD_REQUEST);
    assert_eq!(
        invalid.body,
        Value::String("訊息格式不正確，無法保存。".to_owned())
    );
    let not_json = conversation_call(
        &app,
        "POST",
        &format!("/api/conversations/{fresh}/messages"),
        Some(&token),
        Some(Value::String("not an object".to_owned())),
    )
    .await?;
    assert_eq!(not_json.status, StatusCode::BAD_REQUEST);

    let max = crate::conversations::MAX_MESSAGE_BYTES;
    let mut at_limit = user_message("q1", "");
    let overhead = serde_json::to_vec(&at_limit)?.len();
    at_limit["content"][0]["text"] = Value::String("a".repeat(max - overhead));
    assert_eq!(serde_json::to_vec(&at_limit)?.len(), max);
    let accepted = save_conversation_message(&app, &token, &fresh, None, at_limit).await?;
    assert_eq!(accepted.status, StatusCode::OK);
    for extra in [1, 64 * 1024] {
        let mut too_large = user_message("q2", "");
        too_large["content"][0]["text"] = Value::String("a".repeat(max - overhead + extra));
        let rejected =
            save_conversation_message(&app, &token, &fresh, Some("q1"), too_large).await?;
        assert_eq!(
            rejected.status,
            StatusCode::PAYLOAD_TOO_LARGE,
            "+{extra} bytes"
        );
        assert_eq!(
            rejected.body,
            Value::String("單則訊息超過 256 KiB，無法保存。".to_owned())
        );
    }
    let count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM conversation_messages WHERE conversation_id = $1::uuid",
    )
    .bind(&fresh)
    .fetch_one(&pool)
    .await?;
    assert_eq!(count, 1);
    Ok(())
}

#[sqlx::test]
async fn the_first_question_titles_a_conversation_without_any_model_request(
    pool: PgPool,
) -> anyhow::Result<()> {
    let (models, mock, server) = spawn_embedding_mock(false).await;
    let app = api_router(test_state(&pool, models, Storage::memory()));
    let token = crate::conversations::test_support::new_token();
    let id = create_conversation_for(&app, &token).await?;

    let question = "  𠀋𡈽三號廠區\n\n冷卻水塔今年第三季有哪些異常紀錄、各自如何處理、負責人是誰？";
    let saved =
        save_conversation_message(&app, &token, &id, None, user_message("q1", question)).await?;
    let title = "𠀋𡈽三號廠區 冷卻水塔今年第三季有哪些異常紀錄、各自如何處理";
    assert_eq!(title.chars().count(), 30);
    assert_eq!(saved.body["conversation"]["title"], title);

    for (parent, message) in [
        ("q1", assistant_message("a1", "回答", serde_json::json!([]))),
        ("a1", user_message("q2", "第二個問題不會改標題")),
    ] {
        let reply = save_conversation_message(&app, &token, &id, Some(parent), message).await?;
        assert_eq!(reply.body["conversation"]["title"], title);
    }
    let fetched = conversation_call(
        &app,
        "GET",
        &format!("/api/conversations/{id}"),
        Some(&token),
        None,
    )
    .await?;
    assert_eq!(fetched.body["title"], title);

    for requests in [
        &mock.requests,
        &mock.summary_requests,
        &mock.rerank_requests,
    ] {
        assert!(
            requests.lock().expect("mock requests").is_empty(),
            "titling must not reach a model endpoint"
        );
    }
    server.abort();
    Ok(())
}
