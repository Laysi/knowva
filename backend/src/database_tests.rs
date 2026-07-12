use super::*;
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
    state
        .summary_requests
        .lock()
        .expect("summary request lock")
        .push(body);
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
    };
    let app = Router::new()
        .route("/embeddings", post(embedding_mock))
        .route("/chat/completions", post(summary_mock))
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
    let client = CloudModelClient::for_test_endpoints(&chat_endpoint, &embed_endpoint, timeout);
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
         (document_id, kind, unit_index, content, embedding) \
         VALUES ($1, 'text', $2, $3, $4) RETURNING id",
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
         (document_id, kind, unit_index, page_number, content, image_path, embedding) \
         VALUES ($1, 'page', $2, $3, $4, $5, $6) RETURNING id",
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
async fn unit_constraints_dimensions_and_cascade_are_enforced(pool: PgPool) -> anyhow::Result<()> {
    let document_id = insert_document(&pool, "processing").await?;
    let embedding = vector_with(&[(0, 1.0)]);

    sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, page_number, image_path, embedding) \
         VALUES ($1, 'page', 0, 1, '/pages/1.png', $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await?;
    insert_text_unit(&pool, document_id, 0, "raw text", &embedding).await?;

    let page_zero = sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, page_number, image_path, embedding) \
         VALUES ($1, 'page', 1, 0, '/pages/0.png', $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await;
    assert!(page_zero.is_err());

    let page_without_number = sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, image_path, embedding) \
         VALUES ($1, 'page', 1, '/pages/missing-number.png', $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await;
    assert!(page_without_number.is_err());

    let page_without_image = sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, page_number, embedding) \
         VALUES ($1, 'page', 1, 2, $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await;
    assert!(page_without_image.is_err());

    let text_without_content = sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, embedding) \
         VALUES ($1, 'text', 1, $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await;
    assert!(text_without_content.is_err());

    let duplicate_page_number = sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, page_number, image_path, embedding) \
         VALUES ($1, 'page', 1, 1, '/pages/duplicate.png', $2)",
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
         (document_id, kind, unit_index, content, embedding) \
         VALUES ($1, 'text', 2, 'wrong dimension', $2)",
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
         (document_id, page_number, image_path, width, height) \
         VALUES ($1, 1, '/files/run/pages/0001.png', 100, 200)",
    )
    .bind(document_id)
    .execute(&pool)
    .await?;

    for statement in [
        "INSERT INTO page_artifacts (document_id, page_number, image_path, width, height) VALUES ($1, 0, '/bad.png', 1, 1)",
        "INSERT INTO page_artifacts (document_id, page_number, image_path, width, height) VALUES ($1, 2, '', 1, 1)",
        "INSERT INTO page_artifacts (document_id, page_number, image_path, width, height) VALUES ($1, 2, '/bad.png', 0, 1)",
        "INSERT INTO page_artifacts (document_id, page_number, image_path, width, height) VALUES ($1, 1, '/duplicate.png', 1, 1)",
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
        "SELECT km_bm25_tokenize('知識，管理！PostgreSQL／ＢＭ２５ 版本１２３ ＮＸ－２０４８－Ｖ２')",
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
        sqlx::query_scalar("SELECT km_bm25_tokenize('ＮＸ－２０４８－Ｖ２')")
            .fetch_one(&pool)
            .await?;
    let half_width: Vec<String> = sqlx::query_scalar("SELECT km_bm25_tokenize('nx-2048-v2')")
        .fetch_one(&pool)
        .await?;
    assert_eq!(full_width, half_width);

    let punctuation_variant: Vec<String> =
        sqlx::query_scalar("SELECT km_bm25_tokenize('知識。管理 PostgreSQL, BM25; 123')")
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
        "SELECT document_count, total_length FROM km_bm25_corpus_stats WHERE singleton",
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
        "SELECT document_count, total_length FROM km_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(ready_stats, (1, i64::from(first_length)));
    let alpha_df: i64 = sqlx::query_scalar(
        "SELECT document_frequency FROM km_bm25_term_stats WHERE term = 'alpha'",
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
        "SELECT document_count, total_length FROM km_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(changed_stats, (1, i64::from(changed_length)));
    let alpha_df: Option<i64> = sqlx::query_scalar(
        "SELECT document_frequency FROM km_bm25_term_stats WHERE term = 'alpha'",
    )
    .fetch_optional(&pool)
    .await?;
    assert!(alpha_df.is_none());
    let beta_df: i64 =
        sqlx::query_scalar("SELECT document_frequency FROM km_bm25_term_stats WHERE term = 'beta'")
            .fetch_one(&pool)
            .await?;
    assert_eq!(beta_df, 1);

    let second_document = insert_document(&pool, "ready").await?;
    insert_text_unit(&pool, second_document, 0, "beta gamma", &embedding).await?;
    let two_document_stats: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM km_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(two_document_stats.0, 2);
    let beta_df: i64 =
        sqlx::query_scalar("SELECT document_frequency FROM km_bm25_term_stats WHERE term = 'beta'")
            .fetch_one(&pool)
            .await?;
    assert_eq!(beta_df, 2);

    sqlx::query("UPDATE documents SET status = 'failed' WHERE id = $1")
        .bind(first_document)
        .execute(&pool)
        .await?;
    let failed_stats: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM km_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(failed_stats.0, 1);
    let beta_df: i64 =
        sqlx::query_scalar("SELECT document_frequency FROM km_bm25_term_stats WHERE term = 'beta'")
            .fetch_one(&pool)
            .await?;
    assert_eq!(beta_df, 1);

    sqlx::query("UPDATE documents SET status = 'ready' WHERE id = $1")
        .bind(first_document)
        .execute(&pool)
        .await?;
    let restored_count: i64 =
        sqlx::query_scalar("SELECT document_count FROM km_bm25_corpus_stats WHERE singleton")
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
        sqlx::query_scalar("SELECT document_count FROM km_bm25_corpus_stats WHERE singleton")
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
        sqlx::query_scalar("SELECT document_count FROM km_bm25_corpus_stats WHERE singleton")
            .fetch_one(&pool)
            .await?;
    assert_eq!(canonical_count, 1);

    sqlx::query("UPDATE document_units SET document_id = $1 WHERE id = $2")
        .bind(first_document)
        .bind(first_unit)
        .execute(&pool)
        .await?;
    let moved_back_count: i64 =
        sqlx::query_scalar("SELECT document_count FROM km_bm25_corpus_stats WHERE singleton")
            .fetch_one(&pool)
            .await?;
    assert_eq!(moved_back_count, 2);

    sqlx::query("DELETE FROM documents WHERE id = $1")
        .bind(second_document)
        .execute(&pool)
        .await?;
    let after_first_delete: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM km_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(after_first_delete, (1, i64::from(changed_length)));

    sqlx::query("DELETE FROM documents WHERE id = $1")
        .bind(first_document)
        .execute(&pool)
        .await?;
    let final_stats: (i64, i64) = sqlx::query_as(
        "SELECT document_count, total_length FROM km_bm25_corpus_stats WHERE singleton",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(final_stats, (0, 0));
    let term_count: i64 = sqlx::query_scalar("SELECT count(*) FROM km_bm25_term_stats")
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
           FROM unnest(km_bm25_tokenize($1)) AS token \
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
    let files_dir = std::env::temp_dir().join(format!("km-server-{document_id}"));
    let document_dir = files_dir.join(document_id.to_string());
    let artifact_dir = document_dir.join("artifacts-fixture");
    let stale_temp_dir = document_dir.join(".artifacts-stale.tmp");
    tokio::fs::create_dir_all(artifact_dir.join("pages")).await?;
    tokio::fs::create_dir_all(&stale_temp_dir).await?;
    let storage_path = document_dir.join("original.pdf");
    let manifest_path = artifact_dir.join("manifest.json");
    let page_path = artifact_dir.join("pages/0001.png");
    tokio::fs::write(&storage_path, b"test document").await?;
    tokio::fs::write(&manifest_path, b"{}").await?;
    tokio::fs::write(&page_path, b"fixture").await?;
    tokio::fs::write(stale_temp_dir.join("partial.png"), b"partial").await?;

    sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, artifact_status, error) \
         VALUES ($1, 'api.pdf', $2, 'application/pdf', 'processing', 'ready', \
                 'provider-secret Authorization: Bearer should-never-leak')",
    )
    .bind(document_id)
    .bind(storage_path.to_string_lossy().as_ref())
    .execute(&pool)
    .await?;
    sqlx::query(
        "INSERT INTO document_artifact_sets \
         (document_id, source_kind, artifact_dir, manifest_path) \
         VALUES ($1, 'pdf', $2, $3)",
    )
    .bind(document_id)
    .bind(artifact_dir.to_string_lossy().as_ref())
    .bind(manifest_path.to_string_lossy().as_ref())
    .execute(&pool)
    .await?;
    sqlx::query(
        "INSERT INTO page_artifacts \
         (document_id, page_number, image_path, width, height, extracted_text) \
         VALUES ($1, 1, $2, 640, 480, 'native text')",
    )
    .bind(document_id)
    .bind(page_path.to_string_lossy().as_ref())
    .execute(&pool)
    .await?;

    let embedding = vector_with(&[(0, 1.0)]);
    sqlx::query(
        "INSERT INTO document_units \
         (document_id, kind, unit_index, page_number, image_path, embedding) \
         VALUES ($1, 'page', 0, 1, '/pages/1.png', $2)",
    )
    .bind(document_id)
    .bind(&embedding)
    .execute(&pool)
    .await?;
    insert_text_unit(&pool, document_id, 0, "api text", &embedding).await?;

    let state = Arc::new(AppState {
        pool: pool.clone(),
        models: CloudModelClient::for_test(),
        files_dir: files_dir.to_string_lossy().into_owned(),
        artifact_config: crate::page_artifacts::ArtifactConfig::for_test(),
        document_locks: crate::ingest::DocumentLocks::default(),
    });
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
    assert!(!document_dir.exists());

    tokio::fs::remove_dir_all(files_dir).await?;
    Ok(())
}

#[sqlx::test]
async fn image_embedding_publish_is_idempotent_and_never_duplicates_units(
    pool: PgPool,
) -> anyhow::Result<()> {
    let document_id = Uuid::new_v4();
    let files_dir = std::env::temp_dir().join(format!("km-artifact-retry-{document_id}"));
    let document_dir = files_dir.join(document_id.to_string());
    let storage_path = document_dir.join("original.png");
    tokio::fs::create_dir_all(&document_dir).await?;
    image::RgbaImage::from_pixel(11, 7, image::Rgba([20, 40, 60, 255]))
        .save_with_format(&storage_path, image::ImageFormat::Png)?;

    sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, artifact_status) \
         VALUES ($1, 'retry.png', $2, 'image/png', 'processing', 'processing')",
    )
    .bind(document_id)
    .bind(storage_path.to_string_lossy().as_ref())
    .execute(&pool)
    .await?;

    let (models, mock, server) = spawn_embedding_mock(false).await;
    let state = Arc::new(AppState {
        pool: pool.clone(),
        models,
        files_dir: files_dir.to_string_lossy().into_owned(),
        artifact_config: crate::page_artifacts::ArtifactConfig::for_test(),
        document_locks: crate::ingest::DocumentLocks::default(),
    });
    crate::ingest::process_document(
        state.clone(),
        document_id,
        storage_path.clone(),
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
    assert!(std::path::Path::new(&first_artifact_dir).is_dir());
    assert!(
        std::path::Path::new(&first_artifact_dir)
            .join("manifest.json")
            .is_file()
    );

    let stale_temp_dir = document_dir.join(".artifacts-interrupted.tmp");
    tokio::fs::create_dir_all(&stale_temp_dir).await?;
    tokio::fs::write(stale_temp_dir.join("partial.png"), b"partial").await?;
    crate::ingest::process_document(
        state,
        document_id,
        storage_path,
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
    assert!(!std::path::Path::new(&first_artifact_dir).exists());
    assert!(!stale_temp_dir.exists());
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
    let document_id = Uuid::new_v4();
    let files_dir = std::env::temp_dir().join(format!("km-embedding-failure-{document_id}"));
    let document_dir = files_dir.join(document_id.to_string());
    let storage_path = document_dir.join("original.png");
    tokio::fs::create_dir_all(&document_dir).await?;
    image::RgbaImage::from_pixel(12, 8, image::Rgba([80, 40, 20, 255]))
        .save_with_format(&storage_path, image::ImageFormat::Png)?;
    sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, artifact_status) \
         VALUES ($1, 'failure.png', $2, 'image/png', 'processing', 'processing')",
    )
    .bind(document_id)
    .bind(storage_path.to_string_lossy().as_ref())
    .execute(&pool)
    .await?;

    let (models, mock, server) = spawn_embedding_mock(true).await;
    let state = Arc::new(AppState {
        pool: pool.clone(),
        models,
        files_dir: files_dir.to_string_lossy().into_owned(),
        artifact_config: crate::page_artifacts::ArtifactConfig::for_test(),
        document_locks: crate::ingest::DocumentLocks::default(),
    });
    crate::ingest::process_document(
        state.clone(),
        document_id,
        storage_path.clone(),
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
    assert!(!error.contains(storage_path.to_string_lossy().as_ref()));
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
        state,
        document_id,
        storage_path,
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

    server.abort();
    tokio::fs::remove_dir_all(files_dir).await?;
    Ok(())
}

#[sqlx::test]
async fn summary_success_and_permanent_failures_publish_ready_without_reembedding(
    pool: PgPool,
) -> anyhow::Result<()> {
    async fn process_text_fixture(
        pool: &PgPool,
        state: Arc<AppState>,
        root: &std::path::Path,
        filename: &str,
    ) -> anyhow::Result<Uuid> {
        let document_id = Uuid::new_v4();
        let document_dir = root.join(document_id.to_string());
        tokio::fs::create_dir_all(&document_dir).await?;
        let storage_path = document_dir.join("original.md");
        tokio::fs::write(
            &storage_path,
            "# 測試文件\n\n這是供摘要與索引測試使用的繁體中文內容。",
        )
        .await?;
        sqlx::query(
            "INSERT INTO documents \
             (id, filename, storage_path, mime_type, status, artifact_status) \
             VALUES ($1, $2, $3, 'text/markdown', 'processing', 'pending')",
        )
        .bind(document_id)
        .bind(filename)
        .bind(storage_path.to_string_lossy().as_ref())
        .execute(pool)
        .await?;
        crate::ingest::process_document(
            state,
            document_id,
            storage_path,
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

    let root = std::env::temp_dir().join(format!("km-summary-{}", Uuid::new_v4()));
    tokio::fs::create_dir_all(&root).await?;
    let (models, mock, server) =
        spawn_embedding_mock_with_timeout(false, Duration::from_millis(20)).await;
    let state = Arc::new(AppState {
        pool: pool.clone(),
        models,
        files_dir: root.to_string_lossy().into_owned(),
        artifact_config: crate::page_artifacts::ArtifactConfig::for_test(),
        document_locks: crate::ingest::DocumentLocks::default(),
    });

    let success = process_text_fixture(&pool, state.clone(), &root, "success.md").await?;
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
    let rate_limited = process_text_fixture(&pool, state.clone(), &root, "rate-limit.md").await?;
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
    let malformed = process_text_fixture(&pool, state.clone(), &root, "malformed.md").await?;
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
    let timed_out = process_text_fixture(&pool, state, &root, "timeout.md").await?;
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
    tokio::fs::remove_dir_all(root).await?;
    Ok(())
}

#[sqlx::test]
async fn page_image_and_text_are_promoted_to_one_matching_halfvec_unit(
    pool: PgPool,
) -> anyhow::Result<()> {
    let document_id = Uuid::new_v4();
    let root = std::env::temp_dir().join(format!("km-mixed-page-{document_id}"));
    let artifact_dir = root.join("artifacts-fixture");
    let page_path = artifact_dir.join("pages/0001.png");
    tokio::fs::create_dir_all(page_path.parent().expect("page parent")).await?;
    image::RgbaImage::from_pixel(16, 9, image::Rgba([10, 30, 50, 255]))
        .save_with_format(&page_path, image::ImageFormat::Png)?;
    let original_path = root.join("original.pdf");
    tokio::fs::write(&original_path, b"fixture").await?;
    sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, artifact_status) \
         VALUES ($1, 'mixed.pdf', $2, 'application/pdf', 'processing', 'ready')",
    )
    .bind(document_id)
    .bind(original_path.to_string_lossy().as_ref())
    .execute(&pool)
    .await?;
    sqlx::query(
        "INSERT INTO document_artifact_sets \
         (document_id, source_kind, artifact_dir, manifest_path) \
         VALUES ($1, 'pdf', $2, $3)",
    )
    .bind(document_id)
    .bind(artifact_dir.to_string_lossy().as_ref())
    .bind(
        artifact_dir
            .join("manifest.json")
            .to_string_lossy()
            .as_ref(),
    )
    .execute(&pool)
    .await?;
    let page = crate::page_artifacts::PageArtifact {
        document_id,
        page_number: 1,
        image_path: page_path.to_string_lossy().into_owned(),
        width: 16,
        height: 9,
        extracted_text: Some("native same-page text".to_string()),
    };
    sqlx::query(
        "INSERT INTO page_artifacts \
         (document_id, page_number, image_path, width, height, extracted_text) \
         VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind(document_id)
    .bind(page.page_number)
    .bind(&page.image_path)
    .bind(page.width)
    .bind(page.height)
    .bind(&page.extracted_text)
    .execute(&pool)
    .await?;

    let (models, mock, server) = spawn_embedding_mock(false).await;
    let state = AppState {
        pool: pool.clone(),
        models,
        files_dir: root.to_string_lossy().into_owned(),
        artifact_config: crate::page_artifacts::ArtifactConfig::for_test(),
        document_locks: crate::ingest::DocumentLocks::default(),
    };
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
    assert_eq!(unit.4, page_path.to_string_lossy());
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
        assert_eq!(requests[0]["modality"], "text_image");
        let input = requests[0]["input"][0]
            .as_str()
            .expect("mixed input string");
        assert!(input.starts_with("native same-page text\n<img src=\"data:image/png;base64,"));
        assert!(!input.contains(page_path.to_string_lossy().as_ref()));
    }

    server.abort();
    tokio::fs::remove_dir_all(root).await?;
    Ok(())
}

#[sqlx::test]
async fn failed_conversion_can_retry_without_partial_artifacts(pool: PgPool) -> anyhow::Result<()> {
    let document_id = Uuid::new_v4();
    let files_dir = std::env::temp_dir().join(format!("km-artifact-failure-{document_id}"));
    let document_dir = files_dir.join(document_id.to_string());
    let storage_path = document_dir.join("original.png");
    tokio::fs::create_dir_all(&document_dir).await?;
    tokio::fs::write(&storage_path, b"not an image").await?;

    sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, artifact_status) \
         VALUES ($1, 'retry-after-failure.png', $2, 'image/png', 'processing', 'processing')",
    )
    .bind(document_id)
    .bind(storage_path.to_string_lossy().as_ref())
    .execute(&pool)
    .await?;

    let (models, mock, server) = spawn_embedding_mock(false).await;
    let state = Arc::new(AppState {
        pool: pool.clone(),
        models,
        files_dir: files_dir.to_string_lossy().into_owned(),
        artifact_config: crate::page_artifacts::ArtifactConfig::for_test(),
        document_locks: crate::ingest::DocumentLocks::default(),
    });
    crate::ingest::process_document(
        state.clone(),
        document_id,
        storage_path.clone(),
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
    let mut entries = tokio::fs::read_dir(&document_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name().to_string_lossy().into_owned();
        assert!(
            !name.starts_with(".artifacts-") && !name.starts_with("artifacts-"),
            "failed run left a partial artifact directory: {name}"
        );
    }

    image::RgbaImage::from_pixel(9, 5, image::Rgba([30, 60, 90, 255]))
        .save_with_format(&storage_path, image::ImageFormat::Png)?;
    crate::ingest::process_document(
        state,
        document_id,
        storage_path,
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
    tokio::fs::remove_dir_all(files_dir).await?;
    Ok(())
}

#[sqlx::test]
async fn delete_keeps_a_retry_handle_when_filesystem_cleanup_fails(
    pool: PgPool,
) -> anyhow::Result<()> {
    let document_id = Uuid::new_v4();
    let files_dir = std::env::temp_dir().join(format!("km-delete-retry-{document_id}"));
    tokio::fs::create_dir_all(&files_dir).await?;
    let document_path = files_dir.join(document_id.to_string());
    // A regular file at the expected directory path makes remove_dir_all fail
    // deterministically without relying on platform-specific permissions.
    tokio::fs::write(&document_path, b"not a directory").await?;
    let storage_path = document_path.join("original.pdf");

    sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, artifact_status) \
         VALUES ($1, 'cleanup.pdf', $2, 'application/pdf', 'failed', 'failed')",
    )
    .bind(document_id)
    .bind(storage_path.to_string_lossy().as_ref())
    .execute(&pool)
    .await?;

    let state = Arc::new(AppState {
        pool: pool.clone(),
        models: CloudModelClient::for_test(),
        files_dir: files_dir.to_string_lossy().into_owned(),
        artifact_config: crate::page_artifacts::ArtifactConfig::for_test(),
        document_locks: crate::ingest::DocumentLocks::default(),
    });
    let app = api_router(state);
    let request = Request::builder()
        .method("DELETE")
        .uri(format!("/api/documents/{document_id}"))
        .body(Body::empty())?;
    let response = app.clone().oneshot(request).await?;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let document_count: i64 = sqlx::query_scalar("SELECT count(*) FROM documents WHERE id = $1")
        .bind(document_id)
        .fetch_one(&pool)
        .await?;
    assert_eq!(document_count, 1, "failed cleanup must remain retryable");

    tokio::fs::remove_file(&document_path).await?;
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

    tokio::fs::remove_dir_all(files_dir).await?;
    Ok(())
}

#[sqlx::test]
async fn markdown_and_text_fixtures_keep_heading_aware_staging(pool: PgPool) -> anyhow::Result<()> {
    let root = std::env::temp_dir().join(format!("km-text-fixtures-{}", Uuid::new_v4()));
    let fixture_source_dir = root.join("sources");
    let fixtures = crate::test_fixtures::write_all(&fixture_source_dir)?;
    let (models, mock, server) = spawn_embedding_mock(false).await;
    let state = Arc::new(AppState {
        pool: pool.clone(),
        models,
        files_dir: root.to_string_lossy().into_owned(),
        artifact_config: crate::page_artifacts::ArtifactConfig::for_test(),
        document_locks: crate::ingest::DocumentLocks::default(),
    });

    for (filename, fixture) in [
        ("headings.md", fixtures.markdown),
        ("plain.txt", fixtures.text),
        ("hybrid-retrieval.md", fixtures.hybrid_retrieval),
    ] {
        let document_id = Uuid::new_v4();
        let document_dir = root.join(document_id.to_string());
        tokio::fs::create_dir(&document_dir).await?;
        let storage_path = document_dir.join(format!(
            "original.{}",
            std::path::Path::new(filename)
                .extension()
                .unwrap()
                .to_string_lossy()
        ));
        tokio::fs::copy(fixture, &storage_path).await?;
        sqlx::query(
            "INSERT INTO documents \
             (id, filename, storage_path, mime_type, status, artifact_status) \
             VALUES ($1, $2, $3, 'text/plain', 'processing', 'processing')",
        )
        .bind(document_id)
        .bind(filename)
        .bind(storage_path.to_string_lossy().as_ref())
        .execute(&pool)
        .await?;

        crate::ingest::process_document(
            state.clone(),
            document_id,
            storage_path,
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
                && request["modality"] == "text"
                && request["input"][0]
                    .as_str()
                    .is_some_and(|input| !input.trim().is_empty())
        }));
    }

    server.abort();
    tokio::fs::remove_dir_all(root).await?;
    Ok(())
}
