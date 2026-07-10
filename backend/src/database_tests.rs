use super::*;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use pgvector::HalfVector;
use serde_json::Value;
use sqlx::{PgPool, Row};
use tower::ServiceExt;
use uuid::Uuid;

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

    let index_definition: String = sqlx::query_scalar(
        "SELECT indexdef FROM pg_indexes \
         WHERE schemaname = current_schema() \
           AND indexname = 'document_units_embedding_hnsw'",
    )
    .fetch_one(&pool)
    .await?;
    assert!(index_definition.contains("USING hnsw"));
    assert!(index_definition.contains("halfvec_cosine_ops"));

    let chunks_table: Option<String> =
        sqlx::query_scalar("SELECT to_regclass('public.chunks')::text")
            .fetch_one(&pool)
            .await?;
    assert!(chunks_table.is_none());
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
         ORDER BY embedding <=> $1 LIMIT 3",
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
         SELECT id FROM document_units ORDER BY embedding <=> $1 LIMIT 2",
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
async fn document_list_and_delete_api_use_the_new_schema(pool: PgPool) -> anyhow::Result<()> {
    let document_id = Uuid::new_v4();
    let files_dir = std::env::temp_dir().join(format!("km-server-{document_id}"));
    tokio::fs::create_dir_all(&files_dir).await?;
    let storage_path = files_dir.join(format!("{document_id}.md"));
    tokio::fs::write(&storage_path, b"test document").await?;

    sqlx::query(
        "INSERT INTO documents (id, filename, storage_path, mime_type, status) \
         VALUES ($1, 'api.md', $2, 'text/markdown', 'ready')",
    )
    .bind(document_id)
    .bind(storage_path.to_string_lossy().as_ref())
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
    });
    let app = api_router(state);

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
    assert_eq!(document["page_count"], 1);
    assert_eq!(document["unit_count"], 2);
    assert_eq!(document["chunk_count"], 2);

    let response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/documents/{document_id}"))
                .body(Body::empty())?,
        )
        .await?;
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
    assert_eq!(document_count, 0);
    assert_eq!(unit_count, 0);
    assert!(!storage_path.exists());

    tokio::fs::remove_dir_all(files_dir).await?;
    Ok(())
}
