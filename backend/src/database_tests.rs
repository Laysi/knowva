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
         (id, filename, storage_path, mime_type, status, artifact_status) \
         VALUES ($1, 'api.pdf', $2, 'application/pdf', 'processing', 'ready')",
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
    assert_eq!(document["artifact_status"], "ready");
    assert_eq!(document["page_count"], 1);
    assert_eq!(document["unit_count"], 2);
    assert_eq!(document["chunk_count"], 2);

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
async fn image_artifact_publish_is_idempotent_and_never_fabricates_embeddings(
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

    let state = Arc::new(AppState {
        pool: pool.clone(),
        models: CloudModelClient::for_test(),
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
    assert_eq!(state_row, ("processing".to_string(), "ready".to_string()));
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
    assert_eq!(
        unit_count, 0,
        "issue #4 must not invent placeholder vectors"
    );

    tokio::fs::remove_dir_all(files_dir).await?;
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

    let state = Arc::new(AppState {
        pool: pool.clone(),
        models: CloudModelClient::for_test(),
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
    assert_eq!(
        retry_statuses,
        ("processing".to_string(), "ready".to_string())
    );
    let counts: (i64, i64, i64) = sqlx::query_as(
        "SELECT \
           (SELECT count(*) FROM document_artifact_sets WHERE document_id = $1), \
           (SELECT count(*) FROM page_artifacts WHERE document_id = $1), \
           (SELECT count(*) FROM document_units WHERE document_id = $1)",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;
    assert_eq!(counts, (1, 1, 0));

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
    let state = Arc::new(AppState {
        pool: pool.clone(),
        models: CloudModelClient::for_test(),
        files_dir: root.to_string_lossy().into_owned(),
        artifact_config: crate::page_artifacts::ArtifactConfig::for_test(),
        document_locks: crate::ingest::DocumentLocks::default(),
    });

    for (filename, fixture) in [
        ("headings.md", fixtures.markdown),
        ("plain.txt", fixtures.text),
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
        assert_eq!(unit_count, 0);
        assert_eq!(statuses, ("processing".to_string(), "ready".to_string()));
    }

    tokio::fs::remove_dir_all(root).await?;
    Ok(())
}
