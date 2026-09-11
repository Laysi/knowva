use crate::AppState;
use crate::cloud_models::{CloudModelClient, CloudModelError, MAX_EMBED_IMAGE_BYTES};
use crate::page_artifacts::{
    ArtifactError, ArtifactManifest, PageArtifact, SourceKind, TextArtifact,
};
use crate::storage::{IO_CONCURRENCY, Storage};
use axum::Json;
use axum::extract::{Multipart, Path as AxumPath, State};
use axum::http::StatusCode;
use futures::{StreamExt, TryStreamExt, stream};
use pgvector::HalfVector;
use serde::Serialize;
use sqlx::{Postgres, Transaction};
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use tokio::sync::{Mutex, OwnedMutexGuard};
use uuid::Uuid;

const MAX_CHUNK_CHARS: usize = 2_000;
const HARD_MAX_CHUNK_CHARS: usize = 3_000;
const OVERLAP_CHARS: usize = 200;
const MAX_ERROR_CHARS: usize = 1_000;
const EMBEDDING_CONCURRENCY: usize = 4;
const EMBED_IMAGE_SHORT_SIDE_LIMIT: i32 = 8_192;
const EMBED_IMAGE_LONG_SIDE_LIMIT: i32 = 16_384;
pub const MAX_UPLOAD_BYTES: usize = 50 * 1_048_576;
pub const MULTIPART_BODY_LIMIT_BYTES: usize = MAX_UPLOAD_BYTES + 1_048_576;

#[derive(Debug, Clone, Default)]
pub(crate) struct DocumentLocks {
    locks: Arc<Mutex<HashMap<Uuid, Weak<Mutex<()>>>>>,
}

impl DocumentLocks {
    pub(crate) async fn lock(&self, document_id: Uuid) -> OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.locks.lock().await;
            locks.retain(|_, lock| lock.strong_count() > 0);
            if let Some(lock) = locks.get(&document_id).and_then(Weak::upgrade) {
                lock
            } else {
                let lock = Arc::new(Mutex::new(()));
                locks.insert(document_id, Arc::downgrade(&lock));
                lock
            }
        };
        lock.lock_owned().await
    }
}

#[derive(Debug, Serialize)]
pub struct UploadResponse {
    id: Uuid,
    status: String,
    artifact_status: String,
    #[serde(rename = "processingStage")]
    processing_stage: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluationIngestResult {
    pub document_id: Uuid,
    pub status: String,
    pub artifact_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationIngestError {
    pub code: &'static str,
    pub safe_message: &'static str,
}

impl EvaluationIngestError {
    fn new(code: &'static str, safe_message: &'static str) -> Self {
        Self { code, safe_message }
    }
}

impl fmt::Display for EvaluationIngestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.safe_message)
    }
}

impl std::error::Error for EvaluationIngestError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProcessOptions {
    generate_summary: bool,
}

impl ProcessOptions {
    fn production() -> Self {
        Self {
            generate_summary: true,
        }
    }

    fn evaluation() -> Self {
        Self {
            generate_summary: false,
        }
    }
}

struct CreatedDocument {
    id: Uuid,
    storage_key: String,
    source_kind: SourceKind,
}

/// A converted run in local scratch space plus the object keys it will be
/// published under. Nothing here is visible until `persist_artifact_set` commits.
#[derive(Debug)]
struct PreparedArtifactSet {
    source_kind: SourceKind,
    artifact_dir: String,
    manifest_path: String,
    converted_pdf_path: Option<String>,
    pages: Vec<PageArtifact>,
    text_units: Vec<TextArtifact>,
    files: Vec<(PathBuf, String)>,
    manifest: Vec<u8>,
}

/// Private local directory for Pdfium/LibreOffice, removed on every exit path.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn create(run_id: Uuid) -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!("km-ingest-{run_id}"));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.create(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug, Clone)]
enum PendingEmbeddingUnit {
    Page(PageArtifact),
    Text(TextArtifact),
}

#[derive(Debug)]
struct EmbeddedUnit {
    kind: &'static str,
    unit_index: i32,
    page_number: Option<i32>,
    content: Option<String>,
    image_path: Option<String>,
    embedding: HalfVector,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EmbeddingStageError {
    stage: &'static str,
    unit_kind: Option<&'static str>,
    unit_index: Option<i32>,
    page_number: Option<i32>,
    message: String,
}

impl EmbeddingStageError {
    fn document(stage: &'static str, message: impl Into<String>) -> Self {
        Self {
            stage,
            unit_kind: None,
            unit_index: None,
            page_number: None,
            message: message.into(),
        }
    }

    fn unit(
        stage: &'static str,
        unit_kind: &'static str,
        unit_index: i32,
        page_number: Option<i32>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            stage,
            unit_kind: Some(unit_kind),
            unit_index: Some(unit_index),
            page_number,
            message: message.into(),
        }
    }
}

impl fmt::Display for EmbeddingStageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.stage)?;
        if let (Some(kind), Some(index)) = (self.unit_kind, self.unit_index) {
            write!(formatter, ": {kind} unit={index}")?;
        }
        if let Some(page) = self.page_number {
            write!(formatter, " page={page}")?;
        }
        write!(formatter, ": {}", self.message)
    }
}

impl std::error::Error for EmbeddingStageError {}

/// POST /api/documents - save one supported file, publish artifacts, then embed every unit.
pub async fn upload_document(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<UploadResponse>), (StatusCode, String)> {
    let mut upload: Option<(String, Vec<u8>)> = None;

    loop {
        let field = multipart.next_field().await.map_err(multipart_error)?;
        let Some(field) = field else { break };
        if field.name() != Some("file") {
            continue;
        }
        if upload.is_some() {
            return Err((StatusCode::BAD_REQUEST, "只允許一個 file 欄位".to_string()));
        }
        let filename = field
            .file_name()
            .map(str::to_owned)
            .ok_or((StatusCode::BAD_REQUEST, "file 欄位缺少檔名".to_string()))?;
        let bytes = field.bytes().await.map_err(multipart_error)?;
        if bytes.is_empty() {
            return Err((StatusCode::BAD_REQUEST, "不可上傳空檔案".to_string()));
        }
        upload = Some((filename, bytes.to_vec()));
    }

    let (original_filename, file_bytes) =
        upload.ok_or((StatusCode::BAD_REQUEST, "缺少 file 欄位".to_string()))?;
    let created = create_document(&state, &original_filename, &file_bytes).await?;
    let id = created.id;

    let state_clone = state.clone();
    tokio::spawn(async move {
        process_document(state_clone, id, created.storage_key, created.source_kind).await;
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(UploadResponse {
            id,
            status: "processing".to_string(),
            artifact_status: "pending".to_string(),
            processing_stage: "queued".to_string(),
        }),
    ))
}

async fn create_document(
    state: &AppState,
    original_filename: &str,
    file_bytes: &[u8],
) -> Result<CreatedDocument, (StatusCode, String)> {
    if file_bytes.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "不可上傳空檔案".to_owned()));
    }
    if file_bytes.len() > MAX_UPLOAD_BYTES {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("檔案超過 {} MiB 上限", MAX_UPLOAD_BYTES / 1_048_576),
        ));
    }
    let extension = extension_for(original_filename)?;
    let source_kind = SourceKind::from_extension(&extension).ok_or((
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        "不支援此格式；請上傳 PDF、PNG/JPEG/WebP、DOCX/PPTX/XLSX 或 MD/TXT".to_owned(),
    ))?;
    let mime_type = mime_type_for(&extension);
    let id = Uuid::new_v4();
    let storage_key = format!("{id}/original.{extension}");
    state
        .storage
        .put(&storage_key, file_bytes.to_vec())
        .await
        .map_err(|error| internal_error(format!("original-storage: 原檔寫入失敗: {error}")))?;

    let insert_result = sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, artifact_status) \
         VALUES ($1, $2, $3, $4, 'processing', 'pending')",
    )
    .bind(id)
    .bind(original_filename)
    .bind(&storage_key)
    .bind(mime_type)
    .execute(&state.pool)
    .await;
    if let Err(error) = insert_result {
        // A lost reply can hide a committed INSERT whose row references the original.
        if !document_exists(state, id).await {
            let _ = state.storage.delete_prefix(&id.to_string()).await;
        }
        return Err(internal_error(error));
    }

    Ok(CreatedDocument {
        id,
        storage_key,
        source_kind,
    })
}

/// Synchronously ingests immutable evaluation bytes through the production
/// artifact and embedding pipeline while skipping non-critical summary calls.
pub async fn ingest_evaluation_bytes(
    state: Arc<AppState>,
    filename: &str,
    bytes: &[u8],
) -> Result<EvaluationIngestResult, EvaluationIngestError> {
    if filename.trim().is_empty()
        || Path::new(filename)
            .file_name()
            .and_then(|value| value.to_str())
            != Some(filename)
    {
        return Err(EvaluationIngestError::new(
            "fixture_filename_invalid",
            "evaluation fixture filename is invalid",
        ));
    }
    if bytes.is_empty() || bytes.len() > MAX_UPLOAD_BYTES {
        return Err(EvaluationIngestError::new(
            "fixture_size_invalid",
            "evaluation fixture size is invalid",
        ));
    }
    let created = create_document(&state, filename, bytes)
        .await
        .map_err(|_| {
            EvaluationIngestError::new(
                "fixture_create_failed",
                "evaluation fixture could not be staged",
            )
        })?;
    process_document_with_options(
        state.clone(),
        created.id,
        created.storage_key,
        created.source_kind,
        ProcessOptions::evaluation(),
    )
    .await;

    let row: Option<(String, String)> =
        sqlx::query_as("SELECT status, artifact_status FROM documents WHERE id = $1")
            .bind(created.id)
            .fetch_optional(&state.pool)
            .await
            .map_err(|_| {
                EvaluationIngestError::new(
                    "fixture_status_failed",
                    "evaluation fixture status could not be verified",
                )
            })?;
    match row {
        Some((status, artifact_status)) if status == "ready" && artifact_status == "ready" => {
            Ok(EvaluationIngestResult {
                document_id: created.id,
                status,
                artifact_status,
            })
        }
        Some(_) => Err(EvaluationIngestError::new(
            "fixture_ingest_failed",
            "evaluation fixture ingestion did not complete successfully",
        )),
        None => Err(EvaluationIngestError::new(
            "fixture_missing_after_ingest",
            "evaluation fixture disappeared during ingestion",
        )),
    }
}

fn multipart_error(error: axum::extract::multipart::MultipartError) -> (StatusCode, String) {
    let status = error.status();
    let message = error.body_text();
    (status, message)
}

fn extension_for(filename: &str) -> Result<String, (StatusCode, String)> {
    Path::new(filename)
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.is_empty())
        .map(str::to_ascii_lowercase)
        .ok_or((
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "檔名必須包含支援的副檔名".to_string(),
        ))
}

fn mime_type_for(extension: &str) -> &'static str {
    match extension {
        "pdf" => "application/pdf",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "md" | "markdown" => "text/markdown",
        "txt" => "text/plain",
        "stp" | "step" => "application/step",
        "eml" => "message/rfc822",
        "msg" => "application/vnd.ms-outlook",
        _ => "application/octet-stream",
    }
}

async fn set_failed(state: &AppState, id: Uuid, error: &str) {
    let snippet = bounded_error(error);
    if let Err(db_error) = sqlx::query(
        "UPDATE documents \
         SET status = 'failed', artifact_status = 'failed', error = $2 \
         WHERE id = $1",
    )
    .bind(id)
    .bind(snippet)
    .execute(&state.pool)
    .await
    {
        tracing::warn!(document_id = %id, "failed to persist ingest error: {db_error}");
    }
}

async fn set_embedding_failed(state: &AppState, id: Uuid, error: &EmbeddingStageError) {
    let snippet = bounded_error(&error.to_string());
    if let Err(db_error) =
        sqlx::query("UPDATE documents SET status = 'failed', error = $2 WHERE id = $1")
            .bind(id)
            .bind(&snippet)
            .execute(&state.pool)
            .await
    {
        tracing::warn!(document_id = %id, "failed to persist embedding error: {db_error}");
    }
    tracing::warn!(
        document_id = %id,
        stage = error.stage,
        unit_kind = ?error.unit_kind,
        unit_index = ?error.unit_index,
        page_number = ?error.page_number,
        error = %snippet,
        "document embedding failed"
    );
}

fn bounded_error(error: &str) -> String {
    let data_urls = regex::Regex::new(r"(?i)data:image/(?:png|jpeg);base64,[A-Za-z0-9+/=_-]*")
        .expect("valid data URL regex");
    let without_images = data_urls.replace_all(error, "[REDACTED-IMAGE-DATA]");
    let long_base64 = regex::Regex::new(r"[A-Za-z0-9+/]{64,}={0,2}").expect("valid base64 regex");
    long_base64
        .replace_all(&without_images, "[REDACTED-BASE64]")
        .chars()
        .take(MAX_ERROR_CHARS)
        .collect()
}

pub(crate) async fn process_document(
    state: Arc<AppState>,
    id: Uuid,
    storage_key: String,
    source_kind: SourceKind,
) {
    process_document_with_options(
        state,
        id,
        storage_key,
        source_kind,
        ProcessOptions::production(),
    )
    .await;
}

async fn process_document_with_options(
    state: Arc<AppState>,
    id: Uuid,
    storage_key: String,
    source_kind: SourceKind,
    options: ProcessOptions,
) {
    let _guard = state.document_locks.lock(id).await;
    let document_prefix = id.to_string();

    let marked_processing = sqlx::query(
        "UPDATE documents \
         SET status = 'processing', artifact_status = 'processing', \
             summary = NULL, tags = NULL, error = NULL \
         WHERE id = $1",
    )
    .bind(id)
    .execute(&state.pool)
    .await;
    match marked_processing {
        Ok(result) if result.rows_affected() == 0 => {
            let _ = state.storage.delete_prefix(&document_prefix).await;
            return;
        }
        Ok(_) => {}
        Err(error) => {
            set_failed(
                &state,
                id,
                &format!("artifact-db: 無法更新處理狀態: {error}"),
            )
            .await;
            return;
        }
    }

    let run_id = Uuid::new_v4();
    let run_name = format!("artifacts-{run_id}");
    let run_prefix = format!("{id}/{run_name}");
    // Pdfium and LibreOffice only read local files, so conversion always runs in
    // a scratch directory; dropping the guard deletes it on every exit path.
    let scratch = match ScratchDir::create(run_id) {
        Ok(scratch) => scratch,
        Err(error) => {
            set_failed(
                &state,
                id,
                &format!("artifact-stage: 無法建立暫存目錄: {error}"),
            )
            .await;
            return;
        }
    };

    let prepared = prepare_artifact_set(
        &state,
        id,
        source_kind,
        &storage_key,
        scratch.path(),
        &run_prefix,
    )
    .await;
    let prepared = match prepared {
        Ok(prepared) => prepared,
        Err(error) => {
            set_failed(&state, id, &error.to_string()).await;
            return;
        }
    };

    if !document_exists(&state, id).await {
        let _ = state.storage.delete_prefix(&document_prefix).await;
        return;
    }
    // Objects under the fresh run prefix stay unreferenced until the DB commit in
    // persist_artifact_set, the only publication switch. Any failure before that
    // commit removes this run again, so no DB row can point at a partial set.
    if let Err(error) = upload_artifact_set(&state.storage, &prepared).await {
        let _ = state.storage.delete_prefix(&run_prefix).await;
        set_failed(&state, id, &error.to_string()).await;
        return;
    }
    drop(scratch);

    if let Err(error) = persist_artifact_set(&state, id, &prepared).await {
        // A lost COMMIT reply can hide a successful publish, so re-read what the DB
        // references before deleting. When the DB cannot answer, keep the objects:
        // an unreferenced run is harmless, a row pointing at deleted objects is not.
        let published: Result<Option<Option<String>>, sqlx::Error> = sqlx::query_scalar(
            "SELECT a.artifact_dir FROM documents d \
             LEFT JOIN document_artifact_sets a ON a.document_id = d.id \
             WHERE d.id = $1",
        )
        .bind(id)
        .fetch_optional(&state.pool)
        .await;
        match published {
            Ok(None) => {
                let _ = state.storage.delete_prefix(&document_prefix).await;
                return;
            }
            Ok(Some(artifact_dir)) if artifact_dir.as_deref() != Some(run_prefix.as_str()) => {
                let _ = state.storage.delete_prefix(&run_prefix).await;
            }
            Ok(Some(_)) => {}
            Err(db_error) => tracing::warn!(
                document_id = %id,
                "cannot confirm artifact publication; keeping run objects: {db_error}"
            ),
        }
        set_failed(&state, id, &error.to_string()).await;
        return;
    }

    let embedded_units = match embed_artifact_units(
        &state.models,
        &state.storage,
        &prepared.pages,
        &prepared.text_units,
    )
    .await
    {
        Ok(units) => units,
        Err(error) => {
            set_embedding_failed(&state, id, &error).await;
            return;
        }
    };
    let metadata = if options.generate_summary {
        crate::document_summary::generate_best_effort(
            &state.models,
            id,
            &prepared.pages,
            &prepared.text_units,
            &state.storage,
        )
        .await
    } else {
        None
    };
    if let Err(error) = persist_embedded_units(&state, id, &embedded_units, metadata.as_ref()).await
    {
        set_embedding_failed(&state, id, &error).await;
        return;
    }
    // Superseded runs can still back the previous document_units until the
    // promotion above commits, so they are removed only after it.
    cleanup_obsolete_artifact_runs(&state.storage, id, &run_name).await;
}

async fn prepare_artifact_set(
    state: &AppState,
    document_id: Uuid,
    source_kind: SourceKind,
    storage_key: &str,
    scratch: &Path,
    run_prefix: &str,
) -> Result<PreparedArtifactSet, ArtifactError> {
    let original = state
        .storage
        .get(document_id, storage_key, MAX_UPLOAD_BYTES)
        .await
        .map_err(|error| ArtifactError::new("original-read", format!("無法讀取原檔: {error}")))?;
    let original_path = scratch.join(storage_key.rsplit('/').next().unwrap_or("original"));
    let original_path = original_path.as_path();
    let run_dir = scratch.join("run");
    let staged = async {
        tokio::fs::write(original_path, original).await?;
        tokio::fs::create_dir(&run_dir).await
    };
    staged.await.map_err(|error| {
        ArtifactError::new("artifact-stage", format!("無法寫入暫存區: {error}"))
    })?;

    let mut files = Vec::new();
    let mut converted_pdf_path = None;
    let rendered_pages = match source_kind {
        SourceKind::Pdf => {
            crate::page_artifacts::render_pdf(&state.artifact_config, original_path, &run_dir)
                .await?
        }
        SourceKind::Image => {
            crate::page_artifacts::normalize_image(&state.artifact_config, original_path, &run_dir)
                .await?
        }
        SourceKind::Office => {
            let temporary_pdf = crate::page_artifacts::convert_office_to_pdf(
                &state.artifact_config,
                original_path,
                &run_dir,
            )
            .await?;
            let pages =
                crate::page_artifacts::render_pdf(&state.artifact_config, &temporary_pdf, &run_dir)
                    .await?;
            let key = format!("{run_prefix}/converted.pdf");
            files.push((temporary_pdf, key.clone()));
            converted_pdf_path = Some(key);
            pages
        }
        SourceKind::Text | SourceKind::Binary => Vec::new(),
    };

    let mut pages = Vec::with_capacity(rendered_pages.len());
    for page in rendered_pages {
        let key = format!(
            "{run_prefix}/{}",
            page.relative_image_path.to_string_lossy()
        );
        files.push((run_dir.join(&page.relative_image_path), key.clone()));
        pages.push(PageArtifact {
            document_id,
            page_number: page.page_number,
            image_path: key,
            width: page.width,
            height: page.height,
            extracted_text: page.extracted_text,
        });
    }

    let text_units = if source_kind == SourceKind::Text {
        let text = tokio::fs::read_to_string(original_path)
            .await
            .map_err(|error| {
                ArtifactError::new("text-read", format!("無法以 UTF-8 讀取文字檔: {error}"))
            })?;
        let chunks = chunk_markdown(&text);
        if chunks.is_empty() {
            return Err(ArtifactError::new("text-chunk", "文字檔沒有可建立的內容"));
        }
        chunks
            .into_iter()
            .enumerate()
            .map(|(index, content)| {
                Ok(TextArtifact {
                    document_id,
                    unit_index: i32::try_from(index).map_err(|_| {
                        ArtifactError::new("text-chunk", "文字段落數超過系統可追蹤範圍")
                    })?,
                    content,
                })
            })
            .collect::<Result<Vec<_>, ArtifactError>>()?
    } else if source_kind == SourceKind::Binary {
        let extension = original_path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("file");
        vec![TextArtifact {
            document_id,
            unit_index: 0,
            content: format!(
                "工業物聯網媒體檔案：{}。檔案格式：{}。原始檔案已保存，可供後續影音或郵件解析流程使用。",
                original_path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("unknown"),
                extension.to_ascii_uppercase()
            ),
        }]
    } else {
        Vec::new()
    };

    if source_kind != SourceKind::Text && source_kind != SourceKind::Binary && pages.is_empty() {
        return Err(ArtifactError::new(
            "artifact-validate",
            "轉換結果不含任何頁面",
        ));
    }
    validate_page_sequence(&pages)?;

    let manifest = ArtifactManifest {
        document_id,
        source_kind,
        original_path: storage_key.to_owned(),
        converted_pdf_path: converted_pdf_path.clone(),
        pages: pages.clone(),
        text_units: text_units.clone(),
    };
    let manifest = serde_json::to_vec_pretty(&manifest).map_err(|error| {
        ArtifactError::new(
            "artifact-manifest",
            format!("無法序列化 artifact manifest: {error}"),
        )
    })?;

    Ok(PreparedArtifactSet {
        source_kind,
        artifact_dir: run_prefix.to_owned(),
        manifest_path: format!("{run_prefix}/manifest.json"),
        converted_pdf_path,
        pages,
        text_units,
        files,
        manifest,
    })
}

async fn upload_artifact_set(
    storage: &Storage,
    prepared: &PreparedArtifactSet,
) -> Result<(), ArtifactError> {
    fn publish_error(error: impl fmt::Display) -> ArtifactError {
        ArtifactError::new(
            "artifact-publish",
            format!("無法上傳完整 artifact set: {error}"),
        )
    }

    let uploads = prepared
        .files
        .iter()
        .map(|(local_path, key)| async move {
            let bytes = tokio::fs::read(local_path).await.map_err(publish_error)?;
            storage.put(key, bytes).await.map_err(publish_error)
        })
        .collect::<Vec<_>>();
    stream::iter(uploads)
        .buffer_unordered(IO_CONCURRENCY)
        .try_collect::<Vec<()>>()
        .await?;
    // Uploaded last, so an existing manifest always describes a complete run.
    storage
        .put(&prepared.manifest_path, prepared.manifest.clone())
        .await
        .map_err(publish_error)
}

fn validate_page_sequence(pages: &[PageArtifact]) -> Result<(), ArtifactError> {
    for (index, page) in pages.iter().enumerate() {
        let expected = i32::try_from(index + 1)
            .map_err(|_| ArtifactError::new("artifact-validate", "頁數超過系統可追蹤範圍"))?;
        if page.page_number != expected {
            return Err(ArtifactError::new(
                "artifact-validate",
                format!(
                    "轉換結果頁序不連續：預期第 {expected} 頁，收到第 {} 頁",
                    page.page_number
                ),
            ));
        }
    }
    Ok(())
}

async fn persist_artifact_set(
    state: &AppState,
    document_id: Uuid,
    prepared: &PreparedArtifactSet,
) -> Result<(), ArtifactError> {
    let mut transaction = state.pool.begin().await.map_err(artifact_db_error)?;
    let document_state: Option<(String, String)> =
        sqlx::query_as("SELECT status, artifact_status FROM documents WHERE id = $1 FOR UPDATE")
            .bind(document_id)
            .fetch_optional(&mut *transaction)
            .await
            .map_err(artifact_db_error)?;
    let Some((status, artifact_status)) = document_state else {
        return Err(ArtifactError::new(
            "artifact-db",
            "文件已刪除，取消 artifact 發布",
        ));
    };
    if status != "processing" || artifact_status != "processing" {
        return Err(ArtifactError::new(
            "artifact-db",
            "文件狀態已變更，取消過期的 artifact 發布",
        ));
    }

    sqlx::query("DELETE FROM document_artifact_sets WHERE document_id = $1")
        .bind(document_id)
        .execute(&mut *transaction)
        .await
        .map_err(artifact_db_error)?;
    sqlx::query(
        "INSERT INTO document_artifact_sets \
         (document_id, source_kind, artifact_dir, manifest_path, converted_pdf_path) \
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(document_id)
    .bind(prepared.source_kind.as_str())
    .bind(&prepared.artifact_dir)
    .bind(&prepared.manifest_path)
    .bind(&prepared.converted_pdf_path)
    .execute(&mut *transaction)
    .await
    .map_err(artifact_db_error)?;

    insert_pages(&mut transaction, &prepared.pages).await?;
    insert_text_units(&mut transaction, &prepared.text_units).await?;
    sqlx::query("UPDATE documents SET artifact_status = 'ready', error = NULL WHERE id = $1")
        .bind(document_id)
        .execute(&mut *transaction)
        .await
        .map_err(artifact_db_error)?;
    transaction.commit().await.map_err(artifact_db_error)
}

async fn embed_artifact_units(
    models: &CloudModelClient,
    storage: &Storage,
    pages: &[PageArtifact],
    text_units: &[TextArtifact],
) -> Result<Vec<EmbeddedUnit>, EmbeddingStageError> {
    let pending = pages
        .iter()
        .cloned()
        .map(PendingEmbeddingUnit::Page)
        .chain(text_units.iter().cloned().map(PendingEmbeddingUnit::Text))
        .collect::<Vec<_>>();
    if pending.is_empty() {
        return Err(EmbeddingStageError::document(
            "embedding-prepare",
            "artifact set contains no units",
        ));
    }

    let mut embedded = stream::iter(pending)
        .map(|unit| embed_artifact_unit(models, storage, unit))
        .buffer_unordered(EMBEDDING_CONCURRENCY)
        .try_collect::<Vec<_>>()
        .await?;
    embedded.sort_by_key(|unit| (unit.kind, unit.unit_index));
    Ok(embedded)
}

async fn embed_artifact_unit(
    models: &CloudModelClient,
    storage: &Storage,
    unit: PendingEmbeddingUnit,
) -> Result<EmbeddedUnit, EmbeddingStageError> {
    match unit {
        PendingEmbeddingUnit::Text(text) => {
            let embedding = models
                .embed_passage(&text.content)
                .await
                .map_err(|error| embedding_model_error("text", text.unit_index, None, error))?;
            let embedding = crate::embedding::to_halfvec(embedding).map_err(|error| {
                EmbeddingStageError::unit(
                    "embedding-validate",
                    "text",
                    text.unit_index,
                    None,
                    error.to_string(),
                )
            })?;
            Ok(EmbeddedUnit {
                kind: "text",
                unit_index: text.unit_index,
                page_number: None,
                content: Some(text.content),
                image_path: None,
                embedding,
            })
        }
        PendingEmbeddingUnit::Page(page) => {
            let unit_index = page.page_number.checked_sub(1).ok_or_else(|| {
                EmbeddingStageError::unit(
                    "embedding-prepare",
                    "page",
                    0,
                    Some(page.page_number),
                    "page number must be one-based",
                )
            })?;
            validate_embed_image_dimensions(&page, unit_index)?;
            let image = storage
                .get(page.document_id, &page.image_path, MAX_EMBED_IMAGE_BYTES)
                .await
                .map_err(|error| {
                    EmbeddingStageError::unit(
                        "embedding-image-read",
                        "page",
                        unit_index,
                        Some(page.page_number),
                        format!("cannot read page image artifact: {error}"),
                    )
                })?;
            let embedding = match page
                .extracted_text
                .as_deref()
                .filter(|text| !text.trim().is_empty())
            {
                Some(text) => models.embed_image_text_passage(text, &image).await,
                None => models.embed_image_passage(&image).await,
            }
            .map_err(|error| {
                embedding_model_error("page", unit_index, Some(page.page_number), error)
            })?;
            let embedding = crate::embedding::to_halfvec(embedding).map_err(|error| {
                EmbeddingStageError::unit(
                    "embedding-validate",
                    "page",
                    unit_index,
                    Some(page.page_number),
                    error.to_string(),
                )
            })?;
            Ok(EmbeddedUnit {
                kind: "page",
                unit_index,
                page_number: Some(page.page_number),
                content: page.extracted_text,
                image_path: Some(page.image_path),
                embedding,
            })
        }
    }
}

fn embedding_model_error(
    unit_kind: &'static str,
    unit_index: i32,
    page_number: Option<i32>,
    error: CloudModelError,
) -> EmbeddingStageError {
    let stage = match &error {
        CloudModelError::InvalidRequest { .. } => "embedding-input",
        CloudModelError::InvalidResponse { .. } => "embedding-response",
        CloudModelError::Http { .. }
        | CloudModelError::Timeout { .. }
        | CloudModelError::Transport { .. } => "embedding-request",
    };
    EmbeddingStageError::unit(stage, unit_kind, unit_index, page_number, error.to_string())
}

fn validate_embed_image_dimensions(
    page: &PageArtifact,
    unit_index: i32,
) -> Result<(), EmbeddingStageError> {
    let fits_vendor_limit = (page.width <= EMBED_IMAGE_SHORT_SIDE_LIMIT
        && page.height <= EMBED_IMAGE_LONG_SIDE_LIMIT)
        || (page.width <= EMBED_IMAGE_LONG_SIDE_LIMIT
            && page.height <= EMBED_IMAGE_SHORT_SIDE_LIMIT);
    if page.width <= 0 || page.height <= 0 || !fits_vendor_limit {
        return Err(EmbeddingStageError::unit(
            "embedding-image-validate",
            "page",
            unit_index,
            Some(page.page_number),
            format!(
                "page image dimensions {}x{} exceed the supported orientation limits",
                page.width, page.height
            ),
        ));
    }
    Ok(())
}

async fn persist_embedded_units(
    state: &AppState,
    document_id: Uuid,
    units: &[EmbeddedUnit],
    metadata: Option<&crate::document_summary::DocumentSummary>,
) -> Result<(), EmbeddingStageError> {
    let expected_count = i64::try_from(units.len()).map_err(|_| {
        EmbeddingStageError::document(
            "embedding-db",
            "unit count exceeds the database tracking range",
        )
    })?;
    if expected_count == 0 {
        return Err(EmbeddingStageError::document(
            "embedding-db",
            "cannot publish an empty embedding set",
        ));
    }

    let mut transaction = state.pool.begin().await.map_err(embedding_db_error)?;
    let document_state: Option<(String, String)> =
        sqlx::query_as("SELECT status, artifact_status FROM documents WHERE id = $1 FOR UPDATE")
            .bind(document_id)
            .fetch_optional(&mut *transaction)
            .await
            .map_err(embedding_db_error)?;
    let Some((status, artifact_status)) = document_state else {
        return Err(EmbeddingStageError::document(
            "embedding-db",
            "document was deleted before embedding publication",
        ));
    };
    if status != "processing" || artifact_status != "ready" {
        return Err(EmbeddingStageError::document(
            "embedding-db",
            "document state changed before embedding publication",
        ));
    }

    let artifact_count: i64 = sqlx::query_scalar(
        "SELECT \
           (SELECT count(*) FROM page_artifacts WHERE document_id = $1) + \
           (SELECT count(*) FROM text_artifacts WHERE document_id = $1)",
    )
    .bind(document_id)
    .fetch_one(&mut *transaction)
    .await
    .map_err(embedding_db_error)?;
    if artifact_count != expected_count {
        return Err(EmbeddingStageError::document(
            "embedding-db-count",
            format!(
                "artifact count {artifact_count} does not match embedded unit count {expected_count}"
            ),
        ));
    }

    sqlx::query("DELETE FROM document_units WHERE document_id = $1")
        .bind(document_id)
        .execute(&mut *transaction)
        .await
        .map_err(embedding_db_error)?;
    for unit in units {
        sqlx::query(
            "INSERT INTO document_units \
             (document_id, kind, unit_index, page_number, content, image_path, embedding) \
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(document_id)
        .bind(unit.kind)
        .bind(unit.unit_index)
        .bind(unit.page_number)
        .bind(&unit.content)
        .bind(&unit.image_path)
        .bind(&unit.embedding)
        .execute(&mut *transaction)
        .await
        .map_err(embedding_db_error)?;
    }

    let stored_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM document_units WHERE document_id = $1")
            .bind(document_id)
            .fetch_one(&mut *transaction)
            .await
            .map_err(embedding_db_error)?;
    if stored_count != expected_count {
        return Err(EmbeddingStageError::document(
            "embedding-db-count",
            format!(
                "stored unit count {stored_count} does not match expected count {expected_count}"
            ),
        ));
    }

    let summary = metadata.map(|metadata| metadata.summary.clone());
    let tags = metadata.map(|metadata| metadata.tags.clone());
    let update = sqlx::query(
        "UPDATE documents SET status = 'ready', summary = $2, tags = $3, error = NULL \
         WHERE id = $1 AND status = 'processing' AND artifact_status = 'ready'",
    )
    .bind(document_id)
    .bind(summary)
    .bind(tags)
    .execute(&mut *transaction)
    .await
    .map_err(embedding_db_error)?;
    if update.rows_affected() != 1 {
        return Err(EmbeddingStageError::document(
            "embedding-db",
            "document was not eligible to become ready",
        ));
    }
    transaction.commit().await.map_err(embedding_db_error)
}

#[cfg(test)]
pub(crate) async fn embed_published_artifacts_for_test(
    state: &AppState,
    document_id: Uuid,
    pages: &[PageArtifact],
    text_units: &[TextArtifact],
) -> Result<(), String> {
    let units = embed_artifact_units(&state.models, &state.storage, pages, text_units)
        .await
        .map_err(|error| error.to_string())?;
    persist_embedded_units(state, document_id, &units, None)
        .await
        .map_err(|error| error.to_string())
}

fn embedding_db_error(error: sqlx::Error) -> EmbeddingStageError {
    EmbeddingStageError::document("embedding-db", format!("embedding write failed: {error}"))
}

async fn insert_pages(
    transaction: &mut Transaction<'_, Postgres>,
    pages: &[PageArtifact],
) -> Result<(), ArtifactError> {
    for page in pages {
        sqlx::query(
            "INSERT INTO page_artifacts \
             (document_id, page_number, image_path, width, height, extracted_text) \
             VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(page.document_id)
        .bind(page.page_number)
        .bind(&page.image_path)
        .bind(page.width)
        .bind(page.height)
        .bind(&page.extracted_text)
        .execute(&mut **transaction)
        .await
        .map_err(artifact_db_error)?;
    }
    Ok(())
}

async fn insert_text_units(
    transaction: &mut Transaction<'_, Postgres>,
    units: &[TextArtifact],
) -> Result<(), ArtifactError> {
    for unit in units {
        sqlx::query(
            "INSERT INTO text_artifacts (document_id, unit_index, content) VALUES ($1, $2, $3)",
        )
        .bind(unit.document_id)
        .bind(unit.unit_index)
        .bind(&unit.content)
        .execute(&mut **transaction)
        .await
        .map_err(artifact_db_error)?;
    }
    Ok(())
}

fn artifact_db_error(error: sqlx::Error) -> ArtifactError {
    ArtifactError::new("artifact-db", format!("artifact 寫入失敗: {error}"))
}

/// `false` only when the row is confirmed gone: callers delete every object of the
/// document on `false`, so an unanswerable database must never read as deleted.
async fn document_exists(state: &AppState, document_id: Uuid) -> bool {
    sqlx::query_scalar::<_, bool>("SELECT EXISTS(SELECT 1 FROM documents WHERE id = $1)")
        .bind(document_id)
        .fetch_one(&state.pool)
        .await
        .unwrap_or(true)
}

/// Runs after the publishing commit only: removes superseded runs and runs left
/// by an interrupted process. Failures are logged; DELETE retries them later.
async fn cleanup_obsolete_artifact_runs(storage: &Storage, document_id: Uuid, active_run: &str) {
    let keys = match storage.list(&document_id.to_string()).await {
        Ok(keys) => keys,
        Err(error) => {
            tracing::warn!(%document_id, "failed to list stale artifact runs: {error}");
            return;
        }
    };
    let stale = keys
        .iter()
        .filter_map(|key| key.split('/').nth(1))
        .filter(|run| run.starts_with("artifacts-") && *run != active_run)
        .collect::<BTreeSet<_>>();
    for run in stale {
        if let Err(error) = storage.delete_prefix(&format!("{document_id}/{run}")).await {
            tracing::warn!(%document_id, run, "failed to remove stale artifact run: {error}");
        }
    }
}

/// Split markdown by headings, merge adjacent sections, then split oversized sections.
fn chunk_markdown(markdown: &str) -> Vec<String> {
    let heading_re = regex::Regex::new(r"(?m)^#{1,6} ").expect("static heading regex");
    let mut starts = heading_re
        .find_iter(markdown)
        .map(|match_| match_.start())
        .collect::<Vec<_>>();
    if starts.first() != Some(&0) {
        starts.insert(0, 0);
    }

    let mut sections = Vec::new();
    for (index, start) in starts.iter().copied().enumerate() {
        let end = starts.get(index + 1).copied().unwrap_or(markdown.len());
        let section = &markdown[start..end];
        if !section.trim().is_empty() {
            sections.push(section.to_string());
        }
    }
    if sections.is_empty() && !markdown.trim().is_empty() {
        sections.push(markdown.to_string());
    }

    let expanded = sections
        .into_iter()
        .flat_map(|section| {
            if section.chars().count() <= HARD_MAX_CHUNK_CHARS {
                vec![section]
            } else {
                split_oversized(&section)
            }
        })
        .collect::<Vec<_>>();

    let mut merged = Vec::new();
    let mut current = String::new();
    for section in expanded {
        if current.is_empty() {
            current = section;
        } else if current.chars().count() + section.chars().count() < HARD_MAX_CHUNK_CHARS {
            current.push('\n');
            current.push_str(&section);
        } else {
            merged.push(std::mem::take(&mut current));
            current = section;
        }
    }
    if !current.trim().is_empty() {
        merged.push(current);
    }
    merged
}

fn split_oversized(section: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    for paragraph in section.split("\n\n") {
        let separator = usize::from(!current.is_empty()) * 2;
        if current.is_empty() {
            current = paragraph.to_string();
        } else if current.chars().count() + paragraph.chars().count() + separator <= MAX_CHUNK_CHARS
        {
            current.push_str("\n\n");
            current.push_str(paragraph);
        } else {
            let overlap = current
                .chars()
                .rev()
                .take(OVERLAP_CHARS)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<String>();
            result.push(std::mem::take(&mut current));
            current = format!("{overlap}\n\n{paragraph}");
        }

        while current.chars().count() > HARD_MAX_CHUNK_CHARS {
            result.push(current.chars().take(MAX_CHUNK_CHARS).collect());
            current = current
                .chars()
                .skip(MAX_CHUNK_CHARS - OVERLAP_CHARS)
                .collect();
        }
    }
    if !current.trim().is_empty() {
        result.push(current);
    }
    result
}

/// DELETE /api/documents/:id. The same per-document lock serializes this with conversion.
pub async fn delete_document(
    State(state): State<Arc<AppState>>,
    AxumPath(id): AxumPath<Uuid>,
) -> Result<StatusCode, (StatusCode, String)> {
    let _guard = state.document_locks.lock(id).await;
    // Withdraw the document from ready-only retrieval before any object goes, so a
    // DELETE that fails after cleanup never leaves a searchable row without objects.
    let withdrawn = sqlx::query("UPDATE documents SET status = 'failed' WHERE id = $1")
        .bind(id)
        .execute(&state.pool)
        .await
        .map_err(internal_error)?;
    if withdrawn.rows_affected() == 0 {
        return Err((StatusCode::NOT_FOUND, "document not found".to_string()));
    }

    // Keep the DB row as the retry handle until every object under the document
    // prefix is gone. If cleanup fails, the client can safely repeat DELETE
    // instead of receiving a permanent 404 while orphaned objects remain.
    state
        .storage
        .delete_prefix(&id.to_string())
        .await
        .map_err(internal_error)?;

    sqlx::query("DELETE FROM documents WHERE id = $1")
        .bind(id)
        .execute(&state.pool)
        .await
        .map_err(internal_error)?;
    Ok(StatusCode::NO_CONTENT)
}

fn internal_error(error: impl fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires Pdfium and LibreOffice"]
    async fn native_conversion_runs_in_scratch_and_publishes_relative_keys() -> anyhow::Result<()> {
        let fixtures_root =
            std::env::temp_dir().join(format!("km-ingest-native-{}", Uuid::new_v4()));
        let fixtures = crate::test_fixtures::write_all(&fixtures_root)?;
        let storage = Storage::memory();
        let state = AppState {
            // Conversion and upload never touch the database; this pool never connects.
            pool: sqlx::postgres::PgPoolOptions::new()
                .connect_lazy("postgres://unused@127.0.0.1:9/unused")?,
            models: CloudModelClient::for_test(),
            storage: storage.clone(),
            artifact_config: crate::page_artifacts::ArtifactConfig::from_env()?,
            document_locks: DocumentLocks::default(),
        };

        for (extension, source, kind, page_count) in [
            ("pdf", &fixtures.native_pdf, SourceKind::Pdf, 2),
            ("pdf", &fixtures.scanned_pdf, SourceKind::Pdf, 1),
            ("docx", &fixtures.docx, SourceKind::Office, 2),
            ("pptx", &fixtures.pptx, SourceKind::Office, 2),
            ("xlsx", &fixtures.xlsx, SourceKind::Office, 1),
        ] {
            let document_id = Uuid::new_v4();
            let storage_key = format!("{document_id}/original.{extension}");
            storage.put(&storage_key, std::fs::read(source)?).await?;
            let run_prefix = format!("{document_id}/artifacts-native");
            let scratch = ScratchDir::create(Uuid::new_v4())?;
            let scratch_path = scratch.path().to_owned();

            let prepared = prepare_artifact_set(
                &state,
                document_id,
                kind,
                &storage_key,
                scratch.path(),
                &run_prefix,
            )
            .await?;
            upload_artifact_set(&storage, &prepared).await?;
            drop(scratch);
            assert!(!scratch_path.exists(), "scratch must be removed");

            assert_eq!(prepared.pages.len(), page_count, "{extension} page count");
            let stored = storage.list(&run_prefix).await?;
            for (index, page) in prepared.pages.iter().enumerate() {
                assert_eq!(page.page_number, index as i32 + 1);
                assert_eq!(
                    page.image_path,
                    format!("{run_prefix}/pages/{:04}.png", index + 1)
                );
                assert!(stored.contains(&page.image_path), "{stored:?}");
            }
            assert!(stored.contains(&prepared.manifest_path));
            assert_eq!(
                prepared.converted_pdf_path.is_some(),
                kind == SourceKind::Office
            );
            if let Some(converted) = &prepared.converted_pdf_path {
                assert!(stored.contains(converted), "{stored:?}");
            }
            let manifest: ArtifactManifest = serde_json::from_slice(
                &storage
                    .get(document_id, &prepared.manifest_path, usize::MAX)
                    .await?,
            )?;
            assert_eq!(manifest.original_path, storage_key);
            assert_eq!(manifest.pages, prepared.pages);
        }

        std::fs::remove_dir_all(fixtures_root)?;
        Ok(())
    }

    #[test]
    fn evaluation_ingest_disables_only_best_effort_summary() {
        assert!(ProcessOptions::production().generate_summary);
        assert!(!ProcessOptions::evaluation().generate_summary);
    }

    #[test]
    fn heading_aware_chunking_remains_model_free_and_bounded() {
        let document = format!("# 第一章\n\n{}\n\n# 第二章\n內容", "字".repeat(7_000));
        let chunks = chunk_markdown(&document);
        assert!(chunks.len() >= 3);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.chars().count() <= HARD_MAX_CHUNK_CHARS)
        );
    }

    #[test]
    fn page_sequence_must_be_one_based_and_contiguous() {
        let id = Uuid::new_v4();
        let pages = vec![PageArtifact {
            document_id: id,
            page_number: 2,
            image_path: "page.png".to_string(),
            width: 1,
            height: 1,
            extracted_text: None,
        }];
        assert!(validate_page_sequence(&pages).is_err());
    }

    #[test]
    fn embedding_image_dimensions_follow_vendor_orientation_limits() {
        let page = |width, height| PageArtifact {
            document_id: Uuid::new_v4(),
            page_number: 1,
            image_path: "never-read.png".to_string(),
            width,
            height,
            extracted_text: None,
        };
        assert!(validate_embed_image_dimensions(&page(8_192, 16_384), 0).is_ok());
        assert!(validate_embed_image_dimensions(&page(16_384, 8_192), 0).is_ok());
        let error = validate_embed_image_dimensions(&page(8_193, 8_193), 0)
            .expect_err("both sides cannot exceed the short-side limit");
        assert_eq!(error.stage, "embedding-image-validate");
        assert_eq!(error.page_number, Some(1));
    }

    #[test]
    fn persisted_error_snippets_redact_image_data_and_long_base64() {
        let error = format!(
            "page=2 data:image/png;base64,{} trailing",
            "A".repeat(MAX_ERROR_CHARS + 100)
        );
        let bounded = bounded_error(&error);
        assert!(bounded.contains("page=2"));
        assert!(bounded.contains("[REDACTED-IMAGE-DATA]"));
        assert!(!bounded.contains("data:image"));
        assert!(bounded.chars().count() <= MAX_ERROR_CHARS);
    }
}
