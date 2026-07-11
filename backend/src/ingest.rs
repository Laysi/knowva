use crate::AppState;
use crate::page_artifacts::{
    ArtifactError, ArtifactManifest, PageArtifact, SourceKind, TextArtifact,
};
use axum::Json;
use axum::extract::{Multipart, Path as AxumPath, State};
use axum::http::StatusCode;
use serde::Serialize;
use sqlx::{Postgres, Transaction};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use tokio::sync::{Mutex, OwnedMutexGuard};
use uuid::Uuid;

const MAX_CHUNK_CHARS: usize = 2_000;
const HARD_MAX_CHUNK_CHARS: usize = 3_000;
const OVERLAP_CHARS: usize = 200;
const MAX_ERROR_CHARS: usize = 1_000;
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
}

#[derive(Debug)]
struct PreparedArtifactSet {
    source_kind: SourceKind,
    artifact_dir: PathBuf,
    manifest_path: PathBuf,
    converted_pdf_path: Option<PathBuf>,
    pages: Vec<PageArtifact>,
    text_units: Vec<TextArtifact>,
}

/// POST /api/documents - save one supported file and start model-free artifact conversion.
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
        if bytes.len() > MAX_UPLOAD_BYTES {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("檔案超過 {} MiB 上限", MAX_UPLOAD_BYTES / 1_048_576),
            ));
        }
        upload = Some((filename, bytes.to_vec()));
    }

    let (original_filename, file_bytes) =
        upload.ok_or((StatusCode::BAD_REQUEST, "缺少 file 欄位".to_string()))?;
    let extension = extension_for(&original_filename)?;
    let source_kind = SourceKind::from_extension(&extension).ok_or((
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        "不支援此格式；請上傳 PDF、PNG/JPEG/WebP、DOCX/PPTX/XLSX 或 MD/TXT".to_string(),
    ))?;
    let mime_type = mime_type_for(&extension);
    let id = Uuid::new_v4();
    let files_dir = PathBuf::from(&state.files_dir);
    let storage_path = publish_original(&files_dir, id, &extension, &file_bytes)
        .await
        .map_err(internal_error)?;
    let storage_path_string = path_string(&storage_path);

    let insert_result = sqlx::query(
        "INSERT INTO documents \
         (id, filename, storage_path, mime_type, status, artifact_status) \
         VALUES ($1, $2, $3, $4, 'processing', 'processing')",
    )
    .bind(id)
    .bind(&original_filename)
    .bind(&storage_path_string)
    .bind(mime_type)
    .execute(&state.pool)
    .await;
    if let Err(error) = insert_result {
        let _ = remove_document_tree(&files_dir, id).await;
        return Err(internal_error(error));
    }

    let state_clone = state.clone();
    tokio::spawn(async move {
        process_document(state_clone, id, storage_path, source_kind).await;
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(UploadResponse {
            id,
            status: "processing".to_string(),
            artifact_status: "processing".to_string(),
        }),
    ))
}

fn multipart_error(error: axum::extract::multipart::MultipartError) -> (StatusCode, String) {
    let status = error.status();
    let message = if status == StatusCode::PAYLOAD_TOO_LARGE {
        format!("檔案超過 {} MiB 上限", MAX_UPLOAD_BYTES / 1_048_576)
    } else {
        error.body_text()
    };
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
        _ => "application/octet-stream",
    }
}

async fn publish_original(
    files_dir: &Path,
    document_id: Uuid,
    extension: &str,
    bytes: &[u8],
) -> Result<PathBuf, ArtifactError> {
    tokio::fs::create_dir_all(files_dir)
        .await
        .map_err(|error| {
            ArtifactError::new("original-storage", format!("無法建立檔案目錄: {error}"))
        })?;
    let document_dir = files_dir.join(document_id.to_string());
    if tokio::fs::try_exists(&document_dir).await.unwrap_or(false) {
        return Err(ArtifactError::new(
            "original-storage",
            "文件儲存目錄已存在，請重試",
        ));
    }

    let staging_dir = files_dir.join(format!(".{document_id}.upload-{}.tmp", Uuid::new_v4()));
    let original_path = staging_dir.join(format!("original.{extension}"));
    let result = async {
        tokio::fs::create_dir(&staging_dir).await?;
        tokio::fs::write(&original_path, bytes).await?;
        tokio::fs::rename(&staging_dir, &document_dir).await
    }
    .await;
    if let Err(error) = result {
        let _ = tokio::fs::remove_dir_all(&staging_dir).await;
        return Err(ArtifactError::new(
            "original-storage",
            format!("原檔寫入失敗: {error}"),
        ));
    }
    Ok(document_dir.join(format!("original.{extension}")))
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

fn bounded_error(error: &str) -> String {
    error.chars().take(MAX_ERROR_CHARS).collect()
}

pub(crate) async fn process_document(
    state: Arc<AppState>,
    id: Uuid,
    storage_path: PathBuf,
    source_kind: SourceKind,
) {
    let _guard = state.document_locks.lock(id).await;
    let document_dir = match storage_path.parent() {
        Some(path) => path.to_owned(),
        None => {
            set_failed(&state, id, "original-storage: 原檔路徑缺少文件目錄").await;
            return;
        }
    };

    let marked_processing = sqlx::query(
        "UPDATE documents \
         SET status = 'processing', artifact_status = 'processing', error = NULL \
         WHERE id = $1",
    )
    .bind(id)
    .execute(&state.pool)
    .await;
    match marked_processing {
        Ok(result) if result.rows_affected() == 0 => {
            let _ = tokio::fs::remove_dir_all(&document_dir).await;
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
    let temporary_dir = document_dir.join(format!(".artifacts-{run_id}.tmp"));
    let artifact_dir = document_dir.join(format!("artifacts-{run_id}"));
    if let Err(error) = tokio::fs::create_dir(&temporary_dir).await {
        set_failed(
            &state,
            id,
            &format!("artifact-stage: 無法建立暫存目錄: {error}"),
        )
        .await;
        return;
    }

    let prepared = prepare_artifact_set(
        &state,
        id,
        source_kind,
        &storage_path,
        &temporary_dir,
        &artifact_dir,
    )
    .await;
    let mut prepared = match prepared {
        Ok(prepared) => prepared,
        Err(error) => {
            let _ = tokio::fs::remove_dir_all(&temporary_dir).await;
            set_failed(&state, id, &error.to_string()).await;
            return;
        }
    };

    if !document_exists(&state, id).await {
        let _ = tokio::fs::remove_dir_all(&temporary_dir).await;
        let _ = tokio::fs::remove_dir_all(&document_dir).await;
        return;
    }
    if let Err(error) = tokio::fs::rename(&temporary_dir, &artifact_dir).await {
        let _ = tokio::fs::remove_dir_all(&temporary_dir).await;
        set_failed(
            &state,
            id,
            &format!("artifact-publish: 無法發布完整 artifact set: {error}"),
        )
        .await;
        return;
    }
    prepared.artifact_dir = artifact_dir.clone();

    if let Err(error) = persist_artifact_set(&state, id, &prepared).await {
        let _ = tokio::fs::remove_dir_all(&artifact_dir).await;
        if document_exists(&state, id).await {
            set_failed(&state, id, &error.to_string()).await;
        } else {
            let _ = tokio::fs::remove_dir_all(&document_dir).await;
        }
        return;
    }

    cleanup_obsolete_artifact_runs(&document_dir, &artifact_dir).await;
}

async fn prepare_artifact_set(
    state: &AppState,
    document_id: Uuid,
    source_kind: SourceKind,
    original_path: &Path,
    temporary_dir: &Path,
    final_dir: &Path,
) -> Result<PreparedArtifactSet, ArtifactError> {
    let mut converted_pdf_path = None;
    let rendered_pages = match source_kind {
        SourceKind::Pdf => {
            crate::page_artifacts::render_pdf(&state.artifact_config, original_path, temporary_dir)
                .await?
        }
        SourceKind::Image => {
            crate::page_artifacts::normalize_image(
                &state.artifact_config,
                original_path,
                temporary_dir,
            )
            .await?
        }
        SourceKind::Office => {
            let temporary_pdf = crate::page_artifacts::convert_office_to_pdf(
                &state.artifact_config,
                original_path,
                temporary_dir,
            )
            .await?;
            let pages = crate::page_artifacts::render_pdf(
                &state.artifact_config,
                &temporary_pdf,
                temporary_dir,
            )
            .await?;
            converted_pdf_path = Some(final_dir.join("converted.pdf"));
            pages
        }
        SourceKind::Text => Vec::new(),
    };

    let pages = rendered_pages
        .into_iter()
        .map(|page| PageArtifact {
            document_id,
            page_number: page.page_number,
            image_path: path_string(&final_dir.join(page.relative_image_path)),
            width: page.width,
            height: page.height,
            extracted_text: page.extracted_text,
        })
        .collect::<Vec<_>>();

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
    } else {
        Vec::new()
    };

    if source_kind != SourceKind::Text && pages.is_empty() {
        return Err(ArtifactError::new(
            "artifact-validate",
            "轉換結果不含任何頁面",
        ));
    }
    validate_page_sequence(&pages)?;

    let manifest_path = final_dir.join("manifest.json");
    let manifest = ArtifactManifest {
        document_id,
        source_kind,
        original_path: path_string(original_path),
        converted_pdf_path: converted_pdf_path.as_ref().map(|path| path_string(path)),
        pages: pages.clone(),
        text_units: text_units.clone(),
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|error| {
        ArtifactError::new(
            "artifact-manifest",
            format!("無法序列化 artifact manifest: {error}"),
        )
    })?;
    tokio::fs::write(temporary_dir.join("manifest.json"), manifest_bytes)
        .await
        .map_err(|error| {
            ArtifactError::new(
                "artifact-manifest",
                format!("無法寫入 artifact manifest: {error}"),
            )
        })?;

    Ok(PreparedArtifactSet {
        source_kind,
        artifact_dir: final_dir.to_owned(),
        manifest_path,
        converted_pdf_path,
        pages,
        text_units,
    })
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
    .bind(path_string(&prepared.artifact_dir))
    .bind(path_string(&prepared.manifest_path))
    .bind(
        prepared
            .converted_pdf_path
            .as_ref()
            .map(|path| path_string(path)),
    )
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

async fn document_exists(state: &AppState, document_id: Uuid) -> bool {
    sqlx::query_scalar::<_, bool>("SELECT EXISTS(SELECT 1 FROM documents WHERE id = $1)")
        .bind(document_id)
        .fetch_one(&state.pool)
        .await
        .unwrap_or(false)
}

async fn cleanup_obsolete_artifact_runs(document_dir: &Path, active_dir: &Path) {
    let active_name = active_dir.file_name();
    let Ok(mut entries) = tokio::fs::read_dir(document_dir).await else {
        return;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name();
        let Some(name_text) = name.to_str() else {
            continue;
        };
        let is_artifact_run = name_text.starts_with("artifacts-")
            || (name_text.starts_with(".artifacts-") && name_text.ends_with(".tmp"));
        if is_artifact_run
            && Some(name.as_os_str()) != active_name
            && let Err(error) = tokio::fs::remove_dir_all(entry.path()).await
        {
            tracing::warn!(path = %entry.path().display(), "failed to remove stale artifact run: {error}");
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
    let storage_path: Option<String> =
        sqlx::query_scalar("SELECT storage_path FROM documents WHERE id = $1")
            .bind(id)
            .fetch_optional(&state.pool)
            .await
            .map_err(internal_error)?;
    let Some(storage_path) = storage_path else {
        return Err((StatusCode::NOT_FOUND, "document not found".to_string()));
    };

    // Keep the DB row as the retry handle until every known filesystem path is
    // gone. If cleanup fails, the client can safely repeat DELETE instead of
    // receiving a permanent 404 while orphaned artifacts remain on disk.
    remove_document_tree(Path::new(&state.files_dir), id)
        .await
        .map_err(internal_error)?;

    // Compatibility cleanup for documents uploaded before the per-document layout.
    let legacy_path = PathBuf::from(storage_path);
    if legacy_path.parent() == Some(Path::new(&state.files_dir)) {
        remove_legacy_files(Path::new(&state.files_dir), id).await?;
    }

    sqlx::query("DELETE FROM documents WHERE id = $1")
        .bind(id)
        .execute(&state.pool)
        .await
        .map_err(internal_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn remove_document_tree(files_dir: &Path, id: Uuid) -> std::io::Result<()> {
    let document_dir = files_dir.join(id.to_string());
    match tokio::fs::remove_dir_all(document_dir).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

async fn remove_legacy_files(files_dir: &Path, id: Uuid) -> Result<(), (StatusCode, String)> {
    let mut entries = match tokio::fs::read_dir(files_dir).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(internal_error(error)),
    };
    let file_prefix = format!("{id}.");
    let upload_prefix = format!(".{id}.upload-");
    while let Some(entry) = entries.next_entry().await.map_err(internal_error)? {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name.starts_with(&file_prefix) {
            tokio::fs::remove_file(entry.path())
                .await
                .map_err(internal_error)?;
        } else if name.starts_with(&upload_prefix) && name.ends_with(".tmp") {
            tokio::fs::remove_dir_all(entry.path())
                .await
                .map_err(internal_error)?;
        }
    }
    Ok(())
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn internal_error(error: impl fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
