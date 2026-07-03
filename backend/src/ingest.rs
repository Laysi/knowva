use crate::bedrock;
use crate::AppState;
use axum::extract::{Multipart, Path as AxumPath, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;
use std::sync::Arc;
use uuid::Uuid;

const MAX_CHUNK_CHARS: usize = 2000;
const HARD_MAX_CHUNK_CHARS: usize = 3000;
const OVERLAP_CHARS: usize = 200;

#[derive(Debug, Serialize)]
pub struct UploadResponse {
    id: Uuid,
    status: String,
}

/// POST /api/documents - multipart upload, saves file, kicks off background processing.
pub async fn upload_document(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<UploadResponse>), (StatusCode, String)> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut original_filename: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        if field.name() == Some("file") {
            original_filename = field.file_name().map(|s| s.to_string());
            let bytes = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            file_bytes = Some(bytes.to_vec());
        }
    }

    let file_bytes = file_bytes.ok_or((StatusCode::BAD_REQUEST, "missing file field".to_string()))?;
    let original_filename =
        original_filename.unwrap_or_else(|| "upload".to_string());

    let id = Uuid::new_v4();
    let ext = std::path::Path::new(&original_filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin")
        .to_lowercase();

    let files_dir = state.files_dir.clone();
    tokio::fs::create_dir_all(&files_dir)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let storage_path = format!("{files_dir}/{id}.{ext}");
    tokio::fs::write(&storage_path, &file_bytes)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    sqlx::query(
        "INSERT INTO documents (id, filename, storage_path, mime_type, status) VALUES ($1, $2, $3, $4, 'processing')",
    )
    .bind(id)
    .bind(&original_filename)
    .bind(&storage_path)
    .bind(Option::<String>::None)
    .execute(&state.pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let state_clone = state.clone();
    tokio::spawn(async move {
        process_document(state_clone, id, storage_path, ext).await;
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(UploadResponse {
            id,
            status: "processing".to_string(),
        }),
    ))
}

async fn set_failed(state: &AppState, id: Uuid, error: &str) {
    let snippet: String = error.chars().take(1000).collect();
    let _ = sqlx::query("UPDATE documents SET status = 'failed', error = $2 WHERE id = $1")
        .bind(id)
        .bind(snippet)
        .execute(&state.pool)
        .await;
}

async fn process_document(state: Arc<AppState>, id: Uuid, storage_path: String, ext: String) {
    // Step 1: get markdown text.
    let markdown = match ext.as_str() {
        "md" | "markdown" | "txt" => match tokio::fs::read_to_string(&storage_path).await {
            Ok(text) => text,
            Err(e) => {
                set_failed(&state, id, &format!("failed to read file: {e}")).await;
                return;
            }
        },
        _ => match run_markitdown(&storage_path).await {
            Ok(md) => {
                let md_path = derive_md_path(&storage_path);
                if let Err(e) = tokio::fs::write(&md_path, &md).await {
                    tracing::warn!("failed to write converted markdown for {id}: {e}");
                }
                md
            }
            Err(e) => {
                set_failed(&state, id, &e).await;
                return;
            }
        },
    };

    if markdown.trim().is_empty() {
        set_failed(&state, id, "conversion produced empty output").await;
        return;
    }

    // Step 2: chunk.
    let chunks = chunk_markdown(&markdown);
    if chunks.is_empty() {
        set_failed(&state, id, "no chunks produced from document").await;
        return;
    }

    // Step 3: embeddings + insert.
    for (idx, chunk) in chunks.iter().enumerate() {
        let embedding =
            match bedrock::embed_text(&state.bedrock_client, &state.embed_model_id, chunk).await {
                Ok(v) => v,
                Err(e) => {
                    set_failed(&state, id, &format!("embedding failed: {e}")).await;
                    return;
                }
            };
        let vector = pgvector::Vector::from(embedding);

        if let Err(e) = sqlx::query(
            "INSERT INTO chunks (document_id, chunk_index, content, embedding) VALUES ($1, $2, $3, $4)",
        )
        .bind(id)
        .bind(idx as i32)
        .bind(chunk)
        .bind(vector)
        .execute(&state.pool)
        .await
        {
            set_failed(&state, id, &format!("db insert failed: {e}")).await;
            return;
        }
    }

    // Step 4: summary + tags (best-effort).
    let (summary, tags) = generate_summary_and_tags(&state, &markdown).await;

    let update_result = sqlx::query(
        "UPDATE documents SET status = 'ready', summary = $2, tags = $3 WHERE id = $1",
    )
    .bind(id)
    .bind(summary)
    .bind(tags)
    .execute(&state.pool)
    .await;

    if let Err(e) = update_result {
        set_failed(&state, id, &format!("final update failed: {e}")).await;
    }
}

fn derive_md_path(storage_path: &str) -> String {
    let path = std::path::Path::new(storage_path);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("doc");
    let parent = path.parent().and_then(|p| p.to_str()).unwrap_or(".");
    format!("{parent}/{stem}.md")
}

async fn run_markitdown(storage_path: &str) -> Result<String, String> {
    let candidates = ["markitdown", "~/.local/bin/markitdown"];
    let home = std::env::var("HOME").unwrap_or_default();

    for candidate in candidates {
        let bin = if candidate.starts_with("~/") {
            format!("{home}/{}", &candidate[2..])
        } else {
            candidate.to_string()
        };

        let output = tokio::process::Command::new(&bin)
            .arg(storage_path)
            .output()
            .await;

        match output {
            Ok(out) => {
                if out.status.success() && !out.stdout.is_empty() {
                    return Ok(String::from_utf8_lossy(&out.stdout).to_string());
                } else {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let snippet: String = stderr.chars().take(500).collect();
                    return Err(format!(
                        "markitdown exited with {}: {}",
                        out.status,
                        if snippet.is_empty() {
                            "(no stderr, empty output)".to_string()
                        } else {
                            snippet
                        }
                    ));
                }
            }
            Err(e) => {
                // Binary not found at this candidate path; try next.
                tracing::debug!("markitdown candidate {bin} unavailable: {e}");
                continue;
            }
        }
    }

    Err("markitdown binary not found in PATH or ~/.local/bin".to_string())
}

/// Split markdown by headings, merge adjacent sections up to ~2000 chars (hard max 3000),
/// and split oversized sections on paragraph boundaries with overlap.
fn chunk_markdown(markdown: &str) -> Vec<String> {
    let heading_re = regex::Regex::new(r"(?m)^#{1,6} ").unwrap();

    // Split into sections, each starting at a heading (first section may have no heading).
    let mut sections: Vec<String> = Vec::new();
    let mut last_start = 0;
    let mut starts: Vec<usize> = heading_re.find_iter(markdown).map(|m| m.start()).collect();
    if starts.first() != Some(&0) {
        starts.insert(0, 0);
    }
    for i in 0..starts.len() {
        let start = starts[i];
        let end = if i + 1 < starts.len() {
            starts[i + 1]
        } else {
            markdown.len()
        };
        if start >= last_start {
            let section = markdown[start..end].to_string();
            if !section.trim().is_empty() {
                sections.push(section);
            }
        }
        last_start = end;
    }
    if sections.is_empty() && !markdown.trim().is_empty() {
        sections.push(markdown.to_string());
    }

    // Further split oversized sections on paragraph boundaries with overlap.
    let mut expanded: Vec<String> = Vec::new();
    for section in sections {
        if section.chars().count() <= HARD_MAX_CHUNK_CHARS {
            expanded.push(section);
        } else {
            expanded.extend(split_oversized(&section));
        }
    }

    // Merge adjacent sections up to ~2000 chars (hard max 3000).
    let mut merged: Vec<String> = Vec::new();
    let mut current = String::new();
    for section in expanded {
        if current.is_empty() {
            current = section;
            continue;
        }
        let combined_len = current.chars().count() + section.chars().count();
        if combined_len <= MAX_CHUNK_CHARS {
            current.push('\n');
            current.push_str(&section);
        } else if combined_len <= HARD_MAX_CHUNK_CHARS {
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

    merged.into_iter().filter(|c| !c.trim().is_empty()).collect()
}

fn split_oversized(section: &str) -> Vec<String> {
    let paragraphs: Vec<&str> = section.split("\n\n").collect();
    let mut result: Vec<String> = Vec::new();
    let mut current = String::new();

    for para in paragraphs {
        let combined_len = current.chars().count() + para.chars().count();
        if current.is_empty() {
            current = para.to_string();
        } else if combined_len <= MAX_CHUNK_CHARS {
            current.push_str("\n\n");
            current.push_str(para);
        } else {
            let overlap: String = current
                .chars()
                .rev()
                .take(OVERLAP_CHARS)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            result.push(std::mem::take(&mut current));
            current = format!("{overlap}\n\n{para}");
        }

        // If a single paragraph itself exceeds the hard max, hard-split it.
        while current.chars().count() > HARD_MAX_CHUNK_CHARS {
            let take_count = MAX_CHUNK_CHARS;
            let piece: String = current.chars().take(take_count).collect();
            let remainder: String = current.chars().skip(take_count.saturating_sub(OVERLAP_CHARS)).collect();
            result.push(piece);
            current = remainder;
        }
    }
    if !current.trim().is_empty() {
        result.push(current);
    }
    result
}

async fn generate_summary_and_tags(
    state: &AppState,
    markdown: &str,
) -> (Option<String>, Option<Vec<String>>) {
    let excerpt: String = markdown.chars().take(6000).collect();
    let prompt = format!(
        "請閱讀以下文件內容,並以嚴格 JSON 格式回覆(不要加任何說明文字或 markdown 標記),格式為 {{\"summary\":\"繁體中文,100字以內\",\"tags\":[\"3到6個繁體中文短標籤\"]}}。\n\n文件內容:\n{excerpt}"
    );

    let result = bedrock::converse_once(
        &state.bedrock_client,
        &state.chat_model_id,
        bedrock::SYSTEM_PROMPT,
        &prompt,
    )
    .await;

    let raw = match result {
        Ok(text) => text,
        Err(e) => {
            tracing::warn!("summary/tags generation failed: {e}");
            return (None, None);
        }
    };

    let cleaned = strip_code_fences(&raw);

    match serde_json::from_str::<serde_json::Value>(&cleaned) {
        Ok(value) => {
            let summary = value
                .get("summary")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let tags = value.get("tags").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|t| t.as_str().map(|s| s.to_string()))
                    .collect::<Vec<String>>()
            });
            (summary, tags)
        }
        Err(e) => {
            tracing::warn!("failed to parse summary/tags JSON: {e}; raw: {raw}");
            (None, None)
        }
    }
}

fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let rest = rest.trim_start_matches("json").trim_start();
        if let Some(end) = rest.rfind("```") {
            return rest[..end].trim().to_string();
        }
        return rest.trim().to_string();
    }
    trimmed.to_string()
}

/// DELETE /api/documents/:id
pub async fn delete_document(
    State(state): State<Arc<AppState>>,
    AxumPath(id): AxumPath<Uuid>,
) -> Result<StatusCode, (StatusCode, String)> {
    let row: Option<(String,)> = sqlx::query_as("SELECT storage_path FROM documents WHERE id = $1")
        .bind(id)
        .fetch_optional(&state.pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if row.is_none() {
        return Err((StatusCode::NOT_FOUND, "document not found".to_string()));
    }

    sqlx::query("DELETE FROM documents WHERE id = $1")
        .bind(id)
        .execute(&state.pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Remove FILES_DIR/{id}.* files (original + any converted markdown).
    if let Ok(mut entries) = tokio::fs::read_dir(&state.files_dir).await {
        let prefix = format!("{id}.");
        while let Ok(Some(entry)) = entries.next_entry().await {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with(&prefix) {
                    let _ = tokio::fs::remove_file(entry.path()).await;
                }
            }
        }
    }

    Ok(StatusCode::NO_CONTENT)
}
