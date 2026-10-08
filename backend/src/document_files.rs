//! What an answer cited, read back for the person asking (issue #56): a unit's
//! page image or text, a catalog answer's document card, and the document's
//! original file.
//!
//! Only what the knowledge base answers from right now is served: the
//! retrieval-ready units of a ready document, and that document's card and original.
//! Everything else, a malformed id included, answers the same 404, so a deleted,
//! reprocessed or still-processing document looks like one that never existed.
//! Stored keys go through the document-scoped storage checks, and neither a key
//! nor a path reaches a response or a log.

use crate::AppState;
use crate::storage::{StorageError, StoredObject};
use axum::Router;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use futures::TryStreamExt;
use std::fmt::Write as _;
use std::sync::Arc;
use uuid::Uuid;

/// The same bound a chat context puts on the page images it reads.
const MAX_PAGE_IMAGE_BYTES: usize = 25 * 1024 * 1024;

pub(crate) fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/documents/{id}/file", get(original_file))
        .route("/api/documents/{id}/units/{unit_id}/image", get(unit_image))
        .route(
            "/api/documents/{id}/units/{unit_id}/catalog",
            get(catalog_card),
        )
        .route("/api/documents/{id}/units/{unit_id}/text", get(unit_text))
}

/// Every failure a client can see. The text is fixed and safe to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DocumentFileError {
    /// Missing, malformed or not ready: indistinguishable on purpose.
    NotFound,
    Unavailable,
}

impl DocumentFileError {
    fn status(self) -> StatusCode {
        match self {
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::Unavailable => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::NotFound => "找不到這個來源。",
            Self::Unavailable => "來源暫時無法讀取，請稍後再試。",
        }
    }
}

impl IntoResponse for DocumentFileError {
    fn into_response(self) -> Response {
        let mut response = (self.status(), self.message()).into_response();
        set_private(response.headers_mut());
        response
    }
}

fn database_error(stage: &'static str) -> impl FnOnce(sqlx::Error) -> DocumentFileError {
    move |error| {
        tracing::error!(stage, "document file query failed: {error}");
        DocumentFileError::Unavailable
    }
}

/// `StorageError` text never holds a key, a path or a response body.
fn storage_error(stage: &'static str) -> impl FnOnce(StorageError) -> DocumentFileError {
    move |error| match error {
        // A ready row whose object is gone: a delete got there first.
        StorageError::NotFound => {
            tracing::warn!(stage, "a ready document's object is missing");
            DocumentFileError::NotFound
        }
        error => {
            tracing::error!(stage, "document file read failed: {error}");
            DocumentFileError::Unavailable
        }
    }
}

/// Stored content is never cached, so a deleted document stops being viewable,
/// and is never sniffed into something other than what it is declared as.
fn set_private(headers: &mut HeaderMap) {
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
}

fn parse_document_id(id: &str) -> Result<Uuid, DocumentFileError> {
    Uuid::parse_str(id).map_err(|_| DocumentFileError::NotFound)
}

fn parse_unit_id(unit_id: &str) -> Result<i64, DocumentFileError> {
    unit_id.parse().map_err(|_| DocumentFileError::NotFound)
}

/// Streams `object` with `headers`, so a large original never sits in memory.
fn object_response(
    object: StoredObject,
    stage: &'static str,
    headers: impl IntoIterator<Item = (HeaderName, HeaderValue)>,
) -> Response {
    let length = object.size();
    let body = object.into_stream().inspect_err(move |error| {
        tracing::error!(stage, "document file stream failed: {error}");
    });
    let mut response = Body::from_stream(body).into_response();
    let response_headers = response.headers_mut();
    response_headers.extend(headers);
    response_headers.insert(header::CONTENT_LENGTH, HeaderValue::from(length));
    set_private(response_headers);
    response
}

#[derive(Debug, sqlx::FromRow)]
struct UnitRow {
    image_path: Option<String>,
    content: Option<String>,
}

/// The unit as retrieval would see it: ready, and in a ready document.
async fn ready_unit(
    state: &AppState,
    document_id: Uuid,
    unit_id: i64,
) -> Result<UnitRow, DocumentFileError> {
    sqlx::query_as::<_, UnitRow>(
        "SELECT u.image_path, u.content \
         FROM document_units AS u \
         JOIN documents AS d ON d.id = u.document_id \
         WHERE u.document_id = $1 AND u.id = $2 \
           AND u.retrieval_ready AND d.status = 'ready'",
    )
    .bind(document_id)
    .bind(unit_id)
    .fetch_optional(&state.pool)
    .await
    .map_err(database_error("unit_lookup"))?
    .ok_or(DocumentFileError::NotFound)
}

/// Page images are written as PNG; the others are what an image upload may
/// have been stored as before that.
fn image_content_type(key: &str) -> Option<&'static str> {
    let (_, extension) = key.rsplit_once('.')?;
    match extension.to_ascii_lowercase().as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

async fn unit_image(
    State(state): State<Arc<AppState>>,
    Path((id, unit_id)): Path<(String, String)>,
) -> Result<Response, DocumentFileError> {
    let document_id = parse_document_id(&id)?;
    let unit = ready_unit(&state, document_id, parse_unit_id(&unit_id)?).await?;
    let key = unit
        .image_path
        .as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .ok_or(DocumentFileError::NotFound)?;
    let content_type = image_content_type(key).ok_or_else(|| {
        tracing::error!(stage = "unit_image", "page image has an unknown format");
        DocumentFileError::Unavailable
    })?;
    let object = state
        .storage
        .open(document_id, key, MAX_PAGE_IMAGE_BYTES)
        .await
        .map_err(storage_error("unit_image"))?;
    Ok(object_response(
        object,
        "unit_image",
        [(header::CONTENT_TYPE, HeaderValue::from_static(content_type))],
    ))
}

async fn unit_text(
    State(state): State<Arc<AppState>>,
    Path((id, unit_id)): Path<(String, String)>,
) -> Result<Response, DocumentFileError> {
    let document_id = parse_document_id(&id)?;
    let unit = ready_unit(&state, document_id, parse_unit_id(&unit_id)?).await?;
    let text = unit
        .content
        .filter(|text| !text.trim().is_empty())
        .ok_or(DocumentFileError::NotFound)?;
    let mut response = text.into_response();
    set_private(response.headers_mut());
    Ok(response)
}

#[derive(Debug, sqlx::FromRow)]
struct CatalogRow {
    filename: String,
    summary: Option<String>,
    tags: Option<Vec<String>>,
}

/// What a catalog answer cited for this document: its card, built as the
/// answer built it. The citation's `unitId` is the document's first
/// retrieval-ready unit, and it must still be: processing the document again
/// replaces its units and its summary in one transaction, so a citation from
/// before then is not found, as a unit citation would not be.
async fn catalog_card(
    State(state): State<Arc<AppState>>,
    Path((id, unit_id)): Path<(String, String)>,
) -> Result<Response, DocumentFileError> {
    let document_id = parse_document_id(&id)?;
    let document = sqlx::query_as::<_, CatalogRow>(
        "SELECT d.filename, d.summary, d.tags FROM documents AS d \
         WHERE d.id = $1 AND d.status = 'ready' \
           AND $2 = (SELECT MIN(u.id) FROM document_units AS u \
                     WHERE u.document_id = d.id AND u.retrieval_ready)",
    )
    .bind(document_id)
    .bind(parse_unit_id(&unit_id)?)
    .fetch_optional(&state.pool)
    .await
    .map_err(database_error("catalog_lookup"))?
    .ok_or(DocumentFileError::NotFound)?;
    let card = crate::search::catalog_card(
        &document.filename,
        document.summary.as_deref(),
        document.tags.as_deref(),
    );
    let mut response = card.into_response();
    set_private(response.headers_mut());
    Ok(response)
}

#[derive(Debug, sqlx::FromRow)]
struct OriginalRow {
    filename: String,
    storage_path: String,
    mime_type: Option<String>,
}

async fn original_file(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, DocumentFileError> {
    let document_id = parse_document_id(&id)?;
    let original = sqlx::query_as::<_, OriginalRow>(
        "SELECT filename, storage_path, mime_type FROM documents \
         WHERE id = $1 AND status = 'ready'",
    )
    .bind(document_id)
    .fetch_optional(&state.pool)
    .await
    .map_err(database_error("original_lookup"))?
    .ok_or(DocumentFileError::NotFound)?;
    let object = state
        .storage
        .open(
            document_id,
            &original.storage_path,
            crate::ingest::MAX_UPLOAD_BYTES,
        )
        .await
        .map_err(storage_error("original_file"))?;
    let content_type = original
        .mime_type
        .and_then(|mime_type| HeaderValue::try_from(mime_type).ok())
        .unwrap_or(HeaderValue::from_static("application/octet-stream"));
    Ok(object_response(
        object,
        "original_file",
        [
            (header::CONTENT_TYPE, content_type),
            (
                header::CONTENT_DISPOSITION,
                content_disposition(&original.filename),
            ),
            // An upload is someone's file, not part of this site: if a browser
            // ever renders it instead of saving it, it runs with no origin.
            (
                header::CONTENT_SECURITY_POLICY,
                HeaderValue::from_static("sandbox"),
            ),
        ],
    ))
}

/// `attachment` under the uploaded filename (RFC 6266): the exact name as
/// RFC 8187 UTF-8, and an ASCII stand-in for clients that only read `filename`.
fn content_disposition(filename: &str) -> HeaderValue {
    let fallback: String = filename
        .chars()
        .map(|c| {
            if c == ' ' || (c.is_ascii_graphic() && c != '"' && c != '\\') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let mut encoded = String::with_capacity(filename.len());
    for byte in filename.bytes() {
        // attr-char: everything else is percent-encoded.
        if byte.is_ascii_alphanumeric() || b"!#$&+-.^_`|~".contains(&byte) {
            encoded.push(char::from(byte));
        } else {
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    HeaderValue::try_from(format!(
        "attachment; filename=\"{fallback}\"; filename*=UTF-8''{encoded}"
    ))
    .unwrap_or(HeaderValue::from_static("attachment"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disposition(filename: &str) -> String {
        content_disposition(filename)
            .to_str()
            .expect("visible ASCII header")
            .to_owned()
    }

    #[test]
    fn content_disposition_keeps_the_exact_name_in_utf8_and_an_ascii_stand_in() {
        assert_eq!(
            disposition("設備保養手冊 v2.pdf"),
            "attachment; filename=\"______ v2.pdf\"; \
             filename*=UTF-8''%E8%A8%AD%E5%82%99%E4%BF%9D%E9%A4%8A%E6%89%8B%E5%86%8A%20v2.pdf"
        );
        assert_eq!(
            disposition("report.pdf"),
            "attachment; filename=\"report.pdf\"; filename*=UTF-8''report.pdf"
        );
    }

    #[test]
    fn content_disposition_cannot_be_broken_out_of() {
        let header = disposition("a\"b\\c\r\nSet-Cookie: x=1;\t.txt");
        assert_eq!(
            header,
            "attachment; filename=\"a_b_c__Set-Cookie: x=1;_.txt\"; \
             filename*=UTF-8''a%22b%5Cc%0D%0ASet-Cookie%3A%20x%3D1%3B%09.txt"
        );
        assert_eq!(
            disposition("報告"),
            "attachment; filename=\"__\"; filename*=UTF-8''%E5%A0%B1%E5%91%8A"
        );
    }

    #[test]
    fn unit_and_document_ids_that_do_not_parse_are_not_found() {
        assert_eq!(
            parse_document_id("../etc"),
            Err(DocumentFileError::NotFound)
        );
        assert_eq!(parse_unit_id("1.5"), Err(DocumentFileError::NotFound));
        assert_eq!(parse_unit_id("７"), Err(DocumentFileError::NotFound));
        assert_eq!(parse_unit_id("42"), Ok(42));
    }

    #[test]
    fn page_images_are_served_as_the_format_their_key_names() {
        assert_eq!(
            image_content_type("d/artifacts-r/pages/0001.png"),
            Some("image/png")
        );
        assert_eq!(image_content_type("d/pages/0001.JPG"), Some("image/jpeg"));
        assert_eq!(image_content_type("d/pages/0001.svg"), None);
        assert_eq!(image_content_type("d/pages/0001"), None);
    }
}
