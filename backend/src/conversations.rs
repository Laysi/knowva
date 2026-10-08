//! Conversation history, kept per browser (issue #52).
//!
//! There is no login, so each browser is an anonymous owner named by the
//! `knowva_client` cookie: 256 random bits this module issues as an HttpOnly
//! cookie. Only the token's SHA-256 is stored or compared, and the token never
//! reaches a log or an error. Every query is scoped to the owner, and anything
//! that belongs to someone else answers 404 exactly like something that never
//! existed. `/api/chat` stays stateless: the frontend saves each finished
//! message here.

use crate::AppState;
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, Extension, Path, Request, State};
use axum::http::uri::Scheme;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use uuid::Uuid;

pub(crate) const CLIENT_COOKIE: &str = "knowva_client";
const CLIENT_TOKEN_BYTES: usize = 32;
/// 32 bytes in unpadded base64url.
const CLIENT_TOKEN_CHARS: usize = 43;
const CLIENT_COOKIE_MAX_AGE_SECS: u64 = 400 * 24 * 60 * 60;
const ANONYMOUS_OWNER: &str = "anonymous";

/// Past this, creating a conversation deletes the owner's least recently updated one.
pub(crate) const MAX_CONVERSATIONS: i64 = 50;
pub(crate) const MAX_MESSAGES_PER_CONVERSATION: i64 = 200;
/// Measured on the message re-serialized as compact JSON.
pub(crate) const MAX_MESSAGE_BYTES: usize = 256 * 1024;
const MAX_MESSAGE_ID_CHARS: usize = 128;
pub(crate) const TITLE_CHARS: usize = 30;
/// Room for `{"parentId":…,"message":…}` around a message at its limit.
const MESSAGE_BODY_LIMIT_BYTES: usize = MAX_MESSAGE_BYTES + 4 * 1024;

const SUMMARY_COLUMNS: &str = "id, title, created_at, updated_at";

/// Who the request acts for. Built only by [`identify_owner`].
#[derive(Debug, Clone)]
pub(crate) struct Owner {
    kind: &'static str,
    id: String,
}

impl Owner {
    fn anonymous(token: &str) -> Self {
        Self {
            kind: ANONYMOUS_OWNER,
            id: token_hash(token),
        }
    }
}

/// Every failure a client can see. The text is fixed and safe to show; causes go
/// to the log without the token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConversationError {
    /// Missing, malformed or someone else's: indistinguishable on purpose.
    NotFound,
    InvalidMessage,
    MessageTooLarge,
    TooManyMessages,
    MissingParent,
    Unavailable,
}

impl ConversationError {
    fn status(self) -> StatusCode {
        match self {
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::InvalidMessage => StatusCode::BAD_REQUEST,
            Self::MessageTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::TooManyMessages | Self::MissingParent => StatusCode::CONFLICT,
            Self::Unavailable => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::NotFound => "找不到這串對話。",
            Self::InvalidMessage => "訊息格式不正確，無法保存。",
            Self::MessageTooLarge => "單則訊息超過 256 KiB，無法保存。",
            Self::TooManyMessages => "這串對話已達 200 則訊息，無法再保存。",
            Self::MissingParent => "前一則訊息尚未保存，這則訊息無法保存。",
            Self::Unavailable => "對話紀錄暫時無法使用，請稍後再試。",
        }
    }
}

impl IntoResponse for ConversationError {
    fn into_response(self) -> Response {
        (self.status(), self.message()).into_response()
    }
}

fn database_error(stage: &'static str) -> impl FnOnce(sqlx::Error) -> ConversationError {
    move |error| {
        tracing::error!(stage, "conversation query failed: {error}");
        ConversationError::Unavailable
    }
}

pub(crate) fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/api/conversations",
            get(list_conversations)
                .post(create_conversation)
                .delete(clear_conversations),
        )
        .route(
            "/api/conversations/{id}",
            get(fetch_conversation).delete(delete_conversation),
        )
        .route(
            "/api/conversations/{id}/messages",
            get(list_messages)
                .post(save_message)
                .layer(DefaultBodyLimit::max(MESSAGE_BODY_LIMIT_BYTES)),
        )
        .route_layer(middleware::from_fn(identify_owner))
}

fn new_client_token() -> Result<String, getrandom::Error> {
    let mut bytes = [0_u8; CLIENT_TOKEN_BYTES];
    getrandom::fill(&mut bytes)?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

/// Exactly what [`new_client_token`] produces; anything else is treated as absent.
fn is_client_token(value: &str) -> bool {
    value.len() == CLIENT_TOKEN_CHARS
        && URL_SAFE_NO_PAD
            .decode(value)
            .is_ok_and(|bytes| bytes.len() == CLIENT_TOKEN_BYTES)
}

/// The first well-formed `knowva_client` value across every `Cookie` header.
fn presented_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(';'))
        .filter_map(|pair| pair.trim().split_once('='))
        .filter(|(name, _)| *name == CLIENT_COOKIE)
        .map(|(_, value)| value)
        .find(|value| is_client_token(value))
}

fn token_hash(token: &str) -> String {
    Sha256::digest(token.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The backend never terminates TLS itself; a proxy in front says so.
fn arrived_over_https(request: &Request) -> bool {
    request.uri().scheme() == Some(&Scheme::HTTPS)
        || request
            .headers()
            .get("x-forwarded-proto")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(',').next())
            .is_some_and(|proto| proto.trim().eq_ignore_ascii_case("https"))
}

/// `Secure` only over HTTPS: the appliance may well be plain HTTP on the intranet,
/// where a Secure cookie would never come back.
fn client_cookie(token: &str, secure: bool) -> Option<HeaderValue> {
    let secure = if secure { "; Secure" } else { "" };
    HeaderValue::from_str(&format!(
        "{CLIENT_COOKIE}={token}; Path=/api; Max-Age={CLIENT_COOKIE_MAX_AGE_SECS}; HttpOnly; \
         SameSite=Strict{secure}"
    ))
    .ok()
}

/// Names the owner of every conversation request, issuing a new cookie when the
/// request has none or a malformed one. Responses are never cached.
async fn identify_owner(mut request: Request, next: Next) -> Response {
    let (owner, cookie) = match presented_token(request.headers()) {
        Some(token) => (Owner::anonymous(token), None),
        None => {
            let issued = new_client_token().ok().and_then(|token| {
                client_cookie(&token, arrived_over_https(&request))
                    .map(|cookie| (Owner::anonymous(&token), cookie))
            });
            let Some((owner, cookie)) = issued else {
                tracing::error!(stage = "client_cookie", "could not issue a client cookie");
                return ConversationError::Unavailable.into_response();
            };
            (owner, Some(cookie))
        }
    };
    request.extensions_mut().insert(owner);

    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    if let Some(cookie) = cookie {
        headers.append(header::SET_COOKIE, cookie);
        tracing::info!(stage = "client_cookie", "issued an anonymous client cookie");
    }
    response
}

/// The first 30 Unicode code points of a question, whitespace runs folded into one
/// space and the ends trimmed. No model is involved.
pub(crate) fn conversation_title(question: &str) -> Option<String> {
    let folded = question.split_whitespace().collect::<Vec<_>>().join(" ");
    let title = folded.chars().take(TITLE_CHARS).collect::<String>();
    (!title.is_empty()).then_some(title)
}

#[derive(Debug, sqlx::FromRow, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ConversationSummary {
    id: Uuid,
    title: Option<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CreatedConversation {
    conversation: ConversationSummary,
    /// Older conversations deleted to stay at the limit.
    removed_ids: Vec<Uuid>,
}

#[derive(Debug, Serialize)]
struct SavedMessage {
    conversation: ConversationSummary,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StoredMessage {
    parent_id: Option<String>,
    message: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveMessageRequest {
    #[serde(default)]
    parent_id: Option<String>,
    message: Value,
}

/// A message that passed every check, with what the handler needs from it.
#[derive(Debug)]
struct ValidMessage {
    id: String,
    parent_id: Option<String>,
    /// The text of a user message, for the title.
    question: Option<String>,
    message: Value,
}

fn is_message_id(value: &str) -> bool {
    !value.is_empty() && value.chars().count() <= MAX_MESSAGE_ID_CHARS
}

/// PostgreSQL's jsonb cannot hold U+0000 anywhere.
fn holds_nul(value: &Value) -> bool {
    match value {
        Value::String(text) => text.contains('\0'),
        Value::Array(items) => items.iter().any(holds_nul),
        Value::Object(fields) => fields
            .iter()
            .any(|(key, value)| key.contains('\0') || holds_nul(value)),
        _ => false,
    }
}

impl ValidMessage {
    fn parse(request: SaveMessageRequest) -> Result<Self, ConversationError> {
        let size = serde_json::to_vec(&request.message)
            .map_err(|_| ConversationError::InvalidMessage)?
            .len();
        if size > MAX_MESSAGE_BYTES {
            return Err(ConversationError::MessageTooLarge);
        }
        let Value::Object(fields) = &request.message else {
            return Err(ConversationError::InvalidMessage);
        };
        let id = fields
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| is_message_id(id))
            .ok_or(ConversationError::InvalidMessage)?;
        let role = fields.get("role").and_then(Value::as_str);
        if !matches!(role, Some("user" | "assistant")) {
            return Err(ConversationError::InvalidMessage);
        }
        let content = fields
            .get("content")
            .and_then(Value::as_array)
            .ok_or(ConversationError::InvalidMessage)?;
        if let Some(parent_id) = &request.parent_id
            && (!is_message_id(parent_id) || parent_id == id)
        {
            return Err(ConversationError::InvalidMessage);
        }
        if holds_nul(&request.message) {
            return Err(ConversationError::InvalidMessage);
        }
        let question = (role == Some("user")).then(|| {
            content
                .iter()
                .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        });
        Ok(Self {
            id: id.to_owned(),
            parent_id: request.parent_id,
            question,
            message: request.message,
        })
    }
}

fn conversation_id(raw: &str) -> Result<Uuid, ConversationError> {
    Uuid::parse_str(raw).map_err(|_| ConversationError::NotFound)
}

async fn list_conversations(
    State(state): State<Arc<AppState>>,
    Extension(owner): Extension<Owner>,
) -> Result<Json<Vec<ConversationSummary>>, ConversationError> {
    let conversations = sqlx::query_as::<_, ConversationSummary>(&format!(
        "SELECT {SUMMARY_COLUMNS} FROM conversations \
         WHERE owner_kind = $1 AND owner_id = $2 \
         ORDER BY updated_at DESC, created_at DESC, id DESC"
    ))
    .bind(owner.kind)
    .bind(&owner.id)
    .fetch_all(&state.pool)
    .await
    .map_err(database_error("conversation_list"))?;
    Ok(Json(conversations))
}

async fn create_conversation(
    State(state): State<Arc<AppState>>,
    Extension(owner): Extension<Owner>,
) -> Result<(StatusCode, Json<CreatedConversation>), ConversationError> {
    let mut tx = state
        .pool
        .begin()
        .await
        .map_err(database_error("conversation_create"))?;
    // One owner's creations take turns, so two at once cannot both keep 50 others.
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
        .bind(format!("conversations:{}:{}", owner.kind, owner.id))
        .execute(&mut *tx)
        .await
        .map_err(database_error("conversation_create"))?;
    let conversation = sqlx::query_as::<_, ConversationSummary>(&format!(
        "INSERT INTO conversations (owner_kind, owner_id) VALUES ($1, $2) \
         RETURNING {SUMMARY_COLUMNS}"
    ))
    .bind(owner.kind)
    .bind(&owner.id)
    .fetch_one(&mut *tx)
    .await
    .map_err(database_error("conversation_create"))?;
    let removed_ids = sqlx::query_scalar::<_, Uuid>(
        "DELETE FROM conversations WHERE id IN ( \
           SELECT id FROM conversations \
           WHERE owner_kind = $1 AND owner_id = $2 AND id <> $3 \
           ORDER BY updated_at DESC, created_at DESC, id DESC \
           OFFSET $4 \
         ) RETURNING id",
    )
    .bind(owner.kind)
    .bind(&owner.id)
    .bind(conversation.id)
    .bind(MAX_CONVERSATIONS - 1)
    .fetch_all(&mut *tx)
    .await
    .map_err(database_error("conversation_trim"))?;
    tx.commit()
        .await
        .map_err(database_error("conversation_create"))?;
    Ok((
        StatusCode::CREATED,
        Json(CreatedConversation {
            conversation,
            removed_ids,
        }),
    ))
}

async fn fetch_conversation(
    State(state): State<Arc<AppState>>,
    Extension(owner): Extension<Owner>,
    Path(id): Path<String>,
) -> Result<Json<ConversationSummary>, ConversationError> {
    let id = conversation_id(&id)?;
    sqlx::query_as::<_, ConversationSummary>(&format!(
        "SELECT {SUMMARY_COLUMNS} FROM conversations \
         WHERE id = $1 AND owner_kind = $2 AND owner_id = $3"
    ))
    .bind(id)
    .bind(owner.kind)
    .bind(&owner.id)
    .fetch_optional(&state.pool)
    .await
    .map_err(database_error("conversation_fetch"))?
    .map(Json)
    .ok_or(ConversationError::NotFound)
}

async fn delete_conversation(
    State(state): State<Arc<AppState>>,
    Extension(owner): Extension<Owner>,
    Path(id): Path<String>,
) -> Result<StatusCode, ConversationError> {
    let id = conversation_id(&id)?;
    let deleted = sqlx::query(
        "DELETE FROM conversations WHERE id = $1 AND owner_kind = $2 AND owner_id = $3",
    )
    .bind(id)
    .bind(owner.kind)
    .bind(&owner.id)
    .execute(&state.pool)
    .await
    .map_err(database_error("conversation_delete"))?;
    if deleted.rows_affected() == 0 {
        return Err(ConversationError::NotFound);
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn clear_conversations(
    State(state): State<Arc<AppState>>,
    Extension(owner): Extension<Owner>,
) -> Result<StatusCode, ConversationError> {
    sqlx::query("DELETE FROM conversations WHERE owner_kind = $1 AND owner_id = $2")
        .bind(owner.kind)
        .bind(&owner.id)
        .execute(&state.pool)
        .await
        .map_err(database_error("conversation_clear"))?;
    Ok(StatusCode::NO_CONTENT)
}

async fn list_messages(
    State(state): State<Arc<AppState>>,
    Extension(owner): Extension<Owner>,
    Path(id): Path<String>,
) -> Result<Json<Vec<StoredMessage>>, ConversationError> {
    let id = conversation_id(&id)?;
    let mut tx = state
        .pool
        .begin()
        .await
        .map_err(database_error("conversation_messages"))?;
    let owned = sqlx::query_scalar::<_, Uuid>(
        "SELECT id FROM conversations WHERE id = $1 AND owner_kind = $2 AND owner_id = $3",
    )
    .bind(id)
    .bind(owner.kind)
    .bind(&owner.id)
    .fetch_optional(&mut *tx)
    .await
    .map_err(database_error("conversation_messages"))?;
    if owned.is_none() {
        return Err(ConversationError::NotFound);
    }
    let rows = sqlx::query_as::<_, (Option<String>, sqlx::types::Json<Value>)>(
        "SELECT parent_id, message FROM conversation_messages \
         WHERE conversation_id = $1 ORDER BY position",
    )
    .bind(id)
    .fetch_all(&mut *tx)
    .await
    .map_err(database_error("conversation_messages"))?;
    tx.commit()
        .await
        .map_err(database_error("conversation_messages"))?;
    Ok(Json(
        rows.into_iter()
            .map(|(parent_id, message)| StoredMessage {
                parent_id,
                message: message.0,
            })
            .collect(),
    ))
}

/// Saves one message, replacing the one with the same id if it was saved before, and
/// marks the conversation updated. The first question names the conversation.
async fn save_message(
    State(state): State<Arc<AppState>>,
    Extension(owner): Extension<Owner>,
    Path(id): Path<String>,
    body: Result<Json<SaveMessageRequest>, JsonRejection>,
) -> Result<Json<SavedMessage>, ConversationError> {
    let id = conversation_id(&id)?;
    let Json(request) = body.map_err(|rejection| {
        if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ConversationError::MessageTooLarge
        } else {
            ConversationError::InvalidMessage
        }
    })?;
    let message = ValidMessage::parse(request)?;

    let mut tx = state
        .pool
        .begin()
        .await
        .map_err(database_error("conversation_save"))?;
    // Locking the conversation row puts its saves in turn, so positions and the
    // message count stay exact.
    let owned = sqlx::query_scalar::<_, Uuid>(
        "SELECT id FROM conversations \
         WHERE id = $1 AND owner_kind = $2 AND owner_id = $3 FOR UPDATE",
    )
    .bind(id)
    .bind(owner.kind)
    .bind(&owner.id)
    .fetch_optional(&mut *tx)
    .await
    .map_err(database_error("conversation_save"))?;
    if owned.is_none() {
        return Err(ConversationError::NotFound);
    }

    let position_of = |message_id: String| {
        sqlx::query_scalar::<_, i32>(
            "SELECT position FROM conversation_messages \
             WHERE conversation_id = $1 AND message_id = $2",
        )
        .bind(id)
        .bind(message_id)
    };
    let existing = position_of(message.id.clone())
        .fetch_optional(&mut *tx)
        .await
        .map_err(database_error("conversation_save"))?;
    if let Some(parent_id) = &message.parent_id {
        // The parent has to be saved already, and before this message.
        let parent = position_of(parent_id.clone())
            .fetch_optional(&mut *tx)
            .await
            .map_err(database_error("conversation_save"))?;
        match (parent, existing) {
            (None, _) => return Err(ConversationError::MissingParent),
            (Some(parent), Some(existing)) if parent >= existing => {
                return Err(ConversationError::MissingParent);
            }
            _ => {}
        }
    }

    if existing.is_some() {
        sqlx::query(
            "UPDATE conversation_messages \
             SET parent_id = $3, message = $4, updated_at = now() \
             WHERE conversation_id = $1 AND message_id = $2",
        )
        .bind(id)
        .bind(&message.id)
        .bind(&message.parent_id)
        .bind(sqlx::types::Json(&message.message))
        .execute(&mut *tx)
        .await
        .map_err(database_error("conversation_save"))?;
    } else {
        let count = sqlx::query_scalar::<_, i64>(
            "SELECT count(*) FROM conversation_messages WHERE conversation_id = $1",
        )
        .bind(id)
        .fetch_one(&mut *tx)
        .await
        .map_err(database_error("conversation_save"))?;
        if count >= MAX_MESSAGES_PER_CONVERSATION {
            return Err(ConversationError::TooManyMessages);
        }
        sqlx::query(
            "INSERT INTO conversation_messages \
             (conversation_id, message_id, parent_id, position, message) \
             VALUES ($1, $2, $3, \
                     (SELECT COALESCE(MAX(position) + 1, 0) FROM conversation_messages \
                      WHERE conversation_id = $1), \
                     $4)",
        )
        .bind(id)
        .bind(&message.id)
        .bind(&message.parent_id)
        .bind(sqlx::types::Json(&message.message))
        .execute(&mut *tx)
        .await
        .map_err(database_error("conversation_save"))?;
    }

    let title = message.question.as_deref().and_then(conversation_title);
    let conversation = sqlx::query_as::<_, ConversationSummary>(&format!(
        "UPDATE conversations SET title = COALESCE(title, $2), updated_at = now() \
         WHERE id = $1 RETURNING {SUMMARY_COLUMNS}"
    ))
    .bind(id)
    .bind(title)
    .fetch_one(&mut *tx)
    .await
    .map_err(database_error("conversation_save"))?;
    tx.commit()
        .await
        .map_err(database_error("conversation_save"))?;
    Ok(Json(SavedMessage { conversation }))
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    pub(crate) fn new_token() -> String {
        new_client_token().expect("draw a client token")
    }

    pub(crate) fn hash(token: &str) -> String {
        token_hash(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use tower::ServiceExt;

    fn probe() -> Router {
        Router::new()
            .route(
                "/api/probe",
                get(|Extension(owner): Extension<Owner>| async move { owner.id }),
            )
            .route_layer(middleware::from_fn(identify_owner))
    }

    async fn call(cookie: Option<&str>, forwarded_proto: Option<&str>) -> (Response, String) {
        let mut request = axum::http::Request::get("/api/probe");
        if let Some(cookie) = cookie {
            request = request.header(header::COOKIE, cookie);
        }
        if let Some(proto) = forwarded_proto {
            request = request.header("x-forwarded-proto", proto);
        }
        let response = probe()
            .oneshot(request.body(Body::empty()).expect("request"))
            .await
            .expect("route the request");
        let (parts, body) = response.into_parts();
        let body = to_bytes(body, usize::MAX).await.expect("body");
        (
            Response::from_parts(parts, Body::empty()),
            String::from_utf8(body.to_vec()).expect("UTF-8 body"),
        )
    }

    fn set_cookie(response: &Response) -> Option<String> {
        response
            .headers()
            .get(header::SET_COOKIE)
            .map(|value| value.to_str().expect("ASCII cookie").to_owned())
    }

    fn issued_token(cookie: &str) -> &str {
        cookie
            .strip_prefix("knowva_client=")
            .and_then(|rest| rest.split(';').next())
            .expect("cookie value")
    }

    #[test]
    fn client_tokens_are_256_random_bits_in_base64url() {
        let first = new_client_token().expect("token");
        let second = new_client_token().expect("token");
        assert_ne!(first, second);
        for token in [&first, &second] {
            assert_eq!(token.len(), 43);
            assert!(is_client_token(token));
            assert_eq!(URL_SAFE_NO_PAD.decode(token).expect("decode").len(), 32);
        }
    }

    #[test]
    fn only_a_well_formed_client_cookie_is_read() {
        let token = new_client_token().expect("token");
        let mut headers = HeaderMap::new();
        headers.append(
            header::COOKIE,
            HeaderValue::from_str(&format!(
                "theme=dark; knowva_client=short; knowva_client={token}; other=1"
            ))
            .expect("header"),
        );
        assert_eq!(presented_token(&headers), Some(token.as_str()));

        let padded = format!("{}=", &token[..42]);
        let wrong_alphabet = format!("{}+", &token[..42]);
        for bad in [
            String::new(),
            "knowva_client=".to_string(),
            format!("knowva_client={padded}"),
            format!("knowva_client={wrong_alphabet}"),
            format!("knowva_client={token}A"),
            format!("Knowva_Client={token}"),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(header::COOKIE, HeaderValue::from_str(&bad).expect("header"));
            assert_eq!(presented_token(&headers), None, "{bad:?}");
        }
    }

    #[test]
    fn owners_are_named_by_the_token_hash() {
        assert_eq!(
            token_hash("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let token = new_client_token().expect("token");
        let owner = Owner::anonymous(&token);
        assert_eq!(owner.kind, "anonymous");
        assert_eq!(owner.id, token_hash(&token));
        assert!(!owner.id.contains(&token));
    }

    #[tokio::test]
    async fn a_request_without_a_cookie_gets_an_http_only_strict_api_cookie() {
        let (response, owner_id) = call(None, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let cookie = set_cookie(&response).expect("a cookie is issued");
        let token = issued_token(&cookie);
        assert!(is_client_token(token));
        assert_eq!(owner_id, token_hash(token));
        let attributes = cookie.split("; ").skip(1).collect::<Vec<_>>();
        assert_eq!(
            attributes,
            [
                "Path=/api",
                "Max-Age=34560000",
                "HttpOnly",
                "SameSite=Strict"
            ]
        );
        assert_eq!(
            response.headers().get(header::CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store"))
        );
    }

    #[tokio::test]
    async fn only_a_request_that_arrived_over_https_gets_a_secure_cookie() {
        let (response, _) = call(None, Some("https")).await;
        let cookie = set_cookie(&response).expect("a cookie is issued");
        assert!(cookie.ends_with("; Secure"), "{cookie}");

        let (response, _) = call(None, Some("HTTPS, http")).await;
        assert!(set_cookie(&response).expect("cookie").ends_with("; Secure"));

        for proto in ["http", "http, https"] {
            let (response, _) = call(None, Some(proto)).await;
            assert!(!set_cookie(&response).expect("cookie").contains("Secure"));
        }
    }

    #[tokio::test]
    async fn a_valid_cookie_is_kept_and_a_malformed_one_replaced() {
        let token = new_client_token().expect("token");
        let (response, owner_id) = call(Some(&format!("knowva_client={token}")), None).await;
        assert!(set_cookie(&response).is_none());
        assert_eq!(owner_id, token_hash(&token));

        let (response, owner_id) = call(Some("knowva_client=not-a-token"), None).await;
        let cookie = set_cookie(&response).expect("a new cookie is issued");
        assert_ne!(issued_token(&cookie), "not-a-token");
        assert_eq!(owner_id, token_hash(issued_token(&cookie)));
    }

    #[test]
    fn titles_are_the_first_30_code_points_of_the_folded_question() {
        assert_eq!(
            conversation_title("  馬達\n\n多久\t保養一次？ ").as_deref(),
            Some("馬達 多久 保養一次？")
        );
        let astral = "𠀋".repeat(31);
        assert_eq!(conversation_title(&astral), Some("𠀋".repeat(30)));
        assert_eq!(
            conversation_title(&format!("　{}", "一".repeat(40)))
                .map(|title| title.chars().count()),
            Some(30)
        );
        assert_eq!(conversation_title(" \n\t　"), None);
    }

    fn request(parent_id: Option<&str>, message: Value) -> SaveMessageRequest {
        SaveMessageRequest {
            parent_id: parent_id.map(str::to_owned),
            message,
        }
    }

    #[test]
    fn messages_are_checked_before_they_are_saved() {
        let valid = ValidMessage::parse(request(
            Some("q1"),
            serde_json::json!({
                "id": "a1",
                "role": "assistant",
                "content": [{ "type": "text", "text": "回答" }]
            }),
        ))
        .expect("valid assistant message");
        assert_eq!(valid.id, "a1");
        assert_eq!(valid.question, None);

        let question = ValidMessage::parse(request(
            None,
            serde_json::json!({
                "id": "q1",
                "role": "user",
                "content": [
                    { "type": "text", "text": "第一行" },
                    { "type": "image", "image": "ignored" },
                    { "type": "text", "text": "第二行" }
                ]
            }),
        ))
        .expect("valid user message");
        assert_eq!(question.question.as_deref(), Some("第一行\n第二行"));

        let long_id = "m".repeat(MAX_MESSAGE_ID_CHARS + 1);
        for (parent_id, message) in [
            (None, serde_json::json!(["not", "an", "object"])),
            (None, serde_json::json!({ "role": "user", "content": [] })),
            (
                None,
                serde_json::json!({ "id": "", "role": "user", "content": [] }),
            ),
            (
                None,
                serde_json::json!({ "id": long_id, "role": "user", "content": [] }),
            ),
            (
                None,
                serde_json::json!({ "id": "m1", "role": "system", "content": [] }),
            ),
            (
                None,
                serde_json::json!({ "id": "m1", "role": "user", "content": "text" }),
            ),
            (
                Some("m1"),
                serde_json::json!({ "id": "m1", "role": "user", "content": [] }),
            ),
            (
                Some(long_id.as_str()),
                serde_json::json!({ "id": "m1", "role": "user", "content": [] }),
            ),
            (
                None,
                serde_json::json!({ "id": "m1", "role": "user", "content": [{ "type": "text", "text": "a\u{0}b" }] }),
            ),
        ] {
            assert_eq!(
                ValidMessage::parse(request(parent_id, message.clone())).map(|_| ()),
                Err(ConversationError::InvalidMessage),
                "{parent_id:?} {message}"
            );
        }

        let oversized = serde_json::json!({
            "id": "m1",
            "role": "assistant",
            "content": [{ "type": "text", "text": "長".repeat(MAX_MESSAGE_BYTES / 3) }]
        });
        assert_eq!(
            ValidMessage::parse(request(None, oversized)).map(|_| ()),
            Err(ConversationError::MessageTooLarge)
        );
    }

    #[test]
    fn errors_answer_with_fixed_safe_text() {
        for error in [
            ConversationError::NotFound,
            ConversationError::InvalidMessage,
            ConversationError::MessageTooLarge,
            ConversationError::TooManyMessages,
            ConversationError::MissingParent,
            ConversationError::Unavailable,
        ] {
            assert!(error.status().is_client_error() || error.status().is_server_error());
            assert!(!error.message().is_empty());
        }
        assert_eq!(ConversationError::NotFound.status(), StatusCode::NOT_FOUND);
    }
}
