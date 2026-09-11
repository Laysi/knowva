use base64::Engine;
use futures::StreamExt;
use reqwest::header::CONTENT_TYPE;
use reqwest::{Response, StatusCode, Url};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::mpsc::Sender;

pub const SYSTEM_PROMPT: &str =
    "你是企業內部的 AI 知識管理助理,依據知識庫內容協助回答問題與整理資訊。一律使用繁體中文回答。";

pub const EMBED_MODEL_FAMILY: &str = "llama-nemotron-embed-vl-1b-v2";
pub const RERANK_MODEL_FAMILY: &str = "llama-nemotron-rerank-vl-1b-v2";
pub const DEFAULT_EMBED_MODEL_ID: &str = "nvidia/llama-nemotron-embed-vl-1b-v2";
pub const DEFAULT_RERANK_MODEL_ID: &str = "nvidia/llama-nemotron-rerank-vl-1b-v2";
pub const EMBEDDING_DIMENSIONS: usize = 2048;
pub const MAX_EMBED_IMAGE_BYTES: usize = 25 * 1024 * 1024;
pub const DEFAULT_RERANK_CANDIDATE_TOP_K: usize = 50;
pub const DEFAULT_RERANK_TOP_N: usize = 6;
pub const DEFAULT_RERANK_BATCH_SIZE: usize = 8;
pub const MIN_RERANK_TOP_N: usize = 3;
pub const MAX_RERANK_TOP_N: usize = 8;
pub const MAX_RERANK_BATCH_SIZE: usize = 512;

/// All cloud model traffic is routed through a Cloudflare AI Gateway. The two
/// NVIDIA hosts are registered on the gateway as custom providers with the
/// slugs `nvidia-build` (https://integrate.api.nvidia.com) and `nvidia-ai`
/// (https://ai.api.nvidia.com); requests address them as `custom-<slug>`.
/// Provider API keys are stored gateway-side via BYOK, so the backend never
/// sends an upstream Authorization header; its only secret is the gateway
/// Run token sent as `cf-aig-authorization`.
const GATEWAY_NVIDIA_BUILD_SEGMENT: &str = "custom-nvidia-build";
const GATEWAY_NVIDIA_AI_SEGMENT: &str = "custom-nvidia-ai";
const GATEWAY_BASE_URL_HINT: &str =
    "https://gateway.ai.cloudflare.com/v1/<account_id>/<gateway_id>";
const GATEWAY_AUTHORIZATION_HEADER: &str = "cf-aig-authorization";
const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_CHAT_REQUEST_BYTES: usize = 24 * 1024 * 1024;
const MAX_EMBED_REQUEST_BYTES: usize = 36 * 1024 * 1024;
const DEFAULT_EMBED_MAX_ATTEMPTS: usize = 3;
const DEFAULT_EMBED_RETRY_BASE_DELAY: Duration = Duration::from_millis(250);
const DEFAULT_EMBED_RETRY_MAX_DELAY: Duration = Duration::from_secs(2);
const DEFAULT_RERANK_MAX_ATTEMPTS: usize = 3;
const DEFAULT_RERANK_RETRY_BASE_DELAY: Duration = Duration::from_millis(250);
const DEFAULT_RERANK_RETRY_MAX_DELAY: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum SseEvent {
    #[serde(rename = "text-delta")]
    TextDelta { text: String },
    #[serde(rename = "sources")]
    Sources {
        sources: Vec<crate::rag_context::SourceRef>,
    },
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "error")]
    Error { message: String },
}

#[derive(Clone)]
pub struct CloudModelConfig {
    gateway_token: String,
    chat_endpoint: Url,
    embed_endpoint: Url,
    rerank_endpoint: Url,
    chat_model: String,
    embed_model: String,
    rerank_model: String,
    request_timeout: Duration,
    rerank_candidate_top_k: usize,
    rerank_top_n: usize,
    rerank_batch_size: usize,
}

impl fmt::Debug for CloudModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CloudModelConfig")
            .field("gateway_token", &"[REDACTED]")
            .field("chat_endpoint", &self.chat_endpoint)
            .field("embed_endpoint", &self.embed_endpoint)
            .field("rerank_endpoint", &self.rerank_endpoint)
            .field("chat_model", &self.chat_model)
            .field("embed_model", &self.embed_model)
            .field("rerank_model", &self.rerank_model)
            .field("request_timeout", &self.request_timeout)
            .field("rerank_candidate_top_k", &self.rerank_candidate_top_k)
            .field("rerank_top_n", &self.rerank_top_n)
            .field("rerank_batch_size", &self.rerank_batch_size)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudModelConfigError(String);

impl fmt::Display for CloudModelConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CloudModelConfigError {}

impl CloudModelConfig {
    pub fn from_env() -> Result<Self, CloudModelConfigError> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    pub(crate) fn from_lookup(
        mut get: impl FnMut(&str) -> Option<String>,
    ) -> Result<Self, CloudModelConfigError> {
        // Provider API keys live in the Cloudflare AI Gateway BYOK store; the
        // gateway Run token is the backend's only secret and BYOK requires the
        // gateway to be authenticated, so the token is mandatory.
        let gateway_token = required_value(&mut get, "CF_AI_GATEWAY_TOKEN")?;
        let chat_model = required_value(&mut get, "GEMMA_4_MODEL_ID")?;
        if !is_gemma_4_model_id(&chat_model) {
            return Err(CloudModelConfigError(
                "GEMMA_4_MODEL_ID must identify a cloud-hosted Gemma 4 model".to_string(),
            ));
        }

        let embed_model = optional_value(&mut get, "NEMOTRON_EMBED_MODEL_ID")
            .unwrap_or_else(|| DEFAULT_EMBED_MODEL_ID.to_string());
        validate_locked_model(
            "NEMOTRON_EMBED_MODEL_ID",
            &embed_model,
            DEFAULT_EMBED_MODEL_ID,
        )?;

        let rerank_model = optional_value(&mut get, "NEMOTRON_RERANK_MODEL_ID")
            .unwrap_or_else(|| DEFAULT_RERANK_MODEL_ID.to_string());
        validate_locked_model(
            "NEMOTRON_RERANK_MODEL_ID",
            &rerank_model,
            DEFAULT_RERANK_MODEL_ID,
        )?;

        let gateway_base = optional_value(&mut get, "CF_AI_GATEWAY_BASE_URL")
            .map(|value| value.trim_end_matches('/').to_string());

        let chat_endpoint = capability_endpoint(
            &mut get,
            "CLOUD_CHAT_URL",
            &gateway_base,
            &format!("{GATEWAY_NVIDIA_BUILD_SEGMENT}/chat/completions"),
        )?;
        let embed_endpoint = capability_endpoint(
            &mut get,
            "CLOUD_EMBED_URL",
            &gateway_base,
            &format!("{GATEWAY_NVIDIA_BUILD_SEGMENT}/embeddings"),
        )?;
        let embed_model = optional_value(&mut get, "NEMOTRON_EMBED_MODEL_ID")
            .unwrap_or_else(|| DEFAULT_EMBED_MODEL_ID.to_string());
        let rerank_endpoint = capability_endpoint(
            &mut get,
            "CLOUD_RERANK_URL",
            &gateway_base,
            &format!(
                "{GATEWAY_NVIDIA_AI_SEGMENT}/retrieval/nvidia/{RERANK_MODEL_FAMILY}/reranking"
            ),
        )?;
        validate_production_gateway_endpoints(&chat_endpoint, &embed_endpoint, &rerank_endpoint)?;

        let timeout_secs = match optional_value(&mut get, "CLOUD_MODEL_REQUEST_TIMEOUT_SECS") {
            Some(raw) => raw
                .parse::<u64>()
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    CloudModelConfigError(
                        "CLOUD_MODEL_REQUEST_TIMEOUT_SECS must be a positive integer".to_string(),
                    )
                })?,
            None => DEFAULT_TIMEOUT_SECS,
        };
        let rerank_candidate_top_k = positive_usize_setting(
            &mut get,
            "RERANK_CANDIDATE_TOP_K",
            DEFAULT_RERANK_CANDIDATE_TOP_K,
        )?;
        let rerank_top_n = positive_usize_setting(&mut get, "RERANK_TOP_N", DEFAULT_RERANK_TOP_N)?;
        if !(MIN_RERANK_TOP_N..=MAX_RERANK_TOP_N).contains(&rerank_top_n) {
            return Err(CloudModelConfigError(format!(
                "RERANK_TOP_N must be between {MIN_RERANK_TOP_N} and {MAX_RERANK_TOP_N}"
            )));
        }
        if rerank_candidate_top_k < rerank_top_n {
            return Err(CloudModelConfigError(format!(
                "RERANK_CANDIDATE_TOP_K ({rerank_candidate_top_k}) must be at least RERANK_TOP_N ({rerank_top_n})"
            )));
        }
        let rerank_batch_size =
            positive_usize_setting(&mut get, "RERANK_BATCH_SIZE", DEFAULT_RERANK_BATCH_SIZE)?;
        if rerank_batch_size > MAX_RERANK_BATCH_SIZE {
            return Err(CloudModelConfigError(format!(
                "RERANK_BATCH_SIZE must not exceed {MAX_RERANK_BATCH_SIZE}"
            )));
        }

        Ok(Self {
            gateway_token,
            chat_endpoint,
            embed_endpoint,
            rerank_endpoint,
            chat_model,
            embed_model,
            rerank_model,
            request_timeout: Duration::from_secs(timeout_secs),
            rerank_candidate_top_k,
            rerank_top_n,
            rerank_batch_size,
        })
    }

    pub fn chat_endpoint(&self) -> &Url {
        &self.chat_endpoint
    }

    pub fn embed_endpoint(&self) -> &Url {
        &self.embed_endpoint
    }

    pub fn rerank_endpoint(&self) -> &Url {
        &self.rerank_endpoint
    }

    pub fn chat_model(&self) -> &str {
        &self.chat_model
    }

    pub fn embed_model(&self) -> &str {
        &self.embed_model
    }

    pub fn rerank_model(&self) -> &str {
        &self.rerank_model
    }

    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    pub fn rerank_candidate_top_k(&self) -> usize {
        self.rerank_candidate_top_k
    }

    pub fn rerank_top_n(&self) -> usize {
        self.rerank_top_n
    }

    pub fn rerank_batch_size(&self) -> usize {
        self.rerank_batch_size
    }
}

fn required_value(
    get: &mut impl FnMut(&str) -> Option<String>,
    name: &'static str,
) -> Result<String, CloudModelConfigError> {
    optional_value(get, name).ok_or_else(|| {
        CloudModelConfigError(format!(
            "{name} is required and must be non-empty before the server can start"
        ))
    })
}

fn optional_value(
    get: &mut impl FnMut(&str) -> Option<String>,
    name: &'static str,
) -> Option<String> {
    get(name)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Resolves one capability URL: an explicit full gateway route wins, otherwise
/// the default NVIDIA custom-provider path is appended to the gateway base.
/// Missing both is a startup error so nothing can silently bypass the gateway.
fn capability_endpoint(
    get: &mut impl FnMut(&str) -> Option<String>,
    name: &'static str,
    gateway_base: &Option<String>,
    default_gateway_path: &str,
) -> Result<Url, CloudModelConfigError> {
    if let Some(url) = optional_value(get, name) {
        return parse_endpoint(name, &url);
    }
    match gateway_base {
        Some(base) => parse_endpoint(
            "CF_AI_GATEWAY_BASE_URL",
            &format!("{base}/{default_gateway_path}"),
        ),
        None => Err(CloudModelConfigError(format!(
            "{name} is empty and CF_AI_GATEWAY_BASE_URL is not set; configure the Cloudflare AI \
             Gateway base URL ({GATEWAY_BASE_URL_HINT}) or a full gateway route in {name}"
        ))),
    }
}

fn positive_usize_setting(
    get: &mut impl FnMut(&str) -> Option<String>,
    name: &'static str,
    default: usize,
) -> Result<usize, CloudModelConfigError> {
    match optional_value(get, name) {
        Some(raw) => raw
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| CloudModelConfigError(format!("{name} must be a positive integer"))),
        None => Ok(default),
    }
}

fn validate_locked_model(
    setting: &'static str,
    model: &str,
    expected_model: &'static str,
) -> Result<(), CloudModelConfigError> {
    if model == expected_model {
        Ok(())
    } else {
        Err(CloudModelConfigError(format!(
            "{setting} must remain locked to {expected_model}"
        )))
    }
}

fn validate_production_gateway_endpoints(
    chat: &Url,
    embed: &Url,
    rerank: &Url,
) -> Result<(), CloudModelConfigError> {
    let chat_identity = production_gateway_identity("CLOUD_CHAT_URL", chat)?;
    let embed_identity = production_gateway_identity("CLOUD_EMBED_URL", embed)?;
    let rerank_identity = production_gateway_identity("CLOUD_RERANK_URL", rerank)?;
    if chat_identity != embed_identity || chat_identity != rerank_identity {
        return Err(CloudModelConfigError(
            "all cloud capability URLs must use the same Cloudflare AI Gateway account and gateway path"
                .to_owned(),
        ));
    }
    Ok(())
}

fn production_gateway_identity(
    name: &'static str,
    endpoint: &Url,
) -> Result<(String, String), CloudModelConfigError> {
    if endpoint.scheme() != "https"
        || endpoint.host_str() != Some("gateway.ai.cloudflare.com")
        || endpoint.port_or_known_default() != Some(443)
    {
        return Err(CloudModelConfigError(format!(
            "{name} must use HTTPS on gateway.ai.cloudflare.com"
        )));
    }
    let segments = endpoint
        .path_segments()
        .map(|segments| segments.collect::<Vec<_>>())
        .unwrap_or_default();
    if segments.len() < 4
        || segments.first() != Some(&"v1")
        || segments[1].is_empty()
        || segments[2].is_empty()
    {
        return Err(CloudModelConfigError(format!(
            "{name} must include /v1/<account-id>/<gateway-id>/..."
        )));
    }
    Ok((segments[1].to_owned(), segments[2].to_owned()))
}

fn is_gemma_4_model_id(model: &str) -> bool {
    let normalized = model.to_ascii_lowercase();
    let bytes = normalized.as_bytes();
    normalized.match_indices("gemma").any(|(index, _)| {
        let has_left_boundary = index == 0 || !bytes[index - 1].is_ascii_alphanumeric();
        if !has_left_boundary {
            return false;
        }

        let mut cursor = index + "gemma".len();
        while bytes
            .get(cursor)
            .is_some_and(|byte| matches!(byte, b'-' | b'_' | b'.'))
        {
            cursor += 1;
        }
        if bytes.get(cursor) != Some(&b'4') {
            return false;
        }
        bytes
            .get(cursor + 1)
            .is_none_or(|byte| !byte.is_ascii_alphanumeric())
    })
}

fn parse_endpoint(name: &'static str, value: &str) -> Result<Url, CloudModelConfigError> {
    let endpoint = Url::parse(value)
        .map_err(|_| CloudModelConfigError(format!("{name} must be a valid HTTP(S) URL")))?;
    if !matches!(endpoint.scheme(), "http" | "https")
        || endpoint.host_str().is_none()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
    {
        return Err(CloudModelConfigError(format!(
            "{name} must be an HTTP(S) URL without credentials, query, or fragment"
        )));
    }
    Ok(endpoint)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelOperation {
    Chat,
    Embed,
    Rerank,
}

impl fmt::Display for ModelOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Chat => "chat",
            Self::Embed => "embedding",
            Self::Rerank => "reranking",
        })
    }
}

#[derive(Debug)]
pub enum CloudModelError {
    Http {
        operation: ModelOperation,
        status: StatusCode,
        response_snippet: String,
        retryable: bool,
    },
    Timeout {
        operation: ModelOperation,
        timeout: Duration,
    },
    Transport {
        operation: ModelOperation,
        message: String,
        retryable: bool,
    },
    InvalidRequest {
        operation: ModelOperation,
        message: String,
    },
    InvalidResponse {
        operation: ModelOperation,
        message: String,
    },
}

impl CloudModelError {
    pub fn status(&self) -> Option<StatusCode> {
        match self {
            Self::Http { status, .. } => Some(*status),
            _ => None,
        }
    }

    pub fn response_snippet(&self) -> Option<&str> {
        match self {
            Self::Http {
                response_snippet, ..
            } => Some(response_snippet),
            _ => None,
        }
    }

    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout { .. })
    }

    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Http { retryable, .. } | Self::Transport { retryable, .. } => *retryable,
            Self::Timeout { .. } => true,
            Self::InvalidRequest { .. } | Self::InvalidResponse { .. } => false,
        }
    }

    pub fn safe_kind(&self) -> &'static str {
        match self {
            Self::Http { .. } => "http",
            Self::Timeout { .. } => "timeout",
            Self::Transport { .. } => "transport",
            Self::InvalidRequest { .. } => "invalid_request",
            Self::InvalidResponse { .. } => "invalid_response",
        }
    }

    /// Stable text that can be exposed to chat clients without provider or request content.
    pub fn chat_client_message(&self) -> String {
        match self {
            Self::Http { status, .. } => format!(
                "雲端回答服務拒絕請求（錯誤代碼：chat_model_http；HTTP {}）",
                status.as_u16()
            ),
            Self::Timeout { .. } => "雲端回答服務逾時（錯誤代碼：chat_model_timeout）".to_string(),
            Self::Transport { .. } => {
                "雲端回答服務連線失敗（錯誤代碼：chat_model_transport）".to_string()
            }
            Self::InvalidRequest { .. } => {
                "雲端回答請求設定無效（錯誤代碼：chat_model_invalid_request）".to_string()
            }
            Self::InvalidResponse { .. } => {
                "雲端回答服務回應格式無效（錯誤代碼：chat_model_invalid_response）".to_string()
            }
        }
    }
}

impl fmt::Display for CloudModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http {
                operation,
                status,
                response_snippet,
                retryable,
            } => write!(
                f,
                "{operation} returned HTTP {status} (retryable={retryable}): {response_snippet}"
            ),
            Self::Timeout { operation, timeout } => write!(
                f,
                "{operation} request timed out after {}s (retryable=true)",
                timeout.as_secs()
            ),
            Self::Transport {
                operation,
                message,
                retryable,
            } => write!(
                f,
                "{operation} request failed (retryable={retryable}): {message}"
            ),
            Self::InvalidRequest { operation, message } => {
                write!(f, "invalid {operation} request: {message}")
            }
            Self::InvalidResponse { operation, message } => {
                write!(f, "invalid {operation} response: {message}")
            }
        }
    }
}

impl std::error::Error for CloudModelError {}

pub type CloudModelResult<T> = Result<T, CloudModelError>;

#[derive(Clone, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ApiChatMessage>,
    pub stream: bool,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

impl fmt::Debug for ChatCompletionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChatCompletionRequest")
            .field("model", &self.model)
            .field("message_count", &self.messages.len())
            .field("stream", &self.stream)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiChatMessage {
    pub role: String,
    pub content: ApiChatContent,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ApiChatContent {
    Text(String),
    Parts(Vec<ApiChatContentPart>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ApiChatContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ApiImageUrl },
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiImageUrl {
    pub url: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponseMessage {
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatStreamChunk {
    pub choices: Vec<ChatStreamChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChatStreamChoice {
    pub delta: ChatStreamDelta,
}

#[derive(Debug, Deserialize)]
pub struct ChatStreamDelta {
    pub content: Option<String>,
}

#[derive(Clone, Serialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    pub encoding_format: String,
    pub embedding_type: String,
    pub input_type: String,
    pub modality: Vec<String>,
    pub truncate: String,
}

impl fmt::Debug for EmbeddingRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EmbeddingRequest")
            .field("model", &self.model)
            .field(
                "input",
                &format_args!("[REDACTED; {} item(s)]", self.input.len()),
            )
            .field("encoding_format", &self.encoding_format)
            .field("embedding_type", &self.embedding_type)
            .field("input_type", &self.input_type)
            .field("modality", &self.modality)
            .field("truncate", &self.truncate)
            .finish()
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Serialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: RerankQuery,
    pub passages: Vec<RerankPassage>,
    pub truncate: String,
}

impl fmt::Debug for RerankRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RerankRequest")
            .field("model", &self.model)
            .field("query", &"[REDACTED]")
            .field("passage_count", &self.passages.len())
            .field("truncate", &self.truncate)
            .finish()
    }
}

#[derive(Clone, Serialize)]
pub struct RerankQuery {
    pub text: String,
}

#[derive(Clone, Serialize)]
pub struct RerankPassage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
}

impl RerankPassage {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            image: None,
        }
    }

    pub fn image(image: &[u8]) -> CloudModelResult<Self> {
        Ok(Self {
            text: None,
            image: Some(image_data_url_for(ModelOperation::Rerank, image)?),
        })
    }

    pub fn image_text(text: impl Into<String>, image: &[u8]) -> CloudModelResult<Self> {
        Ok(Self {
            text: Some(text.into()),
            image: Some(image_data_url_for(ModelOperation::Rerank, image)?),
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct RerankResponse {
    pub rankings: Vec<RerankResult>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RerankResult {
    pub index: usize,
    pub logit: f64,
}

#[derive(Debug, Clone)]
struct EmbeddingRetryConfig {
    max_attempts: usize,
    base_delay: Duration,
    max_delay: Duration,
    jitter: bool,
}

impl Default for EmbeddingRetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_EMBED_MAX_ATTEMPTS,
            base_delay: DEFAULT_EMBED_RETRY_BASE_DELAY,
            max_delay: DEFAULT_EMBED_RETRY_MAX_DELAY,
            jitter: true,
        }
    }
}

#[derive(Debug, Clone)]
struct RerankRetryConfig {
    max_attempts: usize,
    base_delay: Duration,
    max_delay: Duration,
    jitter: bool,
}

impl Default for RerankRetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_RERANK_MAX_ATTEMPTS,
            base_delay: DEFAULT_RERANK_RETRY_BASE_DELAY,
            max_delay: DEFAULT_RERANK_RETRY_MAX_DELAY,
            jitter: true,
        }
    }
}

/// Content-free counters for actual HTTP attempts made by the cloud client.
///
/// A snapshot deliberately contains no endpoint, model input, document data, or
/// authentication material. Cloned clients share the same underlying atomics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CloudRequestTelemetry {
    pub chat_attempts: u64,
    pub embedding_attempts: u64,
    pub rerank_attempts: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloudRetryPolicy {
    pub max_attempts: u32,
    pub base_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub jitter: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloudRetryPolicies {
    pub embedding: CloudRetryPolicy,
    pub rerank: CloudRetryPolicy,
    pub generation: CloudRetryPolicy,
}

impl CloudRequestTelemetry {
    pub fn delta_since(self, earlier: Self) -> Self {
        Self {
            chat_attempts: self.chat_attempts.saturating_sub(earlier.chat_attempts),
            embedding_attempts: self
                .embedding_attempts
                .saturating_sub(earlier.embedding_attempts),
            rerank_attempts: self.rerank_attempts.saturating_sub(earlier.rerank_attempts),
        }
    }
}

#[derive(Debug, Default)]
struct CloudAttemptCounters {
    chat: AtomicU64,
    embedding: AtomicU64,
    rerank: AtomicU64,
}

#[derive(Clone)]
pub struct CloudModelClient {
    http: reqwest::Client,
    gateway_token: String,
    chat_endpoint: Url,
    embed_endpoint: Url,
    rerank_endpoint: Url,
    chat_model: String,
    embed_model: String,
    rerank_model: String,
    request_timeout: Duration,
    rerank_candidate_top_k: usize,
    rerank_top_n: usize,
    rerank_batch_size: usize,
    embedding_retry: EmbeddingRetryConfig,
    rerank_retry: RerankRetryConfig,
    attempts: Arc<CloudAttemptCounters>,
}

impl CloudModelClient {
    pub fn new(config: CloudModelConfig) -> Result<Self, CloudModelConfigError> {
        let http = reqwest::Client::builder()
            .connect_timeout(config.request_timeout.min(Duration::from_secs(10)))
            .timeout(config.request_timeout)
            .build()
            .map_err(|_| {
                CloudModelConfigError("failed to initialize cloud HTTP client".to_string())
            })?;

        Ok(Self {
            http,
            gateway_token: config.gateway_token,
            chat_endpoint: config.chat_endpoint,
            embed_endpoint: config.embed_endpoint,
            rerank_endpoint: config.rerank_endpoint,
            chat_model: config.chat_model,
            embed_model: config.embed_model,
            rerank_model: config.rerank_model,
            request_timeout: config.request_timeout,
            rerank_candidate_top_k: config.rerank_candidate_top_k,
            rerank_top_n: config.rerank_top_n,
            rerank_batch_size: config.rerank_batch_size,
            embedding_retry: EmbeddingRetryConfig::default(),
            rerank_retry: RerankRetryConfig::default(),
            attempts: Arc::new(CloudAttemptCounters::default()),
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        let endpoint = |path: &str| {
            Url::parse(&format!("http://127.0.0.1/{path}"))
                .expect("static test endpoint must be valid")
        };
        let config = CloudModelConfig {
            gateway_token: "test-gateway-token".to_string(),
            chat_endpoint: endpoint("chat/completions"),
            embed_endpoint: endpoint("embeddings"),
            rerank_endpoint: endpoint("ranking"),
            chat_model: "test/gemma-4".to_string(),
            embed_model: DEFAULT_EMBED_MODEL_ID.to_string(),
            rerank_model: DEFAULT_RERANK_MODEL_ID.to_string(),
            request_timeout: Duration::from_secs(1),
            rerank_candidate_top_k: DEFAULT_RERANK_CANDIDATE_TOP_K,
            rerank_top_n: DEFAULT_RERANK_TOP_N,
            rerank_batch_size: DEFAULT_RERANK_BATCH_SIZE,
        };
        let mut client = Self::new(config).expect("static cloud model test config must be valid");
        client.embedding_retry = EmbeddingRetryConfig {
            max_attempts: DEFAULT_EMBED_MAX_ATTEMPTS,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            jitter: false,
        };
        client.rerank_retry = RerankRetryConfig {
            max_attempts: DEFAULT_RERANK_MAX_ATTEMPTS,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            jitter: false,
        };
        client
    }

    #[cfg(test)]
    pub(crate) fn for_test_embed_endpoint(endpoint: &str, timeout: Duration) -> Self {
        Self::for_test_endpoints("http://127.0.0.1/chat/completions", endpoint, timeout)
    }

    #[cfg(test)]
    pub(crate) fn for_test_endpoints(
        chat_endpoint: &str,
        embed_endpoint: &str,
        timeout: Duration,
    ) -> Self {
        let mut client = Self::for_test();
        client.chat_endpoint = Url::parse(chat_endpoint).expect("test chat endpoint must be valid");
        client.embed_endpoint =
            Url::parse(embed_endpoint).expect("test embed endpoint must be valid");
        client.request_timeout = timeout;
        client.http = reqwest::Client::builder()
            .connect_timeout(timeout.min(Duration::from_millis(100)))
            .timeout(timeout)
            .build()
            .expect("test HTTP client must be valid");
        client
    }

    #[cfg(test)]
    pub(crate) fn with_test_rerank_endpoint(mut self, endpoint: &str) -> Self {
        self.rerank_endpoint = Url::parse(endpoint).expect("test rerank endpoint must be valid");
        self
    }

    #[cfg(test)]
    fn embedding_attempt_count(&self) -> usize {
        self.telemetry_snapshot().embedding_attempts as usize
    }

    #[cfg(test)]
    fn rerank_attempt_count(&self) -> usize {
        self.telemetry_snapshot().rerank_attempts as usize
    }

    pub fn telemetry_snapshot(&self) -> CloudRequestTelemetry {
        CloudRequestTelemetry {
            chat_attempts: self.attempts.chat.load(Ordering::Relaxed),
            embedding_attempts: self.attempts.embedding.load(Ordering::Relaxed),
            rerank_attempts: self.attempts.rerank.load(Ordering::Relaxed),
        }
    }

    pub fn with_deterministic_evaluation_retries(mut self) -> Self {
        self.embedding_retry.jitter = false;
        self.rerank_retry.jitter = false;
        self
    }

    pub fn evaluation_retry_policies(&self) -> CloudRetryPolicies {
        let policy = |max_attempts: usize, base_delay: Duration, max_delay: Duration, jitter| {
            CloudRetryPolicy {
                max_attempts: u32::try_from(max_attempts).unwrap_or(u32::MAX),
                base_backoff_ms: u64::try_from(base_delay.as_millis()).unwrap_or(u64::MAX),
                max_backoff_ms: u64::try_from(max_delay.as_millis()).unwrap_or(u64::MAX),
                jitter,
            }
        };
        CloudRetryPolicies {
            embedding: policy(
                self.embedding_retry.max_attempts,
                self.embedding_retry.base_delay,
                self.embedding_retry.max_delay,
                self.embedding_retry.jitter,
            ),
            rerank: policy(
                self.rerank_retry.max_attempts,
                self.rerank_retry.base_delay,
                self.rerank_retry.max_delay,
                self.rerank_retry.jitter,
            ),
            generation: CloudRetryPolicy {
                max_attempts: 1,
                base_backoff_ms: 0,
                max_backoff_ms: 0,
                jitter: false,
            },
        }
    }

    pub fn rerank_model(&self) -> &str {
        &self.rerank_model
    }

    pub fn chat_model(&self) -> &str {
        &self.chat_model
    }

    pub fn embed_model(&self) -> &str {
        &self.embed_model
    }

    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    pub fn rerank_candidate_top_k(&self) -> usize {
        self.rerank_candidate_top_k
    }

    pub fn rerank_top_n(&self) -> usize {
        self.rerank_top_n
    }

    pub fn rerank_batch_size(&self) -> usize {
        self.rerank_batch_size
    }

    /// Fits a prepared RAG context to the complete serialized Gemma request.
    ///
    /// The context builder bounds its own multipart payload first. This second pass includes the
    /// system prompt, model id, generation settings, and conversation history; if necessary it
    /// removes only lowest-ranked sources while keeping model parts and structured metadata in
    /// lockstep.
    pub fn fit_rag_context_to_request(
        &self,
        system_prompt: &str,
        context: crate::rag_context::RagContext,
        messages: &[ChatMessage],
    ) -> CloudModelResult<crate::rag_context::RagContext> {
        self.fit_rag_context_to_request_with_limit(
            system_prompt,
            context,
            messages,
            MAX_CHAT_REQUEST_BYTES,
        )
    }

    /// Validates a catalog context against the complete request budget without dropping any
    /// document source. Catalog completeness is more important than silently fitting a prefix.
    pub fn fit_catalog_context_to_request(
        &self,
        system_prompt: &str,
        context: crate::rag_context::RagContext,
        messages: &[ChatMessage],
    ) -> CloudModelResult<crate::rag_context::RagContext> {
        let request = self.chat_request_with_context(
            system_prompt.to_string(),
            context.content.clone(),
            messages.to_vec(),
            true,
            4096,
            None,
        );
        let payload = serialize_chat_request(&request)?;
        if payload.len() > MAX_CHAT_REQUEST_BYTES {
            return Err(chat_request_too_large(
                payload.len(),
                MAX_CHAT_REQUEST_BYTES,
            ));
        }
        Ok(context)
    }

    fn fit_rag_context_to_request_with_limit(
        &self,
        system_prompt: &str,
        mut context: crate::rag_context::RagContext,
        messages: &[ChatMessage],
        maximum_bytes: usize,
    ) -> CloudModelResult<crate::rag_context::RagContext> {
        loop {
            let request = self.chat_request_with_context(
                system_prompt.to_string(),
                context.content.clone(),
                messages.to_vec(),
                true,
                4096,
                None,
            );
            let payload = serialize_chat_request(&request)?;
            if payload.len() <= maximum_bytes {
                return Ok(context);
            }
            if !context.drop_lowest_source() {
                return Err(chat_request_too_large(payload.len(), maximum_bytes));
            }
        }
    }

    pub async fn stream_chat(&self, messages: Vec<ChatMessage>, tx: Sender<SseEvent>) {
        self.stream_chat_with_context(
            SYSTEM_PROMPT.to_string(),
            ApiChatContent::Parts(Vec::new()),
            messages,
            tx,
        )
        .await;
    }

    pub async fn stream_chat_with_context(
        &self,
        system_prompt: String,
        context: ApiChatContent,
        messages: Vec<ChatMessage>,
        tx: Sender<SseEvent>,
    ) {
        if let Err(error) = self
            .try_stream_chat_with_context(system_prompt, context, messages, tx.clone())
            .await
        {
            tracing::error!(
                stage = "chat_stream",
                error_kind = error.safe_kind(),
                http_status = ?error.status().map(|status| status.as_u16()),
                "Gemma 4 chat stream failed"
            );
            let _ = tx
                .send(SseEvent::Error {
                    message: error.chat_client_message(),
                })
                .await;
        }
    }

    pub async fn try_stream_chat_with_context(
        &self,
        system_prompt: String,
        context: ApiChatContent,
        messages: Vec<ChatMessage>,
        tx: Sender<SseEvent>,
    ) -> CloudModelResult<()> {
        let request =
            self.chat_request_with_context(system_prompt, context, messages, true, 4096, None);
        let response = self.send_chat_request(&request).await?;

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut filter = ThinkFilter::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| self.request_error(ModelOperation::Chat, error))?;
            buffer.extend_from_slice(&chunk);

            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=newline).collect::<Vec<_>>();
                match forward_chat_sse_line(&line, &tx, &mut filter).await? {
                    ChatSseProgress::Continue => {}
                    ChatSseProgress::Complete | ChatSseProgress::ReceiverClosed => return Ok(()),
                }
            }
        }

        if !buffer.is_empty() {
            match forward_chat_sse_line(&buffer, &tx, &mut filter).await? {
                ChatSseProgress::Complete | ChatSseProgress::ReceiverClosed => return Ok(()),
                ChatSseProgress::Continue => {}
            }
        }
        Err(CloudModelError::InvalidResponse {
            operation: ModelOperation::Chat,
            message: "stream ended before the [DONE] marker".to_string(),
        })
    }

    pub async fn chat_once(
        &self,
        system_prompt: &str,
        user_message: &str,
    ) -> CloudModelResult<String> {
        let request = self.chat_request(
            system_prompt.to_string(),
            vec![ChatMessage {
                role: "user".to_string(),
                content: user_message.to_string(),
            }],
            false,
            2048,
            Some(0.2),
        );
        let response = self.send_chat_request(&request).await?;
        self.chat_content(response).await
    }

    /// Runs one bounded non-streaming Gemma 4 request over prepared multimodal content.
    ///
    /// This is used for best-effort document metadata generation. The caller remains
    /// responsible for treating the content as untrusted and validating the returned schema.
    pub(crate) async fn chat_once_with_context(
        &self,
        system_prompt: &str,
        context: ApiChatContent,
    ) -> CloudModelResult<String> {
        let request = self.chat_request_with_context(
            system_prompt.to_string(),
            context,
            Vec::new(),
            false,
            512,
            Some(0.0),
        );
        let response = self.send_chat_request(&request).await?;
        self.chat_content(response).await
    }

    /// Runs one deterministic non-streaming Gemma 4 RAG answer request.
    ///
    /// Evaluation modes A and B both use this exact method so the only changed
    /// input is their retrieved context. The context precedes the user query in
    /// the same order as the production streaming request.
    pub async fn answer_rag_once(
        &self,
        system_prompt: &str,
        context: ApiChatContent,
        user_query: &str,
    ) -> CloudModelResult<String> {
        if user_query.trim().is_empty() {
            return Err(CloudModelError::InvalidRequest {
                operation: ModelOperation::Chat,
                message: "evaluation query must not be empty".to_owned(),
            });
        }
        let request = self.chat_request_with_context(
            system_prompt.to_owned(),
            context,
            vec![ChatMessage {
                role: "user".to_owned(),
                content: user_query.to_owned(),
            }],
            false,
            4096,
            Some(0.0),
        );
        let response = self.send_chat_request(&request).await?;
        self.chat_content(response).await
    }

    pub async fn embed_query(&self, text: &str) -> CloudModelResult<Vec<f32>> {
        self.embed_text(text, "query").await
    }

    pub async fn embed_passage(&self, text: &str) -> CloudModelResult<Vec<f32>> {
        self.embed_text(text, "passage").await
    }

    pub async fn embed_image_passage(&self, image: &[u8]) -> CloudModelResult<Vec<f32>> {
        let data_url = image_data_url_for(ModelOperation::Embed, image)?;
        self.embed_input(data_url, "passage", "image").await
    }

    pub async fn embed_image_text_passage(
        &self,
        text: &str,
        image: &[u8],
    ) -> CloudModelResult<Vec<f32>> {
        if text.trim().is_empty() {
            return Err(CloudModelError::InvalidRequest {
                operation: ModelOperation::Embed,
                message: "text_image input requires non-empty text".to_string(),
            });
        }
        let data_url = image_data_url_for(ModelOperation::Embed, image)?;
        let input = format!("{}\n<img src=\"{data_url}\"/>", text.trim());
        self.embed_input(input, "passage", "text_image").await
    }

    pub async fn embed_text(&self, text: &str, input_type: &str) -> CloudModelResult<Vec<f32>> {
        if text.trim().is_empty() {
            return Err(CloudModelError::InvalidRequest {
                operation: ModelOperation::Embed,
                message: "input must not be empty".to_string(),
            });
        }
        if !matches!(input_type, "query" | "passage") {
            return Err(CloudModelError::InvalidRequest {
                operation: ModelOperation::Embed,
                message: "input_type must be query or passage".to_string(),
            });
        }

        self.embed_input(text.to_string(), input_type, "text").await
    }

    async fn embed_input(
        &self,
        input: String,
        input_type: &str,
        modality: &str,
    ) -> CloudModelResult<Vec<f32>> {
        let request = EmbeddingRequest {
            model: self.embed_model.clone(),
            input: vec![input],
            encoding_format: "float".to_string(),
            embedding_type: "float".to_string(),
            input_type: input_type.to_string(),
            // `input` is a one-item list, so modality must also contain one
            // item. `text_image` is one multimodal input, not two inputs.
            modality: vec![modality.to_string()],
            truncate: "NONE".to_string(),
        };
        let payload_size = serde_json::to_vec(&request)
            .map_err(|_| CloudModelError::InvalidRequest {
                operation: ModelOperation::Embed,
                message: "request payload could not be serialized".to_string(),
            })?
            .len();
        if payload_size > MAX_EMBED_REQUEST_BYTES {
            return Err(CloudModelError::InvalidRequest {
                operation: ModelOperation::Embed,
                message: format!(
                    "request payload is {payload_size} bytes; maximum is {MAX_EMBED_REQUEST_BYTES} bytes"
                ),
            });
        }

        let response = self.send_embedding_request(&request).await?;
        let embedding = response
            .data
            .into_iter()
            .find(|item| item.index == 0)
            .map(|item| item.embedding)
            .ok_or_else(|| CloudModelError::InvalidResponse {
                operation: ModelOperation::Embed,
                message: "missing data[0].embedding".to_string(),
            })?;
        validate_embedding(embedding)
    }

    pub async fn rerank(
        &self,
        query: &str,
        passages: &[RerankPassage],
    ) -> CloudModelResult<RerankResponse> {
        if query.trim().is_empty() || passages.is_empty() {
            return Err(CloudModelError::InvalidRequest {
                operation: ModelOperation::Rerank,
                message: "query and at least one passage are required".to_string(),
            });
        }
        for (index, passage) in passages.iter().enumerate() {
            if passage.text.is_none() && passage.image.is_none() {
                return Err(CloudModelError::InvalidRequest {
                    operation: ModelOperation::Rerank,
                    message: format!("passage {index} requires text, image, or both"),
                });
            }
            if passage
                .text
                .as_deref()
                .is_some_and(|text| text.trim().is_empty())
            {
                return Err(CloudModelError::InvalidRequest {
                    operation: ModelOperation::Rerank,
                    message: format!("passage {index} text must not be empty"),
                });
            }
            if passage.image.as_deref().is_some_and(|image| {
                !image.starts_with("data:image/png;base64,")
                    && !image.starts_with("data:image/jpeg;base64,")
            }) {
                return Err(CloudModelError::InvalidRequest {
                    operation: ModelOperation::Rerank,
                    message: format!(
                        "passage {index} image must be an in-memory PNG or JPEG data URL"
                    ),
                });
            }
        }

        let request = RerankRequest {
            model: self.rerank_model.clone(),
            query: RerankQuery {
                text: query.to_string(),
            },
            passages: passages.to_vec(),
            truncate: "NONE".to_string(),
        };
        self.send_rerank_request(&request, passages.len()).await
    }

    fn chat_request(
        &self,
        system_prompt: String,
        messages: Vec<ChatMessage>,
        stream: bool,
        max_tokens: u32,
        temperature: Option<f32>,
    ) -> ChatCompletionRequest {
        self.chat_request_with_optional_context(
            system_prompt,
            None,
            messages,
            stream,
            max_tokens,
            temperature,
        )
    }

    fn chat_request_with_context(
        &self,
        system_prompt: String,
        context: ApiChatContent,
        messages: Vec<ChatMessage>,
        stream: bool,
        max_tokens: u32,
        temperature: Option<f32>,
    ) -> ChatCompletionRequest {
        let context = match context {
            ApiChatContent::Parts(parts) if parts.is_empty() => None,
            context => Some(context),
        };
        self.chat_request_with_optional_context(
            system_prompt,
            context,
            messages,
            stream,
            max_tokens,
            temperature,
        )
    }

    fn chat_request_with_optional_context(
        &self,
        system_prompt: String,
        context: Option<ApiChatContent>,
        messages: Vec<ChatMessage>,
        stream: bool,
        max_tokens: u32,
        temperature: Option<f32>,
    ) -> ChatCompletionRequest {
        let mut api_messages = vec![ApiChatMessage {
            role: "system".to_string(),
            content: ApiChatContent::Text(system_prompt),
        }];
        if let Some(context) = context {
            api_messages.push(ApiChatMessage {
                role: "user".to_string(),
                content: context,
            });
        }
        api_messages.extend(messages.into_iter().map(|message| ApiChatMessage {
            role: if message.role == "assistant" {
                "assistant".to_string()
            } else {
                "user".to_string()
            },
            content: ApiChatContent::Text(message.content),
        }));

        ChatCompletionRequest {
            model: self.chat_model.clone(),
            messages: api_messages,
            stream,
            max_tokens,
            temperature,
        }
    }

    async fn send_chat_request(
        &self,
        request: &ChatCompletionRequest,
    ) -> CloudModelResult<Response> {
        let payload = serialize_chat_request_with_limit(request, MAX_CHAT_REQUEST_BYTES)?;
        self.attempts.chat.fetch_add(1, Ordering::Relaxed);
        let response = self
            .authorized(self.http.post(self.chat_endpoint.clone()))
            .header(CONTENT_TYPE, "application/json")
            .body(payload)
            .send()
            .await
            .map_err(|error| self.request_error(ModelOperation::Chat, error))?;

        if response.status().is_success() {
            Ok(response)
        } else {
            Err(self.http_error(ModelOperation::Chat, response).await)
        }
    }

    async fn chat_content(&self, response: Response) -> CloudModelResult<String> {
        let response: ChatCompletionResponse =
            self.decode_json(ModelOperation::Chat, response).await?;
        let content = response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .ok_or_else(|| CloudModelError::InvalidResponse {
                operation: ModelOperation::Chat,
                message: "missing choices[0].message.content".to_string(),
            })?;
        let think_re = regex::Regex::new(r"(?s)<think>.*?</think>").expect("valid think regex");
        Ok(think_re.replace_all(&content, "").trim().to_string())
    }

    async fn send_embedding_request(
        &self,
        request: &EmbeddingRequest,
    ) -> CloudModelResult<EmbeddingResponse> {
        for attempt in 1..=self.embedding_retry.max_attempts {
            let result = async {
                let response = self
                    .send_json(ModelOperation::Embed, &self.embed_endpoint, request)
                    .await?;
                self.decode_json(ModelOperation::Embed, response).await
            }
            .await;
            match result {
                Ok(response) => return Ok(response),
                Err(error)
                    if error.is_retryable() && attempt < self.embedding_retry.max_attempts =>
                {
                    tokio::time::sleep(self.embedding_retry_delay(attempt)).await;
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!("embedding retry configuration always has at least one attempt")
    }

    async fn send_rerank_request(
        &self,
        request: &RerankRequest,
        expected_count: usize,
    ) -> CloudModelResult<RerankResponse> {
        for attempt in 1..=self.rerank_retry.max_attempts {
            let result = async {
                let response = self
                    .send_json(ModelOperation::Rerank, &self.rerank_endpoint, request)
                    .await?;
                let response = self.decode_json(ModelOperation::Rerank, response).await?;
                validate_rerank_response(response, expected_count)
            }
            .await;
            match result {
                Ok(response) => return Ok(response),
                Err(error) if error.is_retryable() && attempt < self.rerank_retry.max_attempts => {
                    let error_kind = error.safe_kind();
                    tracing::warn!(
                        stage = "rerank_request",
                        attempt,
                        max_attempts = self.rerank_retry.max_attempts,
                        model_id = self.rerank_model(),
                        error_kind,
                        http_status = ?error.status().map(|status| status.as_u16()),
                        "rerank request failed transiently; retrying"
                    );
                    tokio::time::sleep(bounded_retry_delay(
                        self.rerank_retry.base_delay,
                        self.rerank_retry.max_delay,
                        self.rerank_retry.jitter,
                        attempt,
                    ))
                    .await;
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!("rerank retry configuration always has at least one attempt")
    }

    fn embedding_retry_delay(&self, failed_attempt: usize) -> Duration {
        bounded_retry_delay(
            self.embedding_retry.base_delay,
            self.embedding_retry.max_delay,
            self.embedding_retry.jitter,
            failed_attempt,
        )
    }

    /// Authenticates the request to the Cloudflare AI Gateway. Upstream
    /// provider keys live in the gateway's BYOK store, so no provider
    /// `Authorization` header is ever sent from the backend.
    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        request.header(
            GATEWAY_AUTHORIZATION_HEADER,
            format!("Bearer {}", self.gateway_token),
        )
    }

    async fn send_json<T: Serialize + ?Sized>(
        &self,
        operation: ModelOperation,
        endpoint: &Url,
        request: &T,
    ) -> CloudModelResult<Response> {
        match operation {
            ModelOperation::Embed => self.attempts.embedding.fetch_add(1, Ordering::Relaxed),
            ModelOperation::Rerank => self.attempts.rerank.fetch_add(1, Ordering::Relaxed),
            ModelOperation::Chat => self.attempts.chat.fetch_add(1, Ordering::Relaxed),
        };
        let response = self
            .authorized(self.http.post(endpoint.clone()))
            .json(request)
            .send()
            .await
            .map_err(|error| self.request_error(operation, error))?;

        if response.status().is_success() {
            Ok(response)
        } else {
            Err(self.http_error(operation, response).await)
        }
    }

    async fn decode_json<T: for<'de> Deserialize<'de>>(
        &self,
        operation: ModelOperation,
        response: Response,
    ) -> CloudModelResult<T> {
        let bytes = response
            .bytes()
            .await
            .map_err(|error| self.request_error(operation, error))?;
        serde_json::from_slice(&bytes).map_err(|error| CloudModelError::InvalidResponse {
            operation,
            message: format!("response body is not valid JSON: {error}"),
        })
    }

    async fn http_error(&self, operation: ModelOperation, response: Response) -> CloudModelError {
        let status = response.status();
        let mut stream = response.bytes_stream();
        let mut body = Vec::with_capacity(4_096);
        while body.len() < 4_096
            && let Some(chunk) = stream.next().await
        {
            let Ok(chunk) = chunk else { break };
            let remaining = 4_096 - body.len();
            body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
        }
        let body = String::from_utf8_lossy(&body);
        CloudModelError::Http {
            operation,
            status,
            response_snippet: self.sanitize(&body).chars().take(500).collect(),
            retryable: retryable_status(status),
        }
    }

    fn request_error(&self, operation: ModelOperation, error: reqwest::Error) -> CloudModelError {
        if error.is_timeout() {
            CloudModelError::Timeout {
                operation,
                timeout: self.request_timeout,
            }
        } else {
            let retryable = error.is_connect() || error.is_request() || error.is_body();
            CloudModelError::Transport {
                operation,
                message: self.sanitize(&error.to_string()),
                retryable,
            }
        }
    }

    fn sanitize(&self, text: &str) -> String {
        let without_key = if self.gateway_token.is_empty() {
            text.to_string()
        } else {
            text.replace(&self.gateway_token, "[REDACTED]")
        };
        let data_urls = regex::Regex::new(r"(?i)data:image/(?:png|jpeg);base64,[A-Za-z0-9+/=_-]*")
            .expect("valid data URL regex");
        let without_images = data_urls
            .replace_all(&without_key, "[REDACTED-IMAGE-DATA]")
            .to_string();
        let long_base64 =
            regex::Regex::new(r"[A-Za-z0-9+/]{64,}={0,2}").expect("valid base64 regex");
        let without_base64 = long_base64
            .replace_all(&without_images, "[REDACTED-BASE64]")
            .to_string();
        let auth_header = regex::Regex::new("(?i)authorization").expect("valid auth regex");
        auth_header
            .replace_all(&without_base64, "[REDACTED-HEADER]")
            .to_string()
    }
}

fn serialize_chat_request_with_limit(
    request: &ChatCompletionRequest,
    maximum_bytes: usize,
) -> CloudModelResult<Vec<u8>> {
    let payload = serialize_chat_request(request)?;
    if payload.len() > maximum_bytes {
        return Err(chat_request_too_large(payload.len(), maximum_bytes));
    }
    Ok(payload)
}

fn serialize_chat_request(request: &ChatCompletionRequest) -> CloudModelResult<Vec<u8>> {
    serde_json::to_vec(request).map_err(|_| CloudModelError::InvalidRequest {
        operation: ModelOperation::Chat,
        message: "chat request could not be serialized".to_string(),
    })
}

fn chat_request_too_large(actual_bytes: usize, maximum_bytes: usize) -> CloudModelError {
    CloudModelError::InvalidRequest {
        operation: ModelOperation::Chat,
        message: format!(
            "serialized request is {actual_bytes} bytes; maximum is {maximum_bytes} bytes"
        ),
    }
}

fn bounded_retry_delay(
    base_delay: Duration,
    max_delay: Duration,
    jitter: bool,
    failed_attempt: usize,
) -> Duration {
    let exponent = u32::try_from(failed_attempt.saturating_sub(1))
        .unwrap_or(u32::MAX)
        .min(31);
    let multiplier = 1_u32 << exponent;
    let exponential = base_delay
        .checked_mul(multiplier)
        .unwrap_or(max_delay)
        .min(max_delay);
    if !jitter || exponential.is_zero() {
        return exponential;
    }

    // Equal jitter keeps a minimum delay while spreading retries over the
    // upper half of the exponentially growing window.
    let half = exponential / 2;
    let jitter_window_ms = exponential.saturating_sub(half).as_millis();
    let upper_bound = jitter_window_ms.saturating_add(1);
    let random = (uuid::Uuid::new_v4().as_u128() % upper_bound) as u64;
    half.saturating_add(Duration::from_millis(random))
}

#[cfg(test)]
fn image_data_url(image: &[u8]) -> CloudModelResult<String> {
    image_data_url_for(ModelOperation::Embed, image)
}

fn image_data_url_for(operation: ModelOperation, image: &[u8]) -> CloudModelResult<String> {
    if image.is_empty() {
        return Err(CloudModelError::InvalidRequest {
            operation,
            message: "image input must not be empty".to_string(),
        });
    }
    if image.len() > MAX_EMBED_IMAGE_BYTES {
        return Err(CloudModelError::InvalidRequest {
            operation,
            message: format!(
                "decoded image payload is {} bytes; maximum is {MAX_EMBED_IMAGE_BYTES} bytes",
                image.len()
            ),
        });
    }

    let mime = if image.starts_with(b"\x89PNG\r\n\x1a\n") {
        "image/png"
    } else if image.starts_with(&[0xff, 0xd8, 0xff]) {
        "image/jpeg"
    } else {
        return Err(CloudModelError::InvalidRequest {
            operation,
            message: "image input must be a PNG or JPEG payload".to_string(),
        });
    };
    let encoded = base64::engine::general_purpose::STANDARD.encode(image);
    Ok(format!("data:{mime};base64,{encoded}"))
}

fn validate_rerank_response(
    response: RerankResponse,
    expected_count: usize,
) -> CloudModelResult<RerankResponse> {
    if response.rankings.len() != expected_count {
        return Err(CloudModelError::InvalidResponse {
            operation: ModelOperation::Rerank,
            message: format!(
                "expected {expected_count} ranking entries, received {}",
                response.rankings.len()
            ),
        });
    }

    let mut seen = vec![false; expected_count];
    for (position, ranking) in response.rankings.iter().enumerate() {
        if ranking.index >= expected_count {
            return Err(CloudModelError::InvalidResponse {
                operation: ModelOperation::Rerank,
                message: format!(
                    "ranking entry {position} has out-of-range index {} for {expected_count} passages",
                    ranking.index
                ),
            });
        }
        if seen[ranking.index] {
            return Err(CloudModelError::InvalidResponse {
                operation: ModelOperation::Rerank,
                message: format!("ranking index {} is duplicated", ranking.index),
            });
        }
        if !ranking.logit.is_finite() {
            return Err(CloudModelError::InvalidResponse {
                operation: ModelOperation::Rerank,
                message: format!("ranking index {} has a non-finite logit", ranking.index),
            });
        }
        seen[ranking.index] = true;
    }

    Ok(response)
}

fn validate_embedding(embedding: Vec<f32>) -> CloudModelResult<Vec<f32>> {
    if embedding.len() != EMBEDDING_DIMENSIONS {
        return Err(CloudModelError::InvalidResponse {
            operation: ModelOperation::Embed,
            message: format!(
                "expected {EMBEDDING_DIMENSIONS} dimensions from {EMBED_MODEL_FAMILY}, received {}",
                embedding.len()
            ),
        });
    }
    if let Some(index) = embedding.iter().position(|value| !value.is_finite()) {
        return Err(CloudModelError::InvalidResponse {
            operation: ModelOperation::Embed,
            message: format!("embedding contains a non-finite value at index {index}"),
        });
    }
    Ok(embedding)
}

fn retryable_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS
    ) || status.is_server_error()
}

enum ChatSseProgress {
    Continue,
    Complete,
    ReceiverClosed,
}

async fn forward_chat_sse_line(
    line: &[u8],
    tx: &Sender<SseEvent>,
    filter: &mut ThinkFilter,
) -> CloudModelResult<ChatSseProgress> {
    let line = std::str::from_utf8(line)
        .map_err(|_| CloudModelError::InvalidResponse {
            operation: ModelOperation::Chat,
            message: "SSE stream contained a non-UTF-8 line".to_string(),
        })?
        .trim_end_matches(['\r', '\n']);
    let Some(payload) = line
        .strip_prefix("data:")
        .map(str::trim_start)
        .filter(|payload| !payload.is_empty())
    else {
        return Ok(ChatSseProgress::Continue);
    };

    if payload == "[DONE]" {
        if !send_filter_remainder(tx, filter).await {
            return Ok(ChatSseProgress::ReceiverClosed);
        }
        return Ok(if tx.send(SseEvent::Done).await.is_ok() {
            ChatSseProgress::Complete
        } else {
            ChatSseProgress::ReceiverClosed
        });
    }

    let chunk = serde_json::from_str::<ChatStreamChunk>(payload).map_err(|error| {
        CloudModelError::InvalidResponse {
            operation: ModelOperation::Chat,
            message: format!(
                "SSE data was not a valid chat chunk at line {}, column {}",
                error.line(),
                error.column()
            ),
        }
    })?;
    let Some(content) = chunk
        .choices
        .first()
        .and_then(|choice| choice.delta.content.as_deref())
    else {
        return Ok(ChatSseProgress::Continue);
    };
    let visible = filter.feed(content);
    if visible.is_empty() {
        return Ok(ChatSseProgress::Continue);
    }
    Ok(
        if tx.send(SseEvent::TextDelta { text: visible }).await.is_ok() {
            ChatSseProgress::Continue
        } else {
            ChatSseProgress::ReceiverClosed
        },
    )
}

async fn send_filter_remainder(tx: &Sender<SseEvent>, filter: &mut ThinkFilter) -> bool {
    match filter.flush() {
        Some(text) if !text.is_empty() => tx.send(SseEvent::TextDelta { text }).await.is_ok(),
        _ => true,
    }
}

struct ThinkFilter {
    in_think: bool,
    pending: String,
    emitted_visible: bool,
}

impl ThinkFilter {
    fn new() -> Self {
        Self {
            in_think: false,
            pending: String::new(),
            emitted_visible: false,
        }
    }

    fn feed(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        let mut output = String::new();

        loop {
            if self.in_think {
                if let Some(index) = self.pending.find("</think>") {
                    self.pending.drain(..index + "</think>".len());
                    self.in_think = false;
                    continue;
                }
                let keep = partial_suffix_len(&self.pending, "</think>");
                self.pending.drain(..self.pending.len() - keep);
                break;
            }

            if let Some(index) = self.pending.find("<think>") {
                let visible = self.pending.drain(..index).collect::<String>();
                self.pending.drain(.."<think>".len());
                self.in_think = true;
                if !visible.is_empty() {
                    self.emitted_visible = true;
                }
                output.push_str(&visible);
                continue;
            }

            let keep = partial_suffix_len(&self.pending, "<think>");
            let emit_len = self.pending.len() - keep;
            let mut visible = self.pending.drain(..emit_len).collect::<String>();
            if !self.emitted_visible {
                visible = visible.trim_start().to_string();
            }
            if !visible.is_empty() {
                self.emitted_visible = true;
            }
            output.push_str(&visible);
            break;
        }

        output
    }

    fn flush(&mut self) -> Option<String> {
        if self.in_think {
            self.pending.clear();
            None
        } else {
            let pending = std::mem::take(&mut self.pending);
            (!pending.is_empty()).then_some(pending)
        }
    }
}

fn partial_suffix_len(value: &str, tag: &str) -> usize {
    let max = value.len().min(tag.len() - 1);
    (1..=max)
        .rev()
        .find(|length| value.ends_with(&tag[..*length]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::{Body, Bytes, to_bytes};
    use axum::extract::{Request, State};
    use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
    use axum::response::{IntoResponse, Response as AxumResponse};
    use axum::routing::post;
    use std::collections::HashMap;
    use std::convert::Infallible;
    use std::sync::{Arc, Mutex};

    /// Paths produced by the default CF_AI_GATEWAY_BASE_URL derivation when the
    /// mock server address is used as the gateway base.
    const MOCK_GATEWAY_CHAT_PATH: &str = "/custom-nvidia-build/chat/completions";
    const MOCK_GATEWAY_EMBED_PATH: &str = "/custom-nvidia-build/embeddings";
    const MOCK_GATEWAY_RERANK_PATH: &str =
        "/custom-nvidia-ai/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking";
    const TEST_GATEWAY_BASE: &str =
        "https://gateway.ai.cloudflare.com/v1/test-account/test-gateway";

    #[derive(Clone, Copy)]
    enum MockMode {
        Success,
        SplitSse,
        ChatStatus(StatusCode),
        EchoSecretError,
        Slow,
        RetryOnce(StatusCode),
        PermanentBadRequest,
        WrongDimensions,
        MalformedRerank,
        MissingRerank,
        DuplicateRerank,
        OutOfRangeRerank,
    }

    #[derive(Debug)]
    struct RecordedRequest {
        path: String,
        authorization: Option<String>,
        gateway_authorization: Option<String>,
        body: serde_json::Value,
    }

    #[derive(Clone)]
    struct MockState {
        mode: MockMode,
        requests: Arc<Mutex<Vec<RecordedRequest>>>,
    }

    async fn mock_cloud(State(state): State<MockState>, request: Request) -> AxumResponse {
        let (parts, body) = request.into_parts();
        let path = parts.uri.path().to_string();
        let authorization = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let gateway_authorization = parts
            .headers
            .get(super::GATEWAY_AUTHORIZATION_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let bytes = to_bytes(body, usize::MAX).await.expect("read mock request");
        let body: serde_json::Value =
            serde_json::from_slice(&bytes).expect("mock request contains JSON");
        let request_number = {
            let mut requests = state.requests.lock().expect("request lock");
            requests.push(RecordedRequest {
                path: path.clone(),
                authorization: authorization.clone(),
                gateway_authorization: gateway_authorization.clone(),
                body: body.clone(),
            });
            requests.len()
        };

        match state.mode {
            MockMode::EchoSecretError => {
                let response_body = format!(
                    "Authorization: {} cf-aig-authorization: {} echoed-input={} {}",
                    authorization.unwrap_or_default(),
                    gateway_authorization.unwrap_or_default(),
                    body.pointer("/input/0")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default(),
                    "錯誤".repeat(700)
                );
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [(CONTENT_TYPE, "text/plain")],
                    response_body,
                )
                    .into_response();
            }
            MockMode::Slow => tokio::time::sleep(Duration::from_millis(100)).await,
            MockMode::RetryOnce(status) if request_number == 1 => {
                return status.into_response();
            }
            MockMode::PermanentBadRequest => {
                return (StatusCode::BAD_REQUEST, "invalid request").into_response();
            }
            MockMode::ChatStatus(status)
                if matches!(path.as_str(), MOCK_GATEWAY_CHAT_PATH | "/capabilities/chat") =>
            {
                return (
                    status,
                    "provider-secret-body data:image/jpeg;base64,PRIVATEPAYLOAD /private/path",
                )
                    .into_response();
            }
            MockMode::Success
            | MockMode::SplitSse
            | MockMode::ChatStatus(_)
            | MockMode::RetryOnce(_)
            | MockMode::WrongDimensions
            | MockMode::MalformedRerank
            | MockMode::MissingRerank
            | MockMode::DuplicateRerank
            | MockMode::OutOfRangeRerank => {}
        }

        match path.as_str() {
            MOCK_GATEWAY_CHAT_PATH | "/capabilities/chat"
                if body.get("stream") == Some(&serde_json::Value::Bool(true))
                    && matches!(state.mode, MockMode::SplitSse) =>
            {
                // Includes OpenRouter-style `:` keep-alive comment lines and
                // unknown JSON fields, which the parser must ignore.
                let chunks = [
                    ": OPENROUTER PROCESSING\n\n",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"<thi",
                    "nk>hidden</think>跨\"}}]}\n\n: OPENROUTER PROCESSING\n\ndata: {\"id\":\"gen-1\",\"choices\":[{\"delta\":{\"con",
                    "tent\":\"區塊 [S1]\",\"reasoning\":null}}],\"usage\":{\"total_tokens\":3}}\n\ndata: [DO",
                    "NE]\n\n",
                ]
                .into_iter()
                .map(|chunk| Ok::<_, Infallible>(Bytes::from_static(chunk.as_bytes())));
                (
                    StatusCode::OK,
                    [(CONTENT_TYPE, "text/event-stream")],
                    Body::from_stream(futures::stream::iter(chunks)),
                )
                    .into_response()
            }
            MOCK_GATEWAY_CHAT_PATH | "/capabilities/chat"
                if body.get("stream") == Some(&serde_json::Value::Bool(true)) =>
            {
                let sse = concat!(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"<think>hidden</think>雲\"}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"端回答\"}}]}\n\n",
                    "data: [DONE]\n\n"
                );
                (StatusCode::OK, [(CONTENT_TYPE, "text/event-stream")], sse).into_response()
            }
            MOCK_GATEWAY_CHAT_PATH | "/capabilities/chat" => (
                StatusCode::OK,
                [(CONTENT_TYPE, "application/json")],
                r#"{"choices":[{"message":{"content":"ok"}}]}"#,
            )
                .into_response(),
            MOCK_GATEWAY_EMBED_PATH | "/capabilities/embed" => {
                let dimensions = if matches!(state.mode, MockMode::WrongDimensions) {
                    EMBEDDING_DIMENSIONS - 1
                } else {
                    EMBEDDING_DIMENSIONS
                };
                let embedding = vec![0.25_f32; dimensions];
                let response_body = serde_json::json!({
                    "data": [{"index": 0, "embedding": embedding}]
                })
                .to_string();
                (
                    StatusCode::OK,
                    [(CONTENT_TYPE, "application/json")],
                    response_body,
                )
                    .into_response()
            }
            MOCK_GATEWAY_RERANK_PATH | "/capabilities/rerank" => {
                let passage_count = body["passages"]
                    .as_array()
                    .map(Vec::len)
                    .unwrap_or_default();
                let response_body = match state.mode {
                    MockMode::MalformedRerank => "not-json".to_string(),
                    MockMode::MissingRerank => serde_json::json!({ "rankings": [] }).to_string(),
                    MockMode::DuplicateRerank => serde_json::json!({
                        "rankings": (0..passage_count)
                            .map(|index| serde_json::json!({
                                "index": if index == 0 { 0 } else { index - 1 },
                                "logit": index as f64,
                            }))
                            .collect::<Vec<_>>()
                    })
                    .to_string(),
                    MockMode::OutOfRangeRerank => serde_json::json!({
                        "rankings": (0..passage_count)
                            .map(|index| serde_json::json!({
                                "index": if index == 0 { passage_count } else { index },
                                "logit": index as f64,
                            }))
                            .collect::<Vec<_>>()
                    })
                    .to_string(),
                    _ => serde_json::json!({
                        "rankings": (0..passage_count)
                            .rev()
                            .map(|index| serde_json::json!({
                                "index": index,
                                "logit": index as f64 + 0.1,
                            }))
                            .collect::<Vec<_>>()
                    })
                    .to_string(),
                };
                (
                    StatusCode::OK,
                    [(CONTENT_TYPE, "application/json")],
                    response_body,
                )
                    .into_response()
            }
            _ => StatusCode::NOT_FOUND.into_response(),
        }
    }

    async fn spawn_mock(mode: MockMode) -> (String, MockState, tokio::task::JoinHandle<()>) {
        let state = MockState {
            mode,
            requests: Arc::new(Mutex::new(Vec::new())),
        };
        let app = Router::new()
            .route(MOCK_GATEWAY_CHAT_PATH, post(mock_cloud))
            .route(MOCK_GATEWAY_EMBED_PATH, post(mock_cloud))
            .route(MOCK_GATEWAY_RERANK_PATH, post(mock_cloud))
            .route("/capabilities/chat", post(mock_cloud))
            .route("/capabilities/embed", post(mock_cloud))
            .route("/capabilities/rerank", post(mock_cloud))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock listener");
        let address = listener.local_addr().expect("mock address");
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve mock");
        });
        (format!("http://{address}"), state, handle)
    }

    fn config_from(values: &[(&str, &str)]) -> Result<CloudModelConfig, CloudModelConfigError> {
        let values = values
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect::<HashMap<_, _>>();
        CloudModelConfig::from_lookup(|name| values.get(name).cloned())
    }

    fn test_config(gateway_base_url: &str, gateway_token: &str) -> CloudModelConfig {
        test_config_with_endpoints(
            &format!("{gateway_base_url}{MOCK_GATEWAY_CHAT_PATH}"),
            &format!("{gateway_base_url}{MOCK_GATEWAY_EMBED_PATH}"),
            &format!("{gateway_base_url}{MOCK_GATEWAY_RERANK_PATH}"),
            gateway_token,
        )
    }

    fn test_config_with_endpoints(
        chat_endpoint: &str,
        embed_endpoint: &str,
        rerank_endpoint: &str,
        gateway_token: &str,
    ) -> CloudModelConfig {
        CloudModelConfig {
            gateway_token: gateway_token.to_owned(),
            chat_endpoint: Url::parse(chat_endpoint).expect("test chat endpoint"),
            embed_endpoint: Url::parse(embed_endpoint).expect("test embed endpoint"),
            rerank_endpoint: Url::parse(rerank_endpoint).expect("test rerank endpoint"),
            chat_model: "hosted/google-gemma-4-test".to_owned(),
            embed_model: DEFAULT_EMBED_MODEL_ID.to_owned(),
            rerank_model: DEFAULT_RERANK_MODEL_ID.to_owned(),
            request_timeout: Duration::from_secs(1),
            rerank_candidate_top_k: DEFAULT_RERANK_CANDIDATE_TOP_K,
            rerank_top_n: DEFAULT_RERANK_TOP_N,
            rerank_batch_size: DEFAULT_RERANK_BATCH_SIZE,
        }
    }

    fn test_client(base_url: &str, gateway_token: &str, timeout: Duration) -> CloudModelClient {
        let mut config = test_config(base_url, gateway_token);
        config.request_timeout = timeout;
        let mut client = CloudModelClient::new(config).expect("cloud client");
        client.embedding_retry = EmbeddingRetryConfig {
            max_attempts: DEFAULT_EMBED_MAX_ATTEMPTS,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            jitter: false,
        };
        client.rerank_retry = RerankRetryConfig {
            max_attempts: DEFAULT_RERANK_MAX_ATTEMPTS,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            jitter: false,
        };
        client
    }

    fn minimal_png() -> Vec<u8> {
        b"\x89PNG\r\n\x1a\nmock-png-payload".to_vec()
    }

    #[tokio::test]
    async fn telemetry_snapshots_are_content_free_deltas_shared_across_clones() {
        let (base_url, _state, server) = spawn_mock(MockMode::Success).await;
        let client = test_client(&base_url, "sensitive-gateway-token", Duration::from_secs(1));
        let clone = client.clone();
        let before = client.telemetry_snapshot();

        clone
            .embed_passage("sensitive passage that must never enter telemetry")
            .await
            .expect("embedding request");
        clone
            .rerank(
                "sensitive query",
                &[RerankPassage::text("sensitive passage")],
            )
            .await
            .expect("rerank request");
        clone
            .chat_once("fixed system prompt", "sensitive user content")
            .await
            .expect("chat request");

        let delta = client.telemetry_snapshot().delta_since(before);
        assert_eq!(
            delta,
            CloudRequestTelemetry {
                chat_attempts: 1,
                embedding_attempts: 1,
                rerank_attempts: 1,
            }
        );
        let debug = format!("{delta:?}");
        for forbidden in [
            "sensitive-gateway-token",
            &base_url,
            "sensitive query",
            "sensitive passage",
            "authorization",
        ] {
            assert!(!debug.contains(forbidden));
        }
        server.abort();
    }

    #[tokio::test]
    async fn non_streaming_rag_answer_uses_fixed_gemma_request_contract() {
        let (base_url, state, server) = spawn_mock(MockMode::Success).await;
        let client = test_client(&base_url, "gateway-token", Duration::from_secs(1));
        let answer = client
            .answer_rag_once(
                "fixed RAG system prompt",
                ApiChatContent::Parts(vec![ApiChatContentPart::Text {
                    text: "[S1] bounded context".to_owned(),
                }]),
                "使用者問題",
            )
            .await
            .expect("non-streaming RAG answer");
        assert_eq!(answer, "ok");

        let requests = state.requests.lock().expect("requests lock");
        let request = requests.last().expect("recorded chat request");
        assert_eq!(request.body["model"], client.chat_model());
        assert_eq!(request.body["stream"], false);
        assert_eq!(request.body["max_tokens"], 4096);
        assert_eq!(request.body["temperature"], 0.0);
        let messages = request.body["messages"].as_array().expect("messages array");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["content"][0]["text"], "[S1] bounded context");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"], "使用者問題");
        drop(requests);
        server.abort();
    }

    #[test]
    fn evaluation_client_locks_the_reported_retry_policy_without_jitter() {
        let client = CloudModelClient::new(test_config(TEST_GATEWAY_BASE, "gateway-token"))
            .expect("cloud client")
            .with_deterministic_evaluation_retries();
        assert_eq!(
            client.evaluation_retry_policies(),
            CloudRetryPolicies {
                embedding: CloudRetryPolicy {
                    max_attempts: 3,
                    base_backoff_ms: 250,
                    max_backoff_ms: 2_000,
                    jitter: false,
                },
                rerank: CloudRetryPolicy {
                    max_attempts: 3,
                    base_backoff_ms: 250,
                    max_backoff_ms: 2_000,
                    jitter: false,
                },
                generation: CloudRetryPolicy {
                    max_attempts: 1,
                    base_backoff_ms: 0,
                    max_backoff_ms: 0,
                    jitter: false,
                },
            }
        );
    }

    #[test]
    fn config_requires_gateway_token_and_explicit_chat_model() {
        let missing_token = config_from(&[]).expect_err("gateway token must be required");
        assert!(missing_token.to_string().contains("CF_AI_GATEWAY_TOKEN"));

        let config = config_from(&[
            ("CF_AI_GATEWAY_TOKEN", "secret"),
            ("GEMMA_4_MODEL_ID", "google/gemma-4-31b-it"),
            (
                "CLOUD_CHAT_URL",
                "https://gateway.ai.cloudflare.com/v1/acct/gw/custom-nvidia-build/chat/completions",
            ),
            ("CF_AI_GATEWAY_BASE_URL", TEST_GATEWAY_BASE),
        ])
        .expect("full provider URL and explicit model should be sufficient");
        assert_eq!(config.chat_model(), "google/gemma-4-31b-it");

        let explicit = config_from(&[
            ("CF_AI_GATEWAY_TOKEN", "secret"),
            ("GEMMA_4_MODEL_ID", "provider/custom-chat-model"),
            ("CF_AI_GATEWAY_BASE_URL", TEST_GATEWAY_BASE),
        ])
        .expect("explicit model id remains supported");
        assert_eq!(explicit.chat_model(), "provider/custom-chat-model");
    }

    #[test]
    fn config_uses_explicit_embed_and_rerank_models_with_full_urls() {
        let config = config_from(&[
            ("CF_AI_GATEWAY_TOKEN", "gateway-secret"),
            ("GEMMA_4_MODEL_ID", "hosted/gemma-4-production"),
            ("CF_AI_GATEWAY_BASE_URL", TEST_GATEWAY_BASE),
        ])
        .expect("default model config");
        assert_eq!(config.embed_model(), DEFAULT_EMBED_MODEL_ID);
        assert_eq!(config.rerank_model(), DEFAULT_RERANK_MODEL_ID);
        let debug = format!("{config:?}").to_ascii_lowercase();
        assert!(!debug.contains("secret"));
        assert!(!debug.contains("authorization"));

        let configured = config_from(&[
            ("CF_AI_GATEWAY_TOKEN", "secret"),
            ("GEMMA_4_MODEL_ID", "hosted/gemma-4-production"),
            ("NEMOTRON_EMBED_MODEL_ID", "vendor/custom-embedder"),
            ("NEMOTRON_RERANK_MODEL_ID", "vendor/custom-reranker"),
            (
                "CLOUD_CHAT_URL",
                "https://gateway.ai.cloudflare.com/v1/acct/gw/provider/chat/completions",
            ),
            (
                "CLOUD_EMBED_URL",
                "https://gateway.ai.cloudflare.com/v1/acct/gw/provider/embeddings",
            ),
            (
                "CLOUD_RERANK_URL",
                "https://gateway.ai.cloudflare.com/v1/acct/gw/provider/reranking",
            ),
        ])
        .expect("full provider routes should select capability models");
        assert_eq!(configured.embed_model(), "vendor/custom-embedder");
        assert_eq!(configured.rerank_model(), "vendor/custom-reranker");
    }

    #[test]
    fn config_derives_gateway_routes_and_requires_base_or_full_urls() {
        let config = config_from(&[
            ("CF_AI_GATEWAY_TOKEN", "secret"),
            ("GEMMA_4_MODEL_ID", "hosted/gemma-4-production"),
            // Trailing slash must not produce a double slash in derived routes.
            (
                "CF_AI_GATEWAY_BASE_URL",
                "https://gateway.ai.cloudflare.com/v1/acct/gw/",
            ),
        ])
        .expect("gateway-derived config");
        assert_eq!(
            config.chat_endpoint().as_str(),
            "https://gateway.ai.cloudflare.com/v1/acct/gw/custom-nvidia-build/chat/completions"
        );
        assert_eq!(
            config.embed_endpoint().as_str(),
            "https://gateway.ai.cloudflare.com/v1/acct/gw/custom-nvidia-build/embeddings"
        );
        assert_eq!(
            config.rerank_endpoint().as_str(),
            "https://gateway.ai.cloudflare.com/v1/acct/gw/custom-nvidia-ai/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking"
        );

        let missing_base = config_from(&[
            ("CF_AI_GATEWAY_TOKEN", "secret"),
            ("GEMMA_4_MODEL_ID", "hosted/gemma-4-production"),
        ])
        .expect_err("no gateway base and no full URLs must fail before startup");
        assert!(missing_base.to_string().contains("CF_AI_GATEWAY_BASE_URL"));
        assert!(missing_base.to_string().contains("CLOUD_CHAT_URL"));

        let overrides_only = config_from(&[
            ("CF_AI_GATEWAY_TOKEN", "secret"),
            ("GEMMA_4_MODEL_ID", "hosted/gemma-4-production"),
            (
                "CLOUD_CHAT_URL",
                "https://gateway.ai.cloudflare.com/v1/acct/gw/openrouter/v1/chat/completions",
            ),
            (
                "CLOUD_EMBED_URL",
                "https://gateway.ai.cloudflare.com/v1/acct/gw/custom-nvidia-build/embeddings",
            ),
            (
                "CLOUD_RERANK_URL",
                "https://gateway.ai.cloudflare.com/v1/acct/gw/custom-nvidia-ai/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking",
            ),
        ])
        .expect("three full gateway routes do not need the shared base");
        assert_eq!(
            overrides_only.chat_endpoint().as_str(),
            "https://gateway.ai.cloudflare.com/v1/acct/gw/openrouter/v1/chat/completions"
        );
    }

    #[test]
    fn config_validates_rerank_pool_batch_and_top_n() {
        let config = config_from(&[
            ("CF_AI_GATEWAY_TOKEN", "secret"),
            ("GEMMA_4_MODEL_ID", "hosted/gemma-4-production"),
            ("CF_AI_GATEWAY_BASE_URL", TEST_GATEWAY_BASE),
        ])
        .expect("default rerank config");
        assert_eq!(
            config.rerank_candidate_top_k(),
            DEFAULT_RERANK_CANDIDATE_TOP_K
        );
        assert_eq!(config.rerank_top_n(), DEFAULT_RERANK_TOP_N);
        assert_eq!(config.rerank_batch_size(), DEFAULT_RERANK_BATCH_SIZE);

        for top_n in [MIN_RERANK_TOP_N, DEFAULT_RERANK_TOP_N, MAX_RERANK_TOP_N] {
            let raw = top_n.to_string();
            let config = config_from(&[
                ("CF_AI_GATEWAY_TOKEN", "secret"),
                ("GEMMA_4_MODEL_ID", "hosted/gemma-4-production"),
                ("CF_AI_GATEWAY_BASE_URL", TEST_GATEWAY_BASE),
                ("RERANK_TOP_N", &raw),
            ])
            .expect("supported top_n");
            assert_eq!(config.rerank_top_n(), top_n);
        }

        for (name, value) in [
            ("RERANK_TOP_N", "2"),
            ("RERANK_TOP_N", "9"),
            ("RERANK_BATCH_SIZE", "0"),
            ("RERANK_CANDIDATE_TOP_K", "0"),
        ] {
            let error = config_from(&[
                ("CF_AI_GATEWAY_TOKEN", "secret"),
                ("GEMMA_4_MODEL_ID", "hosted/gemma-4-production"),
                ("CF_AI_GATEWAY_BASE_URL", TEST_GATEWAY_BASE),
                (name, value),
            ])
            .expect_err("invalid rerank config must fail before startup");
            assert!(error.to_string().contains(name));
        }

        let oversized_batch = (MAX_RERANK_BATCH_SIZE + 1).to_string();
        let error = config_from(&[
            ("CF_AI_GATEWAY_TOKEN", "secret"),
            ("GEMMA_4_MODEL_ID", "hosted/gemma-4-production"),
            ("CF_AI_GATEWAY_BASE_URL", TEST_GATEWAY_BASE),
            ("RERANK_BATCH_SIZE", &oversized_batch),
        ])
        .expect_err("hosted batch limit must fail before startup");
        assert!(error.to_string().contains("RERANK_BATCH_SIZE"));
    }

    #[tokio::test]
    async fn mock_server_observes_capability_specific_models_and_sse_contract() {
        let (base_url, state, server) = spawn_mock(MockMode::Success).await;
        let chat_url = format!("{base_url}/capabilities/chat");
        let embed_url = format!("{base_url}/capabilities/embed");
        let rerank_url = format!("{base_url}/capabilities/rerank");
        let mut config =
            test_config_with_endpoints(&chat_url, &embed_url, &rerank_url, "gateway-run-token");
        config.chat_model = "hosted/google-gemma-4-test".to_owned();
        let client = CloudModelClient::new(config).expect("cloud client");

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        client
            .stream_chat(
                vec![ChatMessage {
                    role: "user".to_string(),
                    content: "問題".to_string(),
                }],
                tx,
            )
            .await;
        let mut text = String::new();
        let mut done = false;
        while let Some(event) = rx.recv().await {
            match event {
                SseEvent::TextDelta { text: delta } => text.push_str(&delta),
                SseEvent::Done => {
                    done = true;
                    break;
                }
                SseEvent::Error { message } => panic!("unexpected SSE error: {message}"),
                SseEvent::Sources { .. } => {}
            }
        }
        assert_eq!(text, "雲端回答");
        assert!(done);

        let query_embedding = client.embed_query("query").await.expect("query embedding");
        let passage_embedding = client
            .embed_passage("passage")
            .await
            .expect("passage embedding");
        let image = minimal_png();
        let image_embedding = client
            .embed_image_passage(&image)
            .await
            .expect("image embedding");
        let mixed_embedding = client
            .embed_image_text_passage("same-page text", &image)
            .await
            .expect("image and text embedding");
        for embedding in [
            query_embedding,
            passage_embedding,
            image_embedding,
            mixed_embedding,
        ] {
            assert_eq!(embedding.len(), EMBEDDING_DIMENSIONS);
        }

        let ranking = client
            .rerank(
                "query",
                &[
                    RerankPassage::image_text("same-page text", &image)
                        .expect("image+text rerank passage"),
                    RerankPassage::image(&image).expect("image-only rerank passage"),
                    RerankPassage::text("text-only passage"),
                ],
            )
            .await
            .expect("ranking");
        assert_eq!(ranking.rankings[0].index, 2);

        let requests = state.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 6);
        assert!(requests.iter().all(|request| {
            request.gateway_authorization.as_deref() == Some("Bearer gateway-run-token")
        }));
        // Provider keys live in the gateway's BYOK store; the backend must
        // never send an upstream Authorization header.
        assert!(
            requests
                .iter()
                .all(|request| request.authorization.is_none())
        );

        let chat = requests
            .iter()
            .find(|request| request.path == "/capabilities/chat")
            .expect("chat request");
        assert_eq!(chat.body["model"], "hosted/google-gemma-4-test");
        assert_eq!(chat.body["stream"], true);
        let chat_messages = chat.body["messages"].as_array().expect("chat messages");
        assert_eq!(chat_messages.len(), 2);
        assert_eq!(chat_messages[0]["role"], "system");
        assert_eq!(chat_messages[1]["role"], "user");
        assert_eq!(chat_messages[1]["content"], "問題");

        let embeds = requests
            .iter()
            .filter(|request| request.path == "/capabilities/embed")
            .collect::<Vec<_>>();
        assert_eq!(embeds.len(), 4);
        assert!(embeds.iter().all(|request| {
            request.body["model"] == DEFAULT_EMBED_MODEL_ID
                && request.body["embedding_type"] == "float"
                && request.body["encoding_format"] == "float"
                && request.body["truncate"] == "NONE"
                && request.body.get("dimensions").is_none()
        }));
        assert_eq!(embeds[0].body["input_type"], "query");
        assert_eq!(embeds[0].body["modality"], serde_json::json!(["text"]));
        assert_eq!(embeds[0].body["input"][0], "query");
        assert_eq!(embeds[1].body["input_type"], "passage");
        assert_eq!(embeds[1].body["modality"], serde_json::json!(["text"]));
        assert_eq!(embeds[1].body["input"][0], "passage");
        assert_eq!(embeds[2].body["input_type"], "passage");
        assert_eq!(embeds[2].body["modality"], serde_json::json!(["image"]));
        let image_input = embeds[2].body["input"][0]
            .as_str()
            .expect("image input string");
        assert!(image_input.starts_with("data:image/png;base64,"));
        assert!(!image_input.contains("/tmp/"));
        assert!(!image_input.contains("pages/"));
        assert_eq!(embeds[3].body["input_type"], "passage");
        assert_eq!(
            embeds[3].body["modality"],
            serde_json::json!(["text_image"])
        );
        let mixed_input = embeds[3].body["input"][0]
            .as_str()
            .expect("text_image input string");
        assert!(mixed_input.starts_with("same-page text\n<img src=\"data:image/png;base64,"));
        assert!(mixed_input.ends_with("\"/>"));

        let rerank = requests
            .iter()
            .find(|request| request.path == "/capabilities/rerank")
            .expect("rerank request");
        assert_eq!(rerank.body["model"], DEFAULT_RERANK_MODEL_ID);
        assert_eq!(rerank.body["query"], serde_json::json!({ "text": "query" }));
        assert_eq!(rerank.body["truncate"], "NONE");
        let passages = rerank.body["passages"]
            .as_array()
            .expect("rerank passages array");
        assert_eq!(passages.len(), 3);
        assert_eq!(passages[0]["text"], "same-page text");
        assert!(
            passages[0]["image"]
                .as_str()
                .is_some_and(|image| image.starts_with("data:image/png;base64,"))
        );
        assert!(passages[1].get("text").is_none());
        assert!(
            passages[1]["image"]
                .as_str()
                .is_some_and(|image| image.starts_with("data:image/png;base64,"))
        );
        assert_eq!(
            passages[2],
            serde_json::json!({ "text": "text-only passage" })
        );
        let rerank_debug = format!(
            "{:?}",
            RerankRequest {
                model: DEFAULT_RERANK_MODEL_ID.to_string(),
                query: RerankQuery {
                    text: "private query".to_string(),
                },
                passages: vec![RerankPassage::image(&image).expect("debug passage")],
                truncate: "NONE".to_string(),
            }
        );
        assert!(!rerank_debug.contains("private query"));
        assert!(!rerank_debug.contains("base64"));
        server.abort();
    }

    #[tokio::test]
    async fn one_shot_multimodal_chat_uses_locked_bounded_generation_contract() {
        let (base_url, state, server) = spawn_mock(MockMode::Success).await;
        let client = test_client(&base_url, "test-api-key", Duration::from_secs(1));
        let content = ApiChatContent::Parts(vec![
            ApiChatContentPart::Text {
                text: "untrusted document source".to_string(),
            },
            ApiChatContentPart::ImageUrl {
                image_url: ApiImageUrl {
                    url: "data:image/png;base64,fixture".to_string(),
                },
            },
        ]);

        assert_eq!(
            client
                .chat_once_with_context("strict summary system prompt", content)
                .await
                .expect("one-shot response"),
            "ok"
        );

        let requests = state.requests.lock().expect("request lock");
        let request = requests.last().expect("summary request");
        assert_eq!(request.body["model"], "hosted/google-gemma-4-test");
        assert_eq!(request.body["stream"], false);
        assert_eq!(request.body["max_tokens"], 512);
        assert_eq!(request.body["temperature"], 0.0);
        let messages = request.body["messages"].as_array().expect("messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(
            messages[1]["content"]
                .as_array()
                .expect("multimodal parts")
                .len(),
            2
        );
        drop(requests);
        server.abort();
    }

    #[tokio::test]
    async fn gemma_multimodal_request_places_three_to_six_page_context_before_history() {
        for source_count in [3_usize, 6] {
            let (base_url, state, server) = spawn_mock(MockMode::Success).await;
            let client = test_client(&base_url, "test-api-key", Duration::from_secs(1));
            let mut parts = Vec::new();
            for page in 1..=source_count {
                parts.push(ApiChatContentPart::Text {
                    text: format!(
                        "[S{page}] source_id: S{page}\nfilename: manual.pdf\npage: {page}"
                    ),
                });
                parts.push(ApiChatContentPart::ImageUrl {
                    image_url: ApiImageUrl {
                        url: format!("data:image/jpeg;base64,PAGE{page}"),
                    },
                });
            }
            let (tx, mut rx) = tokio::sync::mpsc::channel(8);
            client
                .stream_chat_with_context(
                    "只依來源回答".to_string(),
                    ApiChatContent::Parts(parts),
                    vec![
                        ChatMessage {
                            role: "user".to_string(),
                            content: "歷史問題".to_string(),
                        },
                        ChatMessage {
                            role: "assistant".to_string(),
                            content: "歷史回答".to_string(),
                        },
                    ],
                    tx,
                )
                .await;
            assert!(rx.recv().await.is_some());

            let requests = state.requests.lock().expect("request lock");
            let chat = requests.last().expect("chat request");
            assert_eq!(chat.body["model"], "hosted/google-gemma-4-test");
            let messages = chat.body["messages"].as_array().expect("messages array");
            assert_eq!(messages.len(), 4);
            assert_eq!(messages[0]["role"], "system");
            assert_eq!(messages[1]["role"], "user");
            assert_eq!(messages[2]["content"], "歷史問題");
            assert_eq!(messages[3]["role"], "assistant");
            let context = messages[1]["content"]
                .as_array()
                .expect("multipart context");
            assert_eq!(context.len(), source_count * 2);
            for page in 1..=source_count {
                let text = context[(page - 1) * 2]["text"]
                    .as_str()
                    .expect("source text block");
                assert!(text.contains(&format!("source_id: S{page}")));
                assert!(text.contains(&format!("page: {page}")));
                assert!(
                    context[(page - 1) * 2 + 1]["image_url"]["url"]
                        .as_str()
                        .is_some_and(|url| url.starts_with("data:image/jpeg;base64,"))
                );
            }
            drop(requests);
            server.abort();
        }
    }

    #[tokio::test]
    async fn candidates_flow_to_matching_gemma_parts_and_structured_sources_without_inline_citation()
     {
        let storage = crate::storage::Storage::memory();
        let page_document_id = uuid::Uuid::new_v4();
        let page_path = format!("{page_document_id}/artifacts-run/pages/0005.png");
        let mut png = Vec::new();
        image::RgbImage::from_pixel(8, 8, image::Rgb([20, 40, 60]))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("encode valid integration PNG");
        storage
            .put(&page_path, png)
            .await
            .expect("store integration PNG");
        let text_document_id = uuid::Uuid::new_v4();
        let candidates = vec![
            crate::search::RetrievalCandidate {
                unit_id: 41,
                unit_index: 4,
                document_id: page_document_id,
                filename: "shared-name.pdf".to_string(),
                kind: "page".to_string(),
                page_number: Some(5),
                image_path: Some(page_path.clone()),
                content: Some("page fact".to_string()),
                dense_rank: Some(1),
                dense_score: Some(0.9),
                bm25_rank: None,
                bm25_score: None,
                rrf_score: 0.1,
                rerank_score: Some(0.8),
                rerank_rank: Some(1),
            },
            crate::search::RetrievalCandidate {
                unit_id: 42,
                unit_index: 1,
                document_id: text_document_id,
                filename: "shared-name.pdf".to_string(),
                kind: "text".to_string(),
                page_number: None,
                image_path: None,
                content: Some("text fact".to_string()),
                dense_rank: Some(2),
                dense_score: Some(0.8),
                bm25_rank: Some(1),
                bm25_score: Some(2.0),
                rrf_score: 0.09,
                rerank_score: Some(0.7),
                rerank_rank: Some(2),
            },
        ];
        let context = crate::rag_context::build_rag_context(&candidates, &storage)
            .await
            .expect("build real RAG context");
        assert_eq!(context.sources.len(), 2);
        assert_eq!(context.sources[0].source_id, "S1");
        assert_eq!(context.sources[0].page_number, Some(5));
        assert_eq!(context.sources[1].source_id, "S2");
        assert_eq!(context.sources[1].page_number, None);
        assert_ne!(
            context.sources[0].document_id,
            context.sources[1].document_id
        );

        let (base_url, state, server) = spawn_mock(MockMode::Success).await;
        let client = test_client(&base_url, "test-api-key", Duration::from_secs(1));
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "question".to_string(),
        }];
        let context = client
            .fit_rag_context_to_request(crate::rag_context::RAG_SYSTEM_PROMPT, context, &messages)
            .expect("fit complete request");
        let sources = context.sources.clone();
        let (inner_tx, mut inner_rx) = tokio::sync::mpsc::channel(8);
        client
            .stream_chat_with_context(
                crate::rag_context::RAG_SYSTEM_PROMPT.to_string(),
                context.content,
                messages,
                inner_tx,
            )
            .await;

        let (outer_tx, mut outer_rx) = tokio::sync::mpsc::channel(8);
        let upstream = tokio::spawn(std::future::pending::<()>());
        let abort_handle = upstream.abort_handle();
        assert!(crate::forward_rag_events(&mut inner_rx, &outer_tx, &sources, &abort_handle).await);
        upstream.abort();
        drop(outer_tx);
        let mut answer = String::new();
        let structured_sources = loop {
            match outer_rx.recv().await {
                Some(SseEvent::TextDelta { text }) => answer.push_str(&text),
                Some(SseEvent::Sources { sources }) => break sources,
                other => panic!("expected text deltas followed by sources, got {other:?}"),
            }
        };
        assert_eq!(answer, "雲端回答");
        assert!(!answer.contains("[S"));
        assert_eq!(structured_sources.len(), 2);
        assert_eq!(structured_sources[0].source_id, "S1");
        assert_eq!(structured_sources[0].page_number, Some(5));
        assert_eq!(structured_sources[1].source_id, "S2");
        assert!(structured_sources[1].page_number.is_none());
        assert!(matches!(outer_rx.recv().await, Some(SseEvent::Done)));

        let requests = state.requests.lock().expect("request lock");
        let context_parts = requests[0].body["messages"][1]["content"]
            .as_array()
            .expect("real multipart context");
        assert!(
            context_parts[0]["text"]
                .as_str()
                .is_some_and(|text| text.contains("source_id: S1") && text.contains("page: 5"))
        );
        assert!(
            context_parts[1]["image_url"]["url"]
                .as_str()
                .is_some_and(|url| url.starts_with("data:image/jpeg;base64,"))
        );
        assert!(
            context_parts[2]["text"]
                .as_str()
                .is_some_and(|text| text.contains("source_id: S2") && !text.contains("page:"))
        );
        drop(requests);
        server.abort();
    }

    #[tokio::test]
    async fn chat_sse_parser_handles_split_chunks_comment_keepalives_and_unknown_fields() {
        let (base_url, state, server) = spawn_mock(MockMode::SplitSse).await;
        let client = test_client(&base_url, "test-api-key", Duration::from_secs(1));
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        client
            .stream_chat_with_context(
                "system".to_string(),
                ApiChatContent::Parts(Vec::new()),
                vec![ChatMessage {
                    role: "user".to_string(),
                    content: "question".to_string(),
                }],
                tx,
            )
            .await;

        let mut text = String::new();
        let mut done_count = 0;
        while let Some(event) = rx.recv().await {
            match event {
                SseEvent::TextDelta { text: delta } => text.push_str(&delta),
                SseEvent::Done => done_count += 1,
                SseEvent::Error { message } => panic!("unexpected error: {message}"),
                SseEvent::Sources { .. } => panic!("provider must not emit sources"),
            }
        }
        assert_eq!(text, "跨區塊 [S1]");
        assert_eq!(done_count, 1);

        // The default gateway-derived chat route was used, the gateway token
        // authenticated the request, and no provider Authorization header
        // left the backend (provider keys live in the gateway's BYOK store).
        let requests = state.requests.lock().expect("request lock");
        assert_eq!(requests[0].path, MOCK_GATEWAY_CHAT_PATH);
        assert_eq!(
            requests[0].gateway_authorization.as_deref(),
            Some("Bearer test-api-key")
        );
        assert!(requests[0].authorization.is_none());
        drop(requests);
        server.abort();
    }

    #[test]
    fn serialized_chat_request_hard_cap_counts_entire_multimodal_payload() {
        let client = CloudModelClient::for_test();
        let secret = "DO_NOT_LEAK_REQUEST_CONTENT";
        let request = client.chat_request_with_context(
            format!("system-{secret}"),
            ApiChatContent::Parts(vec![
                ApiChatContentPart::Text {
                    text: format!("[S1] {secret}"),
                },
                ApiChatContentPart::ImageUrl {
                    image_url: ApiImageUrl {
                        url: format!("data:image/jpeg;base64,{secret}"),
                    },
                },
            ]),
            vec![ChatMessage {
                role: "user".to_string(),
                content: format!("history-{secret}"),
            }],
            true,
            4096,
            None,
        );
        let request_debug = format!("{request:?}");
        assert!(request_debug.contains("message_count"));
        assert!(!request_debug.contains(secret));
        assert!(!request_debug.contains("base64"));
        let complete = serialize_chat_request_with_limit(&request, usize::MAX)
            .expect("serialize complete request");
        let error = serialize_chat_request_with_limit(&request, complete.len() - 1)
            .expect_err("one-byte-short cap must reject the complete request");
        assert!(matches!(
            error,
            CloudModelError::InvalidRequest {
                operation: ModelOperation::Chat,
                ..
            }
        ));
        assert!(!error.chat_client_message().contains(secret));

        let oversized = client.chat_request(
            "x".repeat(MAX_CHAT_REQUEST_BYTES),
            Vec::new(),
            true,
            4096,
            None,
        );
        assert!(matches!(
            serialize_chat_request_with_limit(&oversized, MAX_CHAT_REQUEST_BYTES),
            Err(CloudModelError::InvalidRequest {
                operation: ModelOperation::Chat,
                ..
            })
        ));
    }

    #[test]
    fn complete_request_budget_evicts_lowest_source_before_rejecting_rank_one() {
        let client = CloudModelClient::for_test();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: format!("history-overhead-{}", "h".repeat(300)),
        }];
        let first_part = ApiChatContentPart::Text {
            text: format!("[S1] {}", "a".repeat(400)),
        };
        let second_part = ApiChatContentPart::Text {
            text: format!("[S2] {}", "b".repeat(400)),
        };
        let source = |index: u128| crate::rag_context::SourceRef {
            source_id: format!("S{index}"),
            document_id: uuid::Uuid::from_u128(index),
            unit_id: index as i64,
            unit_index: index as i32 - 1,
            filename: format!("source-{index}.md"),
            page_number: None,
            kind: "text".to_string(),
        };

        let one_source_request = client.chat_request_with_context(
            "system".to_string(),
            ApiChatContent::Parts(vec![first_part.clone()]),
            messages.clone(),
            true,
            4096,
            None,
        );
        let one_source_bytes =
            serialize_chat_request(&one_source_request).expect("measure one-source request");
        let context = crate::rag_context::RagContext {
            content: ApiChatContent::Parts(vec![first_part, second_part]),
            sources: vec![source(1), source(2)],
        };
        let fitted = client
            .fit_rag_context_to_request_with_limit(
                "system",
                context,
                &messages,
                one_source_bytes.len(),
            )
            .expect("lowest source should be evicted");
        assert_eq!(fitted.sources.len(), 1);
        assert_eq!(fitted.sources[0].source_id, "S1");
        let ApiChatContent::Parts(parts) = &fitted.content else {
            panic!("fitted context must remain multipart");
        };
        assert_eq!(parts.len(), 1);

        let error = client
            .fit_rag_context_to_request_with_limit(
                "system",
                fitted,
                &messages,
                one_source_bytes.len() - 1,
            )
            .expect_err("rank one must not be silently removed");
        assert!(matches!(
            error,
            CloudModelError::InvalidRequest {
                operation: ModelOperation::Chat,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn chat_4xx_and_5xx_emit_one_safe_error_without_done() {
        for status in [StatusCode::BAD_REQUEST, StatusCode::INTERNAL_SERVER_ERROR] {
            let (base_url, state, server) = spawn_mock(MockMode::ChatStatus(status)).await;
            let client = test_client(&base_url, "never-print-api-key", Duration::from_secs(1));
            let (tx, mut rx) = tokio::sync::mpsc::channel(4);
            client
                .stream_chat_with_context(
                    "system".to_string(),
                    ApiChatContent::Parts(vec![ApiChatContentPart::ImageUrl {
                        image_url: ApiImageUrl {
                            url: "data:image/jpeg;base64,PRIVATEPAYLOAD".to_string(),
                        },
                    }]),
                    Vec::new(),
                    tx,
                )
                .await;

            let message = match rx.recv().await {
                Some(SseEvent::Error { message }) => message,
                other => panic!("expected one error, got {other:?}"),
            };
            for forbidden in [
                "provider-secret-body",
                "never-print-api-key",
                "PRIVATEPAYLOAD",
                "/private/path",
            ] {
                assert!(!message.contains(forbidden));
            }
            assert!(message.contains("chat_model_http"));
            assert!(rx.recv().await.is_none());
            assert_eq!(state.requests.lock().expect("request lock").len(), 1);
            server.abort();
        }
    }

    #[tokio::test]
    async fn http_errors_are_bounded_retryable_and_secret_safe() {
        let (base_url, state, server) = spawn_mock(MockMode::EchoSecretError).await;
        let client = test_client(&base_url, "never-print-this-token", Duration::from_secs(1));
        let image = minimal_png();
        let encoded_image = base64::engine::general_purpose::STANDARD.encode(&image);

        let error = client
            .embed_image_passage(&image)
            .await
            .expect_err("mock returns an error");
        assert_eq!(error.status(), Some(StatusCode::TOO_MANY_REQUESTS));
        assert!(error.is_retryable());
        assert!(!error.is_timeout());
        assert_eq!(
            error
                .response_snippet()
                .expect("HTTP snippet")
                .chars()
                .count(),
            500
        );
        let rendered = error.to_string().to_ascii_lowercase();
        assert!(!rendered.contains("never-print-this-token"));
        assert!(!rendered.contains("authorization"));
        assert!(!rendered.contains("data:image"));
        assert!(!rendered.contains(&encoded_image.to_ascii_lowercase()));
        assert_eq!(client.embedding_attempt_count(), 3);
        assert!(state.requests.lock().expect("request lock").len() <= 3);
        server.abort();
    }

    #[tokio::test]
    async fn retryable_http_statuses_retry_once_then_succeed() {
        for status in [
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::INTERNAL_SERVER_ERROR,
        ] {
            let (base_url, state, server) = spawn_mock(MockMode::RetryOnce(status)).await;
            let client = test_client(&base_url, "test-api-key", Duration::from_secs(1));

            let embedding = client
                .embed_passage("retry passage")
                .await
                .unwrap_or_else(|error| panic!("second attempt after {status} failed: {error}"));
            assert_eq!(embedding.len(), EMBEDDING_DIMENSIONS);
            assert_eq!(client.embedding_attempt_count(), 2);
            assert_eq!(state.requests.lock().expect("request lock").len(), 2);
            server.abort();
        }
    }

    #[tokio::test]
    async fn rerank_retries_429_and_5xx_then_succeeds() {
        for status in [
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
        ] {
            let (base_url, state, server) = spawn_mock(MockMode::RetryOnce(status)).await;
            let client = test_client(&base_url, "test-api-key", Duration::from_secs(1));
            let response = client
                .rerank(
                    "query",
                    &[RerankPassage::text("first"), RerankPassage::text("second")],
                )
                .await
                .unwrap_or_else(|error| panic!("rerank retry after {status} failed: {error}"));
            assert_eq!(response.rankings.len(), 2);
            assert_eq!(client.rerank_attempt_count(), 2);
            assert_eq!(state.requests.lock().expect("request lock").len(), 2);
            server.abort();
        }
    }

    #[tokio::test]
    async fn rerank_timeout_is_typed_retried_and_safe_for_stage_fallback() {
        let (base_url, state, server) = spawn_mock(MockMode::Slow).await;
        let client = test_client(&base_url, "test-api-key", Duration::from_millis(10));
        let error = client
            .rerank("query", &[RerankPassage::text("passage")])
            .await
            .expect_err("mock rerank should time out");
        assert!(error.is_timeout());
        assert!(error.is_retryable());
        assert_eq!(client.rerank_attempt_count(), 3);
        assert!(state.requests.lock().expect("request lock").len() <= 3);
        server.abort();
    }

    #[tokio::test]
    async fn rerank_rejects_malformed_count_duplicate_and_out_of_range_without_retry() {
        for mode in [
            MockMode::MalformedRerank,
            MockMode::MissingRerank,
            MockMode::DuplicateRerank,
            MockMode::OutOfRangeRerank,
        ] {
            let (base_url, state, server) = spawn_mock(mode).await;
            let client = test_client(&base_url, "test-api-key", Duration::from_secs(1));
            let error = client
                .rerank(
                    "query",
                    &[RerankPassage::text("first"), RerankPassage::text("second")],
                )
                .await
                .expect_err("invalid rerank response must fail");
            assert!(matches!(&error, CloudModelError::InvalidResponse { .. }));
            assert!(!error.is_retryable());
            assert_eq!(client.rerank_attempt_count(), 1);
            assert_eq!(state.requests.lock().expect("request lock").len(), 1);
            server.abort();
        }
    }

    #[test]
    fn rerank_validation_rejects_non_finite_scores() {
        for logit in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = validate_rerank_response(
                RerankResponse {
                    rankings: vec![RerankResult { index: 0, logit }],
                },
                1,
            )
            .expect_err("non-finite rerank logit must fail");
            assert!(matches!(&error, CloudModelError::InvalidResponse { .. }));
            assert!(!error.is_retryable());
        }
    }

    #[tokio::test]
    async fn permanent_client_error_is_not_retried() {
        let (base_url, state, server) = spawn_mock(MockMode::PermanentBadRequest).await;
        let client = test_client(&base_url, "test-api-key", Duration::from_secs(1));

        let error = client
            .embed_passage("invalid passage")
            .await
            .expect_err("400 must fail immediately");
        assert_eq!(error.status(), Some(StatusCode::BAD_REQUEST));
        assert!(!error.is_retryable());
        assert_eq!(client.embedding_attempt_count(), 1);
        assert_eq!(state.requests.lock().expect("request lock").len(), 1);
        server.abort();
    }

    #[tokio::test]
    async fn request_timeout_is_typed_and_retryable() {
        let (base_url, state, server) = spawn_mock(MockMode::Slow).await;
        let client = test_client(&base_url, "test-api-key", Duration::from_millis(10));

        let error = client
            .embed_query("query")
            .await
            .expect_err("mock should time out");
        assert!(error.is_timeout());
        assert!(error.is_retryable());
        assert_eq!(error.status(), None);
        assert_eq!(client.embedding_attempt_count(), 3);
        assert!(state.requests.lock().expect("request lock").len() <= 3);
        server.abort();
    }

    #[tokio::test]
    async fn wrong_embedding_dimensions_are_rejected_without_retry() {
        let (base_url, state, server) = spawn_mock(MockMode::WrongDimensions).await;
        let client = test_client(&base_url, "test-api-key", Duration::from_secs(1));

        let error = client
            .embed_query("query")
            .await
            .expect_err("2047-dimensional response must fail");
        assert!(error.to_string().contains("expected 2048"));
        assert!(!error.is_retryable());
        assert_eq!(client.embedding_attempt_count(), 1);
        assert_eq!(state.requests.lock().expect("request lock").len(), 1);
        server.abort();
    }

    #[test]
    fn embedding_validation_rejects_nan_and_infinity() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut embedding = vec![0.0; EMBEDDING_DIMENSIONS];
            embedding[17] = invalid;
            let error = validate_embedding(embedding).expect_err("non-finite value must fail");
            assert!(error.to_string().contains("index 17"));
            assert!(!error.is_retryable());
        }
    }

    #[test]
    fn image_payload_rejects_unsupported_mime_and_oversize_data() {
        let error = image_data_url(b"GIF89a").expect_err("GIF must not be sent");
        assert!(error.to_string().contains("PNG or JPEG"));

        let oversized = vec![0_u8; MAX_EMBED_IMAGE_BYTES + 1];
        let error = image_data_url(&oversized).expect_err("oversize image must not be encoded");
        assert!(error.to_string().contains("maximum"));
    }

    #[test]
    fn only_contract_http_statuses_are_retryable() {
        for status in [
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
        ] {
            assert!(retryable_status(status), "{status} should be retryable");
        }
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::CONFLICT,
            StatusCode::TOO_EARLY,
            StatusCode::NOT_FOUND,
        ] {
            assert!(!retryable_status(status), "{status} must fail immediately");
        }
    }

    #[tokio::test]
    #[ignore = "requires explicit NVIDIA cloud credentials and incurs four embedding calls"]
    async fn real_nemotron_embed_vl_smoke_covers_all_input_modalities() -> anyhow::Result<()> {
        dotenvy::dotenv().ok();
        let client = CloudModelClient::new(CloudModelConfig::from_env()?)?;
        let image = image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            16,
            16,
            image::Rgba([32, 64, 96, 255]),
        ));
        let mut cursor = std::io::Cursor::new(Vec::new());
        image.write_to(&mut cursor, image::ImageFormat::Png)?;
        let image = cursor.into_inner();

        let embeddings = [
            client.embed_query("What is this document about?").await?,
            client
                .embed_passage("A short text document passage.")
                .await?,
            client.embed_image_passage(&image).await?,
            client
                .embed_image_text_passage("Text extracted from the same page.", &image)
                .await?,
        ];
        assert!(embeddings.iter().all(|embedding| {
            embedding.len() == EMBEDDING_DIMENSIONS
                && embedding.iter().all(|value| value.is_finite())
        }));
        Ok(())
    }

    #[test]
    fn think_filter_handles_tags_split_across_deltas() {
        let mut filter = ThinkFilter::new();
        let mut output = String::new();
        output.push_str(&filter.feed("visible <thi"));
        output.push_str(&filter.feed("nk>secret reasoning</thi"));
        output.push_str(&filter.feed("nk>after"));
        if let Some(remainder) = filter.flush() {
            output.push_str(&remainder);
        }
        assert_eq!(output, "visible after");
    }
}
