use base64::Engine;
use futures::StreamExt;
use reqwest::{Response, StatusCode, Url};
use serde::{Deserialize, Serialize};
use std::fmt;
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

const DEFAULT_BASE_URL: &str = "https://integrate.api.nvidia.com/v1";
const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_EMBED_REQUEST_BYTES: usize = 36 * 1024 * 1024;
const DEFAULT_EMBED_MAX_ATTEMPTS: usize = 3;
const DEFAULT_EMBED_RETRY_BASE_DELAY: Duration = Duration::from_millis(250);
const DEFAULT_EMBED_RETRY_MAX_DELAY: Duration = Duration::from_secs(2);

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
        sources: Vec<crate::search::SourceRef>,
    },
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "error")]
    Error { message: String },
}

#[derive(Clone)]
pub struct CloudModelConfig {
    api_key: String,
    chat_endpoint: Url,
    embed_endpoint: Url,
    rerank_endpoint: Url,
    chat_model: String,
    embed_model: String,
    rerank_model: String,
    request_timeout: Duration,
}

impl fmt::Debug for CloudModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CloudModelConfig")
            .field("api_key", &"[REDACTED]")
            .field("chat_endpoint", &self.chat_endpoint)
            .field("embed_endpoint", &self.embed_endpoint)
            .field("rerank_endpoint", &self.rerank_endpoint)
            .field("chat_model", &self.chat_model)
            .field("embed_model", &self.embed_model)
            .field("rerank_model", &self.rerank_model)
            .field("request_timeout", &self.request_timeout)
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

    fn from_lookup(
        mut get: impl FnMut(&str) -> Option<String>,
    ) -> Result<Self, CloudModelConfigError> {
        let api_key = required_value(&mut get, "NVIDIA_API_KEY")?;
        let chat_model = required_value(&mut get, "GEMMA_4_MODEL_ID")?;
        if !is_gemma_4_model_id(&chat_model) {
            return Err(CloudModelConfigError(
                "GEMMA_4_MODEL_ID must identify a cloud-hosted Gemma 4 model".to_string(),
            ));
        }

        let embed_model = optional_value(&mut get, "NEMOTRON_EMBED_MODEL_ID")
            .unwrap_or_else(|| DEFAULT_EMBED_MODEL_ID.to_string());
        validate_locked_model("NEMOTRON_EMBED_MODEL_ID", &embed_model, EMBED_MODEL_FAMILY)?;

        let rerank_model = optional_value(&mut get, "NEMOTRON_RERANK_MODEL_ID")
            .unwrap_or_else(|| DEFAULT_RERANK_MODEL_ID.to_string());
        validate_locked_model(
            "NEMOTRON_RERANK_MODEL_ID",
            &rerank_model,
            RERANK_MODEL_FAMILY,
        )?;

        let base_url = optional_value(&mut get, "CLOUD_MODEL_BASE_URL")
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let base_url = base_url.trim_end_matches('/');

        let chat_endpoint = optional_value(&mut get, "CLOUD_CHAT_URL")
            .unwrap_or_else(|| format!("{base_url}/chat/completions"));
        let embed_endpoint = optional_value(&mut get, "CLOUD_EMBED_URL")
            .unwrap_or_else(|| format!("{base_url}/embeddings"));
        let rerank_endpoint = optional_value(&mut get, "CLOUD_RERANK_URL")
            .unwrap_or_else(|| format!("{base_url}/ranking"));

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

        Ok(Self {
            api_key,
            chat_endpoint: parse_endpoint("CLOUD_CHAT_URL", &chat_endpoint)?,
            embed_endpoint: parse_endpoint("CLOUD_EMBED_URL", &embed_endpoint)?,
            rerank_endpoint: parse_endpoint("CLOUD_RERANK_URL", &rerank_endpoint)?,
            chat_model,
            embed_model,
            rerank_model,
            request_timeout: Duration::from_secs(timeout_secs),
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

fn validate_locked_model(
    setting: &'static str,
    model: &str,
    expected_family: &'static str,
) -> Result<(), CloudModelConfigError> {
    if model.rsplit('/').next() == Some(expected_family) {
        Ok(())
    } else {
        Err(CloudModelConfigError(format!(
            "{setting} must remain locked to {expected_family}"
        )))
    }
}

fn is_gemma_4_model_id(model: &str) -> bool {
    let normalized = model.to_ascii_lowercase();
    normalized.match_indices("gemma").any(|(index, _)| {
        let suffix = normalized[index + "gemma".len()..]
            .trim_start_matches(|character: char| !character.is_ascii_alphanumeric());
        suffix.strip_prefix('4').is_some_and(|rest| {
            rest.chars()
                .next()
                .is_none_or(|character| !character.is_ascii_digit())
        })
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

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ApiChatMessage>,
    pub stream: bool,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
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
    pub modality: String,
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

#[derive(Debug, Clone, Serialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: RerankQuery,
    pub passages: Vec<RerankPassage>,
    pub truncate: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankQuery {
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
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
}

#[derive(Debug, Deserialize)]
pub struct RerankResponse {
    pub rankings: Vec<RerankResult>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RerankResult {
    pub index: usize,
    pub logit: f32,
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

#[derive(Clone)]
pub struct CloudModelClient {
    http: reqwest::Client,
    api_key: String,
    chat_endpoint: Url,
    embed_endpoint: Url,
    rerank_endpoint: Url,
    chat_model: String,
    embed_model: String,
    rerank_model: String,
    request_timeout: Duration,
    embedding_retry: EmbeddingRetryConfig,
    #[cfg(test)]
    embedding_attempts: std::sync::Arc<std::sync::atomic::AtomicUsize>,
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
            api_key: config.api_key,
            chat_endpoint: config.chat_endpoint,
            embed_endpoint: config.embed_endpoint,
            rerank_endpoint: config.rerank_endpoint,
            chat_model: config.chat_model,
            embed_model: config.embed_model,
            rerank_model: config.rerank_model,
            request_timeout: config.request_timeout,
            embedding_retry: EmbeddingRetryConfig::default(),
            #[cfg(test)]
            embedding_attempts: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        let endpoint = |path: &str| {
            Url::parse(&format!("http://127.0.0.1/{path}"))
                .expect("static test endpoint must be valid")
        };
        let config = CloudModelConfig {
            api_key: "test-api-key".to_string(),
            chat_endpoint: endpoint("chat/completions"),
            embed_endpoint: endpoint("embeddings"),
            rerank_endpoint: endpoint("ranking"),
            chat_model: "test/gemma-4".to_string(),
            embed_model: DEFAULT_EMBED_MODEL_ID.to_string(),
            rerank_model: DEFAULT_RERANK_MODEL_ID.to_string(),
            request_timeout: Duration::from_secs(1),
        };
        let mut client = Self::new(config).expect("static cloud model test config must be valid");
        client.embedding_retry = EmbeddingRetryConfig {
            max_attempts: DEFAULT_EMBED_MAX_ATTEMPTS,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            jitter: false,
        };
        client
    }

    #[cfg(test)]
    pub(crate) fn for_test_embed_endpoint(endpoint: &str, timeout: Duration) -> Self {
        let mut client = Self::for_test();
        client.embed_endpoint = Url::parse(endpoint).expect("test embed endpoint must be valid");
        client.request_timeout = timeout;
        client.http = reqwest::Client::builder()
            .connect_timeout(timeout.min(Duration::from_millis(100)))
            .timeout(timeout)
            .build()
            .expect("test HTTP client must be valid");
        client
    }

    #[cfg(test)]
    fn embedding_attempt_count(&self) -> usize {
        self.embedding_attempts
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    pub async fn stream_chat(&self, messages: Vec<ChatMessage>, tx: Sender<SseEvent>) {
        self.stream_chat_with_system(SYSTEM_PROMPT.to_string(), messages, tx)
            .await;
    }

    pub async fn stream_chat_with_system(
        &self,
        system_prompt: String,
        messages: Vec<ChatMessage>,
        tx: Sender<SseEvent>,
    ) {
        if let Err(error) = self
            .try_stream_chat_with_system(system_prompt, messages, tx.clone())
            .await
        {
            let _ = tx
                .send(SseEvent::Error {
                    message: error.to_string(),
                })
                .await;
        }
    }

    pub async fn try_stream_chat_with_system(
        &self,
        system_prompt: String,
        messages: Vec<ChatMessage>,
        tx: Sender<SseEvent>,
    ) -> CloudModelResult<()> {
        let request = self.chat_request(system_prompt, messages, true, 4096, None);
        let response = self
            .send_json(ModelOperation::Chat, &self.chat_endpoint, &request)
            .await?;

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut filter = ThinkFilter::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| self.request_error(ModelOperation::Chat, error))?;
            buffer.extend_from_slice(&chunk);

            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=newline).collect::<Vec<_>>();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim_end_matches(['\r', '\n']);
                let Some(payload) = line
                    .strip_prefix("data:")
                    .map(str::trim_start)
                    .filter(|payload| !payload.is_empty())
                else {
                    continue;
                };

                if payload == "[DONE]" {
                    if !send_filter_remainder(&tx, &mut filter).await {
                        return Ok(());
                    }
                    let _ = tx.send(SseEvent::Done).await;
                    return Ok(());
                }

                let Ok(chunk) = serde_json::from_str::<ChatStreamChunk>(payload) else {
                    continue;
                };
                let Some(content) = chunk
                    .choices
                    .first()
                    .and_then(|choice| choice.delta.content.as_deref())
                else {
                    continue;
                };
                let visible = filter.feed(content);
                if !visible.is_empty()
                    && tx
                        .send(SseEvent::TextDelta { text: visible })
                        .await
                        .is_err()
                {
                    return Ok(());
                }
            }
        }

        if !send_filter_remainder(&tx, &mut filter).await {
            return Ok(());
        }
        let _ = tx.send(SseEvent::Done).await;
        Ok(())
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
        let response = self
            .send_json(ModelOperation::Chat, &self.chat_endpoint, &request)
            .await?;
        self.chat_content(response).await
    }

    pub async fn embed_query(&self, text: &str) -> CloudModelResult<Vec<f32>> {
        self.embed_text(text, "query").await
    }

    pub async fn embed_passage(&self, text: &str) -> CloudModelResult<Vec<f32>> {
        self.embed_text(text, "passage").await
    }

    pub async fn embed_image_passage(&self, image: &[u8]) -> CloudModelResult<Vec<f32>> {
        let data_url = image_data_url(image)?;
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
        let data_url = image_data_url(image)?;
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
            modality: modality.to_string(),
            truncate: "END".to_string(),
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
        if passages
            .iter()
            .any(|passage| passage.text.is_none() && passage.image.is_none())
        {
            return Err(CloudModelError::InvalidRequest {
                operation: ModelOperation::Rerank,
                message: "each passage requires text, image, or both".to_string(),
            });
        }

        let request = RerankRequest {
            model: self.rerank_model.clone(),
            query: RerankQuery {
                text: query.to_string(),
            },
            passages: passages.to_vec(),
            truncate: "END".to_string(),
        };
        let response = self
            .send_json(ModelOperation::Rerank, &self.rerank_endpoint, &request)
            .await?;
        let response: RerankResponse = self.decode_json(ModelOperation::Rerank, response).await?;
        if response
            .rankings
            .iter()
            .any(|ranking| ranking.index >= passages.len())
        {
            return Err(CloudModelError::InvalidResponse {
                operation: ModelOperation::Rerank,
                message: "ranking index is outside the submitted passage list".to_string(),
            });
        }
        Ok(response)
    }

    fn chat_request(
        &self,
        system_prompt: String,
        messages: Vec<ChatMessage>,
        stream: bool,
        max_tokens: u32,
        temperature: Option<f32>,
    ) -> ChatCompletionRequest {
        let mut api_messages = vec![ApiChatMessage {
            role: "system".to_string(),
            content: ApiChatContent::Text(system_prompt),
        }];
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
            #[cfg(test)]
            self.embedding_attempts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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

    fn embedding_retry_delay(&self, failed_attempt: usize) -> Duration {
        let exponent = u32::try_from(failed_attempt.saturating_sub(1))
            .unwrap_or(u32::MAX)
            .min(31);
        let multiplier = 1_u32 << exponent;
        let exponential = self
            .embedding_retry
            .base_delay
            .checked_mul(multiplier)
            .unwrap_or(self.embedding_retry.max_delay)
            .min(self.embedding_retry.max_delay);
        if !self.embedding_retry.jitter || exponential.is_zero() {
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

    async fn send_json<T: Serialize + ?Sized>(
        &self,
        operation: ModelOperation,
        endpoint: &Url,
        request: &T,
    ) -> CloudModelResult<Response> {
        let response = self
            .http
            .post(endpoint.clone())
            .bearer_auth(&self.api_key)
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
        let without_key = if self.api_key.is_empty() {
            text.to_string()
        } else {
            text.replace(&self.api_key, "[REDACTED]")
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

fn image_data_url(image: &[u8]) -> CloudModelResult<String> {
    if image.is_empty() {
        return Err(CloudModelError::InvalidRequest {
            operation: ModelOperation::Embed,
            message: "image input must not be empty".to_string(),
        });
    }
    if image.len() > MAX_EMBED_IMAGE_BYTES {
        return Err(CloudModelError::InvalidRequest {
            operation: ModelOperation::Embed,
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
            operation: ModelOperation::Embed,
            message: "image input must be a PNG or JPEG payload".to_string(),
        });
    };
    let encoded = base64::engine::general_purpose::STANDARD.encode(image);
    Ok(format!("data:{mime};base64,{encoded}"))
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
    use axum::body::to_bytes;
    use axum::extract::{Request, State};
    use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
    use axum::response::{IntoResponse, Response as AxumResponse};
    use axum::routing::post;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Copy)]
    enum MockMode {
        Success,
        EchoSecretError,
        Slow,
        RetryOnce(StatusCode),
        PermanentBadRequest,
        WrongDimensions,
    }

    #[derive(Debug)]
    struct RecordedRequest {
        path: String,
        authorization: Option<String>,
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
        let bytes = to_bytes(body, usize::MAX).await.expect("read mock request");
        let body: serde_json::Value =
            serde_json::from_slice(&bytes).expect("mock request contains JSON");
        let request_number = {
            let mut requests = state.requests.lock().expect("request lock");
            requests.push(RecordedRequest {
                path: path.clone(),
                authorization: authorization.clone(),
                body: body.clone(),
            });
            requests.len()
        };

        match state.mode {
            MockMode::EchoSecretError => {
                let response_body = format!(
                    "Authorization: {} echoed-input={} {}",
                    authorization.unwrap_or_default(),
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
            MockMode::Success | MockMode::RetryOnce(_) | MockMode::WrongDimensions => {}
        }

        match path.as_str() {
            "/chat/completions" | "/capabilities/chat"
                if body.get("stream") == Some(&serde_json::Value::Bool(true)) =>
            {
                let sse = concat!(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"<think>hidden</think>雲\"}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"端回答\"}}]}\n\n",
                    "data: [DONE]\n\n"
                );
                (StatusCode::OK, [(CONTENT_TYPE, "text/event-stream")], sse).into_response()
            }
            "/chat/completions" | "/capabilities/chat" => (
                StatusCode::OK,
                [(CONTENT_TYPE, "application/json")],
                r#"{"choices":[{"message":{"content":"ok"}}]}"#,
            )
                .into_response(),
            "/embeddings" | "/capabilities/embed" => {
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
            "/ranking" | "/capabilities/rerank" => (
                StatusCode::OK,
                [(CONTENT_TYPE, "application/json")],
                r#"{"rankings":[{"index":1,"logit":0.9},{"index":0,"logit":0.1}]}"#,
            )
                .into_response(),
            _ => StatusCode::NOT_FOUND.into_response(),
        }
    }

    async fn spawn_mock(mode: MockMode) -> (String, MockState, tokio::task::JoinHandle<()>) {
        let state = MockState {
            mode,
            requests: Arc::new(Mutex::new(Vec::new())),
        };
        let app = Router::new()
            .route("/chat/completions", post(mock_cloud))
            .route("/embeddings", post(mock_cloud))
            .route("/ranking", post(mock_cloud))
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

    fn test_config(base_url: &str, api_key: &str) -> CloudModelConfig {
        config_from(&[
            ("NVIDIA_API_KEY", api_key),
            ("GEMMA_4_MODEL_ID", "hosted/google-gemma-4-test"),
            ("CLOUD_MODEL_BASE_URL", base_url),
        ])
        .expect("valid test config")
    }

    fn test_client(base_url: &str, api_key: &str, timeout: Duration) -> CloudModelClient {
        let mut config = test_config(base_url, api_key);
        config.request_timeout = timeout;
        let mut client = CloudModelClient::new(config).expect("cloud client");
        client.embedding_retry = EmbeddingRetryConfig {
            max_attempts: DEFAULT_EMBED_MAX_ATTEMPTS,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            jitter: false,
        };
        client
    }

    fn minimal_png() -> Vec<u8> {
        b"\x89PNG\r\n\x1a\nmock-png-payload".to_vec()
    }

    #[test]
    fn config_requires_api_key_and_explicit_gemma_4_model() {
        let missing_key = config_from(&[]).expect_err("API key must be required");
        assert!(missing_key.to_string().contains("NVIDIA_API_KEY"));

        let missing_model =
            config_from(&[("NVIDIA_API_KEY", "secret")]).expect_err("model must be required");
        assert!(missing_model.to_string().contains("GEMMA_4_MODEL_ID"));

        let wrong_family = config_from(&[
            ("NVIDIA_API_KEY", "secret"),
            ("GEMMA_4_MODEL_ID", "some-other-chat-model"),
        ])
        .expect_err("chat family must be locked");
        assert!(wrong_family.to_string().contains("Gemma 4"));

        let misleading_version = config_from(&[
            ("NVIDIA_API_KEY", "secret"),
            ("GEMMA_4_MODEL_ID", "hosted/gemma-14b"),
        ])
        .expect_err("Gemma 14 must not be mistaken for Gemma 4");
        assert!(misleading_version.to_string().contains("Gemma 4"));
    }

    #[test]
    fn config_locks_embed_and_rerank_families() {
        let config = config_from(&[
            ("NVIDIA_API_KEY", "secret"),
            ("GEMMA_4_MODEL_ID", "hosted/gemma-4-production"),
        ])
        .expect("default model config");
        assert_eq!(config.embed_model(), DEFAULT_EMBED_MODEL_ID);
        assert_eq!(config.rerank_model(), DEFAULT_RERANK_MODEL_ID);
        let debug = format!("{config:?}").to_ascii_lowercase();
        assert!(!debug.contains("secret"));
        assert!(!debug.contains("authorization"));

        let replacement = config_from(&[
            ("NVIDIA_API_KEY", "secret"),
            ("GEMMA_4_MODEL_ID", "hosted/gemma-4-production"),
            ("NEMOTRON_EMBED_MODEL_ID", "vendor/arbitrary-embedder"),
        ])
        .expect_err("alternative embed family must be rejected");
        assert!(replacement.to_string().contains(EMBED_MODEL_FAMILY));

        let replacement = config_from(&[
            ("NVIDIA_API_KEY", "secret"),
            ("GEMMA_4_MODEL_ID", "hosted/gemma-4-production"),
            ("NEMOTRON_RERANK_MODEL_ID", "vendor/arbitrary-reranker"),
        ])
        .expect_err("alternative rerank family must be rejected");
        assert!(replacement.to_string().contains(RERANK_MODEL_FAMILY));
    }

    #[tokio::test]
    async fn mock_server_observes_capability_specific_models_and_sse_contract() {
        let (base_url, state, server) = spawn_mock(MockMode::Success).await;
        let chat_url = format!("{base_url}/capabilities/chat");
        let embed_url = format!("{base_url}/capabilities/embed");
        let rerank_url = format!("{base_url}/capabilities/rerank");
        let config = config_from(&[
            ("NVIDIA_API_KEY", "test-api-key"),
            ("GEMMA_4_MODEL_ID", "hosted/google-gemma-4-test"),
            ("CLOUD_MODEL_BASE_URL", "https://unused.example/v1"),
            ("CLOUD_CHAT_URL", &chat_url),
            ("CLOUD_EMBED_URL", &embed_url),
            ("CLOUD_RERANK_URL", &rerank_url),
        ])
        .expect("capability-specific config");
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
                &[RerankPassage::text("first"), RerankPassage::text("second")],
            )
            .await
            .expect("ranking");
        assert_eq!(ranking.rankings[0].index, 1);

        let requests = state.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 6);
        assert!(
            requests
                .iter()
                .all(|request| request.authorization.as_deref() == Some("Bearer test-api-key"))
        );

        let chat = requests
            .iter()
            .find(|request| request.path == "/capabilities/chat")
            .expect("chat request");
        assert_eq!(chat.body["model"], "hosted/google-gemma-4-test");
        assert_eq!(chat.body["stream"], true);

        let embeds = requests
            .iter()
            .filter(|request| request.path == "/capabilities/embed")
            .collect::<Vec<_>>();
        assert_eq!(embeds.len(), 4);
        assert!(embeds.iter().all(|request| {
            request.body["model"] == DEFAULT_EMBED_MODEL_ID
                && request.body["embedding_type"] == "float"
                && request.body["encoding_format"] == "float"
                && request.body["truncate"] == "END"
                && request.body.get("dimensions").is_none()
        }));
        assert_eq!(embeds[0].body["input_type"], "query");
        assert_eq!(embeds[0].body["modality"], "text");
        assert_eq!(embeds[0].body["input"][0], "query");
        assert_eq!(embeds[1].body["input_type"], "passage");
        assert_eq!(embeds[1].body["modality"], "text");
        assert_eq!(embeds[1].body["input"][0], "passage");
        assert_eq!(embeds[2].body["input_type"], "passage");
        assert_eq!(embeds[2].body["modality"], "image");
        let image_input = embeds[2].body["input"][0]
            .as_str()
            .expect("image input string");
        assert!(image_input.starts_with("data:image/png;base64,"));
        assert!(!image_input.contains("/tmp/"));
        assert!(!image_input.contains("pages/"));
        assert_eq!(embeds[3].body["input_type"], "passage");
        assert_eq!(embeds[3].body["modality"], "text_image");
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
        server.abort();
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
