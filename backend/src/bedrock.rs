use anyhow::{anyhow, Result};
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ContentBlockDelta, ConversationRole, ConverseStreamOutput, Message,
    SystemContentBlock,
};
use aws_sdk_bedrockruntime::Client as BedrockClient;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;

pub const SYSTEM_PROMPT: &str = "你是企業內部的 AI 知識管理助理,依據知識庫內容協助回答問題與整理資訊。一律使用繁體中文回答。";

#[derive(Debug, Clone, Deserialize)]
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
    Sources { sources: Vec<crate::search::SourceRef> },
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "error")]
    Error { message: String },
}

fn to_bedrock_message(msg: &ChatMessage) -> Message {
    let role = if msg.role == "assistant" {
        ConversationRole::Assistant
    } else {
        ConversationRole::User
    };
    Message::builder()
        .role(role)
        .content(ContentBlock::Text(msg.content.clone()))
        .build()
        .expect("valid message")
}

/// Drives a ConverseStream call, sending SSE-shaped events into `tx`.
/// Never panics; all failures are surfaced as `SseEvent::Error`.
pub async fn stream_chat(
    client: BedrockClient,
    model_id: String,
    messages: Vec<ChatMessage>,
    tx: Sender<SseEvent>,
) {
    stream_chat_with_system(client, model_id, SYSTEM_PROMPT.to_string(), messages, tx).await
}

/// Same as `stream_chat` but with a caller-supplied system prompt (used for RAG-augmented chat).
pub async fn stream_chat_with_system(
    client: BedrockClient,
    model_id: String,
    system_prompt: String,
    messages: Vec<ChatMessage>,
    tx: Sender<SseEvent>,
) {
    let msgs: Vec<Message> = messages.iter().map(to_bedrock_message).collect();

    let stream_result = client
        .converse_stream()
        .model_id(&model_id)
        .system(SystemContentBlock::Text(system_prompt))
        .set_messages(Some(msgs))
        .send()
        .await;

    let mut output = match stream_result {
        Ok(output) => output,
        Err(err) => {
            let _ = tx
                .send(SseEvent::Error {
                    message: err.to_string(),
                })
                .await;
            return;
        }
    };

    loop {
        match output.stream.recv().await {
            Ok(Some(ConverseStreamOutput::ContentBlockDelta(delta_event))) => {
                if let Some(ContentBlockDelta::Text(text)) = delta_event.delta {
                    if tx.send(SseEvent::TextDelta { text }).await.is_err() {
                        return;
                    }
                }
            }
            Ok(Some(_)) => {
                // ignore other event kinds (message start/stop, metadata, etc.)
            }
            Ok(None) => {
                let _ = tx.send(SseEvent::Done).await;
                return;
            }
            Err(err) => {
                let _ = tx
                    .send(SseEvent::Error {
                        message: err.to_string(),
                    })
                    .await;
                return;
            }
        }
    }
}

#[allow(dead_code)]
pub type BedrockResult<T> = Result<T>;

/// Calls Titan Text Embeddings V2 via invoke_model, returns the 1024-dim embedding.
pub async fn embed_text(
    client: &BedrockClient,
    model_id: &str,
    text: &str,
) -> Result<Vec<f32>> {
    let body = serde_json::json!({
        "inputText": text,
        "dimensions": 1024,
        "normalize": true,
    });
    let body_bytes = serde_json::to_vec(&body)?;

    let output = client
        .invoke_model()
        .model_id(model_id)
        .content_type("application/json")
        .accept("application/json")
        .body(Blob::new(body_bytes))
        .send()
        .await
        .map_err(|e| anyhow!("invoke_model failed: {e}"))?;

    let resp: serde_json::Value = serde_json::from_slice(output.body.as_ref())?;
    let embedding = resp
        .get("embedding")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing embedding field in Titan response"))?;

    let vec: Vec<f32> = embedding
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();

    Ok(vec)
}

/// One-shot (non-streaming) Converse call, used for summary/tags generation.
pub async fn converse_once(
    client: &BedrockClient,
    model_id: &str,
    system_prompt: &str,
    user_message: &str,
) -> Result<String> {
    let message = Message::builder()
        .role(ConversationRole::User)
        .content(ContentBlock::Text(user_message.to_string()))
        .build()?;

    let output = client
        .converse()
        .model_id(model_id)
        .system(SystemContentBlock::Text(system_prompt.to_string()))
        .messages(message)
        .send()
        .await
        .map_err(|e| anyhow!("converse failed: {e}"))?;

    let text = output
        .output
        .and_then(|o| o.as_message().ok().cloned())
        .and_then(|m| {
            m.content
                .into_iter()
                .find_map(|c| c.as_text().ok().cloned())
        })
        .ok_or_else(|| anyhow!("no text content in converse response"))?;

    Ok(text)
}
