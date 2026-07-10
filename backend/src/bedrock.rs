use anyhow::{anyhow, Result};
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::error::DisplayErrorContext;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ContentBlockDelta, ConversationRole, ConverseStreamOutput, DocumentBlock,
    DocumentFormat, DocumentSource, ImageBlock, ImageFormat, ImageSource, InferenceConfiguration,
    Message, SystemContentBlock,
};
use aws_sdk_bedrockruntime::Client as BedrockClient;
use tokio::sync::mpsc::Sender;

use crate::llm::{ChatMessage, SseEvent};

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

/// Transcribes an uploaded file into Markdown by sending it to the multimodal model as a
/// Converse document/image block — no local conversion tooling involved.
pub async fn transcribe_file(
    client: &BedrockClient,
    model_id: &str,
    ext: &str,
    bytes: Vec<u8>,
) -> Result<String> {
    let content = match ext {
        "png" | "jpg" | "jpeg" | "gif" | "webp" => {
            let format = match ext {
                "png" => ImageFormat::Png,
                "jpg" | "jpeg" => ImageFormat::Jpeg,
                "gif" => ImageFormat::Gif,
                _ => ImageFormat::Webp,
            };
            ContentBlock::Image(
                ImageBlock::builder()
                    .format(format)
                    .source(ImageSource::Bytes(Blob::new(bytes)))
                    .build()?,
            )
        }
        _ => {
            let format = match ext {
                "pdf" => DocumentFormat::Pdf,
                "doc" => DocumentFormat::Doc,
                "docx" => DocumentFormat::Docx,
                "xls" => DocumentFormat::Xls,
                "xlsx" => DocumentFormat::Xlsx,
                "csv" => DocumentFormat::Csv,
                "html" | "htm" => DocumentFormat::Html,
                _ => {
                    return Err(anyhow!(
                        "不支援的檔案格式 .{ext}(支援 PDF、Word、Excel、CSV、HTML、PNG/JPG/GIF/WebP、Markdown、TXT;PPT 請先匯出為 PDF)"
                    ));
                }
            };
            // Bedrock restricts document names to a narrow character set; use a fixed safe name.
            ContentBlock::Document(
                DocumentBlock::builder()
                    .format(format)
                    .name("document")
                    .source(DocumentSource::Bytes(Blob::new(bytes)))
                    .build()?,
            )
        }
    };

    let message = Message::builder()
        .role(ConversationRole::User)
        .content(content)
        .content(ContentBlock::Text(crate::llm::TRANSCRIBE_PROMPT.to_string()))
        .build()?;

    let output = client
        .converse()
        .model_id(model_id)
        .messages(message)
        .inference_config(
            InferenceConfiguration::builder()
                .max_tokens(8000)
                .temperature(0.1)
                .build(),
        )
        .send()
        .await
        .map_err(|e| anyhow!("converse (transcribe) failed: {}", DisplayErrorContext(&e)))?;

    let text = output
        .output
        .and_then(|o| o.as_message().ok().cloned())
        .and_then(|m| {
            m.content
                .into_iter()
                .find_map(|c| c.as_text().ok().cloned())
        })
        .ok_or_else(|| anyhow!("no text content in transcribe response"))?;

    Ok(text)
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
