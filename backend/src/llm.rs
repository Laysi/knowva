use aws_sdk_bedrockruntime::Client as BedrockClient;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;

use crate::bedrock;
use crate::nvidia::NvidiaClient;

pub const SYSTEM_PROMPT: &str = "你是企業內部的 AI 知識管理助理,依據知識庫內容協助回答問題與整理資訊。一律使用繁體中文回答。";

/// Prompt used to transcribe uploaded files (PDF/Office/images) into Markdown.
pub const TRANSCRIBE_PROMPT: &str = "請將這份文件的內容完整轉錄為 Markdown:保留原有標題階層與段落順序;表格轉成 Markdown 表格;圖表與圖片改以文字簡述其內容與數據;保留所有數字、單位與代號。不要省略內容,不要輸出任何前言、評論或結尾說明,只輸出轉錄後的 Markdown。";

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

/// Provider-neutral LLM handle. Constructed once at startup based on `MODEL_PROVIDER`.
#[derive(Clone)]
pub enum Llm {
    Bedrock {
        client: BedrockClient,
        chat_model: String,
        embed_model: String,
    },
    Nvidia(NvidiaClient),
}

impl Llm {
    /// Drives a streaming chat completion using the default system prompt.
    pub async fn stream_chat(&self, messages: Vec<ChatMessage>, tx: Sender<SseEvent>) {
        self.stream_chat_with_system(SYSTEM_PROMPT.to_string(), messages, tx)
            .await
    }

    /// Same as `stream_chat` but with a caller-supplied system prompt (used for RAG-augmented chat).
    pub async fn stream_chat_with_system(
        &self,
        system_prompt: String,
        messages: Vec<ChatMessage>,
        tx: Sender<SseEvent>,
    ) {
        match self {
            Llm::Bedrock {
                client,
                chat_model,
                ..
            } => {
                bedrock::stream_chat_with_system(
                    client.clone(),
                    chat_model.clone(),
                    system_prompt,
                    messages,
                    tx,
                )
                .await
            }
            Llm::Nvidia(nvidia) => {
                nvidia.stream_chat_with_system(system_prompt, messages, tx).await
            }
        }
    }

    /// One-shot (non-streaming) completion, used for summary/tags generation.
    pub async fn converse_once(
        &self,
        system_prompt: &str,
        user_message: &str,
    ) -> anyhow::Result<String> {
        match self {
            Llm::Bedrock {
                client, chat_model, ..
            } => bedrock::converse_once(client, chat_model, system_prompt, user_message).await,
            Llm::Nvidia(nvidia) => nvidia.chat_once(system_prompt, user_message).await,
        }
    }

    /// Transcribes an uploaded file (PDF/Office/image) into Markdown via the multimodal model.
    /// `ext` is the lowercased file extension without the dot.
    pub async fn transcribe_file(&self, ext: &str, bytes: Vec<u8>) -> anyhow::Result<String> {
        match self {
            Llm::Bedrock {
                client, chat_model, ..
            } => bedrock::transcribe_file(client, chat_model, ext, bytes).await,
            Llm::Nvidia(nvidia) => nvidia.transcribe_file(ext, bytes).await,
        }
    }

    /// Embeds a search query. Bedrock (Titan) has no query/passage distinction.
    pub async fn embed_query(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        match self {
            Llm::Bedrock {
                client, embed_model, ..
            } => bedrock::embed_text(client, embed_model, text).await,
            Llm::Nvidia(nvidia) => nvidia.embed_text(text, "query").await,
        }
    }

    /// Embeds a document chunk for storage. Bedrock (Titan) has no query/passage distinction.
    pub async fn embed_passage(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        match self {
            Llm::Bedrock {
                client, embed_model, ..
            } => bedrock::embed_text(client, embed_model, text).await,
            Llm::Nvidia(nvidia) => nvidia.embed_text(text, "passage").await,
        }
    }
}
