use anyhow::{anyhow, bail, Result};
use futures::StreamExt;
use std::time::Duration;
use tokio::sync::mpsc::Sender;

use crate::llm::{ChatMessage, SseEvent};

/// OpenAI-compatible REST client for NVIDIA Build (build.nvidia.com / NIM).
#[derive(Clone)]
pub struct NvidiaClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    chat_model: String,
    embed_model: String,
    // Matryoshka truncation to match the DB's vector(1024); None for natively-1024 models,
    // which reject the `dimensions` field outright.
    embed_dimensions: Option<u32>,
}

impl NvidiaClient {
    pub fn new(
        api_key: String,
        base_url: String,
        chat_model: String,
        embed_model: String,
        embed_dimensions: Option<u32>,
    ) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build reqwest client");
        Self {
            http,
            api_key,
            base_url,
            chat_model,
            embed_model,
            embed_dimensions,
        }
    }

    fn to_openai_role(role: &str) -> &str {
        if role == "assistant" {
            "assistant"
        } else {
            "user"
        }
    }

    /// Drives a streaming chat completion, sending SSE-shaped events into `tx`.
    /// Never panics; all failures are surfaced as `SseEvent::Error`.
    pub async fn stream_chat_with_system(
        &self,
        system_prompt: String,
        messages: Vec<ChatMessage>,
        tx: Sender<SseEvent>,
    ) {
        let mut body_messages = vec![serde_json::json!({
            "role": "system",
            "content": system_prompt,
        })];
        for m in &messages {
            body_messages.push(serde_json::json!({
                "role": Self::to_openai_role(&m.role),
                "content": m.content,
            }));
        }

        let body = serde_json::json!({
            "model": self.chat_model,
            "messages": body_messages,
            "stream": true,
            "max_tokens": 4096,
        });

        let resp = match self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let _ = tx
                    .send(SseEvent::Error {
                        message: e.to_string(),
                    })
                    .await;
                return;
            }
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            let snippet: String = text.chars().take(500).collect();
            let _ = tx
                .send(SseEvent::Error {
                    message: format!("NVIDIA chat completions returned {status}: {snippet}"),
                })
                .await;
            return;
        }

        let mut stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut filter = ThinkFilter::new();

        loop {
            let chunk = match stream.next().await {
                Some(Ok(bytes)) => bytes,
                Some(Err(e)) => {
                    let _ = tx
                        .send(SseEvent::Error {
                            message: e.to_string(),
                        })
                        .await;
                    return;
                }
                None => {
                    // Stream ended without an explicit [DONE]; still emit Done.
                    if let Some(visible) = filter.flush() {
                        if !visible.is_empty() && tx.send(SseEvent::TextDelta { text: visible }).await.is_err() {
                            return;
                        }
                    }
                    let _ = tx.send(SseEvent::Done).await;
                    return;
                }
            };

            buf.extend_from_slice(&chunk);

            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
                let mut line = String::from_utf8_lossy(&line_bytes).to_string();
                if line.ends_with('\n') {
                    line.pop();
                }
                if line.ends_with('\r') {
                    line.pop();
                }

                let payload = if let Some(rest) = line.strip_prefix("data: ") {
                    rest
                } else if let Some(rest) = line.strip_prefix("data:") {
                    rest
                } else {
                    continue;
                };

                if payload == "[DONE]" {
                    if let Some(visible) = filter.flush() {
                        if !visible.is_empty() && tx.send(SseEvent::TextDelta { text: visible }).await.is_err() {
                            return;
                        }
                    }
                    let _ = tx.send(SseEvent::Done).await;
                    return;
                }

                let value: serde_json::Value = match serde_json::from_str(payload) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let content = value
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str());

                if let Some(content) = content {
                    if content.is_empty() {
                        continue;
                    }
                    let visible = filter.feed(content);
                    if !visible.is_empty() && tx.send(SseEvent::TextDelta { text: visible }).await.is_err() {
                        return;
                    }
                }
            }
        }
    }

    /// One-shot (non-streaming) completion, used for summary/tags generation.
    pub async fn chat_once(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        let body = serde_json::json!({
            "model": self.chat_model,
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_message },
            ],
            "stream": false,
            "max_tokens": 2048,
            "temperature": 0.2,
        });

        let resp = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("NVIDIA chat completions request failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let snippet: String = text.chars().take(500).collect();
            bail!("NVIDIA chat completions returned {status}: {snippet}");
        }

        let value: serde_json::Value = resp.json().await?;
        let content = value
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| anyhow!("missing choices[0].message.content in NVIDIA response"))?;

        let think_re = regex::Regex::new(r"(?s)<think>.*?</think>").unwrap();
        let cleaned = think_re.replace_all(content, "");
        Ok(cleaned.trim().to_string())
    }

    /// Transcribes an uploaded image into Markdown via the (multimodal) chat model.
    /// Only images are supported on the NVIDIA path: OpenAI-compatible chat has no standard
    /// document input, so PDF/Office need the bedrock provider (or a future parser stage).
    pub async fn transcribe_file(&self, ext: &str, bytes: Vec<u8>) -> Result<String> {
        let mime = match ext {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "webp" => "image/webp",
            _ => bail!(
                "MODEL_PROVIDER=nvidia 僅支援圖片(PNG/JPG/GIF/WebP)與 Markdown/TXT 上傳;PDF/Office 請改用 bedrock provider,或先轉存為圖片"
            ),
        };

        use base64::Engine as _;
        let data_url = format!(
            "data:{mime};base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&bytes)
        );

        let body = serde_json::json!({
            "model": self.chat_model,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": crate::llm::TRANSCRIBE_PROMPT },
                    { "type": "image_url", "image_url": { "url": data_url } },
                ],
            }],
            "stream": false,
            "max_tokens": 4096,
            "temperature": 0.1,
        });

        let resp = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .timeout(Duration::from_secs(180))
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("NVIDIA transcribe request failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let snippet: String = text.chars().take(500).collect();
            bail!(
                "NVIDIA transcribe returned {status}: {snippet}(注意:NVIDIA_CHAT_MODEL 必須是支援影像輸入的 VLM)"
            );
        }

        let value: serde_json::Value = resp.json().await?;
        let content = value
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| anyhow!("missing choices[0].message.content in NVIDIA response"))?;

        let think_re = regex::Regex::new(r"(?s)<think>.*?</think>").unwrap();
        let cleaned = think_re.replace_all(content, "");
        Ok(cleaned.trim().to_string())
    }

    /// Calls the embeddings endpoint, returns a 1024-dim embedding.
    pub async fn embed_text(&self, text: &str, input_type: &str) -> Result<Vec<f32>> {
        let mut body = serde_json::json!({
            "model": self.embed_model,
            "input": [text],
            "encoding_format": "float",
            "input_type": input_type,
            "truncate": "END",
        });
        if let Some(dims) = self.embed_dimensions {
            body["dimensions"] = serde_json::json!(dims);
        }

        let resp = self
            .http
            .post(format!("{}/embeddings", self.base_url))
            .bearer_auth(&self.api_key)
            .timeout(Duration::from_secs(60))
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("NVIDIA embeddings request failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() {
            let text_body = resp.text().await.unwrap_or_default();
            let snippet: String = text_body.chars().take(500).collect();
            bail!("NVIDIA embeddings returned {status}: {snippet}");
        }

        let value: serde_json::Value = resp.json().await?;
        let embedding = value
            .get("data")
            .and_then(|d| d.get(0))
            .and_then(|d| d.get("embedding"))
            .and_then(|e| e.as_array())
            .ok_or_else(|| anyhow!("missing data[0].embedding in NVIDIA response"))?;

        let vec: Vec<f32> = embedding
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        if vec.len() != 1024 {
            bail!(
                "embedding dim {} != 1024; NVIDIA_EMBED_MODEL must be a 1024-dim model to match the DB schema",
                vec.len()
            );
        }

        Ok(vec)
    }
}

/// Stateful filter that strips `<think>...</think>` reasoning blocks out of a delta stream,
/// tolerating the tags being split across multiple deltas.
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

    /// Feed a content delta, returns the visible text to emit now (may be empty).
    fn feed(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        let mut out = String::new();

        loop {
            if self.in_think {
                if let Some(idx) = self.pending.find("</think>") {
                    self.pending.drain(..idx + "</think>".len());
                    self.in_think = false;
                    continue;
                } else {
                    // Hold back a possible partial prefix of "</think>" at the end.
                    let keep = partial_suffix_len(&self.pending, "</think>");
                    let drop_len = self.pending.len() - keep;
                    self.pending.drain(..drop_len);
                    break;
                }
            } else if let Some(idx) = self.pending.find("<think>") {
                let visible: String = self.pending.drain(..idx).collect();
                self.pending.drain(.."<think>".len());
                self.in_think = true;
                if !visible.is_empty() {
                    self.emitted_visible = true;
                }
                out.push_str(&visible);
                continue;
            } else {
                // Hold back a possible partial prefix of "<think>" at the end.
                let keep = partial_suffix_len(&self.pending, "<think>");
                let emit_len = self.pending.len() - keep;
                let mut visible: String = self.pending.drain(..emit_len).collect();
                if !self.emitted_visible {
                    let trimmed = visible.trim_start();
                    if trimmed.len() != visible.len() {
                        visible = trimmed.to_string();
                    }
                }
                if !visible.is_empty() {
                    self.emitted_visible = true;
                }
                out.push_str(&visible);
                break;
            }
        }

        out
    }

    /// Flush any remaining non-think text held in the buffer.
    fn flush(&mut self) -> Option<String> {
        if self.in_think {
            self.pending.clear();
            return None;
        }
        let remainder = std::mem::take(&mut self.pending);
        if remainder.is_empty() {
            None
        } else {
            Some(remainder)
        }
    }
}

/// Returns the length of the longest suffix of `s` that is a proper prefix of `tag`
/// (i.e. could be the start of `tag` split across a delta boundary).
fn partial_suffix_len(s: &str, tag: &str) -> usize {
    let max = s.len().min(tag.len() - 1);
    for len in (1..=max).rev() {
        if s.ends_with(&tag[..len]) {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_no_tags() {
        let mut f = ThinkFilter::new();
        let mut out = String::new();
        out.push_str(&f.feed("Hello, "));
        out.push_str(&f.feed("world!"));
        if let Some(rest) = f.flush() {
            out.push_str(&rest);
        }
        assert_eq!(out, "Hello, world!");
    }

    #[test]
    fn tag_split_across_deltas() {
        let mut f = ThinkFilter::new();
        let mut out = String::new();
        out.push_str(&f.feed("visible <thi"));
        out.push_str(&f.feed("nk>secret reasoning</thi"));
        out.push_str(&f.feed("nk>after"));
        if let Some(rest) = f.flush() {
            out.push_str(&rest);
        }
        assert_eq!(out, "visible after");
    }

    #[test]
    fn whole_think_block_dropped() {
        let mut f = ThinkFilter::new();
        let mut out = String::new();
        out.push_str(&f.feed("<think>reasoning here</think>answer"));
        if let Some(rest) = f.flush() {
            out.push_str(&rest);
        }
        assert_eq!(out, "answer");
    }
}
