use crate::AppState;
use serde::Serialize;
use uuid::Uuid;

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct RetrievedChunk {
    pub content: Option<String>,
    pub document_id: Uuid,
    pub filename: String,
    pub page_number: Option<i32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceRef {
    #[serde(rename = "documentId")]
    pub document_id: Uuid,
    pub filename: String,
    #[serde(rename = "pageNumber", skip_serializing_if = "Option::is_none")]
    pub page_number: Option<i32>,
}

/// Embeds the query text and retrieves the top-8 most similar chunks from ready documents.
pub async fn retrieve(state: &AppState, query: &str) -> anyhow::Result<Vec<RetrievedChunk>> {
    let embedding = state.models.embed_query(query).await?;
    let vector = crate::embedding::to_halfvec(embedding)?;

    let rows = sqlx::query_as::<_, RetrievedChunk>(
        "SELECT u.content, d.id AS document_id, d.filename, u.page_number \
         FROM document_units u JOIN documents d ON d.id = u.document_id \
         WHERE d.status = 'ready' \
         ORDER BY u.embedding <=> $1 \
         LIMIT 8",
    )
    .bind(vector)
    .fetch_all(&state.pool)
    .await?;

    Ok(rows)
}

/// Builds the RAG-augmented system prompt from retrieved chunks.
pub fn build_rag_system_prompt(chunks: &[RetrievedChunk]) -> String {
    let mut prompt = String::new();
    prompt.push_str(crate::cloud_models::SYSTEM_PROMPT);
    prompt.push_str(
        "\n\n以下是知識庫檢索到的段落,回答時優先依據這些內容,並在回答中標註出處檔名;若段落與問題無關或不足,請直接說明知識庫沒有相關資料。\n\n",
    );
    for (i, chunk) in chunks.iter().enumerate() {
        let page = chunk
            .page_number
            .map(|number| format!(" 第 {number} 頁"))
            .unwrap_or_default();
        let content = chunk
            .content
            .as_deref()
            .unwrap_or("[此頁只有頁面影像，尚無文字層]");
        prompt.push_str(&format!(
            "[{}] {}{}\n{}\n\n",
            i + 1,
            chunk.filename,
            page,
            content
        ));
    }
    prompt
}

/// Dedupe retrieved units by document/page, preserving retrieval rank order.
pub fn dedupe_sources(chunks: &[RetrievedChunk]) -> Vec<SourceRef> {
    let mut seen = std::collections::HashSet::new();
    let mut sources = Vec::new();
    for chunk in chunks {
        if seen.insert((chunk.document_id, chunk.page_number)) {
            sources.push(SourceRef {
                document_id: chunk.document_id,
                filename: chunk.filename.clone(),
                page_number: chunk.page_number,
            });
        }
    }
    sources
}
