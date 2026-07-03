# 通用 AI 知識管理系統 — Demo 規劃

## 目標

使用者上傳文件(手冊、SOP、規範、報告、簡報、表格等),
系統透過 LLM 處理建立知識庫,並提供 RAG 問答介面,以繁體中文回答並附上文件來源。

## 技術選型(已定案)

| 項目 | 技術 |
|------|------|
| 前端 | React + assistant-ui(Vite SPA) |
| 後端 | Rust(axum + tokio + sqlx) |
| 資料庫 | PostgreSQL + pgvector(Docker,地端) |
| 對話 LLM | AWS Bedrock — Nova 2 Lite(唯一雲端元件) |
| Embedding | AWS Bedrock — Titan Text Embeddings V2(1024 維) |
| 檔案儲存 | 本機檔案系統 `./data/files/` |

## 資料流

**上傳處理(ingest):**

```
前端上傳 → axum 存檔到本機 → tokio::spawn 背景處理:
  markitdown 轉成 Markdown(TXT/MD 直接讀;轉檔結果存原檔旁)
  → 依標題切段、過長再切塊(約 800 token,10% 重疊)
  → Bedrock 產生 embedding → 寫入 pgvector
  → Nova 2 Lite 產生文件摘要與標籤 → status = ready
```

**問答(RAG):**

```
提問 → 問題 embedding → pgvector 餘弦相似度 top 8
  → 組 prompt(chunk 內容 + 檔名出處)→ Nova 2 Lite ConverseStream
  → SSE 串流回前端 → assistant-ui 顯示回答與引用來源
```

## Demo 簡化決策

- 不做登入驗證、不做多租戶、不做權限
- 不用 job queue / Redis:上傳後直接 `tokio::spawn`,狀態記在 `documents.status`
- 統一用 markitdown(Python CLI,後端以子行程呼叫)轉 Markdown:支援 PDF、DOCX、XLSX、PPTX;TXT/MD 直接讀;掃描 PDF 不支援
- 檢索只用向量搜尋,不做混合搜尋、不做 rerank
- 聊天記錄不落地,只存在前端狀態
- 開發時前端跑 Vite dev server(proxy 到後端);打包後由 axum 服務靜態檔案

## 資料庫 Schema(2 張表)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  filename     text NOT NULL,
  storage_path text NOT NULL,
  mime_type    text,
  status       text NOT NULL DEFAULT 'processing',  -- processing | ready | failed
  summary      text,
  tags         text[],
  error        text,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  id          bigserial PRIMARY KEY,
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index int NOT NULL,
  content     text NOT NULL,
  embedding   vector(1024)
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

## API(4 支)

| Method | Path | 說明 |
|--------|------|------|
| POST | `/api/documents` | multipart 上傳,立即回 id,背景處理 |
| GET | `/api/documents` | 文件列表(含狀態、摘要、標籤) |
| DELETE | `/api/documents/:id` | 刪除文件(連同 chunks 與實體檔案) |
| POST | `/api/chat` | 傳入訊息陣列,回 SSE 串流(RAG 回答 + 來源) |

前端 assistant-ui 用 `useLocalRuntime` + 自訂 `ChatModelAdapter`,
fetch `/api/chat` 讀 SSE 逐字附加,結尾事件夾帶引用來源清單。

## 專案結構

```
ai-km-demo/
├── backend/              # 單一 axum crate
│   ├── src/
│   │   ├── main.rs       # 路由、啟動
│   │   ├── bedrock.rs    # Nova 2 Lite(converse_stream)+ Titan embedding
│   │   ├── ingest.rs     # markitdown 轉檔、切塊、embedding、摘要
│   │   ├── search.rs     # pgvector 相似度查詢
│   │   └── db.rs
│   ├── migrations/
│   └── data/files/       # 上傳檔案(gitignore;伺服器從 backend/ 執行)
├── frontend/             # Vite + React + assistant-ui
├── docker-compose.yml    # 只有 postgres(pgvector/pgvector:pg17)
└── PLAN.md
```

## 環境變數

```
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
BEDROCK_CHAT_MODEL_ID=us.amazon.nova-2-lite-v1:0   # 以 Bedrock 主控台實際 ID 為準
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
DATABASE_URL=postgres://km:km@localhost:5432/km
FILES_DIR=./data/files
```

地端只需開放對 `bedrock-runtime.<region>.amazonaws.com` 的 HTTPS 出站連線。

前置需求:Docker、Rust toolchain、Node.js、Python 3.10+ 並 `pipx install "markitdown[all]"`。

## 實作步驟

1. **骨架**:docker compose 啟 pgvector、axum 專案 + migrations、Vite + assistant-ui 前端
2. **聊天直通**:`/api/chat` → Bedrock Nova 2 Lite → SSE,前端接上(先不含 RAG,驗證 Bedrock 連線與串流)
3. **上傳管線**:存檔 → 抽字 → 切塊 → embedding → 入庫 → 摘要/標籤
4. **RAG**:檢索 top 8 接進聊天 prompt,回答附引用來源
5. **收尾**:文件列表頁(上傳、狀態、刪除)、系統提示詞為通用企業知識助理(繁中回答)
