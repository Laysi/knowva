# AI 知識中樞（MVP）

通用企業 AI 知識管理系統 demo：保存使用者文件、建立檢索資料，並以 SSE 串流回答問題與附上來源。MVP 的所有模型推論只呼叫雲端託管 API。

## 已鎖定的模型契約

三種能力各自有獨立的 model id 與 endpoint，不會以任意模型互相替代：

| 能力 | 固定契約 |
|---|---|
| 回答／多模態閱讀 | 雲端 Gemma 4；`GEMMA_4_MODEL_ID` 必填，不提供猜測值或其他 family fallback |
| 向量嵌入 | `nvidia/llama-nemotron-embed-vl-1b-v2`，輸出 2048 維 float embedding |
| 精排 | `nvidia/llama-nemotron-rerank-vl-1b-v2` |

後端保留 OpenAI-compatible `/chat/completions` SSE，並提供 typed embedding 與 reranking client。頁面化 ingestion、2048 維資料表 migration 與把 rerank 接入檢索流程由後續 MVP issue 完成。

## 技術架構

| 層 | 技術 |
|---|---|
| 前端 | React + Vite + assistant-ui（純 CSS、繁中介面） |
| 後端 | Rust（axum + tokio + sqlx），單一 binary |
| 資料庫 | PostgreSQL 17 + pgvector |
| 模型呼叫 | 單一 cloud-model HTTP client；Gemma 4 + Nemotron Embed/Rerank VL |
| 檔案儲存 | `backend/data/files/` |

## 前置需求

- Docker、Rust toolchain、Node.js
- 可呼叫上述三個固定模型的雲端 API key
- 由實際雲端端點確認過的 Gemma 4 model id

## 設定

先複製範例：

```bash
cp .env.example .env
```

最少必須填入：

```dotenv
NVIDIA_API_KEY=your-cloud-api-key
GEMMA_4_MODEL_ID=the-confirmed-cloud-gemma-4-id
```

其他 cloud-model 設定：

| 環境變數 | 必要性 | 說明 |
|---|---|---|
| `CLOUD_MODEL_BASE_URL` | 有預設 | 共用雲端 API base URL |
| `CLOUD_CHAT_URL` | 選填 | 完整 chat URL；空值時使用 common base 的 `/chat/completions` |
| `CLOUD_EMBED_URL` | 選填 | 完整 embedding URL；空值時使用 common base 的 `/embeddings` |
| `CLOUD_RERANK_URL` | 選填 | 完整 reranking URL；範例已填 NVIDIA 託管端點 |
| `NEMOTRON_EMBED_MODEL_ID` | 有鎖定預設 | 只接受 `llama-nemotron-embed-vl-1b-v2` family |
| `NEMOTRON_RERANK_MODEL_ID` | 有鎖定預設 | 只接受 `llama-nemotron-rerank-vl-1b-v2` family |
| `CLOUD_MODEL_REQUEST_TIMEOUT_SECS` | 有預設 | 所有模型 HTTP/SSE request timeout，必須是正整數 |

API key 或 Gemma 4 model id 缺失時，後端會在連接資料庫與接受流量前停止並顯示可理解的錯誤。啟動 log 只會列 endpoint、model id 與 timeout，不會列出 token。

## 快速開始

```bash
docker compose up -d

cd backend
cargo run                         # http://localhost:8080
```

前端開發模式（另開終端）：

```bash
cd frontend
npm install
npm run dev                       # http://localhost:5173，proxy /api 到 :8080
```

或建立 production bundle，交由後端服務：

```bash
cd frontend
npm run build
```

## API

| Method | Path | 說明 |
|---|---|---|
| GET | `/api/health` | 健康檢查 |
| GET | `/api/documents` | 文件列表（含狀態、摘要、標籤、段落數） |
| POST | `/api/documents` | multipart 上傳（欄位 `file`），立即回 id，背景處理 |
| DELETE | `/api/documents/{id}` | 刪除文件與實體檔案 |
| POST | `/api/chat` | SSE 串流問答（`text-delta` / `sources` / `done` / `error`） |

`/api/chat` 的純文字串流契約維持不變；有檢索來源時，`sources` 一定在 `done` 前送出。模型端 HTTP 錯誤包含 status、最多 500 字的安全 response snippet 與可重試狀態；timeout 也會以統一錯誤回傳。

## 驗證

```bash
cd backend
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

詳細 MVP 規劃見 [docs/mvp-now.md](docs/mvp-now.md)。
