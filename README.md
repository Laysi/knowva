# AI 知識中樞(Demo)

通用企業 AI 知識管理系統 demo:上傳文件 → 自動轉檔、切塊、向量化 → RAG 智慧問答(回答標註引用來源)。
除了 AWS Bedrock 的模型呼叫之外,所有元件皆在地端執行。

## 功能

- **文件處理管線**:上傳後自動以 markitdown 轉 Markdown → 標題感知切塊 → Titan 向量化 → 寫入 pgvector;並由 LLM 產生文件摘要與標籤
- **RAG 問答**:pgvector 餘弦相似度 top-8 檢索,Nova 2 Lite 串流回答,結尾附引用來源;知識庫無資料時自動退回一般對話
- **文件管理**:處理狀態(處理中/完成/失敗)、摘要、標籤、段落數、刪除

## 技術架構

| 層 | 技術 |
|---|---|
| 前端 | React + Vite + assistant-ui(純 CSS,繁中介面) |
| 後端 | Rust(axum + tokio + sqlx),單一 binary |
| 資料庫 | PostgreSQL 17 + pgvector(HNSW 索引,1024 維) |
| LLM | AWS Bedrock:Nova 2 Lite(對話/摘要)、Titan Text Embeddings V2(向量) |
| 檔案儲存 | 本機檔案系統 `backend/data/files/` |

## 前置需求

- Docker、Rust toolchain、Node.js、Python 3.10+
- `pipx install "markitdown[all]"`(需在 PATH 或 `~/.local/bin`)
- AWS 憑證(`aws login` 或 default chain 的其他來源),帳號已開通上述兩個 Bedrock 模型
- 對外連線僅需 `bedrock-runtime.<region>.amazonaws.com`

## 快速開始

```bash
cp .env.example .env            # 填入 AWS_REGION 等設定
docker compose up -d            # 啟動 pgvector Postgres(:5432)

cd backend && cargo run         # :8080,啟動時自動跑 migrations(必須在 backend/ 目錄執行)

# 前端開發模式(另開終端):
cd frontend && npm install && npm run dev     # :5173,proxy /api 到 :8080

# 或 production 模式:
cd frontend && npm run build    # 打包後由後端直接在 :8080 服務
```

## API

| Method | Path | 說明 |
|---|---|---|
| GET | `/api/health` | 健康檢查 |
| GET | `/api/documents` | 文件列表(含狀態、摘要、標籤、段落數) |
| POST | `/api/documents` | multipart 上傳(欄位 `file`),立即回 id,背景處理 |
| DELETE | `/api/documents/{id}` | 刪除文件(連同向量與實體檔案) |
| POST | `/api/chat` | SSE 串流問答(`text-delta` / `sources` / `done` / `error` 事件) |

## 已知限制(demo 取捨)

- 無登入驗證與權限控管;聊天記錄僅存於前端狀態
- 掃描型 PDF 不支援(無 OCR)
- 僅向量檢索,無混合搜尋與 rerank
- `aws login` session 過期時聊天會回 `{"type":"error","message":"dispatch failure"}`,重新執行 `aws login` 即可,無需重啟伺服器

詳細規劃見 [PLAN.md](PLAN.md),開發注意事項見 [CLAUDE.md](CLAUDE.md)。
