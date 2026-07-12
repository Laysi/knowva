# AI 知識中樞（MVP）

通用企業 AI 知識管理系統 demo：保存使用者文件、建立檢索資料，並以 SSE 串流回答問題與附上來源。MVP 的所有模型推論只呼叫雲端託管 API。

## 已鎖定的模型契約

三種能力各自有獨立的 model id 與 endpoint，不會以任意模型互相替代：

| 能力 | 固定契約 |
|---|---|
| 回答／多模態閱讀 | 雲端 Gemma 4；`GEMMA_4_MODEL_ID` 必填，不提供猜測值或其他 family fallback |
| 向量嵌入 | `nvidia/llama-nemotron-embed-vl-1b-v2`，輸出 2048 維 float embedding |
| 精排 | `nvidia/llama-nemotron-rerank-vl-1b-v2` |

後端保留 OpenAI-compatible `/chat/completions` SSE，並提供 typed embedding 與 reranking client。頁面化 ingestion 先以模型無關的 artifact staging 產生可追蹤頁圖或標題感知文字單元，再用固定 Nemotron Embed VL 建立同一個 `halfvec(2048)` 檢索空間：文字 query 使用 `query/text`，MD/TXT 使用 `passage/text`，無文字層頁面使用 `passage/image`，有同頁文字時使用 `passage/text_image`。查詢會各取 dense 50 筆與文字 BM25 50 筆，再以 deterministic RRF 融合；沒有文字層的掃描頁仍可由 dense 路徑召回。

## 技術架構

| 層 | 技術 |
|---|---|
| 前端 | React + Vite + assistant-ui（純 CSS、繁中介面） |
| 後端 | Rust（axum + tokio + sqlx），單一 binary |
| 資料庫 | PostgreSQL 17 + pgvector |
| 模型呼叫 | 單一 cloud-model HTTP client；Gemma 4 + Nemotron Embed/Rerank VL |
| 檔案儲存 | `backend/data/files/{document_id}/`；原檔與每次完整 artifact run 分開保存 |

## 前置需求

- Docker（完整容器啟動只需要 Docker）
- 本機開發另需 [proto](https://moonrepo.dev/proto)；repository 的 `.prototools` 會安裝固定版本的 Rust、Node.js 與 npm
- 不使用容器轉檔時，主機需另裝 Pdfium 動態函式庫；可安裝在系統 library path，或以 `PDFIUM_LIBRARY_PATH` 指定檔案／目錄
- 不使用容器處理 DOCX/PPTX/XLSX 時，主機需另裝 LibreOffice headless 與所需 CJK 字型
- 一個 Cloudflare AI Gateway(所有模型流量一律經 gateway 轉送,不再直連供應商端點),並啟用 Authenticated Gateway 與一個 Run token
- 可呼叫上述三個固定模型的 NVIDIA Build API key(chat 改走 OpenRouter 則另備 OpenRouter key),以 BYOK 存放在 gateway;後端不保存任何 provider key
- 由實際雲端端點確認過的 Gemma 4 model id

## 設定

先複製範例：

```bash
cp .env.example .env
```

先在 Cloudflare dashboard 依序完成：建立 AI Gateway → 啟用 Authenticated Gateway 並建立 Run token → 註冊兩個 custom providers（slug `nvidia-build` → `https://integrate.api.nvidia.com`、slug `nvidia-ai` → `https://ai.api.nvidia.com`）→ 以 BYOK 為兩個 custom providers 存入 NVIDIA Build API key（chat 走 OpenRouter 時另存 OpenRouter key）。provider key 一律存於 gateway，後端不會送出上游 `Authorization` header。最少必須填入：

```dotenv
CF_AI_GATEWAY_BASE_URL=https://gateway.ai.cloudflare.com/v1/<account_id>/<gateway_id>
CF_AI_GATEWAY_TOKEN=your-ai-gateway-run-token
GEMMA_4_MODEL_ID=the-confirmed-cloud-gemma-4-id
```

其他 cloud-model 設定：

| 環境變數 | 必要性 | 說明 |
|---|---|---|
| `CLOUD_CHAT_URL` | 選填 | 完整 chat gateway route；空值時使用 `${CF_AI_GATEWAY_BASE_URL}/custom-nvidia-build/v1/chat/completions`，改走 OpenRouter 則填 `${CF_AI_GATEWAY_BASE_URL}/openrouter/v1/chat/completions` |
| `CLOUD_EMBED_URL` | 選填 | 完整 embedding gateway route；空值時使用 `${CF_AI_GATEWAY_BASE_URL}/custom-nvidia-build/v1/embeddings` |
| `CLOUD_RERANK_URL` | 選填 | 完整 reranking gateway route；空值時使用 `${CF_AI_GATEWAY_BASE_URL}/custom-nvidia-ai/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking` |
| `NEMOTRON_EMBED_MODEL_ID` | 有鎖定預設 | 只接受 `llama-nemotron-embed-vl-1b-v2` family |
| `NEMOTRON_RERANK_MODEL_ID` | 有鎖定預設 | 只接受 `llama-nemotron-rerank-vl-1b-v2` family |
| `RERANK_CANDIDATE_TOP_K` | `50` | RRF 候選池送入 VL rerank 的數量，必須不小於 `RERANK_TOP_N` |
| `RERANK_TOP_N` | `6` | 精排後回答 context 數量，只允許 3–8 |
| `RERANK_BATCH_SIZE` | `8` | 每次 hosted rerank request 的 passage 數量，範圍為 1–512 |
| `CLOUD_MODEL_REQUEST_TIMEOUT_SECS` | 有預設 | 所有模型 HTTP/SSE request timeout，必須是正整數 |

Gateway base URL、gateway token、Gemma 4 model id 缺失或 rerank 設定無效時，後端會在連接資料庫與接受流量前停止並顯示可理解的錯誤。啟動 log 只會列 endpoint、model id、timeout 與非敏感 rerank limits，不會列出 token。

文件前處理設定：

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `FILES_DIR` | `./data/files` | 原檔、頁圖、轉出 PDF 與 manifest 的根目錄 |
| `PDF_RENDER_DPI` | `180` | Pdfium rasterize DPI；允許 72–600 |
| `DOCUMENT_CONVERSION_TIMEOUT_SECS` | `120` | 每次 Pdfium 圖片處理或 LibreOffice 轉檔 timeout，必須為正整數 |
| `PDFIUM_LIBRARY_PATH` | 空 | 完整 library 路徑或所在目錄；空值時依序搜尋執行檔旁與系統 library path |
| `LIBREOFFICE_BIN` | `libreoffice` | LibreOffice executable 路徑或命令名 |

支援 PDF、PNG/JPEG/WebP、DOCX/PPTX/XLSX、MD/Markdown/TXT，HTTP 上傳上限為 50 MiB。CSV/HTML 與舊版 Office 格式目前不在支援清單。

## 使用 Docker Compose

容器映像已包含 Pdfium、LibreOffice 與 Noto CJK 字型，不需要主機另外提供 PDF API 或 native library。先設定 `.env` 的 `CF_AI_GATEWAY_BASE_URL`、`CF_AI_GATEWAY_TOKEN` 與 `GEMMA_4_MODEL_ID`，再啟動完整服務：

```bash
cp .env.example .env
# 編輯 .env，填入必要的雲端模型設定
docker compose up -d --build
docker compose ps
curl http://localhost:8080/api/health
```

預設網址是 `http://localhost:8080`；可在 `.env` 以 `APP_PORT` 改變主機 port。`app_files` 與 `pgdata` named volumes 會保存上傳檔案與資料庫。查看或停止服務：

```bash
docker compose logs -f app
docker compose down
```

若要在 build 階段實際驗證 Pdfium 與 LibreOffice 對 PDF、掃描 PDF、DOCX、PPTX、XLSX 的分頁與頁序：

```bash
docker build --target native-test -t ai-km-demo:native-test .
```

## 本機開發

先依 `.prototools` 安裝工具，再只啟動資料庫：

```bash
proto install
docker compose up -d postgres

cd backend
proto exec rust -- cargo run      # http://localhost:8080
```

前端開發模式（另開終端）：

```bash
cd frontend
proto run npm -- ci
proto run npm -- run dev          # http://localhost:5173，proxy /api 到 :8080
```

或建立 production bundle，交由後端服務：

```bash
cd frontend
proto run npm -- run build
```

## API

| Method | Path | 說明 |
|---|---|---|
| GET | `/api/health` | 健康檢查 |
| GET | `/api/documents` | 文件列表（含文件／artifact 狀態、頁數與檢索單元數） |
| POST | `/api/documents` | multipart 上傳（單一 `file` 欄位），立即回 id，背景處理 |
| DELETE | `/api/documents/{id}` | 刪除 DB row、原檔、轉出 PDF、頁圖、manifest 與暫存物 |
| POST | `/api/chat` | SSE 串流問答（`text-delta` / `sources` / `done` / `error`） |

`/api/chat` 的純文字串流契約維持不變；有檢索來源時，`sources` 一定在 `done` 前送出。模型端 HTTP 錯誤包含 status、最多 500 字的安全 response snippet 與可重試狀態；timeout 也會以統一錯誤回傳。

## Artifact 契約

分頁文件的典型布局如下；每次重試會先建立同檔案系統的暫存 run，全部成功後才以 rename 發布，DB 也在單一 transaction 內切換 current artifact set：

```text
FILES_DIR/{document_id}/
├── original.ext
└── artifacts-{run_id}/
    ├── converted.pdf       # 僅 Office
    ├── manifest.json
    └── pages/
        ├── 0001.png
        └── 0002.png
```

`page_artifacts` 保存 1-based `page_number`、圖片路徑、尺寸與可空的 PDF 文字層；掃描頁沒有文字仍是有效頁面。極端 PDF MediaBox 仍依設定 DPI 縮放，但輸出單邊上限為 8192 px，避免 native bitmap 無界配置。MD/TXT 不製造假頁碼，改存 `text_artifacts`。artifact 發布完成後，每份文件最多同時處理四個 embedding unit；只有所有回應都恰為 2048 維 finite float、artifact 與 unit 計數一致，才在單一 transaction 內替換 `document_units` 並把文件設為 `ready`。任一 unit 失敗時已完成的 artifact 保持 `artifact_status=ready`，文件則為 `failed`，不會留下可搜尋的部分新結果。

頁圖只以記憶體中的 PNG/JPEG data URL 送出，不會傳送本機 artifact path；decoded image 上限 25 MiB、整體 embedding request 上限 36 MiB。Embedding 對 408、429、5xx、transport error 與 timeout 最多嘗試三次並使用 exponential backoff + jitter，其他 4xx 立即失敗。HTTP error snippet、API key、Authorization 與 base64 圖片資料都會在記錄前限長或遮蔽。

## Hybrid retrieval 契約

- Dense 使用 query 的固定 2048 維 embedding 與 ready-only partial HNSW `halfvec_cosine_ops` index，cosine top-k 預設 50。HNSW nearest-neighbor query 先 materialize，再以 distance、`unit_id` 做穩定排序。
- Lexical 是標準 Okapi BM25，不使用 `ILIKE`、`ts_rank` 或 `ts_rank_cd`。公式參數固定 `k1=1.2`、`b=0.75`；migration `0004_hybrid_retrieval.sql` 以 trigger 維護 ready-only corpus 的文件數、總 token 數與每詞 DF，query 在 PostgreSQL 內計算 TF/IDF/長度正規化，不會把 corpus 載入 Rust。
- Lexical candidate index 是 non-empty `content` 上的 ready-only partial GIN token-array index。tokenizer 先做 Unicode NFKC、lowercase 與 dash normalization；繁中採 character unigram + adjacent bigram，原因是不用可變字典也能重現、可涵蓋未知企業詞；英文、數字與產品代號保留完整 token，hyphenated code 另產 compact 與 component variants，讓全形／半形或常見標點差異不會讓基本詞失效。
- RRF 固定 `k=60`，`rrf_score(d) = Σ 1 / (60 + rank_i(d))`。以 `unit_id` 去重；同分依最佳 route rank、`unit_id` 排序。兩路預設各 50，融合預設保留 50，再以固定 `llama-nemotron-rerank-vl-1b-v2` 精排成 top 6（可設 3–8）。
- Page passage 只讀已發布 artifact：有文字層時送 image+text，掃描頁送 image-only（完全省略 `text`），MD/TXT chunk 送 text-only。本機路徑不進 payload，圖片只以記憶體 data URL 傳送；預設每批 8 筆，API 回應即使重排順序也依每批 local index 映回原 unit。
- 每筆 `RetrievalCandidate` 直接帶 `unit_id`、`document_id`、`filename`、`kind`、1-based nullable `page_number`、nullable `image_path/content`、兩路 rank/score、`rrf_score` 與 nullable `rerank_score/rerank_rank`，後續階段不需重新查來源 metadata。
- Rerank 每批都嚴格驗證回應筆數、index 唯一且不越界、score finite，再驗證跨批映射。408、429、5xx、transport error 與 timeout 最多嘗試三次；耗盡後以原 RRF top_n 原序回退，且 rerank 欄位保持 null。設定、artifact 或 hosted response contract 錯誤不回退，會回傳可診斷錯誤。Structured log 僅記錄各階段筆數、延遲、fallback 與 model id，不含 query、文件內容、secret 或 base64。

## 驗證

```bash
cd backend
proto exec rust -- cargo fmt --check
proto exec rust -- cargo clippy --all-targets -- -D warnings
proto exec rust -- cargo test
```

一般測試以本機 mock cloud endpoint 驗證四種 embedding payload、三種 rerank modality、跨批映射、嚴格 response contract、有限重試／fallback 與 transaction/DB 契約，不會呼叫真實雲端。安裝 Pdfium 與 LibreOffice 後，可明確執行 native fixture smoke；它也會確認原生 PDF 使用 image+text embedding、掃描 PDF 使用 image-only embedding：

```bash
cd backend
PDFIUM_LIBRARY_PATH=/path/to/libpdfium.so \
LIBREOFFICE_BIN=libreoffice \
proto exec rust -- cargo test page_artifacts::tests::generated_pdf_and_office_fixtures_preserve_page_order \
  -- --ignored --exact --nocapture
```

fixture 的來源、預期頁數與涵蓋範圍見 [backend/tests/fixtures/README.md](backend/tests/fixtures/README.md)。

若要選擇性驗證經 Cloudflare AI Gateway 的真實 NVIDIA Build endpoint，先在 `.env` 填妥 `CF_AI_GATEWAY_BASE_URL`（或完整的 `CLOUD_EMBED_URL` gateway route）、`CF_AI_GATEWAY_TOKEN` 與 `GEMMA_4_MODEL_ID`，並確認 NVIDIA key 已以 BYOK 存於 gateway，再明確執行下列 ignored test。它會各送一次 text query、text passage、image passage 與 image+text passage（共四次雲端呼叫）；一般 `cargo test` 與 CI 不會執行：

```bash
cd backend
proto exec rust -- cargo test \
  cloud_models::tests::real_nemotron_embed_vl_smoke_covers_all_input_modalities \
  -- --ignored --exact --nocapture
```

詳細 MVP 規劃見 [docs/mvp-now.md](docs/mvp-now.md)。
