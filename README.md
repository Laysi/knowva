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

`GET /api/documents` 的 canonical response 使用 camelCase；下列範例同時省略了過渡期保留的 `artifact_status`、`page_count`、`unit_count`、`chunk_count`、`created_at` 與安全字串 `error` aliases：

```json
[
  {
    "id": "00000000-0000-0000-0000-000000000001",
    "filename": "操作手冊.pdf",
    "status": "processing",
    "processingStage": "indexing",
    "processingError": null,
    "summary": null,
    "tags": null,
    "createdAt": "2026-07-12T08:00:00Z",
    "pageCount": 12,
    "unitCount": 12
  }
]
```

`status` 維持 `processing` / `ready` / `failed`；`processingStage` 進一步區分 `queued` / `extracting` / `indexing` / `ready` / `failed`。失敗時 `processingError` 只提供固定的繁中 code/message，不會回傳資料庫診斷、provider response、路徑或 secret。Deprecated aliases 只供現有 MVP client 過渡，`chunk_count` 固定等於 `unitCount`。

`/api/chat` 的純文字串流契約維持不變；有檢索來源時，`sources` 一定在 `done` 前送出。模型端 HTTP 錯誤包含 status、最多 500 字的安全 response snippet 與可重試狀態；timeout 也會以統一錯誤回傳。

分頁 source 的 `pageNumber` 是 1-based；文字 source 的 `pageNumber` 固定為 JSON `null`，並以 0-based `unitIndex` 標識文字單元（UI 顯示時加一）。同文件的不同頁不會去重，完整 SSE 範例如下：

```text
data: {"type":"text-delta","text":"答案 [S1]"}

data: {"type":"sources","sources":[{"sourceId":"S1","documentId":"00000000-0000-0000-0000-000000000001","unitId":101,"unitIndex":1,"filename":"操作手冊.pdf","pageNumber":2,"kind":"page"},{"sourceId":"S2","documentId":"00000000-0000-0000-0000-000000000002","unitId":202,"unitIndex":0,"filename":"備註.md","pageNumber":null,"kind":"text"}]}

data: {"type":"done"}
```

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

所有 embedding 完成後，系統會以同一個雲端 Gemma 4 best-effort 產生繁中摘要與 3–6 個標籤：分頁文件固定只取前 3 頁，MD/TXT 固定只取前 3 個文字單元；每個文字單元最多 3000 字，每頁 extracted text 最多 2000 字，頁圖沿用最長邊 1600 px、JPEG quality 80、單一 serialized image part 5 MiB 與完整 chat request 24 MiB 的界線。輸出必須是 typed JSON，摘要非空且最多 500 字，每個唯一標籤非空且最多 30 字；只容許單一 code fence 或單一 JSON object 前後 chatter 的 JSON-aware salvage。暫時性錯誤最多兩次 summary request，malformed JSON 不重試；任一 summary 失敗都只記安全 warning，仍會把完整 embeddings 與 `ready` 在同一 transaction 發布，不會重跑 embedding。

## Hybrid retrieval 契約

- Dense 使用 query 的固定 2048 維 embedding 與 ready-only partial HNSW `halfvec_cosine_ops` index，cosine top-k 預設 50。HNSW nearest-neighbor query 先 materialize，再以 distance、`unit_id` 做穩定排序。
- Lexical 是標準 Okapi BM25，不使用 `ILIKE`、`ts_rank` 或 `ts_rank_cd`。公式參數固定 `k1=1.2`、`b=0.75`；migration `0004_hybrid_retrieval.sql` 以 trigger 維護 ready-only corpus 的文件數、總 token 數與每詞 DF，query 在 PostgreSQL 內計算 TF/IDF/長度正規化，不會把 corpus 載入 Rust。
- Lexical candidate index 是 non-empty `content` 上的 ready-only partial GIN token-array index。tokenizer 先做 Unicode NFKC、lowercase 與 dash normalization；繁中採 character unigram + adjacent bigram，原因是不用可變字典也能重現、可涵蓋未知企業詞；英文、數字與產品代號保留完整 token，hyphenated code 另產 compact 與 component variants，讓全形／半形或常見標點差異不會讓基本詞失效。
- RRF 固定 `k=60`，`rrf_score(d) = Σ 1 / (60 + rank_i(d))`。以 `unit_id` 去重；同分依最佳 route rank、`unit_id` 排序。兩路預設各 50，融合預設保留 50，再以固定 `llama-nemotron-rerank-vl-1b-v2` 精排成 top 6（可設 3–8）。
- Page passage 只讀已發布 artifact：有文字層時送 image+text，掃描頁送 image-only（完全省略 `text`），MD/TXT chunk 送 text-only。本機路徑不進 payload，圖片只以記憶體 data URL 傳送；預設每批 8 筆，API 回應即使重排順序也依每批 local index 映回原 unit。
- 每筆 `RetrievalCandidate` 直接帶 `unit_id`、0-based `unit_index`、`document_id`、`filename`、`kind`、1-based nullable `page_number`、nullable `image_path/content`、兩路 rank/score、`rrf_score` 與 nullable `rerank_score/rerank_rank`，後續階段不需重新查來源 metadata。
- Rerank 每批都嚴格驗證回應筆數、index 唯一且不越界、score finite，再驗證跨批映射。408、429、5xx、transport error 與 timeout 最多嘗試三次；耗盡後以原 RRF top_n 原序回退，且 rerank 欄位保持 null。設定、artifact 或 hosted response contract 錯誤不回退，會回傳可診斷錯誤。Structured log 僅記錄各階段筆數、延遲、fallback 與 model id，不含 query、文件內容、secret 或 base64。

## A/B 評測

評測資料集位於 `backend/eval/fixtures/manifest.json`。在已安裝 proto 的全新 checkout，從 repository root 執行下列命令即可做離線 dry-run；若固定版 Rust 尚未安裝，先執行 `proto install`：

```bash
cd backend && proto exec rust -- cargo run -- eval --dry-run
```

dry-run 只讀取並驗證 fixture，不連資料庫、不載入 native converter，也不呼叫雲端模型。它會檢查 dataset schema/version、未知欄位、唯一 document/query/unit id、安全相對路徑、檔案類型與副檔名、SHA-256、非空 license/source、預期 page/unit inventory，以及 query scope、expected source、required facts、answerable/not-found 等契約。所有內附文件都是 deterministic synthetic CC0 fixture，不含客戶或正式環境內容；涵蓋繁中原生 PDF、純影像掃描 PDF、PPTX、表格／圖表 PDF、PNG、Markdown 與純文字，也涵蓋同名文件隔離、相鄰頁干擾、多頁來源、無頁碼文字與 visual-only facts。完整來源與 coverage 見 [backend/eval/fixtures/README.md](backend/eval/fixtures/README.md)。

可重複使用 `--document <document_key>` 與 `--query <query_id>` 選 subset；指定 document 時，只會選取其 `document_scope` 全部落在所選 documents 的 queries。dry-run 的 JSON 會列出實際選到的 documents/queries，未知或空 subset 會 fail closed。例如：

```bash
cd backend
proto exec rust -- cargo run -- eval --dry-run \
  --document markdown-operations \
  --query markdown-escalation
```

### 正式 cloud A/B run

Cloud run 是明確 opt-in，會 ingest fixtures 並實際呼叫 embedding、rerank 與 generation endpoint，因此會產生雲端費用；CI 與一般測試絕不自動執行。A 是 dense-only control，B 共用 A 的同一批 dense candidates，再執行 BM25、RRF 與固定 VL reranker。兩組固定使用相同 embedding 與 Gemma 4 generation model id、相同 prompt version、相同 query、相同 context top_n 與 deterministic retry policy，只有 retrieval path 不同。provider 不支援 seed 時會在 run manifest 明確記為 false。為確保 telemetry 可歸因，目前 cloud run 強制 `--concurrency 1`。

執行前必須完成下列事項：

1. 由固定核准者 GitHub login **`Ruisi-Lu`** 先在 `Laysi/outsourcing-ai-km` issue #1 建立一則留言，鎖定 correct-page recall 的 stage、k、minimum，citation precision/recall、answer facts recall，以及 B 相對 A improvement 的實際數值。GitHub API 回傳的 comment author 必須精確為 `Ruisi-Lu`，且 `author_association` 必須為 `OWNER`、`MEMBER` 或 `COLLABORATOR`；login 大小寫不同、其他作者或僅為 `CONTRIBUTOR` 都不會通過。所有 minimum 必須是 0 到 1 的 JSON number，k 必須是正整數；不要由執行者臨時決定門檻。先取得該留言固定的 `https://github.com/Laysi/outsourcing-ai-km/issues/1#issuecomment-<id>` permalink，再把它寫入本機 threshold JSON 的 `issue_comment_url`。下列只是 schema 示意；尖括號欄位必須換成留言已鎖定的真實內容，不能直接執行，也不要填一個看似可通過的假 URL 或猜測值：

   ```text
   {
     "schema_version": "1.0",
     "version": "<owner-locked-version>",
     "issue_comment_url": "<real-Laysi/outsourcing-ai-km-issue-1-comment-permalink>",
     "minimums": {
       "correct_page_recall_at_k": {
         "stage": "<dense|bm25|rrf|rerank>",
         "k": <owner-locked-positive-integer>,
         "minimum": <owner-locked-number-0-to-1>
       },
       "citation_precision": <owner-locked-number-0-to-1>,
       "citation_recall": <owner-locked-number-0-to-1>,
       "answer_facts_recall": <owner-locked-number-0-to-1>,
       "b_relative_improvement": <owner-locked-number-0-to-1>
     }
   }
   ```

2. 完成 threshold JSON 的所有欄位與排版後，對**最終檔案 bytes**計算 SHA-256（例如 `sha256sum /absolute/path/to/thresholds.json`；之後只要檔案有任何 byte 改動就必須重算）。由同一位 owner 編輯上一步的同一則留言，加入獨立一行 `threshold_sha256: <sha256sum 輸出的 64 字元雜湊>`；編輯留言不會改變 permalink。runner 會從 GitHub 讀取該固定 URL，並要求 URL 與 hash line 都精確符合本機 snapshot。
3. 由 PostgreSQL 管理者建立一個名稱以 `km_eval_` 開頭的全新空 database，以及名稱可辨識用途的專用登入角色（例如組織自訂的 `km_eval_runner_<識別碼>`；尖括號內容必須換成真實且唯一的值）。以該角色作為 database owner，讓它能執行 migration；接著在任何 migration 前，以 database owner 連線並執行 `COMMENT ON DATABASE <實際的 km_eval_ database 名稱> IS 'ai-km-evaluation-only:v1';`。最後把使用該專用角色與 database 的連線字串設為 `EVAL_DATABASE_URL`。runner 會檢查 current database 名稱、connect role 身分與精確 marker；它也會拒絕 production `DATABASE_URL`、repository 預設 DB，以及 migration 後含有任何 document 的 DB。不要把 placeholder、示意密碼或假 marker 當成正式設定。
4. 準備專用且為空的 `EVAL_FILES_DIR`。它不可等於、包含或位於 production `FILES_DIR` 或預設 `./data/files` 之下；失敗後會保留完整隔離 run 供診斷，下次不可重用非空目錄。
5. 設定 Cloudflare AI Gateway/BYOK 與確認過的 Gemma 4 model id：至少提供 `CF_AI_GATEWAY_BASE_URL`、`CF_AI_GATEWAY_TOKEN`、`GEMMA_4_MODEL_ID`；自訂 route 時另設 `CLOUD_CHAT_URL`、`CLOUD_EMBED_URL`、`CLOUD_RERANK_URL`。確認 gateway 已能存取固定 Nemotron embedding/rerank models。
6. 主機安裝 Pdfium 與 LibreOffice/CJK 字型，並以 `PDFIUM_LIBRARY_PATH`、`LIBREOFFICE_BIN` 指向可用 binary/library，讓 PDF、掃描頁與 PPTX fixtures 能完整 ingest。

完成上述前置作業後，從 `backend/` 以單一 cloud command 執行完整資料集；以下所有尖括號都必須替換為真實的隔離環境設定：

```bash
cd backend
EVAL_DATABASE_URL='postgres://<eval-user>:<password>@<host>:<port>/<brand-new-empty-eval-db>' \
EVAL_FILES_DIR='/absolute/path/to/empty/eval-files' \
CF_AI_GATEWAY_BASE_URL='https://gateway.ai.cloudflare.com/v1/<account-id>/<gateway-id>' \
CF_AI_GATEWAY_TOKEN='<gateway-run-token>' \
GEMMA_4_MODEL_ID='<confirmed-cloud-gemma-4-model-id>' \
PDFIUM_LIBRARY_PATH='/absolute/path/to/libpdfium.so' \
LIBREOFFICE_BIN='/absolute/path/to/libreoffice' \
proto exec rust -- cargo run -- eval --cloud \
  --manifest eval/fixtures/manifest.json \
  --thresholds /absolute/path/to/local/thresholds.json \
  --output /absolute/path/to/eval-results \
  --concurrency 1
```

正式 run 也接受上述 `--document` / `--query` subset flags。成功寫檔後，stdout 會回傳 `passed` 與路徑；每次輸出位於 `<output>/<run_id>/report.json`、`report.csv`、`report.md`、`run-manifest.json`。`passed: true` 只有在至少一組有效 A/B pair，且 correct-page、citation precision/recall、answer facts recall、B 相對 A improvement 全部達到 owner 鎖定門檻時成立；command 本身完成不代表門檻通過。`report.*` 會列出各 query 的 improvement/same/regression/failure、分數與 failure reasons，`run-manifest.json` 則保存 locked model ids、prompt/top_n、dataset/threshold hash、git revision、subsets、retry/latency/request/fallback telemetry 與安全化失敗案例。preflight、native conversion、cloud contract 或 DB isolation 錯誤會以非零狀態停止；已開始寫入後不會局部清掉 DB/files，以免破壞診斷一致性。

Threshold JSON 只應保存在本機安全位置：不得提交本機 threshold 檔、API/gateway key、`.env` secret 或任何 customer/production 文件。fixture 只能使用 repository 內已審核、標明來源與 license 的 synthetic files。

## 驗證

```bash
cd backend
proto exec rust -- cargo fmt --check
proto exec rust -- cargo clippy --all-targets -- -D warnings
proto exec rust -- cargo test

cd ../frontend
proto run npm -- ci
proto run npm -- run lint
proto run npm -- run test
proto run npm -- run build
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
