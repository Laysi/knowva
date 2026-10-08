# Knowva 知星（MVP）

> 點亮你的知識星系 — Your knowledge supernova

通用企業 AI 知識管理系統，以企業地端部署為目標：保存使用者文件、建立檢索資料，並以 SSE 串流回答問題與附上來源。正式環境的模型推論都在企業地端推論服務執行；開發與測試時，同一組端點設定也可以指向雲端託管的 OpenAI 相容服務。換服務只需要改 URL。

## 已鎖定的模型契約

三種能力各自有獨立的 model id 與 endpoint，不會以任意模型互相替代：

| 能力 | 固定契約 |
|---|---|
| 回答／多模態閱讀 | OpenAI 相容 chat endpoint（正式環境為地端服務）；model 由 `GEMMA_4_MODEL_ID` 放入 request body |
| 向量嵌入 | `nvidia/llama-nemotron-embed-vl-1b-v2`，輸出 2048 維 float embedding |
| 精排 | `nvidia/llama-nemotron-rerank-vl-1b-v2` |

後端保留 OpenAI-compatible `/chat/completions` SSE，並提供 typed embedding 與 reranking client。頁面化 ingestion 先以模型無關的 artifact staging 產生可追蹤頁圖或標題感知文字單元，再用固定 Nemotron Embed VL 建立同一個 `halfvec(2048)` 檢索空間：文字 query 使用 `query/text`，MD/TXT 與文字層足以代表整頁的頁面（`needs_vision = false`）使用 `passage/text`，其餘頁面中無文字層者使用 `passage/image`、有同頁文字者使用 `passage/text_image`。每頁要不要送圖只在轉檔時判斷一次，之後 embedding、rerank、回答與摘要都依同一個結果，見「頁面是否送圖」。查詢取 dense 前 20 筆送 Nemotron Rerank VL 精排，再取前 6 筆給 Gemma；沒有文字層的掃描頁同樣由 dense 召回。

## 技術架構

| 層 | 技術 |
|---|---|
| 前端 | React + Vite + assistant-ui（npm workspaces：共用的 `@knowva/core` 加上主題；純 CSS、繁中介面） |
| 後端 | Rust（axum + tokio + sqlx），單一 binary |
| 資料庫 | PostgreSQL 17 + pgvector |
| 模型呼叫 | 單一模型服務 HTTP client（`cloud_models.rs`，地端與雲端端點共用）；Gemma 4 + Nemotron Embed/Rerank VL |
| 檔案儲存 | 初始化時選定本機 `FILES_DIR` 或 S3 相容物件儲存（`object_store`）；DB 只存 `{document_id}/...` 相對 key，原檔與每次完整 artifact run 分開保存 |
| 產業 profile | `profiles/<id>/`：部署時以 `KNOWVA_PROFILE` 選定前端主題、預設文案、總覽關鍵字與產業段 prompt（見「產業 profile」） |

## 前置需求

- Docker（完整容器啟動只需要 Docker）
- 本機開發另需 [proto](https://moonrepo.dev/proto)；repository 的 `.prototools` 會安裝固定版本的 Rust、Node.js 與 npm
- 不使用容器轉檔時，主機需另裝 Pdfium 動態函式庫；可安裝在系統 library path，或以 `PDFIUM_LIBRARY_PATH` 指定檔案／目錄
- 不使用容器處理 DOCX/PPTX/XLSX 時，主機需另裝 LibreOffice headless 與所需 CJK 字型
- 一套地端推論服務,提供 OpenAI 相容的 chat/embeddings route 與 NVIDIA ranking route,並已載入上述三個固定模型
- 服務若需認證,準備一個 bearer token;不需認證時可完全不設
- 由實際端點確認過的 Gemma 4 model id

## 設定

先複製範例：

```bash
cp .env.example .env
```

三個模型都在地端服務上時，最少只需要 base URL 與 chat model id：

```dotenv
MODEL_BASE_URL=http://inference-host:8000
GEMMA_4_MODEL_ID=google/gemma-4-31b-it
```

chat 與兩個 Nemotron NIM 分屬不同主機時（地端常見），改成三個完整 URL，`MODEL_BASE_URL` 可留空：

```dotenv
MODEL_CHAT_URL=http://chat-host:8000/v1/chat/completions
MODEL_EMBED_URL=http://embed-host:8000/v1/embeddings
MODEL_RERANK_URL=http://rerank-host:8000/v1/ranking
GEMMA_4_MODEL_ID=google/gemma-4-31b-it
```

服務需要認證時另外填 `MODEL_API_TOKEN`，後端會以標準 `Authorization: Bearer` 送出；留空則完全不送認證 header。其他模型設定：

| 環境變數 | 必要性 | 說明 |
|---|---|---|
| `MODEL_BASE_URL` | 三個 `MODEL_*_URL` 未全填時必填 | 地端推論服務 base URL，預設 route 為 `v1/chat/completions`、`v1/embeddings`、`v1/ranking` |
| `MODEL_API_TOKEN` | 選填 | bearer token；留空代表服務不需認證，後端不送 `Authorization` |
| `MODEL_CHAT_URL` | 選填 | 完整 chat route；model 放在 JSON body；空值時由 base URL 推導 |
| `MODEL_EMBED_URL` | 選填 | 完整 embedding route，可獨立指向其他主機；空值時由 base URL 推導 |
| `MODEL_RERANK_URL` | 選填 | 完整 reranking route，可獨立指向其他主機；空值時由 base URL 推導 |
| `GEMMA_4_MODEL_ID` | 必填 | 放入 Chat request body 的 provider/model ID |
| `NEMOTRON_EMBED_MODEL_ID` | 有鎖定預設 | 只接受 `llama-nemotron-embed-vl-1b-v2` family |
| `NEMOTRON_RERANK_MODEL_ID` | 有鎖定預設 | 只接受 `llama-nemotron-rerank-vl-1b-v2` family |
| `RERANK_CANDIDATE_TOP_K` | `20` | dense 前幾筆送入 VL rerank，必須不小於 `RERANK_TOP_N` |
| `RERANK_TOP_N` | `6` | 精排後回答 context 數量，只允許 3–8 |
| `RERANK_BATCH_SIZE` | `8` | 每次 rerank request 的 passage 數量，範圍為 1–512 |
| `RERANK_ENABLED` | `true` | `false` 時不呼叫 rerank，直接把 dense 前 `RERANK_TOP_N` 筆給 Gemma；推論服務不載入 rerank 模型，記憶體較省但會漏掉部分來源（見 `docs/measurements/README.md` §8.16） |
| `MODEL_REQUEST_TIMEOUT_SECS` | 有預設 | 所有模型 HTTP/SSE request timeout，必須是正整數 |

Chat model、endpoint 或 rerank 設定無效時，後端會在連接資料庫與接受流量前停止並顯示可理解的錯誤。URL 只指定服務的 operation route，model 分別放在 JSON request body；因此換一套服務只需修改 URL，不需修改程式。

Embedding 仍要求 2048 維向量，Rerank 仍要求目前的 ranking response contract；切換到不相容的模型需要另外調整模型契約與重建索引。

文件前處理設定：

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `FILES_DIR` | `./data/files` | `STORAGE_BACKEND=local` 時原檔、頁圖、轉出 PDF 與 manifest 的根目錄 |
| `PDF_RENDER_DPI` | `180` | Pdfium rasterize DPI；允許 72–600 |
| `DOCUMENT_CONVERSION_TIMEOUT_SECS` | `120` | 每次 Pdfium 圖片處理或 LibreOffice 轉檔 timeout，必須為正整數 |
| `PDFIUM_LIBRARY_PATH` | 空 | 完整 library 路徑或所在目錄；空值時依序搜尋執行檔旁與系統 library path |
| `LIBREOFFICE_BIN` | `libreoffice` | LibreOffice executable 路徑或命令名 |

支援 PDF、PNG/JPEG/WebP、DOCX/PPTX/XLSX、MD/Markdown/TXT、STP/STEP 與 EML/MSG。STP/STEP 與 EML 會以文字內容建立檢索單元；MSG 保存原檔並建立檔案描述單元。MP4/AVI 目前不接受上傳，待 VLM 支援影片後再開放。CSV/HTML 與舊版 Office 格式目前不在支援清單。

### 檔案儲存後端（初始化時鎖定）

原檔、頁圖、轉出 PDF 與 manifest 可存放在本機檔案系統或 S3 相容物件儲存（AWS S3、Cloudflare R2、地端 MinIO），兩者走同一條程式路徑；未設定時與過去一樣使用本機 `FILES_DIR`。

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `STORAGE_BACKEND` | `local` | `local` 或 `s3`；其他值啟動即失敗 |
| `FILES_DIR` | `./data/files` | `local` 的根目錄 |
| `STORAGE_BUCKET` | 空 | `s3` 必填 |
| `STORAGE_PREFIX` | 空 | `s3` 選填的 key 前綴，例如 `tenant-a/prod` |
| `STORAGE_ENDPOINT` | 空 | R2／MinIO 等非 AWS endpoint |
| `STORAGE_REGION` | 空 | AWS region（預設 `us-east-1`）；R2 填 `auto` |
| `STORAGE_ACCESS_KEY_ID`、`STORAGE_SECRET_ACCESS_KEY` | 空 | 存取金鑰；不設時改用平台提供的認證：EC2 instance role，或由 ECS／EKS 等平台直接注入程序環境的標準 `AWS_*` 變數。以本 repo 的 docker compose 啟動時，主機上的 `AWS_*` 不會傳進容器，請填入金鑰或使用 EC2 instance role |
| `STORAGE_ALLOW_HTTP` | `false` | 僅限 HTTP endpoint（例如本機 MinIO） |
| `STORAGE_TIMEOUT` | `30s` | 單次請求逾時，上行慢時可調高 |

```dotenv
# 本機（預設）
STORAGE_BACKEND=local
FILES_DIR=./data/files

# AWS S3（IAM role 時省略 STORAGE_ACCESS_KEY_ID/STORAGE_SECRET_ACCESS_KEY）
STORAGE_BACKEND=s3
STORAGE_BUCKET=knowva-files
STORAGE_PREFIX=prod
STORAGE_REGION=ap-northeast-1

# Cloudflare R2
STORAGE_BACKEND=s3
STORAGE_BUCKET=knowva-files
STORAGE_REGION=auto
STORAGE_ENDPOINT=https://<account_id>.r2.cloudflarestorage.com
STORAGE_ACCESS_KEY_ID=<r2-access-key-id>
STORAGE_SECRET_ACCESS_KEY=<r2-secret-access-key>
```

- 設定在連資料庫前解析；`s3` 會先列出一次 bucket／prefix 驗證可存取，失敗即停止啟動。log 只記錄 backend、bucket、prefix 與 endpoint，不含 secret、簽章或 S3 錯誤內文。
- **初始化鎖定**：第一次對空資料庫啟動時，會把 backend（`s3` 另含 bucket 與 prefix）寫進 `storage_settings`。之後每次啟動都會比對，不一致即停止並顯示「儲存後端已於初始化時鎖定，變更需重新初始化 DB 與儲存空間」與已鎖定／目前設定。`FILES_DIR` 與 S3 endpoint 不列入鎖定，可更換掛載點或 endpoint。
- **正式部署（`deploy.yml`）**：儲存設定放在 repository 設定，每次部署都寫進主機 `.env`，確保與初始化時的鎖定一致。Variables：`STORAGE_BACKEND`、`STORAGE_BUCKET`、`STORAGE_PREFIX`、`STORAGE_ENDPOINT`、`STORAGE_REGION`；Secrets：`STORAGE_ACCESS_KEY_ID`、`STORAGE_SECRET_ACCESS_KEY`。未設定即為空值，`STORAGE_BACKEND` 未設定即為 `local`。`STORAGE_ALLOW_HTTP=false` 與 `STORAGE_TIMEOUT=120s` 寫死在 `docker-compose.prod.yml`。首次部署到空 DB 前就要設好；`s3` 缺 `STORAGE_BUCKET` 或值不合法時，workflow 會在碰主機前失敗。
- **已部署環境升級到此版需清空 DB 與檔案重新初始化**：DB 改存相對 key，不提供舊路徑資料轉換；資料庫已有文件但沒有鎖定紀錄時會拒絕啟動。
- Pdfium／LibreOffice 仍只能讀本機檔，每次轉檔都在系統暫存區（`TMPDIR`）下獨立的 `knowva-ingest-<run_id>` 目錄進行，不論成功或失敗都會刪除。
- evaluation 固定使用本機 `EVAL_FILES_DIR`，不受 `STORAGE_BACKEND` 影響；它的獨立 `EVAL_DATABASE_URL` 有自己的鎖定紀錄，與 production 互不影響。

### 產業 profile

同一個 image 出貨所有產業：產業差異是 profile（產業段 prompt、預設文案）加上前端主題，部署時以 `KNOWVA_PROFILE` 選定。一台機器只跑一個 profile。

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `KNOWVA_PROFILE` | `default` | profile id，也就是 `PROFILES_DIR` 下的目錄名；只允許小寫英數與 `-` |
| `PROFILES_DIR` | `../profiles` | profile 根目錄，相對於後端工作目錄（`backend/`）；image 內建所有 profile 於 `/app/profiles`，容器不需設定 |

每個 profile 是一個目錄：

```text
profiles/<id>/
├── profile.json   # 主題、品牌、頁面標題與說明、建議問題、輸入框提示、總覽關鍵字
└── prompt.md      # 產業段；可以是空檔
```

`profile.json` 的欄位全部必填，也不接受未知欄位；完整範例見 `profiles/default/profile.json`：

| 欄位 | 說明 |
|---|---|
| `theme` | 前端主題名稱，後端提供 `../frontend/dist/<theme>`；同一個主題可以給多個 profile 共用。只允許小寫英數與 `-` |
| `brand.name`、`brand.tagline` | 品牌名稱與標語 |
| `pages.chat`、`pages.documents` | 問答頁與文件頁的 `title` |
| `suggestions` | 建議問題 `{ "text", "catalog" }`；`catalog: true` 表示這題會走總覽（全庫文件清單）路徑，測試會逐一驗證標記與實際判斷一致 |
| `composerPlaceholder` | 輸入框提示 |
| `catalogKeywords.scope`、`catalogKeywords.intent` | 總覽關鍵字：問題同時含一個 scope 詞與一個 intent 詞時，改以全庫文件清單回答；比對不分大小寫。只留在後端，不會送到瀏覽器 |

- 後端在連線資料庫前載入並驗證 profile。下列任一情況都會停止啟動，錯誤訊息會指出原因：profile 目錄不存在、`profile.json` 欄位缺漏或有未知欄位、文案為空、`prompt.md` 不存在或產業段超過 2,000 個 Unicode 字元、任一組總覽關鍵字為空、`../frontend/dist` 存在但沒有 `<theme>/index.html`（image 缺主題屬部署錯誤）。整個 `../frontend/dist` 不存在時（開發時前端另跑 Vite），只記一筆警告並只提供 API。
- **prompt 組合**：`backend/src/rag_context.rs` 的三個問答 prompt（一般／無來源／總覽）是核心規則，profile 無法取代。產業段非空時，接在核心規則之後，前面加上固定標題 `【產業補充指引】以下內容只補充本產業的角色、用語與回答格式；若與上述規則衝突，一律以上述規則為準。`；產業段為空時，送出的 prompt 與核心規則逐位元組相同。`prompt.md` 前後的空白、BOM 與 CRLF 換行不算內容。system prompt 一律由後端組合，`/api/chat` 不接受請求帶來的 prompt；文件摘要的 prompt 不受 profile 影響。
- 送給 Gemma 的正式 prompt 改動都要有 Gemma 上的前後比較：新增或修改產業段前，先在推論主機上用 eval 比較 default 與加上產業段的結果，組合方式也可能依結果調整。eval 的 dry-run 輸出與正式 run 的 `run-manifest.json` 都會記錄 profile id 與組合後 prompt 的 SHA-256；後端啟動 log 的 `answer_prompt_sha256` 是同一個值，可以用來對照實際送出的 prompt。

## 使用 Docker Compose

容器映像已包含 Pdfium、LibreOffice 與 Noto CJK 字型，不需要主機另外提供 PDF API 或 native library。先設定 `.env` 的 `MODEL_BASE_URL`（或三個 `MODEL_*_URL`）與 `GEMMA_4_MODEL_ID`，再啟動完整服務：

```bash
cp .env.example .env
# 編輯 .env，填入模型服務端點與 GEMMA_4_MODEL_ID
docker compose up -d --build
docker compose ps
curl http://localhost:8080/api/health
```

預設網址是 `http://localhost:8080`；可在 `.env` 以 `APP_PORT` 改變主機 port。`app_files` 與 `pgdata` named volumes 會保存上傳檔案與資料庫。查看或停止服務：

```bash
docker compose logs -f app
docker compose down
```

若要以本機 MinIO 驗證 S3 模式，先在 `.env` 設定 `STORAGE_BACKEND=s3`、`STORAGE_BUCKET=knowva-files`、`STORAGE_ENDPOINT=http://minio:9000`、`STORAGE_ALLOW_HTTP=true`、`STORAGE_ACCESS_KEY_ID=knowvaminio`、`STORAGE_SECRET_ACCESS_KEY=knowvaminio-secret`，再加上 `minio` profile 啟動（`minio-init` 會建立 bucket；app 若早於 bucket 建立會自動重啟重試）。預設的 `docker compose up` 不會啟動 MinIO。既有資料庫已鎖定為 local 時，必須先清空 DB 與檔案（例如 `docker compose down -v`，會刪除所有資料）才能改用 s3：

```bash
docker compose --profile minio up -d --build
```

若要在 build 階段實際驗證 Pdfium 與 LibreOffice 對 PDF、掃描 PDF、DOCX、PPTX、XLSX 的分頁、頁序與每頁是否送圖：

```bash
docker build --target native-test -t knowva:native-test .
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
proto run npm -- run dev          # default 主題，http://localhost:5173，proxy /api 到 :8080
```

或建立 production bundle，交由後端服務：

```bash
cd frontend
proto run npm -- run build        # 先清空 frontend/dist/，每個主題輸出到 frontend/dist/<theme>/
```

後端提供部署 profile 指定的主題，也就是 `frontend/dist/<theme>/`（default 主題是 `frontend/dist/default/`）。`frontend/dist/` 存在但缺少該主題時後端不會啟動，所以舊版留下的平鋪 `frontend/dist/index.html` 要用根目錄的 `run build` 重建；各主題自己的 build 只清空自己的目錄，不會刪到其他主題。

`frontend/` 是 npm workspaces，lockfile 只有根目錄的 `frontend/package-lock.json`，指令一律在 `frontend/` 執行：

| Workspace | 內容 |
|---|---|
| `packages/core`（`@knowva/core`） | `BackendAdapter`、`useKnowvaRuntime`（問答 runtime 與對話紀錄）、`DocumentModel`、`useDocuments`（文件列表、輪詢、上傳、刪除）、`useProfile`（讀 `GET /api/profile`）、`useSourcePreview`／`useMessageSources`／`remarkSourceMarkers`（引用來源的預覽與內文 `[S1]` 標記）與安全化的錯誤訊息；不含 CSS 與版面。`@knowva/core/testing` 提供主題契約測試，`@knowva/core/vite` 提供擋外部網址的 build plugin |
| `themes/default`（`@knowva/theme-default`） | 現行介面：元件、CSS、`index.html` 與 `main.tsx`；輸出到 `frontend/dist/default/` |

根目錄的 `run lint`、`run test`、`run build` 依序跑過每個 workspace；`run dev` 與 `run preview` 啟動 default 主題，也可以用 `proto run npm -- run dev --workspace @knowva/theme-default` 指定。

主題不寫死品牌名稱與標語、頁面標題與說明、建議問題和輸入框提示，一律由 `useProfile()` 從 `GET /api/profile` 讀取。它回傳 `status` 為 `loading`、`ready`（附 `profile`）或 `error`（附 core 的安全化 `message`）的狀態，以及重新載入用的 `reload()`。default 主題在載入期間保留版面但不顯示文案；載入失敗時在頁首顯示該訊息與「重新載入」按鈕，問答與文件功能照常可用。載入完成後的畫面與改版前相同，分頁標題也改用 profile 的品牌名稱與標語。

每個主題的 vitest 都要呼叫 `describeThemeContract`，用主題自己的元件確認產品承諾：回答完成後列出每個來源的檔名、分頁來源另列頁碼；點來源清單的項目或內文的 `[S1]` 會讀出並顯示該單元的頁面圖或文字（總覽來源則是文件卡）、提供原檔下載連結，來源已不存在時顯示 core 的安全化訊息；SSE `error` 或連線失敗時顯示 core 的安全化訊息；串流途中可以停止生成；不支援的副檔名或超過 50 MiB 的檔案顯示 core 的訊息而且不送出請求；問答頁標題與建議問題來自 `/api/profile`；`/api/profile` 失敗時顯示 core 的安全化訊息，問答仍可使用；重新整理後對話與來源都還在、追問送出的 `previousSources` 和沒重新整理時相同；新對話、切換、刪除單一對話與畫面內確認的「清除全部對話」都反映在伺服器上；清單顯示後端用第一個問題前 30 個字元產生的標題，除了 `/api/chat` 之外不發模型請求；保存失敗時問答照常並顯示 core 的無法保存提示；對話清單載入失敗時顯示 core 的安全化訊息，問答照常。契約用記憶體裡的 `/api/conversations` 假後端代替伺服器，重新 render 就是重新整理。主題也要在 `:root` 宣告 `--theme-text`、`--theme-background`、`--theme-brand`、`--theme-on-brand`、`--theme-success`、`--theme-warning`、`--theme-error`，契約測試會檢查內文與三種狀態色在底色上、`on-brand` 在品牌色上的對比都達到 WCAG AA（4.5:1）。正式環境是離線 appliance，主題的 HTML、SVG 與 CSS 不能從其他主機載入字型、圖片或樣式，build 遇到這類 `src`、`href`、`url()`、`@import` 會直接失敗。

### 對話紀錄

問答頁的對話紀錄存在伺服器的 PostgreSQL（migration `0008_conversations.sql`），以瀏覽器 cookie 區分擁有者，每個瀏覽器最多 50 串。`/api/chat` 維持無狀態：回答完成後，前端另外呼叫 `/api/conversations` 保存每一則訊息。系統沒有登入，部署前請先了解這種做法的代價：

- **換瀏覽器或清除網站資料，就找不回原本的對話。** 擁有者是後端發給瀏覽器的 `knowva_client` cookie；換電腦、換瀏覽器或清除 cookie 之後就是另一個擁有者，也沒有跨裝置同步。原本的對話仍留在伺服器，直到被數量上限或刪除清掉。
- **共用電腦的同一個瀏覽器，會看到前一位使用者的紀錄。** 同一個瀏覽器設定檔共用同一個 cookie，誰打開問答頁都看得到之前的對話。共用的電腦請在離開前按「清除全部對話」，或讓每位使用者使用自己的瀏覽器設定檔。
- **能存取資料庫的人，讀得到問答內容。** 問題與回答以明文存在 `conversations` 與 `conversation_messages` 兩張表；資料庫只存 cookie token 的 SHA-256，不存 token 本身。

其餘行為：

- 上限：每個瀏覽器最多 50 串，建立第 51 串時刪除最久沒更新的那一串；每串最多 200 則訊息；單則訊息的 JSON 最多 256 KiB。超過時 API 回 4xx 與固定的安全化訊息，這串對話不再保存，問答照常。
- Cookie：對話 API 收到沒有 `knowva_client` cookie（或格式不對）的請求時，產生 256 位元的隨機 token，以 `HttpOnly`、`SameSite=Strict`、`Path=/api`、`Max-Age` 400 天的 cookie 回傳；請求經由 HTTPS 進來（直接 HTTPS 或 `X-Forwarded-Proto: https`）時再加 `Secure`。正式環境可能是純 HTTP 的內網，所以不一律加 `Secure`。log 與錯誤訊息不會出現 token。
- 隔離完全靠 token 保密。每個查詢都限定目前 cookie 的擁有者，存取不屬於自己或不存在的對話一律回 404，不透露那串對話是否存在。資料表以 `owner_kind`（`anonymous` 或 `user`，目前只寫入 `anonymous`）與 `owner_id` 表示擁有者，之後加入登入時可以用一個 `UPDATE` 把匿名對話轉給帳號。
- 標題由後端在保存第一則使用者訊息時產生：去掉頭尾空白、連續空白併成一個空格，再取前 30 個 Unicode 字元，不呼叫模型。
- 每則回答連同 `sources` data part 一起保存，串流中斷或出錯的回答也照樣保存（狀態為 `incomplete`）；重新整理後追問，送出的 `previousSources` 和沒重新整理時相同。重新整理後會回到最近更新的那串對話。
- 側欄的「近期對話」依最後更新時間排列，每串標示多久前更新；在較舊的對話裡發問後，那串會移到最上面。
- 保存失敗時問答照常，畫面顯示一行「這串對話無法保存」的提示。連線失敗、408、429 與 5xx 會在這串對話的下一則訊息前重送；其他 4xx 讓這串對話在這個頁面裡停止保存。對話清單載入失敗時，清單位置顯示 core 的安全化訊息與「重新載入」按鈕。
- 不做重新命名、封存、搜尋、匯出、自動過期與速率限制。

### 引用來源

回答內文的 `[S1]`、`[S2]` 與下方「資料來源」清單的每一項都可以點，點了會開啟對話框顯示模型引用的內容：頁面來源（PDF、Office、圖片）顯示那一頁的頁面圖，文字來源（Markdown、純文字）顯示那一段文字，總覽問答的來源顯示文件卡（檔名、摘要、標籤）；對話框底部可以下載原檔。內文中對不到來源的 `[Sn]` 維持純文字。

- 只讀得到問答目前會用到的內容：文件處理完成（`ready`）才讀得到，處理中、失敗或刪除中的文件一律當作不存在。文件刪除或重新處理後單元會換新，舊回答的來源會顯示「找不到這個來源，文件可能已刪除或重新處理。」
- 頁面圖就是入庫時轉出的那張圖，圖上不標示引用的位置。原檔只提供下載，不在瀏覽器裡開啟；回應帶 `Content-Security-Policy: sandbox`，上傳的內容不會在本站執行。
- 這些回應都帶 `Cache-Control: no-store`，瀏覽器不留快取，刪除的文件不會再從快取被看到。

## API

| Method | Path | 說明 |
|---|---|---|
| GET | `/api/health` | 健康檢查 |
| GET | `/api/profile` | 部署 profile 的介面欄位（主題、品牌、頁面標題與說明、建議問題、輸入框提示）；不含 prompt 與總覽關鍵字 |
| GET | `/api/documents` | 文件列表（含文件／artifact 狀態、頁數與檢索單元數） |
| POST | `/api/documents` | multipart 上傳（單一 `file` 欄位），立即回 id，背景處理 |
| DELETE | `/api/documents/{id}` | 刪除 `{id}/` 前綴下所有物件（原檔、轉出 PDF、頁圖、manifest）後才刪 DB row；清除失敗時保留 row，可重送 DELETE |
| GET | `/api/documents/{id}/file` | 下載原檔，串流回傳，`Content-Disposition: attachment`（中文檔名以 RFC 8187 編碼）；只限 `ready` 文件 |
| GET | `/api/documents/{id}/units/{unitId}/catalog` | 總覽問答引用的文件卡（`text/plain`，檔名、摘要、標籤），與模型拿到的相同；`unitId` 必須仍是該文件第一個可檢索的單元，文件重新處理後舊引用回 404 |
| GET | `/api/documents/{id}/units/{unitId}/image` | 來源單元的頁面圖；只限 `ready` 文件中可檢索的單元 |
| GET | `/api/documents/{id}/units/{unitId}/text` | 來源單元的文字（`text/plain`），與模型拿到的相同；條件同上 |
| POST | `/api/chat` | SSE 串流問答（`text-delta` / `sources` / `done` / `error`） |
| GET | `/api/conversations` | 這個瀏覽器的對話清單，依 `updatedAt` 新到舊 |
| POST | `/api/conversations` | 新增對話（201），id 由後端產生；回傳 `conversation` 與為了維持 50 串上限而刪除的 `removedIds` |
| DELETE | `/api/conversations` | 清除這個瀏覽器的全部對話（204） |
| GET | `/api/conversations/{id}` | 單串對話的 id、標題與時間 |
| DELETE | `/api/conversations/{id}` | 刪除一串（204） |
| GET | `/api/conversations/{id}/messages` | 依保存順序讀出 `{ parentId, message }` |
| POST | `/api/conversations/{id}/messages` | 保存一則 `{ parentId, message }`；同一個 `message.id` 重送時覆寫（冪等），並更新對話的 `updatedAt`；回傳 `{ conversation }` |

`GET /api/profile` 的形狀與 `profile.json` 相同，但多了 profile `id`，拿掉 `catalogKeywords`；產業段 prompt 與總覽關鍵字只留在後端。default profile 的回應如下：

```json
{
  "id": "default",
  "theme": "default",
  "brand": { "name": "Knowva 知星", "tagline": "點亮你的知識星系" },
  "pages": {
    "chat": { "title": "知識問答" },
    "documents": { "title": "文件管理" }
  },
  "suggestions": [
    { "text": "知識庫裡有哪些主題？", "catalog": true },
    { "text": "整理知識庫的重點", "catalog": false },
    { "text": "新人該先看哪幾份文件？", "catalog": false }
  ],
  "composerPlaceholder": "問問你的文件…"
}
```

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

`POST /api/chat` 的 request body 是 `{"messages": [...], "previousSources": [...]}`。`previousSources` 可省略，內容是上一則 assistant 訊息收到的 `sources` 事件原樣回送（只讀 `unitId`，其餘欄位忽略），用來決定本輪 context 區塊的排列順序；它不影響檢索到哪些來源，也不會把沒被檢索到的來源加回 context。

`/api/chat` 的純文字串流契約維持不變；有檢索來源時，`sources` 一定在 `done` 前送出。模型端 HTTP 錯誤包含 status、最多 500 字的安全 response snippet 與可重試狀態；timeout 也會以統一錯誤回傳。

對話 API 只作用在目前 `knowva_client` cookie 的擁有者身上，回應一律帶 `Cache-Control: no-store`；失敗時回傳固定的繁中純文字。`message` 是 assistant-ui 的完整訊息（JSON 物件，`role` 為 `user` 或 `assistant`），`parentId` 必須是這串對話裡已保存的訊息。保存第一個問題後的回應如下：

```json
{
  "conversation": {
    "id": "6f1c2a8e-3b7d-4c55-9a43-0d2e8f61b9a7",
    "title": "馬達多久保養一次？",
    "createdAt": "2026-10-06T08:00:00Z",
    "updatedAt": "2026-10-06T08:00:05Z"
  }
}
```

分頁 source 的 `pageNumber` 是 1-based；文字 source 的 `pageNumber` 固定為 JSON `null`，並以 0-based `unitIndex` 標識文字單元（UI 顯示時加一）。總覽問答（「知識庫裡有哪些主題？」這類問題）的 source 另帶 `"catalog": true`：模型拿到的是整份文件的文件卡（檔名、摘要、標籤），`unitId` 只是該文件的第一個單元，引用內容要用 `GET /api/documents/{id}/units/{unitId}/catalog` 讀；其他 source 不帶這個欄位。同文件的不同頁不會去重，完整 SSE 範例如下：

```text
data: {"type":"text-delta","text":"答案 [S1]"}

data: {"type":"sources","sources":[{"sourceId":"S1","documentId":"00000000-0000-0000-0000-000000000001","unitId":101,"unitIndex":1,"filename":"操作手冊.pdf","pageNumber":2,"kind":"page"},{"sourceId":"S2","documentId":"00000000-0000-0000-0000-000000000002","unitId":202,"unitIndex":0,"filename":"備註.md","pageNumber":null,"kind":"text"}]}

data: {"type":"done"}
```

## Artifact 契約

分頁文件的典型布局如下（根目錄為 `FILES_DIR`，或 `s3://STORAGE_BUCKET/STORAGE_PREFIX`）。每次處理都在本機暫存區轉檔，完成後把整組 artifact 上傳到新的 `artifacts-{run_id}/` 前綴（manifest 最後上傳）；DB transaction commit 是唯一的發布切換點，commit 後才清除舊 run。上傳或 commit 失敗會清掉本次前綴，DB 不會引用殘缺的 artifact set：

```text
{document_id}/
├── original.ext
└── artifacts-{run_id}/
    ├── converted.pdf       # 僅 Office
    ├── manifest.json
    └── pages/
        ├── 0001.png
        └── 0002.png
```

DB 欄位（`documents.storage_path`、`document_artifact_sets.*_path`／`artifact_dir`、`page_artifacts.image_path`、`document_units.image_path`）與 manifest 內容都存上述相對 key。讀取頁圖時 key 必須位於該文件的 `{document_id}/` 之下、不得含 `..` 或為絕對路徑，且在讀取內文前先檢查大小上限；錯誤訊息不含 key 或路徑。

`page_artifacts` 保存 1-based `page_number`、圖片 key、尺寸、可空的 PDF 文字層與 `needs_vision`；掃描頁沒有文字仍是有效頁面。極端 PDF MediaBox 仍依設定 DPI 縮放，但輸出單邊上限為 8192 px，避免 native bitmap 無界配置。MD/TXT 不製造假頁碼，改存 `text_artifacts`。artifact 發布完成後，每份文件最多同時處理四個 embedding unit；只有所有回應都恰為 2048 維 finite float、artifact 與 unit 計數一致，才在單一 transaction 內替換 `document_units` 並把文件設為 `ready`。任一 unit 失敗時已完成的 artifact 保持 `artifact_status=ready`，文件則為 `failed`，不會留下可搜尋的部分新結果。

頁圖只以記憶體中的 PNG/JPEG data URL 送出，不會傳送本機 artifact path；decoded image 上限 25 MiB、整體 embedding request 上限 36 MiB。Embedding 對 408、429、5xx、transport error 與 timeout 最多嘗試三次並使用 exponential backoff + jitter，其他 4xx 立即失敗。HTTP error snippet、API key、Authorization 與 base64 圖片資料都會在記錄前限長或遮蔽。

所有 embedding 完成後，系統會以同一個 Gemma 4 chat 端點 best-effort 產生繁中摘要與 3–6 個標籤：分頁文件固定只取前 3 頁（只有 `needs_vision` 頁附圖），MD/TXT 固定只取前 3 個文字單元；每個文字單元最多 3000 字，每頁 extracted text 最多 2000 字，頁圖沿用最長邊 1600 px、JPEG quality 80、單一 serialized image part 5 MiB 與完整 chat request 24 MiB 的界線。輸出必須是 typed JSON，摘要非空且最多 500 字，每個唯一標籤非空且最多 30 字；只容許單一 code fence 或單一 JSON object 前後 chatter 的 JSON-aware salvage。暫時性錯誤最多兩次 summary request，malformed JSON 不重試；任一 summary 失敗都只記安全 warning，仍會把完整 embeddings 與 `ready` 在同一 transaction 發布，不會重跑 embedding。

## 頁面是否送圖

地端推理的成本幾乎都在圖片：在 Mac mini M6 上 rerank 50 筆候選圖 + 文 12.9s、純文字 4.9s；Gemma 首字附頁圖 21–23s、純文字 7–8.5s（`docs/measurements/README.md` §8）。多數頁面的圖沒有文字層以外的資訊，所以每頁在 Pdfium 轉圖時決定一次 `needs_vision`，存進 `page_artifacts`、`document_units` 與 manifest（migration `0007_page_vision.sql`）。判斷是決定性的、不呼叫模型，PDF 與 Office 轉出的 PDF 用同一套規則，門檻是 `backend/src/page_vision.rs` 的常數。

下列任一成立即需要看圖，其餘頁面一律只送文字層：

- 有效的非空白文字少於 20 字，或 U+FFFD、控制字元、Private Use Area 字元超過 5%（掃描頁、文字轉外框、缺 ToUnicode 的字型）。前後都是空白的單一 U+F000–U+F0FF 字元是 Wingdings／Symbol 的項目符號或圖示，兩邊都不算。
- 除 Popup 與沒有外觀的 Link 以外的 annotation（印章、手寫、FreeText、表單欄位，以及帶 appearance stream 的 Link 會畫進頁圖但不在文字層）。
- 每張至少佔頁面 0.5% 的點陣圖合計達 3%（小 logo 不算）。
- 向量圖形合計達頁面 5% 或達 20 個：只算短邊至少 3pt、小於頁面 90%、不是近白色或透明的填色，以及含曲線、斜線或在旋轉矩陣下畫的筆畫；表格格線、外框、底線與整頁背景都不算。
- 上面兩項都不算頁面家具：投影片母片的色帶與邊條、信頭、頁尾圖這類每頁都畫一樣的圖。同一份文件中，內容相同（點陣圖比原始串流；向量圖比路徑、變換矩陣、填色與筆畫）且位置相同（四捨五入到 1pt）的圖，出現在至少 3 頁且至少一半的頁面，就是家具候選。內容與位置相同仍可能畫成不同的樣子（`/Decode`、遮罩、裁切、翻轉、stencil 填色、疊在上面的其他圖），所以有候選的頁面會再用 72 dpi 轉一次圖，比對候選範圍內的像素；這些頁面的所有文字框位置先一律塗掉，寫在家具上的頁尾文字才不會讓它變成內容。候選範圍有 95% 以上被塗掉時沒東西可比，照算。看起來也相同的頁面同樣要達到 3 頁且一半的門檻才算家具。每份文件的轉圖與比對最多 3 億像素、文字框最多 200,000 個，超過或轉圖失敗時照算。佔頁面一半以上的點陣圖不會被當成家具，因為重複的整頁背景可能就是內容本身（例如填了不同資料的掃描表單）。
- 不支援的物件、Pdfium 讀取錯誤、單頁超過 20,000 個物件或 form 巢狀超過 8 層。

面積以 CropBox 為分母，form XObject 內的物件會套上外層矩陣再計算。頁面家具要看完整份文件才知道，所以轉完所有頁才逐頁決定。不確定時一律偏向看圖：誤判成看圖只多一次推理，誤判成文字頁會丟資訊。圖片上傳一律看圖。

- 頁圖照常轉出與儲存（`image_path` 仍是 NOT NULL），只是文字頁不拿它推理，也不讀取它。
- 文字頁的 embedding 是 `passage/text`；rerank passage 只有 `text`；回答 context 的 block 沒有 `page_image:` 行與圖片 part；摘要只送文字。SSE `sources` 仍是 `kind: page` 加頁碼。
- Rerank 依「有圖／純文字」分組送 batch，每組內維持 dense 順序；推論服務對同一個 request 只用一種 processor，混合會讓文字 passage 也付圖片成本。
- 頁圖讀取失敗時，看圖頁照舊報錯，不會退回只送文字。
- migration 前的頁面都是 `needs_vision = true`（它們的向量是用頁圖建的），重新處理後才套用新規則；不需要重建索引。
- 入庫時每份文件記一行 info log（看圖頁數、文字頁數、原因），rerank 與回答也各記錄圖片與文字的筆數。

## 檢索契約

- 檢索只有 dense 一路：query 的固定 2048 維 embedding 走 ready-only partial HNSW `halfvec_cosine_ops` index，取 cosine 前 `RERANK_CANDIDATE_TOP_K` 筆（預設 20）。HNSW nearest-neighbor query 先 materialize，再以 distance、`unit_id` 做穩定排序。
- 不做 BM25、也不做 RRF 融合（migration `0009_dense_retrieval.sql` 移除了 `0004` 的 token 欄位、GIN index 與語料統計）。#60 量了 79 題（含 18 題型號、條號、人名等精確字詞題），沒有一題要靠 BM25 才取得到來源。只有圖的頁面、用中文問英文文件時，BM25 給不了分，這些 dense 找到的來源反而在 RRF 被往後擠。數據見 `docs/measurements/README.md` §8.16。
- 候選再以固定 `llama-nemotron-rerank-vl-1b-v2` 精排成 top 6（可設 3–8）。`RERANK_ENABLED=false` 時只取 dense 前 `RERANK_TOP_N` 筆，不呼叫 rerank。
- Page passage 只讀已發布 artifact：`needs_vision` 頁有文字層時送 image+text，掃描頁送 image-only（完全省略 `text`）；文字頁與 MD/TXT chunk 送 text-only，不讀頁圖。物件 key 與本機路徑都不進 payload，圖片以有上限的並行讀取後只以記憶體 data URL 傳送；候選先依有圖／純文字分組，每組依 dense 順序每批預設 8 筆，API 回應即使重排順序也依每批 local index 映回原 unit。
- 每筆 `RetrievalCandidate` 直接帶 `unit_id`、0-based `unit_index`、`document_id`、`filename`、`kind`、1-based nullable `page_number`、nullable `image_path/content`、`needs_vision`、dense rank/score 與 nullable `rerank_score/rerank_rank`，後續階段不需重新查來源 metadata。
- Rerank 每批都嚴格驗證回應筆數、index 唯一且不越界、score finite，再驗證跨批映射。408、429、5xx、transport error 與 timeout 最多嘗試三次；耗盡後以 dense 前 top_n 筆原序回退，且 rerank 欄位保持 null。設定、artifact 或模型服務 response contract 錯誤不回退，會回傳可診斷錯誤。Structured log 僅記錄各階段筆數、延遲、fallback 與 model id，不含 query、文件內容、secret 或 base64。
- context 區塊採追加式排列：`POST /api/chat` 的 `previousSources` 帶回上一輪答案的來源與順序，本輪仍檢索到的來源就留在原位、原順序，本輪新進的來源再依排名接在尾端。推論服務的 prompt cache 是前綴比對，唯有這種排列能把「追問檢索到重疊來源」穩定換成快取命中；來源完全沒變時整段 prompt（連同它後面的對話歷史）都會命中。沒有 `previousSources` 時（第一個問題、catalog、文件摘要）就直接照排名排列——最相關的來源最可能在下一輪續留，把它放在最前面，下一輪能重用的前綴才最長。`[S1]`..`[Sn]` 依版面位置連號並與 SSE `sources` 事件一致；payload 或完整 request 超限時仍是從排名最低的來源開始整筆移除，再重新編號。

## A/B 評測

評測資料集位於 `backend/eval/fixtures/manifest.json`。在已安裝 proto 的全新 checkout，從 repository root 執行下列命令即可做離線 dry-run；若固定版 Rust 尚未安裝，先執行 `proto install`：

```bash
cd backend && proto exec rust -- cargo run -- eval --dry-run
```

dry-run 只讀取並驗證 fixture 與 profile，不連資料庫、不載入 native converter，也不呼叫任何模型端點。profile 依 `KNOWVA_PROFILE`／`PROFILES_DIR` 選定，和正式 run 一樣會讀 `.env`；輸出的 JSON 含 `profileId` 與 `systemPromptSha256`（回答時送出的組合後 system prompt 的 SHA-256），比較產業段前後時可以先確認兩次用的是不同 prompt。fixture 方面，它會檢查 dataset schema/version、未知欄位、唯一 document/query/unit id、安全相對路徑、檔案類型與副檔名、SHA-256、非空 license/source、預期 page/unit inventory，以及 query scope、expected source、required facts、answerable/not-found 等契約。所有內附文件都是 deterministic synthetic CC0 fixture，不含客戶或正式環境內容；涵蓋繁中原生 PDF、純影像掃描 PDF、PPTX、表格／圖表 PDF、PNG、Markdown 與純文字，也涵蓋同名文件隔離、相鄰頁干擾、多頁來源、無頁碼文字與 visual-only facts。完整來源與 coverage 見 [backend/eval/fixtures/README.md](backend/eval/fixtures/README.md)。

可重複使用 `--document <document_key>` 與 `--query <query_id>` 選 subset；指定 document 時，只會選取其 `document_scope` 全部落在所選 documents 的 queries。dry-run 的 JSON 會列出實際選到的 documents/queries，未知或空 subset 會 fail closed。例如：

```bash
cd backend
proto exec rust -- cargo run -- eval --dry-run \
  --document markdown-operations \
  --query markdown-escalation
```

### 正式 A/B run（`--cloud`）

正式 run 是明確 opt-in，會 ingest fixtures 並實際呼叫設定好的 embedding、rerank 與 generation endpoint。端點是雲端服務時會產生費用，是地端服務時會佔用推論主機；CI 與一般測試絕不自動執行。A 是 dense-only control，直接取 dense 前 top_n 筆；B 是正式流程，共用 A 的同一批 dense candidates，再以固定 VL reranker 精排。兩組固定使用相同 embedding 與 Gemma 4 generation model id、相同 prompt version、相同 system prompt（選定 profile 的一般問答 prompt，與 `/api/chat` 用同一個組合函式產生）、相同 query、相同 context top_n 與 deterministic retry policy，只有 retrieval path 不同。provider 不支援 seed 時會在 run manifest 明確記為 false。為確保 telemetry 可歸因，目前正式 run 強制 `--concurrency 1`。

執行前必須完成下列事項：

1. 準備一份本機門檻 JSON，寫下 correct-page recall 的 stage、k、minimum，citation precision／recall、answer facts recall，以及 B 相對 A improvement 的數值。所有 minimum 必須是 0 到 1 的 JSON number，k 必須是正整數，`version` 是自訂的門檻版本名稱。報告的 `run-manifest.json` 會記錄這份檔案的 `version` 與 SHA-256。要比較的幾次 run 請用同一份檔案，並在執行前就決定好數值，不要看完結果才調整。下列只是 schema 示意，尖括號欄位必須換成實際數值：

   ```text
   {
     "schema_version": "2.0",
     "version": "<threshold-version>",
     "minimums": {
       "correct_page_recall_at_k": {
         "stage": "<dense|rerank>",
         "k": <positive-integer>,
         "minimum": <number-0-to-1>
       },
       "citation_precision": <number-0-to-1>,
       "citation_recall": <number-0-to-1>,
       "answer_facts_recall": <number-0-to-1>,
       "b_relative_improvement": <number-0-to-1>
     }
   }
   ```

2. 由 PostgreSQL 管理者建立一個名稱以 `knowva_eval_` 開頭的全新空 database，以及名稱可辨識用途的專用登入角色（例如組織自訂的 `knowva_eval_runner_<識別碼>`；尖括號內容必須換成真實且唯一的值）。以該角色作為 database owner，讓它能執行 migration；接著在任何 migration 前，以 database owner 連線並執行 `COMMENT ON DATABASE <實際的 knowva_eval_ database 名稱> IS 'knowva-evaluation-only:v1';`。最後把使用該專用角色與 database 的連線字串設為 `EVAL_DATABASE_URL`。runner 會檢查 current database 名稱、connect role 身分與精確 marker；它也會拒絕 production `DATABASE_URL`、repository 預設 DB，以及 migration 後含有任何 document 的 DB。不要把 placeholder、示意密碼或假 marker 當成正式設定。
3. 準備專用且為空的 `EVAL_FILES_DIR`。它不可等於、包含或位於 production `FILES_DIR` 或預設 `./data/files` 之下；失敗後會保留完整隔離 run 供診斷，下次不可重用非空目錄。
4. 設定地端推論服務與確認過的 Gemma 4 model id：至少提供 `MODEL_BASE_URL` 與 `GEMMA_4_MODEL_ID`；各能力分屬不同主機時改設 `MODEL_CHAT_URL`、`MODEL_EMBED_URL`、`MODEL_RERANK_URL`，服務需認證時另設 `MODEL_API_TOKEN`。確認服務已載入固定的 Nemotron embedding/rerank models。
5. 主機安裝 Pdfium 與 LibreOffice/CJK 字型，並以 `PDFIUM_LIBRARY_PATH`、`LIBREOFFICE_BIN` 指向可用 binary/library，讓 PDF、掃描頁與 PPTX fixtures 能完整 ingest。

完成上述前置作業後，從 `backend/` 以單一 `eval --cloud` command 執行完整資料集；以下所有尖括號都必須替換為真實的隔離環境設定：

```bash
cd backend
EVAL_DATABASE_URL='postgres://<eval-user>:<password>@<host>:<port>/<brand-new-empty-eval-db>' \
EVAL_FILES_DIR='/absolute/path/to/empty/eval-files' \
MODEL_BASE_URL='http://<inference-host>:<port>' \
MODEL_API_TOKEN='<optional-bearer-token>' \
GEMMA_4_MODEL_ID='<confirmed-gemma-4-model-id>' \
PDFIUM_LIBRARY_PATH='/absolute/path/to/libpdfium.so' \
LIBREOFFICE_BIN='/absolute/path/to/libreoffice' \
proto exec rust -- cargo run -- eval --cloud \
  --manifest eval/fixtures/manifest.json \
  --thresholds /absolute/path/to/local/thresholds.json \
  --output /absolute/path/to/eval-results \
  --concurrency 1
```

正式 run 也接受上述 `--document` / `--query` subset flags。成功寫檔後，stdout 會回傳 `passed` 與路徑；每次輸出位於 `<output>/<run_id>/report.json`、`report.csv`、`report.md`、`run-manifest.json`。`passed: true` 只有在至少一組有效 A/B pair，且 correct-page、citation precision/recall、answer facts recall、B 相對 A improvement 全部達到門檻檔的數值時成立；command 本身完成不代表門檻通過。`report.*` 會列出各 query 的 improvement/same/regression/failure、分數與 failure reasons，`run-manifest.json`（schema `1.1`）則保存 locked model ids、prompt version、profile id（`profile_id`）與組合後 system prompt 的 SHA-256（`system_prompt_sha256`）、top_n、dataset/threshold hash、git revision、subsets、retry/latency/request/fallback telemetry 與安全化失敗案例。preflight、native conversion、模型服務 contract 或 DB isolation 錯誤會以非零狀態停止；已開始寫入後不會局部清掉 DB/files，以免破壞診斷一致性。

Threshold JSON 只應保存在本機安全位置：不得提交本機 threshold 檔、API token、`.env` secret 或任何 customer/production 文件。fixture 只能使用 repository 內已審核、標明來源與 license 的 synthetic files。

## CI 與快取

`ci.yml` 的三個檢查 job 前面先跑一個 `changes` job，用改動路徑決定哪幾個要跑：`backend/**` 觸發 backend 與 native-conversion，`frontend/**` 觸發 frontend，`profiles/**` 觸發 backend 與 frontend，`Dockerfile` 與 `.dockerignore` 觸發 native-conversion，`.prototools` 與 workflow 本身觸發全部。只改文件的 PR 因此只花一個短 job。

只有 pull request 會過濾。那是 paths-filter 能直接從 API 回答的事件，不需要 checkout 也不需要 history；要過濾 push 就得在每一次 run 完整 fetch 整個 repo，只為了省下「每次合併一次」而且 PR 已經檢查過的那一趟。所以 push 到 main 與手動 dispatch 一律全跑。

只有在 filter 確實執行且成功時才採信它的輸出。一個寫出部分 output 之後才失敗的 step，否則會被照單全收；而一個無法判斷改了什麼的 filter，絕不能有權決定「什麼都不用檢查」。它出錯的每一種方式（包含被 skip）都會退回這個 workflow 原本的全跑。

快取有兩條。cargo 這條由 `Swatinem/rust-cache` 帶走 cargo registry 與 `backend/target`，clippy 與測試不必每次從零編譯；Docker layer 這條走 buildx 的 GitHub Actions cache，`native-conversion` 用 `scope=native-test`、`release.yml` 用 `scope=release`，兩者不會互相驅逐。

兩條都只在 `main` 寫入。GitHub 的 cache 依分支隔離：在 PR 分支寫的 entry 只有那個分支讀得到，若每個 PR 都寫入，只會用滿 repository 的配額並擠掉所有 PR 共用的那一份 main entry。PR 一律讀 main 的快取。

Dockerfile 把相依圖編在自己的 layer：`backend-builder` 與 `native-test` 都先以 placeholder main 編一次相依，再蓋上真正的 source。registry cache 帶得動 layer，帶不動 cache mount，所以相依改放 layer 之後才真正被快取住——只改 source 時只重編 `knowva-server`，改 `Cargo.lock` 才會重編相依。placeholder 自己的 fingerprint 與產物在同一層刪掉，cargo 不會把 stub 當成最新的 `knowva-server` 直接送進 runtime。

`native-test` 從 `backend-base` 分支出來而不是從 `backend-builder`：ignored test 跑的是 test profile，用不到 release binary，繼承它只是把整個 crate 編兩次。代價是 CI 不再順便驗證 release build 編得過，那要等 tag 觸發 `release.yml`，與「未經審查的 tag 只賠掉一次 build」這個既有取捨一致。

`backend/Cargo.toml` 的 `[profile.dev] debug = "line-tables-only"` 只影響 dev 與 test profile：panic backtrace 仍有檔名與行號，但不再寫入與連結完整 debug info，測試 build 與被快取的 target 目錄都因此小很多。release profile 不受影響。

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

一般測試以本機 mock 模型端點驗證四種 embedding payload、三種 rerank modality、依 modality 分組的 batch、跨批映射、嚴格 response contract、有限重試／fallback 與 transaction/DB 契約，不會呼叫真實的模型服務。安裝 Pdfium 與 LibreOffice 後，可明確執行 native fixture smoke；它也會確認 Pdfium 的 image+text 與 image-only embedding payload，並逐頁檢查 `needs_vision` 判斷（含 `example_file/iiot_industry_report.pdf` 只有第 7 頁看圖）：

```bash
cd backend
PDFIUM_LIBRARY_PATH=/path/to/libpdfium.so \
LIBREOFFICE_BIN=libreoffice \
proto exec rust -- cargo test -- --ignored --exact --nocapture --test-threads=1 \
  page_artifacts::tests::generated_pdf_and_office_fixtures_preserve_page_order \
  page_artifacts::tests::page_vision_decisions_match_fixtures_and_the_example_report \
  ingest::tests::native_conversion_runs_in_scratch_and_publishes_relative_keys
```

fixture 的來源、預期頁數與涵蓋範圍見 [backend/tests/fixtures/README.md](backend/tests/fixtures/README.md)。

一般測試已用本機目錄與 in-memory store 跑完上傳 → ready → 問答（頁圖 rerank 與 RAG）→ 刪除。S3 模式另有 opt-in 的 MinIO smoke：它以真的 MinIO 跑同一條流程並確認刪除後文件前綴為空，同時驗證錯誤帳密與不存在的 bucket 會失敗且錯誤訊息不含 secret 或簽章；模型一律走本機 mock endpoint，不需模型服務憑證，資料庫沿用 `.env` 的 `DATABASE_URL`（`sqlx::test` 會建立並刪除獨立的測試 DB）：

```bash
docker compose --profile minio run --rm minio-init   # 啟動 MinIO，建好 bucket 才返回
cd backend
STORAGE_BACKEND=s3 STORAGE_BUCKET=knowva-files STORAGE_PREFIX=smoke \
STORAGE_ENDPOINT=http://localhost:9000 STORAGE_ALLOW_HTTP=true STORAGE_REGION=us-east-1 \
STORAGE_ACCESS_KEY_ID=knowvaminio STORAGE_SECRET_ACCESS_KEY=knowvaminio-secret \
proto exec rust -- cargo test minio_smoke -- --ignored --nocapture
cd ..
docker compose --profile minio rm -sfv minio
```

若要選擇性驗證真實 Embed VL endpoint，先在 `.env` 填妥 `MODEL_BASE_URL`（或完整的 `MODEL_EMBED_URL`）與 `GEMMA_4_MODEL_ID`，服務需認證時另填 `MODEL_API_TOKEN`，再明確執行下列 ignored test。它會各送一次 text query、text passage、image passage 與 image+text passage（共四次真實呼叫）；一般 `cargo test` 與 CI 不會執行：

```bash
cd backend
proto exec rust -- cargo test \
  cloud_models::tests::real_nemotron_embed_vl_smoke_covers_all_input_modalities \
  -- --ignored --exact --nocapture
```

詳細 MVP 規劃見 [docs/mvp-now.md](docs/mvp-now.md)。
