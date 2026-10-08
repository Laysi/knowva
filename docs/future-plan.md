# 未來方案(正式地端產品)

> 總覽與不變原則見 [README.md](README.md);目前進行中的工作見 [mvp-now.md](mvp-now.md)。

**目標:同一套 Knowva Core 完全離線落在客戶內網。客戶用「買機器的預算」決定落點;軟體、pipeline、資料格式全落點相同。**

## 1. 統一模型棧

| 層級 | 模型 / 元件 | 設定 | 理由 |
| --- | --- | --- | --- |
| LLM / VLM(生成) | Google Gemma 4 family | 入門硬體:12B;高階硬體:31B | 同一模型家族;多模態 — 回答時直讀來源頁圖,另供摘要與選配轉錄。 |
| Embedding(文字+視覺) | NVIDIA llama-nemotron-embed-vl-1b-v2(約 1.7B:Llama 3.2 1B + SigLIP2 400M) | 全面固定;2048 維單向量 | 頁圖、文字、「圖+文」嵌入同一向量空間 — 檢索不依賴轉錄;多語(MIRACL/MLQA);NVIDIA Open Model License 可商用。 |
| Rerank(文字+視覺) | NVIDIA llama-nemotron-rerank-vl-1b-v2(約 1.7B,cross-encoder) | 全面固定;低階機降候選數 | 單一 reranker 同時精排文字 chunk 與頁面候選(官方評測 recall@5 約 +6~7 個百分點);純文字 fallback:llama-nemotron-rerank-1b-v2。 |
| 文件頁面化 | pdfium(Rust 原生)+ LibreOffice headless(Office→PDF,含 CJK 字型) | 全面固定 | 前處理的頁轉圖基礎;無 Python 依賴。 |
| 推理 runtime | vLLM 優先(Embed/Rerank VL 僅 vLLM 0.17+ / NIM 可服務化);llama.cpp 系僅作 LLM fallback | 全面固定;全線 CUDA 容器 | 全容器、一套部署法;單一 runtime 服務三顆模型最簡;NIM 因授權成本僅列選配。詳見 [deployment.md](deployment.md)。 |
| Vector DB | PostgreSQL + pgvector | 全面固定;2048 維以 `halfvec(2048)` 建 HNSW(vector 型別索引上限 2000 維) | Rust / Postgres / pgvector 對 x86 與 Arm Linux 部署皆成熟。 |
| 原檔保存 | 本機檔案系統或 S3-compatible storage | 全面固定 | 保留原檔與頁圖,支援重新解析與重建索引。 |

## 2. 硬體落點(客戶採購參考)

**預算級距指客戶「買機器的錢」**,與軟體無關 — 軟體只有一套,三個 profile 跑的是同一系統。硬體價格依市況浮動,級距僅供採購溝通。

| Profile | 硬體預算級距 | 建議硬體 | LLM 設定 | 建議 context | 定位 |
| --- | --- | --- | --- | --- | --- |
| Basic Local | 約 10 萬 | x86 主機 + RTX 5060 Ti 16GB ×2(合計 32GB) | Gemma 4 12B Q4 獨占卡 0(Q8 視實測;31B 不承諾);Embed/Rerank VL 獨占卡 1(FP16 免量化) | 16K-32K | 低併發部門知識庫、文件問答、最低可交付自部署;入庫批次與互動問答分卡並行。 |
| Enterprise Local | 約 20 萬 | DGX Spark 128GB(容量型;GB10 迷你機如 ASUS GX10 實售 17 萬+,同級替代)或 x86 + RTX 5090 32GB(速度型);96GB 工作站另估 | Spark:12B Q8 常駐 + 31B Q4/Q8 品質模式;5090:12B 極速(31B Q4 勉強、不承諾) | 32K-64K+ | 長文件、多模態批次解析、31B 品質模式或極速問答。 |

**x86 主機 + RTX 5060 Ti 16GB ×2(最低可交付規格,約 10 萬)**

- 雙卡分工:**卡 0 全給 Gemma 4 12B Q4**(權重約 7-8GB + KV cache 與頁圖 vision tokens,16K-32K context;獨占 16GB 後 12B Q8 約 13GB 有機會,視實測)、**卡 1 跑 Embed VL + Rerank VL**(FP16 各約 3.5GB,免量化,餘裕留給批次頁面嵌入)。retriever 不再與 LLM 搶 VRAM,大量文件入庫與互動問答互不干擾。
- 5060 Ti 16GB 單卡行情約 1.4-1.7 萬,雙卡加整機仍壓在 10 萬內;單卡頻寬 448GB/s 低於 5070 Ti/5080,但仍遠高於統一記憶體方案,12B Q4 互動問答體感佳;單卡 180W,一般塔式機殼 + 750W 電源即可,主機板需雙 PCIe 槽(x8/x8 即可 — 模型分卡部署,卡間溝通極少)。
- 合計 32GB 理論可測 31B Q4(vLLM tensor parallel ×2),但受 PCIe 與頻寬限制,不承諾、不作賣點 — 31B 仍屬 Enterprise 落點。
- 消費卡的保固與採購觀感需評估;偏好單卡時 RTX PRO 4000 24GB(觀感/保固較佳)或 5070 Ti/5080 16GB 為替代,但三顆模型同卡即回到量化 + 降候選的緊配置。
- retriever 資源不足時先降候選數(rerank 候選預設 dense 前 20、rerank top_n 3-8 頁),再考慮暫關 rerank(`RERANK_ENABLED=false`),不改 API。
- **工控選配:Jetson AGX Orin 64GB(約 6-8 萬)** — 低功耗/無風扇/長供貨,64GB 記憶體充足,但算力弱:RAG 長 prompt 的首字延遲預估 15-30 秒,僅適合批次解析或可容忍延遲的場景,不作互動式預設;需 Jetson 專用(l4t)容器 image。
- macOS 不在標準支援清單;個別客戶堅持 Mac 時另案評估,不列 SKU。

**DGX Spark 128GB(高階地端 appliance,約 20 萬)**

- 價值在 128GB unified memory、NVIDIA AI stack 與地端產品機形態;頻寬不等同高階 GDDR/HBM,不以最高 token/s 作賣點。
- 適合 Gemma 4 31B、長上下文、批次頁面嵌入與解析、rerank、Postgres/pgvector 與 worker 同機部署。
- 適合高隱私、內網、製造業文件場景;若單純追求單模型速度,高階 RTX / RTX PRO 工作站可能更快。
- 同級替代:GB10 合作夥伴迷你機(ASUS GX10 等,實售 17 萬+,規格同級);重視 12B 速度而非 31B/長上下文時,x86 + RTX 5090 32GB(單卡 15 萬+,整機約 20 萬)為速度型替代 — 5090 的 32GB 在單卡上、頻寬約 1.8TB/s,與 Basic 雙 5060 Ti 的「合計 32GB」不同級,單模型上限與速度都更高。

## 3. 完整 Pipeline(標準流程)

| 步驟 | 內容 |
| --- | --- |
| 1. 原檔保存 | PDF、DOCX、PPTX、PNG、JPG、email 等原檔保存到本機 storage(未來 S3-compatible)。 |
| 2. 頁面化 | Office 檔經 LibreOffice headless 轉 PDF;PDF 以 pdfium 逐頁轉圖(約 150-200 DPI,PNG/WebP);圖片檔視為單頁;md/txt 不轉圖,依標題感知切 chunk。 |
| 3. 文字層(加值,非關鍵路徑) | PDF 內嵌文字直接抽出;掃描頁選配 OCR / VLM 轉錄。供文字頁嵌入、引用片段、摘要與「圖+文」嵌入;缺文字層不阻擋入庫。 |
| 4. 標準化輸出 | 每頁輸出 document_id、page、content(可後補)、table、figure、bbox、source_hash、confidence。 |
| 5. 建立索引 | Embed VL 產生向量:頁面用「圖+文」(有文字層)或純圖嵌入,md/txt chunk 用文字嵌入 — 同一 2048 維空間;Postgres/pgvector(`halfvec` + HNSW)保存 vector 與 metadata。 |
| 6. 查詢召回 | query 文字嵌入(同一空間)dense 召回前 20 筆。原本併 BM25 再做 RRF,#60 量測後拿掉(見 `docs/measurements/README.md` §8.16)。 |
| 7. Rerank 精排 | Rerank VL 對候選頁/chunk 精排,取 top 3-8 作為回答 context。 |
| 8. 回答與引用 | Gemma 4 直讀來源頁圖(+文字片段)產生回答,附文件名、頁碼、原文片段或 chunk id。 |

## 4. 部署架構

部署不是未來項目,從 MVP 就生效:**全線 NVIDIA + Linux、整套系統(含模型)容器化、一份 compose 通用全落點**。完整內容獨立成 [deployment.md](deployment.md)。

## 5. 從 MVP 到正式交付的升級路徑

模型從 MVP 第一天就與交付一致(見 [mvp-now.md](mvp-now.md) §2),所以升級主要是「換執行位置、加模組」:

| 項目 | 動作 | 是否重建索引 |
| --- | --- | --- |
| LLM:雲端 Gemma → 地端 Gemma | 換推理端點(雲端 API → 本地 runtime);模型相同,prompt 與 SSE 協定不動 | 否 |
| Embedding | MVP 起即固定 llama-nemotron-embed-vl-1b-v2,交付時不用換、不用重建。例外是現行 demo(Titan / NVIDIA 文字 embed):切換當下需一次性 schema migration(`vector(1024)` → `halfvec(2048)`)+ 重跑前處理(向量空間不同,整庫重嵌) | 原則上否(demo 汰換時一次) |
| 檢索:加 Rerank VL | 加 Rerank VL 服務,既有向量照用(BM25 + RRF 曾加入,#60 量測後拿掉) | 否 |
| 前處理:VLM 直讀轉錄 → 頁面化 + VL 嵌入 | 重建 ingestion 前段(轉檔 + 轉圖 + 嵌入);之後 parser/OCR 只是文字層加值的升級,不再動檢索主路徑 | 一次(原檔都在,重跑前處理即可) |
| Schema:版本欄位 | `chunks` 加 `embedding_model` / `embedding_version` 欄位與 migration 流程,支援日後 embedding 大版本升級 | — |

## 6. 風險與待驗證項目

| 風險 | 處理方式 |
| --- | --- |
| 推理容器與 arm64 | vLLM 容器在 GB10(arm64/sbsa)的 image 支援與效能需實測;Embed/Rerank VL 的 vLLM(0.17+)服務化與 arm64 支援需逐一驗證,fallback 為 NIM(授權成本)或 transformers 自建輕量服務(同一份 compose)。 |
| Basic 雙卡分工 | LLM 獨占的卡 0 仍只有 16GB:12B Q4 承諾、Q8 視實測、31B 不承諾(合計 32GB 的 TP=2 不作賣點);頁圖回答的 KV/vision tokens 需實測。各服務綁卡(compose `device_ids`)是新增部署細節,一份 compose 仍通用;不足時升級 Enterprise。 |
| Jetson Orin 選配的 TTFT | 若採 Orin,RAG 長 prompt 首字延遲需實測(預估 15-30 秒);SLA 須按批次/可容忍延遲場景撰寫,並驗證 l4t 容器變體。 |
| DGX Spark 31B 速度 | 31B 可用但不一定快;定位為高品質模式或批次解析,不一定作所有即時問答預設。 |
| 前處理成本 | 頁面化後關鍵路徑只剩 1.7B 嵌入,較 VLM 全文轉錄大降;但「圖+文」最佳嵌入模式依賴文字層 — 掃描件選配 OCR 的品質/成本、與純圖嵌入的命中率差距需實測。 |
| 繁中視覺檢索品質 | Embed/Rerank VL 官方評測(ViDoRe 系列)以英文文件為主;模型標多語(MIRACL/MLQA),繁中掃描件/表格/簡報需自建測集實測,不達標時以「圖+文」嵌入與 OCR 文字層補強。 |
| 頁圖 context 的延遲與 VRAM | 回答夾帶 3-6 張頁圖對 Gemma TTFT 與 VRAM 的影響需在 Basic 落點實測;不達標時降 top_n 或改送文字層+關鍵單頁。 |
| halfvec(2048) | 2048 維超過 pgvector HNSW 對 vector 型別的 2000 維上限,以 `halfvec` 建索引(pg17 image 已支援)需驗證召回與效能;模型未標 Matryoshka,不假設可截斷降維。 |
| 頁圖儲存 | 每頁約 100-300KB(WebP 可再壓),百頁文件約 10-30MB;隨頁數線性成長,可接受但需納入容量規劃。 |
| Embedding 版本升級 | 升級 embedding 必然 re-index;以版本欄位 + migration 流程管理(見 §5)。 |
| 非中資聲明 | 以模型家族與供應商所在地作實務篩選;不承諾逐層股權完全無中資,除非另做法務/採購調查。 |
