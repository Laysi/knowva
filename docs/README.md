# Knowva 知星 冷啟動規劃(總覽)

> **系統定位:本產品不是 SaaS。** 它是部署在「單一企業內網」的簡易知識管理系統(文件上傳 → 建立知識庫 → RAG 問答附來源)。一套系統只服務一個客戶,沒有多租戶、沒有雲端帳號體系;軟體只有一套,產業差異是 profile(產業段 prompt、預設文案)加上主題,由部署時的 `KNOWVA_PROFILE` 選定,其餘差異只在客戶採購的硬體與掛載的模型尺寸。
>
> 文中的 **10 萬 / 20 萬指「客戶買機器的預算」(新台幣)**:10 萬級 = x86 + RTX 5060 Ti 16GB ×2 主機(合計 32GB)、20 萬級 = DGX Spark 128GB 或 RTX 5090 主機(市場行情:5060 Ti 16GB 單卡約 1.4-1.7 萬、GB10 迷你機實售 17 萬+、RTX 5090 單卡 15 萬+),與 MVP / 正式產品的階段無關。**產品線全部落在 NVIDIA + Linux(macOS 不在標準支援清單)**,整套系統含模型皆可容器化,一套部署法通用全落點。

## 文件結構

| 文件 | 內容 |
| --- | --- |
| [mvp-now.md](mvp-now.md) | **現在做(MVP)**:範圍、模型策略(雲端也用與交付相同的模型)、現況差距、驗證問題、本次系統調整。 |
| [future-plan.md](future-plan.md) | **未來方案(正式地端產品)**:統一模型棧、硬體落點與預算級距、完整 pipeline、升級路徑、風險。 |
| [deployment.md](deployment.md) | **部署架構(從 MVP 就生效)**:全容器 compose 拓撲、推理 runtime 選型、離線交付、CPU 架構。 |

## 一頁摘要

| 面向 | 現在做(MVP) | 未來方案(正式地端產品) |
| --- | --- | --- |
| 交付形態 | 單一租戶、單機部署;開發與對內測試可走雲端 | 完全離線,落在客戶內網的地端 appliance |
| LLM / VLM | 雲端 Gemma 4 — **與正式交付相同的模型,只是跑在雲端** | Gemma 4 12B(入門硬體)/ 31B(高階硬體),地端推理 |
| Embedding | Llama Nemotron Embed VL 1B v2(頁圖與文字同一向量空間),從 MVP 第一天就固定(現行 demo 的 Titan/文字版 embed 屬過渡,汰換時整庫重嵌一次) | Llama Nemotron Embed VL 1B v2(不變,交付時免重建索引) |
| 檢索 | 頁面級向量檢索 + Nemotron Rerank VL 精排(文字層 BM25 + RRF 在 #60 量測後拿掉) | dense 召回 → Nemotron Rerank VL 精排(標配,單一 reranker 蓋文字與頁面) |
| 文件解析(前處理) | 頁面化直檢:每頁轉圖 → Embed VL 直接嵌入,檢索不依賴轉錄(markitdown 與 VLM 全文轉錄路線皆已棄用) | 主路徑同左;PDF 文字層/OCR 只作加值(文字頁嵌入、引用、「圖+文」嵌入) |
| 硬體 | 不綁硬體;任一 NVIDIA GPU 機器即可驗證 | 雙 RTX 5060 Ti 16GB 主機(合計 32GB)約 10 萬 → DGX Spark 128GB / RTX 5090 主機 約 20 萬(全線 NVIDIA Linux) |
| DB / 儲存 | PostgreSQL + pgvector;本機檔案系統 | 相同;原檔儲存可換 S3-compatible |
| 部署 | docker compose;模型一律走 OpenAI-compatible endpoint(雲端) | **同一套全容器部署法**:app + DB + 模型皆容器,一份 compose 通用全落點(見 [deployment.md](deployment.md)) |

## 全程不變的原則(現在與未來共用)

這些原則從 MVP 第一天就生效,目的只有一個:**升級只換硬體與執行位置,不重寫系統、不換模型。**

| 原則 | 說明 |
| --- | --- |
| 一套系統 | 軟體不分版本;產業版本是同一個 image 內以 `KNOWVA_PROFILE` 選定的 profile 與主題,核心問答規則不變。Basic / Enterprise 只是「硬體採購與模型尺寸」的建議落點,pipeline、API、資料格式完全相同;**部署手順也相同**(全線 NVIDIA Linux,同一份 compose、安裝腳本與權重包,含模型全容器)。 |
| 模型與交付一致 | LLM/VLM 主線固定 Google Gemma 4 family,只在家族內調尺寸(12B/31B)、量化(Q4/Q8/SFP8)、context 與併發。**開發/測試走雲端時,也用同一家族與版本 — 雲端只改執行位置,不改模型。** |
| Embedding 固定 | 換 embedding = 全庫向量重建。全程固定 Llama Nemotron Embed VL 1B v2(2048 維;文字、頁圖、「圖+文」同一空間),從 MVP 就開始。demo 已實測:Bedrock Titan 與 NVIDIA embed 的向量空間互不相容,切換 provider 後必須刪除並重傳所有文件 — 這就是此原則的成本證明。 |
| Rerank 固定 family | 固定 NVIDIA Llama Nemotron Rerank VL 1B v2 — 單一 cross-encoder 同時精排文字與頁面候選(官方評測 recall@5 約 +6~7 個百分點)。低階機只調低候選數(top-k / top-n)或改低併發批次,不換模型、不改 retrieval API 與資料格式。 |
| 資料格式一致 | 解析輸出固定欄位:`document_id、page、content、table、figure、bbox、source_hash、confidence`。MVP 可先只填一部分(頁面化檢索不依賴 `content`,文字層可後補),但欄位先定,正式產品不重寫 ingestion pipeline。 |
| 非中資 | 核心模型避開 Qwen、DeepSeek、GLM、Kimi、MiniMax、BAAI/BGE 等中國系模型家族,維持企業採購與資料治理說法一致。 |
| 基礎設施不換 | PostgreSQL + pgvector 全程固定;原檔永遠保存(本機檔案系統,未來可換 S3-compatible),支援任何時候重新解析、重建索引。 |

## 對外說法與對內原則

**對外(銷售/提案用):**

- 本系統支援從 10 萬級 NVIDIA 主機起跳的全地端部署;需要更長上下文、更強多模態解析或更大文件庫時,升級至 DGX Spark / NVIDIA 工作站即可 — 換機器,不換系統。
- 所有落點使用同一套 Knowva pipeline 與模型家族;入門可升級到高階,不需要更換系統或重建產品流程。
- 文件資料可完全保留於客戶內網;雲端模型僅作 MVP 或客戶明確同意時使用。
- Embedding 模型固定,不會每次升級都重建向量資料庫;僅正式大版本升級才執行計畫內的 re-index migration。

**對內:**

- 開發與測試可用雲端加速迭代,但**雲端跑的模型必須與正式交付相同**(同家族、同版本);雲端只是執行位置,不是另一套模型棧。
- 對外文件維持「可完全地端部署」的產品定位,並明確區分 MVP、Basic Local、Enterprise Local 三個落點。

## 結論

產品規格定為「**同一套 Knowva Core,兩種硬體落點**」:

- 客戶用買機器的預算決定落點:約 10 萬 = 雙 RTX 5060 Ti 16GB 主機(合計 32GB;LLM 與 retriever 分卡)+ Gemma 4 12B Q4(最低可交付);約 20 萬 = DGX Spark 128GB(31B/長上下文容量型)或 RTX 5090 主機(12B 速度型)。全線 NVIDIA Linux、一套全容器部署;升級換機器,軟體與資料格式不變。
- 模型從第一天就對齊交付家族(Gemma 4 / Nemotron Embed VL / Nemotron Rerank VL),把「雲端 → 地端」的切換縮小成換執行環境,MVP 驗證到的命中率即交付品質。

## 參考資料

- NVIDIA 合作夥伴:GB10 迷你機(ASUS Ascent GX10 等)規格與台灣通路行情(實售 17 萬+),128GB unified memory。
- NVIDIA:Jetson AGX Orin 64GB 規格(64GB LPDDR5、204.8GB/s、JetPack/l4t),工控選配評估用。
- NVIDIA:DGX Spark User Guide / Hardware Overview,128GB LPDDR5x unified system memory、273GB/s bandwidth。
- Google AI for Developers:Gemma 4 model overview / model card,12B 與 31B 記憶體需求與多模態定位。
- NVIDIA Hugging Face / build.nvidia.com:llama-nemotron-embed-vl-1b-v2 model card — 約 1.7B(Llama 3.2 1B + SigLIP2 400M)、2048 維單向量、輸入支援文字/頁圖/「圖+文」、多語(MIRACL/MLQA)、NVIDIA Open Model License 可商用、vLLM 0.17+ / NIM 部署。
- NVIDIA Hugging Face / build.nvidia.com:llama-nemotron-rerank-vl-1b-v2 model card — 同架構 cross-encoder,對「查詢 × 頁面」輸出相關性分數,官方評測 recall@5 約 +6~7 個百分點;純文字 fallback:llama-nemotron-rerank-1b-v2。
- NVIDIA blog「Small Yet Mighty: Improve Accuracy in Multimodal Search and Visual Document Retrieval with Llama Nemotron RAG Models」— Embed VL + Rerank VL 的視覺文件檢索 pipeline 與 ViDoRe 系列評測。
