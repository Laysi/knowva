# 部署架構(全程適用 — 從 MVP 就生效,不是未來項目)

> 總覽見 [README.md](README.md);MVP 待辦見 [mvp-now.md](mvp-now.md) §3;硬體落點見 [future-plan.md](future-plan.md) §2。

**核心決策:產品線全部落在 NVIDIA + Linux,macOS 不在標準支援清單。** 因此「一套部署方法」可以做到最徹底的形態:**整套系統 — 含模型 — 全部容器化**,一份 docker compose 從開發機用到每一台客戶機,沒有任何 per-platform 分支。

## 1. 拓撲(全落點相同)

| 服務 | 形態 | 說明 |
| --- | --- | --- |
| app(前端 + 後端) | 容器 | 前端 build 產物烘進後端 image(axum 本就 serve `frontend/dist`);前處理 = pdfium 頁面轉圖(原生庫烘進 image)+ 呼叫嵌入端點,無 Python 依賴,客戶機器不需裝任何語言環境。 |
| 文件轉檔(Office→PDF) | 容器 | LibreOffice headless,image 含 CJK 字型;僅 DOCX/PPTX/XLSX 上傳時使用,PDF/圖片/md/txt 不經過它。 |
| PostgreSQL + pgvector | 容器 | 沿用現行 `docker-compose.yml`(pgvector/pgvector:pg17)— 現在的 compose 就是全套 compose 的起點;2048 維索引走 `halfvec`。 |
| LLM 推理 | 容器(GPU) | 經 NVIDIA Container Toolkit 直通 GPU;對 app 永遠只是一個 OpenAI-compatible base URL。 |
| embedding / rerank | 容器(GPU) | Embed VL / Rerank VL 1B v2 以 vLLM(0.17+)服務化,與 LLM 共用同一 runtime 最簡;TEI 不支援 VL 模型,棄選;NIM 僅選配。 |

模型 = 「一個 OpenAI-compatible endpoint」:demo 的 nvidia provider 已是此形狀;VL 嵌入/精排是 NVIDIA 對 `/v1/embeddings`、`/v1/ranking` 的多模態擴充(頁圖以 base64 傳入),app 以 provider 模組封裝。雲端開發期 base URL 指向 build.nvidia.com(compose 不含推理容器);地端交付時 compose 多推理 service、URL 改指本機 — **app 層零改動**。

多 GPU 落點(Basic 雙 RTX 5060 Ti 16GB)以 compose 的 `device_ids` 分卡:卡 0 = LLM、卡 1 = Embed/Rerank VL;單 GPU / 統一記憶體落點(5090、DGX Spark)同一份 compose 指同一顆 GPU — 拓撲與 image 不變,只差 device 指定。

## 2. 推理 runtime(MVP 定案,全線同一個)

| 候選 | 特性 | 定位 |
| --- | --- | --- |
| vLLM 容器 | 0.17+ 可同時服務 Gemma 與 Embed/Rerank VL;併發與吞吐強(批次嵌入、多人問答);NVIDIA 對 DGX Spark 有官方支援 | 預設先測:單一 runtime 服務三顆模型 |
| llama.cpp server / Ollama 容器 | 最輕;GGUF Q4/Q8 與量化規劃一致;但只能跑 LLM — 不支援 Embed/Rerank VL | 僅當 vLLM 跑 Gemma 的量化/VRAM 不達標時,作 LLM 專用容器(雙 runtime) |
| NIM | 介面最省事,但生產環境需 NVIDIA AI Enterprise 授權,成本不符 10-20 萬級 appliance | 僅列選配,不進標準 |

定案準則:Embed/Rerank VL 只能 vLLM(或 NIM)服務化 → retriever 固定 vLLM;LLM 先測同一 vLLM(單 runtime 最簡),不達標才拆出 llama.cpp 系 LLM 專用容器。**不論結果,全產品線同一套,不做 per-SKU 分支。**

## 3. 打包與離線交付

- 模型權重不放進 image(Gemma 動輒 10-60GB;Embed/Rerank VL 各約 3-4GB):獨立檔案隨安裝包交付,掛 volume 載入;版本 = image tag + 權重版本。
- **離線安裝**:`docker save` 打包全部 image + 權重檔 → 拷入客戶內網 → 安裝腳本執行 `docker load` → `docker compose up`。
- 升級 = 換 image / 換權重檔;不污染客戶機器環境。

## 4. CPU 架構

- RTX 主機(Basic 預設、5090 速度型)、96GB 工作站與開發機是 x86_64;DGX Spark / GB10 機器與 Jetson(工控選配)是 arm64。
- 不存在「另外開發 x86 版」:Rust + docker buildx 用同一套程式碼、同一份 Dockerfile 產出 `linux/amd64` + `linux/arm64` 雙架構 image,成本只是 CI 多一個 build target。
- 推理容器用官方多架構 image;arm64(sbsa)版支援需逐一驗證(見 §5)。

## 5. 待驗證(MVP 期完成)

1. 推理 runtime 定案:vLLM 在目標機(Basic:雙 RTX 5060 Ti 16GB;Enterprise:DGX Spark)實測 Gemma 12B / 31B 的 TTFT、token/s 與併發(含回答夾帶頁圖),Basic 加測雙卡分工(卡 0 LLM、卡 1 retriever)與 12B Q8 於獨占 16GB 的可行性;不達標才拆 llama.cpp 系 LLM 專用容器。
2. Embed/Rerank VL 服務化:vLLM 0.17+ 跑 llama-nemotron-embed-vl-1b-v2 / rerank-vl-1b-v2 的 API 形狀、吞吐與 INT8/FP8 量化 VRAM;NIM 對照(選配)。
3. `halfvec(2048)` + HNSW 在 pgvector/pgvector:pg17 的建索引與查詢效能。
4. 頁面化定案:pdfium 轉圖 DPI(150-200)與輸出格式(PNG/WebP);LibreOffice 轉檔品質(PPTX 版面、CJK 字型)。
5. arm64 image 完整性:app(自建,無虞)、pgvector(官方有)、vLLM(sbsa 版)、LibreOffice(含字型)。
6. 一鍵部署演練:同一份 compose + 安裝腳本在目標機離線完成部署。
7. (若客戶採 Jetson Orin 工控選配)l4t 容器變體建置與 RAG 長 prompt TTFT 實測。
