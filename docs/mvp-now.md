# 現在做(MVP)

> 總覽與不變原則見 [README.md](README.md);正式地端產品見 [future-plan.md](future-plan.md)。

**目標:驗證「客戶自己的文件能不能被正確問答」。** 不承諾完整企業級知識庫。MVP 的另一個價值是把 pipeline、資料格式與模型家族一次定對,讓之後的升級只剩換硬體、換執行位置。

## 1. 範圍

| 分類 | 內容 |
| --- | --- |
| 應做 | 文件上傳、原檔保存、頁面化前處理(PDF/Office/圖片逐頁轉圖 + Embed VL 嵌入,見 §6)、md/txt 標題感知 chunk、pgvector、文字層 hybrid search、Rerank VL、AI 回答(Gemma 直讀來源頁圖)、來源引用(檔名+頁碼)。 |
| 可做但限縮 | 文字層加值:PDF 內嵌文字抽出供 BM25/引用片段/「圖+文」嵌入;掃描頁 MVP 先走純圖嵌入(OCR 列選配),不承諾所有格式完美。 |
| 不做(非目標 — 本產品不是 SaaS) | 多租戶、雲端帳號體系、計費。 |
| 先不做(留給未來方案或視客戶需求) | SSO/複雜權限繼承、GraphRAG、agent workflow、知識圖譜、完整 admin 後台、部署器、維運監控、標準欄位全填(bbox/confidence)。 |

## 2. 模型策略:雲端可以,換模型不行

- **開發與對內測試可走雲端,但雲端跑的模型必須與正式交付相同**:LLM 用 Gemma 4(雲端託管版,例如 NVIDIA NIM 的 OpenAI-compatible 端點)、embedding 用 llama-nemotron-embed-vl-1b-v2、rerank 用 llama-nemotron-rerank-vl-1b-v2(兩者在 build.nvidia.com 都有託管端點)。雲端只是執行位置,不是另一套模型。
- 為什麼:
  1. **Prompt 與行為一致** — 為雲端調好的 prompt 與參數,交付地端時原樣可用,不用重調。
  2. **Embedding 空間一致** — MVP 期建立的向量索引與交付版相容,不需交付前重建(demo 已實測:Titan 與 NVIDIA embed 空間互不相容,切換即整庫報廢)。
  3. **驗證可信** — MVP 測到的命中率就是交付時的命中率,驗證結果不會因換模型失效。
- Embed VL / Rerank VL 各約 1.7B,build.nvidia.com 已託管可直接用;需要時也能在開發機本機跑(vLLM 0.17+ / transformers),同樣符合「與交付相同」原則。
- 對客戶:雲端 Gemma 僅在客戶明確同意時作過渡;對外定位始終是「可完全地端部署」。

## 3. 現況(demo)與 MVP 的差距

| 狀態 | 項目 |
| --- | --- |
| 已有 | 上傳、原檔保存、VLM 直讀解析(PDF/Office/圖片直送模型轉錄)、標題感知 chunk、embedding、pgvector(HNSW)、向量檢索 top-8、SSE 串流問答、來源引用、文件摘要/標籤、模型 provider 切換架構。 |
| 待補:模型對齊交付 | demo 目前預設 Bedrock Nova / Titan(早期驗證用),依 §2 應切到交付棧 — 聊天模型把 `MODEL_PROVIDER=nvidia` 的 `NVIDIA_CHAT_MODEL` 指向 NIM 上的 Gemma;embedding 換 llama-nemotron-embed-vl-1b-v2 時需一次性 schema migration(`vector(1024)` → `halfvec(2048)`;pgvector HNSW 對 vector 型別上限 2000 維)並重跑前處理(向量空間不同,整庫重嵌)。 |
| 待補:檢索 | dense 召回(頁與 chunk 同一索引)+ 文字層 BM25,RRF 合併;Rerank VL 精排(低候選數:dense top_k 50、BM25 top_k 50、rerank top_n 3-8 頁)。 |
| 待補:前處理頁面化 | 依 §6 重建 ingestion:Office→PDF 轉檔(LibreOffice headless)、pdfium 逐頁轉圖、頁圖(+PDF 文字層)送 Embed VL;逐頁處理天然解決單檔 4MB 上限與 PPT 不支援問題;輸出補 `page` 等標準欄位。 |
| 待補:部署打包 | 後端 Dockerfile(multi-stage)+ 全套 compose(含轉檔容器)+ 離線安裝腳本;推理 runtime 在目標機(Basic:雙 RTX 5060 Ti 16GB;Enterprise:DGX Spark)實測定案 — Embed/Rerank VL 只能 vLLM/NIM 服務化,LLM 先測同一 vLLM,見 [deployment.md](deployment.md)。 |
| 本次已調整 | UI 整體放大、對話框視覺強化(純 CSS,見 §5);前處理重新規劃為頁面視覺檢索(2026-07-10,見 §5、§6)。 |

## 4. 驗證問題(MVP 結案標準)

1. 客戶自己的文件能否被正確問答?
2. 答案是否穩定附上正確來源(檔名、頁碼)?
3. rerank 是否明顯改善命中率?(對照組:純向量 vs hybrid + Rerank VL)
4. 掃描頁 / PPT / 表格走「頁圖直檢 + Gemma 讀頁圖」後,問答是否可用?繁中文件的視覺檢索命中率是否達標(官方評測以英文為主)?
5. 回答 context 夾帶 3-6 張頁圖時,Basic 落點(雙 RTX 5060 Ti 16GB;12B Q4 獨占一卡)的 VRAM 與首字延遲是否可接受?
6. 同一份 compose + 安裝腳本,能否在目標機上離線一鍵部署成功?

## 5. MVP 系統調整(2026-07-10)

- UI 整體放大:基準字級與各元件(側欄、訊息、文件列表)同步放大,投影與展示情境更易讀。
- 對話框視覺更明顯:輸入框加大加高、邊框加粗、陰影提級、聚焦光暈加強;對話欄位寬度加大。
- 已實作於 `frontend/`(純 CSS 調整,不動元件結構)。
- **移除 markitdown**(品質不佳):解析改為 VLM 直讀 — 非 MD/TXT 檔直送多模態模型轉錄(Bedrock document/image block);單檔 ≤4MB,PPT 需先轉 PDF;後端不再依賴 Python。
- **前處理重新規劃(同日,規劃)**:「先轉錄成文字再檢索」整條路線退場(含上一條的 VLM 直讀轉錄,僅作 demo 現況過渡),改採 llama-nemotron-embed-vl-1b-v2 頁面視覺檢索 — 頁轉圖直接嵌入,檢索不依賴轉錄;EmbeddingGemma 自模型棧移除,rerank 統一 llama-nemotron-rerank-vl-1b-v2。MVP 版 pipeline 見 §6,完整版見 [future-plan.md](future-plan.md) §3。

## 6. 前處理規劃(頁面化,2026-07-10 起)

檢索單位從「轉錄文字的 chunk」改為「文件頁」;文字化不再是檢索前提,只是加值層。

1. **原檔保存**(不變)。
2. **頁面化**:PDF 以 pdfium 逐頁轉圖(約 150-200 DPI);圖片檔視為單頁;DOCX/PPTX/XLSX 先經 LibreOffice headless 轉 PDF 再轉圖(容器含 CJK 字型);md/txt 不轉圖,沿用標題感知 chunk。
3. **嵌入**:頁圖與文字 chunk 都用 llama-nemotron-embed-vl-1b-v2,同一 2048 維空間、同一索引(pgvector `halfvec(2048)` + HNSW);有 PDF 文字層時用「圖+文」合併嵌入(官方評測三種輸入中命中率最高)。
4. **文字層(加值,非關鍵路徑)**:PDF 內嵌文字直接抽出填 `content` — 供 BM25、引用片段、摘要;掃描頁 MVP 先純圖嵌入,OCR/VLM 轉錄列選配,缺文字層不阻擋入庫。
5. **摘要/標籤**:Gemma 讀前幾頁頁圖產生(best-effort,失敗不影響 ready)。
6. **查詢與回答**:query 文字嵌入(同空間)→ dense 召回(+文字層 BM25)→ Rerank VL 精排 → 取 top 3-6 頁,Gemma 4 直讀頁圖(+文字片段)回答,附檔名+頁碼。

相對「VLM 全文轉錄」路線的收益:前處理關鍵路徑只跑 1.7B 嵌入,不佔用 12B/31B;表格/圖表/掃描頁不經轉錄失真(Gemma 回答時看的是原頁);逐頁處理,無單檔 4MB 上限;PPT 經轉檔自然支援;檢索單位就是頁,頁碼引用天然精確。
