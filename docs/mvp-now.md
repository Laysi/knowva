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

## 2. 模型策略:固定雲端模型家族

- **MVP 的模型推論只走雲端託管 API,模型家族不可替換**:LLM 用 Gemma 4、embedding 用 llama-nemotron-embed-vl-1b-v2、rerank 用 llama-nemotron-rerank-vl-1b-v2。不新增本地模型、vLLM、本地 NIM、量化或替代模型路徑。
- 為什麼:
  1. **Prompt 與行為一致** — 開發、測試與展示都使用同一組模型契約。
  2. **Embedding 空間一致** — 文字 query、文字 passage、頁圖與圖+文都固定在同一個 2048 維空間;切換 embedding 會使整庫索引失效。
  3. **驗證可信** — MVP 測到的命中率不會因測試時偷偷使用替代模型而失真。

## 3. 現況(demo)與 MVP 的差距

| 狀態 | 項目 |
| --- | --- |
| 已有 | 上傳與原檔保存、PDF/Office/圖片頁面化、PDF 文字層抽取、MD/TXT 標題感知 chunk、四種 Nemotron Embed VL 輸入、`halfvec(2048)` + HNSW、dense/BM25 各 top-50 + deterministic RRF、SSE 串流問答與來源引用。 |
| 已完成:模型與入庫對齊 | Gemma 4 / Nemotron Embed VL / Nemotron Rerank VL family 已鎖定;舊 `vector(1024)` 已淘汰。artifact unit 以 bounded concurrency 嵌入,全部成功且 transaction 內計數一致才把文件設為 ready;失敗不會發布部分新向量。 |
| 已完成:候選檢索 | ready-only dense 召回(頁與 chunk 同一 2048d index)+ 文字層真正 Okapi BM25，再以固定 k=60 的 RRF 合併；兩路各 top_k 50。 |
| 待補:精排與回答多模態整合 | Gemma 4 直接讀取召回頁圖，以及 RRF 候選串接 Rerank VL（top_n 3-8 頁）仍是後續 issue。 |
| 待補:部署驗證 | 現有 multi-stage Dockerfile、compose、Pdfium、LibreOffice 與 CJK 字型需持續做 cloud API 環境的整合 smoke;不在 MVP 新增本地推論 runtime。 |
| 本次已調整 | UI 整體放大、對話框視覺強化(純 CSS,見 §5);前處理重新規劃為頁面視覺檢索(2026-07-10,見 §5、§6)。 |

## 4. 驗證問題(MVP 結案標準)

1. 客戶自己的文件能否被正確問答?
2. 答案是否穩定附上正確來源(檔名、頁碼)?
3. rerank 是否明顯改善命中率?(對照組:純向量 vs hybrid + Rerank VL)
4. 掃描頁 / PPT / 表格走「頁圖直檢 + Gemma 讀頁圖」後,問答是否可用?繁中文件的視覺檢索命中率是否達標(官方評測以英文為主)?
5. 回答 context 夾帶 3-6 張頁圖時,雲端 Gemma 4 的首字延遲與 payload 成本是否可接受?
6. 同一份 compose 能否穩定完成轉檔、雲端推論與資料庫入庫?

## 5. MVP 系統調整(2026-07-10)

- UI 整體放大:基準字級與各元件(側欄、訊息、文件列表)同步放大,投影與展示情境更易讀。
- 對話框視覺更明顯:輸入框加大加高、邊框加粗、陰影提級、聚焦光暈加強;對話欄位寬度加大。
- 已實作於 `frontend/`(純 CSS 調整,不動元件結構)。
- **全文轉錄路徑已退場**:不再使用 markitdown、VLM 全文轉錄或 converted markdown;非 MD/TXT 文件一律先頁面化,掃描頁沒有 OCR 也能以純圖嵌入。
- **頁面化與 Embed VL 入庫已實作**:PDF/Office/圖片先發布 atomic artifact set,再以 llama-nemotron-embed-vl-1b-v2 建立 2048 維向量;MD/TXT 直接嵌入標題感知 chunk。MVP pipeline 見 §6。

## 6. 現行前處理(頁面化 + Embed VL)

檢索單位從「轉錄文字的 chunk」改為「文件頁」;文字化不再是檢索前提,只是加值層。

1. **原檔保存**(不變)。
2. **頁面化**:PDF 以 pdfium 逐頁轉圖(約 150-200 DPI);圖片檔視為單頁;DOCX/PPTX/XLSX 先經 LibreOffice headless 轉 PDF 再轉圖(容器含 CJK 字型);md/txt 不轉圖,沿用標題感知 chunk。
3. **嵌入**:頁圖與文字 chunk 都用 llama-nemotron-embed-vl-1b-v2,同一 2048 維空間、同一索引(pgvector `halfvec(2048)` + HNSW);有 PDF 文字層時用「圖+文」合併嵌入(官方評測三種輸入中命中率最高)。
4. **文字層(加值,非關鍵路徑)**:PDF 內嵌文字直接抽出填 `content` — 供 BM25、引用片段、摘要;掃描頁 MVP 先純圖嵌入,OCR/VLM 轉錄列選配,缺文字層不阻擋入庫。
5. **ready 契約**:每文件 bounded concurrency 呼叫 embedding;每個回應必須恰為 2048 維 finite float。所有 unit 成功後才在單一 transaction 內替換 `document_units`、核對 artifact/unit 計數並設為 ready。摘要/標籤目前不在關鍵路徑產生。
6. **查詢與回答**:query 使用同模型的 text query embedding；dense top-50 與文字 BM25 top-50 已用 RRF(k=60) 合併，既有 chat 暫取融合前 8 筆。Rerank VL 與 Gemma 4 直讀召回頁圖由後續 issue 串接。

相對「VLM 全文轉錄」路線的收益:前處理關鍵路徑只呼叫固定 Embed VL;表格/圖表/掃描頁不經轉錄失真;逐頁處理並限制單一 payload;PPT 經轉檔自然支援;檢索單位就是頁,頁碼引用天然精確。
