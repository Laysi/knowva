# Knowva 知星 — Active MVP

> 初版 demo 規劃已由 GitHub tracker 取代。實作優先序為目前子 issue、[tracker #1](https://github.com/Laysi/outsourcing-ai-km/issues/1)、`docs/mvp-now.md`；本檔只保留方便進入專案的摘要，不再描述已退場的 provider 或 ingestion 路徑。

## 已鎖定決策

- 所有模型推論僅呼叫雲端託管 API。
- 回答／多模態閱讀固定 Gemma 4 家族。
- embedding 固定 `llama-nemotron-embed-vl-1b-v2`，使用 2048 維向量。
- rerank 固定 `llama-nemotron-rerank-vl-1b-v2`。
- 不新增 Bedrock Nova/Titan fallback、本地模型、vLLM、本地 NIM、量化或 GPU/VRAM 工作。

## 目前資料流

```text
上傳原檔
  → PDF 以 Pdfium 逐頁轉圖；Office 先由 LibreOffice 轉 PDF；圖片視為單頁
  → PDF 文字層 best-effort 抽取；MD/TXT 保留標題感知文字單元
  → 原子發布 artifact run 與 manifest
  → #5 以固定 Nemotron Embed VL 建立真正的 halfvec(2048) document_units
  → #6 dense + BM25 + RRF
  → #7 Nemotron Rerank VL
  → #8 Gemma 4 讀來源頁圖並輸出頁碼引用
```

目前支援 PDF、DOCX/PPTX/XLSX、PNG/JPEG/WebP、MD/TXT，HTTP 上傳上限為 50 MiB。CSV/HTML、舊版 Office 與全文 VLM 轉錄不在此階段支援範圍。

## 開發入口

- 設定、啟動方式、artifact 契約與驗證命令：`README.md`
- 固定模型與完整 MVP 工作順序：GitHub tracker #1
- 文件轉換 fixture 與預期頁數：`backend/tests/fixtures/README.md`
- 後續規格：GitHub issues #5–#10
