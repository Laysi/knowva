# 文件轉換 fixtures

這組 fixture 由 `backend/src/test_fixtures.rs` 在測試期間直接產生，不含客戶文件或第三方內容，因此 repository 不需要保存不透明的 binary 檔案。

| 測試輸入 | 預期結果 | 驗證重點 |
|---|---:|---|
| `native-two-page.pdf` | 2 頁 | 頁碼依序為 1、2，兩頁各有可辨識的文字層；`needs_vision = false` |
| `scanned-one-page.pdf` | 1 頁 | 只有 image XObject、沒有文字層，仍產生頁圖並送 image-only passage embedding；`needs_vision = true` |
| 單張 WebP | 1 頁 | 正規化為 PNG，尺寸與 1-based 頁碼正確 |
| `ordered-two-page.docx` | 2 頁 | 明確 page break 經 LibreOffice 轉 PDF 後保留順序；`needs_vision = false` |
| `ordered-two-slide.pptx` | 2 頁 | 兩張投影片經 LibreOffice 轉 PDF 後順序不變；整頁背景矩形不算圖形，`needs_vision = false` |
| `single-sheet.xlsx` | 1 頁 | 單一 worksheet 轉為一頁 PDF；`needs_vision = false` |
| `headings.md` | 至少 1 個文字單元 | 使用既有標題感知 chunk，不轉頁圖 |
| `plain.txt` | 至少 1 個文字單元 | 使用文字 chunk，不虛構頁碼 |
| `hybrid-retrieval.md` | 至少 1 個文字單元 | 含繁中同義詞 `人工智慧`/`AI`、精確產品代號 `NX-2048-V2` 與兩段刻意重複內容 |

各 fixture 的文字都是一般句子長度：只有幾個字的頁面依規則會判成需要看圖，所以文字頁路徑需要真正的文字層。

`write_vision_fixtures` 另外產生判斷規則 fixture，每頁都有同一句文字層，只差在畫了什麼；多頁的逐頁列出：

| fixture | `needs_vision` | 驗證重點 |
|---|:---:|---|
| `vision-text-page.pdf` | false | 只有文字 |
| `vision-table-grid.pdf` | false | 水平垂直的筆畫格線與細框線不算圖形 |
| `vision-bar-chart.pdf` | true | 填色長條合計約 8% |
| `vision-line-chart.pdf` | true | 含斜線的折線筆畫 |
| `vision-background-and-logo.pdf` | false | 整頁背景矩形與 0.2% 的小 logo 都不算 |
| `vision-form-image-half.pdf` | true | form XObject 內的 400 pt 圖片縮放 0.5 後約 8% |
| `vision-form-image-tenth.pdf` | false | 同一張圖縮放 0.1 後約 0.3%，低於單張下限；沒有組合矩陣就會誤判 |
| `vision-stamp-annotation.pdf` | true | Stamp annotation 會畫進頁圖 |
| `vision-link-annotation-with-appearance.pdf` | true | 帶 appearance stream 的 Link 會畫進頁圖 |
| `vision-link-annotation.pdf` | false | 沒有外觀的 Link annotation 不畫進頁圖 |
| `vision-template-deck.pdf` | false, false, true, false | 4 頁都有同一條佔 7.6% 的母片頁尾圖，是頁面家具；只有第 3 頁另有自己的圖。第 3、4 頁畫的是另一個內容相同的 image XObject，所以比的是內容而不是物件編號 |
| `vision-repeated-page-background.pdf` | true, true, true | 3 頁都是同一張整頁圖；佔頁面一半以上的圖不當家具 |
| `vision-vector-template.pdf` | false, false, true, false | 4 頁都有同一個佔 6.2% 的深色三角色帶（向量），是頁面家具；只有第 3 頁另有長條圖 |
| `vision-text-over-template.pdf` | false, false, false, false | 4 頁都有母片頁尾圖，第 2 頁另在圖上寫了一行字；比對前會塗掉文字框，所以 4 頁仍看起來相同 |
| `vision-text-covered-variants.pdf` | true, true, true, true | 同 `decode-variants`，但第 1 頁有一層隱形 OCR 文字整個蓋住圖；塗掉文字框後沒有像素可比，不能當成看起來相同 |
| `vision-clip-variants.pdf` | true, true, true, true | 4 頁同一張圖、同一個位置，第 1、2 頁裁成左半、第 3、4 頁裁成右半；畫出來是兩種各 2 頁，都不是家具 |
| `vision-decode-variants.pdf` | true, true, true, true | 4 頁同一份串流、同一個位置，第 3、4 頁用 `/Decode` 反相；畫出來是兩種各 2 頁 |
| `vision-flipped-copies.pdf` | true, true, true, true | 4 頁同一張圖、同一個 150 pt 方框，第 3、4 頁左右鏡射；外框相同但畫出來不同 |
| `vision-soft-mask-variants.pdf` | true, true, true, true | 4 頁同一份串流，第 3、4 頁套不同的 `/SMask`；畫出來不同 |
| `vision-explicit-mask-variants.pdf` | true, true, true, true | 4 頁同一份串流，第 3、4 頁套不同的 stencil `/Mask`；畫出來不同 |
| `vision-stencil-colors.pdf` | true, true, true, true | 4 頁同一個 stencil 遮罩，第 1、2 頁紅色、第 3、4 頁藍色；填色不同就不是同一張 |

其中 `scanned-one-page.pdf` 同時是 hybrid retrieval 的 dense-only fixture：它沒有 `content`，不得出現在 BM25 route，但仍須由 2048 維頁圖 embedding 召回。database fixtures 另以相同文字的不同 1-based page 驗證它們保留不同 `unit_id`，只有同一頁在 dense/BM25 重疊時才會去重。

一般 `cargo test` 會用 mock cloud endpoint 驗證 embedding payload、圖片正規化、MD/TXT 入庫、DB constraint、重試與刪除清理。真實 Pdfium／LibreOffice 轉換測試標記為 ignored，避免缺少 native dependency 的 CI 假失敗；安裝依賴後執行：

```bash
cd backend
PDFIUM_LIBRARY_PATH=/path/to/libpdfium.so \
LIBREOFFICE_BIN=libreoffice \
proto exec rust -- cargo test -- --ignored --exact --nocapture --test-threads=1 \
  page_artifacts::tests::generated_pdf_and_office_fixtures_preserve_page_order \
  page_artifacts::tests::page_vision_decisions_match_fixtures_and_the_example_report \
  ingest::tests::native_conversion_runs_in_scratch_and_publishes_relative_keys
```

或直接使用包含固定 Pdfium、LibreOffice 與 CJK 字型的 Docker 驗證 target：

```bash
docker build --target native-test -t knowva:native-test .
```

此 smoke test 會實際比較頁數與頁序，並以 mock endpoint 確認原生 PDF 逐頁送出 image+text、掃描 PDF 沒有文字時仍送出 image-only embedding；不會耗用真實雲端額度。第二支測試逐頁檢查上表 fixture、既有 PDF／Office fixture 與 `example_file/iiot_industry_report.pdf`（14 頁只有第 7 頁看圖）的 `needs_vision`，並印出每頁的判斷原因。第三支測試走實際 ingest 路徑：原檔從 storage 取回本機暫存區轉檔，再以 `{document_id}/artifacts-{run_id}/...` 相對 key 上傳頁圖、`converted.pdf` 與 manifest，確認每頁的 `needs_vision` 已寫進 manifest，並確認暫存區已刪除。

PPTX 的文字層取決於 LibreOffice 版本：Fedora 的 LibreOffice 轉出的投影片 PDF 沒有文字層，這兩頁會依規則判成需要看圖，在主機上跑時兩支 PPTX 斷言都會失敗。以 `native-test` image（Debian LibreOffice）的結果為準。真實 NVIDIA Embed VL 的可選 smoke 指令見 repository 根目錄 `README.md`。
