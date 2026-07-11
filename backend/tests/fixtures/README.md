# 文件轉換 fixtures

這組 fixture 由 `backend/src/test_fixtures.rs` 在測試期間直接產生，不含客戶文件或第三方內容，因此 repository 不需要保存不透明的 binary 檔案。

| 測試輸入 | 預期結果 | 驗證重點 |
|---|---:|---|
| `native-two-page.pdf` | 2 頁 | 頁碼依序為 1、2，兩頁各有可辨識的文字層並送 image+text passage embedding |
| `scanned-one-page.pdf` | 1 頁 | 只有 image XObject、沒有文字層，仍產生頁圖並送 image-only passage embedding |
| 單張 WebP | 1 頁 | 正規化為 PNG，尺寸與 1-based 頁碼正確 |
| `ordered-two-page.docx` | 2 頁 | 明確 page break 經 LibreOffice 轉 PDF 後保留順序 |
| `ordered-two-slide.pptx` | 2 頁 | 兩張投影片經 LibreOffice 轉 PDF 後順序不變 |
| `single-sheet.xlsx` | 1 頁 | 單一 worksheet 轉為一頁 PDF |
| `headings.md` | 至少 1 個文字單元 | 使用既有標題感知 chunk，不轉頁圖 |
| `plain.txt` | 至少 1 個文字單元 | 使用文字 chunk，不虛構頁碼 |

一般 `cargo test` 會用 mock cloud endpoint 驗證 embedding payload、圖片正規化、MD/TXT 入庫、DB constraint、重試與刪除清理。真實 Pdfium／LibreOffice 轉換測試標記為 ignored，避免缺少 native dependency 的 CI 假失敗；安裝依賴後執行：

```bash
cd backend
PDFIUM_LIBRARY_PATH=/path/to/libpdfium.so \
LIBREOFFICE_BIN=libreoffice \
proto exec rust -- cargo test page_artifacts::tests::generated_pdf_and_office_fixtures_preserve_page_order \
  -- --ignored --exact --nocapture
```

或直接使用包含固定 Pdfium、LibreOffice 與 CJK 字型的 Docker 驗證 target：

```bash
docker build --target native-test -t ai-km-demo:native-test .
```

此 smoke test 會實際比較頁數與頁序，並以 mock endpoint 確認原生 PDF 逐頁送出 image+text、掃描 PDF 沒有文字時仍送出 image-only embedding；不會耗用真實雲端額度。真實 NVIDIA Embed VL 的可選 smoke 指令見 repository 根目錄 `README.md`。
