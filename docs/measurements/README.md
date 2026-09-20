# 本機推理實測紀錄

2026-09-16 ～ 09-18 在兩台機器上把整套系統跑起來所量到的數字。這裡只放**實際量到的東西與量它的腳本**，不含規劃或建議。

| 機器 | 規格 | 用途 |
| --- | --- | --- |
| knowva | Intel i5-8265U（4 核、AVX2、無 AVX-512）、23GB RAM、無 GPU | 全 CPU 跑完整系統 |
| Mac | Apple M1（8 GPU 核心）、8GB、macOS 14.6.1 | 只驗 MPS 相容性與速度 |

推理端用 `nim_shim.py`（附於本目錄）把兩個 Nemotron VL 模型包成 NIM 契約，chat 轉給 Ollama。

---

## 1. 結構性結論

**瓶頸是 vision encoder，不是 LLM 參數量。** knowva 上 prompt 處理固定落在 20–28 tokens/s，換哪個 chat 模型都一樣，因為成本來自每頁圖片變成的 token 數（72 dpi 約 221 個），不是模型大小。

這解釋了為什麼「換更小的模型」在 CPU 上沒有用（見 §5）。

---

## 2. knowva：全 CPU 的優化過程

同一個第一題，從頭到尾同一份 14 頁 PDF：

| 設定 | 查詢時間 |
| --- | --- |
| 起點：`gemma4:e4b`、180 dpi、8 候選、4 頁 context | 816s |
| `e2b`、4 候選、3 頁、rerank 1 tile | 390s |
| 100 dpi | 208s |
| 72 dpi | 171s |
| rerank 改純文字 | 146s |
| embed 常駐、拉長 keep-alive | 141s |
| embed 改 bfloat16、兩個模型同時常駐 | **102s** |

單項成本：

| 操作 | 時間 |
| --- | --- |
| query embed（float32 / bfloat16） | 0.21s / 0.64s |
| 單頁 image embed | ~21s |
| 單頁 image+text embed | ~25s |
| rerank 一頁（純文字 / 圖片，float32） | ~3s / 6.4s |
| rerank 一頁（圖片，bfloat16） | ~25s |
| 14 頁 PDF 上傳到 ready | 298s |

### dtype 在 AVX2 上的反直覺結果

這顆 CPU **沒有原生 bfloat16**，torch 用模擬，代價 5–7 倍。所以：

- **rerank 用 float32**：bfloat16 會讓整輪查詢的 rerank 從 6.4s 變 131s
- **embed 用 bfloat16**：query embed 本來就只有零點幾秒，快慢無所謂，但模型從 6.8GB 減半到 3.4GB，這是兩個模型能同時常駐的原因
- bfloat16 的 query 向量與既有 float32 向量相容：對同一 query 的 cosine 0.9999，對不同 query 是 0.038

---

## 3. prompt cache：追問為什麼時快時慢

推論服務快取的是 **prefix**，逐 token 從頭比對。請求結構是 `[system][RAG context][history][question]`，所以 context 區塊的排列順序決定能重用多少。

### v0.4.0（依 `(document_id, page_number)` 排序）

三輪對話，Ollama log 的 `prompt eval` / `task.n_tokens`：

| 輪次 | 頁碼 | prompt eval / 總 token |
| --- | --- | --- |
| 1 | 2, 3, 6 | 2095 / 2095 |
| 2 | 1, 3, 6 | 2326 / 2326 |
| 3 | 2, 4, 6 | 2548 / 2548 |

**三輪全量重算，零重用。** 原因：決定性排序不等於前綴穩定 —— 第 2 輪新進的頁 1 頁碼最小，插到最前面，prompt 從第一個 block 起全部不同。

### v0.4.1（追加式排列，client 回送 `previousSources`）

| 輪次 | 頁碼 | prompt eval / 總 token | 總時間 |
| --- | --- | --- | --- |
| 1 | 3, 6, 2 | 2095 / 2095 | 110s |
| 2 | 3, 6, 1 | **505 / 2127** | **53s** |
| 3 | 6, 2, 4 | 2225 / 2225 | 133s |

第 2 輪重用 1622 tokens，LCP similarity 0.763，prompt eval 從 77.8s 降到 22.8s。

第 3 輪仍全量重算，而且不是 bug：上一輪版面是 units `[32, 35, 30]`，這輪檢索到 `{35, 31, 33}`，佔位置 0 的 unit 32 掉出檢索結果，沿用的 35 遞補到最前，prompt 從第一個 block 分岔。

**可重用長度 = 上一輪版面中從位置 0 起連續存活到本輪的長度。** 位置 0 的來源一旦落選就是 0，跟排版規則無關。

測試腳本：`three_turn.py`（會回送 `previousSources`，不回送的話行為等同 v0.4.0）。

---

## 4. Mac / MPS 相容性與速度（M1 8GB）

**兩個 VL 模型在 MPS + float16 上都跑得起來，沒有缺任何 op，不需要 `PYTORCH_ENABLE_MPS_FALLBACK`。**

| 操作 | knowva CPU | M1 MPS | |
| --- | --- | --- | --- |
| image+text embed（一頁） | ~25s | **3.64s** | 6.9x |
| query embed（暖機後） | 0.21s | 0.89s | 較慢 |
| rerank 一頁（純文字，短字串） | ~3s | **0.61s** | 5x |
| rerank 一頁（文字 + 圖片） | 6.4s (fp32) | **1.05s** | 6x |

文字 query 在 MPS 上反而慢，工作量太小、kernel 啟動成本蓋過收益。

**圖片比文字便宜**：72 dpi 單 tile 的頁面圖片約 221 tokens，一頁的文字層約 1183 tokens。這反轉了 knowva 上「改純文字 rerank」的前提。

### 載入方式（實作上的必要修正）

rerank 模型的 config 帶 `torch_dtype: null`，用 `device_map="mps"` 時 accelerate 逐 shard 分派的中間態是 float32，在 8GB 上直接卡死在 `Loading weights 0/601`，無錯誤訊息（被 SIGKILL）。

改成先以明確 dtype 在 CPU 載入、再 `.to("mps")`：

```
cpu load 8.2s  peak_rss=2.90GB
move to mps 1.7s
```

### 吞吐量與 50 筆候選

50 筆、每筆 1183 tokens、真實頁面長度的文字：

| 批次 | 總時間 | 每筆 | 吞吐 |
| --- | --- | --- | --- |
| 4 | **69.2s** | 1.38s | 890 tok/s |
| 16 | 79.9s | 1.60s | 757 tok/s |

批次放大沒有幫助（此機記憶體壓力下反而更慢，8GB 無法排除這個干擾）。50 筆 = 59,150 tokens，除以吞吐量就是下限。

記憶體：模型 2.90GB，圖片 activations 約 0.24GB/筆（4 筆實測 0.95GB）。

腳本：`mps_embed.py`、`mps_rerank.py`、`mps_rerank50.py`。用 `uv` 建 venv 後直接跑，環境變數 `DEV` / `DT` / `MODALITY` / `N` / `BATCH` 控制。

---

## 5. 測過但不採用的路（都有數字）

| 做法 | 結果 |
| --- | --- |
| 換更小的 chat 模型 `gemma3:4b` | 同一頁 82s，`gemma4:e2b` 只要 39s |
| 換 `qwen3.5:2b` | 87s，且同一張圖花更多 token |
| `gemma3n:e2b` | Ollama 回 HTTP 400，不支援多模態 |
| `qwen3.5:0.8b` | Ollama 0.34.1 拉不下來（manifest 解析錯誤），登錄檔本身正常 |
| rerank 用 int8 動態量化 | 快 2 倍、省 4GB，但**排序壞掉**：5 個 passage 分數位移 -7.9 到 +4.4，13 分差距塌成 2 分，不相關的排到第二 |
| 把 context 移到 history 之後 | 會讓唯一有效的情境（追問命中同來源）失效 |

float16 在 MPS 上**沒有**重現 int8 的問題：5 個 passage 的排序正確，相關的在前、不相關的在後。

---

## 6. 檔案

| 檔案 | 用途 |
| --- | --- |
| `nim_shim.py` | CPU 路線的推理前端：NIM 契約的 embed/rerank + chat 轉 Ollama。環境變數見檔首 docstring |
| `three_turn.py` | 三輪對話量測，會回送 `previousSources` |
| `mps_embed.py` | embed 模型在 MPS 上的相容性與速度 |
| `mps_rerank.py` | rerank 模型同上，一次一個 modality |
| `mps_rerank50.py` | 50 筆候選的吞吐量與批次大小比較 |

Python 腳本用 `uv` 建環境（`uv venv` + `uv pip install`），不要用 pip。

## 7. 未量測

- image rerank 在高候選數下的 activations（只量到 4 筆）
- PyTorch MPS 是否會用上 M5/M6 GPU 內的 Neural Accelerator —— M1 沒有這個單元，測不出來
- Gemma 4 12B 的 encoder-free 設計對每頁 token 數的實際影響
