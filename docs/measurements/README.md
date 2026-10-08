# 本機推理實測紀錄

2026-09-16 ～ 09-18 在兩台機器上把整套系統跑起來所量到的數字；§8 是 2026-10-02 在 Mac mini M6 上的。這裡只放**實際量到的東西與量它的腳本**，不含規劃或建議。

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

---

## 8. Mac mini M6（24GB、macOS 27.0，2026-10-02）

整套系統裝在一台 Mac mini 上：app 與 PostgreSQL 在 Colima VM（vz，3 GiB），Gemma 4 由 Ollama 0.35.0（內建 llama-server）跑 Metal，兩個 Nemotron VL 在 MPS float16（torch 2.14.1、transformers 5.18.0）。腳本在 `deploy/mac-mini/bench/`，設定與取捨見 `deploy/mac-mini/README.md`。

### 8.1 chat 模型

6 頁來源（1600px JPEG + 文字層）、context 16384、embed 與 rerank 常駐，每格 3 輪：

| 模型 | 載入後 | 首字（FA 關 / 開） | 解碼 tok/s（FA 關 / 開） |
| --- | --- | --- | --- |
| `gemma4:e2b-it-q4_K_M` | 100% GPU | 19.5–22.8s / 13.3–15.4s | 93–98 / 90–101 |
| `gemma4:12b-it-qat` | 7.4 GiB、100% GPU | 20.2–22.0s / 20.9–23.4s | 15.5 / 17.2 |
| `gemma4:12b-it-q4_K_M` | 100% GPU | 21.8–23.9s / 22.6–25.4s | 23.7 / 24.2 |
| `gemma4:26b-a4b-it-q4_K_M` | 17.6–19.5 GiB、79–83% GPU | 無輸出 | — |

A4B 每輪都在編碼頁圖時 Metal OOM（`kIOGPUCommandBufferCallbackErrorOutOfMemory`，llama-server 被 kill），swap 漲到 10GB。**Ollama 這時仍回 HTTP 200、串流是空的**，錯誤只在 server log。單獨跑短 prompt 時它是 20GB、84% GPU、32.7 tok/s。

### 8.2 首字延遲花在哪（12B QAT，FA 開）

| 送入 | prompt tokens | 首字 |
| --- | --- | --- |
| 只有文字 | 2.1–3.8K | 7.1–8.5s |
| 只有頁圖 | 5368 | 13.8s |
| 文字 + 頁圖 | 7.3–9.0K | 21–23s |

圖片 token 跟像素成正比，llama-server log 的 prefill 速率約 290 tok/s，影像編碼本身每張 13ms：

| 頁圖長邊 | 每張約 | 6 張只送圖的首字 | 讀「IL050」 |
| --- | --- | --- | --- |
| 1600 | 860 tokens | 13.8s | 對 |
| 1280 | 570 | 9.5s | — |
| 1024 | 340 | 5.9s | 對 |
| 768 | 195 | 4.1s | 錯（「11050」） |

文字 + 1280px 頁圖的完整 prompt，12 輪平均首字 19.2–19.7s（範圍 14.3–24.4s）。

### 8.3 檢索（MPS float16）

- query embed 暖機後 0.03s；頁面 embed（圖 + 文、6 tiles）1.02–1.33s/頁。
- rerank 一次查詢 50 筆、批次 8：

| 送入 | tiles | 總時間 | 14 頁中的前 3 名 |
| --- | --- | --- | --- |
| 圖 + 文 | 6 | 57.4s | — |
| 圖 + 文 | 1 | 12.9s | 9, 2, 6 |
| 只文字 | — | 4.9s | 9, 2, 6 |
| 只圖 | 6 | 49.8s | 9, 2, 6 |
| 只圖 | 2 | 21.8s | 9, 6, 2 |
| 只圖 | 1 | 7.9s | 9, 2, 4 |

查詢是「JADE 在 IoT4Industry 裡是怎麼部署的？」，最相關的是第 9、10 頁；只有一題，分不出模式之間的品質差異。

### 8.4 記憶體

- llama-server 的 prompt cache（`--cache-ram`）預設 8192 MiB：實體佔用 10.8 GB，是 swap 的主因。設 2048 後 3.3 GB；兩種設定交替各 6 輪，首字平均 19.2s 對 19.7s，沒有差別。2 GB 約留得住最近兩份 prompt。
- Gemma 4 的 sliding-window 層每個 context checkpoint 320 MiB，預設上限 32 個。
- 模型卸載後重新載入：單獨測時 embed 0.95s；Gemma 常駐時權重已被擠出檔案快取，embed 4.2–5.3s、rerank 第一批多 5s 左右。

### 8.5 端到端（Knowva v0.5.0）

14 頁 PDF 上傳到 ready（含 Gemma 寫摘要）51.4s。三輪對話（`bench/e2e.ts`，回送 `previousSources`）：

| | 第 1 輪 | 第 2 輪 | 第 3 輪 |
| --- | --- | --- | --- |
| 檢索模型閒置 2s 就卸載：首字 | 33.6s | 39.3s | 43.6s |
| 　dense + rerank | 4.5s + 8.8s | 5.3s + 11.8s | 4.8s + 7.9s |
| 　Gemma prefill | 21.5s | 21.9s | 27.3s |
| 檢索模型常駐：首字 | 38.9s（剛啟動） | 30.4s | 30.4s |
| 　dense + rerank | 4.2s + 11.7s | 1.1s + 5.9s | 0.1s + 3.7s |
| 　Gemma prefill | 37.4s | 26.3s | 31.8s |

常駐時 swap 最高 4.4 GB，Gemma prefill 慢 4–5s，但省下的重新載入更多。三輪的頁面與回答都正確（IL050、Contract-Net 流程、Arduino / RIOT-OS）。

### 8.6 與 §3 不同的地方：追問沒有重用 prefix

三輪的 llama-server log 都是從 `cached n_tokens = 0` 開始算。Gemma 4 的 sliding-window 層只能從 checkpoint 接續，而 checkpoint 只建在上一份 prompt 的尾端；新 prompt 只要在中途分岔（第 3 輪保留了第一個來源，`stable_prefix = 1`）就整段重算。§3 在 Ollama 0.34.1 + e2b 上量到的部分重用（第 2 輪 1622 tokens），在 0.35.0 的 llama-server 上不成立；完全相同的 prompt 仍會命中（首字 0.46s）。`--swa-full` 可以換回任意 prefix 重用，但 16K context 下 sliding 層的 KV 會多約 5 GB，這台放不下。

### 8.7 主機本身

- 沒有使用者登入時 `audiomxd` 常駐約 0.7 顆核心、`configd` 約 0.4 顆，從開機就開始。
- 持續負載下 Cloudflare Tunnel 的 QUIC 連線每 ~30s 逾時（`no recent network activity`），SSH 斷線；區網 ping 與 SSH 正常。

### 8.8 頁面依內容決定是否送圖（#38，2026-10-03）

`v0.5.0` 與 `feat/page-vision-gating`（只換 `knowva-server`）在同一台、同一組設定（檢索模型 `KM_IDLE_SECS=600` 常駐、rerank 1 tile、chat 頁圖長邊 1280）下各跑一次。知識庫同樣 8 份文件；報告在新版重新上傳，其餘 5 份頁面文件（DOCX／XLSX／PPTX／PNG／JPG）是 `v0.5.0` 入庫的，仍是 `needs_vision = true`。每次先暖機一題，再單輪問三題表格題，最後跑 `bench/e2e.ts --reuse` 三輪對話：

| | v0.5.0 | 本分支 |
| --- | --- | --- |
| 14 頁上傳到 ready（含摘要） | 34.2s（摘要 18.4s） | 16.1s（摘要 12.1s） |
| 看圖頁 | 14 | 1（第 7 頁，`raster`） |
| 第 1 頁：作者、QA、起訖日期 | 首字 24.4s | 13.7s |
| 第 2 頁：目錄 3.2 底下的框架 | 25.7s | 13.2s |
| 第 4 頁：Task 1 的四個 Goals | 24.3s | 14.5s |
| e2e 第 1／2／3 輪首字 | 25.5／25.3／30.2s | 13.2／18.6／17.2s |
| 每次 rerank（50 筆） | 8.4–9.0s | 6.6–7.2s |
| 每次回答附圖 | 每個頁面來源都附 | 0–3 張 |

- 兩版的六題都答對（Felix Strohmeier／Dr. Violeta Damjanovic-Behrendt、AKKA／JIAC、四個 Goals、IL050、Contract-Net 流程、Arduino／RIOT-OS），第一名頁面都相同；表格題與 e2e 第 3 輪的前三名相同，e2e 第 1、2 輪的第 2–3 名有變動。
- rerank 只快約 20%：50 筆候選大多本來就是文字 unit，新版每次只剩 1–6 筆圖片 passage，主要來自另外 5 份舊文件。
- 新版有附圖的回答，是因為來源裡有看圖頁：第 4 頁那題附的是報告第 7 頁，e2e 第 2 輪的 3 張都來自舊文件。舊文件要重新上傳才會套用新規則。
- `v0.5.0` 第 3 輪生成期間另有一筆不屬於本次量測的查詢（server log 17:09:56），首字 30.2s 可能偏高。

### 8.9 範例驗收（`bench/examples.ts`，2026-10-03）

`example_file/` 的 10 個檔案全部重新上傳到 `dev-page-vision-20261003`，再問 `bench/examples.json` 的 34 題。上傳階段 10/10 通過：8 份 ready、頁數正確，送圖的只有報告第 7 頁與兩張圖片，MP4／AVI 回 415。STP 入庫 61.2s，其餘 7–18s。

| 類別 | 通過 | 引用到正確來源 |
| --- | --- | --- |
| PDF 文字頁 | 10/10 | 9/10 |
| PDF 看圖頁（第 7 頁，含只能從照片回答的 Figure 1） | 2/2 | 2/2 |
| 追問 | 1/1 | 1/1 |
| DOCX／XLSX／EML／STP | 9/9 | 9/9 |
| PPTX | 0/1 | 0/1 |
| 圖片 | 1/4 | 4/4 |
| 找不到答案 | 6/6 | — |
| 知識庫總覽 | 1/1 | — |

- 第一輪有一段時間另一個 session 在用推論服務，還把常駐的 Gemma 換掉：`pdf-p13-jiac` 逾時、`pdf-p14-reference` 首字 129.7s。Mac 空下來後重跑這兩題都通過（首字 13–14s），表內是重跑後的結果；`pdf-p14-reference` 重跑時引用成第 9 頁。其餘題目首字中位數 21.1s。
- PPTX：投影片只有 Lorem ipsum，問「這份簡報的內容」時沒有取回它。檔名不在檢索內容裡，沒辦法用檔名找到這份文件。
- 圖片：`png-logo` 單張直接問 Gemma 也讀錯（IDRA 讀成「SPX」，問答時是「1924」），`jpg-task` 單張也只看出「黃色籃框」、沒認出麵包，都是 500px 原圖上的小細節。`png-arm-color` 單張問答對（紅色），但問答時同一個 context 有兩張圖，就回答「無法確認」。
- STP：266 個 unit 中有 245 個是 STP 的幾何資料段。192 個來源欄位中 81 個是 STP 段，20 題與 STP 無關的問題也帶到 STP 段。

### 8.10 真實文件驗收（`bench/realistic.json`，2026-10-03）

9 份公開文件：行政院智慧機械推動方案簡報 PDF、OSHA 3080 手冊、NASA 掃描 Tech Brief、高雄市勞檢處起重機自動檢查表 DOCX、職安署重大職災統計 XLSX、OSHA Harwood「Machine Guarding」75 頁 PPTX、職安署兩張宣導海報、open62541 README。`--fetch` 依網址下載並核對 sha256，跟 §8.9 的 8 份範例放在同一個知識庫，問 39 題。上傳 9/9 通過，問答 32/39，首字中位數 21.3s。

| 類別 | 通過 | 引用到正確來源 |
| --- | --- | --- |
| PDF（中文簡報／英文手冊／掃描）、追問 | 13/13 | 13/13 |
| DOCX、Markdown、跨文件 | 8/8 | 8/8 |
| PPTX | 5/5 | 5/5 |
| XLSX | 0/4 | 4/4 |
| 海報圖片 | 1/4 | 2/4 |
| 找不到答案、知識庫總覽 | 5/5 | — |

送圖的頁：

| 文件 | 頁數 | 看圖頁 | 原因 |
| --- | --- | --- | --- |
| 智慧機械推動方案 | 6 | 4 | 第 3、4 頁是圖表；第 5、6 頁只有兩條約 17:1 的版型橫幅 |
| OSHA 3080 | 32 | 4 | 封面、封底、整頁掃描的第 12 頁 |
| NASA Tech Brief | 1 | 1 | 掃描頁（有 OCR 文字層） |
| DOCX／XLSX | 1／9 | 0 | — |
| Machine Guarding PPTX | 75 | 75 | 70 頁 `raster`：母片上 5 張版型圖（合計約 22%）每頁都有；3 頁 `unreliable-text`：Wingdings 項目符號是 U+F07D，字少的頁超過 5% |

- XLSX：四題都取回並引用了正確的表，但數字對錯欄。LibreOffice 把寬工作表依欄切成好幾頁，例如營建業 112 年的死亡人數 151 在第 7 頁，那一頁沒有年份和表頭。把單頁送給 Gemma 比較：只送文字 1/4，圖 + 文 2/4，兩種都救不回被切開的欄。改用 `calc_pdf_Export` 的 `SinglePageSheets` 時，3 張工作表各輸出成一頁，同一列的所有欄位都在一起。
- 海報：問「機械災害預防海報」的兩題都沒取回這張圖（取回的是 PPTX 與另一張海報）；其中一題把起重吊掛海報的內容當成機械海報回答。起重吊掛海報有取回，但 2504×3531 的原圖縮到長邊 1280 後，底部小字讀錯。
- 在本機估算：如果不計重複出現在半數以上頁面的同位置同尺寸圖片，也不把 U+F000–U+F0FF 的符號字型字元算成亂碼，PPTX 的看圖頁會從 75 降到約 11（只估點陣圖與文字兩條規則）。

### 8.11 校準後重跑（2026-10-03）

`dev-page-vision-20261003-2` 在判斷規則加上符號字型項目符號與頁面家具（README「頁面是否送圖」），其餘設定與 §8.9、§8.10 相同。先跑 `bench/realistic.json`、再跑 `bench/examples.json`，兩次都用 `--replace` 重新上傳，問答時知識庫都是同樣的 17 份文件。題組在兩次之間有改：每題都要引用正確來源才算通過；`ey-p5`、`osha-p14`、`md-license` 三題改了題目或補上同義關鍵字，因為原本的題意有歧義或關鍵字太窄。

| | §8.9／§8.10 | 本次 |
| --- | --- | --- |
| Machine Guarding（75 頁）看圖頁 | 75 | 11：9 頁 `raster`，第 1、75 頁 `sparse-text` |
| Machine Guarding 上傳到 ready | 93.4s | 28.1s |
| 其他 16 份的看圖頁 | — | 與 §8.9、§8.10 完全相同 |
| Machine Guarding 五題首字 | 21.8–23.0s | 10.2–11.2s |
| realistic 問答 | 32/39 | 32/39 |
| examples 問答 | 30/34（含重跑的兩題） | 31/34 |
| 首字中位數 realistic／examples | 21.3／21.1s | 18.5／19.7s |

- Machine Guarding 本機轉檔加上家具比對共 7–10s。11 個看圖頁裡，9 頁是投影片自己的照片，第 1、75 頁字太少。另外 64 頁拼成縮圖人工檢查過，都只有標題、條列文字和母片裝飾。
- Machine Guarding 五題都通過；四題的第一名來源與 §8.10 相同，`pptx-guard-types` 原本的第一名第 23 頁仍在前三。
- 沒過的題目都在 §8.9、§8.10 列過：realistic 的 XLSX 四題數字對錯欄（#40）、海報三題讀不出小字；examples 的 PPTX 一題（只有 Lorem ipsum，檔名不在檢索內容裡，#41）與 `png-logo`、`png-arm-color`。examples 的 `jpg-task` 這次通過，但圖片上傳本來就一律看圖、送進模型的內容沒變，只是回答本身有波動。
- `--replace` 刪舊 STP 時，`UPDATE documents SET status = 'failed'` 的慢查詢警告是刪除流程先把文件撤出檢索；245 個 unit 的 BM25 統計要 2.7s，不是錯誤。

### 8.12 Gemma 同時處理數（2026-10-05）

`dev-page-vision-20261003-2`、知識庫同 §8.11 的 17 份文件，其餘設定同 README（`KM_IDLE_SECS=600`、rerank 1 tile、chat 頁圖長邊 1280）。`bench/parallel-sweep.sh` 依序把 `OLLAMA_NUM_PARALLEL` 設成 1、2、3（三組都 `LLAMA_ARG_CTX_CHECKPOINTS=2`），每組重啟 Ollama、暖機一題，再跑 `bench/concurrency.ts`：1–4 人同時送出單輪問題，每種人數兩輪，共 20 題、題題不同，三組問同一批題目、同樣的組合。每 2 秒記一次 swap 與記憶體，app 與 Ollama 的 log 拆出各階段時間。

全部答完的時間（括號內是平均／最慢的首字）：

| 同時人數 | 1 格 | 2 格 | 3 格 |
| --- | --- | --- | --- |
| 1 | 19.1s（16.4／16.4s） | 19.3s（17.0／17.8s） | 20.8s（18.4／19.1s） |
| 2 | 38.9s（29.7／37.2s） | 40.1s（36.3／38.5s） | 48.0s（44.2／48.4s） |
| 3 | 69.3s（45.5／65.7s） | 67.6s（47.7／64.0s） | 85.1s（72.3／76.8s） |
| 4 | 85.3s（59.4／84.3s） | 86.3s（64.3／86.0s） | 104.3s（83.9／100.2s） |
| 每分鐘答完（2–4 人） | 2.6–3.1 題 | 2.7–3.0 題 | 2.1–2.5 題 |
| swap 最高 | 2.9 GiB | 4.7 GiB | 7.0 GiB |

- 一格時每題的 GPU 時間：rerank 7.3s（50 筆候選、平均 7 筆圖片 passage）、prefill 8.2s（3.3K tokens，約 400 tok/s）、生成 40–90 tokens 約 3–5s。推論服務的 embed／rerank 共用一把鎖，一次只做一題，4 人同時時 rerank 平均 29s、最慢 36s。
- 2 格：兩題同時 prefill 時每題 236 tok/s，合計和一格相同；生成每題 13 tok/s。吞吐沒有增加，首字反而平均晚一些，因為兩題一起做完而不是先做完一題。
- 3 格：swap 到 7.0 GiB，生成每題 8.7–9.3 tok/s，3–4 人同時時 rerank 平均 32–38s，吞吐比一格少約 20%。
- KV cache 每格 896 MiB（全域層 256 + sliding-window 層 640）；`/api/ps` 的大小 1／2／3 格為 7.97／8.67／9.41 GB。三組 Ollama log 都沒有 OOM，可用記憶體最低都是 0.1 GiB；測試開始前 swap 已有 1.6 GiB。
- 60 題答對 59 題。沒過的是 2 格、3 人那輪的 `jpg-brand`（KUKA 答成 KURODA），一格與三格同一題都對。回答請求沒有指定 temperature，用的是 Ollama 的預設取樣，§8.11 也記過圖片題的回答會波動。
- 這組題目的回答都短，每題時間由 rerank 與 prefill 決定；回答長的時候（同日另一筆 619 tokens 生成 34.6s），多格合批生成的效果沒有在這裡量到。

### 8.13 Gemma 改跑 MLX、拿掉 rerank（2026-10-05）

Ollama 0.35.0 在 Mac 上另有 MLX runner（`gemma4:12b-mlx`，nvfp4 權重），QAT 版（`gemma4:12b-it-qat`）走的是 llama.cpp。知識庫、題目與其他設定同 §8.12（一格）。四組依序跑，每組先卸載模型，再跑 `bench/concurrency.ts`（同 §8.12 的 20 題）、`bench/examples.ts` 與 `bench/examples.ts --dataset bench/realistic.json`：

- `mlx-1280`、`mlx-1152`：`GEMMA_4_MODEL_ID=gemma4:12b-mlx`，chat 頁圖長邊 1280／1152。
- `qat-k6`：QAT、`RERANK_CANDIDATE_TOP_K=6`。rerank 只收到 RRF 前 6 名，等於由 RRF 決定送進 context 的來源，rerank 只重排順序。
- `qat-base`：QAT 原設定，只跑兩組問答；表中 QAT 的同時處理數字取 §8.12 一格。

切換設定、取樣記憶體、跑完自動還原 `.env` 與推論服務 plist 的腳本（`mlx-e2e.sh`），以及比較各組的 `compare.ts`，和結果放在一起：`knowva-mac:~/knowva-inference/bench-results/mlx-20261005-214342/`。

| | QAT | MLX 1280 | MLX 1152 | QAT、候選 6 |
| --- | --- | --- | --- | --- |
| 每分鐘答完（1–4 人） | 2.6–3.1 | 3.2–4.1 | 3.2–3.9 | 3.8–5.2 |
| 4 人同時，平均／最慢首字 | 59.4／84.3s | 53.4／69.0s | 56.2／77.8s | 36.4／62.4s |
| 同時處理那 20 題答對 | 20/20 | 20/20 | 19/20 | 12/20 |
| rerank 平均／最慢（同時處理那 21 題，含排隊） | 22.5／35.9s | 21.6／32.8s | 21.6／32.6s | 4.8／14.2s |
| examples | 31/34 | 30/34 | 28/34 | 26/34 |
| realistic | 32/39 | 31/39 | 32/39 | 26/39 |
| 首字中位數 examples／realistic（一次一題） | 19.4／18.9s | 16.8／16.8s | 17.1／16.3s | 11.3／11.8s |
| swap 最高 | 2.9 GiB（`qat-base` 3.2） | 12.5 GiB | 12.1 GiB | 2.3 GiB |

- 單次 chat（`bench/chat.ts`，6 頁，每格 3 輪）：MLX prefill 546–566 tok/s、生成 26.8–29.5 tok/s；QAT 384–394、17.7–17.9 tok/s。
- MLX 的頁圖 token 分 280／560／1120 三檔：長邊 1280 的頁約 1104 token，896–1152 約 539 token；QAT 在 1280 是 570 token。1152 的 token 少一半，但首字與吞吐和 1280 差不多。
- MLX runner 載入後 `msg=memory held` 是 7.81 GiB，之後隨 prefix cache 一路漲到 15.0–15.4 GiB（peak 17.5 GiB），兩組都一樣；結束時 `/api/ps` 的大小是 16.1 GB（QAT 8.0 GB）。Ollama 0.35.0 的 `OLLAMA_*` 環境變數裡沒有限制它的設定，QAT 那條路有 `LLAMA_ARG_CACHE_RAM=2048`。`mlx-1152` 跑到一半時的 `top`：Ollama MLX runner 16 GB（其中 7.2 GB 被壓縮）、推論服務 10 GB、Colima VM 3.1 GB。四組都沒有 OOM 或 Metal 錯誤，可用記憶體最低 0.06–0.08 GiB。
- 和 `qat-base` 相比多錯的題目：`mlx-1280` 是 `png-arm-color`、`cross-crane`、`poster-machine-measures`；`mlx-1152` 是 `jpg-brand`（KUKA 讀成 FANUC）、`nf-press-tonnage`、`png-arm-color`、`poster-machine-measures`。反過來，`qat-base` 沒過的 `osha-follow-up` 在兩組 MLX 都通過。
- `qat-k6` 比 `qat-base` 多錯 14 題（examples 5、realistic 9），其中 11 題是正確來源沒進前 6 名（`not-retrieved`）。這組只拿掉 rerank，BM25 與 RRF 仍在，送進 context 的是 RRF 前 6 名，所以結論只適用於「BM25＋RRF 不加 rerank」；只用 dense 的量測見 §8.16。多錯的題目包括圖片與 EML 題、OSHA 單頁題、PPTX 統計與機器人事故題、海報題。它多答對三題 XLSX：`xlsx-deaths-112` 的 6 個來源與順序和 `qat-base` 完全一樣，差在回答本身；`xlsx-peak-month`、`xlsx-construction` 各有一到兩頁來源不同，`xlsx-construction` 的答案引自 `qat-base` 沒選進的第 7 頁。
- `qat-k6` 一次一題的首字比 `qat-base` 快約 8s：rerank 平均 1.0s（`qat-base` 7.4s），每題 context 的圖片平均 1.7 張（`qat-base` 3.0 張），prefill 也比較短。
- 一次一題時 rerank 平均 7.4s（`qat-base`，50 筆候選），和 §8.12 一格的 7.3s 相近；多人同時時平均多出約 14s，包含 embed／rerank 共用一把鎖的排隊與同時跑 Gemma 的 GPU 競爭，這次沒有拆開量。
- 上傳壓力（2026-10-06，`gemma4:12b-mlx`、頁圖 1280）：先跑一次 realistic（32/39）把 prefix cache 填滿，再用 `concurrency.ts --levels 4 --rounds 3` 連續三輪 4 人同時提問，同時依序上傳 Machine Guarding（75 頁）與 OSHA 3080（32 頁）的複本，ready 後就刪掉。每輪全部答完 69.0–81.3s（平均 77.0s），平均／最慢首字 59.0／79.8s（沒有上傳的 `mlx-1280` 是 70.5s、53.4／69.0s），12 題答對 11 題（`jpg-brand` 又讀成別的廠牌）。兩份上傳分別 64.4s、66.4s ready（§8.11 單獨上傳 Machine Guarding 是 28.1s）。swap 最高 11.8 GiB；系統記憶體壓力有 4 個取樣點（約 8 秒）到 critical，都在 Gemma 寫上傳摘要或 4 題同時 prefill 時，wired 從約 9.6 GiB 升到 15.4 GiB。Ollama、推論服務、app 的 log 都沒有錯誤；系統 log 只有 macOS 把閒置的系統 daemon 以 idle-exit 結束，沒有砍到這三個服務。腳本與結果在 `knowva-mac:~/knowva-inference/bench-results/stress-20261005-235428/`。
- MLX prefix cache 上限 2 GiB（2026-10-06）：`maxPagedOutBytes` 是常數，上游讓它可設定的 PR #18556 沒有被接受（issue #18131 仍開著）。在 knowva-mac 以 v0.35.0 原始碼（cc40693）套上 #18556，只用 `go build` 重編 Go 執行檔（Go 1.26.8），MLX 與 llama.cpp 沿用 release 附的函式庫；設 `OLLAMA_MLX_MAX_PAGED_OUT_BYTES=2147483648`，其餘同 `mlx-1280`，結果在 `knowva-mac:~/knowva-inference/bench-results/km1-20261006-002433/`。

  | | `mlx-1280`（官方 8 GiB） | 上限 2 GiB |
  | --- | --- | --- |
  | `msg=memory held` 最高／peak | 15.4／17.5 GiB | 9.35／11.4 GiB |
  | `/api/ps` 大小 | 16.1 GB | 9.8 GB |
  | 每分鐘答完（1–4 人） | 3.2–4.1 | 3.3–4.0 |
  | 4 人同時，平均／最慢首字 | 53.4／69.0s | 48.4／65.1s |
  | 同時處理那 20 題答對 | 20/20 | 19/20 |
  | examples／realistic | 30/34、31/39 | 29/34、32/39 |
  | 重用超過 300 token 的請求（共 96 個） | 6 | 1 |
  | 追問首字 `pdf-follow-up`／`osha-follow-up` | 19.1／17.2s | 18.7／17.2s |
  | swap 最高 | 12.5 GiB | 5.2 GiB |
  | 記憶體壓力 critical | —（未記錄） | 0 次 |
  | 上傳壓力：每輪平均、平均／最慢首字 | 77.0s、59.0／79.8s | 75.2s、56.7／76.9s |
  | 上傳壓力：ready 時間 | 64.4、66.4s | 54.3、69.4s |
  | 上傳壓力：swap 最高、critical | 11.8 GiB、4 個取樣點 | 4.8 GiB、0 次 |

  答錯的題目兩組差在 `jpg-brand`（2 GiB 這組在同時處理與 examples 各錯一次）與 `cross-crane`（2 GiB 這組通過）；上傳壓力的第二輪兩組也都錯在 `jpg-brand`。上限 2 GiB 後，追問少了前一輪的重用，但首字時間沒有變。兩組的 log 都沒有錯誤。

### 8.14 Rerank 與 Embed 改跑 MLX（#47，2026-10-06）

把 Nemotron Rerank VL 移植到 MLX 0.32.3（直接讀原始 `model.safetensors`），和推論服務現行的 PyTorch 2.14.1 MPS 比。前處理和模型的 transformers processor 逐 token、逐 pixel 相同（tiles 1 與 6、文字／圖／圖＋文、截斷都比過）。

先用合成候選比（正式服務改跑 MLX 之後擷取的實際候選在本節後段）：`examples.json` 與 `realistic.json` 裡來源是 PDF 或圖片的 34 題（兩題追問的每輪各算一題，共 36 題），以 BM25 從 4 份 PDF（共 53 頁）與 4 張圖片選出每題前 50 名；其中 7 筆帶頁圖（上傳的圖片與 iiot 報告第 7 頁優先，不足依名次補），頁圖是 180 DPI、長邊上限 1600 的 PNG。送進模型的方式照後端與推論服務：同一 modality 一批、每批最多 8 筆、循序，`KM_RERANK_TILES=1`，token 上限依 modality。每組各自一個程序、先跑 1 題暖機；每組開跑前推論服務都沒有載入模型，Gemma（MLX、上限 2 GiB）常駐。這份候選每題 rerank 5.87s，比 §8.13 實際候選的 7.4s 短。

| 每題 50 筆 | 平均 | 與 fp32 參考的 logit 最大差 | 前 6 名集合不同 | 前 6 名順序不同 | 使用中記憶體峰值（不含快取） |
| --- | --- | --- | --- | --- | --- |
| PyTorch MPS fp32（參考） | 18.16s | — | — | — | 9.12 GiB |
| PyTorch MPS fp16（現行） | 5.87s | 0.0106 | 0/36 | 1 | 4.89 GiB |
| PyTorch MPS fp16，逐筆不補 padding | 4.43s | 0.0094 | 0/36 | 1 | 3.89 GiB |
| MLX fp32（`MLX_ENABLE_TF32=0`） | 18.33s | 0.00001 | 0/36 | 0 | 7.69 GiB |
| MLX bf16 | 5.03s | 0.0776 | 0/36 | 2 | 3.85 GiB |
| MLX fp16 | 5.09s | 0.0103 | 0/36 | 0 | 3.87 GiB |
| MLX fp16，逐筆不補 padding | 3.79s | 0.0086 | 0/36 | 0 | 3.83 GiB |

- 同一種批次方式下，MLX fp16 比 PyTorch MPS fp16 快 1.15 倍（補 padding）與 1.17 倍（不補）。最快的 MLX 比現行快 1.55 倍，其中 1.33 倍來自不補 padding，PyTorch 也做得到（#48）。#47 的門檻是 2 倍。
- 這台的 PyTorch MPS 與 MLX 跑 matmul 一樣快（TFLOPS，`deploy/mac-mini/nemotron-vl-mlx/tools/matmul_peak.py`）：

  | M×K×N | MLX fp16 | MPS fp16 | MLX bf16 | MPS bf16 |
  | --- | --- | --- | --- | --- |
  | 4096×4096×4096 | 18.0 | 18.7 | 19.1 | 18.7 |
  | 19200×2048×8192 | 18.2 | 18.8 | 19.6 | 18.8 |
  | 600×2048×8192 | 13.6 | 16.7 | 13.8 | 15.6 |
  | 600×8192×2048 | 10.2 | 13.4 | 12.1 | 13.0 |

  每題有效 token 約 2.0 萬、補齊後約 3.2 萬（有效 63%），約 79 TFLOP；PyTorch MPS fp16 每題 5.87s 相當於約 13.5 TFLOPS。
- 精度：PyTorch MPS fp32 和 CPU fp32 在 4 個固定批次上差 1e-5，MLX fp32 和它們也差 1e-5。MLX 在 M6 上預設讓 float32 matmul 走 TF32（`MLX_ENABLE_TF32`），沒關掉時差 0.0036，比 PyTorch fp16 還大。
- 記憶體要看程序的 phys_footprint（`top` 的 MEM）：macOS 的 RSS 不算 GPU buffer，`mx.get_peak_memory()` 不算 MLX 的 buffer cache。MLX 的 buffer cache 預設沒有實際上限，每筆長度不同，釋放的 buffer 被留著重用，36 題後長到 13.8 GiB。單獨跑 rerank、逐筆不補 padding、fp16（`rerank-cache-20261006-030622/`）：

  | | 每題 | 程序 footprint |
  | --- | --- | --- |
  | MLX，預設 cache | 3.80s | 17 GB |
  | MLX，cache 上限 0 | 3.99s | 3.6 GB |
  | MLX，cache 上限 256 MB | 3.88s | 3.8 GB |
  | MLX，cache 上限 1 GB | 3.88s | 4.6 GB |
  | PyTorch MPS | 4.34s | 5.0 GB |

- 推論服務本身（`server.py` 加 `KM_RERANK_BACKEND`，在 9502 埠另跑一份，正式服務沒動）：先 embed 一次查詢讓 embed 常駐，等同有人在問的狀態，再用 HTTP 照後端格式送 36 題（`nemotron-vl-mlx/tools/knowva/server-test.sh`）。

  | `KM_RERANK_BACKEND` | 每題 rerank | footprint 峰值 | 跑完時 footprint | 記憶體壓力 |
  | --- | --- | --- | --- | --- |
  | torch | 5.90s | 10.0 GB | 9.2 GB | normal |
  | mlx，`KM_MLX_CACHE_MB=256` | 3.96s | 9.9 GB | 8.4 GB | normal |
  | mlx，沒設上限 | 4.08s | 23 GB | — | warn，swap 增加 4.1 GB |

  embed 仍跑 PyTorch，程序裡還是有 PyTorch runtime 與 MPS 快取，所以峰值幾乎一樣。結果在 `rerank-server2-20261006-025831/`（沒設上限）與 `rerank-server3-20261006-031952/`。
- MLX 第一版載入時同時持有 bf16 原檔與 fp16 副本，可用記憶體一度剩 0.15 GiB、記憶體壓力 critical 2 秒、swap 增加 2.3 GiB；改成逐個 tensor 轉換後沒有再出現。
- 移植與量測工具在 `deploy/mac-mini/nemotron-vl-mlx/`（`tools/knowva/build_sets.py` 組候選、`bench_sets.py` 量時間與 logit、`compare_runs.py` 比較、`run-sets.sh` 依序執行並取樣記憶體、`cache-sweep.sh` 與 `server-test.sh`）。以上結果都在 `knowva-mac:~/knowva-inference/bench-results/` 底下，36 題的比較在 `rerank-sets-20261006-015536/` 與 `rerank-sets2-20261006-023640/`。

正式推論服務 09:40 改成 `KM_RERANK_BACKEND=mlx`、`KM_MLX_CACHE_MB=256`（embed 仍是 PyTorch），照 §8.13 跑一輪 e2e（`mlx-rerank-e2e-20261006-094336/`），對照同一天 PyTorch 的 `km1-20261006-002433/`。這一輪 app 的 rerank 請求經過 `tools/knowva/capture-proxy.ts` 轉送，原樣記下請求與回應（`capture-e2e.sh`，跑完路由改回直連）。

| | MLX rerank | PyTorch（km1） |
| --- | --- | --- |
| 1／2／3／4 人同時提問，每分鐘答完 | 5.15／4.44／3.98／4.18 | 4.03／3.86／3.30／3.65 |
| 同時提問時每題 rerank 平均／最長 | 14.6s／26.2s | 21.5s／33.9s |
| examples／realistic 答對 | 30/34、33/39 | 29/34、32/39 |
| 首字中位數 examples／realistic | 13.4s／12.9s | 16.8s／16.2s |
| 記憶體壓力 warn 的取樣比例 | 2.8% | 36% |
| 跑的期間 swap | 1.5→2.7 GiB | 1.4→5.2 GiB |

- 同時提問時，各人的 rerank 批次在推論服務裡排隊（一次一個 forward），所以每題 rerank 比單人時長；單人時 MLX 每題約 5.8s。
- swap 最高 9.7 GiB 是開跑前留下的（第一筆取樣時推論服務還沒載入模型），跑的期間一路降到 1.5 GiB 再回升。
- 答對題不同的有 jpg-brand、xlsx-peak-month（MLX 對、km1 錯）與 poster-machine-causes、poster-hoisting-hazards（兩邊都錯，缺的關鍵字不同）。用下面的重播比，只有 poster-machine-causes 的前 6 名和 PyTorch fp16 不同（第 3、4 名對調）；其他三題前 6 名相同，差異來自 rerank 之後。

擷取到的 520 個請求是 73 題（examples 34、realistic 39）、每題 50 筆的實際候選，3650 筆裡 524 筆帶頁圖（24 張不同的圖），以 `tools/knowva/capture_to_sets.py` 轉成 `bench_sets.py` 的格式。MLX 照服務設定（TF32 開、cache 256 MB）重播時，73 題的 logit 和服務當時回傳的完全相同，所以重播能代表正式服務。結果在 `rerank-captured-20261006/`：

| 實際候選，每題 50 筆 | 平均 | 與 fp32 參考的 logit 最大差 | 前 6 名集合不同 | 前 6 名順序不同 | 程序 footprint |
| --- | --- | --- | --- | --- | --- |
| PyTorch MPS fp32（參考） | 22.12s | — | — | — | 12 GB |
| PyTorch MPS fp16（原本的服務） | 7.18s | 0.0082 | 2/73 | 0 | 7.3 GB |
| MLX fp16（服務設定） | 4.91s | 0.0104 | 1/73 | 1 | 4.0 GB |

- 前 6 名集合不同的題目，參考值的第 6、7 名只差 0.0018 與 0.0003；MLX 不同的那一題，PyTorch fp16 也一樣換掉。MLX 和 PyTorch fp16 之間 logit 最大差 0.0107，前 6 名集合不同 1 題（pptx-content，第 6 名）、順序不同 1 題（poster-machine-causes）。
- MLX 比 PyTorch fp16 快 1.46 倍（中位數 1.54 倍）。實際候選比合成的長（PyTorch 每題 7.18s 對 5.87s）。
- TF32 開或關，fp16 的 logit 完全相同：fp16 模型裡只有 score head 是 float32 matmul。
- 頁面文字的 `\r\n` 要原樣送：工具一開始用文字模式讀候選，換行被改成 `\n`，重播和服務的 logit 差到 0.45、前 6 名 18 題不同。合成候選（`build_sets.py`）同樣把 `\r\n` 換成 `\n`；各組拿到的文字相同，前面的比較仍成立，只是和 app 送出的差在換行。

Embed VL 用同一套移植（`Embedder`）：和 sentence-transformers 一樣的 `query: `／`passage: ` prompt、4096 token 截斷、不正規化的 mean pooling；前處理和原版逐 token、逐 pixel 相同（`tools/check_embed_processing.py`）。量測用 `bench-pages/` 的 14 頁（iiot 報告，頁圖與文字層）和 `realistic.json` 前 20 題，每筆各送一次，和後端一樣；`KM_MAX_TILES=6`，每組各自一個程序，開跑前推論服務沒有載入模型（`tools/knowva/run-embed.sh`，結果在 `embed-20261006/`）。

| 每筆 | 查詢 | 文字頁 | 頁圖 | 頁圖＋文字 | 程序 footprint | 與 PyTorch fp32 的 cosine 最小值 |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch MPS fp32（參考） | 0.054s | 0.196s | 2.78s | 2.99s | 9.9 GB | — |
| PyTorch MPS fp16（原本的服務） | 0.026s | 0.068s | 1.00s | 1.06s | 4.7 GB | 0.999992 |
| MLX fp16 | 0.022s | 0.055s | 0.84s | 0.91s | 4.3 GB | 0.999990 |
| MLX fp32（`MLX_ENABLE_TF32=0`） | 0.038s | 0.189s | 2.74s | 2.94s | 7.4 GB | 1.000000 |

- 用每題查詢向量對 14 頁取 cosine 前 5 頁（`tools/compare_embed.py`）：MLX fp16 和 PyTorch fp16 一樣，在文字頁、頁圖、頁圖＋文字各有 0–1 題的前 5 名集合和 fp32 參考不同。
- 不用重新 embed：MLX fp16 的查詢向量去比 PyTorch fp16 的頁向量（也就是資料庫裡現有的），三種頁的前 5 名都和 PyTorch fp16 自己比的完全相同（0/20）。
- 推論服務在 9502 埠另跑一份、embed 與 rerank 都設 MLX：後端會送的四種輸入（查詢、文字、頁圖、頁圖＋文字）經 HTTP 拿到的向量和直接呼叫 `Embedder` 逐位相同；程序沒有載入任何 PyTorch 函式庫，兩個模型都載入後 footprint 6.95 GB（`embed-service-smoke-20261006/`）。

接著正式推論服務改成 `KM_EMBED_BACKEND=mlx`（rerank 已是 MLX），之後的 e2e 不改任何設定（`tools/knowva/acceptance-e2e.sh`，`mlx-all-e2e-20261006-112335/`）：examples、realistic、1–4 人同時提問，再上傳兩份文件的複本、同時 4 人各問 3 輪（`stress-upload.ts`，上傳完即刪除，和 km1 的上傳階段相同）。

| | PyTorch（km1） | Rerank MLX | 全 MLX |
| --- | --- | --- | --- |
| 1／2／3／4 人同時提問，每分鐘答完 | 4.03／3.86／3.30／3.65 | 5.15／4.44／3.98／4.18 | 5.11／4.36／3.85／4.34 |
| 同時提問時每題 rerank 平均／最長 | 21.5s／33.9s | 14.6s／26.2s | 14.8s／25.3s |
| examples／realistic 答對 | 29/34、32/39 | 30/34、33/39 | 28/34、32/39 |
| 首字中位數 examples／realistic | 16.8s／16.2s | 13.4s／12.9s | 13.3s／12.9s |
| 推論服務 footprint 中位數／最高 | — | 8.7／10.2 GB | 7.5／8.4 GB |

| 上傳 + 4 人各問 3 輪 | PyTorch（km1） | 全 MLX（兩次） |
| --- | --- | --- |
| 每分鐘答完 | 3.19 | 3.93、3.83 |
| 首字平均 | 56.7s | 44.4s、47.4s |
| 每題 rerank 平均 | 34.6s | 24.2s、25.0s |
| pptx（75 頁）可問答 | 54.3s | 88.4s、92.4s |
| PDF（32 頁）可問答 | 69.4s | 69.3s、68.3s |
| 記憶體壓力 warn 的取樣比例 | 57% | 20%（第一次） |
| 階段中 swap 增加 | 0.8 GiB | 0.4 GiB（第一次） |

- 全 MLX 和 rerank MLX 的 73 題取到的來源完全相同；和 km1 比只有 pptx-content、poster-machine-causes 兩題不同（就是重播裡前 6 名和 PyTorch fp16 不同的兩題，三輪都答錯）。答對題不同的 jpg-brand、jpg-task、xlsx-peak-month 來源都相同，是 Gemma 對同樣的來源給了不同答案，例如 jpg-brand 一次讀成 KUKA、一次讀成 FANUC。
- pptx 在有人提問時變慢：同一份 pptx 沒人提問時 23.1s 可問答，其中 embed 11.6s。有人提問時 embed 要 48s，推論服務的存取紀錄顯示上傳的 embed 和 4 人的 rerank 批次輪流取得 GPU（約 4 個 embed、4 個 rerank 交替），上傳的進度被 rerank 綁住；km1 沒留推論服務紀錄，無法比較它當時的順序。第二次在 `mlx-all-upload-20261006/`，沒有取樣記憶體。同一段時間答題變多、首字變快，PDF 沒有變慢。
- 推論服務不再載入 PyTorch，footprint 中位數少 1.2 GB、最高少 1.8 GB。上傳階段推論服務最高 8.3 GB。

之後權重改從 Hugging Face 的 `Laysi-ai/llama-nemotron-{embed,rerank}-vl-1b-v2-mlx` 下載（`setup/fetch-models.ts`）。這是 NVIDIA 權重換成移植的參數名稱與版面，bf16 數值不變。切換前後，推論服務對同一組請求的回應逐位相同：15 個 embed（查詢、文字、頁圖、頁圖＋文字）和 9 個 rerank（文字、頁圖、頁圖＋文字各 3 題）。切換前同一份服務連跑兩次也逐位相同，所以這個比對有效。切換後透過 app 問 realistic 第一題，答案正確，答案用到的頁排第一。比對工具是 `tools/knowva/service_outputs.py`，結果在 `hf-weights-switch-20261006/`。

### 8.15 EmbeddingGemma 2（MLX）對 Nemotron Embed VL（2026-10-08）

Google 10/06 發布的 EmbeddingGemma 2（740M：文字 270M、視覺 170M、音訊 300M；768 維，可截成 512／256／128；Apache 2.0），用 mlx-community 的 MLX 轉檔（`embeddinggemma-2-bf16` revision 1a4ffdd、`embeddinggemma-2-mxfp8` 0536fa4；mlx-vlm 3d87e884、MLX 0.32.3、transformers 5.19.0）和現行的 Nemotron Embed VL（MLX fp16、6 tiles，§8.14）比。只換 dense 這一段，BM25、RRF 與 Nemotron rerank 都不變。

- 語料：用 main（ed45d8b）的映像在開發機另起一份 Knowva、接全新的資料庫，依序上傳 §8.9 的 8 份範例與 §8.10 的 9 份真實文件；入庫的 Nemotron 向量經 SSH 通道由 knowva-mac 的正式推論服務計算，和正式環境同一條路。共 397 個 unit，25 個送圖（21 個頁圖＋文字、4 張只有圖），各文件的看圖頁和 §8.11 相同。
- 題目：兩份題組中有標準來源的 61 題，用最後一輪的提問檢索（後端只看最後一則使用者訊息，所以兩題追問的第二輪沒有上下文）。命中的定義同題組：檔名相符，有 `pages` 時頁碼也要相符（`null` 是文字 unit）。
- EG2 的輸入照模型卡：查詢加 `task: search result | query: `（另試 `task: question answering | query: `）；文字 unit 與不送圖的頁是 `title: none | text: ` 加內容；送圖的頁是同一段前綴、頁面文字、再接圖（順序同 Nemotron 的 text_image）；只有圖的不加前綴（前綴只用在文字）。圖片預設 280 soft tokens，另試 1120。
- 和官方 PyTorch fp32（sentence-transformers 6.1.0）在 11 個 unit（四種輸入都有）與 8 題 × 2 種查詢前綴上比 cosine：MLX bf16 最小 0.99984，mxfp8 最小 0.9982；1120 的圖片輸入和 280 的參考只有 0.91–0.98，預算確實改變了圖片的表示。

速度與記憶體（`embed_corpus.py`）：每筆一個 forward、循序，和入庫時每頁一個請求相同；每組各自一個程序，MLX cache 上限 256 MB 同推論服務；開跑前正式推論服務沒有載入模型，Gemma 常駐。`run_embeds.sh` 依序跑完全部各組；先前另跑過一輪，向量逐位相同，每種輸入的平均時間差最多 6.2%（那一輪的查詢時間沒算前處理，不比）。

| 每筆平均 | 文字（372） | 頁圖＋文字（21） | 只有圖（4） | 查詢 | 397 筆合計 | 程序 footprint 載入後／最高 | MLX 峰值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Nemotron MLX fp16，6 tiles（現行） | 0.087s | 0.870s | 0.772s | 0.021s | 53.9s | 3.25／4.40 GiB | 3.81 GiB |
| EG2 bf16，280 | 0.047s | 0.175s | 0.209s | 0.006s | 21.9s | 2.30／3.35 GiB | 2.11 GiB |
| EG2 bf16，280，不載音訊塔 | 0.047s | 0.177s | 0.204s | 0.006s | 21.9s | 1.44／2.49 GiB | 1.54 GiB |
| EG2 mxfp8，280 | 0.051s | 0.184s | 0.215s | 0.008s | 23.5s | 2.05／3.09 GiB | 1.86 GiB |
| EG2 mxfp8，280，不載音訊塔 | 0.051s | 0.185s | 0.216s | 0.008s | 23.6s | 1.19／2.26 GiB | 1.29 GiB |
| EG2 bf16，1120 | 0.046s | 1.216s | 1.244s | 0.006s | 47.8s | 2.29／4.73 GiB | 3.38 GiB |

- footprint 是程序的 phys_footprint（`proc_pid_rusage`），查詢與每筆時間都含前處理。EG2 的程序含 PyTorch 與 transformers（見下面的部署條件），Nemotron 的沒有，所以比記憶體要看 footprint，不是 MLX 峰值。
- Nemotron 這組用推論服務自己的 venv 與 `nemotron_vl_mlx`，算出的向量和資料庫裡的完全相同（cosine 1.0），所以量到的是正式服務的路徑。
- 不載音訊塔的向量和完整模型逐位相同。EG2 文字 unit 的 token 數中位數 1237。

檢索（`eval.ts`）：dense 取前 50，BM25 直接跑 `search.rs` 的 SQL，RRF k=60 取前 50，送正式推論服務的 Nemotron rerank（同一 modality 一批、每批 8 筆）取前 6。向量照資料庫轉成 halfvec：沒轉時 dense 第 11 名以後的順序和資料庫不同，轉了以後 61 題的前 50 名和後端 SQL 逐名相同（`dense-check.ts`；這個規模 Postgres 走 seq scan）。同一個實例用 `/api/chat` 問 4 題（pdf-p7-figure、poster-machine-causes、eml-subject、osha-p12-air-hose），後端回傳的 6 個來源和重現結果逐名相同（`backend-check.ts`）。

| 61 題，標準來源在前幾名 | dense 第 1 | dense 前 6 | dense 前 10 | dense 前 50 | dense MRR | RRF 前 6 | RRF 前 50（rerank 候選） | 最終第 1 | 最終前 6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nemotron（現行） | 48 | 57 | 59 | 60 | 0.851 | 45 | 60 | 51 | 60 |
| EG2 bf16，280 | 47 | 58 | 58 | 60 | 0.841 | 45 | 60 | 51 | 60 |
| EG2 bf16，280，256 維 | 47 | 57 | 58 | 60 | 0.830 | 43 | 59 | — | — |
| EG2 bf16，280，QA 前綴 | 46 | 58 | 59 | 60 | 0.827 | 46 | 60 | — | — |
| EG2 bf16，1120 | 48 | 56 | 57 | 60 | 0.834 | 45 | 60 | — | — |
| EG2 mxfp8，280 | 47 | 58 | 59 | 60 | 0.840 | 45 | 60 | — | — |

- rerank 之後完全打平；最終前 6 名找不到的都是 pptx-content（投影片只有 Lorem ipsum，#41）。兩邊的最終前 6 名有 28 題逐名相同，平均重疊 4.9 筆。每題 rerank 6.3s 對 6.7s。在這份語料上換 embed 不改變最終取到的來源，差別在入庫速度（合計 53.9s→21.9s，頁圖＋文字快 5.0 倍）與記憶體（程序 footprint 最高 4.40→3.35 GiB，不載音訊塔 2.49 GiB，已含 PyTorch 與 transformers）。問答時查詢 embed 只要 0.02s 對 0.006s，每題 rerank 6.3s，換 embed 對問答速度幾乎沒有影響。
- dense 的差異：EG2 把 §8.10 取不回的「機械災害預防海報」兩題從第 17、10 名拉到第 1 名，但 rerank 後兩邊都只排第 5、3 名。EG2 較差的是 eml-subject（dense 7→20，最終 3→5）、jpg-task（dense 1→6，最終都是第 1）與 pdf-follow-up 的第二輪（dense 1→11）。
- 語料只有 397 個 unit，dense 前 50 兩邊都已涵蓋 60/61。文件多的時候候選池要靠 dense 前段的名次，這次前 10 是 59 對 58，看不出差別。
- 1120 對文字頁沒有影響，圖片題反而較差（png-logo dense 2→12、png-arm-color 2→18）。截成 256 維 dense 幾乎不變，但 RRF 前 50 少 1 題。
- rerank 偏好送圖的 passage，兩個模型都一樣：答案在文字的 40 題，RRF 候選中送圖的 passage 佔 12.4%（EG2 17.5%），最終前 6 名佔 37.5%（EG2 42.9%）。例如 osha-p9-circular-saw 的前 3 名是 OSHA 的第 3 頁、封面與封底（都是看圖頁），答案的第 9 頁（文字頁）排第 4。EG2 帶進較多圖片 passage，送給 Gemma 的圖也會多一些。

部署條件：

- transformers 5.19 的 `EmbeddingGemma2Processor` 沒有 PyTorch 就載入不了。推論服務若改用 EG2 要裝回 PyTorch（§8.14 剛拿掉），或像 `nemotron_vl_mlx` 一樣把前處理改寫成 numpy。空程序逐步 import 的 footprint：mlx、numpy、PIL 0.02 GiB，加 torch 與 torchvision 0.23 GiB，再加 transformers 與 EG2 處理器（詞表 26 萬）0.59 GiB，前處理 5 張頁圖後 0.67 GiB；現行服務的 import（`nemotron_vl_mlx`、fastapi）是 0.05 GiB。改寫前處理只省得掉 torch 那一段，tokenizer 仍要載入。
- 向量從 2048 維改成 768 維，`document_units.embedding`（`halfvec(2048)`）與索引都要改，所有文件要重新 embed。rerank 仍是 Nemotron，它的權重還是要留。
- mlx-vlm 的 EG2 支援在 3d87e884 這個 commit，main 已合併但還沒發版（PyPI 最新 0.7.6）。

工具在 `deploy/mac-mini/bench/embeddinggemma2/`：`upload.ts` 依序上傳、`export-corpus.sh` 匯出語料與題目、`run_embeds.sh` 在 knowva-mac 上依序跑各組（`embed_corpus.py`）與 PyTorch 參考（`reference.py`）、`eval.ts` 重現檢索、`summarize.ts` 彙總、`dense-check.ts` 與 `backend-check.ts` 核對重現結果。結果在 `knowva-mac:~/knowva-inference/bench-results/embeddinggemma2-20261008/`，同一目錄另有入庫後的資料庫 dump（`knowva_eg2.dump`，`pg_dump -Fc`）與它的 FILES_DIR（`knowva_eg2-files.tgz`），還原後不必重新入庫就能重跑 `eval.ts`。

### 8.16 只用 dense 檢索（#60，2026-10-08）

現行 `/api/chat` 的檢索是 dense 前 50 ＋ BM25 前 50 → RRF（k=60）前 50 → Nemotron Rerank VL → 前 6 名。§8.13 的 `qat-k6` 只拿掉 rerank，BM25 與 RRF 仍在。這次比較三種只用 dense 的設定：

- `dense-6`、`dense-8`：dense 前 6／8 名直接送 Gemma，不呼叫 BM25，也不呼叫 rerank（`RETRIEVAL_MODE=dense`、`RERANK_ENABLED=false`、`RERANK_TOP_N=6／8`）。
- `dense-rerank-20`：dense 前 20 名送 rerank，取前 6（`RETRIEVAL_MODE=dense`、`RERANK_CANDIDATE_TOP_K=20`）。20 由下面的重播決定。

後端開關在分支 `feat/dense-retrieval-mode`：`RETRIEVAL_MODE=hybrid|dense`、`RERANK_ENABLED=true|false`，未設定時仍走 BM25＋RRF＋rerank（有單元測試）。同一分支也把 `rag_context.rs` 的 `MAX_RAG_SOURCES` 從固定 6 改成 `RERANK_TOP_N` 的上限 8：原本 `RERANK_TOP_N` 設 7、8 時，送進 context 的仍只有前 6 筆；預設 6 筆不變。第一次跑 `dense-8` 時還沒改這個上限，送給 Gemma 的其實是 dense 前 6 名，那一輪保留下來，當作 `dense-6` 的重跑。

- 語料：§8.15 的資料庫 dump 與 FILES_DIR（17 份文件、397 個 unit），還原到 knowva-mac 上另起的一份 Knowva（compose 專案 `knowva-dense60`，port 8081，有自己的資料庫與檔案）。app 是正式映像 `dev-page-vision-20261003-2` 換上這個分支編出的 `knowva-server`。推論服務與 Gemma 都用正式的那一套：`gemma4:12b-mlx`、prefix cache 上限 2 GiB、頁圖長邊 1280、embed 與 rerank 都跑 MLX。
- 正式環境不受影響：正式的 `~/knowva/.env`、`docker-compose.yml` 與推論服務 plist 都沒有動，實測前後的 sha256 相同。正式 app 全程沒有重啟，實測後 `/api/health` 正常，透過它問 `md-port` 答對。
- 題目：`examples.json`（34 題）、`realistic.json`（39 題），加上新的 `identifiers.json`（18 題）。新題組都是精確字詞查詢，來源都在現有的 17 份文件裡：
  - 出版品、補助與表格編號：OSHA 3104、SH-05029-SH8、表 7-2
  - 通訊錄：Birmingham, AL 的 Area Office、Region V、電話 (847) 297-4810
  - 法規條號：1910.212(a)(1)、1926.300(b)(2)、1928.57(b)(1)(i)、職安衛管理辦法第八十條
  - 型號、料號、SKU：ENC28J60、STEP 檔 93520-GA、SKU OCR-002
  - 縮寫：ANSI B11.19、AN128
  - 人名：Jason Bloomberg
  - 識別字與函式名：Message-ID、`UA_Server_runUntilInterrupt`
- 新題組評分：有兩題的關鍵字原本太窄（Gemma 答「以太網」「主護盾」），實測後放寬，四組都用放寬後的關鍵字重新評分（`rescore.ts`）。是否取到來源、是否引用正確，判斷方式不變。

重播（`replay.ts`）：直接在這個實例的資料庫上跑 `search.rs` 的 dense 與 BM25 SQL，查詢向量與 rerank 都打正式推論服務，作法同 §8.15 的 `eval.ts`。舊的 61 題 dense 名次和 §8.15 完全相同（第 1 名 48、前 6 名 57、前 10 名 59、前 50 名 60）。

| 標準來源進入 context | examples（27） | realistic（34） | 新題組（18） | 合計（79） | 排第 1 | 每題 rerank（一次一題） |
| --- | --- | --- | --- | --- | --- | --- |
| 現行 | 26 | 34 | 18 | 78 | 69 | 5.10s |
| dense-6 | 25 | 32 | 15 | 72 | 62 | — |
| dense-8 | 26 | 32 | 16 | 74 | 62 | — |
| dense 前 8 → rerank | 26 | 32 | 16 | 74 | 69 | 0.72s |
| dense 前 10 → rerank | 26 | 33 | 17 | 76 | 70 | 0.89s |
| dense 前 12 → rerank | 26 | 33 | 18 | 77 | 71 | 1.06s |
| dense 前 16 → rerank | 26 | 33 | 18 | 77 | 71 | 1.44s |
| dense 前 20 → rerank | 26 | 34 | 18 | 78 | 71 | 1.80s |
| dense 前 24 → rerank | 26 | 34 | 18 | 78 | 71 | 2.19s |

- 79 題中仍然沒有任何一題是 BM25 前 50 找得到、dense 前 50 漏掉的。三種設定都找不到的只有 `pptx-content`（投影片只有 Lorem ipsum，#41）；除了這題，dense 前 20 涵蓋全部題目。
- 新題組的前 6 名：BM25 16/18，比 dense 的 15/18 多；RRF 18/18。
- dense 前 6 名漏掉、但都在前 20 名內的六題（括號內是 dense 名次）：
  - 新題組：`id-iiot-bloomberg`（8）、`id-osha-847-phone`（10）、`id-iiot-an128`（12）
  - 舊題：`eml-subject`（7）、`poster-machine-measures`（10）、`poster-machine-causes`（17）

  rerank 收 dense 前 20 名，就能把這六題都排進前 6。
- 實測的 `dense-rerank` 用 20：這是命中數追上現行的最小候選數。

實測（`run.sh`）：依序跑 `dense-6`、`dense-8`（上限 6 的那一輪，即重跑）、`dense-rerank-20`、現行，最後用改過上限的執行檔再跑一次 `dense-8`。每組的步驟：

1. 卸載 Gemma。
2. 跑 `bench/concurrency.ts --levels 1,2,3,4`（同 §8.12 的 20 題）。
3. 一次一題跑三份題組。

全程每 3 秒記一次 swap、記憶體壓力與推論服務 footprint（`top` 的 MEM）。兩組只用 dense 的設定在開跑前，先等推論服務釋放 rerank（`/health` 的 `loaded` 只剩 `embed`），所以它們的 footprint 不含 rerank。

| | dense-6 | dense-8 | dense-rerank-20 | 現行 |
| --- | --- | --- | --- | --- |
| 答對：examples／realistic／新題組 | 29／33／15 | 30／32／16 | 29／33／18 | 28／32／18 |
| 答對合計（91 題） | 77 | 78 | 80 | 78 |
| 取到標準來源（可回答的 79 題） | 72 | 74 | 78 | 78 |
| 首字中位數：examples／realistic／新題組（一次一題） | 6.2／6.3／7.5s | 8.1／7.4／9.9s | 9.4／8.7／9.7s | 13.0／12.7／13.6s |
| 每分鐘答完（1／2／3／4 人） | 12.4／7.9／6.7／7.9 | 10.0／6.4／5.3／5.9 | 8.3／6.4／5.1／5.6 | 5.1／4.4／3.9／4.3 |
| 4 人同時：平均／最慢首字 | 16.8／29.9s | 22.8／42.4s | 26.4／44.5s | 38.5／59.5s |
| 同時處理那 20 題答對 | 17/20 | 18/20 | 20/20 | 19/20 |
| rerank 平均：同時處理（含排隊）／一次一題 | — | — | 5.3／1.8s | 14.8／5.1s |
| 每題送給 Gemma 的圖（平均） | 1.16 | 1.40 | 1.85 | 2.94 |
| 推論服務 footprint：最高／中位數 | 3.96／3.89 GiB | 4.49／3.75 GiB | 8.03／7.32 GiB | 8.37／7.60 GiB |
| 記憶體壓力 warn 的取樣點 | 0 | 0 | 13 | 2 |

- 實測取到來源的題數，和重播的命中數逐組相同。
- 全部四組都沒有錯誤，也沒有到 critical。
- swap 不比：各組開跑時就有 0.5–2.1 GiB 是前面留下來的。
- BM25 這一段的延遲中位數是 11 ms，dense 是 55 ms，所以拿掉 BM25 省不到時間。時間差主要來自 rerank 本身與排隊，以及送給 Gemma 的圖變少：現行的 rerank 偏好送圖的頁（#59）。
- 同一設定兩次執行的差距：`dense-6` 的重跑答對 29／32／15，每分鐘答完 12.1／7.3／6.6／7.7，同時處理 18/20。和第一次差在 `xlsx-peak-month` 與同時處理的一題，兩邊取到的來源都一樣，只差在 Gemma 的回答。所以每份題組差 1 題，是同一設定重跑就會出現的差距。

和現行結果不同的題目：

| 題目 | dense-6 | dense-8 | dense-rerank-20 | 現行 | 差在 |
| --- | --- | --- | --- | --- | --- |
| `eml-subject` | 沒取到 | 對 | 對 | 對 | 來源（dense 第 7） |
| `pdf-p14-reference` | 對 | 對 | 錯：沒引用 | 錯：沒引用 | 答案（來源都取到） |
| `xlsx-vendor` | 對 | 對 | 對 | 錯 | 答案 |
| `jpg-task` | 對 | 對 | 錯 | 對 | 答案（同一張圖） |
| `png-arm-color` | 錯：沒引用 | 錯：沒引用 | 對 | 錯 | 答案 |
| `xlsx-peak-month` | 對 | 錯 | 錯 | 錯 | 答案 |
| `xlsx-construction` | 對 | 對 | 錯 | 錯 | 答案 |
| `pptx-guard-types` | 錯 | 錯 | 對 | 對 | 來源（只用 dense 時沒取到列出四種護罩的第 23 張投影片） |
| `poster-machine-measures` | 沒取到 | 沒取到 | 對 | 錯：缺「防護設施」 | 來源（dense 第 10）／讀圖 |
| `id-osha-847-phone` | 沒取到 | 沒取到 | 對 | 對 | 來源（dense 第 10） |
| `id-iiot-an128` | 沒取到 | 沒取到 | 對 | 對 | 來源（dense 第 12） |
| `id-iiot-bloomberg` | 沒取到 | 對 | 對 | 對 | 來源（dense 第 8） |

四組都錯的題目不在表內，都是已知的既有問題：

- `pptx-content`：投影片只有 Lorem ipsum（#41）。
- `jpg-brand`、`png-logo`、`poster-machine-causes`、`poster-hoisting-hazards`：Gemma 讀圖讀錯。`poster-machine-causes` 在 `dense-6`、`dense-8` 是根本沒取到來源（dense 第 17 名）。
- `xlsx-deaths-112`、`xlsx-manufacturing`：XLSX 轉檔把表格切頁。

結論：

- 拿掉 BM25 與 RRF、保留 rerank、候選從 50 降到 20：
  - 答對：比現行多 2 題（80 對 78），取到的來源相同（78/79）。新題組也是 18/18，所以這份語料裡，精確字詞查詢不需要 BM25，只要 rerank 看得到 dense 前 20 名。
  - 速度：一次一題的首字快 3.6–4.0s，1–4 人同時提問每分鐘多答 30–62%，rerank 時間約剩三分之一。
  - 記憶體：rerank 模型仍然常駐，推論服務 footprint 只少 0.3 GiB。
- 完全不 rerank（`dense-6`）：
  - 速度：每分鐘答完是現行的 1.7–2.4 倍。
  - 記憶體：推論服務 footprint 少 4.4 GiB（8.37→3.96 GiB）。
  - 代價：取不到 6 題的來源，其中 3 題是精確字詞題。
  - `dense-8` 救回其中 2 題（`eml-subject`、`id-iiot-bloomberg`），但多送的兩個來源讓它比 `dense-6` 慢，footprint 最高 4.49 GiB。
- 語料只有 17 份文件、397 個 unit，dense 前 20 名占了 5%。文件變多時，前 20 名占的比例會下降，dense 前段的名次會更關鍵。所以如果改用 dense，候選數要在更大的語料上再確認。

工具在 `deploy/mac-mini/bench/dense-only/`：

- `replay.ts`：重播。
- `compose.yml` 與 `Dockerfile`：隔離實例。
- `run.sh`：依序實測各組；可另外指定只跑其中幾組。
- `compare.ts`：彙總。
- `rescore.ts`：關鍵字修改後重新評分。

新題組在 `deploy/mac-mini/bench/identifiers.json`。結果在 `knowva-mac:~/knowva-inference/bench-results/dense-60-20261008/`：

- `replay/`：重播結果。
- `e2e/`：各組的問答、同時處理、記憶體、app／Ollama log，以及 `compare.md`。重跑的結果檔名是 `dense-6b`；`orig-keys/` 是放寬關鍵字前的評分。
- 實例設定、正式設定前後的 sha256、有線網路的切換紀錄。

knowva-mac 的有線網路很慢（約 0.85 MB/s，Wi-Fi 約 40 MB/s），所以實測期間暫時關閉、只用 Wi-Fi。結束後已恢復，服務順序、IP 與預設路由都和實測前相同。

採用（2026-10-08）：預設改成 `dense-rerank-20`。

- BM25 與 RRF 的程式拿掉，migration `0009_dense_retrieval.sql` 移除 `0004` 加的 token 欄位、GIN index、語料統計表與 trigger。`RERANK_CANDIDATE_TOP_K` 的預設從 50 改成 20。
- `RETRIEVAL_MODE` 開關也一起拿掉，因為只剩 dense 一路。`RERANK_ENABLED=false` 保留，也就是 `dense-6`／`dense-8` 那種不 rerank 的設定，給記憶體優先的機器用。
- 上面的工具和 §8.15 的 `eval.ts` 會直接跑 BM25 的 SQL，`run.sh` 也會設 `RETRIEVAL_MODE`，所以只能在 `0008` 之前的資料庫和映像上重跑：bench-results 裡的 dump，以及隔離實例的映像 `knowva:dense60-ctx8`。
