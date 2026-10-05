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
- `qat-k6` 比 `qat-base` 多錯 14 題（examples 5、realistic 9），其中 11 題是正確來源沒進前 6 名（`not-retrieved`）：圖片與 EML 題、OSHA 單頁題、PPTX 統計與機器人事故題、海報題。它多答對三題 XLSX：`xlsx-deaths-112` 的 6 個來源與順序和 `qat-base` 完全一樣，差在回答本身；`xlsx-peak-month`、`xlsx-construction` 各有一到兩頁來源不同，`xlsx-construction` 的答案引自 `qat-base` 沒選進的第 7 頁。
- `qat-k6` 一次一題的首字比 `qat-base` 快約 8s：rerank 平均 1.0s（`qat-base` 7.4s），每題 context 的圖片平均 1.7 張（`qat-base` 3.0 張），prefill 也比較短。
- 一次一題時 rerank 平均 7.4s（`qat-base`，50 筆候選），和 §8.12 一格的 7.3s 相近；多人同時時平均多出約 14s，包含 embed／rerank 共用一把鎖的排隊與同時跑 Gemma 的 GPU 競爭，這次沒有拆開量。
- 上傳壓力（2026-10-06，`gemma4:12b-mlx`、頁圖 1280）：先跑一次 realistic（32/39）把 prefix cache 填滿，再用 `concurrency.ts --levels 4 --rounds 3` 連續三輪 4 人同時提問，同時依序上傳 Machine Guarding（75 頁）與 OSHA 3080（32 頁）的複本，ready 後就刪掉。每輪全部答完 69.0–81.3s（平均 77.0s），平均／最慢首字 59.0／79.8s（沒有上傳的 `mlx-1280` 是 70.5s、53.4／69.0s），12 題答對 11 題（`jpg-brand` 又讀成別的廠牌）。兩份上傳分別 64.4s、66.4s ready（§8.11 單獨上傳 Machine Guarding 是 28.1s）。swap 最高 11.8 GiB；系統記憶體壓力有 4 個取樣點（約 8 秒）到 critical，都在 Gemma 寫上傳摘要或 4 題同時 prefill 時，wired 從約 9.6 GiB 升到 15.4 GiB。Ollama、推論服務、app 的 log 都沒有錯誤；系統 log 只有 macOS 把閒置的系統 daemon 以 idle-exit 結束，沒有砍到這三個服務。腳本與結果在 `knowva-mac:~/knowva-inference/bench-results/stress-20261005-235428/`。
