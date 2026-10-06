# Knowva on a Mac mini（M6, 24 GB）

一台 Mac mini 跑完整的 Knowva：應用程式與資料庫在 Colima 的 Linux VM 裡，三個模型在 macOS 原生跑、直接用 GPU。這是單機 demo 落點；產品線的標準部署仍是 [docs/deployment.md](../../docs/deployment.md) 的 NVIDIA + Linux。

| 元件 | 跑在哪 | 位址 | 開機啟動 |
| --- | --- | --- | --- |
| app（前後端）+ PostgreSQL | Colima VM（vz，4 CPU / 3 GiB） | `127.0.0.1:8080` | `io.laysi.knowva.colima` |
| 推論服務：Embed VL、Rerank VL（MPS float16），chat 轉給 Ollama | macOS | `127.0.0.1:9500` | `io.laysi.knowva.inference` |
| Gemma 4 12B（`gemma4:12b-mlx`） | macOS，Ollama（自編，見下）的 MLX runner | `127.0.0.1:11434` | `io.laysi.knowva.ollama` |

容器經 `host.docker.internal`（192.168.5.2，lima 轉到主機 loopback）連推論服務，所以三個服務都只綁 loopback，區網看不到。三個都是 LaunchDaemon（`UserName=knowva`）：這台開機不登入，LaunchAgent 不會啟動。

## 為什麼是這些設定

整體結果（QAT 時期）：14 頁 PDF 上傳到可問答 51s；提問到第一個字 30–39s，之後約 12 tok/s。改用 MLX 後，1–4 人同時提問每分鐘答完 3.3–4.0 題（QAT 2.6–3.1）。數字都是在這台機器上用 `bench/` 量的，題目與頁面見 `bench/chat.ts`、`bench/retrieval.ts`。量測紀錄在 [docs/measurements/README.md](../../docs/measurements/README.md) §8。

- **chat 模型 12B，MLX 版**（`gemma4:12b-mlx`）：26B-A4B 單獨就佔 20 GB，超過 GPU 可用的 17.8 GiB；和檢索模型同時存在時，編碼頁圖會 Metal OOM，回答為空。E2B 首字較快，但瓶頸是看圖，12B 只慢幾秒、品質高一級。同一個 12B，Ollama 的 MLX runner 比 llama.cpp 跑的 QAT 版讀入快約 40%（約 550 對 390 tok/s）、生成快約 60%（約 28 對 18 tok/s），多人同時提問每分鐘多答約 30%，examples／realistic 答對 29/34、32/39（QAT 31/34、32/39）。QAT（`gemma4:12b-it-qat`）是退路：它是 Google 針對 int4 訓練的版本，比 q4_K_M 解碼慢（17 對 24 tok/s），但品質較接近原始權重。
- **頁圖縮到長邊 1280 再給 Gemma**（`KM_CHAT_IMAGE_MAX_SIDE`）：QAT 版的圖片 token 跟像素成正比，後端送的 1600px 一頁約 860 tokens，佔首字延遲三分之二；1280 時一頁約 570 tokens，6 頁的首字延遲從約 22s 降到約 19s；768 已經會把「IL050」讀成「11050」。MLX 版的圖片 token 分 280／560／1120 三檔，1280 落在 1120 檔（約 1104 tokens）；縮到 1152 會落進 560 檔，但首字與吞吐沒有變快、答對率少 1 題，所以維持 1280。縮圖是決定性的，相同的 prompt 仍命中 prompt cache。
- **Rerank 用圖 + 文、1 tile**（`KM_RERANK_TILES=1`）：50 筆候選從 57s（6 tiles）降到 12.6s，最相關的頁仍排第一（只量一題）。只用文字更快（4.9s），但掃描頁與圖表頁會失去影像訊號。後端從 #38 起逐頁決定送不送圖（`needs_vision`），文字頁本來就只送文字，所以 `KM_RERANK_INPUT` 維持 `both`；設成 `text` 會連需要看圖的頁也拿掉圖。
- **Embed / Rerank 閒置 10 分鐘才卸載**（`KM_IDLE_SECS=600`）：每題都卸載實測是虧的。Gemma 常駐時權重會被擠出檔案快取，每次重新載入 embed 4–5s、rerank 約 5s，比省下的 Gemma prefill（4–5s）還多；對話中兩個模型保持載入，第 2、3 輪首字從 39–44s 降到 30s。沒人用 10 分鐘後歸還約 6 GB。
- **自編 Ollama，MLX 的 prompt 快照上限 2 GiB**（`OLLAMA_MLX_MAX_PAGED_OUT_BYTES`）：官方 MLX runner 把每題的 KV 快照留著給之後開頭相同的 prompt，上限寫死 8 GiB；每題約 0.7 GiB，十來題就滿，Ollama 佔 16 GB、swap 到 12.5 GiB，上傳加 4 人同時提問時記憶體壓力會短暫到 critical。知識庫的 prompt 之間幾乎只共用系統提示（平均只重用 6.7% 的 token），上限 2 GiB 時速度、答對率、追問首字都沒變，Ollama 降到 9.8 GB、swap 5.2 GiB。上游不收這個設定（ollama/ollama#18556 被關，#18131 未解），所以 `setup/install-ollama.sh` 從 release 的同一個 commit 套 `ollama/prefix-cache-budget.patch` 重編 Go 執行檔（`v0.35.0-km1`），MLX 與 llama.cpp 的函式庫沿用 release 附的。上游內建上限後就改回官方版。
- **Ollama prompt cache 上限 2 GB**（`LLAMA_ARG_CACHE_RAM`，只作用於 QAT 等 llama.cpp 模型）：預設 8 GB，llama-server 實體佔用 10.8 GB，是 swap 的主因；2 GB 時 3.3 GB，仍保留最近約兩份 prompt 給追問；交替 A/B 量不出速度差。Gemma 4 sliding-window 層的 context checkpoint 每個 320 MB、預設上限 32，同理設 `LLAMA_ARG_CTX_CHECKPOINTS=4`。
- **Ollama context 16384、flash attention 開、`keep_alive=-1`**：6 頁 prompt 有 7–9K tokens，預設 4096 會截斷；FA 讓 12B 解碼 15.5→17.2 tok/s；模型常駐省掉每題 3s 載入。
- **Ollama 一次只答一題**（`OLLAMA_NUM_PARALLEL=1`）：每題的 rerank 與 prefill 已經用滿 GPU，1–4 人同時提問時一格每分鐘答完 2.6–3.1 題；2 格吞吐相同、swap 多 1.8 GiB，3 格 swap 到 7 GiB、吞吐少約 20%。多人同時用時，排隊最久的是推論服務裡一次只做一題的 rerank（4 人同時平均 29s）。

## 從零安裝

前提：macOS 已完成無頭設定（不睡眠、斷電後自動開機、SSH via Cloudflare Tunnel），帳號 `knowva`。整個 `deploy/mac-mini/` 同步到 `~/knowva-inference/app/`：

```sh
rsync -a --exclude .venv deploy/mac-mini/ knowva-mac:knowva-inference/app/
```

需要 sudo 的步驟都要在**自己的終端機**跑（`ssh -t knowva-mac 'sudo …'`），Claude Code 的 `!` 沒有 TTY。

1. Xcode Command Line Tools（proto 的 uv/bun 插件需要 git）：`sudo bash ~/knowva-inference/app/setup/install-clt.sh`
2. proto 與工具：照 [proto 安裝說明](https://moonrepo.dev/docs/proto/install) 放進 `~/.proto`，然後在 `~/knowva-inference/app` 執行 `proto install`（版本鎖在 `.prototools`）。
3. 容器工具（Colima、Lima、Docker CLI、Compose，全部驗 SHA-256）：`bash setup/install-containers.sh`
4. Ollama 與模型：`bash setup/install-ollama.sh`（下載官方 release，再從同一個 commit 套 patch 重編 Go 執行檔，需要步驟 1 的 clang 與步驟 2 的 Go），再以任一 `ollama serve` 執行中時 `ollama pull gemma4:12b-mlx`；要保留 QAT 退路就再 pull `gemma4:12b-it-qat`。
5. Nemotron VL（鎖 revision、逐檔驗證）：`bun setup/fetch-models.ts`
6. Python 環境：`cd inference && uv sync --frozen`
7. VM：`colima start --cpus 4 --memory 3 --disk 60 --vm-type vz --mount none --runtime docker`（設定存進 `~/.colima/default/colima.yaml`，之後由 daemon 沿用）。
8. 應用程式：image 是私有的，不在 Mac 上存 registry 憑證；在有權限的機器 `docker pull --platform linux/arm64 ghcr.io/laysi/knowva:<版本>`、`docker save`，複製過去後 `docker load`。然後：
   ```sh
   mkdir -p ~/knowva && cp ~/knowva-inference/app/docker-compose.yml ~/knowva/
   cp ~/knowva-inference/app/.env.example ~/knowva/.env   # 填 POSTGRES_PASSWORD
   cd ~/knowva && docker compose -p knowva up -d
   ```
9. 開機服務：`sudo bash ~/knowva-inference/app/setup/install-services.sh`（會先停掉手動啟動的同名服務與 VM）。
10. Tunnel 設定（SSH + 公開網頁、走 HTTP/2）：`sudo bash ~/knowva-inference/app/setup/install-tunnel-config.sh`。`cloudflared/config.yml` 是唯一來源；`~/cloudflared-install/config.yml` 要保持相同，否則重跑 `install-cloudflared.sh` 會蓋回舊版。DNS 另外在 Cloudflare 建 proxied CNAME `knowva-mac.laysi.io` → `92fba54d-96ba-4ef3-a392-f3897374008e.cfargotunnel.com`。

## 維運

- 日誌：`~/Library/Logs/knowva/{ollama,inference,colima}.log`；應用程式 `docker compose -p knowva logs app`。
- 狀態：`curl 127.0.0.1:9500/health`、`curl 127.0.0.1:11434/api/ps`、`curl 127.0.0.1:8080/api/health`。
- 改設定：改 `launchd/*.plist` 後同步過去，再跑一次 `install-services.sh`。
- 換回官方 Ollama：`ln -sfn v0.35.0 ~/knowva-inference/ollama/current`，再 `sudo launchctl kickstart -k system/io.laysi.knowva.ollama`；官方版不認得 `OLLAMA_MLX_MAX_PAGED_OUT_BYTES`，快照上限回到 8 GiB。
- 退回 QAT：`~/knowva/.env` 的 `GEMMA_4_MODEL_ID` 改成 `gemma4:12b-it-qat`，再 `cd ~/knowva && docker compose -p knowva up -d`。
- 升級 Ollama：改 `setup/install-ollama.sh` 的 `VERSION`、`SHA256`、`COMMIT`，確認 `ollama/prefix-cache-budget.patch` 還套得上，再重跑它並重啟 Ollama daemon。
- 網頁：<https://knowva-mac.laysi.io>，**完全公開、沒有登入**，任何拿到網址的人都能上傳、刪除、提問（擁有者 2026-10-02 決定）。要收回時刪掉 `config.yml` 的那條 ingress 或 DNS 記錄即可。
- 不經 Cloudflare 時：`ssh -L 8080:127.0.0.1:8080 knowva-mac`，再開 <http://127.0.0.1:8080>。
- 驗收：把 `example_file/` 複製到這台後跑 `bun bench/examples.ts --files <目錄> --replace --db-container knowva-postgres-1`。它會重新上傳全部範例，新的一份就緒後才刪掉同名的舊上傳（被拒或失敗時保留舊的），檢查頁數與哪些頁送圖，再問 `bench/examples.json` 的題目。每題要取回並引用正確的來源，答案也要有指定的關鍵字；任何一項檢查失敗就以非 0 結束。
- 真實文件驗收：`bun bench/examples.ts --dataset bench/realistic.json --fetch --files <目錄> --db-container knowva-postgres-1`。`bench/realistic.json` 列的是政府與開源專案公開的 PDF、掃描 PDF、DOCX、XLSX、PPTX、海報與 Markdown，repo 只存網址與 sha256；`--fetch` 會下載缺少的檔案；每個檔案上傳前都會驗 sha256，雜湊不符就停止，上傳的就是驗過的那份內容。

## 已知問題

- **`audiomxd` / `configd` 常駐約一顆核心**：沒有使用者登入時 `audiomxd` 會空轉（[Apple Community](https://discussions.apple.com/thread/255974341)）。自動登入可解，但等於開機就有一個已登入的管理員桌面，所以沒做。
- **Ollama 在 Metal OOM 時仍回 HTTP 200、串流為空**：後端會因缺少 `[DONE]` 回報錯誤，但原因只在 `ollama.log`。
