# Knowva on a Mac mini（M6, 24 GB）

一台 Mac mini 跑完整的 Knowva：應用程式與資料庫在 Colima 的 Linux VM 裡，三個模型在 macOS 原生跑、直接用 GPU。這是單機 demo 落點；產品線的標準部署仍是 [docs/deployment.md](../../docs/deployment.md) 的 NVIDIA + Linux。

| 元件 | 跑在哪 | 位址 | 開機啟動 |
| --- | --- | --- | --- |
| app（前後端）+ PostgreSQL | Colima VM（vz，4 CPU / 3 GiB） | `127.0.0.1:8080` | `io.laysi.knowva.colima` |
| 推論服務：Embed VL、Rerank VL（MPS float16），chat 轉給 Ollama | macOS | `127.0.0.1:9500` | `io.laysi.knowva.inference` |
| Gemma 4 12B（`gemma4:12b-it-qat`） | macOS，Ollama + Metal | `127.0.0.1:11434` | `io.laysi.knowva.ollama` |

容器經 `host.docker.internal`（192.168.5.2，lima 轉到主機 loopback）連推論服務，所以三個服務都只綁 loopback，區網看不到。三個都是 LaunchDaemon（`UserName=knowva`）：這台開機不登入，LaunchAgent 不會啟動。

## 為什麼是這些設定

整體結果：14 頁 PDF 上傳到可問答 51s；提問到第一個字 30–39s，之後約 12 tok/s。數字都是在這台機器上用 `bench/` 量的，題目與頁面見 `bench/chat.ts`、`bench/retrieval.ts`。量測紀錄在 [docs/measurements/README.md](../../docs/measurements/README.md) §8。

- **chat 模型 12B QAT**：26B-A4B 單獨就佔 20 GB，超過 GPU 可用的 17.8 GiB；和檢索模型同時存在時，編碼頁圖會 Metal OOM，回答為空。E2B 首字較快，但瓶頸是看圖，12B 只慢幾秒、品質高一級。QAT 比 q4_K_M 解碼慢（17 對 24 tok/s），但它是 Google 針對 int4 訓練的版本，品質較接近原始權重，檔案也較小。
- **頁圖縮到長邊 1280 再給 Gemma**（`KM_CHAT_IMAGE_MAX_SIDE`）：Gemma 4 的圖片 token 跟像素成正比，後端送的 1600px 一頁約 860 tokens，佔首字延遲三分之二。1280 時一頁約 570 tokens，6 頁的首字延遲從約 22s 降到約 19s；768 已經會把「IL050」讀成「11050」。縮圖是決定性的，相同的 prompt 仍命中 prompt cache。
- **Rerank 用圖 + 文、1 tile**（`KM_RERANK_TILES=1`）：50 筆候選從 57s（6 tiles）降到 12.6s，最相關的頁仍排第一（只量一題）。只用文字更快（4.9s），但掃描頁與圖表頁會失去影像訊號。後端從 #38 起逐頁決定送不送圖（`needs_vision`），文字頁本來就只送文字，所以 `KM_RERANK_INPUT` 維持 `both`；設成 `text` 會連需要看圖的頁也拿掉圖。
- **Embed / Rerank 閒置 10 分鐘才卸載**（`KM_IDLE_SECS=600`）：每題都卸載實測是虧的。Gemma 常駐時權重會被擠出檔案快取，每次重新載入 embed 4–5s、rerank 約 5s，比省下的 Gemma prefill（4–5s）還多；對話中兩個模型保持載入，第 2、3 輪首字從 39–44s 降到 30s。沒人用 10 分鐘後歸還約 6 GB。
- **Ollama prompt cache 上限 2 GB**（`LLAMA_ARG_CACHE_RAM`）：預設 8 GB，llama-server 實體佔用 10.8 GB，是 swap 的主因；2 GB 時 3.3 GB，仍保留最近約兩份 prompt 給追問；交替 A/B 量不出速度差。Gemma 4 sliding-window 層的 context checkpoint 每個 320 MB、預設上限 32，同理設 `LLAMA_ARG_CTX_CHECKPOINTS=4`。
- **Ollama context 16384、flash attention 開、`keep_alive=-1`**：6 頁 prompt 有 7–9K tokens，預設 4096 會截斷；FA 讓 12B 解碼 15.5→17.2 tok/s；模型常駐省掉每題 3s 載入。

## 從零安裝

前提：macOS 已完成無頭設定（不睡眠、斷電後自動開機、SSH via Cloudflare Tunnel），帳號 `knowva`。整個 `deploy/mac-mini/` 同步到 `~/knowva-inference/app/`：

```sh
rsync -a --exclude .venv deploy/mac-mini/ knowva-mac:knowva-inference/app/
```

需要 sudo 的步驟都要在**自己的終端機**跑（`ssh -t knowva-mac 'sudo …'`），Claude Code 的 `!` 沒有 TTY。

1. Xcode Command Line Tools（proto 的 uv/bun 插件需要 git）：`sudo bash ~/knowva-inference/app/setup/install-clt.sh`
2. proto 與工具：照 [proto 安裝說明](https://moonrepo.dev/docs/proto/install) 放進 `~/.proto`，然後在 `~/knowva-inference/app` 執行 `proto install`（版本鎖在 `.prototools`）。
3. 容器工具（Colima、Lima、Docker CLI、Compose，全部驗 SHA-256）：`bash setup/install-containers.sh`
4. Ollama 與模型：`bash setup/install-ollama.sh`，再以任一 `ollama serve` 執行中時 `ollama pull gemma4:12b-it-qat`。
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
- 網頁：<https://knowva-mac.laysi.io>，**完全公開、沒有登入**，任何拿到網址的人都能上傳、刪除、提問（擁有者 2026-10-02 決定）。要收回時刪掉 `config.yml` 的那條 ingress 或 DNS 記錄即可。
- 不經 Cloudflare 時：`ssh -L 8080:127.0.0.1:8080 knowva-mac`，再開 <http://127.0.0.1:8080>。

## 已知問題

- **`audiomxd` / `configd` 常駐約一顆核心**：沒有使用者登入時 `audiomxd` 會空轉（[Apple Community](https://discussions.apple.com/thread/255974341)）。自動登入可解，但等於開機就有一個已登入的管理員桌面，所以沒做。
- **Ollama 在 Metal OOM 時仍回 HTTP 200、串流為空**：後端會因缺少 `[DONE]` 回報錯誤，但原因只在 `ollama.log`。
