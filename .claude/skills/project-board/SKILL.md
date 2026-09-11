---
name: project-board
description: Laysi/outsourcing-ai-km 的 GitHub issue 專案政策 overlay。凡是「建 issue、拆工作、寫驗收條件、連結 PR/commit、標記 blocker、寫 implementation note、close issue、更新 tracker #1」都依本 skill 操作。執行機制＝github-loop-engineering-skill（GitHub MCP 優先）；本 skill 提供本 repo 的座標、語意約定、內文格式、close ritual 與 gh CLI fallback。
---

# outsourcing-ai-km issue 操作手冊

issue 同步是完成定義的一部分，政策規則在 `CLAUDE.md`「Project management (GitHub issues)」章節；本檔是操作細節。

## 執行機制：github-loop-engineering-skill

執行引擎是 vendored skill `github-loop-engineering-skill`（Discover → Frame → Act → Verify → Persist → Continue/Stop，寫後必讀回驗證）。做任何 issue 操作前先載入該 skill；本 skill 只補本 repo 的政策，兩者衝突時**以本 skill 為準**。已知覆寫：

- **沒有 Projects 看板**：本 repo 目前沒有連結任何 Projects V2 看板（2026-09-11 查過）。loop skill 裡的 project、Status、Priority 欄位步驟全部跳過；**不要自行建立或掛到任何看板**（`Laysi` org 底下的其他看板屬於別的專案）。使用者之後若要看板，先問清楚是哪一個，再照 loop skill 現查。
- **狀態只有 open／closed**，進度寫在 comment；總覽靠 tracker issue #1 的 checklist。
- **內文格式照本 repo 既有 issue**（見下方「issue 內文格式」），覆寫 loop skill 的 `references/work-item-format.md`。
- **不新增 label**。
- **MCP 不在時允許 gh fallback**：先回報再 fallback，不是靜默繞過（見文末）。

### 工具選擇

- GitHub MCP（`.mcp.json` 的 `github` entry，需要環境變數 `GITHUB_PERSONAL_ACCESS_TOKEN`）：`issue_write`／`issue_read`／`add_issue_comment`／`sub_issue_write`／`list_label`／`search_issues`。payload 範式見 loop skill 的 `references/github-mcp-tools.md`。
- MCP 沒連上 → 文末 gh fallback。

## 固定座標

- Repo：`Laysi/outsourcing-ai-km`（private），預設分支 `main`。
- **規格優先序：當前 issue > tracker #1 > `docs/mvp-now.md`**（同 CLAUDE.md）。
- Tracker：#1「[MVP] 雲端視覺 RAG：客戶文件可問答」，用內文的「工作項目與順序」checklist 追蹤子工作，**不是** GitHub sub-issue。

## 語意約定

- **agent 自己發現的待辦是提案**：列給使用者看，使用者同意才建 issue。使用者明示要建的才直接建。
- **標題**：屬於 MVP 範圍、會掛進 #1 checklist 的工作用 `[MVP-NN] 標題`，NN 接續現有最大號（先 `search_issues` 查 `[MVP-` 取最大值，不要憑記憶）。MVP 外的 bug 或單點問題用一般標題（如 #15）。
- **Labels**：只用現有的——`bug`／`enhancement`／`documentation`／`ui`／`ci/cd`（另有 GitHub 預設的 `duplicate`／`wontfix`／`invalid`／`question` 等）。新能力配 `enhancement`、行為錯誤配 `bug`，畫面改動加 `ui`，CI/workflow 改動加 `ci/cd`；一張可以多個。新分類先問使用者，不要 `create_label`。MCP `issue_write` 的 `labels` 是**整組替換**——先讀全集再送完整清單。
- **Issue type**：repo 有 Feature／Bug／Task 可用，但既有 issue 都沒設；新 issue 可以設（與 label 對齊），不回補舊的。
- **Assignee**：既有 MVP issue 指派給協作者，不是 agent 的帳號。**不要自己指派任何人**，除非使用者指定。
- **語言**：issue 標題與內文 zh-TW（技術名詞、識別字、指令保留英文）；commit message 英文 conventional commits（`fix: …`、`chore: …`）。

## issue 內文格式

現成範例：#8（完整 MVP 格式）、#15（小型 bug／功能）。空段落整段省略。

**MVP 工作項**（掛 #1 的那種）開頭固定兩行引言，讓沒讀過 #1 的人也不會越界：

```markdown
> 固定限制：模型推論僅使用雲端託管 API；回答／多模態閱讀固定 Gemma 4 家族，embedding 固定 llama-nemotron-embed-vl-1b-v2，rerank 固定 llama-nemotron-rerank-vl-1b-v2。不得新增本地模型、vLLM、本地 NIM、量化或替代模型家族議題。

> 規格優先序：本 issue > #1 > docs/mvp-now.md。

## 背景

現在的行為與為什麼要改。**指到檔案與行號**（`backend/src/search.rs:120-140`），冷啟動接得上。

## 目標

做完之後看得到什麼。

## （契約段，視需要）

API／SSE／資料模型等必須遵守的介面，例如「API/SSE 契約」「Context 與 prompt 契約」。

## 交付內容

- 具體改動

## 驗收條件

- 可觀察的行為或產物，每條獨立可驗證
- 對應的測試或指令

## 非目標

明確不做的事。

## 依賴與交付

依賴 #N；PR 需附的東西。
```

**小型 issue**（#15 型）：`背景`／`問題`／`目標`／`驗收條件` 四段即可，引言可省。

驗收條件不寫「正常運作」「測試通過」這種沒指名預期行為的句子。本 repo 的驗證 gate（與 CLAUDE.md 同步）：

- 後端（在 `backend/`）：`proto exec rust -- cargo fmt --check`、`proto exec rust -- cargo clippy --all-targets -- -D warnings`、`proto exec rust -- cargo test`
- 前端（在 `frontend/`）：`proto run npm -- ci`、`proto run npm -- run lint`、`proto run npm -- run build`
- 評測（CI `evaluation.yml`）：`proto exec rust -- cargo test evaluation::`、`proto exec rust -- cargo run -- eval --dry-run`
- 原生轉檔：`docker build --target native-test -t ai-km-demo:native-test .`

### blocker 段

卡住時加這一段，不改寫其他內文：

```markdown
## 卡在

- 阻礙：#123，或精確的外部原因（例如 AI Gateway BYOK 尚未設定）
- 起始：YYYY-MM-DD
- 解除條件：具體的事件或證據
```

解除時刪掉已解決的項目並在 comment 記錄解法；一個 blocker 都不剩才整段刪。

## 建立 issue

1. **先查重**（`search_issues`）。
2. `issue_write` create：zh-TW 標題、上述格式、label（type 選填），不帶 assignee。
3. 若是 MVP 工作項：更新 #1 內文的「工作項目與順序」checklist 加上 `- [ ] #N`（送**完整**內文，只加那一行）。
4. **讀回驗證** 新 issue 與 #1。

## 拆工作

可獨立驗收的部分拆成獨立 issue，彼此用內文的「依賴與交付」互相引用；要掛 #1 的一律進 #1 checklist。只有使用者要求時才用 `sub_issue_write`（`sub_issue_id` 是子項的**數字 database ID**，不是 issue number）。半天內的細項用驗收條件 checklist 即可。

## implementation note

放在 comment，不改內文：

```markdown
**Implementation note (YYYY-MM-DD)：** 決定、發現或進度。證據：PR／commit／路徑／測試。剩下：具體的下一步。
```

值得寫的是耐久的決定、不明顯的陷阱、交接到一半的狀態、驗證結果、PR 或 commit 連結。

## close ritual

1. 驗收條件每一條都有跑過的證據（MVP issue 的驗收條件是 bullet 不是 checkbox，證據寫進完成 comment）。
2. 補完成 comment：

```markdown
**完成 (YYYY-MM-DD)**

- 成果：改了什麼
- 變更：PR、commit 或產物連結
- 驗證：指令與觀察到的結果
- 範圍變動：無，或明確的縮減決定
- 後續：issue 連結、提案，或無
```

3. close issue（`state_reason: completed`）。PR 用 `Closes #N` 時 merge 會自動關，merge 後讀回確認。
4. 若在 #1 checklist 裡，把該行勾成 `- [x] #N` 並讀回。
5. 掃描下一步：有新待辦或技術債就**提案給使用者**，不自行建。

## gh CLI fallback

MCP 沒連上時走這條，**先回報再 fallback**。`gh` 已登入（帳號 `jerry830401`），issue 與 label 操作直接可用：

```bash
gh issue list --repo Laysi/outsourcing-ai-km --state all --search "[MVP-"
gh issue view <N> --repo Laysi/outsourcing-ai-km --json title,body,labels,state
gh issue create --repo Laysi/outsourcing-ai-km --title "…" --label enhancement --body-file <file>
gh issue edit <N> --repo Laysi/outsourcing-ai-km --body-file <file>
gh issue comment <N> --repo Laysi/outsourcing-ai-km --body-file <file>
gh issue close <N> --repo Laysi/outsourcing-ai-km --reason completed
```

內文一律用 `--body-file`（放 scratchpad），不要把多行 zh-TW 塞進 `--body` 引號裡。
