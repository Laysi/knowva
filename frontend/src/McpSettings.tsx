// Mock view. Nothing here talks to a backend: the MVP has no MCP wiring, this
// is the board panel used to show what the settings screen will look like.
// Replace MOCK_SERVERS with a fetch when the real endpoint exists.
import { PlugIcon } from "./Icons";

type MockServer = {
  id: string;
  name: string;
  purpose: string;
  transport: string;
  endpoint: string;
  status: "ready" | "queued" | "processing" | "failed";
  statusLabel: string;
  tools: number;
  tags: readonly string[];
  note?: string;
};

const MOCK_SERVERS: readonly MockServer[] = [
  {
    id: "knowva-docs",
    name: "knowva-docs",
    purpose: "把文件庫的檢索與摘要開放給其他代理程式呼叫。",
    transport: "HTTP",
    endpoint: "http://knowva-server:8080/mcp",
    status: "ready",
    statusLabel: "已連線",
    tools: 4,
    tags: ["文件檢索", "內部"],
  },
  {
    id: "laysi-one",
    name: "laysi-one",
    purpose: "查詢財務報表、請假與薪資資料，供問答引用。",
    transport: "HTTP",
    endpoint: "https://one.laysi.internal/mcp",
    status: "ready",
    statusLabel: "已連線",
    tools: 12,
    tags: ["財務", "人資"],
  },
  {
    id: "github",
    name: "github",
    purpose: "讀取議題與程式碼，讓工程問題可以連回原始碼。",
    transport: "STDIO",
    endpoint: "npx -y @modelcontextprotocol/server-github",
    status: "queued",
    statusLabel: "待設定",
    tools: 8,
    tags: ["研發"],
    note: "尚未填入存取權杖，啟用後才會列出工具。",
  },
  {
    id: "plm-bridge",
    name: "plm-bridge",
    purpose: "從 PLM 取出料件與版次資訊。",
    transport: "HTTP",
    endpoint: "http://plm-bridge.internal:9100/mcp",
    status: "failed",
    statusLabel: "連線失敗",
    tools: 0,
    tags: ["製造"],
    note: "最後一次交握逾時（30 秒）。請確認服務是否在線。",
  },
];

const readyCount = MOCK_SERVERS.filter(
  (server) => server.status === "ready",
).length;
const toolCount = MOCK_SERVERS.reduce((sum, server) => sum + server.tools, 0);

export const McpSettings = () => (
  <div className="documents">
    <div className="documents-inner">
      <div className="mcp-banner" role="note">
        <span className="mcp-banner-icon">
          <PlugIcon />
        </span>
        <div className="dropzone-text">
          <strong>這是示意畫面</strong>
          <span>
            下列伺服器與狀態為假資料，尚未接上實際的 MCP
            連線；按鈕只是版面示意，不會動作。
          </span>
        </div>
      </div>

      <div className="library-head">
        <h2>MCP 伺服器</h2>
        <span className="library-count">
          {MOCK_SERVERS.length} 個 · {readyCount} 個已連線 · 共 {toolCount} 項工具
        </span>
      </div>

      <div className="doc-list">
        {MOCK_SERVERS.map((server) => (
          <div
            key={server.id}
            className={
              "doc-row" + (server.status === "failed" ? " is-failed" : "")
            }
          >
            <span className="doc-glyph">{server.transport}</span>
            <div className="doc-main">
              <div className="doc-line1">
                <span className="doc-name">{server.name}</span>
                <span className={`status status-${server.status}`}>
                  <span className="status-dot" />
                  {server.statusLabel}
                </span>
              </div>
              <p className="doc-summary">{server.purpose}</p>
              <p className="mcp-endpoint" title={server.endpoint}>
                {server.endpoint}
              </p>
              <div className="doc-meta">
                <span className="doc-tags">
                  {server.tags.map((tag) => (
                    <span key={tag} className="chip">
                      {tag}
                    </span>
                  ))}
                </span>
                <span className="doc-metrics">
                  <span className="doc-metric">{server.transport}</span>
                  <span aria-hidden="true"> · </span>
                  <span className="doc-metric">{server.tools} 項工具</span>
                </span>
              </div>
              {server.note && (
                <p className={server.status === "failed" ? "doc-error" : "doc-note"}>
                  {server.note}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="mcp-actions">
        <button type="button" disabled title="示意畫面，尚未實作">
          新增伺服器
        </button>
        <button type="button" disabled title="示意畫面，尚未實作">
          重新檢查連線
        </button>
      </div>
    </div>
  </div>
);
