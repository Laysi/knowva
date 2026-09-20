import { useState, type ReactNode } from "react";
import { AssistantRuntimeProvider, useLocalRuntime } from "@assistant-ui/react";
import { BackendAdapter } from "./BackendAdapter";
import { Thread } from "./Thread";
import { Documents } from "./Documents";
import { McpSettings } from "./McpSettings";
import { BrandMark, ChatIcon, DocsIcon, PlugIcon } from "./Icons";
import "./App.css";

type View = "chat" | "documents" | "mcp";

type NavItem = { id: View; label: string; icon: ReactNode };

const NAV: NavItem[] = [
  { id: "chat", label: "問答", icon: <ChatIcon /> },
  { id: "documents", label: "文件", icon: <DocsIcon /> },
  { id: "mcp", label: "MCP 設定", icon: <PlugIcon /> },
];

const VIEW_META: Record<View, { title: string; description: string }> = {
  chat: {
    title: "知識問答",
    description: "查詢已建立索引的公司文件。",
  },
  documents: {
    title: "文件管理",
    description: "上傳後，知星會讀完文件、整理摘要，並建立索引供查詢。",
  },
  mcp: {
    title: "MCP 設定",
    description: "接上外部工具伺服器，讓知星能查詢與操作其他系統。",
  },
};

// Mock. Chat history lives in frontend state only and is not persisted, so
// there is nothing to list yet; these entries exist to show the rail's shape.
// Replace with the real conversation list when history gets stored.
const MOCK_HISTORY: readonly { id: string; title: string; when: string }[] = [
  { id: "h1", title: "出貨檢驗規範怎麼判定不合格", when: "剛剛" },
  { id: "h2", title: "ISO 9001 改版後要補哪些文件", when: "2 小時前" },
  { id: "h3", title: "新人前三天該看哪幾份手冊", when: "昨天" },
  { id: "h4", title: "客戶退貨流程誰負責簽核", when: "昨天" },
  { id: "h5", title: "Q3 設備保養排程有沒有衝突", when: "3 天前" },
  { id: "h6", title: "倉儲安全規則的重點摘要", when: "上週" },
];

function App() {
  const runtime = useLocalRuntime(BackendAdapter);
  const [view, setView] = useState<View>("chat");
  const meta = VIEW_META[view];

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="app">
        <aside className="sidebar">
          <div className="brand">
            <span className="brand-mark">
              <BrandMark />
            </span>
            <span className="brand-text">
              <span className="brand-name">Knowva 知星</span>
              <span className="brand-sub">點亮你的知識星系</span>
            </span>
          </div>

          <nav className="nav">
            {NAV.map((item) => (
              <button
                key={item.id}
                type="button"
                className={"nav-item" + (view === item.id ? " active" : "")}
                aria-current={view === item.id ? "page" : undefined}
                onClick={() => setView(item.id)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
              </button>
            ))}
          </nav>

          {/* Mock list: display only, clicking loads nothing. */}
          <section className="history" aria-label="近期對話（示意）">
            <h2 className="history-head">
              近期對話
              <span className="history-mock">示意</span>
            </h2>
            <div className="history-list">
              {MOCK_HISTORY.map((item, index) => (
                <div
                  key={item.id}
                  className={"history-item" + (index === 0 ? " current" : "")}
                >
                  <span className="history-title">{item.title}</span>
                  <span className="history-when">{item.when}</span>
                </div>
              ))}
            </div>
          </section>
        </aside>

        <div className="content">
          <header className="content-header">
            <h1 className="content-title">{meta.title}</h1>
            <p className="content-desc">{meta.description}</p>
          </header>
          <main className="content-body">
            {view === "chat" ? (
              <Thread />
            ) : view === "documents" ? (
              <Documents />
            ) : (
              <McpSettings />
            )}
          </main>
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
}

export default App;
