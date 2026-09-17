import { useState, type ReactNode } from "react";
import { AssistantRuntimeProvider, useLocalRuntime } from "@assistant-ui/react";
import { BackendAdapter } from "./BackendAdapter";
import { Thread } from "./Thread";
import { Documents } from "./Documents";
import { BrandMark, ChatIcon, DocsIcon } from "./Icons";
import "./App.css";

type View = "chat" | "documents";

type NavItem = { id: View; label: string; icon: ReactNode };

const NAV: NavItem[] = [
  { id: "chat", label: "問答", icon: <ChatIcon /> },
  { id: "documents", label: "文件", icon: <DocsIcon /> },
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
};

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
        </aside>

        <div className="content">
          <header className="content-header">
            <h1 className="content-title">{meta.title}</h1>
            <p className="content-desc">{meta.description}</p>
          </header>
          <main className="content-body">
            {view === "chat" ? <Thread /> : <Documents />}
          </main>
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
}

export default App;
