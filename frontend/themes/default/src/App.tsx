import { useEffect, useState, type ReactNode } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import {
  CONVERSATION_NOT_SAVED_MESSAGE,
  useKnowvaRuntime,
  useProfile,
} from "@knowva/core";
import type { ProfileSuggestion } from "@knowva/core";
import { Thread } from "./Thread";
import { Documents } from "./Documents";
import { Conversations } from "./Conversations";
import { BrandMark, ChatIcon, DocsIcon } from "./Icons";
import "./App.css";

type View = "chat" | "documents";

type NavItem = { id: View; label: string; icon: ReactNode };

const NAV: NavItem[] = [
  { id: "chat", label: "問答", icon: <ChatIcon /> },
  { id: "documents", label: "文件", icon: <DocsIcon /> },
];

/** Keeps a line's height while the profile loads, so nothing shifts when it lands. */
const PENDING = " ";
const NO_SUGGESTIONS: readonly ProfileSuggestion[] = [];

function App() {
  const { runtime, history } = useKnowvaRuntime();
  const [view, setView] = useState<View>("chat");
  // Brand, page copy, suggestions and the composer placeholder all come from
  // the deployed profile. Chat and documents work whether or not it loads.
  const profileState = useProfile();
  const profile = profileState.status === "ready" ? profileState.profile : null;
  const page = profile?.pages[view];

  useEffect(() => {
    if (profile) {
      document.title = `${profile.brand.name} — ${profile.brand.tagline}`;
    }
  }, [profile]);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="app">
        <aside className="sidebar">
          <div className="brand">
            <span className="brand-mark">
              <BrandMark />
            </span>
            <span className="brand-text">
              <span className="brand-name">{profile?.brand.name ?? PENDING}</span>
              <span className="brand-sub">{profile?.brand.tagline ?? PENDING}</span>
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
            <h1 className="content-title">{page?.title ?? PENDING}</h1>
            {profileState.status === "error" ? (
              <div className="profile-error" role="alert">
                <span>{profileState.message}</span>
                <button type="button" onClick={profileState.reload}>
                  重新載入
                </button>
              </div>
            ) : (
              <p className="content-desc">{page?.description ?? PENDING}</p>
            )}
          </header>
          <main className="content-body">
            {view === "chat" ? (
              <div className="chat">
                <Conversations history={history} />
                <div className="chat-main">
                  {!history.saved && (
                    <p className="history-notice" role="status">
                      {CONVERSATION_NOT_SAVED_MESSAGE}
                    </p>
                  )}
                  <Thread
                    suggestions={profile?.suggestions ?? NO_SUGGESTIONS}
                    composerPlaceholder={profile?.composerPlaceholder ?? ""}
                  />
                </div>
              </div>
            ) : (
              <Documents />
            )}
          </main>
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
}

export default App;
