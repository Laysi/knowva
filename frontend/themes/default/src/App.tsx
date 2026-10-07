import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { AssistantRuntimeProvider, ThreadListPrimitive } from "@assistant-ui/react";
import { useKnowvaRuntime, useProfile } from "@knowva/core";
import type { ProfileSuggestion } from "@knowva/core";
import { Thread } from "./Thread";
import { Documents } from "./Documents";
import { Conversations } from "./Conversations";
import {
  BrandMark,
  ChatIcon,
  CloseIcon,
  DocsIcon,
  MenuIcon,
  PlusIcon,
} from "./Icons";
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
/** Where App.css turns the rail into a drawer; keep the two in step. */
const DRAWER_QUERY = "(max-width: 720px)";

function App() {
  const { runtime, history } = useKnowvaRuntime();
  const [view, setView] = useState<View>("chat");
  // Phones only: the rail slides over the page instead of standing beside it.
  const [menuOpen, setMenuOpen] = useState(false);
  const menuId = useId();
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  // Set when the person closes the drawer, so focus goes back to its button.
  const refocusMenuButton = useRef(false);
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

  useEffect(() => {
    if (!menuOpen) {
      // Only once the page behind is no longer inert can it take focus.
      if (refocusMenuButton.current) {
        refocusMenuButton.current = false;
        menuButtonRef.current?.focus();
      }
      return;
    }
    closeButtonRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      refocusMenuButton.current = true;
      setMenuOpen(false);
    };
    // Widening past the drawer breakpoint leaves no drawer to close.
    const narrow =
      typeof window.matchMedia === "function"
        ? window.matchMedia(DRAWER_QUERY)
        : null;
    const onResize = () => {
      if (!narrow?.matches) setMenuOpen(false);
    };
    document.addEventListener("keydown", onKeyDown);
    narrow?.addEventListener("change", onResize);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      narrow?.removeEventListener("change", onResize);
    };
  }, [menuOpen]);

  const closeMenu = () => {
    if (!menuOpen) return;
    refocusMenuButton.current = true;
    setMenuOpen(false);
  };
  const show = (next: View) => {
    setView(next);
    closeMenu();
  };

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className={"app" + (menuOpen ? " menu-open" : "")}>
        <aside id={menuId} className="sidebar" aria-label="主選單">
          <div className="brand">
            <span className="brand-mark">
              <BrandMark />
            </span>
            <span className="brand-text">
              <span className="brand-name">{profile?.brand.name ?? PENDING}</span>
              <span className="brand-sub">{profile?.brand.tagline ?? PENDING}</span>
            </span>
            <button
              ref={closeButtonRef}
              type="button"
              className="menu-close"
              aria-label="關閉選單"
              onClick={closeMenu}
            >
              <CloseIcon />
            </button>
          </div>

          <nav className="nav">
            {NAV.map((item) => (
              <button
                key={item.id}
                type="button"
                className={"nav-item" + (view === item.id ? " active" : "")}
                aria-current={view === item.id ? "page" : undefined}
                onClick={() => show(item.id)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
              </button>
            ))}
          </nav>

          <Conversations history={history} onOpen={() => show("chat")} />
        </aside>

        <div className="scrim" aria-hidden="true" onClick={closeMenu} />

        <div className="content" inert={menuOpen}>
          <header className="content-header">
            <button
              ref={menuButtonRef}
              type="button"
              className="menu-button"
              aria-label="開啟選單"
              aria-controls={menuId}
              aria-expanded={menuOpen}
              onClick={() => setMenuOpen(true)}
            >
              <MenuIcon />
            </button>
            <div className="content-heading">
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
            </div>
            {view === "chat" && (
              <ThreadListPrimitive.New
                className="topbar-new"
                aria-label="開始新對話"
                title="新對話"
              >
                <PlusIcon />
              </ThreadListPrimitive.New>
            )}
          </header>
          <main className="content-body">
            {view === "chat" ? (
              <div className="chat">
                {history.notice && (
                  <p className="chat-notice" role="status">
                    {history.notice}
                  </p>
                )}
                <Thread
                  suggestions={profile?.suggestions ?? NO_SUGGESTIONS}
                  composerPlaceholder={profile?.composerPlaceholder ?? ""}
                />
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
