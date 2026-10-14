import {
  AuiIf,
  ThreadPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  ErrorPrimitive,
} from "@assistant-ui/react";
import type { DataMessagePartProps } from "@assistant-ui/react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";
import {
  BrandMark,
  SendIcon,
  StopIcon,
  MicIcon,
  CompassIcon,
  ListIcon,
  UserIcon,
  ArrowRightIcon,
  CloseIcon,
  DownloadIcon,
} from "./Icons";
import { createContext, useContext, useEffect, useId, useRef, useState } from "react";
import type { ComponentProps, ReactNode } from "react";
import {
  documentFileUrl,
  remarkSourceMarkers,
  SOURCE_MARKER_ELEMENT,
  useMessageSources,
  useSourcePreview,
} from "@knowva/core";
import type { ChatSource, ProfileSuggestion } from "@knowva/core";

/** Opens a source in its message's preview; null outside an assistant message. */
const OpenSourceContext = createContext<((source: ChatSource) => void) | null>(
  null,
);

type SourceMarkerProps = ComponentProps<"button"> & {
  "data-source-id"?: unknown;
};

/** An answer's `[S1]`: a button when the message has that source, text otherwise. */
const SourceMarker = ({ children, "data-source-id": sourceId }: SourceMarkerProps) => {
  const open = useContext(OpenSourceContext);
  const source = useMessageSources().find((entry) => entry.sourceId === sourceId);
  if (!open || !source) return <>{children}</>;
  return (
    <button
      type="button"
      className="source-marker"
      title={sourceLabel(source)}
      onClick={() => open(source)}
    >
      {children}
    </button>
  );
};

const REMARK_PLUGINS = [remarkGfm, remarkSourceMarkers];
const MARKDOWN_COMPONENTS = { [SOURCE_MARKER_ELEMENT]: SourceMarker };

const MarkdownText = () => (
  <MarkdownTextPrimitive
    remarkPlugins={REMARK_PLUGINS}
    components={MARKDOWN_COMPONENTS}
    className="aui-md"
  />
);

const sourceLabel = (source: ChatSource): string => {
  // A catalog answer cited the whole document's card, not one passage.
  if (source.catalog === true) return source.filename;
  if (source.kind === "page") {
    return source.pageNumber !== null && Number.isFinite(source.pageNumber)
      ? `${source.filename} · 第 ${source.pageNumber} 頁`
      : `${source.filename} · 頁面`;
  }

  return typeof source.unitIndex === "number" &&
    Number.isInteger(source.unitIndex) &&
    source.unitIndex >= 0
    ? `${source.filename} · 第 ${source.unitIndex + 1} 段`
    : `${source.filename} · 段落`;
};

type SourceListProps = Pick<DataMessagePartProps<ChatSource[]>, "data">;

export const SourceList = ({ data }: SourceListProps) => {
  const open = useContext(OpenSourceContext);
  if (!Array.isArray(data) || data.length === 0) return null;

  return (
    <section className="source-list" aria-label="資料來源">
      <h3>資料來源</h3>
      <ol>
        {data.map((source) => (
          <li key={source.sourceId}>
            <button
              type="button"
              className="source-entry"
              onClick={() => open?.(source)}
            >
              {sourceLabel(source)}
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
};

type SourcePreviewDialogProps = {
  /** The source to show; null keeps the dialog closed. */
  source: ChatSource | null;
  onClose: () => void;
};

/**
 * What one source cited, in a modal dialog: the page image or the passage,
 * and the original file to download. The dialog follows `source`. Escape (its
 * `cancel` event), the close button and a click on the backdrop each call
 * `onClose` straight away: the `close` event comes later, and a hidden tab
 * may not get it at all, so it only covers any other way the dialog closes.
 */
export const SourcePreviewDialog = ({
  source,
  onClose,
}: SourcePreviewDialogProps) => {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const preview = useSourcePreview(source);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (source && !dialog.open) dialog.showModal();
    if (!source && dialog.open) dialog.close();
  }, [source]);

  return (
    <dialog
      ref={dialogRef}
      className="source-preview"
      aria-labelledby={titleId}
      onCancel={onClose}
      onClose={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      {source && (
        <div className="source-preview-panel">
          <header className="source-preview-head">
            <h2 id={titleId}>{sourceLabel(source)}</h2>
            <button
              type="button"
              className="source-preview-close"
              aria-label="關閉"
              onClick={onClose}
            >
              <CloseIcon />
            </button>
          </header>
          <div className="source-preview-body">
            {preview.status === "loading" && (
              <p className="source-preview-status">載入中…</p>
            )}
            {preview.status === "image" && (
              <img src={preview.url} alt={sourceLabel(source)} />
            )}
            {preview.status === "text" && (
              <p className="source-preview-text">{preview.text}</p>
            )}
            {preview.status === "error" && (
              <p className="source-preview-error" role="alert">
                {preview.message}
              </p>
            )}
          </div>
          <footer className="source-preview-foot">
            <a
              className="source-preview-download"
              href={documentFileUrl(source.documentId)}
              download
            >
              <DownloadIcon />
              下載原檔
            </a>
          </footer>
        </div>
      )}
    </dialog>
  );
};

/** Suggestions take these icons in turn, in the order the profile lists them. */
const SUGGESTION_ICONS: readonly ReactNode[] = [
  <CompassIcon key="compass" />,
  <ListIcon key="list" />,
  <UserIcon key="user" />,
];

type WelcomeProps = { suggestions: readonly ProfileSuggestion[] };

const Welcome = ({ suggestions }: WelcomeProps) => (
  <div className="welcome">
    <span className="welcome-mark">
      <BrandMark />
    </span>
    <h2 className="welcome-title">想知道什麼？</h2>
    <div className="suggestions">
      {suggestions.map((s, index) => (
        <ThreadPrimitive.Suggestion
          key={s.text}
          prompt={s.text}
          send
          className="suggestion"
        >
          <span className="suggestion-icon">
            {SUGGESTION_ICONS[index % SUGGESTION_ICONS.length]}
          </span>
          <span className="suggestion-text">{s.text}</span>
          <span className="suggestion-go">
            <ArrowRightIcon />
          </span>
        </ThreadPrimitive.Suggestion>
      ))}
    </div>
  </div>
);

const UserMessage = () => (
  <MessagePrimitive.Root className="msg msg-user">
    <div className="user-bubble">
      <MessagePrimitive.Parts />
    </div>
  </MessagePrimitive.Root>
);

const AssistantMessage = () => {
  const [openedSource, setOpenedSource] = useState<ChatSource | null>(null);

  return (
    <MessagePrimitive.Root className="msg msg-assistant">
      <span className="msg-avatar">
        <BrandMark />
      </span>
      <div className="msg-body">
        <OpenSourceContext.Provider value={setOpenedSource}>
          <MessagePrimitive.Parts
            components={{
              Text: MarkdownText,
              data: { by_name: { sources: SourceList } },
            }}
          />
        </OpenSourceContext.Provider>
        <SourcePreviewDialog
          source={openedSource}
          onClose={() => setOpenedSource(null)}
        />
        <MessagePrimitive.Error>
          <ErrorPrimitive.Root className="msg-error">
            <ErrorPrimitive.Message />
          </ErrorPrimitive.Root>
        </MessagePrimitive.Error>
        <AuiIf
          condition={(s) =>
            s.message.status?.type === "running" && s.message.parts.length === 0
          }
        >
          <div className="typing" aria-label="思考中">
            <span />
            <span />
            <span />
          </div>
        </AuiIf>
        <AuiIf
          condition={(s) =>
            s.message.status?.type === "incomplete" &&
            s.message.status.reason === "cancelled" &&
            s.message.parts.length === 0
          }
        >
          <p className="msg-stopped">已停止回覆</p>
        </AuiIf>
      </div>
    </MessagePrimitive.Root>
  );
};

const Composer = ({ placeholder }: { placeholder: string }) => (
  <ComposerPrimitive.Root className="composer">
    <ComposerPrimitive.Input
      className="composer-input"
      placeholder={placeholder}
      submitMode="enter"
      maxRows={8}
    />
    <div className="composer-actions">
      <AuiIf condition={(s) => s.thread.capabilities.dictation}>
        <AuiIf condition={(s) => s.composer.dictation == null}>
          <ComposerPrimitive.Dictate
            className="composer-btn composer-btn-mic"
            aria-label="語音輸入"
          >
            <MicIcon />
          </ComposerPrimitive.Dictate>
        </AuiIf>
        <AuiIf condition={(s) => s.composer.dictation != null}>
          <ComposerPrimitive.StopDictation
            className="composer-btn composer-btn-mic composer-btn-listening"
            aria-label="停止語音輸入"
          >
            <MicIcon />
          </ComposerPrimitive.StopDictation>
        </AuiIf>
      </AuiIf>
      <ThreadPrimitive.If running={false}>
        <ComposerPrimitive.Send className="composer-btn" aria-label="送出">
          <SendIcon />
        </ComposerPrimitive.Send>
      </ThreadPrimitive.If>
      <ThreadPrimitive.If running>
        <ComposerPrimitive.Cancel
          className="composer-btn composer-btn-stop"
          aria-label="停止生成"
        >
          <StopIcon />
        </ComposerPrimitive.Cancel>
      </ThreadPrimitive.If>
    </div>
  </ComposerPrimitive.Root>
);

type ThreadProps = {
  /** The profile's suggested questions; empty until the profile arrives. */
  suggestions: readonly ProfileSuggestion[];
  /** The profile's composer placeholder; empty until the profile arrives. */
  composerPlaceholder: string;
};

export const Thread = ({ suggestions, composerPlaceholder }: ThreadProps) => (
  <ThreadPrimitive.Root className="thread">
    <ThreadPrimitive.Viewport className="thread-viewport">
      <div className="thread-col">
        <ThreadPrimitive.Empty>
          <Welcome suggestions={suggestions} />
        </ThreadPrimitive.Empty>
        <ThreadPrimitive.Messages
          components={{ UserMessage, AssistantMessage }}
        />
      </div>
    </ThreadPrimitive.Viewport>
    <div className="composer-bar">
      <div className="composer-col">
        <Composer placeholder={composerPlaceholder} />
        <p className="composer-hint">
          內容由語言模型生成，可能有誤，以原始文件為準。
        </p>
      </div>
    </div>
  </ThreadPrimitive.Root>
);
