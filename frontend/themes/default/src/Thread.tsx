import {
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
  CompassIcon,
  ListIcon,
  UserIcon,
  ArrowRightIcon,
} from "./Icons";
import type { ReactNode } from "react";
import type { ChatSource, ProfileSuggestion } from "@knowva/core";

const MarkdownText = () => (
  <MarkdownTextPrimitive remarkPlugins={[remarkGfm]} className="aui-md" />
);

const sourceLabel = (source: ChatSource): string => {
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
  if (!Array.isArray(data) || data.length === 0) return null;

  return (
    <section className="source-list" aria-label="資料來源">
      <h3>資料來源</h3>
      <ol>
        {data.map((source) => (
          <li key={source.sourceId}>{sourceLabel(source)}</li>
        ))}
      </ol>
    </section>
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

const AssistantMessage = () => (
  <MessagePrimitive.Root className="msg msg-assistant">
    <span className="msg-avatar">
      <BrandMark />
    </span>
    <div className="msg-body">
      <MessagePrimitive.Parts
        components={{
          Text: MarkdownText,
          data: { by_name: { sources: SourceList } },
        }}
      />
      <MessagePrimitive.Error>
        <ErrorPrimitive.Root className="msg-error">
          <ErrorPrimitive.Message />
        </ErrorPrimitive.Root>
      </MessagePrimitive.Error>
      <MessagePrimitive.If hasContent={false}>
        <div className="typing" aria-label="思考中">
          <span />
          <span />
          <span />
        </div>
      </MessagePrimitive.If>
    </div>
  </MessagePrimitive.Root>
);

const Composer = ({ placeholder }: { placeholder: string }) => (
  <ComposerPrimitive.Root className="composer">
    <ComposerPrimitive.Input
      className="composer-input"
      placeholder={placeholder}
      submitMode="enter"
      maxRows={8}
    />
    <div className="composer-actions">
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
