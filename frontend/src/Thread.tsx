import {
  ThreadPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  ErrorPrimitive,
} from "@assistant-ui/react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";
import {
  BrandMark,
  SendIcon,
  StopIcon,
  AlertIcon,
  WrenchIcon,
  SlidersIcon,
  ArrowRightIcon,
} from "./Icons";
import type { ReactNode } from "react";

const MarkdownText = () => (
  <MarkdownTextPrimitive remarkPlugins={[remarkGfm]} className="aui-md" />
);

const SUGGESTIONS: { icon: ReactNode; text: string }[] = [
  { icon: <AlertIcon />, text: "知識庫目前涵蓋哪些主題?" },
  { icon: <WrenchIcon />, text: "幫我整理知識庫內容的重點摘要" },
  { icon: <SlidersIcon />, text: "新進人員應該先了解哪些重要資訊?" },
];

const Welcome = () => (
  <div className="welcome">
    <span className="welcome-mark">
      <BrandMark />
    </span>
    <h2 className="welcome-title">請問有什麼可以協助?</h2>
    <p className="welcome-sub">
      詢問知識庫中的任何內容,系統會依據已上傳的文件作答並標註來源。
    </p>
    <div className="suggestions">
      {SUGGESTIONS.map((s) => (
        <ThreadPrimitive.Suggestion
          key={s.text}
          prompt={s.text}
          send
          className="suggestion"
        >
          <span className="suggestion-icon">{s.icon}</span>
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
      <MessagePrimitive.Parts components={{ Text: MarkdownText }} />
      <MessagePrimitive.Error>
        <ErrorPrimitive.Root className="msg-error">
          <ErrorPrimitive.Message />
        </ErrorPrimitive.Root>
      </MessagePrimitive.Error>
      <MessagePrimitive.If hasContent={false}>
        <div className="typing" aria-label="生成回覆中">
          <span />
          <span />
          <span />
        </div>
      </MessagePrimitive.If>
    </div>
  </MessagePrimitive.Root>
);

const Composer = () => (
  <ComposerPrimitive.Root className="composer">
    <ComposerPrimitive.Input
      className="composer-input"
      placeholder="輸入你的問題,系統會依據知識庫回答並附上來源"
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

export const Thread = () => (
  <ThreadPrimitive.Root className="thread">
    <ThreadPrimitive.Viewport className="thread-viewport">
      <div className="thread-col">
        <ThreadPrimitive.Empty>
          <Welcome />
        </ThreadPrimitive.Empty>
        <ThreadPrimitive.Messages
          components={{ UserMessage, AssistantMessage }}
        />
      </div>
    </ThreadPrimitive.Viewport>
    <div className="composer-bar">
      <div className="composer-col">
        <Composer />
        <p className="composer-hint">
          Enter 送出 · Shift + Enter 換行 · 回答由 AI 生成,請以原始文件為準
        </p>
      </div>
    </div>
  </ThreadPrimitive.Root>
);
