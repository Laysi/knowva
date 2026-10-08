import type {
  ExportedMessageRepository,
  ExportedMessageRepositoryItem,
  MessageStatus,
  ThreadAssistantMessagePart,
  ThreadMessage,
} from "@assistant-ui/react";
import { sourceData } from "./BackendAdapter";

/**
 * Conversations live in the server's database, one list per browser: the
 * backend names the browser with an HttpOnly cookie it issues itself, so this
 * code never sees or sends an owner. `/api/chat` stays stateless; each
 * finished message is saved here afterwards.
 */
const CONVERSATIONS_PATH = "/api/conversations";

/** Past this, starting a conversation deletes the least recently updated one. */
export const MAX_CONVERSATIONS = 50;
/** Titles are the first this many Unicode code points of the first question. */
export const CONVERSATION_TITLE_LENGTH = 30;

/**
 * The one line a theme shows while the open conversation is not being saved.
 * Answers keep working.
 */
export const CONVERSATION_NOT_SAVED_MESSAGE =
  "這串對話無法保存到伺服器，重新整理或關閉頁面後就會消失。";
/**
 * Shown instead when the open conversation's messages could not be read back.
 * Nothing is saved to it then, so the stored copy stays intact.
 */
export const CONVERSATION_LOAD_FAILED_MESSAGE =
  "這串對話的內容載入失敗，新的問答也不會保存，請重新整理頁面後再試。";

export type ConversationAction = "list" | "delete" | "clear";

export function conversationRequestErrorMessage(
  action: ConversationAction,
  status?: number,
): string {
  if (status === undefined) {
    if (action === "list") {
      return "連不上伺服器，無法載入對話紀錄，請確認網路後再試。";
    }
    return "連不上伺服器，無法刪除對話，請確認網路後再試。";
  }
  if (action === "list") return "對話紀錄載入失敗，請稍後再試。";
  if (action === "delete") return "刪除對話失敗，請稍後再試。";
  return "清除對話紀錄失敗，請稍後再試。";
}

/** A failed conversation request; `status` is undefined when it never got an answer. */
export class ConversationRequestError extends Error {
  readonly status: number | undefined;

  constructor(status: number | undefined) {
    super(
      status === undefined
        ? "The conversation request did not get an answer."
        : `The conversation request failed with HTTP ${status}.`,
    );
    this.name = "ConversationRequestError";
    this.status = status;
  }
}

/** Worth sending again later: no answer, a timeout, throttling or a server fault. */
export const isTransientFailure = (error: unknown): boolean =>
  !(error instanceof ConversationRequestError) ||
  error.status === undefined ||
  error.status === 408 ||
  error.status === 429 ||
  error.status >= 500;

export type ConversationSummary = {
  readonly id: string;
  /** Null until the first question is saved. */
  readonly title: string | null;
  readonly createdAt: string;
  readonly updatedAt: string;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isTimestamp = (value: unknown): value is string =>
  typeof value === "string" && !Number.isNaN(Date.parse(value));

const isConversationSummary = (value: unknown): value is ConversationSummary =>
  isRecord(value) &&
  typeof value.id === "string" &&
  value.id.length > 0 &&
  (value.title === null || typeof value.title === "string") &&
  isTimestamp(value.createdAt) &&
  isTimestamp(value.updatedAt);

const send = async (path: string, init?: RequestInit): Promise<Response> => {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ConversationRequestError(undefined);
  }
  if (!response.ok) throw new ConversationRequestError(response.status);
  return response;
};

/** A response that is not the expected JSON counts as a server fault. */
const json = async <T>(
  response: Response,
  accept: (value: unknown) => value is T,
): Promise<T> => {
  const value: unknown = await response.json().catch(() => undefined);
  if (!accept(value)) throw new ConversationRequestError(502);
  return value;
};

const conversationPath = (id: string) =>
  `${CONVERSATIONS_PATH}/${encodeURIComponent(id)}`;

const isSummaryList = (value: unknown): value is ConversationSummary[] =>
  Array.isArray(value) && value.every(isConversationSummary);

const isCreated = (
  value: unknown,
): value is { conversation: ConversationSummary; removedIds: string[] } =>
  isRecord(value) &&
  isConversationSummary(value.conversation) &&
  Array.isArray(value.removedIds) &&
  value.removedIds.every((id) => typeof id === "string");

const isSaved = (
  value: unknown,
): value is { conversation: ConversationSummary } =>
  isRecord(value) && isConversationSummary(value.conversation);

/** Most recently updated first. */
export const listConversations = async (): Promise<ConversationSummary[]> =>
  json(await send(CONVERSATIONS_PATH), isSummaryList);

/** The new conversation, and the older ones deleted to stay at the limit. */
export const createConversation = async (): Promise<{
  conversation: ConversationSummary;
  removedIds: string[];
}> => json(await send(CONVERSATIONS_PATH, { method: "POST" }), isCreated);

export const fetchConversation = async (
  id: string,
): Promise<ConversationSummary> =>
  json(await send(conversationPath(id)), isConversationSummary);

export const deleteConversation = async (id: string): Promise<void> => {
  await send(conversationPath(id), { method: "DELETE" });
};

/** Deletes every conversation of this browser. */
export const clearConversations = async (): Promise<void> => {
  await send(CONVERSATIONS_PATH, { method: "DELETE" });
};

/**
 * Saves one message, replacing an earlier save with the same id, and returns
 * the conversation as it is now, titled once its first question is saved.
 */
export const saveMessage = async (
  id: string,
  { parentId, message }: ExportedMessageRepositoryItem,
): Promise<ConversationSummary> => {
  const response = await send(`${conversationPath(id)}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ parentId, message }),
  });
  return (await json(response, isSaved)).conversation;
};

export const loadMessages = async (
  id: string,
): Promise<ExportedMessageRepository> => {
  const response = await send(`${conversationPath(id)}/messages`);
  const value: unknown = await response.json().catch(() => undefined);
  const repository = decodeStoredMessages(value);
  if (!repository) throw new ConversationRequestError(502);
  return repository;
};

/**
 * The first 30 Unicode code points of a question, with whitespace runs folded
 * into one space and the ends trimmed: the backend's rule, used here only for a
 * conversation the server could not keep.
 */
export const conversationTitle = (question: string): string =>
  Array.from(question.trim().replace(/\s+/g, " "))
    .slice(0, CONVERSATION_TITLE_LENGTH)
    .join("");

/** The text of the first question in `messages`. */
export const firstQuestion = (messages: readonly ThreadMessage[]): string => {
  const question = messages.find(({ role }) => role === "user");
  if (!question) return "";
  return question.content
    .flatMap((part) => (part.type === "text" ? [part.text] : []))
    .join("\n");
};

const MAX_MESSAGE_ID_LENGTH = 128;

const isMessageId = (value: unknown): value is string =>
  typeof value === "string" &&
  value.length > 0 &&
  value.length <= MAX_MESSAGE_ID_LENGTH;

const INCOMPLETE_REASONS = [
  "cancelled",
  "tool-calls",
  "length",
  "content-filter",
  "other",
  "error",
] as const;

/** Anything unexpected reads as an answer that was cut off. */
const decodeStatus = (value: unknown): MessageStatus => {
  if (
    isRecord(value) &&
    value.type === "complete" &&
    (value.reason === "stop" || value.reason === "unknown")
  ) {
    return { type: "complete", reason: value.reason };
  }
  const reason = isRecord(value)
    ? INCOMPLETE_REASONS.find((known) => known === value.reason)
    : undefined;
  if (isRecord(value) && value.type === "incomplete" && reason !== undefined) {
    return typeof value.error === "string"
      ? { type: "incomplete", reason, error: value.error }
      : { type: "incomplete", reason };
  }
  return { type: "incomplete", reason: "cancelled" };
};

const decodeTime = (value: unknown): Date => {
  const time = typeof value === "string" ? Date.parse(value) : NaN;
  return new Date(Number.isNaN(time) ? 0 : time);
};

const text = (part: unknown): { type: "text"; text: string } | null =>
  isRecord(part) && part.type === "text" && typeof part.text === "string"
    ? { type: "text", text: part.text }
    : null;

/**
 * Rebuilds one stored message from what the page needs and nothing else: text
 * parts, and an answer's well-formed `sources` data part. Whatever else was
 * stored is dropped, and text is only ever rendered as text.
 */
const decodeMessage = (value: unknown): ThreadMessage | null => {
  if (
    !isRecord(value) ||
    !isMessageId(value.id) ||
    !Array.isArray(value.content)
  ) {
    return null;
  }
  const common = { id: value.id, createdAt: decodeTime(value.createdAt) };
  if (value.role === "user") {
    return {
      ...common,
      role: "user",
      content: value.content.flatMap((part) => text(part) ?? []),
      attachments: [],
      metadata: { custom: {} },
    };
  }
  if (value.role !== "assistant") return null;
  return {
    ...common,
    role: "assistant",
    content: value.content.flatMap((part): ThreadAssistantMessagePart[] => {
      const textPart = text(part);
      if (textPart) return [textPart];
      if (isRecord(part) && part.type === "data" && part.name === "sources") {
        const data = sourceData(part.data);
        if (data) return [{ type: "data", name: "sources", data }];
      }
      return [];
    }),
    status: decodeStatus(value.status),
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {},
    },
  };
};

/**
 * `GET /api/conversations/{id}/messages` as assistant-ui's repository, or null
 * when it is not a list. Messages come in save order; one that cannot be read,
 * repeats an id, or follows a parent that is not there is dropped with
 * everything after it on its branch. The head is the last message kept.
 */
export function decodeStoredMessages(
  value: unknown,
): ExportedMessageRepository | null {
  if (!Array.isArray(value)) return null;
  const ids = new Set<string>();
  const messages: ExportedMessageRepositoryItem[] = [];
  for (const entry of value) {
    if (!isRecord(entry)) continue;
    const parentId = entry.parentId ?? null;
    if (parentId !== null && !(isMessageId(parentId) && ids.has(parentId))) {
      continue;
    }
    const message = decodeMessage(entry.message);
    if (!message || ids.has(message.id)) continue;
    ids.add(message.id);
    messages.push({ parentId, message });
  }
  const head = messages.at(-1);
  return head ? { headId: head.message.id, messages } : { messages: [] };
}
