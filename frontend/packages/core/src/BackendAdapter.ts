import type {
  ChatModelAdapter,
  ChatModelRunOptions,
  ThreadAssistantMessagePart,
} from "@assistant-ui/react";

export type ChatSource = {
  sourceId: string;
  documentId: string;
  unitId: number;
  unitIndex?: number | null;
  filename: string;
  pageNumber: number | null;
  kind: "page" | "text";
  /**
   * Set on a catalog answer's sources: the answer saw the document's card
   * (filename, summary, tags), and `unitId` is only its first unit.
   */
  catalog?: boolean;
};

/**
 * Every chat failure a theme can show. Raw server, SSE and provider text never
 * reaches the thread; it is always replaced by one of these.
 */
export const CHAT_ERROR_MESSAGES = {
  /** The backend sent an SSE `error` event. */
  generation: "系統暫時無法產生回覆，請稍後再試。",
  malformed: "回覆格式異常，請重新送出問題。",
  interrupted: "回覆傳輸中斷，請重新送出問題。",
  emptyBody: "問答服務未傳回內容，請稍後再試。",
} as const;

class SafeChatError extends Error {}

export function chatRequestErrorMessage(status?: number): string {
  if (status === undefined) {
    return "無法連線至問答服務，請確認網路後再試。";
  }
  if (status === 401 || status === 403) {
    return "目前沒有使用知識問答的權限。";
  }
  if (status === 429) return "目前詢問人數較多，請稍後再試。";
  return "問答服務暫時無法使用，請稍後再試。";
}

const isAbortError = (error: unknown, signal: AbortSignal): boolean =>
  signal.aborted ||
  (error instanceof Error && error.name === "AbortError");

const parseSseEvent = (event: string): unknown | null => {
  const data = event
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""))
    .join("\n");
  if (!data) return null;
  return JSON.parse(data) as unknown;
};

async function* readSsePayloads(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer = (buffer + decoder.decode(value, { stream: true })).replace(
        /\r\n/g,
        "\n",
      );
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";
      for (const event of events) {
        const payload = parseSseEvent(event);
        if (payload !== null) yield payload;
      }
    }

    buffer = (buffer + decoder.decode()).replace(/\r\n/g, "\n");
    if (buffer.trim()) {
      const payload = parseSseEvent(buffer);
      if (payload !== null) yield payload;
    }
  } finally {
    reader.releaseLock();
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const isChatSource = (value: unknown): value is ChatSource => {
  if (!isRecord(value)) return false;
  return (
    typeof value.sourceId === "string" &&
    value.sourceId.length > 0 &&
    typeof value.documentId === "string" &&
    value.documentId.length > 0 &&
    typeof value.unitId === "number" &&
    Number.isSafeInteger(value.unitId) &&
    value.unitId > 0 &&
    (value.unitIndex === undefined ||
      value.unitIndex === null ||
      (typeof value.unitIndex === "number" &&
        Number.isSafeInteger(value.unitIndex) &&
        value.unitIndex >= 0)) &&
    typeof value.filename === "string" &&
    value.filename.length > 0 &&
    (value.catalog === undefined || typeof value.catalog === "boolean") &&
    ((value.kind === "page" &&
      typeof value.pageNumber === "number" &&
      Number.isSafeInteger(value.pageNumber) &&
      value.pageNumber > 0) ||
      (value.kind === "text" && value.pageNumber === null))
  );
};

/** The sources of one answer, or null unless every entry is well formed and unique. */
export const sourceData = (value: unknown): ChatSource[] | null => {
  if (!Array.isArray(value) || !value.every(isChatSource)) return null;
  const sourceIds = new Set(value.map((source) => source.sourceId));
  return sourceIds.size === value.length ? value : null;
};

/**
 * Sources of the most recent answer, in the order the backend laid them out.
 *
 * The inference prompt cache matches on a prefix, so the backend keeps those blocks where
 * they already are and appends the newly retrieved ones. `/api/chat` is stateless and the
 * thread lives here, so the order is only known to the backend if this sends it back.
 */
const previousSources = (
  messages: ChatModelRunOptions["messages"],
): ChatSource[] => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "assistant") continue;
    const part = message.content.find(
      (part) => part.type === "data" && part.name === "sources",
    );
    if (part?.type !== "data") return [];
    return sourceData(part.data) ?? [];
  }
  return [];
};

const appendTextDelta = (
  content: readonly ThreadAssistantMessagePart[],
  activeTextIndex: number | null,
  delta: string,
): { content: ThreadAssistantMessagePart[]; activeTextIndex: number } => {
  if (activeTextIndex === null) {
    return {
      content: [...content, { type: "text", text: delta }],
      activeTextIndex: content.length,
    };
  }

  const current = content[activeTextIndex];
  if (current?.type !== "text") {
    throw new SafeChatError(CHAT_ERROR_MESSAGES.malformed);
  }
  const updated = { ...current, text: current.text + delta };
  return {
    content: [
      ...content.slice(0, activeTextIndex),
      updated,
      ...content.slice(activeTextIndex + 1),
    ],
    activeTextIndex,
  };
};

export const BackendAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    let response: Response;
    try {
      response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: messages.map((message) => ({
            role: message.role,
            content: message.content
              .flatMap((part) => (part.type === "text" ? [part.text] : []))
              .join("\n"),
          })),
          previousSources: previousSources(messages),
        }),
        signal: abortSignal,
      });
    } catch (error) {
      if (isAbortError(error, abortSignal)) throw error;
      throw new SafeChatError(chatRequestErrorMessage());
    }

    if (!response.ok) {
      throw new SafeChatError(chatRequestErrorMessage(response.status));
    }
    if (!response.body) {
      throw new SafeChatError(CHAT_ERROR_MESSAGES.emptyBody);
    }

    let content: ThreadAssistantMessagePart[] = [];
    let activeTextIndex: number | null = null;
    let sources: ChatSource[] | null = null;
    let completed = false;

    try {
      for await (const value of readSsePayloads(response.body)) {
        if (!isRecord(value) || typeof value.type !== "string") continue;
        if (completed) {
          throw new SafeChatError(CHAT_ERROR_MESSAGES.malformed);
        }

        if (value.type === "text-delta") {
          if (sources !== null || typeof value.text !== "string") {
            throw new SafeChatError(CHAT_ERROR_MESSAGES.malformed);
          }
          const next = appendTextDelta(content, activeTextIndex, value.text);
          content = next.content;
          activeTextIndex = next.activeTextIndex;
          yield { content };
          continue;
        }

        if (value.type === "sources") {
          if (sources !== null) {
            throw new SafeChatError(CHAT_ERROR_MESSAGES.malformed);
          }
          sources = sourceData(value.sources);
          if (sources === null) {
            throw new SafeChatError(CHAT_ERROR_MESSAGES.malformed);
          }
          continue;
        }

        if (value.type === "done") {
          if (sources === null) {
            throw new SafeChatError(CHAT_ERROR_MESSAGES.malformed);
          }
          content = [
            ...content,
            { type: "data", name: "sources", data: sources },
          ];
          activeTextIndex = null;
          completed = true;
          yield { content };
          continue;
        }

        if (value.type === "error") {
          throw new SafeChatError(CHAT_ERROR_MESSAGES.generation);
        }
      }
      if (!completed) {
        throw new SafeChatError(CHAT_ERROR_MESSAGES.interrupted);
      }
    } catch (error) {
      if (isAbortError(error, abortSignal)) throw error;
      if (error instanceof SafeChatError) throw error;
      throw new SafeChatError(CHAT_ERROR_MESSAGES.interrupted);
    }
  },
};
