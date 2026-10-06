import type {
  ExportedMessageRepository,
  ExportedMessageRepositoryItem,
  MessageStatus,
  ThreadMessage,
} from "@assistant-ui/react";
import { sourceData } from "./BackendAdapter";
import type { ChatSource } from "./BackendAdapter";

/**
 * Conversations live only in this browser's localStorage. `/api/chat` stays
 * stateless and there is no login, so the server could not tell whose history
 * it would be keeping.
 */
export const CONVERSATION_STORAGE_PREFIX = "knowva:v1:";
/** The list: id, title, created and last updated time of each conversation. */
export const CONVERSATION_INDEX_KEY = `${CONVERSATION_STORAGE_PREFIX}conversations`;
const MESSAGES_KEY_PREFIX = `${CONVERSATION_STORAGE_PREFIX}conversation:`;
/** The messages of one conversation. */
export const conversationMessagesKey = (id: string): string =>
  `${MESSAGES_KEY_PREFIX}${id}`;

/** Past this, the least recently updated conversation is deleted. */
export const MAX_STORED_CONVERSATIONS = 50;
export const CONVERSATION_TITLE_LENGTH = 30;
/**
 * The one line a theme shows while the open conversation is not being kept:
 * the browser blocks storage, or it is full even after older conversations
 * were deleted. Answers keep working either way.
 */
export const CONVERSATION_NOT_SAVED_MESSAGE =
  "瀏覽器無法保存這串對話，重新整理或關閉頁面後就會消失。";

const STORAGE_VERSION = 1;
const CONVERSATION_ID = /^[A-Za-z0-9_-]{1,64}$/;
const MAX_MESSAGE_ID_LENGTH = 128;

export type StoredConversation = {
  readonly id: string;
  /** Empty until the first question is stored. */
  readonly title: string;
  readonly createdAt: number;
  readonly updatedAt: number;
};

type StoredPart =
  | { readonly type: "text"; readonly text: string }
  | {
      readonly type: "data";
      readonly name: "sources";
      readonly data: ChatSource[];
    };

const INCOMPLETE_REASONS = [
  "cancelled",
  "tool-calls",
  "length",
  "content-filter",
  "other",
  "error",
] as const;

type StoredStatus =
  | { readonly type: "complete"; readonly reason: "stop" | "unknown" }
  | {
      readonly type: "incomplete";
      readonly reason: (typeof INCOMPLETE_REASONS)[number];
      readonly error?: string;
    };

type StoredMessage = {
  readonly id: string;
  readonly parentId: string | null;
  readonly createdAt: number;
} & (
  | { readonly role: "user"; readonly content: StoredPart[] }
  | {
      readonly role: "assistant";
      readonly content: StoredPart[];
      readonly status: StoredStatus;
    }
);

type StoredThread = {
  readonly headId: string;
  readonly messages: readonly StoredMessage[];
};

/**
 * The first 30 Unicode code points of the first question, with whitespace runs
 * folded into one space. It never involves the model or the network.
 */
export const conversationTitle = (question: string): string =>
  Array.from(question.trim().replace(/\s+/g, " "))
    .slice(0, CONVERSATION_TITLE_LENGTH)
    .join("");

const messageText = (message: {
  readonly content: readonly { readonly type: string; readonly text?: string }[];
}): string =>
  message.content
    .flatMap((part) =>
      part.type === "text" && typeof part.text === "string" ? [part.text] : [],
    )
    .join("\n");

/**
 * The browser's localStorage, or null where it is missing or blocked. Some
 * browsers hand out the object and only throw once it is used.
 */
export const browserConversationStorage = (): Storage | null => {
  try {
    const storage = globalThis.localStorage;
    if (!storage) return null;
    storage.getItem(CONVERSATION_INDEX_KEY);
    return storage;
  } catch {
    return null;
  }
};

const isQuotaError = (error: unknown): boolean =>
  typeof error === "object" &&
  error !== null &&
  "name" in error &&
  (error.name === "QuotaExceededError" ||
    error.name === "NS_ERROR_DOM_QUOTA_REACHED");

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isTime = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value >= 0;

const isMessageId = (value: unknown): value is string =>
  typeof value === "string" &&
  value.length > 0 &&
  value.length <= MAX_MESSAGE_ID_LENGTH;

const parseJson = (raw: string): unknown => {
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return undefined;
  }
};

const parseConversation = (value: unknown): StoredConversation | null => {
  if (
    !isRecord(value) ||
    typeof value.id !== "string" ||
    !CONVERSATION_ID.test(value.id) ||
    typeof value.title !== "string" ||
    Array.from(value.title).length > CONVERSATION_TITLE_LENGTH ||
    !isTime(value.createdAt) ||
    !isTime(value.updatedAt)
  ) {
    return null;
  }
  return {
    id: value.id,
    title: value.title,
    createdAt: value.createdAt,
    updatedAt: value.updatedAt,
  };
};

const parsePart = (
  value: unknown,
  role: StoredMessage["role"],
): StoredPart | null => {
  if (!isRecord(value)) return null;
  if (value.type === "text" && typeof value.text === "string") {
    return { type: "text", text: value.text };
  }
  if (role === "assistant" && value.type === "data" && value.name === "sources") {
    const data = sourceData(value.data);
    return data && { type: "data", name: "sources", data };
  }
  return null;
};

const parseStatus = (value: unknown): StoredStatus | null => {
  if (!isRecord(value)) return null;
  if (
    value.type === "complete" &&
    (value.reason === "stop" || value.reason === "unknown")
  ) {
    return { type: "complete", reason: value.reason };
  }
  const reason = INCOMPLETE_REASONS.find((known) => known === value.reason);
  if (
    value.type === "incomplete" &&
    reason !== undefined &&
    (value.error === undefined || typeof value.error === "string")
  ) {
    return typeof value.error === "string"
      ? { type: "incomplete", reason, error: value.error }
      : { type: "incomplete", reason };
  }
  return null;
};

const parseMessage = (value: unknown): StoredMessage | null => {
  if (
    !isRecord(value) ||
    !isMessageId(value.id) ||
    !(value.parentId === null || isMessageId(value.parentId)) ||
    !isTime(value.createdAt) ||
    (value.role !== "user" && value.role !== "assistant") ||
    !Array.isArray(value.content)
  ) {
    return null;
  }
  const role = value.role;
  const content: StoredPart[] = [];
  for (const part of value.content) {
    const parsed = parsePart(part, role);
    if (!parsed) return null;
    content.push(parsed);
  }
  const common = {
    id: value.id,
    parentId: value.parentId,
    createdAt: value.createdAt,
  };
  if (role === "user") {
    return value.status === undefined ? { ...common, role, content } : null;
  }
  const status = parseStatus(value.status);
  return status && { ...common, role, content, status };
};

/** Every parent comes before its child and the head is one of the messages. */
const parseThread = (value: unknown): StoredThread | null => {
  if (
    !isRecord(value) ||
    value.version !== STORAGE_VERSION ||
    !isMessageId(value.headId) ||
    !Array.isArray(value.messages) ||
    value.messages.length === 0
  ) {
    return null;
  }
  const ids = new Set<string>();
  const messages: StoredMessage[] = [];
  for (const entry of value.messages) {
    const message = parseMessage(entry);
    if (
      !message ||
      ids.has(message.id) ||
      (message.parentId !== null && !ids.has(message.parentId))
    ) {
      return null;
    }
    ids.add(message.id);
    messages.push(message);
  }
  return ids.has(value.headId) ? { headId: value.headId, messages } : null;
};

const storedStatus = (status: MessageStatus | undefined): StoredStatus => {
  if (status?.type === "complete") {
    return { type: "complete", reason: status.reason };
  }
  if (status?.type === "incomplete") {
    return typeof status.error === "string"
      ? { type: "incomplete", reason: status.reason, error: status.error }
      : { type: "incomplete", reason: status.reason };
  }
  // Still running or waiting when it was written: it was cut off.
  return { type: "incomplete", reason: "cancelled" };
};

const encodeMessage = (
  { message, parentId }: ExportedMessageRepositoryItem,
  now: number,
): StoredMessage | null => {
  if (message.role === "system") return null;
  const content: StoredPart[] = [];
  for (const part of message.content) {
    if (part.type === "text") {
      content.push({ type: "text", text: part.text });
    } else if (
      message.role === "assistant" &&
      part.type === "data" &&
      part.name === "sources"
    ) {
      const data = sourceData(part.data);
      if (data) content.push({ type: "data", name: "sources", data });
    }
  }
  const time =
    message.createdAt instanceof Date ? message.createdAt.getTime() : NaN;
  const common = {
    id: message.id,
    parentId,
    createdAt: isTime(time) ? time : now,
  };
  return message.role === "user"
    ? { ...common, role: "user", content }
    : {
        ...common,
        role: "assistant",
        content,
        status: storedStatus(message.status),
      };
};

const decodeMessage = (stored: StoredMessage): ThreadMessage => {
  const common = { id: stored.id, createdAt: new Date(stored.createdAt) };
  if (stored.role === "user") {
    return {
      ...common,
      role: "user",
      content: stored.content.flatMap((part) =>
        part.type === "text" ? [part] : [],
      ),
      attachments: [],
      metadata: { custom: {} },
    };
  }
  return {
    ...common,
    role: "assistant",
    content: stored.content,
    status: stored.status,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {},
    },
  };
};

const byRecency = (conversations: readonly StoredConversation[]) =>
  [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);

const serializeIndex = (conversations: readonly StoredConversation[]) =>
  JSON.stringify({ version: STORAGE_VERSION, conversations });

const randomId = () =>
  Date.now().toString(36) + Math.random().toString(36).slice(2, 10);

/**
 * The storage contract behind the conversation list. Reads never trust what
 * they find: corrupt JSON, an unexpected shape or another version is removed
 * and ignored. No method throws; what could not be kept is reported through
 * `available` and `isSaved`.
 */
export class ConversationStore {
  private readonly storage: Storage | null;
  private readonly now: () => number;
  private usable: boolean;
  private openId: string | undefined;
  /** Deleted or evicted in this page; a late write must not bring one back. */
  private readonly removed = new Set<string>();
  private readonly unsaved = new Set<string>();
  /** What this page knows of each conversation it wrote or opened. */
  private readonly threads = new Map<string, StoredThread>();
  private readonly listeners = new Set<() => void>();
  private readonly evictionListeners = new Set<(id: string) => void>();

  constructor(storage: Storage | null, now: () => number = Date.now) {
    this.storage = storage;
    this.now = now;
    this.usable = storage !== null;
  }

  /** False once the browser refuses to read or write site data. */
  get available(): boolean {
    return this.usable;
  }

  /** False when the last attempt to keep this conversation failed. */
  isSaved(id: string): boolean {
    return this.usable && !this.unsaved.has(id);
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Called with each conversation deleted to stay under the limits. */
  onEvict(listener: (id: string) => void): () => void {
    this.evictionListeners.add(listener);
    return () => this.evictionListeners.delete(listener);
  }

  /** The conversation on screen is never evicted to make room. */
  setOpenConversation(id: string | undefined): void {
    this.openId = id;
  }

  newId(): string {
    const taken = new Set(this.readIndex().map(({ id }) => id));
    let id = randomId();
    while (taken.has(id) || this.removed.has(id) || this.threads.has(id)) {
      id = randomId();
    }
    return id;
  }

  /** The most recently updated conversation, read from the list alone. */
  latestId(): string | undefined {
    return this.readIndex().find(({ id }) => !this.removed.has(id))?.id;
  }

  /**
   * The stored conversations, most recently updated first. Drops entries whose
   * messages are missing or unreadable, anything past the limit, and message
   * keys no entry points to.
   */
  list(): StoredConversation[] {
    const index = this.readIndex();
    const readable = index.filter(
      ({ id }) => !this.removed.has(id) && this.readThread(id) !== null,
    );
    const kept = readable.slice(0, MAX_STORED_CONVERSATIONS);
    for (const { id } of readable.slice(MAX_STORED_CONVERSATIONS)) {
      this.delete(conversationMessagesKey(id));
    }
    if (kept.length !== index.length) this.writeIndex(kept);

    const ids = new Set(kept.map(({ id }) => id));
    for (const key of this.keys()) {
      if (
        key.startsWith(MESSAGES_KEY_PREFIX) &&
        !ids.has(key.slice(MESSAGES_KEY_PREFIX.length))
      ) {
        this.delete(key);
      }
    }
    return kept;
  }

  get(id: string): StoredConversation | undefined {
    if (this.removed.has(id)) return undefined;
    const conversation = this.readIndex().find((entry) => entry.id === id);
    return conversation && this.readThread(id) ? conversation : undefined;
  }

  load(id: string): ExportedMessageRepository {
    const thread = this.removed.has(id) ? null : this.readThread(id);
    if (!thread) return { messages: [] };
    this.threads.set(id, thread);
    return {
      headId: thread.headId,
      messages: thread.messages.map((message) => ({
        parentId: message.parentId,
        message: decodeMessage(message),
      })),
    };
  }

  /**
   * Adds or replaces one message and stores the whole conversation, deleting
   * the least recently updated others while the browser reports it full.
   */
  append(id: string, item: ExportedMessageRepositoryItem): void {
    if (this.removed.has(id) || !CONVERSATION_ID.test(id)) return;
    const now = this.now();
    const message = encodeMessage(item, now);
    if (!message) return;

    const previous = this.threads.get(id) ?? this.readThread(id);
    const messages = [...(previous?.messages ?? [])];
    const at = messages.findIndex(({ id: existing }) => existing === message.id);
    if (at >= 0) messages[at] = message;
    else messages.push(message);
    const thread: StoredThread = { headId: message.id, messages };
    this.threads.set(id, thread);
    if (!this.usable) return;

    const entry = this.readIndex().find((conversation) => conversation.id === id);
    const firstQuestion = messages.find(({ role }) => role === "user");
    const conversation: StoredConversation = {
      id,
      title:
        entry?.title ||
        (firstQuestion ? conversationTitle(messageText(firstQuestion)) : ""),
      createdAt: entry?.createdAt ?? now,
      updatedAt: now,
    };
    const keep = new Set([id]);
    const saved =
      this.write(
        conversationMessagesKey(id),
        () => JSON.stringify({ version: STORAGE_VERSION, ...thread }),
        keep,
      ) && this.putEntry(conversation, keep);
    // A partial copy would come back after a reload as if it were complete.
    if (!saved) this.dropStored(id);
    this.setSaved(id, saved);
  }

  /** The stored title, or one made from the first question in `messages`. */
  titleFor(id: string, messages: readonly ThreadMessage[]): string {
    const firstQuestion = messages.find(({ role }) => role === "user");
    if (firstQuestion) return conversationTitle(messageText(firstQuestion));
    return this.readIndex().find((entry) => entry.id === id)?.title ?? "";
  }

  remove(id: string): void {
    this.forget(id);
    this.emit();
  }

  /** Deletes every key this app keeps, including ones it cannot read. */
  clear(): void {
    for (const { id } of this.readIndex()) this.removed.add(id);
    for (const id of this.threads.keys()) this.removed.add(id);
    this.threads.clear();
    this.unsaved.clear();
    for (const key of this.keys()) {
      if (key.startsWith(CONVERSATION_STORAGE_PREFIX)) this.delete(key);
    }
    this.emit();
  }

  private emit(): void {
    for (const listener of this.listeners) listener();
  }

  private setSaved(id: string, saved: boolean): void {
    if (saved === !this.unsaved.has(id)) return;
    if (saved) this.unsaved.delete(id);
    else this.unsaved.add(id);
    this.emit();
  }

  private markUnusable(): void {
    if (!this.usable) return;
    this.usable = false;
    this.emit();
  }

  private read(key: string): string | null {
    if (!this.storage || !this.usable) return null;
    try {
      return this.storage.getItem(key);
    } catch {
      this.markUnusable();
      return null;
    }
  }

  private delete(key: string): void {
    if (!this.storage || !this.usable) return;
    try {
      this.storage.removeItem(key);
    } catch {
      this.markUnusable();
    }
  }

  private keys(): string[] {
    if (!this.storage || !this.usable) return [];
    try {
      const keys: string[] = [];
      for (let index = 0; index < this.storage.length; index += 1) {
        const key = this.storage.key(index);
        if (key !== null) keys.push(key);
      }
      return keys;
    } catch {
      this.markUnusable();
      return [];
    }
  }

  /**
   * Writes `value()`, recomputed after each eviction, and while the browser
   * reports it full deletes the least recently updated conversation that is
   * neither in `keep` nor on screen.
   */
  private write(
    key: string,
    value: () => string,
    keep: ReadonlySet<string>,
  ): boolean {
    if (!this.storage || !this.usable) return false;
    for (;;) {
      try {
        this.storage.setItem(key, value());
        return true;
      } catch (error) {
        if (!isQuotaError(error)) {
          this.markUnusable();
          return false;
        }
        const victim = this.victim(this.readIndex(), keep);
        if (!victim) return false;
        this.evict(victim.id);
      }
    }
  }

  /** Shrinking writes; a failure leaves entries that the next list drops. */
  private writeIndex(conversations: readonly StoredConversation[]): void {
    if (conversations.length === 0) {
      this.delete(CONVERSATION_INDEX_KEY);
      return;
    }
    if (!this.storage || !this.usable) return;
    try {
      this.storage.setItem(CONVERSATION_INDEX_KEY, serializeIndex(conversations));
    } catch (error) {
      if (!isQuotaError(error)) this.markUnusable();
    }
  }

  private putEntry(
    conversation: StoredConversation,
    keep: ReadonlySet<string>,
  ): boolean {
    const others = () =>
      this.readIndex().filter(
        ({ id }) => id !== conversation.id && !this.removed.has(id),
      );
    for (
      let current = others();
      current.length >= MAX_STORED_CONVERSATIONS;
      current = others()
    ) {
      const victim = this.victim(current, keep);
      if (!victim) break;
      this.evict(victim.id);
    }
    return this.write(
      CONVERSATION_INDEX_KEY,
      () => serializeIndex([conversation, ...others()]),
      keep,
    );
  }

  private victim(
    index: readonly StoredConversation[],
    keep: ReadonlySet<string>,
  ): StoredConversation | undefined {
    return index.findLast(
      ({ id }) => !keep.has(id) && id !== this.openId && !this.removed.has(id),
    );
  }

  private evict(id: string): void {
    this.forget(id);
    for (const listener of this.evictionListeners) listener(id);
  }

  private forget(id: string): void {
    this.removed.add(id);
    this.threads.delete(id);
    this.unsaved.delete(id);
    this.dropStored(id);
  }

  private dropStored(id: string): void {
    this.delete(conversationMessagesKey(id));
    const index = this.readIndex();
    const rest = index.filter((conversation) => conversation.id !== id);
    if (rest.length !== index.length) this.writeIndex(rest);
  }

  private readIndex(): StoredConversation[] {
    const raw = this.read(CONVERSATION_INDEX_KEY);
    if (raw === null) return [];
    const parsed = parseJson(raw);
    if (
      !isRecord(parsed) ||
      parsed.version !== STORAGE_VERSION ||
      !Array.isArray(parsed.conversations)
    ) {
      this.delete(CONVERSATION_INDEX_KEY);
      return [];
    }
    const seen = new Set<string>();
    const conversations: StoredConversation[] = [];
    for (const value of parsed.conversations) {
      const conversation = parseConversation(value);
      if (conversation && !seen.has(conversation.id)) {
        seen.add(conversation.id);
        conversations.push(conversation);
      }
    }
    if (conversations.length !== parsed.conversations.length) {
      this.writeIndex(conversations);
    }
    return byRecency(conversations);
  }

  private readThread(id: string): StoredThread | null {
    const key = conversationMessagesKey(id);
    const raw = this.read(key);
    if (raw === null) return null;
    const thread = parseThread(parseJson(raw));
    if (!thread) this.delete(key);
    return thread;
  }
}
