import { useMemo } from "react";
import type { PropsWithChildren } from "react";
import { RuntimeAdapterProvider, useAui } from "@assistant-ui/react";
import type {
  ExportedMessageRepository,
  ExportedMessageRepositoryItem,
  RemoteThreadListAdapter,
  ThreadHistoryAdapter,
} from "@assistant-ui/react";
import { createAssistantStream } from "assistant-stream";
import {
  CONVERSATION_LOAD_FAILED_MESSAGE,
  CONVERSATION_NOT_SAVED_MESSAGE,
  ConversationRequestError,
  conversationRequestErrorMessage,
  conversationTitle,
  createConversation,
  deleteConversation,
  fetchConversation,
  firstQuestion,
  isTransientFailure,
  listConversations,
  loadMessages,
  saveMessage,
} from "./conversationApi";
import type { ConversationSummary } from "./conversationApi";

export type ConversationListStatus = "loading" | "ready" | "error";

type ConversationMetadata = Awaited<ReturnType<RemoteThreadListAdapter["fetch"]>>;

/** What a conversation's history adapter needs from its list item. */
export type ConversationListItem = {
  remoteId: () => string | undefined;
  /** Resolves with the server id, creating the conversation the first time. */
  initialize: () => Promise<{ remoteId: string }>;
  /**
   * Asks the list adapter for the title again. assistant-ui types it as void
   * but hands back the promise, which is awaited so a failure stays caught.
   */
  generateTitle: () => unknown;
};

/**
 * A conversation the server never created. Its id never leaves the page and
 * nothing of it is saved.
 */
const LOCAL_ID_PREFIX = "local:";
let localIds = 0;
const localId = () =>
  `${LOCAL_ID_PREFIX}${Date.now().toString(36)}-${(localIds += 1)}`;
export const isLocalConversationId = (id: string): boolean =>
  id.startsWith(LOCAL_ID_PREFIX);

const metadata = (conversation: ConversationSummary): ConversationMetadata => ({
  status: "regular",
  remoteId: conversation.id,
  title: conversation.title ?? undefined,
  lastMessageAt: new Date(conversation.updatedAt),
});

// Renaming and archiving are not offered; rejecting keeps the list honest.
const unsupported = (action: string) => () =>
  Promise.reject(new Error(`Conversations cannot be ${action}.`));

type QueuedMessage = { item: ExportedMessageRepositoryItem; saved: boolean };

/**
 * What this page knows of the server's conversations, shared by the list
 * adapter and the history adapter of every conversation. A failure never
 * rejects into assistant-ui: it becomes `listStatus` and `listError`, or a
 * conversation's `notice`, and chat keeps working.
 */
export class ConversationSync {
  private status: ConversationListStatus = "loading";
  private error: string | null = null;
  private readonly titles = new Map<string, string>();
  /** Replaced, never changed in place, so a snapshot of it stays stable. */
  private times: ReadonlyMap<string, Date> = new Map();
  /** Deleted, evicted or cleared: nothing is sent for it again. */
  private readonly gone = new Set<string>();
  /** Saving stopped for good in this page, with the line to show. */
  private readonly stopped = new Map<string, string>();
  /** A save failed for now; it goes again before the next message. */
  private readonly retrying = new Set<string>();
  private readonly queues = new Map<string, QueuedMessage[]>();
  /** Each conversation's saves run one at a time, in the order they came. */
  private readonly chains = new Map<string, Promise<void>>();
  private readonly listeners = new Set<() => void>();
  private readonly removedListeners = new Set<(ids: readonly string[]) => void>();

  get listStatus(): ConversationListStatus {
    return this.status;
  }

  /** Safe text for the last failed list load, otherwise null. */
  get listError(): string | null {
    return this.error;
  }

  /**
   * When each conversation last changed on the server, by server id: as the
   * list said, then as each save answered. A new map on every change.
   */
  get updatedTimes(): ReadonlyMap<string, Date> {
    return this.times;
  }

  /** The line to show for the conversation with this server id, or null. */
  notice(remoteId: string | undefined): string | null {
    if (remoteId === undefined) return null;
    const stopped = this.stopped.get(remoteId);
    if (stopped !== undefined) return stopped;
    return this.retrying.has(remoteId) ? CONVERSATION_NOT_SAVED_MESSAGE : null;
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Called with the server ids of conversations deleted to stay at the limit. */
  onRemoved(listener: (ids: readonly string[]) => void): () => void {
    this.removedListeners.add(listener);
    return () => this.removedListeners.delete(listener);
  }

  /** Forgets a conversation the server no longer has. */
  markGone(remoteId: string): void {
    this.gone.add(remoteId);
    this.titles.delete(remoteId);
    this.queues.delete(remoteId);
    const wasStopped = this.stopped.delete(remoteId);
    const wasRetrying = this.retrying.delete(remoteId);
    const hadTime = this.times.has(remoteId);
    if (hadTime) {
      const times = new Map(this.times);
      times.delete(remoteId);
      this.times = times;
    }
    if (wasStopped || wasRetrying || hadTime) this.emit();
  }

  /** Resolves once every save asked for so far in this conversation has finished. */
  settled(remoteId: string): Promise<void> {
    return this.chains.get(remoteId) ?? Promise.resolve();
  }

  /**
   * Queues one message behind the earlier ones of its conversation. Resolves
   * true once it is saved and false otherwise; never rejects.
   */
  save(remoteId: string, item: ExportedMessageRepositoryItem): Promise<boolean> {
    if (this.gone.has(remoteId) || this.stopped.has(remoteId)) {
      return Promise.resolve(false);
    }
    const queued: QueuedMessage = { item, saved: false };
    const queue = this.queues.get(remoteId) ?? [];
    queue.push(queued);
    this.queues.set(remoteId, queue);
    const run = this.settled(remoteId)
      .then(() => this.flush(remoteId))
      .catch(() => undefined);
    this.chains.set(remoteId, run);
    return run.then(() => queued.saved);
  }

  /** Stops saving a conversation for the rest of this page's life. */
  stop(remoteId: string, notice: string): void {
    this.queues.delete(remoteId);
    this.retrying.delete(remoteId);
    this.stopped.set(remoteId, notice);
    this.emit();
  }

  private async flush(remoteId: string): Promise<void> {
    const queue = this.queues.get(remoteId) ?? [];
    while (queue.length > 0) {
      if (this.gone.has(remoteId) || this.stopped.has(remoteId)) return;
      const next = queue[0]!;
      try {
        const conversation = await saveMessage(remoteId, next.item);
        if (conversation.title) this.titles.set(remoteId, conversation.title);
        this.touch([conversation]);
      } catch (error) {
        if (isTransientFailure(error)) {
          // Kept, in order, for the next message of this conversation.
          this.setRetrying(remoteId, true);
        } else {
          // Refused for good: over a limit, gone, or out of order.
          this.stop(remoteId, CONVERSATION_NOT_SAVED_MESSAGE);
        }
        return;
      }
      queue.shift();
      next.saved = true;
    }
    this.setRetrying(remoteId, false);
  }

  private setRetrying(remoteId: string, retrying: boolean): void {
    if (this.retrying.has(remoteId) === retrying) return;
    if (retrying) this.retrying.add(remoteId);
    else this.retrying.delete(remoteId);
    this.emit();
  }

  private setList(status: ConversationListStatus, error: string | null): void {
    if (this.status === status && this.error === error) return;
    this.status = status;
    this.error = error;
    this.emit();
  }

  /** Keeps the later of the known and the given time of each conversation. */
  private touch(conversations: readonly ConversationSummary[]): void {
    let times: Map<string, Date> | null = null;
    for (const { id, updatedAt } of conversations) {
      if (this.gone.has(id)) continue;
      const time = new Date(updatedAt);
      const known = (times ?? this.times).get(id);
      if (known !== undefined && known.getTime() >= time.getTime()) continue;
      times ??= new Map(this.times);
      times.set(id, time);
    }
    if (times === null) return;
    this.times = times;
    this.emit();
  }

  private emit(): void {
    for (const listener of this.listeners) listener();
  }

  /** The conversation list for `useRemoteThreadListRuntime`. */
  listAdapter(): RemoteThreadListAdapter {
    // Each conversation's runtime gets a history adapter bound to its list item.
    const ConversationHistoryProvider = ({ children }: PropsWithChildren) => {
      const aui = useAui();
      const adapters = useMemo(
        () => ({
          history: this.historyAdapter({
            remoteId: () => aui.threadListItem().getState().remoteId,
            initialize: () => aui.threadListItem().initialize(),
            generateTitle: () => aui.threadListItem().generateTitle(),
          }),
        }),
        [aui],
      );
      return (
        <RuntimeAdapterProvider adapters={adapters}>{children}</RuntimeAdapterProvider>
      );
    };

    return {
      unstable_Provider: ConversationHistoryProvider,
      list: async () => {
        this.setList("loading", null);
        try {
          const conversations = await listConversations();
          for (const { id, title } of conversations) {
            if (title) this.titles.set(id, title);
          }
          this.touch(conversations);
          this.setList("ready", null);
          return {
            threads: conversations
              .filter(({ id }) => !this.gone.has(id))
              .map(metadata),
          };
        } catch (error) {
          const status =
            error instanceof ConversationRequestError ? error.status : undefined;
          this.setList("error", conversationRequestErrorMessage("list", status));
          // assistant-ui keeps the list it already had.
          throw error;
        }
      },
      fetch: async (remoteId) => {
        if (this.gone.has(remoteId) || isLocalConversationId(remoteId)) {
          throw new ConversationRequestError(404);
        }
        const conversation = await fetchConversation(remoteId);
        this.touch([conversation]);
        return metadata(conversation);
      },
      initialize: async () => {
        try {
          const { conversation, removedIds } = await createConversation();
          this.touch([conversation]);
          if (removedIds.length > 0) {
            for (const id of removedIds) this.markGone(id);
            for (const listener of this.removedListeners) listener(removedIds);
          }
          return { remoteId: conversation.id, externalId: undefined };
        } catch {
          // Sending waits on this, so it never fails: answer, and say it is not kept.
          const remoteId = localId();
          this.stop(remoteId, CONVERSATION_NOT_SAVED_MESSAGE);
          return { remoteId, externalId: undefined };
        }
      },
      delete: async (remoteId) => {
        if (!this.gone.has(remoteId) && !isLocalConversationId(remoteId)) {
          try {
            await deleteConversation(remoteId);
          } catch (error) {
            // Already gone is what was asked for.
            const missing =
              error instanceof ConversationRequestError && error.status === 404;
            if (!missing) throw error;
          }
        }
        this.markGone(remoteId);
      },
      generateTitle: async (remoteId, messages) => {
        // The server titles a conversation as its first question is saved.
        await this.settled(remoteId);
        const title =
          this.titles.get(remoteId) ?? conversationTitle(firstQuestion(messages));
        // An empty stream would clear the title, so there is always a part.
        return createAssistantStream((controller) => {
          controller.appendText(title);
        });
      },
      rename: unsupported("renamed"),
      archive: unsupported("archived"),
      unarchive: unsupported("unarchived"),
    };
  }

  /**
   * Reads and saves the messages of one conversation, every part included, so
   * a follow-up after a reload sends the same `previousSources` as one without.
   */
  historyAdapter(listItem: ConversationListItem): ThreadHistoryAdapter {
    return {
      load: async (): Promise<ExportedMessageRepository> => {
        const remoteId = listItem.remoteId();
        if (
          remoteId === undefined ||
          isLocalConversationId(remoteId) ||
          this.gone.has(remoteId)
        ) {
          return { messages: [] };
        }
        try {
          return await loadMessages(remoteId);
        } catch {
          // Saving on top of a conversation that did not load would split it.
          this.stop(remoteId, CONVERSATION_LOAD_FAILED_MESSAGE);
          return { messages: [] };
        }
      },
      // The runtime does not await every append and awaits the last one inside
      // its own error handling, so this never rejects.
      append: async (item) => {
        try {
          const { remoteId } = await listItem.initialize();
          const saved = await this.save(remoteId, item);
          // Show the server's title now rather than when the first answer ends.
          if (saved && item.parentId === null && item.message.role === "user") {
            await listItem.generateTitle();
          }
        } catch {
          // Reported through `notice`.
        }
      },
    };
  }
}
