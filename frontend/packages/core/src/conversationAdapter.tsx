import { useMemo } from "react";
import type { PropsWithChildren } from "react";
import { RuntimeAdapterProvider, useAui } from "@assistant-ui/react";
import type {
  AssistantClient,
  RemoteThreadListAdapter,
  ThreadHistoryAdapter,
} from "@assistant-ui/react";
import { createAssistantStream } from "assistant-stream";
import type { ConversationStore, StoredConversation } from "./conversationStore";

type ConversationMetadata = Awaited<ReturnType<RemoteThreadListAdapter["fetch"]>>;

const metadata = (conversation: StoredConversation): ConversationMetadata => ({
  status: "regular",
  remoteId: conversation.id,
  title: conversation.title || undefined,
  lastMessageAt: new Date(conversation.updatedAt),
});

// Renaming and archiving are not offered; rejecting keeps the list honest.
const unsupported = (action: string) => () =>
  Promise.reject(new Error(`Conversations cannot be ${action}.`));

/** Reads and writes the messages of the conversation it is mounted for. */
const conversationHistory = (
  store: ConversationStore,
  aui: AssistantClient,
): ThreadHistoryAdapter => ({
  async load() {
    const { remoteId } = aui.threadListItem().getState();
    return remoteId === undefined ? { messages: [] } : store.load(remoteId);
  },
  async append(item) {
    // The runtime does not await every append and awaits the last one inside
    // its own error handling, so this must never reject.
    try {
      const { remoteId } = await aui.threadListItem().initialize();
      store.append(remoteId, item);
      // Show the title now rather than when the first answer ends.
      if (item.parentId === null && item.message.role === "user") {
        aui.threadListItem().generateTitle();
      }
    } catch {
      // The store reports anything it could not keep.
    }
  },
});

/**
 * The conversation list for `useRemoteThreadListRuntime`, kept in `store`. Each
 * conversation's runtime gets a history adapter for its own messages, with
 * every message part, `sources` included, so a follow-up after a reload sends
 * the same `previousSources` as one without.
 */
export function createConversationListAdapter(
  store: ConversationStore,
): RemoteThreadListAdapter {
  function ConversationHistoryProvider({ children }: PropsWithChildren) {
    const aui = useAui();
    const adapters = useMemo(
      () => ({ history: conversationHistory(store, aui) }),
      [aui],
    );
    return (
      <RuntimeAdapterProvider adapters={adapters}>{children}</RuntimeAdapterProvider>
    );
  }

  return {
    unstable_Provider: ConversationHistoryProvider,
    async list() {
      return { threads: store.list().map(metadata) };
    },
    async fetch(id) {
      const conversation = store.get(id);
      if (!conversation) throw new Error("Conversation not found.");
      return metadata(conversation);
    },
    async initialize() {
      return { remoteId: store.newId(), externalId: undefined };
    },
    async delete(id) {
      store.remove(id);
    },
    async generateTitle(id, messages) {
      const title = store.titleFor(id, messages);
      return createAssistantStream((controller) => {
        controller.appendText(title);
      });
    },
    rename: unsupported("renamed"),
    archive: unsupported("archived"),
    unarchive: unsupported("unarchived"),
  };
}
