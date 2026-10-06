import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";
import { useLocalRuntime, useRemoteThreadListRuntime } from "@assistant-ui/react";
import type { AssistantRuntime } from "@assistant-ui/react";
import { BackendAdapter } from "./BackendAdapter";
import { createConversationListAdapter } from "./conversationAdapter";
import {
  browserConversationStorage,
  ConversationStore,
} from "./conversationStore";

export type ConversationHistory = {
  /**
   * False while the open conversation is not being kept in this browser. Show
   * `CONVERSATION_NOT_SAVED_MESSAGE`; answers keep working.
   */
  saved: boolean;
  /** Deletes every conversation and its storage. Confirm in the page first. */
  clearAll: () => Promise<void>;
};

export type KnowvaRuntime = {
  /** Give it to `AssistantRuntimeProvider`. */
  runtime: AssistantRuntime;
  history: ConversationHistory;
};

// Every conversation still runs the local runtime against /api/chat.
const useBackendThreadRuntime = () => useLocalRuntime(BackendAdapter);

const openConversationId = (runtime: AssistantRuntime): string | undefined => {
  const { mainThreadId, threadItems } = runtime.threads.getState();
  return threadItems[mainThreadId]?.remoteId;
};

/**
 * The chat runtime with conversation history kept in this browser's
 * localStorage: up to 50 conversations, titled from the first question, and
 * the most recently updated one reopened on load.
 */
export function useKnowvaRuntime(): KnowvaRuntime {
  const [store] = useState(
    () => new ConversationStore(browserConversationStorage()),
  );
  const [adapter] = useState(() => createConversationListAdapter(store));
  const [latestId] = useState(() => store.latestId());
  const runtime = useRemoteThreadListRuntime({
    runtimeHook: useBackendThreadRuntime,
    adapter,
    // Read once: a reload lands on the conversation the person was in.
    threadId: latestId,
  });

  useEffect(() => {
    const sync = () => store.setOpenConversation(openConversationId(runtime));
    sync();
    return runtime.threads.subscribe(sync);
  }, [runtime, store]);

  useEffect(
    () =>
      store.onEvict((id) => {
        try {
          runtime.threads
            .getItemById(id)
            .delete()
            .catch(() => undefined);
        } catch {
          // Not in the list this page shows.
        }
      }),
    [runtime, store],
  );

  const subscribe = useCallback(
    (onChange: () => void) => {
      const unsubscribeStore = store.subscribe(onChange);
      const unsubscribeThreads = runtime.threads.subscribe(onChange);
      return () => {
        unsubscribeStore();
        unsubscribeThreads();
      };
    },
    [runtime, store],
  );
  const saved = useSyncExternalStore(subscribe, () => {
    const id = openConversationId(runtime);
    return store.available && (id === undefined || store.isSaved(id));
  });

  const clearAll = useCallback(async () => {
    const { threadIds, archivedThreadIds } = runtime.threads.getState();
    for (const id of [...threadIds, ...archivedThreadIds]) {
      try {
        await runtime.threads.getItemById(id).delete();
      } catch {
        // Already gone.
      }
    }
    store.clear();
  }, [runtime, store]);

  return useMemo(
    () => ({ runtime, history: { saved, clearAll } }),
    [runtime, saved, clearAll],
  );
}
