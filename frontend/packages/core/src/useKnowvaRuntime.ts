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
import { ConversationSync } from "./conversationAdapter";
import type { ConversationListStatus } from "./conversationAdapter";
import {
  clearConversations,
  ConversationRequestError,
  conversationRequestErrorMessage,
} from "./conversationApi";
import type { ConversationAction } from "./conversationApi";

export type ConversationActionResult =
  | { ok: true }
  | { ok: false; message: string };

export type ConversationHistory = {
  /** Where loading the conversation list stands. */
  status: ConversationListStatus;
  /**
   * Safe text to show while `status` is `"error"`, with a way to `reload`.
   * Chat keeps working; a new conversation is still saved.
   */
  error: string | null;
  /**
   * One line about the open conversation, or null: it is not being saved
   * (`CONVERSATION_NOT_SAVED_MESSAGE`) or its messages did not load
   * (`CONVERSATION_LOAD_FAILED_MESSAGE`). Answers keep working.
   */
  notice: string | null;
  /**
   * When each conversation last changed on the server, by server id (a list
   * item's `remoteId`), kept current as messages are saved. A conversation the
   * server never created has none.
   */
  updatedAt: ReadonlyMap<string, Date>;
  /** Loads the list again. */
  reload: () => void;
  /**
   * Deletes one conversation, by its list item id or server id. Deleting one
   * that is already gone succeeds.
   */
  remove: (id: string) => Promise<ConversationActionResult>;
  /** Deletes every conversation of this browser. Confirm in the page first. */
  clearAll: () => Promise<ConversationActionResult>;
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

const failure = (
  action: ConversationAction,
  error: unknown,
): ConversationActionResult => ({
  ok: false,
  message: conversationRequestErrorMessage(
    action,
    error instanceof ConversationRequestError ? error.status : 500,
  ),
});

/**
 * The chat runtime with conversation history kept on the server for this
 * browser: up to 50 conversations, titled by the server from the first
 * question, and the most recently updated one reopened on load.
 */
export function useKnowvaRuntime(): KnowvaRuntime {
  const [sync] = useState(() => new ConversationSync());
  const [adapter] = useState(() => sync.listAdapter());
  const runtime = useRemoteThreadListRuntime({
    runtimeHook: useBackendThreadRuntime,
    adapter,
  });

  // A reload lands on the conversation the person was last in, unless they
  // already opened one or started typing.
  useEffect(() => {
    let settled = false;
    const openLatest = () => {
      if (settled) return;
      const threads = runtime.threads.getState();
      if (threads.isLoading || sync.listStatus !== "ready") return;
      settled = true;
      const latest = threads.threadIds[0];
      const untouched =
        threads.mainThreadId === threads.newThreadId &&
        runtime.thread.getState().messages.length === 0 &&
        runtime.thread.composer.getState().text === "";
      if (latest !== undefined && untouched) {
        runtime.threads.switchToThread(latest).catch(() => undefined);
      }
    };
    openLatest();
    const unsubscribeThreads = runtime.threads.subscribe(openLatest);
    const unsubscribeSync = sync.subscribe(openLatest);
    return () => {
      unsubscribeThreads();
      unsubscribeSync();
    };
  }, [runtime, sync]);

  // Starting a 51st conversation deletes the oldest on the server; drop it here.
  useEffect(
    () =>
      sync.onRemoved((ids) => {
        for (const id of ids) {
          try {
            runtime.threads
              .getItemById(id)
              .delete()
              .catch(() => undefined);
          } catch {
            // Not in the list this page shows.
          }
        }
      }),
    [runtime, sync],
  );

  const subscribe = useCallback(
    (onChange: () => void) => {
      const unsubscribeSync = sync.subscribe(onChange);
      const unsubscribeThreads = runtime.threads.subscribe(onChange);
      return () => {
        unsubscribeSync();
        unsubscribeThreads();
      };
    },
    [runtime, sync],
  );
  const status = useSyncExternalStore(subscribe, () => sync.listStatus);
  const error = useSyncExternalStore(subscribe, () => sync.listError);
  const notice = useSyncExternalStore(subscribe, () =>
    sync.notice(openConversationId(runtime)),
  );
  const updatedAt = useSyncExternalStore(subscribe, () => sync.updatedTimes);

  const reload = useCallback(() => {
    runtime.threads.reload().catch(() => undefined);
  }, [runtime]);

  const remove = useCallback(
    async (id: string): Promise<ConversationActionResult> => {
      try {
        await runtime.threads.getItemById(id).delete();
        return { ok: true };
      } catch (error) {
        return failure("delete", error);
      }
    },
    [runtime],
  );

  const clearAll = useCallback(async (): Promise<ConversationActionResult> => {
    try {
      await clearConversations();
    } catch (error) {
      return failure("clear", error);
    }
    const { threadIds, archivedThreadIds } = runtime.threads.getState();
    for (const id of [...threadIds, ...archivedThreadIds]) {
      try {
        const item = runtime.threads.getItemById(id);
        const { remoteId } = item.getState();
        // The server already deleted it; take it off the list without asking again.
        if (remoteId !== undefined) sync.markGone(remoteId);
        await item.delete();
      } catch {
        // Already gone.
      }
    }
    return { ok: true };
  }, [runtime, sync]);

  return useMemo(
    () => ({
      runtime,
      history: { status, error, notice, updatedAt, reload, remove, clearAll },
    }),
    [runtime, status, error, notice, updatedAt, reload, remove, clearAll],
  );
}
