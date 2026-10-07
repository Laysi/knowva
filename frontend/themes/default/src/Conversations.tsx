import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  ThreadListItemByIndexProvider,
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  useAuiState,
} from "@assistant-ui/react";
import type { ConversationActionResult, ConversationHistory } from "@knowva/core";
import { PlusIcon, TrashIcon } from "./Icons";
import { exactTime, relativeTime } from "./relativeTime";

const UNTITLED = "未命名對話";
/** How often the relative times move on. */
const CLOCK_TICK_MS = 60_000;

const useNow = () => {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), CLOCK_TICK_MS);
    return () => window.clearInterval(timer);
  }, []);
  return now;
};

const ConversationItem = ({
  updatedAt,
  now,
  onOpen,
  onDelete,
}: {
  updatedAt: Date | undefined;
  now: Date;
  onOpen: () => void;
  onDelete: (id: string) => void;
}) => {
  const id = useAuiState((s) => s.threadListItem.id);
  const title = useAuiState((s) => s.threadListItem.title) || UNTITLED;

  return (
    <ThreadListItemPrimitive.Root className="history-item">
      <ThreadListItemPrimitive.Trigger
        className="history-open"
        title={title}
        onClick={onOpen}
      >
        <span className="history-title">{title}</span>
        {updatedAt && (
          <time
            className="history-when"
            dateTime={updatedAt.toISOString()}
            title={exactTime(updatedAt)}
          >
            {relativeTime(updatedAt, now)}
          </time>
        )}
      </ThreadListItemPrimitive.Trigger>
      <button
        type="button"
        className="history-delete"
        aria-label={`刪除對話「${title}」`}
        title="刪除"
        onClick={() => onDelete(id)}
      >
        <TrashIcon />
      </button>
    </ThreadListItemPrimitive.Root>
  );
};

/** Asked in the page, not through window.confirm. */
const ClearAll = ({
  count,
  onClear,
}: {
  count: number;
  onClear: () => void;
}) => {
  const [confirming, setConfirming] = useState(false);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const promptId = useId();

  useEffect(() => {
    if (confirming) cancelRef.current?.focus();
  }, [confirming]);

  if (!confirming) {
    return (
      <button
        type="button"
        className="history-clear"
        onClick={() => setConfirming(true)}
      >
        清除全部對話
      </button>
    );
  }

  return (
    <div className="history-confirm" role="group" aria-labelledby={promptId}>
      <p id={promptId}>確定清除全部 {count} 串對話？清除後無法復原。</p>
      <div className="history-confirm-actions">
        <button
          ref={cancelRef}
          type="button"
          className="history-confirm-cancel"
          onClick={() => setConfirming(false)}
        >
          取消
        </button>
        <button
          type="button"
          className="history-confirm-clear"
          onClick={() => {
            setConfirming(false);
            onClear();
          }}
        >
          確定清除
        </button>
      </div>
    </div>
  );
};

/**
 * 近期對話 in the navigation rail, most recently updated first. `onOpen` runs
 * when a conversation is opened or started, so the page can show the chat.
 */
export const Conversations = ({
  history,
  onOpen,
}: {
  history: ConversationHistory;
  onOpen: () => void;
}) => {
  const [actionError, setActionError] = useState<string | null>(null);
  const headingId = useId();
  const now = useNow();
  const threadIds = useAuiState((s) => s.threads.threadIds);
  const threadItems = useAuiState((s) => s.threads.threadItems);

  // assistant-ui keeps the order it loaded; a conversation asked in moves up.
  // One the server has no time for yet stays on top, in assistant-ui's order.
  const order = useMemo(() => {
    const remoteIds = new Map(threadItems.map((item) => [item.id, item.remoteId]));
    return threadIds
      .map((id, index) => {
        const remoteId = remoteIds.get(id);
        const updatedAt =
          remoteId === undefined ? undefined : history.updatedAt.get(remoteId);
        return { id, index, updatedAt };
      })
      .sort(
        (a, b) =>
          (b.updatedAt?.getTime() ?? Infinity) -
            (a.updatedAt?.getTime() ?? Infinity) || a.index - b.index,
      );
  }, [threadIds, threadItems, history.updatedAt]);

  const report = (result: ConversationActionResult) =>
    setActionError(result.ok ? null : result.message);
  const remove = (id: string) => {
    setActionError(null);
    void history.remove(id).then(report);
  };
  const clearAll = () => {
    setActionError(null);
    void history.clearAll().then(report);
  };

  const listError = history.status === "error" ? history.error : null;
  const count = order.length;

  return (
    <section className="history" aria-labelledby={headingId}>
      <div className="history-head">
        <h2 id={headingId} className="history-heading">
          近期對話
        </h2>
        <ThreadListPrimitive.New className="history-new" onClick={onOpen}>
          <PlusIcon />
          <span>新對話</span>
        </ThreadListPrimitive.New>
      </div>

      {listError && (
        <div className="history-alert" role="alert">
          <span>{listError}</span>
          <button type="button" onClick={history.reload}>
            重新載入
          </button>
        </div>
      )}
      {actionError && (
        <p className="history-alert" role="alert">
          {actionError}
        </p>
      )}

      {count === 0 && history.status === "loading" && (
        <p className="history-loading">正在載入對話紀錄…</p>
      )}
      {count > 0 && (
        <>
          <ThreadListPrimitive.Root className="history-list">
            {order.map(({ id, index, updatedAt }) => (
              <ThreadListItemByIndexProvider key={id} index={index} archived={false}>
                <ConversationItem
                  updatedAt={updatedAt}
                  now={now}
                  onOpen={onOpen}
                  onDelete={remove}
                />
              </ThreadListItemByIndexProvider>
            ))}
          </ThreadListPrimitive.Root>
          <ClearAll count={count} onClear={clearAll} />
        </>
      )}
    </section>
  );
};
