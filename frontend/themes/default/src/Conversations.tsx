import { useEffect, useId, useRef, useState } from "react";
import {
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  useAuiState,
} from "@assistant-ui/react";
import { MAX_CONVERSATIONS } from "@knowva/core";
import type { ConversationActionResult, ConversationHistory } from "@knowva/core";
import { ChevronDownIcon, HistoryIcon, PlusIcon, TrashIcon } from "./Icons";

const UNTITLED = "未命名對話";

const ConversationItem = ({
  onOpen,
  onDelete,
}: {
  onOpen: () => void;
  onDelete: (id: string) => void;
}) => {
  const id = useAuiState((s) => s.threadListItem.id);
  const title = useAuiState((s) => s.threadListItem.title) || UNTITLED;

  return (
    <ThreadListItemPrimitive.Root className="conversation">
      <ThreadListItemPrimitive.Trigger
        className="conversation-open"
        title={title}
        onClick={onOpen}
      >
        <span className="conversation-title">{title}</span>
      </ThreadListItemPrimitive.Trigger>
      <button
        type="button"
        className="conversation-delete"
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
        className="conversations-clear"
        onClick={() => setConfirming(true)}
      >
        清除全部對話
      </button>
    );
  }

  return (
    <div
      className="conversations-confirm"
      role="group"
      aria-labelledby={promptId}
    >
      <p id={promptId}>確定清除全部 {count} 串對話？清除後無法復原。</p>
      <div className="conversations-confirm-actions">
        <button
          ref={cancelRef}
          type="button"
          className="conversations-confirm-cancel"
          onClick={() => setConfirming(false)}
        >
          取消
        </button>
        <button
          type="button"
          className="conversations-confirm-clear"
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

export const Conversations = ({ history }: { history: ConversationHistory }) => {
  // Only narrow layouts fold the list away; wide ones always show it.
  const [open, setOpen] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const panelId = useId();
  const count = useAuiState((s) => s.threads.threadIds.length);
  const close = () => setOpen(false);

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

  return (
    <aside
      className={"conversations" + (open ? " is-open" : "")}
      aria-label="對話紀錄"
      onKeyDown={(event) => {
        if (event.key === "Escape") close();
      }}
    >
      <div className="conversations-bar">
        <button
          type="button"
          className="conversations-toggle"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={() => setOpen((value) => !value)}
        >
          <HistoryIcon className="conversations-toggle-icon" />
          <span className="conversations-toggle-label">對話紀錄</span>
          {count > 0 && <span className="conversations-count">{count}</span>}
          <ChevronDownIcon className="conversations-chevron" />
        </button>
        <ThreadListPrimitive.New className="conversation-new" onClick={close}>
          <PlusIcon />
          <span>新對話</span>
        </ThreadListPrimitive.New>
      </div>

      <div id={panelId} className="conversations-panel">
        <h2 className="conversations-heading">對話紀錄</h2>
        {listError && (
          <div className="conversations-alert" role="alert">
            <span>{listError}</span>
            <button type="button" onClick={history.reload}>
              重新載入
            </button>
          </div>
        )}
        {actionError && (
          <p className="conversations-alert" role="alert">
            {actionError}
          </p>
        )}
        {count === 0 ? (
          history.status === "loading" ? (
            <p className="conversations-empty">正在載入對話紀錄…</p>
          ) : (
            !listError && (
              <p className="conversations-empty">
                問過的問題會保存在伺服器上，以這個瀏覽器區分，最多{" "}
                {MAX_CONVERSATIONS} 串。
              </p>
            )
          )
        ) : (
          <>
            <ThreadListPrimitive.Root className="conversation-list">
              <ThreadListPrimitive.Items>
                {() => <ConversationItem onOpen={close} onDelete={remove} />}
              </ThreadListPrimitive.Items>
            </ThreadListPrimitive.Root>
            <ClearAll count={count} onClear={clearAll} />
          </>
        )}
      </div>
    </aside>
  );
};
