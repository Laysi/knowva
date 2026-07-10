import { useCallback, useEffect, useRef, useState } from "react";
import { UploadIcon, DocsIcon, TrashIcon } from "./Icons";

type DocumentRow = {
  id: string;
  filename: string;
  status: "processing" | "ready" | "failed";
  summary: string | null;
  tags: string[] | null;
  error: string | null;
  created_at: string;
  chunk_count: number;
};

const statusLabel: Record<DocumentRow["status"], string> = {
  processing: "處理中",
  ready: "完成",
  failed: "失敗",
};

function fileExt(name: string): string {
  const dot = name.lastIndexOf(".");
  if (dot < 0 || dot === name.length - 1) return "檔";
  return name.slice(dot + 1).toUpperCase().slice(0, 4);
}

function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diff = Math.max(0, Date.now() - then);
  const min = 60_000;
  const hr = 3_600_000;
  const day = 86_400_000;
  if (diff < min) return "剛剛";
  if (diff < hr) return `${Math.floor(diff / min)} 分鐘前`;
  if (diff < day) return `${Math.floor(diff / hr)} 小時前`;
  if (diff < 7 * day) return `${Math.floor(diff / day)} 天前`;
  return new Date(iso).toLocaleDateString("zh-TW", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
}

const StatusBadge = ({ status }: { status: DocumentRow["status"] }) => (
  <span className={`status status-${status}`}>
    <span className="status-dot" />
    {statusLabel[status]}
  </span>
);

export const Documents = () => {
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<number | null>(null);

  const fetchDocs = useCallback(async () => {
    const res = await fetch("/api/documents");
    if (!res.ok) return;
    const data: DocumentRow[] = await res.json();
    setDocs(data);
    return data;
  }, []);

  useEffect(() => {
    fetchDocs().finally(() => setLoaded(true));
  }, [fetchDocs]);

  useEffect(() => {
    const hasProcessing = docs.some((d) => d.status === "processing");
    if (hasProcessing && pollRef.current === null) {
      pollRef.current = window.setInterval(() => {
        fetchDocs();
      }, 3000);
    } else if (!hasProcessing && pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [docs, fetchDocs]);

  const uploadFile = async (file: File) => {
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/documents", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        alert(`上傳失敗:${res.status}`);
        return;
      }
      if (fileInputRef.current) fileInputRef.current.value = "";
      await fetchDocs();
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (doc: DocumentRow) => {
    if (!confirm(`確定要刪除「${doc.filename}」嗎?`)) return;
    const res = await fetch(`/api/documents/${doc.id}`, { method: "DELETE" });
    if (!res.ok && res.status !== 204) {
      alert(`刪除失敗:${res.status}`);
      return;
    }
    await fetchDocs();
  };

  const openPicker = () => fileInputRef.current?.click();

  const readyCount = docs.filter((d) => d.status === "ready").length;

  return (
    <div className="documents">
      <div className="documents-inner">
        <div
          className={
            "dropzone" +
            (dragging ? " dragover" : "") +
            (uploading ? " uploading" : "")
          }
          role="button"
          tabIndex={0}
          onClick={openPicker}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              openPicker();
            }
          }}
          onDragOver={(e) => {
            e.preventDefault();
            if (!uploading) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files?.[0];
            if (file && !uploading) uploadFile(file);
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) uploadFile(file);
            }}
          />
          <span className="dropzone-icon">
            {uploading ? <span className="spinner spinner-lg" /> : <UploadIcon />}
          </span>
          <div className="dropzone-text">
            <strong>{uploading ? "上傳與解析中…" : "拖曳檔案到這裡,或點擊選擇檔案"}</strong>
            <span>支援 PDF、Word、Excel、CSV、HTML、圖片、TXT/MD(單檔 ≤ 4MB;PPT 請先匯出 PDF)</span>
          </div>
        </div>

        <div className="library-head">
          <h2>知識庫文件</h2>
          {loaded && docs.length > 0 && (
            <span className="library-count">
              共 {docs.length} 份 · {readyCount} 份可查詢
            </span>
          )}
        </div>

        {!loaded ? (
          <div className="docs-loading">
            <span className="spinner" />
            載入文件中…
          </div>
        ) : docs.length === 0 ? (
          <div className="docs-empty">
            <span className="docs-empty-icon">
              <DocsIcon />
            </span>
            <h3>尚未建立任何文件</h3>
            <p>
              上傳手冊、規範、簡報或報表,系統會自動解析內容並建立可檢索的知識庫。
            </p>
          </div>
        ) : (
          <div className="doc-list">
            {docs.map((doc) => (
              <div
                key={doc.id}
                className={"doc-row" + (doc.status === "failed" ? " is-failed" : "")}
                tabIndex={doc.status === "failed" ? 0 : undefined}
              >
                <span className="doc-glyph">{fileExt(doc.filename)}</span>
                <div className="doc-main">
                  <div className="doc-line1">
                    <span className="doc-name" title={doc.filename}>
                      {doc.filename}
                    </span>
                    <StatusBadge status={doc.status} />
                  </div>
                  <p
                    className={
                      "doc-summary" + (doc.summary ? "" : " is-empty")
                    }
                  >
                    {doc.summary ??
                      (doc.status === "processing"
                        ? "解析中,摘要即將產生…"
                        : doc.status === "failed"
                          ? "處理失敗,無法產生摘要。"
                          : "尚無摘要")}
                  </p>
                  <div className="doc-meta">
                    {(doc.tags ?? []).length > 0 && (
                      <span className="doc-tags">
                        {(doc.tags ?? []).map((tag) => (
                          <span key={tag} className="chip">
                            {tag}
                          </span>
                        ))}
                      </span>
                    )}
                    <span className="doc-metrics">
                      段落 {doc.chunk_count} · {relativeTime(doc.created_at)}
                    </span>
                  </div>
                  {doc.status === "failed" && doc.error && (
                    <p className="doc-error" title={doc.error}>
                      {doc.error}
                    </p>
                  )}
                </div>
                <button
                  type="button"
                  className="doc-delete"
                  aria-label={`刪除 ${doc.filename}`}
                  title="刪除文件"
                  onClick={() => handleDelete(doc)}
                >
                  <TrashIcon />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
