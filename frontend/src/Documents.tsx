import { useCallback, useEffect, useRef, useState } from "react";
import { UploadIcon, DocsIcon, TrashIcon } from "./Icons";
import {
  documentFailureMessage,
  documentRequestErrorMessage,
  documentStatusLabel,
  documentSummary,
  documentVisualStatus,
  shouldPollDocuments,
  UPLOAD_ACCEPT,
  validateUploadFile,
} from "./DocumentModel";
import type { DocumentRow } from "./DocumentModel";

function fileExt(name: string): string {
  const dot = name.lastIndexOf(".");
  if (dot < 0 || dot === name.length - 1) return "檔";
  return name.slice(dot + 1).toUpperCase().slice(0, 4);
}

function relativeTime(iso: string, now: number): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diff = Math.max(0, now - then);
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

const StatusBadge = ({ doc }: { doc: DocumentRow }) => (
  <span className={`status status-${documentVisualStatus(doc)}`}>
    <span className="status-dot" />
    {documentStatusLabel(doc)}
  </span>
);

type DocumentListProps = {
  documents: readonly DocumentRow[];
  onDelete?: (doc: DocumentRow) => void;
  now?: number;
};

export const DocumentList = ({
  documents,
  onDelete,
  now = Date.now(),
}: DocumentListProps) => (
  <div className="doc-list">
    {documents.map((doc) => {
      const created = relativeTime(doc.createdAt, now);
      return (
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
              <StatusBadge doc={doc} />
            </div>
            <p
              className={
                "doc-summary" + (doc.summary?.trim() ? "" : " is-empty")
              }
            >
              {documentSummary(doc)}
            </p>
            <div className="doc-meta">
              {(doc.tags ?? []).length > 0 && (
                <span className="doc-tags">
                  {(doc.tags ?? []).map((tag, index) => (
                    <span key={`${tag}-${index}`} className="chip">
                      {tag}
                    </span>
                  ))}
                </span>
              )}
              <span className="doc-metrics">
                <span className="doc-metric">頁數 {doc.pageCount}</span>
                <span aria-hidden="true"> · </span>
                <span className="doc-metric">檢索單元 {doc.unitCount}</span>
                {created && (
                  <>
                    <span aria-hidden="true"> · </span>
                    <span className="doc-metric">{created}</span>
                  </>
                )}
              </span>
            </div>
            {doc.status === "failed" && (
              <p className="doc-error">{documentFailureMessage(doc)}</p>
            )}
          </div>
          <button
            type="button"
            className="doc-delete"
            aria-label={`刪除 ${doc.filename}`}
            title="刪除文件"
            onClick={() => onDelete?.(doc)}
          >
            <TrashIcon />
          </button>
        </div>
      );
    })}
  </div>
);

const DocumentListError = ({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) => (
  <div className="docs-request-error" role="alert">
    <span>{message}</span>
    <button type="button" onClick={onRetry}>
      重新載入
    </button>
  </div>
);

export const Documents = () => {
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDocs = useCallback(async () => {
    try {
      const res = await fetch("/api/documents");
      if (!res.ok) {
        setListError(documentRequestErrorMessage("list", res.status));
        return undefined;
      }
      const data: unknown = await res.json();
      if (!Array.isArray(data)) {
        setListError(documentRequestErrorMessage("list", 502));
        return undefined;
      }
      const documents = data as DocumentRow[];
      setDocs(documents);
      setListError(null);
      return documents;
    } catch {
      setListError(documentRequestErrorMessage("list"));
      return undefined;
    }
  }, []);

  useEffect(() => {
    void fetchDocs().finally(() => setLoaded(true));
  }, [fetchDocs]);

  const polling = shouldPollDocuments(docs);
  useEffect(() => {
    if (!polling) return undefined;
    const pollId = window.setInterval(() => {
      void fetchDocs();
    }, 3000);
    return () => window.clearInterval(pollId);
  }, [fetchDocs, polling]);

  const uploadFile = async (file: File) => {
    const validationError = validateUploadFile(file);
    if (validationError) {
      window.alert(validationError);
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/documents", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        window.alert(documentRequestErrorMessage("upload", res.status));
        return;
      }
      if (fileInputRef.current) fileInputRef.current.value = "";
      await fetchDocs();
    } catch {
      window.alert(documentRequestErrorMessage("upload"));
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (doc: DocumentRow) => {
    if (!window.confirm(`確定要刪除「${doc.filename}」嗎？`)) return;
    try {
      const res = await fetch(`/api/documents/${doc.id}`, { method: "DELETE" });
      if (!res.ok) {
        window.alert(documentRequestErrorMessage("delete", res.status));
        if (res.status === 404) await fetchDocs();
        return;
      }
      await fetchDocs();
    } catch {
      window.alert(documentRequestErrorMessage("delete"));
    }
  };

  const openPicker = () => {
    if (!uploading) fileInputRef.current?.click();
  };

  const readyCount = docs.filter((doc) => doc.status === "ready").length;

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
          tabIndex={uploading ? -1 : 0}
          aria-disabled={uploading}
          aria-busy={uploading}
          onClick={openPicker}
          onKeyDown={(event) => {
            if (!uploading && (event.key === "Enter" || event.key === " ")) {
              event.preventDefault();
              openPicker();
            }
          }}
          onDragOver={(event) => {
            event.preventDefault();
            if (!uploading) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const file = event.dataTransfer.files?.[0];
            if (file && !uploading) void uploadFile(file);
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={UPLOAD_ACCEPT}
            disabled={uploading}
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void uploadFile(file);
            }}
          />
          <span className="dropzone-icon">
            {uploading ? <span className="spinner spinner-lg" /> : <UploadIcon />}
          </span>
          <div className="dropzone-text">
            <strong>
              {uploading ? "上傳與解析中…" : "拖曳檔案到這裡，或點擊選擇檔案"}
            </strong>
            <span>
              支援 PDF、DOCX、PPTX、XLSX、PNG/JPEG/WebP、TXT/MD（單檔 ≤ 50 MiB）
            </span>
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
          listError ? (
            <DocumentListError
              message={listError}
              onRetry={() => void fetchDocs()}
            />
          ) : (
            <div className="docs-empty">
              <span className="docs-empty-icon">
                <DocsIcon />
              </span>
              <h3>尚未建立任何文件</h3>
              <p>
                上傳手冊、規範、簡報或報表，系統會自動解析內容並建立可檢索的知識庫。
              </p>
            </div>
          )
        ) : (
          <>
            {listError && (
              <DocumentListError
                message={listError}
                onRetry={() => void fetchDocs()}
              />
            )}
            <DocumentList documents={docs} onDelete={handleDelete} />
          </>
        )}
      </div>
    </div>
  );
};
