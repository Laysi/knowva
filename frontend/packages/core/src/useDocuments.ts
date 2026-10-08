import { useCallback, useEffect, useState } from "react";
import {
  documentRequestErrorMessage,
  shouldPollDocuments,
  validateUploadFile,
} from "./DocumentModel";
import type { DocumentRow } from "./DocumentModel";

export const DOCUMENT_POLL_INTERVAL_MS = 3000;

/**
 * How an upload ended. `rejected` never reached the network: the file failed
 * the client-side check. Every message is already safe to show.
 */
export type UploadResult =
  | { status: "uploaded" }
  | { status: "rejected"; message: string }
  | { status: "failed"; message: string };

export type DeleteResult = { ok: true } | { ok: false; message: string };

export type DocumentLibrary = {
  /** The last list the server returned; kept when a later refresh fails. */
  documents: DocumentRow[];
  /** False until the first list request has settled either way. */
  loaded: boolean;
  /** Safe message for the latest failed list request, cleared on success. */
  listError: string | null;
  uploading: boolean;
  refresh: () => Promise<void>;
  upload: (file: File) => Promise<UploadResult>;
  remove: (doc: DocumentRow) => Promise<DeleteResult>;
};

/**
 * The document library's data flow: the list, polling while anything is
 * processing, upload and delete. Themes decide how results and messages look;
 * asking before a delete is theirs as well.
 */
export function useDocuments(): DocumentLibrary {
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/documents");
      if (!res.ok) {
        setListError(documentRequestErrorMessage("list", res.status));
        return;
      }
      const data: unknown = await res.json();
      if (!Array.isArray(data)) {
        setListError(documentRequestErrorMessage("list", 502));
        return;
      }
      setDocuments(data as DocumentRow[]);
      setListError(null);
    } catch {
      setListError(documentRequestErrorMessage("list"));
    }
  }, []);

  useEffect(() => {
    void refresh().finally(() => setLoaded(true));
  }, [refresh]);

  const polling = shouldPollDocuments(documents);
  useEffect(() => {
    if (!polling) return undefined;
    const pollId = window.setInterval(() => {
      void refresh();
    }, DOCUMENT_POLL_INTERVAL_MS);
    return () => window.clearInterval(pollId);
  }, [refresh, polling]);

  const upload = useCallback(
    async (file: File): Promise<UploadResult> => {
      const validationError = validateUploadFile(file);
      if (validationError) {
        return { status: "rejected", message: validationError };
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
          return {
            status: "failed",
            message: documentRequestErrorMessage("upload", res.status),
          };
        }
        await refresh();
        return { status: "uploaded" };
      } catch {
        return {
          status: "failed",
          message: documentRequestErrorMessage("upload"),
        };
      } finally {
        setUploading(false);
      }
    },
    [refresh],
  );

  const remove = useCallback(
    async (doc: DocumentRow): Promise<DeleteResult> => {
      try {
        const res = await fetch(`/api/documents/${doc.id}`, {
          method: "DELETE",
        });
        if (!res.ok) {
          // The row is already gone, so the list is stale. The refresh is not
          // awaited: the message comes first and the list catches up behind it.
          if (res.status === 404) void refresh();
          return {
            ok: false,
            message: documentRequestErrorMessage("delete", res.status),
          };
        }
        await refresh();
        return { ok: true };
      } catch {
        return { ok: false, message: documentRequestErrorMessage("delete") };
      }
    },
    [refresh],
  );

  return { documents, loaded, listError, uploading, refresh, upload, remove };
}
