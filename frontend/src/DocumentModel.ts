export type DocumentProcessingError = {
  code: string;
  message: string;
};

export type DocumentRow = {
  id: string;
  filename: string;
  status: "processing" | "ready" | "failed";
  processingStage: "queued" | "extracting" | "indexing" | "ready" | "failed";
  summary: string | null;
  tags: string[] | null;
  processingError?: DocumentProcessingError | null;
  /** Transitional compatibility only. The message is never rendered. */
  error?: DocumentProcessingError | string | null;
  createdAt: string;
  pageCount: number;
  unitCount: number;
};

type DocumentAction = "list" | "upload" | "delete";

export const SUPPORTED_UPLOAD_EXTENSIONS = [
  ".pdf",
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".docx",
  ".pptx",
  ".xlsx",
  ".md",
  ".markdown",
  ".txt",
  ".stp",
  ".step",
  ".eml",
  ".msg",
] as const;

export const UPLOAD_ACCEPT = SUPPORTED_UPLOAD_EXTENSIONS.join(",");

const isProcessingError = (
  value: DocumentRow["processingError"] | DocumentRow["error"],
): value is DocumentProcessingError =>
  typeof value === "object" &&
  value !== null &&
  typeof value.code === "string" &&
  typeof value.message === "string";

const processingErrorFor = (
  doc: DocumentRow,
): DocumentProcessingError | null => {
  if (isProcessingError(doc.processingError)) return doc.processingError;
  return isProcessingError(doc.error) ? doc.error : null;
};

export function safeDocumentFailureMessage(
  error: DocumentProcessingError | null,
): string {
  switch (error?.code) {
    case "unsupported_file_type":
    case "unsupported_format":
      return "不支援這種檔案格式，請改用上方列出的格式。";
    case "file_too_large":
      return "檔案超過 50 MiB 上限，請壓縮或拆分後再上傳。";
    case "encrypted_document":
    case "password_protected":
      return "文件有密碼保護，請解除後再上傳。";
    case "extraction_failed":
    case "extract_failed":
    case "artifact_processing_failed":
      return "讀不到文件內容，請確認檔案能正常開啟後再上傳。";
    case "indexing_failed":
    case "index_failed":
      return "建立索引失敗，請稍後重新上傳。";
    default:
      return "文件處理失敗，請刪除後重新上傳；若一直失敗，請聯絡管理員。";
  }
}

export const documentFailureMessage = (doc: DocumentRow): string =>
  safeDocumentFailureMessage(processingErrorFor(doc));

export function documentRequestErrorMessage(
  action: DocumentAction,
  status?: number,
): string {
  if (status === undefined) {
    return "連不上伺服器，請確認網路後再試。";
  }

  if (status === 401 || status === 403) {
    if (action === "list") return "目前沒有查看文件的權限。";
    if (action === "upload") return "目前沒有上傳文件的權限。";
    return "目前沒有刪除文件的權限。";
  }

  if (action === "upload") {
    if (status === 413) return "檔案太大，伺服器無法處理。";
    if (status === 400 || status === 415) {
      return "檔案格式不受支援或內容無法解析，請確認檔案後再試。";
    }
    if (status === 409) return "這份文件已經在知識庫裡，或正在處理中。";
    if (status === 429) return "上傳過於頻繁，請稍後再試。";
    return "上傳失敗，伺服器暫時無法處理檔案，請稍後再試。";
  }

  if (action === "delete") {
    if (status === 404) return "這份文件已經不在了，請重新載入清單。";
    if (status === 409) return "文件正在使用中，請稍後再刪除。";
    return "刪除失敗，伺服器暫時無法處理，請稍後再試。";
  }

  if (status === 429) return "重新整理太頻繁，請稍後再試。";
  return "文件清單載入失敗，請稍後再試。";
}

export function validateUploadFile(
  file: Pick<File, "name" | "size">,
): string | null {
  const lowerName = file.name.toLocaleLowerCase();
  const hasSupportedExtension = SUPPORTED_UPLOAD_EXTENSIONS.some((extension) =>
    lowerName.endsWith(extension),
  );

  if (!hasSupportedExtension) {
    return "不支援這種檔案格式，請選擇 PDF、Office、圖片、STP 或 Email 檔案。";
  }
  return null;
}

export function shouldPollDocuments(documents: readonly DocumentRow[]): boolean {
  return documents.some((doc) => doc.status === "processing");
}

export function documentStatusLabel(doc: DocumentRow): string {
  if (doc.status === "ready") return "可查詢";
  if (doc.status === "failed") return "失敗";

  switch (doc.processingStage) {
    case "queued":
      return "排隊中";
    case "extracting":
      return "讀取內容中";
    case "indexing":
      return "建立索引中";
    case "ready":
      return "即將完成";
    case "failed":
      return "處理中";
  }
}

export const documentVisualStatus = (
  doc: DocumentRow,
): "queued" | "processing" | "ready" | "failed" =>
  doc.status === "processing" && doc.processingStage === "queued"
    ? "queued"
    : doc.status;

export const documentSummary = (doc: DocumentRow): string => {
  if (doc.summary?.trim()) return doc.summary;
  if (doc.status === "ready") return "尚無摘要";
  if (doc.status === "failed") return "處理失敗，沒有摘要。";

  switch (doc.processingStage) {
    case "queued":
      return "排隊等待處理…";
    case "extracting":
      return "正在讀取文件內容…";
    case "indexing":
      return "正在建立索引…";
    case "ready":
      return "就快完成了…";
    case "failed":
      return "正在更新處理狀態…";
  }
};
