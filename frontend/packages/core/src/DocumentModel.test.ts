import { describe, expect, it } from "vitest";
import {
  documentStatusLabel,
  MAX_UPLOAD_BYTES,
  shouldPollDocuments,
  validateUploadFile,
} from "./DocumentModel";
import type { DocumentRow } from "./DocumentModel";

const row = (
  status: DocumentRow["status"],
  processingStage: DocumentRow["processingStage"],
): DocumentRow => ({
  id: `${status}-${processingStage}`,
  filename: "手冊.pdf",
  status,
  processingStage,
  summary: null,
  tags: null,
  processingError: null,
  createdAt: "2026-07-12T07:55:00.000Z",
  pageCount: 1,
  unitCount: 1,
});

describe("DocumentModel", () => {
  it("uses explicit extracting and indexing copy", () => {
    expect(documentStatusLabel(row("processing", "extracting"))).toBe(
      "讀取內容中",
    );
    expect(documentStatusLabel(row("processing", "indexing"))).toBe(
      "建立索引中",
    );
  });

  it("keeps polling for every processing stage and stops for terminal rows", () => {
    expect(
      shouldPollDocuments([row("ready", "ready"), row("failed", "failed")]),
    ).toBe(false);
    expect(shouldPollDocuments([row("processing", "queued")])).toBe(true);
    expect(shouldPollDocuments([row("processing", "extracting")])).toBe(true);
    expect(shouldPollDocuments([row("processing", "ready")])).toBe(true);
  });

  it("retains the supported upload formats and the server's 50 MiB limit", () => {
    expect(MAX_UPLOAD_BYTES).toBe(50 * 1024 * 1024);
    expect(
      validateUploadFile({ name: "manual.PDF", size: MAX_UPLOAD_BYTES }),
    ).toBeNull();
    expect(
      validateUploadFile({ name: "manual.pdf", size: MAX_UPLOAD_BYTES + 1 }),
    ).toBe("檔案超過 50 MiB 上限，請壓縮或拆分後再上傳。");
    expect(validateUploadFile({ name: "script.exe", size: 10 })).toContain(
      "不支援",
    );
    expect(validateUploadFile({ name: "sensor.stp", size: 10 })).toBeNull();
    expect(validateUploadFile({ name: "inspection.mp4", size: 10 })).toContain(
      "不支援",
    );
    expect(validateUploadFile({ name: "alert.eml", size: 10 })).toBeNull();
  });
});
