import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { DocumentList } from "./Documents";
import {
  documentFailureMessage,
  documentRequestErrorMessage,
  documentStatusLabel,
  MAX_UPLOAD_SIZE_BYTES,
  shouldPollDocuments,
  validateUploadFile,
} from "./DocumentModel";
import type { DocumentRow } from "./DocumentModel";

const NOW = Date.parse("2026-07-12T08:00:00.000Z");
const RAW_SECRET = "provider response: Authorization Bearer sk-secret-123";

const documents: DocumentRow[] = [
  {
    id: "native-pdf",
    filename: "原生規範.pdf",
    status: "ready",
    processingStage: "ready",
    summary: null,
    tags: ["原生 PDF", "品質規範"],
    processingError: null,
    createdAt: "2026-07-12T07:55:00.000Z",
    pageCount: 5,
    unitCount: 18,
  },
  {
    id: "scan-pdf",
    filename: "掃描手冊.pdf",
    status: "processing",
    processingStage: "extracting",
    summary: null,
    tags: ["掃描 PDF"],
    processingError: null,
    createdAt: "2026-07-12T07:59:30.000Z",
    pageCount: 8,
    unitCount: 0,
  },
  {
    id: "slides",
    filename: "新人訓練.pptx",
    status: "processing",
    processingStage: "queued",
    summary: "簡報涵蓋新人報到與安全訓練。",
    tags: ["簡報", "新人訓練"],
    processingError: null,
    createdAt: "2026-07-12T07:58:00.000Z",
    pageCount: 12,
    unitCount: 24,
  },
  {
    id: "markdown",
    filename: "維運筆記.md",
    status: "failed",
    processingStage: "failed",
    summary: null,
    tags: null,
    processingError: { code: "provider_failure", message: RAW_SECRET },
    error: RAW_SECRET,
    createdAt: "2026-07-11T08:00:00.000Z",
    pageCount: 0,
    unitCount: 6,
  },
];

describe("DocumentList", () => {
  it("renders native PDF, scanned PDF, PPTX and Markdown states with separate counts", () => {
    const html = renderToStaticMarkup(
      <DocumentList documents={documents} now={NOW} />,
    );
    const snapshotMarkup = html.replace(/<svg\b[\s\S]*?<\/svg>/g, "<svg />");

    expect(snapshotMarkup).toMatchSnapshot();
    expect(html).toContain("排隊中");
    expect(html).toContain("處理中 · 擷取內容");
    expect(html).toContain("完成");
    expect(html).toContain("失敗");
    expect(html).toContain("尚無摘要");
    expect(html).toContain("頁數 5");
    expect(html).toContain("檢索單元 18");
    expect(html).toContain("原生 PDF");
    expect(html).toContain("新人訓練");
    expect(html).not.toContain("段落");
    expect(html).not.toContain(RAW_SECRET);
  });

  it("uses explicit extracting and indexing copy", () => {
    expect(documentStatusLabel(documents[1]!)).toBe("處理中 · 擷取內容");
    expect(
      documentStatusLabel({
        ...documents[1]!,
        processingStage: "indexing",
      }),
    ).toBe("處理中 · 建立索引");
  });

  it("keeps polling for every processing stage and stops for terminal rows", () => {
    expect(shouldPollDocuments([documents[0]!, documents[3]!])).toBe(false);
    expect(shouldPollDocuments([documents[2]!])).toBe(true);
    expect(shouldPollDocuments([documents[1]!])).toBe(true);
    expect(
      shouldPollDocuments([
        { ...documents[1]!, processingStage: "ready" },
      ]),
    ).toBe(true);
  });

  it("never exposes canonical or legacy raw failure messages", () => {
    const legacyOnly: DocumentRow = {
      ...documents[3]!,
      id: "legacy-error",
      processingError: null,
      error: RAW_SECRET,
    };
    const rendered = renderToStaticMarkup(
      <DocumentList documents={[legacyOnly]} now={NOW} />,
    );

    expect(documentFailureMessage(documents[3]!)).not.toContain(RAW_SECRET);
    expect(documentFailureMessage(legacyOnly)).not.toContain(RAW_SECRET);
    expect(rendered).not.toContain(RAW_SECRET);
    expect(documentRequestErrorMessage("upload", 500)).not.toContain(
      RAW_SECRET,
    );
  });

  it("retains the supported upload formats and the 50 MiB boundary", () => {
    expect(
      validateUploadFile({ name: "manual.PDF", size: MAX_UPLOAD_SIZE_BYTES }),
    ).toBeNull();
    expect(
      validateUploadFile({
        name: "manual.pdf",
        size: MAX_UPLOAD_SIZE_BYTES + 1,
      }),
    ).toContain("50 MiB");
    expect(validateUploadFile({ name: "script.exe", size: 10 })).toContain(
      "不支援",
    );
  });
});
