import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SourceList } from "./Thread";
import type { ChatSource } from "./BackendAdapter";

const sources: ChatSource[] = [
  {
    sourceId: "manual-page-2",
    documentId: "manual-a",
    unitId: 102,
    unitIndex: 1,
    filename: "操作手冊.pdf",
    pageNumber: 2,
    kind: "page",
  },
  {
    sourceId: "manual-page-5",
    documentId: "manual-a",
    unitId: 105,
    unitIndex: 4,
    filename: "操作手冊.pdf",
    pageNumber: 5,
    kind: "page",
  },
  {
    sourceId: "markdown-unit-0",
    documentId: "notes",
    unitId: 201,
    unitIndex: 0,
    filename: "維運筆記.md",
    pageNumber: null,
    kind: "text",
  },
  {
    sourceId: "markdown-unit-missing",
    documentId: "notes-copy",
    unitId: 301,
    filename: "維運筆記.md",
    pageNumber: null,
    kind: "text",
  },
];

describe("SourceList", () => {
  it("renders every source in server order without page deduplication", () => {
    const html = renderToStaticMarkup(<SourceList data={sources} />);

    expect(html).toMatchSnapshot();
    expect(html).toContain("操作手冊.pdf · 第 2 頁");
    expect(html).toContain("操作手冊.pdf · 第 5 頁");
    expect(html.indexOf("第 2 頁")).toBeLessThan(html.indexOf("第 5 頁"));
    expect(html.match(/操作手冊\.pdf/g)).toHaveLength(2);
  });

  it("uses text units for Markdown and never renders a null page number", () => {
    const html = renderToStaticMarkup(<SourceList data={sources} />);

    expect(html).toContain("維運筆記.md · 第 1 段");
    expect(html).toContain("維運筆記.md · 文字來源");
    expect(html).not.toContain("第 null 頁");
    expect(html).not.toContain("null");
  });
});
