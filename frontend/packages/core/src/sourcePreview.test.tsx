// @vitest-environment jsdom
import { renderHook, waitFor } from "@testing-library/react";
import { renderToStaticMarkup } from "react-dom/server";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatSource } from "./BackendAdapter";
import {
  documentFileUrl,
  remarkSourceMarkers,
  sourcePreviewErrorMessage,
  useSourcePreview,
} from "./sourcePreview";

const PAGE_SOURCE: ChatSource = {
  sourceId: "S1",
  documentId: "manual a",
  unitId: 137,
  unitIndex: 36,
  filename: "設備保養手冊.pdf",
  pageNumber: 37,
  kind: "page",
};

const TEXT_SOURCE: ChatSource = {
  sourceId: "S2",
  documentId: "notes",
  unitId: 202,
  unitIndex: 1,
  filename: "維運交接筆記.md",
  pageNumber: null,
  kind: "text",
};

const IMAGE_ROUTE = "GET /api/documents/manual%20a/units/137/image";
const TEXT_ROUTE = "GET /api/documents/notes/units/202/text";

const markdown = (text: string) =>
  renderToStaticMarkup(
    <Markdown remarkPlugins={[remarkGfm, remarkSourceMarkers]}>{text}</Markdown>,
  );

const marker = (id: string) =>
  `<button type="button" data-source-id="${id}">[${id}]</button>`;

describe("remarkSourceMarkers", () => {
  it("turns each citation into an element naming its source, side by side included", () => {
    expect(markdown("保養週期為六個月[S1][S2]，以手冊為準 [S10]。")).toBe(
      `<p>保養週期為六個月${marker("S1")}${marker("S2")}，以手冊為準 ${marker("S10")}。</p>`,
    );
  });

  it("marks citations inside emphasis, lists and tables", () => {
    const html = markdown(
      "**重點 [S3]**\n\n- 項目 [S4]\n\n| 欄位 |\n| --- |\n| 值 [S5] |",
    );

    expect(html).toContain(`<strong>重點 ${marker("S3")}</strong>`);
    expect(html).toContain(`<li>項目 ${marker("S4")}</li>`);
    expect(html).toContain(`<td>值 ${marker("S5")}</td>`);
  });

  it("leaves code, link text and anything not shaped like a citation alone", () => {
    const html = markdown(
      "`[S1]`\n\n```\n[S2]\n```\n\n[看 [S3]](https://example.com) [S0] [S01] [s4] [S 5]",
    );

    expect(html).not.toContain("<button");
    expect(html).toContain("<code>[S1]</code>");
    expect(html).toContain("[S2]");
    expect(html).toContain('<a href="https://example.com">看 [S3]</a>');
    expect(html).toContain("[S0] [S01] [s4] [S 5]");
  });
});

describe("useSourcePreview", () => {
  let createObjectURL: ReturnType<typeof vi.fn>;
  let revokeObjectURL: ReturnType<typeof vi.fn>;

  /** Answers each `METHOD url` from `routes`; anything else is a 404. */
  const stubFetch = (routes: Record<string, () => Response | Promise<Response>>) => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      const route = routes[`${init?.method ?? "GET"} ${url}`];
      return route ? route() : new Response(null, { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  };

  const png = () =>
    new Response(new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" }), {
      headers: { "Content-Type": "image/png" },
    });

  let restoreUrl: (() => void)[] = [];

  /** Replaces a static `URL` method for one test; jsdom may not have it. */
  const stubUrl = (name: "createObjectURL" | "revokeObjectURL", value: unknown) => {
    const original = Object.getOwnPropertyDescriptor(URL, name);
    Object.defineProperty(URL, name, { configurable: true, writable: true, value });
    restoreUrl.push(() => {
      if (original) Object.defineProperty(URL, name, original);
      else delete (URL as Partial<typeof URL>)[name];
    });
  };

  beforeEach(() => {
    let created = 0;
    createObjectURL = vi.fn(() => `blob:preview/${(created += 1)}`);
    revokeObjectURL = vi.fn();
    stubUrl("createObjectURL", createObjectURL);
    stubUrl("revokeObjectURL", revokeObjectURL);
  });

  afterEach(() => {
    restoreUrl.forEach((restore) => restore());
    restoreUrl = [];
    vi.unstubAllGlobals();
  });

  it("shows a page source's image and releases it when closed", async () => {
    const fetchMock = stubFetch({ [IMAGE_ROUTE]: png });
    const { result, unmount } = renderHook(() => useSourcePreview(PAGE_SOURCE));

    expect(result.current).toEqual({ status: "loading" });
    await waitFor(() =>
      expect(result.current).toEqual({ status: "image", url: "blob:preview/1" }),
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(createObjectURL.mock.calls[0]![0]).toBeInstanceOf(Blob);

    unmount();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:preview/1");
  });

  it("loads a source again when it is reopened, never showing the released image", async () => {
    let answers = 0;
    const fetchMock = stubFetch({
      [IMAGE_ROUTE]: () =>
        (answers += 1) === 1 ? png() : new Promise<Response>(() => undefined),
    });
    const { result, rerender } = renderHook(
      ({ source }: { source: ChatSource | null }) => useSourcePreview(source),
      { initialProps: { source: PAGE_SOURCE as ChatSource | null } },
    );
    await waitFor(() =>
      expect(result.current).toEqual({ status: "image", url: "blob:preview/1" }),
    );

    rerender({ source: null });
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:preview/1");
    rerender({ source: PAGE_SOURCE });

    expect(result.current).toEqual({ status: "loading" });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(result.current).toEqual({ status: "loading" });
  });

  it("shows a text source's passage", async () => {
    stubFetch({
      [TEXT_ROUTE]: () =>
        new Response("交接時確認備品庫存。", {
          headers: { "Content-Type": "text/plain; charset=utf-8" },
        }),
    });
    const { result } = renderHook(() => useSourcePreview(TEXT_SOURCE));

    await waitFor(() =>
      expect(result.current).toEqual({ status: "text", text: "交接時確認備品庫存。" }),
    );
    expect(createObjectURL).not.toHaveBeenCalled();
  });

  it("shows a catalog source's document card, not its first unit", async () => {
    const fetchMock = stubFetch({
      "GET /api/documents/notes/units/202/catalog": () =>
        new Response("文件名稱：維運交接筆記.md\n摘要：交接流程", {
          headers: { "Content-Type": "text/plain; charset=utf-8" },
        }),
    });
    const { result } = renderHook(() =>
      useSourcePreview({ ...TEXT_SOURCE, unitIndex: 0, catalog: true }),
    );

    await waitFor(() =>
      expect(result.current).toEqual({
        status: "text",
        text: "文件名稱：維運交接筆記.md\n摘要：交接流程",
      }),
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/documents/notes/units/202/catalog");
  });

  it("ends every failure in a safe message", async () => {
    const cases: [Record<string, () => Response | Promise<Response>>, string][] = [
      [{}, sourcePreviewErrorMessage(404)],
      [
        { [IMAGE_ROUTE]: () => new Response("raw stack trace", { status: 500 }) },
        sourcePreviewErrorMessage(500),
      ],
      // A server that answers unknown paths with the app shell.
      [
        {
          [IMAGE_ROUTE]: () =>
            new Response("<!doctype html>", { headers: { "Content-Type": "text/html" } }),
        },
        sourcePreviewErrorMessage(502),
      ],
      [
        {
          [IMAGE_ROUTE]: () => Promise.reject(new TypeError("Failed to fetch")),
        },
        sourcePreviewErrorMessage(),
      ],
    ];
    for (const [routes, message] of cases) {
      stubFetch(routes);
      const { result, unmount } = renderHook(() => useSourcePreview(PAGE_SOURCE));
      await waitFor(() => expect(result.current).toEqual({ status: "error", message }));
      expect(JSON.stringify(result.current)).not.toContain("raw stack trace");
      unmount();
    }
    expect(createObjectURL).not.toHaveBeenCalled();
  });

  it("never shows a preview that loaded for the previous source", async () => {
    let answerImage: () => void = () => undefined;
    const fetchMock = stubFetch({
      [IMAGE_ROUTE]: () =>
        new Promise<Response>((resolve) => {
          answerImage = () => resolve(png());
        }),
      [TEXT_ROUTE]: () =>
        new Response("第二個來源", { headers: { "Content-Type": "text/plain" } }),
    });
    const { result, rerender } = renderHook(
      ({ source }: { source: ChatSource | null }) => useSourcePreview(source),
      { initialProps: { source: PAGE_SOURCE as ChatSource | null } },
    );
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const imageSignal = fetchMock.mock.calls[0]![1]!.signal!;

    rerender({ source: TEXT_SOURCE });
    expect(imageSignal.aborted).toBe(true);
    await waitFor(() =>
      expect(result.current).toEqual({ status: "text", text: "第二個來源" }),
    );
    answerImage();
    await Promise.resolve();
    expect(result.current).toEqual({ status: "text", text: "第二個來源" });
    expect(createObjectURL).not.toHaveBeenCalled();

    rerender({ source: null });
    expect(result.current).toEqual({ status: "loading" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("documentFileUrl", () => {
  it("names the document's original file", () => {
    expect(documentFileUrl("4f1c")).toBe("/api/documents/4f1c/file");
    expect(documentFileUrl("a/../b")).toBe("/api/documents/a%2F..%2Fb/file");
  });
});
