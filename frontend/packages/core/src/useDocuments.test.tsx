// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DOCUMENT_POLL_INTERVAL_MS, useDocuments } from "./useDocuments";
import type { DocumentRow } from "./DocumentModel";

const RAW_SECRET = "provider response: Authorization Bearer sk-secret-321";

const row = (overrides: Partial<DocumentRow> = {}): DocumentRow => ({
  id: "manual",
  filename: "手冊.pdf",
  status: "ready",
  processingStage: "ready",
  summary: null,
  tags: null,
  processingError: null,
  createdAt: "2026-07-12T07:55:00.000Z",
  pageCount: 3,
  unitCount: 3,
  ...overrides,
});

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

type Route = (init?: RequestInit) => Response | Promise<Response>;

/** Answers each `METHOD url` from its own queue; the last answer repeats. */
const stubBackend = (routes: Record<string, Route[]>) => {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${url}`;
    const queue = routes[key];
    if (!queue?.length) throw new Error(`unexpected request: ${key}`);
    const route = queue.length > 1 ? queue.shift()! : queue[0]!;
    return route(init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

const requests = (fetchMock: ReturnType<typeof stubBackend>) =>
  fetchMock.mock.calls.map(
    ([url, init]) => `${init?.method ?? "GET"} ${String(url)}`,
  );

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useDocuments", () => {
  it("loads the list once and reports a safe list error without dropping rows", async () => {
    stubBackend({
      "GET /api/documents": [
        () => json([row()]),
        () => new Response(RAW_SECRET, { status: 500 }),
      ],
    });

    const { result } = renderHook(() => useDocuments());
    expect(result.current.loaded).toBe(false);
    await waitFor(() => expect(result.current.loaded).toBe(true));
    expect(result.current.documents).toEqual([row()]);
    expect(result.current.listError).toBeNull();

    await act(() => result.current.refresh());
    expect(result.current.documents).toEqual([row()]);
    expect(result.current.listError).toBe("文件清單載入失敗，請稍後再試。");
  });

  it("treats a non-array list as a server failure", async () => {
    stubBackend({ "GET /api/documents": [() => json({ rows: [] })] });

    const { result } = renderHook(() => useDocuments());
    await waitFor(() => expect(result.current.loaded).toBe(true));
    expect(result.current.documents).toEqual([]);
    expect(result.current.listError).toBe("文件清單載入失敗，請稍後再試。");
  });

  it("polls while a document is processing and stops once every row settles", async () => {
    vi.useFakeTimers();
    const fetchMock = stubBackend({
      "GET /api/documents": [
        () => json([row({ status: "processing", processingStage: "indexing" })]),
        () => json([row()]),
      ],
    });

    const { result } = renderHook(() => useDocuments());
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(result.current.documents[0]?.status).toBe("processing");
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(() => vi.advanceTimersByTimeAsync(DOCUMENT_POLL_INTERVAL_MS));
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.current.documents[0]?.status).toBe("ready");

    await act(() => vi.advanceTimersByTimeAsync(DOCUMENT_POLL_INTERVAL_MS * 3));
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("rejects an unsupported file without sending it", async () => {
    const fetchMock = stubBackend({ "GET /api/documents": [() => json([])] });
    const { result } = renderHook(() => useDocuments());
    await waitFor(() => expect(result.current.loaded).toBe(true));

    const outcome = await act(() =>
      result.current.upload(new File(["MZ"], "setup.exe")),
    );

    expect(outcome).toEqual({
      status: "rejected",
      message: "不支援這種檔案格式，請選擇 PDF、Office、圖片、STP 或 Email 檔案。",
    });
    expect(requests(fetchMock)).toEqual(["GET /api/documents"]);
    expect(result.current.uploading).toBe(false);
  });

  it("uploads, refreshes the list, and clears the busy flag", async () => {
    let releaseUpload: (response: Response) => void = () => undefined;
    const fetchMock = stubBackend({
      "GET /api/documents": [() => json([]), () => json([row()])],
      "POST /api/documents": [
        () => new Promise<Response>((resolve) => (releaseUpload = resolve)),
      ],
    });
    const { result } = renderHook(() => useDocuments());
    await waitFor(() => expect(result.current.loaded).toBe(true));

    let pending: Promise<unknown> = Promise.resolve();
    act(() => {
      pending = result.current.upload(new File(["# notes"], "notes.md"));
    });
    await waitFor(() => expect(result.current.uploading).toBe(true));
    const body = fetchMock.mock.calls[1]?.[1]?.body;
    expect(body).toBeInstanceOf(FormData);
    expect(((body as FormData).get("file") as File).name).toBe("notes.md");

    await act(async () => {
      releaseUpload(json({ id: "manual" }, 201));
      await expect(pending).resolves.toEqual({ status: "uploaded" });
    });
    expect(result.current.uploading).toBe(false);
    expect(result.current.documents).toEqual([row()]);
    expect(requests(fetchMock)).toEqual([
      "GET /api/documents",
      "POST /api/documents",
      "GET /api/documents",
    ]);
  });

  it("returns safe upload failures for a rejected request and a lost connection", async () => {
    stubBackend({
      "GET /api/documents": [() => json([])],
      "POST /api/documents": [
        () => new Response(RAW_SECRET, { status: 413 }),
        () => {
          throw new TypeError("Failed to fetch");
        },
      ],
    });
    const { result } = renderHook(() => useDocuments());
    await waitFor(() => expect(result.current.loaded).toBe(true));

    const file = new File(["%PDF"], "manual.pdf");
    expect(await act(() => result.current.upload(file))).toEqual({
      status: "failed",
      message: "檔案太大，伺服器無法處理。",
    });
    expect(await act(() => result.current.upload(file))).toEqual({
      status: "failed",
      message: "連不上伺服器，請確認網路後再試。",
    });
    expect(result.current.uploading).toBe(false);
  });

  it("deletes and refreshes, and refreshes a stale row after a 404", async () => {
    const fetchMock = stubBackend({
      "GET /api/documents": [
        () => json([row(), row({ id: "gone", filename: "舊檔.pdf" })]),
        () => json([row({ id: "gone", filename: "舊檔.pdf" })]),
        () => json([]),
      ],
      "DELETE /api/documents/manual": [() => new Response(null, { status: 204 })],
      "DELETE /api/documents/gone": [
        () => new Response(RAW_SECRET, { status: 404 }),
      ],
    });
    const { result } = renderHook(() => useDocuments());
    await waitFor(() => expect(result.current.documents).toHaveLength(2));

    expect(await act(() => result.current.remove(row()))).toEqual({ ok: true });
    expect(result.current.documents.map((doc) => doc.id)).toEqual(["gone"]);

    expect(
      await act(() =>
        result.current.remove(row({ id: "gone", filename: "舊檔.pdf" })),
      ),
    ).toEqual({ ok: false, message: "這份文件已經不在了，請重新載入清單。" });
    await waitFor(() => expect(result.current.documents).toEqual([]));
    expect(requests(fetchMock)).toEqual([
      "GET /api/documents",
      "DELETE /api/documents/manual",
      "GET /api/documents",
      "DELETE /api/documents/gone",
      "GET /api/documents",
    ]);
  });

  it("returns a safe delete failure when the server cannot be reached", async () => {
    stubBackend({
      "GET /api/documents": [() => json([row()])],
      "DELETE /api/documents/manual": [
        () => {
          throw new TypeError("Failed to fetch");
        },
      ],
    });
    const { result } = renderHook(() => useDocuments());
    await waitFor(() => expect(result.current.documents).toHaveLength(1));

    expect(await act(() => result.current.remove(row()))).toEqual({
      ok: false,
      message: "連不上伺服器，請確認網路後再試。",
    });
    expect(result.current.documents).toEqual([row()]);
  });
});
