import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  ChatModelRunOptions,
  ChatModelRunResult,
  ThreadAssistantMessagePart,
} from "@assistant-ui/react";
import { BackendAdapter } from "./BackendAdapter";

const runOptions = (
  messages: readonly unknown[] = [],
): ChatModelRunOptions =>
  ({
    messages,
    runConfig: {},
    abortSignal: new AbortController().signal,
    context: {},
    unstable_getMessage: () => {
      throw new Error("not used by this adapter");
    },
  }) as unknown as ChatModelRunOptions;

const collectUpdates = async (
  messages: readonly unknown[] = [],
): Promise<ChatModelRunResult[]> => {
  const updates: ChatModelRunResult[] = [];
  const iterator = BackendAdapter.run(runOptions(messages)) as AsyncGenerator<
    ChatModelRunResult,
    void
  >;
  for await (const update of iterator) updates.push(update);
  return updates;
};

const sseResponse = (events: readonly unknown[]): Response => {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      }
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
};

const chunkedSseResponse = (chunks: readonly string[]): Response => {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } },
  );
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("BackendAdapter", () => {
  it("keeps SSE part and source order in a named assistant-ui data part", async () => {
    const sources = [
      {
        sourceId: "page-5",
        documentId: "manual",
        unitId: 105,
        unitIndex: 4,
        filename: "手冊.pdf",
        pageNumber: 5,
        kind: "page",
      },
      {
        sourceId: "page-2",
        documentId: "manual",
        unitId: 102,
        unitIndex: 1,
        filename: "手冊.pdf",
        pageNumber: 2,
        kind: "page",
      },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        sseResponse([
          { type: "text-delta", text: "前段" },
          { type: "text-delta", text: "後段" },
          { type: "sources", sources },
          { type: "done" },
        ]),
      ),
    );

    const updates = await collectUpdates();
    const finalContent = updates.at(-1)?.content;

    expect(finalContent).toEqual([
      { type: "text", text: "前段後段" },
      { type: "data", name: "sources", data: sources },
    ] satisfies ThreadAssistantMessagePart[]);
  });

  it("echoes the latest answer's sources so the next context keeps their order", async () => {
    const sources = [
      {
        sourceId: "S1",
        documentId: "manual",
        unitId: 103,
        unitIndex: 2,
        filename: "手冊.pdf",
        pageNumber: 3,
        kind: "page",
      },
      {
        sourceId: "S2",
        documentId: "manual",
        unitId: 106,
        unitIndex: 5,
        filename: "手冊.pdf",
        pageNumber: 6,
        kind: "page",
      },
    ];
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      sseResponse([{ type: "sources", sources: [] }, { type: "done" }]),
    );
    vi.stubGlobal("fetch", fetchMock);

    await collectUpdates([
      { role: "user", content: [{ type: "text", text: "第一個問題" }] },
      {
        role: "assistant",
        content: [
          { type: "text", text: "第一個回答" },
          { type: "data", name: "sources", data: sources },
        ],
      },
      { role: "user", content: [{ type: "text", text: "追問" }] },
    ]);

    const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body));
    expect(body.previousSources).toEqual(sources);
  });

  it("sends no previous sources when the answer carried none", async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      sseResponse([{ type: "sources", sources: [] }, { type: "done" }]),
    );
    vi.stubGlobal("fetch", fetchMock);

    await collectUpdates([
      { role: "user", content: [{ type: "text", text: "第一個問題" }] },
    ]);

    const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body));
    expect(body.previousSources).toEqual([]);
  });

  it("parses CRLF SSE events split across transport chunks", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        chunkedSseResponse([
          'data: {"type":"text-de',
          'lta","text":"跨區塊"}\r\n\r\n',
          'data: {"type":"sources","sources":[]}\r\n',
          '\r\ndata: {"type":"done"}\r\n\r\n',
        ]),
      ),
    );

    const updates = await collectUpdates();
    expect(updates.at(-1)?.content).toEqual([
      { type: "text", text: "跨區塊" },
      { type: "data", name: "sources", data: [] },
    ]);
  });

  it("rejects an incomplete stream and invalid source identifiers", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        sseResponse([
          { type: "text-delta", text: "partial" },
          {
            type: "sources",
            sources: [
              {
                sourceId: "S1",
                documentId: "document",
                unitId: "not-a-number",
                unitIndex: 0,
                filename: "notes.md",
                pageNumber: null,
                kind: "text",
              },
            ],
          },
        ]),
      ),
    );

    await expect(collectUpdates()).rejects.toThrow(
      "回覆格式異常，請重新送出問題。",
    );

    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        sseResponse([
          { type: "text-delta", text: "partial" },
          { type: "sources", sources: [] },
        ]),
      ),
    );
    await expect(collectUpdates()).rejects.toThrow(
      "回覆傳輸中斷，請重新送出問題。",
    );
  });

  it("does not expose a provider response or secret in request errors", async () => {
    const rawSecret = "provider response Authorization: Bearer sk-secret-456";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(rawSecret, { status: 500 })),
    );

    let thrown: unknown;
    try {
      await collectUpdates();
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).message).toBe(
      "問答服務暫時無法使用，請稍後再試。",
    );
    expect((thrown as Error).message).not.toContain(rawSecret);
  });

  it("does not expose raw SSE error messages", async () => {
    const rawSecret = "upstream provider leaked sk-secret-789";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        sseResponse([{ type: "error", message: rawSecret }]),
      ),
    );

    let thrown: unknown;
    try {
      await collectUpdates();
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).message).toBe(
      "系統暫時無法產生回覆，請稍後再試。",
    );
    expect((thrown as Error).message).not.toContain(rawSecret);
  });
});
