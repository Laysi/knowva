import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  ExportedMessageRepositoryItem,
  ThreadMessage,
} from "@assistant-ui/react";
import { AssistantMessageStream } from "assistant-stream";
import { ConversationSync, isLocalConversationId } from "./conversationAdapter";
import type { ConversationListItem } from "./conversationAdapter";
import {
  CONVERSATION_LOAD_FAILED_MESSAGE,
  CONVERSATION_NOT_SAVED_MESSAGE,
  ConversationRequestError,
} from "./conversationApi";

const ID = "00000000-0000-4000-8000-000000000001";
const RAW_SECRET = "provider said: Authorization Bearer sk-adapter";

const summary = (id: string, title: string | null = null) => ({
  id,
  title,
  createdAt: "2026-10-06T00:00:00.000Z",
  updatedAt: "2026-10-06T00:00:01.000Z",
});

const userItem = (id: string, text: string): ExportedMessageRepositoryItem => ({
  parentId: null,
  message: {
    id,
    role: "user",
    createdAt: new Date("2026-10-06T00:00:00.000Z"),
    content: [{ type: "text", text }],
    attachments: [],
    metadata: { custom: {} },
  } as ThreadMessage,
});

const answerItem = (
  id: string,
  parentId: string,
  text: string,
): ExportedMessageRepositoryItem => ({
  parentId,
  message: {
    id,
    role: "assistant",
    createdAt: new Date("2026-10-06T00:00:01.000Z"),
    content: [{ type: "text", text }],
    status: { type: "complete", reason: "unknown" },
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {},
    },
  } as ThreadMessage,
});

type Route = (init: RequestInit | undefined) => Response | Promise<Response>;

/** Routes by "METHOD path"; anything else answers 404. Keeps every request. */
const stubFetch = (routes: Record<string, Route>) => {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const route = routes[`${init?.method ?? "GET"} ${String(input)}`];
    return route ? route(init) : new Response(null, { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

const requests = (fetchMock: ReturnType<typeof stubFetch>) =>
  fetchMock.mock.calls.map(([input, init]) => {
    const body = init?.body ? (JSON.parse(String(init.body)) as unknown) : undefined;
    return { route: `${init?.method ?? "GET"} ${String(input)}`, body };
  });

const savedResponse = (title: string | null = null) =>
  Response.json({ conversation: summary(ID, title) });

const listItem = (remoteId: string | undefined = ID) => {
  const item = {
    remoteId: () => remoteId,
    initialize: vi.fn(async () => ({ remoteId: remoteId ?? ID })),
    generateTitle: vi.fn(async () => undefined),
  } satisfies ConversationListItem;
  return item;
};

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};

const titleOf = async (stream: ReadableStream) => {
  let title: string | undefined;
  for await (const message of AssistantMessageStream.fromAssistantStream(stream)) {
    const part = message.parts.find((candidate) => candidate.type === "text");
    title = part?.type === "text" ? part.text : undefined;
  }
  return title;
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("ConversationSync saves", () => {
  it("sends a conversation's messages one at a time, in the order they came", async () => {
    const firstSave = deferred<Response>();
    let saves = 0;
    const fetchMock = stubFetch({
      [`POST /api/conversations/${ID}/messages`]: () => {
        saves += 1;
        return saves === 1 ? firstSave.promise : savedResponse("問題");
      },
    });
    const sync = new ConversationSync();
    const history = sync.historyAdapter(listItem());

    const question = history.append(userItem("q1", "問題"));
    const reply = history.append(answerItem("a1", "q1", "回答"));
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    // The answer waits for its question.
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(fetchMock).toHaveBeenCalledTimes(1);

    firstSave.resolve(savedResponse("問題"));
    await Promise.all([question, reply]);

    expect(requests(fetchMock)).toEqual([
      {
        route: `POST /api/conversations/${ID}/messages`,
        body: expect.objectContaining({ parentId: null }),
      },
      {
        route: `POST /api/conversations/${ID}/messages`,
        body: expect.objectContaining({ parentId: "q1" }),
      },
    ]);
    const [questionBody, replyBody] = requests(fetchMock).map(
      ({ body }) => body as { message: { id: string; createdAt: string } },
    );
    expect(questionBody!.message).toMatchObject({
      id: "q1",
      createdAt: "2026-10-06T00:00:00.000Z",
    });
    expect(replyBody!.message.id).toBe("a1");
    expect(sync.notice(ID)).toBeNull();
  });

  it("asks for the server's title once the first question is saved", async () => {
    stubFetch({
      [`POST /api/conversations/${ID}/messages`]: () => savedResponse("伺服器的標題"),
    });
    const sync = new ConversationSync();
    const item = listItem();
    await sync.historyAdapter(item).append(userItem("q1", "使用者的問題"));

    expect(item.generateTitle).toHaveBeenCalledTimes(1);
    const stream = await sync
      .listAdapter()
      .generateTitle(ID, [userItem("q1", "使用者的問題").message]);
    expect(await titleOf(stream)).toBe("伺服器的標題");
  });

  it("sends a save that failed for now again before the next message, and clears the notice", async () => {
    let fail = true;
    const fetchMock = stubFetch({
      [`POST /api/conversations/${ID}/messages`]: () =>
        fail ? new Response(RAW_SECRET, { status: 503 }) : savedResponse("問題"),
    });
    const sync = new ConversationSync();
    const history = sync.historyAdapter(listItem());

    await history.append(userItem("q1", "問題"));
    expect(sync.notice(ID)).toBe(CONVERSATION_NOT_SAVED_MESSAGE);

    fail = false;
    await history.append(answerItem("a1", "q1", "回答"));

    expect(
      requests(fetchMock).map(({ body }) => (body as { message: { id: string } }).message.id),
    ).toEqual(["q1", "q1", "a1"]);
    expect(sync.notice(ID)).toBeNull();
  });

  it("stops saving a conversation the server refused, without rejecting", async () => {
    const fetchMock = stubFetch({
      [`POST /api/conversations/${ID}/messages`]: () =>
        new Response(RAW_SECRET, { status: 413 }),
    });
    const sync = new ConversationSync();
    const history = sync.historyAdapter(listItem());

    await expect(history.append(userItem("q1", "問題"))).resolves.toBeUndefined();
    await history.append(answerItem("a1", "q1", "回答"));

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(sync.notice(ID)).toBe(CONVERSATION_NOT_SAVED_MESSAGE);
  });
});

describe("ConversationSync list adapter", () => {
  it("answers with a page-only id and says so when the server cannot start a conversation", async () => {
    stubFetch({
      "POST /api/conversations": () => new Response(RAW_SECRET, { status: 500 }),
    });
    const sync = new ConversationSync();

    const { remoteId } = await sync.listAdapter().initialize("__LOCALID_x");

    expect(isLocalConversationId(remoteId)).toBe(true);
    expect(sync.notice(remoteId)).toBe(CONVERSATION_NOT_SAVED_MESSAGE);
    const fetchMock = stubFetch({});
    const item = listItem(remoteId);
    await sync.historyAdapter(item).append(userItem("q1", "問題"));
    await sync.listAdapter().delete(remoteId);
    expect(fetchMock).not.toHaveBeenCalled();
    // Its title still comes from the first question, made here.
    const stream = await sync
      .listAdapter()
      .generateTitle(remoteId, [userItem("q1", "  本機 的\n標題 ").message]);
    expect(await titleOf(stream)).toBe("本機 的 標題");
  });

  it("reports conversations the server deleted to stay at the limit", async () => {
    const evicted = "00000000-0000-4000-8000-000000000009";
    stubFetch({
      "POST /api/conversations": () =>
        Response.json(
          { conversation: summary(ID), removedIds: [evicted] },
          { status: 201 },
        ),
    });
    const sync = new ConversationSync();
    const removed = vi.fn();
    sync.onRemoved(removed);

    const { remoteId } = await sync.listAdapter().initialize("__LOCALID_x");

    expect(remoteId).toBe(ID);
    expect(removed).toHaveBeenCalledWith([evicted]);
    // Taking it off the list sends nothing more.
    const fetchMock = stubFetch({});
    await sync.listAdapter().delete(evicted);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("treats deleting a conversation that is already gone as done, and reports other failures", async () => {
    stubFetch({ [`DELETE /api/conversations/${ID}`]: () => new Response(null, { status: 404 }) });
    const sync = new ConversationSync();
    await expect(sync.listAdapter().delete(ID)).resolves.toBeUndefined();

    stubFetch({
      "DELETE /api/conversations/other": () => new Response(RAW_SECRET, { status: 500 }),
    });
    await expect(sync.listAdapter().delete("other")).rejects.toEqual(
      new ConversationRequestError(500),
    );
  });

  it("turns a failed list into a safe message and keeps going after a retry", async () => {
    stubFetch({ "GET /api/conversations": () => new Response(RAW_SECRET, { status: 500 }) });
    const sync = new ConversationSync();
    const listener = vi.fn();
    sync.subscribe(listener);

    await expect(sync.listAdapter().list()).rejects.toBeInstanceOf(
      ConversationRequestError,
    );
    expect(sync.listStatus).toBe("error");
    expect(sync.listError).toBe("對話紀錄載入失敗，請稍後再試。");
    expect(sync.listError).not.toContain(RAW_SECRET);
    expect(listener).toHaveBeenCalled();

    stubFetch({
      "GET /api/conversations": () => Response.json([summary(ID, "第一個問題")]),
    });
    const { threads } = await sync.listAdapter().list();
    expect(threads).toEqual([
      expect.objectContaining({ remoteId: ID, title: "第一個問題", status: "regular" }),
    ]);
    expect(sync.listStatus).toBe("ready");
    expect(sync.listError).toBeNull();
  });

  it("knows when each conversation last changed, from the list and then from each save", async () => {
    const older = "00000000-0000-4000-8000-000000000002";
    const at = (id: string, title: string, updatedAt: string) => ({
      ...summary(id, title),
      updatedAt,
    });
    stubFetch({
      "GET /api/conversations": () =>
        Response.json([
          at(ID, "問題", "2026-10-06T01:00:00.000Z"),
          at(older, "較舊的問題", "2026-10-05T01:00:00.000Z"),
        ]),
      [`POST /api/conversations/${older}/messages`]: () =>
        Response.json({
          conversation: at(older, "較舊的問題", "2026-10-06T02:00:00.000Z"),
        }),
      [`DELETE /api/conversations/${ID}`]: () => new Response(null, { status: 204 }),
    });
    const sync = new ConversationSync();
    const adapter = sync.listAdapter();
    const listener = vi.fn();
    sync.subscribe(listener);

    await adapter.list();
    const listed = sync.updatedTimes;
    expect(listed.get(ID)?.toISOString()).toBe("2026-10-06T01:00:00.000Z");
    expect(listed.get(older)?.toISOString()).toBe("2026-10-05T01:00:00.000Z");

    listener.mockClear();
    expect(await sync.save(older, userItem("q1", "較舊的問題"))).toBe(true);
    expect(sync.updatedTimes.get(older)?.toISOString()).toBe("2026-10-06T02:00:00.000Z");
    expect(listener).toHaveBeenCalled();
    // A new map each time, so a snapshot taken earlier never changes under it.
    expect(sync.updatedTimes).not.toBe(listed);
    expect(listed.get(older)?.toISOString()).toBe("2026-10-05T01:00:00.000Z");

    // A list answer older than a save does not move a time back.
    await adapter.list();
    expect(sync.updatedTimes.get(older)?.toISOString()).toBe("2026-10-06T02:00:00.000Z");

    await adapter.delete(ID);
    expect(sync.updatedTimes.has(ID)).toBe(false);
  });

  it("reads a list response of the wrong shape as a failure", async () => {
    stubFetch({ "GET /api/conversations": () => Response.json({ conversations: [] }) });
    const sync = new ConversationSync();

    await expect(sync.listAdapter().list()).rejects.toEqual(
      new ConversationRequestError(502),
    );
    expect(sync.listError).toBe("對話紀錄載入失敗，請稍後再試。");
  });
});

describe("ConversationSync history load", () => {
  it("loads a conversation's saved messages", async () => {
    stubFetch({
      [`GET /api/conversations/${ID}/messages`]: () =>
        Response.json([
          { parentId: null, message: userItem("q1", "問題").message },
          { parentId: "q1", message: answerItem("a1", "q1", "回答").message },
        ]),
    });
    const sync = new ConversationSync();

    const repository = await sync.historyAdapter(listItem()).load();

    expect(repository.headId).toBe("a1");
    expect(repository.messages.map(({ message }) => message.id)).toEqual(["q1", "a1"]);
  });

  it("shows an empty conversation and saves nothing to it when its messages do not load", async () => {
    stubFetch({
      [`GET /api/conversations/${ID}/messages`]: () =>
        new Response(RAW_SECRET, { status: 500 }),
    });
    const sync = new ConversationSync();
    const history = sync.historyAdapter(listItem());

    await expect(history.load()).resolves.toEqual({ messages: [] });
    expect(sync.notice(ID)).toBe(CONVERSATION_LOAD_FAILED_MESSAGE);

    const fetchMock = stubFetch({});
    await history.append(userItem("q9", "接著問"));
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("asks nothing for a conversation that does not exist yet", async () => {
    const fetchMock = stubFetch({});
    const sync = new ConversationSync();

    const unsaved = { ...listItem(), remoteId: () => undefined };
    await expect(sync.historyAdapter(unsaved).load()).resolves.toEqual({
      messages: [],
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
