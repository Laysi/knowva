import { describe, expect, it } from "vitest";
import type { ChatSource } from "./BackendAdapter";
import {
  ConversationRequestError,
  conversationRequestErrorMessage,
  conversationTitle,
  decodeStoredMessages,
  isTransientFailure,
} from "./conversationApi";

const SOURCE: ChatSource = {
  sourceId: "S1",
  documentId: "manual",
  unitId: 137,
  unitIndex: 36,
  filename: "設備保養手冊.pdf",
  pageNumber: 37,
  kind: "page",
};

const question = (id: string, text: string) => ({
  id,
  role: "user",
  createdAt: "2026-10-06T01:02:03.000Z",
  content: [{ type: "text", text }],
  attachments: [],
  metadata: { custom: {} },
});

const answer = (id: string, content: unknown[], status: unknown) => ({
  id,
  role: "assistant",
  createdAt: "2026-10-06T01:02:04.000Z",
  content,
  status,
  metadata: { custom: {}, steps: [{ usage: { promptTokens: 1 } }] },
});

describe("decodeStoredMessages", () => {
  it("rebuilds text and sources in save order, with the last message as head", () => {
    const repository = decodeStoredMessages([
      { parentId: null, message: question("q1", "馬達多久保養一次？") },
      {
        parentId: "q1",
        message: answer(
          "a1",
          [
            { type: "text", text: "每三個月。" },
            { type: "data", name: "sources", data: [SOURCE] },
          ],
          { type: "complete", reason: "stop" },
        ),
      },
    ]);

    expect(repository?.headId).toBe("a1");
    expect(repository?.messages.map(({ parentId }) => parentId)).toEqual([
      null,
      "q1",
    ]);
    const [stored, reply] = repository!.messages.map(({ message }) => message);
    expect(stored).toMatchObject({
      id: "q1",
      role: "user",
      content: [{ type: "text", text: "馬達多久保養一次？" }],
      attachments: [],
    });
    expect(stored!.createdAt).toEqual(new Date("2026-10-06T01:02:03.000Z"));
    expect(reply).toMatchObject({
      role: "assistant",
      content: [
        { type: "text", text: "每三個月。" },
        { type: "data", name: "sources", data: [SOURCE] },
      ],
      status: { type: "complete", reason: "stop" },
    });
    // Nothing stored beyond what the page renders comes back.
    expect(reply!.metadata.steps).toEqual([]);
  });

  it("keeps an errored answer as incomplete with its safe error text", () => {
    const repository = decodeStoredMessages([
      { parentId: null, message: question("q1", "今天的產線狀況？") },
      {
        parentId: "q1",
        message: answer("a1", [], {
          type: "incomplete",
          reason: "error",
          error: "系統暫時無法產生回覆，請稍後再試。",
        }),
      },
    ]);

    expect(repository?.messages[1]?.message.status).toEqual({
      type: "incomplete",
      reason: "error",
      error: "系統暫時無法產生回覆，請稍後再試。",
    });
  });

  it("drops parts it does not render and malformed sources", () => {
    const repository = decodeStoredMessages([
      {
        parentId: null,
        message: {
          ...question("q1", "問題"),
          content: [
            { type: "text", text: "問題" },
            { type: "image", image: "data:image/png;base64,AAAA" },
          ],
        },
      },
      {
        parentId: "q1",
        message: answer(
          "a1",
          [
            { type: "text", text: "<img src=x onerror=alert(1)>" },
            { type: "data", name: "sources", data: [{ ...SOURCE, unitId: -1 }] },
            { type: "tool-call", toolName: "x" },
          ],
          { type: "running" },
        ),
      },
    ]);

    expect(repository?.messages[0]?.message.content).toEqual([
      { type: "text", text: "問題" },
    ]);
    const reply = repository?.messages[1]?.message;
    expect(reply?.content).toEqual([
      { type: "text", text: "<img src=x onerror=alert(1)>" },
    ]);
    // Saved while it was still running: it was cut off.
    expect(reply?.status).toEqual({ type: "incomplete", reason: "cancelled" });
  });

  it("drops an unreadable message with everything that follows from it", () => {
    const repository = decodeStoredMessages([
      { parentId: null, message: question("q1", "第一題") },
      { parentId: "q1", message: { id: "a1", role: "system", content: [] } },
      { parentId: "a1", message: question("q2", "接在壞掉的回答後面") },
      { parentId: "missing", message: question("q3", "找不到前一則") },
      { parentId: null, message: question("q1", "重複的 id") },
      "not a message",
    ]);

    expect(repository).toEqual({
      headId: "q1",
      messages: [expect.objectContaining({ parentId: null })],
    });
    expect(repository?.messages[0]?.message.content).toEqual([
      { type: "text", text: "第一題" },
    ]);
  });

  it("refuses anything but a list and reads an empty one as no messages", () => {
    expect(decodeStoredMessages({ messages: [] })).toBeNull();
    expect(decodeStoredMessages("[]")).toBeNull();
    expect(decodeStoredMessages([])).toEqual({ messages: [] });
  });
});

describe("conversationTitle", () => {
  it("folds whitespace and keeps the first 30 code points", () => {
    expect(conversationTitle("  馬達\n\n多久\t保養？  ")).toBe("馬達 多久 保養？");
    const long = "𠀋".repeat(31);
    expect(Array.from(conversationTitle(long))).toHaveLength(30);
    expect(conversationTitle(" \n ")).toBe("");
  });
});

describe("conversation request failures", () => {
  it("retries only what may succeed later", () => {
    for (const status of [undefined, 408, 429, 500, 502, 503]) {
      expect(isTransientFailure(new ConversationRequestError(status))).toBe(true);
    }
    for (const status of [400, 404, 409, 413]) {
      expect(isTransientFailure(new ConversationRequestError(status))).toBe(false);
    }
  });

  it("names every failure with fixed text", () => {
    expect(conversationRequestErrorMessage("list")).toContain("連不上伺服器");
    expect(conversationRequestErrorMessage("list", 500)).toBe(
      "對話紀錄載入失敗，請稍後再試。",
    );
    expect(conversationRequestErrorMessage("delete", 500)).toBe(
      "刪除對話失敗，請稍後再試。",
    );
    expect(conversationRequestErrorMessage("clear", 503)).toBe(
      "清除對話紀錄失敗，請稍後再試。",
    );
  });
});
