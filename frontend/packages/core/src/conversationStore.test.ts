import { describe, expect, it, vi } from "vitest";
import type {
  ExportedMessageRepositoryItem,
  ThreadMessage,
} from "@assistant-ui/react";
import type { ChatSource } from "./BackendAdapter";
import { CHAT_ERROR_MESSAGES } from "./BackendAdapter";
import {
  browserConversationStorage,
  CONVERSATION_INDEX_KEY,
  CONVERSATION_STORAGE_PREFIX,
  conversationMessagesKey,
  ConversationStore,
  conversationTitle,
  MAX_STORED_CONVERSATIONS,
} from "./conversationStore";

/** localStorage in memory; `capacity` counts key and value characters like a browser quota. */
class MemoryStorage {
  readonly items = new Map<string, string>();
  capacity = Number.POSITIVE_INFINITY;
  failWith: Error | null = null;

  get length() {
    return this.items.size;
  }

  key(index: number) {
    return [...this.items.keys()][index] ?? null;
  }

  getItem(key: string) {
    if (this.failWith) throw this.failWith;
    return this.items.get(key) ?? null;
  }

  setItem(key: string, value: string) {
    if (this.failWith) throw this.failWith;
    const current = this.items.get(key);
    const used =
      this.used() - (current === undefined ? 0 : key.length + current.length);
    if (used + key.length + value.length > this.capacity) {
      throw new DOMException("The quota has been exceeded.", "QuotaExceededError");
    }
    this.items.set(key, value);
  }

  removeItem(key: string) {
    if (this.failWith) throw this.failWith;
    this.items.delete(key);
  }

  clear() {
    this.items.clear();
  }

  used() {
    let total = 0;
    for (const [key, value] of this.items) total += key.length + value.length;
    return total;
  }

  asStorage() {
    return this as unknown as Storage;
  }
}

const PAGE_SOURCE: ChatSource = {
  sourceId: "S1",
  documentId: "manual",
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
  filename: "維運交接筆記.md",
  pageNumber: null,
  kind: "text",
};

const question = (
  id: string,
  text: string,
  parentId: string | null = null,
): ExportedMessageRepositoryItem => ({
  parentId,
  message: {
    id,
    role: "user",
    createdAt: new Date(Date.UTC(2026, 9, 6, 8)),
    content: [{ type: "text", text }],
    attachments: [],
    metadata: { custom: {} },
  },
});

const answer = (
  id: string,
  parentId: string,
  text: string,
  sources: ChatSource[] | null,
  status: Extract<ThreadMessage, { role: "assistant" }>["status"] = {
    type: "complete",
    reason: "unknown",
  },
): ExportedMessageRepositoryItem => ({
  parentId,
  message: {
    id,
    role: "assistant",
    createdAt: new Date(Date.UTC(2026, 9, 6, 8, 1)),
    content: [
      { type: "text", text },
      ...(sources ? [{ type: "data" as const, name: "sources", data: sources }] : []),
    ],
    status,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {},
    },
  },
});

/** A store over `storage` whose clock moves one minute per write. */
const storeOver = (storage: MemoryStorage, start = Date.UTC(2026, 0, 1)) => {
  let clock = start;
  return new ConversationStore(storage.asStorage(), () => (clock += 60_000));
};

const addConversation = (store: ConversationStore, title: string) => {
  const id = store.newId();
  store.append(id, question(`${id}-q`, title));
  store.append(id, answer(`${id}-a`, `${id}-q`, `${title}的回答`, [PAGE_SOURCE]));
  return id;
};

const conversationKeys = (storage: MemoryStorage) =>
  [...storage.items.keys()].filter((key) =>
    key.startsWith(CONVERSATION_STORAGE_PREFIX),
  );

describe("conversationTitle", () => {
  it("keeps the first 30 code points, not UTF-16 units", () => {
    const text = "𠀋".repeat(31);
    expect(text.length).toBe(62);
    expect(conversationTitle(text)).toBe("𠀋".repeat(30));
    expect(Array.from(conversationTitle(`  ${"一".repeat(40)}`))).toHaveLength(30);
  });

  it("folds whitespace runs and leaves short questions whole", () => {
    expect(conversationTitle("  馬達\n\n多久\t保養一次？ ")).toBe("馬達 多久 保養一次？");
    expect(conversationTitle("馬達多久保養一次？")).toBe("馬達多久保養一次？");
  });
});

describe("ConversationStore", () => {
  it("brings back every message, sources and a failed answer included, in a new page", () => {
    const storage = new MemoryStorage();
    const store = storeOver(storage);
    const id = store.newId();
    store.append(id, question("q1", "馬達多久保養一次？"));
    store.append(id, answer("a1", "q1", "每三個月一次。", [PAGE_SOURCE, TEXT_SOURCE]));
    store.append(id, question("q2", "交接要注意什麼？", "a1"));
    store.append(
      id,
      answer("a2", "q2", "請先", null, {
        type: "incomplete",
        reason: "error",
        error: CHAT_ERROR_MESSAGES.interrupted,
      }),
    );

    const reloaded = storeOver(storage);
    expect(reloaded.list()).toEqual([
      expect.objectContaining({ id, title: "馬達多久保養一次？" }),
    ]);
    const repository = reloaded.load(id);

    expect(repository.headId).toBe("a2");
    expect(repository.messages.map(({ parentId }) => parentId)).toEqual([
      null,
      "q1",
      "a1",
      "q2",
    ]);
    const [first, second, , fourth] = repository.messages.map(
      ({ message }) => message,
    );
    expect(first).toMatchObject({
      id: "q1",
      role: "user",
      content: [{ type: "text", text: "馬達多久保養一次？" }],
      attachments: [],
    });
    expect(first!.createdAt).toEqual(new Date(Date.UTC(2026, 9, 6, 8)));
    expect(second).toMatchObject({
      role: "assistant",
      content: [
        { type: "text", text: "每三個月一次。" },
        { type: "data", name: "sources", data: [PAGE_SOURCE, TEXT_SOURCE] },
      ],
      status: { type: "complete", reason: "unknown" },
    });
    expect(fourth).toMatchObject({
      content: [{ type: "text", text: "請先" }],
      status: {
        type: "incomplete",
        reason: "error",
        error: CHAT_ERROR_MESSAGES.interrupted,
      },
    });
  });

  it("stores a stopped answer as cancelled and replaces a message written twice", () => {
    const storage = new MemoryStorage();
    const store = storeOver(storage);
    const id = store.newId();
    store.append(id, question("q1", "整理上週的保養紀錄"));
    store.append(id, answer("a1", "q1", "正在", null, { type: "running" }));
    store.append(
      id,
      answer("a1", "q1", "正在整理", null, { type: "incomplete", reason: "cancelled" }),
    );

    const messages = storeOver(storage).load(id).messages;
    expect(messages).toHaveLength(2);
    expect(messages[1]!.message).toMatchObject({
      content: [{ type: "text", text: "正在整理" }],
      status: { type: "incomplete", reason: "cancelled" },
    });
  });

  it("titles a conversation from its first question and keeps that title", () => {
    const storage = new MemoryStorage();
    const store = storeOver(storage);
    const id = store.newId();
    store.append(id, question("q1", "第一個問題"));
    store.append(id, answer("a1", "q1", "回答", null));
    store.append(id, question("q2", "第二個問題", "a1"));

    expect(store.list()[0]!.title).toBe("第一個問題");
    expect(store.titleFor(id, [])).toBe("第一個問題");
  });

  it(`deletes the least recently updated conversation when a ${MAX_STORED_CONVERSATIONS + 1}st one is stored`, () => {
    const storage = new MemoryStorage();
    const store = storeOver(storage);
    const ids = Array.from({ length: MAX_STORED_CONVERSATIONS }, (_, index) =>
      addConversation(store, `對話 ${index + 1}`),
    );
    // The oldest one is followed up, so the second one is now the stalest.
    store.append(ids[0]!, question(`${ids[0]}-q2`, "追問", `${ids[0]}-a`));
    const evicted = vi.fn();
    store.onEvict(evicted);

    const newest = addConversation(store, "第五十一串");

    expect(evicted).toHaveBeenCalledExactlyOnceWith(ids[1]);
    expect(storage.getItem(conversationMessagesKey(ids[1]!))).toBeNull();
    const listed = storeOver(storage).list();
    expect(listed).toHaveLength(MAX_STORED_CONVERSATIONS);
    expect(listed[0]!.id).toBe(newest);
    expect(listed[1]!.id).toBe(ids[0]);
    expect(listed.map(({ id }) => id)).not.toContain(ids[1]);
    expect(conversationKeys(storage)).toHaveLength(MAX_STORED_CONVERSATIONS + 1);
  });

  it("removes corrupt JSON instead of failing", () => {
    const storage = new MemoryStorage();
    storage.setItem(CONVERSATION_INDEX_KEY, "{not json");
    expect(storeOver(storage).list()).toEqual([]);
    expect(storage.getItem(CONVERSATION_INDEX_KEY)).toBeNull();

    const store = storeOver(storage);
    const kept = addConversation(store, "完好的對話");
    const broken = addConversation(store, "壞掉的對話");
    storage.setItem(conversationMessagesKey(broken), '{"version":1,"headId":');

    const reloaded = storeOver(storage);
    expect(reloaded.list().map(({ id }) => id)).toEqual([kept]);
    expect(reloaded.get(broken)).toBeUndefined();
    expect(storage.getItem(conversationMessagesKey(broken))).toBeNull();
    expect(storeOver(storage).list().map(({ id }) => id)).toEqual([kept]);
  });

  it("removes data of another version", () => {
    const storage = new MemoryStorage();
    const store = storeOver(storage);
    const id = addConversation(store, "v1 的對話");
    const thread = JSON.parse(storage.getItem(conversationMessagesKey(id))!);
    storage.setItem(
      conversationMessagesKey(id),
      JSON.stringify({ ...thread, version: 2 }),
    );
    expect(storeOver(storage).list()).toEqual([]);
    expect(conversationKeys(storage)).toEqual([]);

    storage.setItem(
      CONVERSATION_INDEX_KEY,
      JSON.stringify({ version: 2, conversations: [] }),
    );
    expect(storeOver(storage).latestId()).toBeUndefined();
    expect(storage.getItem(CONVERSATION_INDEX_KEY)).toBeNull();
  });

  it("drops entries and messages of the wrong shape, keeping the rest", () => {
    const storage = new MemoryStorage();
    const store = storeOver(storage);
    const kept = addConversation(store, "完好的對話");
    const orphanedParent = addConversation(store, "斷掉的對話");
    const index = JSON.parse(storage.getItem(CONVERSATION_INDEX_KEY)!);
    storage.setItem(
      CONVERSATION_INDEX_KEY,
      JSON.stringify({
        ...index,
        conversations: [
          ...index.conversations,
          { id: "../escape", title: "壞 id", createdAt: 1, updatedAt: 1 },
          { id: "no-times", title: "沒有時間" },
        ],
      }),
    );
    const thread = JSON.parse(storage.getItem(conversationMessagesKey(orphanedParent))!);
    thread.messages[1].parentId = "missing";
    storage.setItem(conversationMessagesKey(orphanedParent), JSON.stringify(thread));
    storage.setItem(conversationMessagesKey("stray"), "{}");
    storage.setItem("someone-else", "untouched");

    expect(storeOver(storage).list().map(({ id }) => id)).toEqual([kept]);
    expect(JSON.parse(storage.getItem(CONVERSATION_INDEX_KEY)!).conversations).toHaveLength(1);
    expect(conversationKeys(storage).sort()).toEqual(
      [CONVERSATION_INDEX_KEY, conversationMessagesKey(kept)].sort(),
    );
    expect(storage.getItem("someone-else")).toBe("untouched");
  });

  it("keeps answering without storage and reports nothing as saved", () => {
    const storage = new MemoryStorage();
    storage.failWith = new DOMException("The operation is insecure.", "SecurityError");
    const store = storeOver(storage);

    expect(store.list()).toEqual([]);
    const id = store.newId();
    expect(() => store.append(id, question("q1", "問題"))).not.toThrow();
    expect(store.available).toBe(false);
    expect(store.isSaved(id)).toBe(false);
    expect(store.load(id)).toEqual({ messages: [] });

    const offline = new ConversationStore(null);
    expect(offline.available).toBe(false);
    expect(() => offline.append(offline.newId(), question("q1", "問題"))).not.toThrow();
  });

  it("treats a failing write that is not about space as storage being unavailable", () => {
    const storage = new MemoryStorage();
    const store = storeOver(storage);
    const listener = vi.fn();
    store.subscribe(listener);
    const id = store.newId();
    storage.setItem = () => {
      throw new DOMException("The operation is insecure.", "SecurityError");
    };

    store.append(id, question("q1", "問題"));

    expect(store.available).toBe(false);
    expect(listener).toHaveBeenCalled();
  });

  it("deletes the least recently updated conversations until a full storage takes the write", () => {
    const storage = new MemoryStorage();
    const store = storeOver(storage);
    const ids = ["一", "二", "三", "四"].map((n) => addConversation(store, `對話${n}`));
    const evicted: string[] = [];
    store.onEvict((id) => evicted.push(id));
    const key = conversationMessagesKey(ids[0]!);
    const footprint = key.length + storage.getItem(key)!.length;
    storage.capacity = storage.used() + 200;

    // More than one conversation's worth past the free space, less than two.
    const long = "長".repeat(footprint + 200);
    store.append(ids[3]!, question(`${ids[3]}-q2`, long, `${ids[3]}-a`));

    expect(evicted).toEqual(ids.slice(0, 2));
    expect(store.isSaved(ids[3]!)).toBe(true);
    expect(storeOver(storage).list().map(({ id }) => id)).toEqual([ids[3], ids[2]]);
    expect(storeOver(storage).load(ids[3]!).messages).toHaveLength(3);
  });

  it("never evicts the conversation on screen to make room", () => {
    const storage = new MemoryStorage();
    const store = storeOver(storage);
    const [oldest, middle, newest] = ["一", "二", "三"].map((n) =>
      addConversation(store, `對話${n}`),
    );
    store.setOpenConversation(oldest);
    storage.capacity = storage.used() + 100;

    store.append(newest!, question(`${newest}-q2`, "長".repeat(300), `${newest}-a`));

    expect(storage.getItem(conversationMessagesKey(oldest!))).not.toBeNull();
    expect(storage.getItem(conversationMessagesKey(middle!))).toBeNull();
    expect(store.isSaved(newest!)).toBe(true);
  });

  it("reports the conversation as not saved when it alone does not fit, and saves it once it does", () => {
    const storage = new MemoryStorage();
    const store = storeOver(storage);
    const other = addConversation(store, "另一串");
    const id = store.newId();
    store.append(id, question("q1", "第一個問題"));
    storage.capacity = storage.used() + 100;

    store.append(id, answer("a1", "q1", "長".repeat(2_000), null));

    expect(store.isSaved(id)).toBe(false);
    expect(storage.getItem(conversationMessagesKey(other))).toBeNull();
    // The partial copy is gone too, so a reload does not pass it off as whole.
    expect(storage.getItem(conversationMessagesKey(id))).toBeNull();
    expect(storeOver(storage).list()).toEqual([]);

    storage.capacity = Number.POSITIVE_INFINITY;
    store.append(id, question("q2", "追問", "a1"));

    expect(store.isSaved(id)).toBe(true);
    expect(storeOver(storage).load(id).messages.map(({ message }) => message.id)).toEqual([
      "q1",
      "a1",
      "q2",
    ]);
  });

  it("removes one conversation's keys and ignores its late writes", () => {
    const storage = new MemoryStorage();
    const store = storeOver(storage);
    const kept = addConversation(store, "留下");
    const removed = addConversation(store, "刪掉");

    store.remove(removed);
    store.append(removed, question(`${removed}-q2`, "晚到的寫入", `${removed}-a`));

    expect(storage.getItem(conversationMessagesKey(removed))).toBeNull();
    expect(storeOver(storage).list().map(({ id }) => id)).toEqual([kept]);
  });

  it("clears every key it owns and nothing else", () => {
    const storage = new MemoryStorage();
    const store = storeOver(storage);
    const id = addConversation(store, "對話");
    storage.setItem(`${CONVERSATION_STORAGE_PREFIX}unknown`, "x");
    storage.setItem("someone-else", "untouched");

    store.clear();
    store.append(id, question(`${id}-q2`, "晚到的寫入", `${id}-a`));

    expect(conversationKeys(storage)).toEqual([]);
    expect(storage.getItem("someone-else")).toBe("untouched");
  });
});

describe("browserConversationStorage", () => {
  it("is null where reading localStorage throws", () => {
    const blocked = {
      getItem() {
        throw new DOMException("The operation is insecure.", "SecurityError");
      },
    };
    vi.stubGlobal("localStorage", blocked);
    try {
      expect(browserConversationStorage()).toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
