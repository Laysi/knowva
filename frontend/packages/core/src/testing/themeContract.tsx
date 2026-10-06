// The contract runs under Vitest in Node. A theme's tsconfig is a browser one,
// so the Node globals this file uses are brought in here rather than there.
/// <reference types="node" />
import { readFileSync } from "node:fs";
import type { ReactElement } from "react";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockInstance } from "vitest";
import { CHAT_ERROR_MESSAGES, chatRequestErrorMessage } from "../BackendAdapter";
import type { ChatSource } from "../BackendAdapter";
import {
  CONVERSATION_INDEX_KEY,
  CONVERSATION_NOT_SAVED_MESSAGE,
  CONVERSATION_STORAGE_PREFIX,
  CONVERSATION_TITLE_LENGTH,
  conversationMessagesKey,
  ConversationStore,
  MAX_STORED_CONVERSATIONS,
} from "../conversationStore";
import type { StoredConversation } from "../conversationStore";
import { MAX_UPLOAD_BYTES, validateUploadFile } from "../DocumentModel";
import { profileRequestErrorMessage } from "../useProfile";
import type { Profile } from "../useProfile";
import { themeContrasts, WCAG_AA_TEXT_CONTRAST } from "./themeColors";

export type ThemeContractHarness = {
  /** Shown in the test names. */
  name: string;
  /**
   * The theme's chat surface with its own runtime wiring (`useKnowvaRuntime`),
   * usually the whole app opened on chat. Rendering it again after `cleanup`
   * stands for a reload. The contract drives it like a person: it types into
   * the only textbox, presses the button named 送出, and stops with the button
   * named 停止. For conversation history it presses the button named 新對話,
   * opens a conversation with the button named by its title, deletes one with
   * the button whose name holds 刪除 and the title, and clears them all with
   * 清除全部對話 followed by 確定清除 in the page. It reads storage through
   * jsdom's localStorage and clears it around every test.
   */
  renderChat: () => ReactElement;
  /**
   * The theme's document library. Uploads go through its
   * `input[type="file"]`.
   */
  renderDocuments: () => ReactElement;
  /**
   * The stylesheets whose `:root` rules declare the `--theme-*` colours, as
   * paths relative to `baseUrl`.
   */
  stylesheets: readonly string[];
  /**
   * The test file's `import.meta.url`. Vite rewrites `new URL("…",
   * import.meta.url)` written in a test, so the contract resolves the paths.
   */
  baseUrl: string;
};

const RAW_SECRET = "provider said: Authorization Bearer sk-theme-contract";

/**
 * What `GET /api/profile` answers in every contract test. Each value differs
 * from the default profile, so a theme that still shows built-in copy fails.
 */
const CONTRACT_PROFILE: Profile = {
  id: "contract",
  theme: "contract",
  brand: { name: "契約品牌名稱", tagline: "契約品牌標語" },
  pages: {
    chat: { title: "契約問答頁標題", description: "契約問答頁說明" },
    documents: { title: "契約文件頁標題", description: "契約文件頁說明" },
  },
  suggestions: [
    { text: "契約總覽建議問題？", catalog: true },
    { text: "契約一般建議問題", catalog: false },
  ],
  composerPlaceholder: "契約輸入框提示",
};

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
  unitIndex: 1,
  filename: "維運交接筆記.md",
  pageNumber: null,
  kind: "text",
};

const encoder = new TextEncoder();
const sseEvent = (event: unknown) =>
  encoder.encode(`data: ${JSON.stringify(event)}\n\n`);

const sseResponse = (events: readonly unknown[]) =>
  new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const event of events) controller.enqueue(sseEvent(event));
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } },
  );

/**
 * Streams one delta, then stays open until the request is aborted, when the
 * body fails with the signal's reason as a browser's fetch does.
 */
const hangingSseResponse = (text: string, signal: AbortSignal) =>
  new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(sseEvent({ type: "text-delta", text }));
        signal.addEventListener("abort", () => controller.error(signal.reason));
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } },
  );

type Handler = (init: RequestInit | undefined) => Response | Promise<Response>;

/**
 * Unlisted routes answer 404. `GET /api/profile` answers the contract profile
 * unless a test lists it.
 */
const stubFetch = (handlers: Record<string, Handler>) => {
  const routes: Record<string, Handler> = {
    "GET /api/profile": () => Response.json(CONTRACT_PROFILE),
    ...handlers,
  };
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${String(input)}`;
    const handler = routes[key];
    if (!handler) return new Response(null, { status: 404 });
    return handler(init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

const sentRequests = (fetchMock: ReturnType<typeof stubFetch>) =>
  fetchMock.mock.calls.map(
    ([input, init]) => `${init?.method ?? "GET"} ${String(input)}`,
  );

const pageText = () => document.body.textContent ?? "";

/**
 * jsdom has no layout. Fill in the few layout APIs chat surfaces call, for the
 * duration of one test, without replacing anything jsdom does provide.
 */
const installLayoutStubs = () => {
  const restore: (() => void)[] = [];
  if (!("ResizeObserver" in globalThis)) {
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      },
    );
  }
  for (const method of ["scrollTo", "scrollIntoView"] as const) {
    if (typeof Element.prototype[method] === "function") continue;
    Object.defineProperty(Element.prototype, method, {
      configurable: true,
      writable: true,
      value: () => undefined,
    });
    restore.push(() => {
      delete (Element.prototype as Partial<Element>)[method];
    });
  }
  return () => restore.forEach((undo) => undo());
};

/**
 * assistant-ui's local runtime puts a failed run's error on the message, which
 * is what a theme shows, and then rethrows it from a promise nothing awaits.
 * A browser only logs that; Vitest would fail the run. For the rest of the
 * test, drop that one rejection; any other still fails the run.
 */
const tolerateUnhandledRejection = (message: string) => {
  const emit = process.emit.bind(process) as (
    event: string,
    ...args: unknown[]
  ) => boolean;
  vi.spyOn(process, "emit").mockImplementation(((
    event: string,
    ...args: unknown[]
  ) => {
    const [reason] = args;
    if (
      event === "unhandledRejection" &&
      reason instanceof Error &&
      reason.message === message
    ) {
      return true;
    }
    return emit(event, ...args);
  }) as typeof process.emit);
};

const ask = async (question: string) => {
  const composer = await screen.findByRole("textbox");
  fireEvent.change(composer, { target: { value: question } });
  const send = await screen.findByRole("button", { name: /送出/ });
  await waitFor(() => expect((send as HTMLButtonElement).disabled).toBe(false));
  fireEvent.click(send);
};

/** The outermost element that names `filename` and none of the others. */
const sourceEntry = (filename: string, others: readonly string[]) =>
  Array.from(document.body.querySelectorAll("*")).find((element) => {
    const text = element.textContent ?? "";
    return text.includes(filename) && others.every((other) => !text.includes(other));
  });

/** Shown either in the page or through the browser's alert dialog. */
const expectShown = async (
  message: string,
  alert: MockInstance<typeof window.alert>,
) => {
  await waitFor(() => {
    const alerted = alert.mock.calls.some(([text]) => text === message);
    expect(alerted || pageText().includes(message)).toBe(true);
  });
};

const answerEvents = (text: string, sources: readonly ChatSource[]) => [
  { type: "text-delta", text },
  { type: "sources", sources },
  { type: "done" },
];

/** Answers the n-th chat request with `answers[n]`, cycling, and keeps every body. */
const stubChat = (answers: readonly (readonly unknown[])[]) => {
  const bodies: unknown[] = [];
  const fetchMock = stubFetch({
    "POST /api/chat": (init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return sseResponse(answers[(bodies.length - 1) % answers.length]!);
    },
  });
  return { fetchMock, bodies };
};

const storedKeys = (): string[] => {
  const keys: string[] = [];
  for (let index = 0; index < localStorage.length; index += 1) {
    const key = localStorage.key(index);
    if (key?.startsWith(CONVERSATION_STORAGE_PREFIX)) keys.push(key);
  }
  return keys.sort();
};

/** Everything kept, as text, to wait for a write to land. */
const storedText = () =>
  storedKeys()
    .map((key) => localStorage.getItem(key))
    .join("\n");

/** The stored list, most recently updated first. */
const storedConversations = (): StoredConversation[] => {
  const raw = localStorage.getItem(CONVERSATION_INDEX_KEY);
  return raw
    ? (JSON.parse(raw) as { conversations: StoredConversation[] }).conversations
    : [];
};

const conversationButtons = (title: string) =>
  screen.queryAllByRole("button", {
    name: (name) => name.includes(title) && !name.includes("刪除"),
  });

const deleteButton = (title: string) =>
  screen.getByRole("button", {
    name: (name) => name.includes("刪除") && name.includes(title),
  });

/** Makes every use of localStorage throw, as a browser blocking site data does. */
const blockStorage = () => {
  const own = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    get() {
      throw new DOMException("The operation is insecure.", "SecurityError");
    },
  });
  return () => {
    if (own) Object.defineProperty(globalThis, "localStorage", own);
    else delete (globalThis as { localStorage?: Storage }).localStorage;
  };
};

/**
 * `count` answered conversations, a minute apart from 2026-01-01 on, so the
 * first is the least recently updated.
 */
const seedConversations = (count: number) => {
  let clock = Date.UTC(2026, 0, 1);
  const store = new ConversationStore(localStorage, () => clock);
  return Array.from({ length: count }, (_, index) => {
    clock += 60_000;
    const id = store.newId();
    const title = `舊對話 ${String(index + 1).padStart(2, "0")}`;
    store.append(id, {
      parentId: null,
      message: {
        id: `question-${index}`,
        role: "user",
        createdAt: new Date(clock),
        content: [{ type: "text", text: title }],
        attachments: [],
        metadata: { custom: {} },
      },
    });
    store.append(id, {
      parentId: `question-${index}`,
      message: {
        id: `answer-${index}`,
        role: "assistant",
        createdAt: new Date(clock),
        content: [{ type: "text", text: `舊回答 ${index + 1}` }],
        status: { type: "complete", reason: "stop" },
        metadata: {
          unstable_state: null,
          unstable_annotations: [],
          unstable_data: [],
          steps: [],
          custom: {},
        },
      },
    });
    return { id, title };
  });
};

/**
 * The product promises every theme keeps, run against the theme's own
 * components. Call it once from a test file in each theme that sets
 * `// @vitest-environment jsdom`.
 */
export function describeThemeContract(harness: ThemeContractHarness): void {
  describe(`theme contract: ${harness.name}`, () => {
    let alert: MockInstance<typeof window.alert>;
    let confirm: MockInstance<typeof window.confirm>;
    let removeLayoutStubs: () => void = () => undefined;
    let unblockStorage: () => void = () => undefined;

    beforeEach(() => {
      if (typeof document === "undefined") {
        throw new Error(
          "the theme contract needs a DOM: add `// @vitest-environment jsdom` to the test file",
        );
      }
      // Conversations from one test must not show up in the next.
      localStorage.clear();
      removeLayoutStubs = installLayoutStubs();
      alert = vi.spyOn(window, "alert").mockImplementation(() => undefined);
      confirm = vi.spyOn(window, "confirm").mockImplementation(() => true);
    });

    afterEach(() => {
      cleanup();
      removeLayoutStubs();
      unblockStorage();
      unblockStorage = () => undefined;
      vi.restoreAllMocks();
      vi.unstubAllGlobals();
      localStorage.clear();
    });

    it("shows every source's filename, and the page of a page source, once the answer completes", async () => {
      stubFetch({
        "POST /api/chat": () =>
          sseResponse([
            { type: "text-delta", text: "保養週期請依照手冊執行。" },
            { type: "sources", sources: [PAGE_SOURCE, TEXT_SOURCE] },
            { type: "done" },
          ]),
      });
      render(harness.renderChat());

      await ask("馬達多久保養一次？");

      await waitFor(() => expect(pageText()).toContain(TEXT_SOURCE.filename));
      const page = sourceEntry(PAGE_SOURCE.filename, [TEXT_SOURCE.filename]);
      const text = sourceEntry(TEXT_SOURCE.filename, [PAGE_SOURCE.filename]);
      expect(page, "page source entry").toBeDefined();
      expect(text, "text source entry").toBeDefined();
      expect(page!.textContent).toMatch(/(?<!\d)37(?!\d)/);
    });

    it("shows core's safe message for an SSE error event", async () => {
      stubFetch({
        "POST /api/chat": () =>
          sseResponse([{ type: "error", message: RAW_SECRET }]),
      });
      tolerateUnhandledRejection(CHAT_ERROR_MESSAGES.generation);
      render(harness.renderChat());

      await ask("今天的產線狀況？");

      await expectShown(CHAT_ERROR_MESSAGES.generation, alert);
      expect(pageText()).not.toContain(RAW_SECRET);
    });

    it("shows core's safe message when the chat request cannot connect", async () => {
      stubFetch({
        "POST /api/chat": () => {
          throw new TypeError(`Failed to fetch: ${RAW_SECRET}`);
        },
      });
      tolerateUnhandledRejection(chatRequestErrorMessage());
      render(harness.renderChat());

      await ask("今天的產線狀況？");

      await expectShown(chatRequestErrorMessage(), alert);
      expect(pageText()).not.toContain(RAW_SECRET);
    });

    it("stops generating while the answer is streaming", async () => {
      let requestSignal: AbortSignal | undefined;
      stubFetch({
        "POST /api/chat": (init) => {
          requestSignal = init?.signal ?? undefined;
          return hangingSseResponse("正在整理保養紀錄", requestSignal!);
        },
      });
      render(harness.renderChat());

      await ask("整理上週的保養紀錄");
      await waitFor(() => expect(pageText()).toContain("正在整理保養紀錄"));
      fireEvent.click(await screen.findByRole("button", { name: /停止/ }));

      await waitFor(() => expect(requestSignal?.aborted).toBe(true));
      await waitFor(() => {
        const stop = screen.queryByRole("button", { name: /停止/ });
        expect(stop === null || (stop as HTMLButtonElement).disabled).toBe(true);
      });
      expect(pageText()).not.toContain(CHAT_ERROR_MESSAGES.interrupted);
    });

    it("shows the chat page title and the suggested questions from /api/profile", async () => {
      stubFetch({});
      render(harness.renderChat());

      await waitFor(() =>
        expect(pageText()).toContain(CONTRACT_PROFILE.pages.chat.title),
      );
      for (const suggestion of CONTRACT_PROFILE.suggestions) {
        expect(pageText()).toContain(suggestion.text);
      }
    });

    it("shows core's safe message and keeps chat usable when /api/profile fails", async () => {
      stubFetch({
        "GET /api/profile": () => new Response(RAW_SECRET, { status: 500 }),
      });
      render(harness.renderChat());

      await expectShown(profileRequestErrorMessage(500), alert);
      expect(pageText()).not.toContain(RAW_SECRET);
      expect(await screen.findByRole("textbox")).toBeDefined();
    });

    const uploadCases = [
      {
        title: "an unsupported extension",
        file: () => new File(["MZ"], "setup.exe"),
      },
      {
        title: "a file over 50 MiB",
        file: () => {
          const file = new File(["%PDF-1.7"], "掃描檔.pdf", {
            type: "application/pdf",
          });
          Object.defineProperty(file, "size", { value: MAX_UPLOAD_BYTES + 1 });
          return file;
        },
      },
    ];

    for (const { title, file } of uploadCases) {
      it(`refuses ${title} with core's message and sends nothing`, async () => {
        const fetchMock = stubFetch({
          "GET /api/documents": () => Response.json([]),
        });
        const upload = file();
        const message = validateUploadFile(upload);
        expect(message).not.toBeNull();
        const { container } = render(harness.renderDocuments());
        const input = container.querySelector<HTMLInputElement>(
          'input[type="file"]',
        );
        expect(input, 'input[type="file"]').not.toBeNull();

        fireEvent.change(input!, { target: { files: [upload] } });

        await expectShown(message!, alert);
        expect(
          sentRequests(fetchMock).filter((request) => !request.startsWith("GET ")),
        ).toEqual([]);
      });
    }

    const FIRST_QUESTION = "馬達多久保養一次？";
    const FIRST_ANSWER = "馬達每三個月保養一次。";
    const FOLLOW_UP = "交接時要注意哪些事項？";
    const FOLLOW_UP_ANSWER = "請依照交接筆記逐項確認。";

    it("keeps both questions, answers and their sources after a reload", async () => {
      stubChat([
        answerEvents(FIRST_ANSWER, [PAGE_SOURCE]),
        answerEvents(FOLLOW_UP_ANSWER, [TEXT_SOURCE]),
      ]);
      render(harness.renderChat());

      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(PAGE_SOURCE.filename));
      await ask(FOLLOW_UP);
      await waitFor(() => expect(pageText()).toContain(TEXT_SOURCE.filename));
      await waitFor(() => expect(storedText()).toContain(FOLLOW_UP_ANSWER));

      cleanup();
      render(harness.renderChat());

      await waitFor(() => {
        for (const text of [
          FIRST_ANSWER,
          FOLLOW_UP_ANSWER,
          PAGE_SOURCE.filename,
          TEXT_SOURCE.filename,
        ]) {
          expect(pageText()).toContain(text);
        }
      });
      const page = sourceEntry(PAGE_SOURCE.filename, [TEXT_SOURCE.filename]);
      expect(page, "page source entry").toBeDefined();
      expect(page!.textContent).toMatch(/(?<!\d)37(?!\d)/);
      expect(sourceEntry(TEXT_SOURCE.filename, [PAGE_SOURCE.filename])).toBeDefined();
    });

    it("sends the same follow-up request after a reload as without one", async () => {
      const { bodies } = stubChat([
        answerEvents(FIRST_ANSWER, [PAGE_SOURCE, TEXT_SOURCE]),
        answerEvents(FOLLOW_UP_ANSWER, [TEXT_SOURCE]),
      ]);

      render(harness.renderChat());
      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(TEXT_SOURCE.filename));
      await ask(FOLLOW_UP);
      await waitFor(() => expect(bodies).toHaveLength(2));
      await waitFor(() => expect(pageText()).toContain(FOLLOW_UP_ANSWER));
      cleanup();
      localStorage.clear();

      render(harness.renderChat());
      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(TEXT_SOURCE.filename));
      await waitFor(() => expect(storedText()).toContain(FIRST_ANSWER));
      cleanup();
      render(harness.renderChat());
      await waitFor(() => expect(pageText()).toContain(FIRST_ANSWER));
      await ask(FOLLOW_UP);
      await waitFor(() => expect(bodies).toHaveLength(4));

      expect(bodies[3]).toEqual(bodies[1]);
      expect(bodies[3]).toMatchObject({
        previousSources: [PAGE_SOURCE, TEXT_SOURCE],
      });
    });

    it("starts, switches, deletes and clears conversations along with their storage", async () => {
      stubChat([
        answerEvents(FIRST_ANSWER, [PAGE_SOURCE]),
        answerEvents(FOLLOW_UP_ANSWER, [TEXT_SOURCE]),
      ]);
      render(harness.renderChat());

      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(PAGE_SOURCE.filename));
      fireEvent.click(await screen.findByRole("button", { name: "新對話" }));
      await waitFor(() => expect(pageText()).not.toContain(FIRST_ANSWER));
      await ask(FOLLOW_UP);
      await waitFor(() => expect(pageText()).toContain(TEXT_SOURCE.filename));
      await waitFor(() =>
        expect(storedConversations().map(({ title }) => title)).toEqual([
          FOLLOW_UP,
          FIRST_QUESTION,
        ]),
      );
      const [second, first] = storedConversations();
      await waitFor(() => expect(storedText()).toContain(FOLLOW_UP_ANSWER));
      expect(storedKeys()).toEqual(
        [
          CONVERSATION_INDEX_KEY,
          conversationMessagesKey(first!.id),
          conversationMessagesKey(second!.id),
        ].sort(),
      );

      await waitFor(() => expect(conversationButtons(FIRST_QUESTION)).toHaveLength(1));
      fireEvent.click(conversationButtons(FIRST_QUESTION)[0]!);
      await waitFor(() => {
        expect(pageText()).toContain(FIRST_ANSWER);
        expect(pageText()).not.toContain(FOLLOW_UP_ANSWER);
      });

      fireEvent.click(deleteButton(FOLLOW_UP));
      await waitFor(() =>
        expect(localStorage.getItem(conversationMessagesKey(second!.id))).toBeNull(),
      );
      await waitFor(() => expect(conversationButtons(FOLLOW_UP)).toHaveLength(0));
      expect(storedConversations().map(({ id }) => id)).toEqual([first!.id]);
      expect(localStorage.getItem(conversationMessagesKey(first!.id))).not.toBeNull();
      expect(pageText()).toContain(FIRST_ANSWER);

      fireEvent.click(screen.getByRole("button", { name: /清除全部對話/ }));
      expect(storedKeys()).toHaveLength(2);
      fireEvent.click(await screen.findByRole("button", { name: /確定清除/ }));
      await waitFor(() => expect(storedKeys()).toEqual([]));
      await waitFor(() => {
        expect(conversationButtons(FIRST_QUESTION)).toHaveLength(0);
        expect(pageText()).not.toContain(FIRST_ANSWER);
      });
      expect(confirm).not.toHaveBeenCalled();
    });

    it("titles a conversation with the first 30 characters of its first question, without a request", async () => {
      // Two characters outside the BMP: 30 characters are 32 UTF-16 units.
      const question =
        "𠀋𡈽三號廠區冷卻水塔今年第三季有哪些異常紀錄、各自如何處理、負責人是誰？";
      const characters = Array.from(question);
      expect(characters.length).toBeGreaterThan(CONVERSATION_TITLE_LENGTH);
      const title = characters.slice(0, CONVERSATION_TITLE_LENGTH).join("");
      const tooLong = characters.slice(0, CONVERSATION_TITLE_LENGTH + 1).join("");
      const { fetchMock } = stubChat([answerEvents(FIRST_ANSWER, [PAGE_SOURCE])]);
      render(harness.renderChat());

      await ask(question);
      await waitFor(() => expect(pageText()).toContain(PAGE_SOURCE.filename));
      await waitFor(() => expect(conversationButtons(title)).toHaveLength(1));

      expect(conversationButtons(tooLong)).toHaveLength(0);
      await waitFor(() =>
        expect(storedConversations().map((stored) => stored.title)).toEqual([title]),
      );
      expect(
        sentRequests(fetchMock).filter((request) => request !== "GET /api/profile"),
      ).toEqual(["POST /api/chat"]);
    });

    it(`deletes the least recently updated conversation when a ${MAX_STORED_CONVERSATIONS + 1}st one starts`, async () => {
      const seeded = seedConversations(MAX_STORED_CONVERSATIONS);
      const oldest = seeded[0]!;
      const next = seeded[1]!;
      stubChat([answerEvents(FIRST_ANSWER, [PAGE_SOURCE])]);
      render(harness.renderChat());
      await waitFor(() => expect(conversationButtons(oldest.title)).toHaveLength(1));

      fireEvent.click(await screen.findByRole("button", { name: "新對話" }));
      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(PAGE_SOURCE.filename));

      await waitFor(() => expect(conversationButtons(oldest.title)).toHaveLength(0));
      expect(localStorage.getItem(conversationMessagesKey(oldest.id))).toBeNull();
      expect(localStorage.getItem(conversationMessagesKey(next.id))).not.toBeNull();
      expect(conversationButtons(next.title)).toHaveLength(1);
      await waitFor(() => {
        const stored = storedConversations();
        expect(stored).toHaveLength(MAX_STORED_CONVERSATIONS);
        expect(stored[0]!.title).toBe(FIRST_QUESTION);
        expect(stored.map(({ id }) => id)).not.toContain(oldest.id);
      });
    });

    it("ignores and removes corrupt stored history, and still answers", async () => {
      localStorage.setItem(CONVERSATION_INDEX_KEY, '{"version":1,"conversations":[');
      localStorage.setItem(conversationMessagesKey("corrupt"), "{not json");
      stubChat([answerEvents(FIRST_ANSWER, [PAGE_SOURCE])]);
      render(harness.renderChat());

      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(PAGE_SOURCE.filename));

      await waitFor(() => expect(storedConversations()).toHaveLength(1));
      expect(localStorage.getItem(conversationMessagesKey("corrupt"))).toBeNull();
      expect(pageText()).not.toContain(CONVERSATION_NOT_SAVED_MESSAGE);
    });

    it("ignores and removes history of an unknown version, and still answers", async () => {
      localStorage.setItem(
        CONVERSATION_INDEX_KEY,
        JSON.stringify({
          version: 2,
          conversations: [
            { id: "future", title: "新版本留下的對話", createdAt: 1, updatedAt: 2 },
          ],
        }),
      );
      localStorage.setItem(
        conversationMessagesKey("future"),
        JSON.stringify({ version: 2, headId: "m1", messages: [] }),
      );
      stubChat([answerEvents(FIRST_ANSWER, [PAGE_SOURCE])]);
      render(harness.renderChat());

      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(PAGE_SOURCE.filename));

      expect(pageText()).not.toContain("新版本留下的對話");
      expect(localStorage.getItem(conversationMessagesKey("future"))).toBeNull();
      await waitFor(() =>
        expect(storedConversations().map(({ title }) => title)).toEqual([
          FIRST_QUESTION,
        ]),
      );
    });

    it("answers without saving, and says so, when the browser blocks storage", async () => {
      unblockStorage = blockStorage();
      stubChat([answerEvents(FIRST_ANSWER, [PAGE_SOURCE])]);
      render(harness.renderChat());

      await waitFor(() =>
        expect(pageText()).toContain(CONVERSATION_NOT_SAVED_MESSAGE),
      );
      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(PAGE_SOURCE.filename));
      expect(pageText()).toContain(FIRST_ANSWER);
      expect(pageText()).toContain(CONVERSATION_NOT_SAVED_MESSAGE);
    });

    it("answers without saving, and says so, when storage is full", async () => {
      vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
        throw new DOMException("The quota has been exceeded.", "QuotaExceededError");
      });
      stubChat([answerEvents(FIRST_ANSWER, [PAGE_SOURCE])]);
      render(harness.renderChat());

      expect(pageText()).not.toContain(CONVERSATION_NOT_SAVED_MESSAGE);
      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(PAGE_SOURCE.filename));
      await waitFor(() =>
        expect(pageText()).toContain(CONVERSATION_NOT_SAVED_MESSAGE),
      );
      expect(pageText()).toContain(FIRST_ANSWER);
      expect(storedKeys()).toEqual([]);
    });

    it("declares --theme-* colours whose pairs meet WCAG AA", () => {
      const css = harness.stylesheets
        .map((path) => readFileSync(new URL(path, harness.baseUrl), "utf8"))
        .join("\n");
      for (const { foreground, background, ratio } of themeContrasts(css)) {
        expect(
          ratio,
          `${foreground} on ${background}: ${ratio.toFixed(2)}:1`,
        ).toBeGreaterThanOrEqual(WCAG_AA_TEXT_CONTRAST);
      }
    });
  });
}
