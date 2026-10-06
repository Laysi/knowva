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
  CONVERSATION_NOT_SAVED_MESSAGE,
  CONVERSATION_TITLE_LENGTH,
  conversationRequestErrorMessage,
} from "../conversationApi";
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
   * stands for a reload; the contract's in-memory copy of the
   * `/api/conversations` routes keeps what was saved. The contract drives it
   * like a person: it types into the only textbox, presses the button named
   * 送出, and stops with the button named 停止. For conversation history it
   * presses the button named 新對話, opens a conversation with the button named
   * by its title, deletes one with the button whose name holds 刪除 and the
   * title, and clears them all with 清除全部對話 followed by 確定清除 in the page.
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

type StoredConversation = {
  id: string;
  title: string | null;
  createdAt: string;
  updatedAt: string;
  messages: { parentId: string | null; message: Record<string, unknown> }[];
};

const MOCK_CONVERSATION_LIMIT = 50;
const CONVERSATION_ROUTE = /^\/api\/conversations(?:\/([^/]+)(\/messages)?)?$/;

/**
 * The backend's conversation routes, in memory, for one test: one browser's
 * list, titled from the first question, with saves that need their parent
 * saved first. A re-render talks to the same copy, so it stands for a reload.
 */
class ConversationServer {
  readonly conversations: StoredConversation[] = [];
  /** When set, answers every message save instead. */
  saveFailure: (() => Response) | null = null;
  /** When set, answers every request to start a conversation instead. */
  createFailure: (() => Response) | null = null;
  private clock = Date.UTC(2026, 0, 1);
  private created = 0;

  /** Most recently updated first. */
  list(): StoredConversation[] {
    return [...this.conversations].sort((a, b) =>
      b.updatedAt.localeCompare(a.updatedAt),
    );
  }

  /** Everything saved, as text, to wait for a save to land. */
  text(): string {
    return JSON.stringify(this.conversations);
  }

  handle(method: string, path: string, init: RequestInit | undefined): Response | null {
    const match = CONVERSATION_ROUTE.exec(path);
    if (!match) return null;
    const [, id, messages] = match;
    if (id === undefined) {
      if (method === "GET") return Response.json(this.list().map(summary));
      if (method === "POST") return this.create();
      if (method === "DELETE") {
        this.conversations.length = 0;
        return new Response(null, { status: 204 });
      }
      return new Response(null, { status: 405 });
    }
    const conversation = this.conversations.find((stored) => stored.id === id);
    if (!conversation) return new Response("找不到這串對話。", { status: 404 });
    if (messages === undefined) {
      if (method === "GET") return Response.json(summary(conversation));
      if (method === "DELETE") {
        this.conversations.splice(this.conversations.indexOf(conversation), 1);
        return new Response(null, { status: 204 });
      }
      return new Response(null, { status: 405 });
    }
    if (method === "GET") {
      return Response.json(
        conversation.messages.map(({ parentId, message }) => ({ parentId, message })),
      );
    }
    if (method === "POST") return this.save(conversation, init);
    return new Response(null, { status: 405 });
  }

  private tick(): string {
    this.clock += 1000;
    return new Date(this.clock).toISOString();
  }

  private create(): Response {
    if (this.createFailure) return this.createFailure();
    this.created += 1;
    const now = this.tick();
    const conversation: StoredConversation = {
      id: `00000000-0000-4000-8000-${String(this.created).padStart(12, "0")}`,
      title: null,
      createdAt: now,
      updatedAt: now,
      messages: [],
    };
    this.conversations.push(conversation);
    const removed = this.list().slice(MOCK_CONVERSATION_LIMIT);
    for (const old of removed) {
      this.conversations.splice(this.conversations.indexOf(old), 1);
    }
    return Response.json(
      {
        conversation: summary(conversation),
        removedIds: removed.map(({ id }) => id),
      },
      { status: 201 },
    );
  }

  private save(conversation: StoredConversation, init: RequestInit | undefined): Response {
    if (this.saveFailure) return this.saveFailure();
    const { parentId, message } = JSON.parse(String(init?.body)) as {
      parentId: string | null;
      message: Record<string, unknown>;
    };
    const saved = conversation.messages;
    if (parentId !== null && !saved.some((entry) => entry.message.id === parentId)) {
      return new Response("前一則訊息尚未保存，這則訊息無法保存。", { status: 409 });
    }
    const at = saved.findIndex((entry) => entry.message.id === message.id);
    if (at >= 0) saved[at] = { parentId, message };
    else saved.push({ parentId, message });
    if (conversation.title === null && message.role === "user") {
      const question = (message.content as { type: string; text?: string }[])
        .flatMap((part) => (part.type === "text" && part.text ? [part.text] : []))
        .join("\n");
      const title = Array.from(question.trim().replace(/\s+/g, " "))
        .slice(0, CONVERSATION_TITLE_LENGTH)
        .join("");
      conversation.title = title || null;
    }
    conversation.updatedAt = this.tick();
    return Response.json({ conversation: summary(conversation) });
  }
}

const summary = ({ id, title, createdAt, updatedAt }: StoredConversation) => ({
  id,
  title,
  createdAt,
  updatedAt,
});

/**
 * Unlisted routes answer 404. `GET /api/profile` answers the contract profile
 * and `/api/conversations` goes to `conversations` unless a test lists the
 * exact route.
 */
const stubFetch = (
  handlers: Record<string, Handler>,
  conversations: ConversationServer = new ConversationServer(),
) => {
  const routes: Record<string, Handler> = {
    "GET /api/profile": () => Response.json(CONTRACT_PROFILE),
    ...handlers,
  };
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const key = `${method} ${String(input)}`;
    const handler = routes[key];
    if (handler) return handler(init);
    return (
      conversations.handle(method, String(input), init) ??
      new Response(null, { status: 404 })
    );
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
const stubChat = (
  answers: readonly (readonly unknown[])[],
  conversations: ConversationServer = new ConversationServer(),
  handlers: Record<string, Handler> = {},
) => {
  const bodies: unknown[] = [];
  const fetchMock = stubFetch(
    {
      "POST /api/chat": (init) => {
        bodies.push(JSON.parse(String(init?.body)));
        return sseResponse(answers[(bodies.length - 1) % answers.length]!);
      },
      ...handlers,
    },
    conversations,
  );
  return { fetchMock, bodies, conversations };
};

const conversationButtons = (title: string) =>
  screen.queryAllByRole("button", {
    name: (name) => name.includes(title) && !name.includes("刪除"),
  });

const deleteButton = (title: string) =>
  screen.getByRole("button", {
    name: (name) => name.includes("刪除") && name.includes(title),
  });

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

    beforeEach(() => {
      if (typeof document === "undefined") {
        throw new Error(
          "the theme contract needs a DOM: add `// @vitest-environment jsdom` to the test file",
        );
      }
      removeLayoutStubs = installLayoutStubs();
      alert = vi.spyOn(window, "alert").mockImplementation(() => undefined);
      confirm = vi.spyOn(window, "confirm").mockImplementation(() => true);
    });

    afterEach(() => {
      cleanup();
      removeLayoutStubs();
      vi.restoreAllMocks();
      vi.unstubAllGlobals();
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
      const { conversations } = stubChat([
        answerEvents(FIRST_ANSWER, [PAGE_SOURCE]),
        answerEvents(FOLLOW_UP_ANSWER, [TEXT_SOURCE]),
      ]);
      render(harness.renderChat());

      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(PAGE_SOURCE.filename));
      await ask(FOLLOW_UP);
      await waitFor(() => expect(pageText()).toContain(TEXT_SOURCE.filename));
      await waitFor(() => expect(conversations.text()).toContain(FOLLOW_UP_ANSWER));

      cleanup();
      render(harness.renderChat());

      await waitFor(() => {
        for (const text of [
          FIRST_QUESTION,
          FIRST_ANSWER,
          FOLLOW_UP,
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
      const { bodies, conversations } = stubChat([
        answerEvents(FIRST_ANSWER, [PAGE_SOURCE, TEXT_SOURCE]),
        answerEvents(FOLLOW_UP_ANSWER, [TEXT_SOURCE]),
      ]);

      render(harness.renderChat());
      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(TEXT_SOURCE.filename));
      await ask(FOLLOW_UP);
      await waitFor(() => expect(bodies).toHaveLength(2));
      await waitFor(() => expect(pageText()).toContain(FOLLOW_UP_ANSWER));
      await waitFor(() => expect(conversations.text()).toContain(FOLLOW_UP_ANSWER));
      cleanup();
      // Start over from an empty history, so nothing is reopened yet.
      conversations.conversations.length = 0;

      render(harness.renderChat());
      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(TEXT_SOURCE.filename));
      await waitFor(() => expect(conversations.text()).toContain(FIRST_ANSWER));
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

    it("starts, switches, deletes and clears conversations on the server", async () => {
      const { conversations } = stubChat([
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
      await waitFor(() => {
        expect(conversations.list().map(({ title }) => title)).toEqual([
          FOLLOW_UP,
          FIRST_QUESTION,
        ]);
        expect(conversations.text()).toContain(FOLLOW_UP_ANSWER);
      });
      const [second, first] = conversations.list();

      await waitFor(() => expect(conversationButtons(FIRST_QUESTION)).toHaveLength(1));
      fireEvent.click(conversationButtons(FIRST_QUESTION)[0]!);
      await waitFor(() => {
        expect(pageText()).toContain(FIRST_ANSWER);
        expect(pageText()).not.toContain(FOLLOW_UP_ANSWER);
      });

      fireEvent.click(deleteButton(FOLLOW_UP));
      await waitFor(() =>
        expect(conversations.list().map(({ id }) => id)).toEqual([first!.id]),
      );
      await waitFor(() => expect(conversationButtons(FOLLOW_UP)).toHaveLength(0));
      expect(conversations.text()).not.toContain(second!.id);
      expect(pageText()).toContain(FIRST_ANSWER);

      fireEvent.click(screen.getByRole("button", { name: /清除全部對話/ }));
      expect(conversations.list()).toHaveLength(1);
      fireEvent.click(await screen.findByRole("button", { name: /確定清除/ }));
      await waitFor(() => expect(conversations.list()).toEqual([]));
      await waitFor(() => {
        expect(conversationButtons(FIRST_QUESTION)).toHaveLength(0);
        expect(pageText()).not.toContain(FIRST_ANSWER);
      });
      expect(confirm).not.toHaveBeenCalled();
    });

    it("shows the title the server made from the first 30 characters of the first question, without a model request", async () => {
      // Two characters outside the BMP: 30 characters are 32 UTF-16 units.
      const question =
        "𠀋𡈽三號廠區冷卻水塔今年第三季有哪些異常紀錄、各自如何處理、負責人是誰？";
      const characters = Array.from(question);
      expect(characters.length).toBeGreaterThan(CONVERSATION_TITLE_LENGTH);
      const title = characters.slice(0, CONVERSATION_TITLE_LENGTH).join("");
      const tooLong = characters.slice(0, CONVERSATION_TITLE_LENGTH + 1).join("");
      const { fetchMock, conversations } = stubChat([
        answerEvents(FIRST_ANSWER, [PAGE_SOURCE]),
      ]);
      render(harness.renderChat());

      await ask(question);
      await waitFor(() => expect(pageText()).toContain(PAGE_SOURCE.filename));
      await waitFor(() => expect(conversationButtons(title)).toHaveLength(1));

      expect(conversationButtons(tooLong)).toHaveLength(0);
      expect(conversations.list().map((stored) => stored.title)).toEqual([title]);
      expect(
        sentRequests(fetchMock).filter(
          (request) =>
            request !== "GET /api/profile" &&
            !request.includes(" /api/conversations"),
        ),
      ).toEqual(["POST /api/chat"]);
    });

    const saveFailures = [
      {
        title: "the server cannot save a message",
        fail: (server: ConversationServer) => {
          server.saveFailure = () => new Response(RAW_SECRET, { status: 500 });
        },
      },
      {
        title: "a message is over the server's limit",
        fail: (server: ConversationServer) => {
          server.saveFailure = () => new Response(RAW_SECRET, { status: 413 });
        },
      },
      {
        title: "the server cannot start the conversation",
        fail: (server: ConversationServer) => {
          server.createFailure = () => new Response(RAW_SECRET, { status: 503 });
        },
      },
    ];

    for (const { title, fail } of saveFailures) {
      it(`keeps answering and says the conversation is not saved when ${title}`, async () => {
        const conversations = new ConversationServer();
        fail(conversations);
        const { bodies } = stubChat(
          [
            answerEvents(FIRST_ANSWER, [PAGE_SOURCE]),
            answerEvents(FOLLOW_UP_ANSWER, [TEXT_SOURCE]),
          ],
          conversations,
        );
        render(harness.renderChat());

        await ask(FIRST_QUESTION);
        await waitFor(() => expect(pageText()).toContain(PAGE_SOURCE.filename));
        await waitFor(() =>
          expect(pageText()).toContain(CONVERSATION_NOT_SAVED_MESSAGE),
        );
        await ask(FOLLOW_UP);
        await waitFor(() => expect(pageText()).toContain(TEXT_SOURCE.filename));

        expect(bodies).toHaveLength(2);
        expect(pageText()).toContain(FIRST_ANSWER);
        expect(pageText()).toContain(FOLLOW_UP_ANSWER);
        expect(pageText()).toContain(CONVERSATION_NOT_SAVED_MESSAGE);
        expect(pageText()).not.toContain(RAW_SECRET);
      });
    }

    it("shows core's safe message and keeps chat usable when the conversation list cannot load", async () => {
      const { conversations } = stubChat([answerEvents(FIRST_ANSWER, [PAGE_SOURCE])], undefined, {
        "GET /api/conversations": () => new Response(RAW_SECRET, { status: 500 }),
      });
      render(harness.renderChat());

      await expectShown(conversationRequestErrorMessage("list", 500), alert);
      expect(pageText()).not.toContain(RAW_SECRET);
      await ask(FIRST_QUESTION);
      await waitFor(() => expect(pageText()).toContain(PAGE_SOURCE.filename));

      expect(pageText()).toContain(FIRST_ANSWER);
      expect(pageText()).not.toContain(RAW_SECRET);
      await waitFor(() => expect(conversations.text()).toContain(FIRST_ANSWER));
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
