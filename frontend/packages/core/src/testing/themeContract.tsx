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
import { MAX_UPLOAD_BYTES, validateUploadFile } from "../DocumentModel";
import { profileRequestErrorMessage } from "../useProfile";
import type { Profile } from "../useProfile";
import { themeContrasts, WCAG_AA_TEXT_CONTRAST } from "./themeColors";

export type ThemeContractHarness = {
  /** Shown in the test names. */
  name: string;
  /**
   * The theme's chat surface with its own runtime wiring, usually the whole
   * app opened on chat. The contract drives it like a person: it types into
   * the only textbox, presses the button named 送出, and stops with the button
   * named 停止.
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

/**
 * The product promises every theme keeps, run against the theme's own
 * components. Call it once from a test file in each theme that sets
 * `// @vitest-environment jsdom`.
 */
export function describeThemeContract(harness: ThemeContractHarness): void {
  describe(`theme contract: ${harness.name}`, () => {
    let alert: MockInstance<typeof window.alert>;
    let removeLayoutStubs: () => void = () => undefined;

    beforeEach(() => {
      if (typeof document === "undefined") {
        throw new Error(
          "the theme contract needs a DOM: add `// @vitest-environment jsdom` to the test file",
        );
      }
      removeLayoutStubs = installLayoutStubs();
      alert = vi.spyOn(window, "alert").mockImplementation(() => undefined);
      vi.spyOn(window, "confirm").mockImplementation(() => true);
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
