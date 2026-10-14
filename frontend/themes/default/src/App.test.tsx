// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const ID = "00000000-0000-4000-8000-000000000001";
const TITLE = "倉儲安全規則的重點摘要";
const OLDER_ID = "00000000-0000-4000-8000-000000000002";
const OLDER_TITLE = "Q3 設備保養排程有沒有衝突";

const minutesAgo = (minutes: number) =>
  new Date(Date.now() - minutes * 60_000).toISOString();
const summary = (id: string, title: string, updatedAt: string) => ({
  id,
  title,
  createdAt: updatedAt,
  updatedAt,
});

type Route = (init?: RequestInit) => Response | Promise<Response>;
let routes: Record<string, Route>;

beforeEach(() => {
  // Thread layout needs these; jsdom has neither.
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  Element.prototype.scrollTo = () => undefined;
  const conversation = summary(ID, TITLE, minutesAgo(5));
  routes = {
    "GET /api/conversations": () => Response.json([conversation]),
    [`GET /api/conversations/${ID}`]: () => Response.json(conversation),
    [`GET /api/conversations/${ID}/messages`]: () => Response.json([]),
    "GET /api/documents": () => Response.json([]),
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const route = routes[`${init?.method ?? "GET"} ${String(input)}`];
      return route ? route(init) : new Response(null, { status: 404 });
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const menuButton = () => screen.getByRole("button", { name: "開啟選單" });
const page = () => menuButton().closest(".content")!;
const conversationNamed = (title: string) =>
  screen.getByRole("button", {
    name: (name) => name.includes(title) && !name.includes("刪除"),
  });
const conversation = () => conversationNamed(TITLE);
const listedTitles = () =>
  Array.from(document.querySelectorAll(".history-title"), (title) => title.textContent);

const sse = (events: readonly unknown[]) =>
  new Response(
    events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    { headers: { "Content-Type": "text/event-stream" } },
  );

describe("navigation rail", () => {
  it("lists a conversation in 近期對話 with how long ago it changed", async () => {
    render(<App />);

    await waitFor(() => expect(conversation().textContent).toContain("5 分鐘前"));
    expect(conversation().closest("section")?.textContent).toContain("近期對話");
  });

  it("opens as a drawer that makes the page inert, and Escape gives focus back", async () => {
    render(<App />);
    const menu = menuButton();
    expect(menu.getAttribute("aria-expanded")).toBe("false");

    fireEvent.click(menu);
    expect(menu.getAttribute("aria-expanded")).toBe("true");
    expect(page().hasAttribute("inert")).toBe(true);
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "關閉選單" }),
    );

    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(document.activeElement).toBe(menu));
    expect(menu.getAttribute("aria-expanded")).toBe("false");
    expect(page().hasAttribute("inert")).toBe(false);
  });

  it("closes once a conversation or a page is chosen", async () => {
    render(<App />);
    const menu = menuButton();

    fireEvent.click(menu);
    fireEvent.click(await waitFor(conversation));
    expect(menu.getAttribute("aria-expanded")).toBe("false");

    fireEvent.click(menu);
    fireEvent.click(screen.getByRole("button", { name: "文件" }));
    expect(menu.getAttribute("aria-expanded")).toBe("false");
    // Documents has no new-conversation button in the header.
    expect(screen.queryByRole("button", { name: "開始新對話" })).toBeNull();

    // A conversation chosen from documents goes back to the chat.
    fireEvent.click(menu);
    fireEvent.click(conversation());
    expect(screen.getByRole("button", { name: "開始新對話" })).toBeDefined();
  });

  it("moves a conversation asked in to the top, as 剛剛", async () => {
    const newer = summary(ID, TITLE, minutesAgo(5));
    const older = summary(OLDER_ID, OLDER_TITLE, minutesAgo(3 * 24 * 60));
    routes["GET /api/conversations"] = () => Response.json([newer, older]);
    routes[`GET /api/conversations/${OLDER_ID}`] = () => Response.json(older);
    routes[`GET /api/conversations/${OLDER_ID}/messages`] = () => Response.json([]);
    routes[`POST /api/conversations/${OLDER_ID}/messages`] = () =>
      Response.json({ conversation: { ...older, updatedAt: new Date().toISOString() } });
    routes["POST /api/chat"] = () =>
      sse([
        { type: "text-delta", text: "回答" },
        { type: "sources", sources: [] },
        { type: "done" },
      ]);
    render(<App />);

    await waitFor(() => expect(listedTitles()).toEqual([TITLE, OLDER_TITLE]));
    expect(conversationNamed(OLDER_TITLE).textContent).toContain("3 天前");
    fireEvent.click(conversationNamed(OLDER_TITLE));
    const composer = await screen.findByRole("textbox");
    fireEvent.change(composer, { target: { value: "追問" } });
    const send = await screen.findByRole("button", { name: "送出" });
    await waitFor(() => expect((send as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(send);

    await waitFor(() => expect(listedTitles()).toEqual([OLDER_TITLE, TITLE]));
    expect(conversationNamed(OLDER_TITLE).textContent).toContain("剛剛");
  });
});

const ask = async (question: string) => {
  const composer = await screen.findByRole("textbox");
  fireEvent.change(composer, { target: { value: question } });
  const send = await screen.findByRole("button", { name: "送出" });
  await waitFor(() => expect((send as HTMLButtonElement).disabled).toBe(false));
  fireEvent.click(send);
};

describe("composer", () => {
  it("stops thinking once an answer that has not started is stopped", async () => {
    routes[`POST /api/conversations/${ID}/messages`] = () =>
      Response.json({ conversation: summary(ID, TITLE, new Date().toISOString()) });
    // Like fetch, the request rejects with the abort reason once stopped.
    routes["POST /api/chat"] = (init) =>
      new Promise((_, reject) => {
        const signal = init?.signal;
        signal?.addEventListener("abort", () => reject(signal.reason));
      });
    render(<App />);

    await ask("倉儲有哪些規定？");
    await screen.findByLabelText("思考中");
    fireEvent.click(screen.getByRole("button", { name: "停止生成" }));

    await screen.findByText("已停止回覆");
    expect(screen.queryByLabelText("思考中")).toBeNull();
  });

  it("offers no voice input where the browser has no speech recognition", async () => {
    render(<App />);

    await screen.findByRole("textbox");
    expect(screen.queryByRole("button", { name: "語音輸入" })).toBeNull();
  });
});
