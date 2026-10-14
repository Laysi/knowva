// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/** Every recognition the page started; a test speaks into the last one. */
const started: FakeRecognition[] = [];
const recognition = () => started.at(-1);

class FakeRecognition extends EventTarget {
  lang = "";
  continuous = true;
  interimResults = false;
  start() {
    started.push(this);
    this.dispatchEvent(new Event("start"));
  }
  stop() {
    this.dispatchEvent(new Event("end"));
  }
  abort() {
    this.dispatchEvent(new Event("end"));
  }
  say(transcript: string, isFinal: boolean) {
    const result = Object.assign([{ transcript }], { isFinal });
    this.dispatchEvent(
      Object.assign(new Event("result"), { resultIndex: 0, results: [result] }),
    );
  }
}

beforeEach(() => {
  vi.resetModules();
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  Element.prototype.scrollTo = () => undefined;
  vi.stubGlobal("webkitSpeechRecognition", FakeRecognition);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) =>
      String(input) === "/api/conversations" || String(input) === "/api/documents"
        ? Response.json([])
        : new Response(null, { status: 404 }),
    ),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  started.length = 0;
});

describe("voice input", () => {
  it("listens in zh-TW and puts what was said into the composer", async () => {
    // The runtime reads the speech API when it loads, so load it after the stub.
    const { default: App } = await import("./App");
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "語音輸入" }));
    await waitFor(() => expect(recognition()).toBeDefined());
    expect(recognition()!.lang).toBe("zh-TW");
    expect(recognition()!.continuous).toBe(false);
    const stop = await screen.findByRole("button", { name: "停止語音輸入" });

    act(() => recognition()!.say("倉儲安全", false));
    await waitFor(() =>
      expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("倉儲安全"),
    );
    act(() => recognition()!.say("倉儲安全規則", true));
    fireEvent.click(stop);

    await screen.findByRole("button", { name: "語音輸入" });
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("倉儲安全規則");
  });
});
