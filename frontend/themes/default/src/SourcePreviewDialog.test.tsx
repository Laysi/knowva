// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatSource } from "@knowva/core";
import { SourcePreviewDialog } from "./Thread";

const SOURCE: ChatSource = {
  sourceId: "S2",
  documentId: "notes",
  unitId: 202,
  unitIndex: 1,
  filename: "維運交接筆記.md",
  pageNumber: null,
  kind: "text",
};

const dialog = () => document.querySelector("dialog")!;

describe("SourcePreviewDialog", () => {
  beforeEach(() => {
    // jsdom has no modal dialogs. This `close` never fires the close event,
    // as a hidden tab may not: closing must not wait for it.
    for (const [name, open] of [
      ["showModal", true],
      ["close", false],
    ] as const) {
      Object.defineProperty(HTMLDialogElement.prototype, name, {
        configurable: true,
        value(this: HTMLDialogElement) {
          this.open = open;
        },
      });
    }
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response("交接時確認備品庫存。", {
            headers: { "Content-Type": "text/plain; charset=utf-8" },
          }),
      ),
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    delete (HTMLDialogElement.prototype as Partial<HTMLDialogElement>).showModal;
    delete (HTMLDialogElement.prototype as Partial<HTMLDialogElement>).close;
  });

  it("opens for a source and closes once the source is cleared", async () => {
    const { rerender } = render(<SourcePreviewDialog source={SOURCE} onClose={() => undefined} />);

    expect(dialog().open).toBe(true);
    expect(await screen.findByText("交接時確認備品庫存。")).toBeDefined();
    expect(
      document.querySelector('a[href="/api/documents/notes/file"]')?.textContent,
    ).toBe("下載原檔");

    rerender(<SourcePreviewDialog source={null} onClose={() => undefined} />);
    expect(dialog().open).toBe(false);
    expect(document.body.textContent).not.toContain(SOURCE.filename);
  });

  it.each([
    ["the close button", () => fireEvent.click(screen.getByRole("button", { name: "關閉" }))],
    ["Escape", () => fireEvent(dialog(), new Event("cancel", { cancelable: true }))],
    ["a click on the backdrop", () => fireEvent.click(dialog())],
  ])("asks to close from %s without waiting for the close event", (_, closeBy) => {
    const onClose = vi.fn();
    render(<SourcePreviewDialog source={SOURCE} onClose={onClose} />);

    closeBy();

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("stays open for a click inside the panel", () => {
    const onClose = vi.fn();
    render(<SourcePreviewDialog source={SOURCE} onClose={onClose} />);

    fireEvent.click(screen.getByRole("heading", { name: /維運交接筆記\.md/ }));

    expect(onClose).not.toHaveBeenCalled();
  });
});
