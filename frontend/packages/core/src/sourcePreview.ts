import { useAuiState } from "@assistant-ui/react";
import { useEffect, useMemo, useState } from "react";
import { sourceData } from "./BackendAdapter";
import type { ChatSource } from "./BackendAdapter";

/** Where a cited document can be downloaded exactly as it was uploaded. */
export const documentFileUrl = (documentId: string): string =>
  `/api/documents/${encodeURIComponent(documentId)}/file`;

/**
 * Where what a source cited is read: a catalog source's document card, a
 * page source's page image, or a text source's passage.
 */
const previewUrl = (source: ChatSource): string => {
  const part =
    source.catalog === true ? "catalog" : source.kind === "page" ? "image" : "text";
  return `/api/documents/${encodeURIComponent(source.documentId)}/units/${source.unitId}/${part}`;
};

export function sourcePreviewErrorMessage(status?: number): string {
  if (status === undefined) {
    return "連不上伺服器，無法載入這個來源，請確認網路後再試。";
  }
  // The document was deleted or processed again, which gives it new units.
  if (status === 404) return "找不到這個來源，文件可能已刪除或重新處理。";
  return "無法載入這個來源，請稍後再試。";
}

/**
 * What a source cited, as far as it has loaded: a page source's page image
 * as an object URL, or the text a text or catalog source cited.
 */
export type SourcePreview =
  | { status: "loading" }
  | { status: "image"; url: string }
  | { status: "text"; text: string }
  | { status: "error"; message: string };

const LOADING: SourcePreview = { status: "loading" };

/**
 * Loads what `source` cited, and nothing while it is null. Changing the
 * source or unmounting stops the request and releases the image's object
 * URL. Every failure, a response of the wrong type included, ends in a safe
 * message.
 */
export function useSourcePreview(source: ChatSource | null): SourcePreview {
  const url = source === null ? null : previewUrl(source);
  const image = source?.kind === "page" && source.catalog !== true;
  const [loaded, setLoaded] = useState<{ url: string; preview: SourcePreview } | null>(
    null,
  );

  useEffect(() => {
    if (url === null) return;
    const controller = new AbortController();
    let objectUrl: string | null = null;
    const settle = (preview: SourcePreview) => {
      if (!controller.signal.aborted) setLoaded({ url, preview });
    };
    const failed = (status?: number) =>
      settle({ status: "error", message: sourcePreviewErrorMessage(status) });

    void (async () => {
      try {
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) return failed(res.status);
        if (image) {
          const image = await res.blob();
          if (!image.type.startsWith("image/")) return failed(502);
          if (controller.signal.aborted) return;
          objectUrl = URL.createObjectURL(image);
          settle({ status: "image", url: objectUrl });
        } else {
          const type = res.headers.get("Content-Type") ?? "";
          if (!type.startsWith("text/plain")) return failed(502);
          settle({ status: "text", text: await res.text() });
        }
      } catch {
        failed();
      }
    })();

    return () => {
      controller.abort();
      if (objectUrl !== null) URL.revokeObjectURL(objectUrl);
      // Opening the same source again loads it again: never its released
      // image, nor content that may have been deleted since.
      setLoaded(null);
    };
  }, [url, image]);

  // A preview loaded for an earlier source is never shown for this one.
  return loaded !== null && loaded.url === url ? loaded.preview : LOADING;
}

/**
 * The sources of the message being rendered, validated as the adapter
 * validates them; empty until they arrive. Call it inside a message.
 */
export function useMessageSources(): readonly ChatSource[] {
  const part = useAuiState((state) =>
    state.message.content.find(
      (part) => part.type === "data" && part.name === "sources",
    ),
  );
  return useMemo(
    () => (part?.type === "data" ? sourceData(part.data) : null) ?? [],
    [part],
  );
}

/**
 * The element `remarkSourceMarkers` renders a marker as. Its `data-source-id`
 * property names the source, so a theme maps the element to its own component.
 */
export const SOURCE_MARKER_ELEMENT = "button";

/** `[S1]`: the inline citation the answer prompt asks the model to write. */
const SOURCE_MARKER = /\[(S[1-9]\d*)\]/g;

/** Inside these the text is code, a link or raw markup, never a citation. */
const NOT_CITED = new Set(["code", "inlineCode", "link", "linkReference", "definition", "html"]);

type MarkdownNode = {
  type: string;
  value?: string;
  children?: MarkdownNode[];
  data?: Record<string, unknown>;
};

const markerNodes = (text: MarkdownNode): MarkdownNode[] => {
  const value = text.value ?? "";
  const nodes: MarkdownNode[] = [];
  let end = 0;
  for (const match of value.matchAll(SOURCE_MARKER)) {
    if (match.index > end) {
      nodes.push({ type: "text", value: value.slice(end, match.index) });
    }
    nodes.push({
      type: "sourceMarker",
      data: {
        hName: SOURCE_MARKER_ELEMENT,
        hProperties: { type: "button", dataSourceId: match[1] },
      },
      children: [{ type: "text", value: match[0] }],
    });
    end = match.index + match[0].length;
  }
  if (end === 0) return [text];
  if (end < value.length) nodes.push({ type: "text", value: value.slice(end) });
  return nodes;
};

const markCitations = (node: MarkdownNode): void => {
  if (node.children === undefined || NOT_CITED.has(node.type)) return;
  node.children = node.children.flatMap((child) => {
    if (child.type === "text") return markerNodes(child);
    markCitations(child);
    return [child];
  });
};

/**
 * A remark plugin that turns every `[S1]` in an answer into a
 * `SOURCE_MARKER_ELEMENT` element carrying `data-source-id="S1"`. Markers in
 * code, links and raw markup stay text.
 */
export function remarkSourceMarkers() {
  return (tree: MarkdownNode) => markCitations(tree);
}
