import type { ChatModelAdapter } from "@assistant-ui/react";

export const BackendAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: messages.map((m) => ({
          role: m.role,
          content: m.content
            .filter((c) => c.type === "text")
            .map((c: any) => c.text)
            .join("\n"),
        })),
      }),
      signal: abortSignal,
    });
    if (!res.ok || !res.body) throw new Error(`API error: ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let text = "";
    let sources: { documentId: string; filename: string }[] = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const events = buf.split("\n\n");
      buf = events.pop() ?? "";
      for (const ev of events) {
        const line = ev.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const payload = JSON.parse(line.slice(6));
        if (payload.type === "text-delta") {
          text += payload.text;
          yield { content: [{ type: "text", text }] };
        } else if (payload.type === "sources") {
          sources = payload.sources ?? [];
        } else if (payload.type === "error") {
          throw new Error(payload.message);
        }
      }
    }
    if (sources.length > 0) {
      const list = sources
        .map((s, i) => `${i + 1}. ${s.filename}`)
        .join("\n");
      text += `\n\n---\n**資料來源**\n${list}`;
      yield { content: [{ type: "text", text }] };
    }
  },
};
