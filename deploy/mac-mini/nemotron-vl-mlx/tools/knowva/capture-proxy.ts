// Forwards every request to the inference service and records each rerank
// call (request, response, seconds) as one JSON file, numbered in arrival
// order, so the application's real rerank traffic can be replayed later.
//
//   bun capture-proxy.ts --out DIR [--listen 9501] [--upstream http://127.0.0.1:9500]

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({
  options: {
    out: { type: "string" },
    listen: { type: "string", default: "9501" },
    upstream: { type: "string", default: "http://127.0.0.1:9500" },
  },
});
if (!args.out) throw new Error("--out is required");
mkdirSync(args.out, { recursive: true });

let seq = 0;
const parse = (text: string) => {
  try {
    return JSON.parse(text);
  } catch {
    return text.slice(0, 500);
  }
};

Bun.serve({
  hostname: "127.0.0.1",
  port: Number(args.listen),
  idleTimeout: 255,
  async fetch(request) {
    const url = new URL(request.url);
    const ranking = request.method === "POST" && url.pathname === "/v1/ranking";
    const n = ranking ? ++seq : 0;
    const body = request.method === "GET" || request.method === "HEAD" ? undefined : await request.text();
    const startedAt = new Date().toISOString();
    const began = performance.now();
    const upstream = await fetch(`${args.upstream}${url.pathname}${url.search}`, {
      method: request.method,
      headers: { "content-type": request.headers.get("content-type") ?? "application/json" },
      body,
    });
    const text = await upstream.text();
    const seconds = (performance.now() - began) / 1000;
    if (ranking) {
      writeFileSync(
        join(args.out!, `${String(n).padStart(5, "0")}.json`),
        JSON.stringify({ seq: n, startedAt, seconds, status: upstream.status, request: parse(body ?? ""), response: parse(text) }),
      );
      console.log(`${startedAt} #${n} ${upstream.status} ${seconds.toFixed(2)}s`);
    }
    return new Response(text, {
      status: upstream.status,
      headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
    });
  },
});
console.log(`capture proxy on 127.0.0.1:${args.listen} -> ${args.upstream}, writing ${args.out}`);
