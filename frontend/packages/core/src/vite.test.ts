import { describe, expect, it } from "vitest";
import { externalReferences } from "./vite";

describe("externalReferences", () => {
  it("finds remote src, href and srcset values in HTML", () => {
    const html = `<!doctype html>
      <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC">
      <link rel=preconnect href=//fonts.gstatic.com>
      <script type="module" src='http://cdn.example.com/app.js'></script>
      <img srcset="/assets/logo.png 1x, https://cdn.example.com/logo@2x.png 2x">
      <a href="/docs">docs</a>`;

    expect(externalReferences("index.html", html)).toEqual([
      'href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC"',
      'href="//fonts.gstatic.com"',
      'src="http://cdn.example.com/app.js"',
      'srcset="https://cdn.example.com/logo@2x.png 2x"',
    ]);
  });

  it("finds remote url() and @import in CSS and in inline HTML styles", () => {
    const css = `@import url("https://fonts.googleapis.com/css2?family=Inter");
      @import 'https://cdn.example.com/reset.css';
      @font-face { src: url(https://fonts.gstatic.com/s/inter.woff2) format("woff2"); }
      .hero { background: url('//cdn.example.com/hero.jpg'); }`;

    expect(externalReferences("assets/index-abc.css", css)).toEqual([
      "url(https://fonts.googleapis.com/css2?family=Inter)",
      "url(https://fonts.gstatic.com/s/inter.woff2)",
      "url(//cdn.example.com/hero.jpg)",
      "@import https://fonts.googleapis.com/css2?family=Inter",
      "@import https://cdn.example.com/reset.css",
    ]);
    expect(
      externalReferences(
        "index.html",
        `<div style="background:url(https://cdn.example.com/a.png)"></div>`,
      ),
    ).toEqual(["url(https://cdn.example.com/a.png)"]);
  });

  it("accepts bundled, relative and data: references, and skips JavaScript", () => {
    const html = `<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3Crect fill='url(%23nova)'/%3E%3C/svg%3E">
      <script type="module" crossorigin src="/assets/index-abc.js"></script>
      <link rel="stylesheet" crossorigin href="/assets/index-abc.css">`;
    const css = `@font-face { src: url(/assets/noto-sans-tc.woff2) format("woff2"); }
      .mark { background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'/%3E"); }
      /* see https://example.com for the licence */`;

    expect(externalReferences("index.html", html)).toEqual([]);
    expect(externalReferences("assets/index-abc.css", css)).toEqual([]);
    expect(
      externalReferences(
        "assets/index-abc.js",
        'const ns = "http://www.w3.org/2000/svg"; fetch("https://example.com")',
      ),
    ).toEqual([]);
  });
});
