import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import type { Plugin } from "vite";

// Knowva ships as an offline appliance: a theme must bundle its fonts, images
// and styles instead of loading them from another host.
//
// Only HTML, SVG and CSS are read. Bundled JavaScript legitimately holds URL
// strings (React's SVG and MathML namespaces, error-page links) that are never
// fetched, so it is not scanned.

const REMOTE = /^(?:https?:)?\/\//i;
const URL_ATTRIBUTES =
  /\b(src|href|srcset|imagesrcset|poster|action|formaction|data|background)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/gi;
const CSS_URL = /url\(\s*(["']?)\s*((?:https?:)?\/\/[^"')\s]*)/gi;
const CSS_IMPORT = /@import\s+(?:url\(\s*)?(["'])?\s*((?:https?:)?\/\/[^"')\s;]*)/gi;

const cssReferences = (source: string): string[] => [
  ...[...source.matchAll(CSS_URL)].map((match) => `url(${match[2]})`),
  ...[...source.matchAll(CSS_IMPORT)].map((match) => `@import ${match[2]}`),
];

const markupReferences = (source: string): string[] => {
  const found: string[] = [];
  for (const match of source.matchAll(URL_ATTRIBUTES)) {
    const attribute = match[1]!.toLowerCase();
    const value = (match[2] ?? match[3] ?? match[4] ?? "").trim();
    const candidates = attribute.endsWith("srcset")
      ? value.split(",").map((candidate) => candidate.trim())
      : [value];
    for (const candidate of candidates) {
      if (REMOTE.test(candidate)) found.push(`${attribute}="${candidate}"`);
    }
  }
  // Inline <style> blocks and style attributes.
  return [...found, ...cssReferences(source)];
};

/**
 * External references in one emitted HTML, SVG or CSS file; other files are
 * skipped. An SVG opened through <object> or <embed> loads its own <image
 * href> and stylesheets, so it is read like HTML.
 */
export function externalReferences(fileName: string, source: string): string[] {
  if (/\.(?:html?|svg)$/i.test(fileName)) return markupReferences(source);
  if (/\.css$/i.test(fileName)) return cssReferences(source);
  return [];
}

const SCANNED = /\.(?:html?|svg|css)$/i;

/**
 * Fails a theme's build if its HTML, SVG or CSS would load anything off the
 * box. It reads the written output directory rather than the bundle, because
 * Vite copies `public/` there without passing it through the bundle.
 */
export function offlineBundle(): Plugin {
  return {
    name: "knowva:offline-bundle",
    apply: "build",
    enforce: "post",
    writeBundle: {
      order: "post",
      handler({ dir }) {
        if (!dir) this.error("offlineBundle needs an output directory");
        const problems = readdirSync(dir, { recursive: true, withFileTypes: true })
          .filter((entry) => entry.isFile() && SCANNED.test(entry.name))
          .flatMap((entry) => {
            const path = join(entry.parentPath, entry.name);
            const name = relative(dir, path);
            return externalReferences(name, readFileSync(path, "utf8")).map(
              (reference) => `${name}: ${reference}`,
            );
          });
        if (problems.length > 0) {
          this.error(
            "the build references external URLs; bundle these with the theme instead:\n" +
              problems.join("\n"),
          );
        }
      },
    },
  };
}
