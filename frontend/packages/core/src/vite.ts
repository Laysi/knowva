import type { Plugin } from "vite";

// Knowva ships as an offline appliance: a theme must bundle its fonts, images
// and styles instead of loading them from another host.
//
// Only HTML and CSS are read. Bundled JavaScript legitimately holds URL
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

const htmlReferences = (source: string): string[] => {
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

/** External references in one emitted HTML or CSS file; other files are skipped. */
export function externalReferences(fileName: string, source: string): string[] {
  if (/\.html?$/i.test(fileName)) return htmlReferences(source);
  if (/\.css$/i.test(fileName)) return cssReferences(source);
  return [];
}

/** Fails a theme's build if its HTML or CSS would load anything off the box. */
export function offlineBundle(): Plugin {
  return {
    name: "knowva:offline-bundle",
    apply: "build",
    enforce: "post",
    generateBundle: {
      order: "post",
      handler(_options, bundle) {
        const problems = Object.values(bundle).flatMap((output) => {
          const source =
            output.type === "asset"
              ? typeof output.source === "string"
                ? output.source
                : new TextDecoder().decode(output.source)
              : "";
          return externalReferences(output.fileName, source).map(
            (reference) => `${output.fileName}: ${reference}`,
          );
        });
        if (problems.length > 0) {
          this.error(
            "the bundle references external URLs; bundle these with the theme instead:\n" +
              problems.join("\n"),
          );
        }
      },
    },
  };
}
