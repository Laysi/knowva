import { describe, expect, it } from "vitest";
import {
  contrastRatio,
  parseColor,
  resolveColor,
  rootCustomProperties,
  themeContrasts,
} from "./themeColors";

const WHITE = { r: 255, g: 255, b: 255 };
const BLACK = { r: 0, g: 0, b: 0 };

describe("theme colours", () => {
  it("parses opaque hex, rgb() and named colours", () => {
    expect(parseColor("#fff")).toEqual(WHITE);
    expect(parseColor("#FFFFFF")).toEqual(WHITE);
    expect(parseColor("#ffff")).toEqual(WHITE);
    expect(parseColor("#000000ff")).toEqual(BLACK);
    expect(parseColor("rgb(109, 74, 255)")).toEqual({ r: 109, g: 74, b: 255 });
    expect(parseColor("rgb(109 74 255 / 100%)")).toEqual({
      r: 109,
      g: 74,
      b: 255,
    });
    expect(parseColor("rgba(0, 0, 0, 1)")).toEqual(BLACK);
    expect(parseColor(" White ")).toEqual(WHITE);
  });

  it("refuses translucent and unsupported colours", () => {
    expect(() => parseColor("#fff8")).toThrow("translucent");
    expect(() => parseColor("rgba(109, 74, 255, 0.24)")).toThrow(
      "translucent",
    );
    expect(() => parseColor("rgb(0 0 0 / 50%)")).toThrow("translucent");
    expect(() => parseColor("hsl(250 100% 64%)")).toThrow("unsupported");
    expect(() => parseColor("linear-gradient(#fff, #000)")).toThrow(
      "unsupported",
    );
  });

  it("reads custom properties from :root rules only, the later rule winning", () => {
    const properties = rootCustomProperties(`
      /* :root { --theme-text: #f00; } */
      :root {
        --theme-text: #111;
        --gradient: linear-gradient(135deg, #6d4aff 0%, #ff9f1c 100%);
      }
      .card { --theme-text: #222; }
      :root { --theme-text: #333; }
    `);

    expect(properties.get("--theme-text")).toBe("#333");
    expect(properties.get("--gradient")).toBe(
      "linear-gradient(135deg, #6d4aff 0%, #ff9f1c 100%)",
    );
  });

  it("follows var() references and fallbacks, and names what is missing", () => {
    const properties = rootCustomProperties(`:root {
      --color-text: #10142a;
      --theme-text: var(--color-text);
      --theme-fallback: var(--undeclared, #fff);
      --loop-a: var(--loop-b);
      --loop-b: var(--loop-a);
    }`);

    expect(resolveColor("--theme-text", properties)).toEqual({
      r: 16,
      g: 20,
      b: 42,
    });
    expect(resolveColor("--theme-fallback", properties)).toEqual(WHITE);
    expect(() => resolveColor("--theme-error", properties)).toThrow(
      "--theme-error is not declared on :root",
    );
    expect(() => resolveColor("--loop-a", properties)).toThrow("circular");
  });

  it("computes WCAG contrast ratios", () => {
    expect(contrastRatio(BLACK, WHITE)).toBeCloseTo(21, 5);
    expect(contrastRatio(WHITE, BLACK)).toBeCloseTo(21, 5);
    expect(contrastRatio(WHITE, WHITE)).toBe(1);
    // #767676 is the lightest grey that still clears 4.5:1 on white.
    expect(contrastRatio(parseColor("#767676"), WHITE)).toBeGreaterThan(4.5);
    expect(contrastRatio(parseColor("#777777"), WHITE)).toBeLessThan(4.5);
  });

  it("reports every contract pair", () => {
    const contrasts = themeContrasts(`:root {
      --theme-text: #000;
      --theme-background: #fff;
      --theme-brand: #000;
      --theme-on-brand: #fff;
      --theme-success: #777;
      --theme-warning: #000;
      --theme-error: #000;
    }`);

    expect(
      contrasts.map(({ foreground, background }) => `${foreground}/${background}`),
    ).toEqual([
      "text/background",
      "onBrand/brand",
      "success/background",
      "warning/background",
      "error/background",
    ]);
    expect(contrasts[2]!.ratio).toBeLessThan(4.5);
  });
});
