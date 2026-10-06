/**
 * Semantic colour roles every theme declares on `:root`. A theme may style
 * anything else however it likes; these name the pairs the contract checks.
 */
export const THEME_COLOR_VARIABLES = {
  text: "--theme-text",
  background: "--theme-background",
  brand: "--theme-brand",
  onBrand: "--theme-on-brand",
  success: "--theme-success",
  warning: "--theme-warning",
  error: "--theme-error",
} as const;

export type ThemeColorRole = keyof typeof THEME_COLOR_VARIABLES;

/** Foreground on background, each held to WCAG AA for normal-size text. */
export const THEME_CONTRAST_PAIRS: readonly {
  foreground: ThemeColorRole;
  background: ThemeColorRole;
}[] = [
  { foreground: "text", background: "background" },
  { foreground: "onBrand", background: "brand" },
  { foreground: "success", background: "background" },
  { foreground: "warning", background: "background" },
  { foreground: "error", background: "background" },
];

export const WCAG_AA_TEXT_CONTRAST = 4.5;

export type Rgb = { r: number; g: number; b: number };

/** Custom properties declared directly in `:root` rules, later ones winning. */
export function rootCustomProperties(css: string): Map<string, string> {
  const properties = new Map<string, string>();
  const source = css.replace(/\/\*[\s\S]*?\*\//g, "");
  for (const rule of source.matchAll(/(?:^|[;{}])\s*:root\s*\{([^{}]*)\}/g)) {
    for (const declaration of rule[1]!.split(";")) {
      const match = /^\s*(--[\w-]+)\s*:\s*([\s\S]*?)\s*$/.exec(declaration);
      if (match) properties.set(match[1]!, match[2]!);
    }
  }
  return properties;
}

const NAMED_COLORS: Record<string, Rgb> = {
  black: { r: 0, g: 0, b: 0 },
  white: { r: 255, g: 255, b: 255 },
};

const channel = (value: string): number => {
  const number = Number.parseFloat(value);
  if (!Number.isFinite(number)) throw new Error(`not a colour channel: ${value}`);
  return value.trim().endsWith("%") ? (number / 100) * 255 : number;
};

const alpha = (value: string | undefined): number => {
  if (value === undefined) return 1;
  const number = Number.parseFloat(value);
  return value.trim().endsWith("%") ? number / 100 : number;
};

/**
 * Parses an opaque hex, rgb() or named colour. A translucent colour is
 * refused: its contrast depends on whatever happens to be behind it.
 */
export function parseColor(value: string): Rgb {
  const text = value.trim().toLowerCase();
  const named = NAMED_COLORS[text];
  if (named) return named;

  const hex = /^#([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$/.exec(text);
  if (hex) {
    const digits =
      hex[1]!.length <= 4
        ? [...hex[1]!].map((digit) => digit + digit).join("")
        : hex[1]!;
    if (digits.length === 8 && digits.slice(6) !== "ff") {
      throw new Error(`translucent colour: ${value}`);
    }
    return {
      r: Number.parseInt(digits.slice(0, 2), 16),
      g: Number.parseInt(digits.slice(2, 4), 16),
      b: Number.parseInt(digits.slice(4, 6), 16),
    };
  }

  const rgb = /^rgba?\(\s*([^)]*)\)$/.exec(text);
  if (rgb) {
    const [colour, opacity] = rgb[1]!.split("/");
    const parts = colour!.trim().split(/[\s,]+/).filter(Boolean);
    const alphaPart = opacity ?? (parts.length === 4 ? parts.pop() : undefined);
    if (parts.length !== 3) throw new Error(`not an rgb() colour: ${value}`);
    if (alpha(alphaPart) < 1) throw new Error(`translucent colour: ${value}`);
    const [r, g, b] = parts.map(channel) as [number, number, number];
    return { r, g, b };
  }

  throw new Error(`unsupported colour (use hex, rgb() or a var()): ${value}`);
}

/** Follows `var()` references through `:root` until a colour remains. */
export function resolveColor(
  name: string,
  properties: ReadonlyMap<string, string>,
  seen: readonly string[] = [],
): Rgb {
  if (seen.includes(name)) {
    throw new Error(`circular var(): ${[...seen, name].join(" -> ")}`);
  }
  const value = properties.get(name);
  if (value === undefined) throw new Error(`${name} is not declared on :root`);

  const reference = /^var\(\s*(--[\w-]+)\s*(?:,\s*([\s\S]+))?\)$/.exec(value);
  if (!reference) return parseColor(value);
  if (!properties.has(reference[1]!) && reference[2] !== undefined) {
    return parseColor(reference[2]);
  }
  return resolveColor(reference[1]!, properties, [...seen, name]);
}

const linear = (value: number): number => {
  const c = value / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
};

const luminance = ({ r, g, b }: Rgb): number =>
  0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);

/** WCAG 2.x contrast ratio, from 1 to 21. */
export function contrastRatio(a: Rgb, b: Rgb): number {
  const [light, dark] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [
    number,
    number,
  ];
  return (light + 0.05) / (dark + 0.05);
}

export type ThemeContrast = {
  foreground: ThemeColorRole;
  background: ThemeColorRole;
  ratio: number;
};

/** Every contract pair's contrast, from the stylesheets' `:root` rules. */
export function themeContrasts(css: string): ThemeContrast[] {
  const properties = rootCustomProperties(css);
  return THEME_CONTRAST_PAIRS.map(({ foreground, background }) => ({
    foreground,
    background,
    ratio: contrastRatio(
      resolveColor(THEME_COLOR_VARIABLES[foreground], properties),
      resolveColor(THEME_COLOR_VARIABLES[background], properties),
    ),
  }));
}
