/**
 * Colour helpers for the confusion matrix and classification report.
 *
 * The matrix uses a single-hue teal intensity ramp (low to #0f6064)
 * applied per row. Intensity only — no good/bad judgment on individual
 * cells. The F1 column uses the project status palette as a diverging
 * scale: #882000 (bad) through #71b7ba (middle) to #0f6064 (good).
 */

export interface SwatchStyle {
  background: string;
  color: string;
}

const TEAL_LOW: [number, number, number] = [227, 240, 240]; // #e3f0f0
const TEAL_HIGH: [number, number, number] = [15, 96, 100]; // #0f6064

const STATUS_BAD: [number, number, number] = [136, 32, 0]; // #882000
const STATUS_MID: [number, number, number] = [113, 183, 186]; // #71b7ba
const STATUS_GOOD: [number, number, number] = [15, 96, 100]; // #0f6064

function clamp01(v: number): number {
  if (Number.isNaN(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

function lerp(
  a: [number, number, number],
  b: [number, number, number],
  t: number
): [number, number, number] {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

function rgb([r, g, b]: [number, number, number]): string {
  return `rgb(${r}, ${g}, ${b})`;
}

/** Pick black or white text depending on background luminance. */
function textOn(bg: [number, number, number]): string {
  // Relative luminance approximation; good enough for categorical swatches.
  const lum = (0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]) / 255;
  return lum > 0.55 ? "#1f2937" : "#ffffff";
}

/**
 * Row-normalised intensity for a matrix cell. `valuePerRow` is
 * `count / rowMax` in `[0, 1]`. A value of 0 renders transparent so the
 * grid background shows through; any other value gets a coloured swatch.
 */
export function matrixCellColor(valuePerRow: number): SwatchStyle {
  const t = clamp01(valuePerRow);
  if (t === 0) {
    return { background: "transparent", color: "var(--color-muted-foreground)" };
  }
  const rgbTuple = lerp(TEAL_LOW, TEAL_HIGH, t);
  return { background: rgb(rgbTuple), color: textOn(rgbTuple) };
}

/**
 * Diverging status colour for an F1 (or precision / recall) value in
 * `[0, 1]`. Null values render neutral. Piecewise: bad → mid at 0.5,
 * then mid → good.
 */
export function f1DivergingColor(value: number | null): SwatchStyle {
  if (value === null) {
    return { background: "transparent", color: "var(--color-muted-foreground)" };
  }
  const v = clamp01(value);
  const rgbTuple = v < 0.5
    ? lerp(STATUS_BAD, STATUS_MID, v / 0.5)
    : lerp(STATUS_MID, STATUS_GOOD, (v - 0.5) / 0.5);
  return { background: rgb(rgbTuple), color: textOn(rgbTuple) };
}
