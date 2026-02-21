/**
 * Shared constants and helpers for detection bbox / label rendering.
 *
 * Used by both AnnotationCanvas (Konva) and VideoPlayer (SVG) so that
 * visual styling stays consistent across both views.
 */

import { getCategoryColor } from "./detection-utils";
import type { DetectionResponse } from "../api/types";

// ── Layout constants ──────────────────────────────────────────────
export const PILL_PAD_X = 6;
export const PILL_PAD_Y = 4;
export const DOT_R = 4;
export const DOT_GAP = 5;
export const LINE_GAP = 2;
export const FONT_SM = 10;
export const FONT_LG = 12;
export const TEXT_START_X = PILL_PAD_X + DOT_R * 2 + DOT_GAP; // 19

export const BBOX_STROKE_WIDTH = 2;
export const BBOX_OPACITY = 0.5;
export const BBOX_CORNER_RADIUS = 4;
export const DIM_FILL = "rgba(0, 0, 0, 0.35)";
export const PILL_BG = "rgba(0,0,0,0.5)";

// ── Text measurement ──────────────────────────────────────────────
let _measureCtx: CanvasRenderingContext2D | null = null;

export function measureTextWidth(
  text: string,
  fontSize: number,
  bold: boolean,
): number {
  if (!_measureCtx) {
    _measureCtx = document.createElement("canvas").getContext("2d")!;
  }
  _measureCtx.font = `${bold ? "bold " : ""}${fontSize}px Arial, sans-serif`;
  return _measureCtx.measureText(text).width;
}

// ── Canvas rounded-rect sub-path (for Konva evenodd overlays) ─────
export function roundedRectPath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  r = Math.min(r, w / 2, h / 2);
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// ── SVG rounded-rect sub-path (for SVG evenodd overlays) ──────────
export function svgRoundedRectPath(
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): string {
  r = Math.min(r, w / 2, h / 2);
  return (
    `M${x + r},${y}` +
    `h${w - 2 * r}` +
    `a${r},${r},0,0,1,${r},${r}` +
    `v${h - 2 * r}` +
    `a${r},${r},0,0,1,${-r},${r}` +
    `h${-(w - 2 * r)}` +
    `a${r},${r},0,0,1,${-r},${-r}` +
    `v${-(h - 2 * r)}` +
    `a${r},${r},0,0,1,${r},${-r}Z`
  );
}

// ── Pill layout computation ───────────────────────────────────────
export interface PillLayout {
  categoryText: string;
  speciesText: string;
  hasSpecies: boolean;
  pillWidth: number;
  pillHeight: number;
  color: string;
}

export function computePillLayout(detection: DetectionResponse): PillLayout {
  const color = getCategoryColor(detection.category);
  const hasSpecies = !!detection.species;

  const categoryText = `${detection.category.charAt(0).toUpperCase() + detection.category.slice(1)} ${(detection.confidence * 100).toFixed(0)}%`;
  const speciesText = hasSpecies
    ? `${detection.species!.charAt(0).toUpperCase() + detection.species!.slice(1)} ${((detection.species_confidence ?? detection.confidence) * 100).toFixed(0)}%`
    : "";

  let pillHeight: number;
  let pillWidth: number;
  if (hasSpecies) {
    pillHeight = PILL_PAD_Y + FONT_SM + LINE_GAP + FONT_LG + PILL_PAD_Y;
    const w1 = measureTextWidth(categoryText, FONT_SM, false);
    const w2 = measureTextWidth(speciesText, FONT_LG, true);
    pillWidth = TEXT_START_X + Math.max(w1, w2) + PILL_PAD_X;
  } else {
    pillHeight = PILL_PAD_Y + FONT_LG + PILL_PAD_Y;
    const tw = measureTextWidth(categoryText, FONT_LG, true);
    pillWidth = TEXT_START_X + tw + PILL_PAD_X;
  }

  return { categoryText, speciesText, hasSpecies, pillWidth, pillHeight, color };
}
