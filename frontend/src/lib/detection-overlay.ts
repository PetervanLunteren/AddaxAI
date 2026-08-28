/**
 * Shared constants and helpers for detection bbox / label rendering.
 *
 * Used by both AnnotationCanvas (Konva) and VideoPlayer (SVG) so that
 * visual styling stays consistent across both views.
 */

import { getCategoryColor } from "./detection-utils";
import { resolveSpeciesName } from "./species-name-mode";
import { getSpeciesColor } from "../utils/species-colors";
import type { DetectionResponse } from "../api/types";

// ── Layout constants ──────────────────────────────────────────────
export const PILL_PAD_X = 6;
export const PILL_PAD_Y = 4;
export const LINE_GAP = 2;
export const FONT = 10; // both pill lines share one size
export const TEXT_START_X = PILL_PAD_X; // no dot, text starts at the pad

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
  labelText: string;
  hasLabel: boolean;
  pillWidth: number;
  pillHeight: number;
  color: string;
}

export function computePillLayout(detection: DetectionResponse): PillLayout {
  const colorKey = detection.label_taxonomy_id || detection.label;
  const color = colorKey
    ? getSpeciesColor(colorKey)
    : getCategoryColor(detection.category);
  const hasLabel = !!detection.label;

  const categoryText = `${detection.category.charAt(0).toUpperCase() + detection.category.slice(1)} ${(detection.confidence * 100).toFixed(0)}%`;
  const displayName = resolveSpeciesName(detection) || detection.label!;
  const labelText = hasLabel
    ? `${displayName.charAt(0).toUpperCase() + displayName.slice(1)} ${((detection.label_confidence ?? detection.confidence) * 100).toFixed(0)}%`
    : "";

  let pillHeight: number;
  let pillWidth: number;
  if (hasLabel) {
    pillHeight = PILL_PAD_Y + FONT + LINE_GAP + FONT + PILL_PAD_Y;
    const w1 = measureTextWidth(categoryText, FONT, false);
    const w2 = measureTextWidth(labelText, FONT, false);
    pillWidth = TEXT_START_X + Math.max(w1, w2) + PILL_PAD_X;
  } else {
    pillHeight = PILL_PAD_Y + FONT + PILL_PAD_Y;
    const tw = measureTextWidth(categoryText, FONT, false);
    pillWidth = TEXT_START_X + tw + PILL_PAD_X;
  }

  return { categoryText, labelText, hasLabel, pillWidth, pillHeight, color };
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Where a box's label pill goes. Above the box when there is room, else
 * below it, else inside at the top (a box spanning the full height, the
 * only case with nowhere else). Clamped to the frame horizontally so a
 * box at the right edge keeps its label on the picture.
 *
 * One rule for every renderer (Konva canvas, SVG, 2D canvas). Each used
 * to carry its own copy that fell back to *inside* the box, which put
 * the pill on the animal's head whenever the box touched the top edge
 * (Grant Hiebert, 2026-08-25). `box`, `pill` and `frame` share one unit:
 * pass the pill size already scaled to it.
 */
export function placePill(
  box: Rect,
  pill: { width: number; height: number },
  frame: { width: number; height: number },
): { x: number; y: number } {
  const x = Math.max(0, Math.min(box.x, frame.width - pill.width));
  let y: number;
  if (box.y - pill.height >= 0) {
    y = box.y - pill.height;
  } else if (box.y + box.height + pill.height <= frame.height) {
    y = box.y + box.height;
  } else {
    y = box.y;
  }
  return { x, y };
}
