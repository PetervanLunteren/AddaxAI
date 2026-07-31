/**
 * Shared detection display utilities used by the verify page and its
 * detail/canvas components.
 */

import type React from "react";
import chroma from "chroma-js";
import { getSpeciesColor } from "../utils/species-colors";
import { resolveSpeciesName } from "./species-name-mode";

/** Get color for a detection: species color if labeled, category color otherwise.
 *  Uses label_taxonomy_id as the color key when available (matches event label chips). */
export function getDetectionColor(detection: {
  label?: string | null;
  label_taxonomy_id?: string | null;
  category: string;
}): string {
  const key = detection.label_taxonomy_id || detection.label;
  return key ? getSpeciesColor(key) : getCategoryColor(detection.category);
}

/**
 * Whether a detection should render as a bounding box on a given file's
 * visible image. Three gates, in priority order:
 *
 *  1. Confidence must meet the project's detection threshold.
 *  2. The detection must have a bbox — event-level observations are
 *     bbox-less by design and never draw.
 *  3. For videos, the detection must be on the frame the JPEG actually
 *     renders (the best frame). Non-best-frame AI detections still
 *     exist in the data and surface in the verification list, but they
 *     don't paint onto the canvas of an unrelated frame.
 *
 * Centralised so every grid tile / canvas / modal applies the same
 * rules; without this, regressions slipped in tile-by-tile.
 */
export function shouldDrawBbox<
  D extends {
    confidence: number;
    bbox_x: number | null;
    bbox_y: number | null;
    bbox_width: number | null;
    bbox_height: number | null;
    frame_number: number | null;
  },
>(
  detection: D,
  file: { file_type: string; best_frame_number: number | null },
  detectionThreshold: number,
): detection is D & {
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
} {
  if (detection.confidence < detectionThreshold) return false;
  if (detection.bbox_x === null) return false;
  if (file.file_type === "video" && file.best_frame_number != null) {
    return detection.frame_number === file.best_frame_number;
  }
  return true;
}

/** Get display name for a detection under the active species-name mode
 *  (common vs scientific), with graceful fallbacks. */
export function getDetectionDisplayName(detection: {
  common_name?: string | null;
  scientific_name?: string | null;
  label?: string | null;
  category: string;
}): string {
  return resolveSpeciesName(detection);
}

/** Get color for a detection category. */
export function getCategoryColor(category: string): string {
  switch (category) {
    case "animal":
      return "#0f6064"; // teal brand
    case "person":
      return "#ff8945"; // orange
    case "vehicle":
      return "#71b7ba"; // light teal
    default:
      return "#882000"; // dark red
  }
}

/** Text color (white or dark) for a category chip whose background is
 *  getCategoryColor(category). Mirrors getSpeciesTextColor's contrast
 *  rule so category chips read the same way as species chips. */
export function getCategoryTextColor(category: string): string {
  const bg = getCategoryColor(category);
  return chroma.contrast(bg, "white") >= 3 ? "white" : "#1f2937";
}

/** Get styled badge props for an observation type. */
export function getObservationBadge(type: string): {
  label: string;
  className: string;
  style?: React.CSSProperties;
} {
  switch (type) {
    case "animal":
      return {
        label: "Animal",
        className: "bg-green-100 text-green-800 border-green-200",
      };
    case "person":
      return {
        label: "Person",
        className: "text-white border-transparent",
        style: { backgroundColor: "#ff8236" },
      };
    case "vehicle":
      return {
        label: "Vehicle",
        className: "bg-blue-100 text-blue-800 border-blue-200",
      };
    case "blank":
      return {
        label: "Blank",
        className: "bg-gray-100 text-gray-600 border-gray-200",
      };
    case "unknown":
      return {
        label: "Unknown",
        className: "bg-yellow-100 text-yellow-800 border-yellow-200",
      };
    default:
      // A category from a detector we know nothing about ("shark",
      // "fish"). Show what it is rather than "Unclassified", which
      // would be a lie: the detector was perfectly clear about it.
      return {
        label: type ? type[0].toUpperCase() + type.slice(1) : "Unclassified",
        className: "bg-gray-50 text-gray-500 border-gray-200",
      };
  }
}

