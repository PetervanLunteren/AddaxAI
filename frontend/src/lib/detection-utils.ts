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
 * Labels that mean "nothing is here". Mirrors `NON_LABEL_CLASSES` in
 * `backend/app/ml/label_exclusion.py` — keep the two in sync.
 *
 * The ingest skip keeps the AI's own such calls out of the database, so
 * the only way one gets in is a person pressing X on the Labels page.
 */
export const NON_LABEL_CLASSES = new Set([
  "bait",
  "blank",
  "empty",
  "false detection",
  "non-animal",
  "none",
  "vide",
]);

/** The same rule as the backend's `is_a_real_detection()`, inverted.
 *  `null` is a real detection the classifier simply never named. */
export function isNonLabel(label: string | null | undefined): boolean {
  return !!label && NON_LABEL_CLASSES.has(label.toLowerCase());
}

/**
 * Everything `shouldDrawBbox` decides except the video best-frame rule.
 *
 * Split out for `VideoPlayer`, which draws every frame's boxes over the
 * real video on purpose, so it needs these gates and must not have the
 * fourth. It had its own inline copy of the confidence test, which is
 * exactly the drift this module exists to prevent: the verified
 * override and the rejected-box rule both landed in `shouldDrawBbox`
 * and neither reached the video, so one event modal drew different
 * boxes in frame mode and in video mode.
 *
 *  1. A box a person rejected is never outlined. Mirrors the backend's
 *     `is_a_real_detection()`, which already keeps it out of every
 *     count, so drawing it argues with the number beside it.
 *  2. Confidence must meet the threshold, **or** the box must be
 *     verified. The same `confidence >= threshold OR verified` rule
 *     every backend query applies (DEVELOPERS.md, "Detection threshold
 *     and verified override"). Relabelling never rewrites
 *     `Detection.confidence`, so a box a human confirmed at 3% keeps
 *     that 3% forever and would otherwise earn a card and a count but
 *     no rectangle.
 *  3. It must have a bbox — event-level observations are bbox-less by
 *     design and never draw.
 */
export function passesDrawFilter(
  detection: {
    confidence: number;
    verified: boolean;
    label: string | null;
    bbox_x: number | null;
  },
  detectionThreshold: number,
): boolean {
  if (isNonLabel(detection.label)) return false;
  if (!detection.verified && detection.confidence < detectionThreshold) {
    return false;
  }
  return detection.bbox_x !== null;
}

/**
 * The frame whose still shows this detection, when it is not the best
 * frame: a track's representative frame, which has its own JPEG
 * (`/image?frame=N`). `undefined` means the file's default picture (the
 * best frame, or the photo itself), which is also the answer for a
 * verified box on a frame nobody has a picture of.
 */
export function stillFrameFor(
  file: { file_type: string; best_frame_number: number | null; tracks?: { representative_frame_number: number; has_frame: boolean }[] },
  detection: { frame_number: number | null },
): number | undefined {
  if (file.file_type !== "video" || detection.frame_number == null) return undefined;
  if (detection.frame_number === file.best_frame_number) return undefined;
  const track = file.tracks?.find(
    (t) => t.representative_frame_number === detection.frame_number && t.has_frame,
  );
  return track ? detection.frame_number : undefined;
}

/**
 * Whether a detection should render as a bounding box on a given file's
 * visible image.
 *
 * Two gates: `passesDrawFilter` above, then the video rule. For videos
 * the detection must be on the frame the JPEG actually renders: the
 * best frame by default, or `frameNumber` when the surface shows
 * another still (a track's representative frame, see `stillFrameFor`).
 * Non-best-frame AI detections still exist in the data and surface in
 * the verification list, but they must not paint onto the canvas of an
 * unrelated frame — that is the crop-service bug in another costume,
 * and it looks perfectly fine until the subject moves.
 *
 * No rule here reads who drew a box. A drawn box is verified at
 * confidence 1.0, so it passes like any confirmed box; one the person
 * later marked as "nothing here" disappears like any rejected box.
 *
 * Centralised so every grid tile / canvas / modal applies the same
 * rules; without this, regressions slipped in tile-by-tile.
 */
export function shouldDrawBbox<
  D extends {
    confidence: number;
    verified: boolean;
    label: string | null;
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
  frameNumber?: number,
): detection is D & {
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
} {
  if (!passesDrawFilter(detection, detectionThreshold)) return false;
  if (detection.bbox_x === null) return false;
  if (file.file_type === "video") {
    const shown = frameNumber ?? file.best_frame_number;
    if (shown != null) return detection.frame_number === shown;
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

/** Get color for a detection category. Any category that is not person
 *  or vehicle is wildlife by another detector's name ("fish",
 *  "elasmobranch") and takes the animal colour; the backend's
 *  CATEGORY_COLORS mirrors this. */
export function getCategoryColor(category: string): string {
  switch (category) {
    case "person":
      return "#ff8945"; // orange
    case "vehicle":
      return "#71b7ba"; // light teal
    default:
      return "#0f6064"; // teal brand, the animal colour
  }
}

/** Text color (white or dark) for a category chip whose background is
 *  getCategoryColor(category). Mirrors getContrastTextColor's rule so
 *  category chips read the same way as species chips. */
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

