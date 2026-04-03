/**
 * Shared detection display utilities.
 *
 * Extracted from ImagesPage for reuse across verify and images pages.
 */

import type React from "react";
import type { DetectionResponse } from "../api/types";
import { getSpeciesColor } from "../utils/species-colors";

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

/** Get display name for a detection, with capitalized fallback. */
export function getDetectionDisplayName(detection: {
  display_name?: string | null;
  label?: string | null;
  category: string;
}): string {
  if (detection.display_name) return detection.display_name;
  if (detection.label) return detection.label;
  return detection.category.charAt(0).toUpperCase() + detection.category.slice(1);
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
    case "human":
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
      return {
        label: "Unclassified",
        className: "bg-gray-50 text-gray-500 border-gray-200",
      };
  }
}

/** Format detection label for bounding box overlay. */
export function getDetectionLabel(detection: DetectionResponse): string {
  const categoryLabel =
    detection.category.charAt(0).toUpperCase() + detection.category.slice(1);
  const confPct = `${(detection.confidence * 100).toFixed(0)}%`;

  if (detection.label && detection.label_confidence != null) {
    const displayName = detection.display_name || detection.label;
    const labelDisplay =
      displayName.charAt(0).toUpperCase() + displayName.slice(1);
    const labelConfPct = `${(detection.label_confidence * 100).toFixed(0)}%`;
    return `${labelDisplay} ${labelConfPct} · ${categoryLabel} ${confPct}`;
  }

  return `${categoryLabel} ${confPct}`;
}
