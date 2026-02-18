/**
 * Shared detection display utilities.
 *
 * Extracted from ImagesPage for reuse across verify and images pages.
 */

import type { DetectionResponse } from "../api/types";

/** Get color for a detection category. */
export function getCategoryColor(category: string): string {
  switch (category) {
    case "animal":
      return "rgb(16, 185, 129)"; // emerald
    case "person":
      return "rgb(244, 63, 94)"; // rose
    case "vehicle":
      return "rgb(99, 102, 241)"; // indigo
    default:
      return "rgb(156, 163, 175)"; // gray
  }
}

/** Get styled badge props for an observation type. */
export function getObservationBadge(type: string): {
  label: string;
  className: string;
} {
  switch (type) {
    case "animal":
      return {
        label: "Animal",
        className: "bg-green-100 text-green-800 border-green-200",
      };
    case "human":
      return {
        label: "Human",
        className: "bg-red-100 text-red-800 border-red-200",
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

  if (detection.species && detection.species_confidence != null) {
    const speciesLabel =
      detection.species.charAt(0).toUpperCase() + detection.species.slice(1);
    const speciesConfPct = `${(detection.species_confidence * 100).toFixed(0)}%`;
    return `${speciesLabel} ${speciesConfPct} · ${categoryLabel} ${confPct}`;
  }

  return `${categoryLabel} ${confPct}`;
}
