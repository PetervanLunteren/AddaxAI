/**
 * Overall percentage for the Labels loading bar.
 *
 * The sort subprocess reports each phase with its own 0 → N counter. To
 * keep the bar moving forward instead of resetting at every phase, each
 * phase the sort actually runs gets an equal slice of 0-100%. The slices
 * follow the sort, not a fixed table: suggestions skip the similarity
 * walk, so a fixed third for it made the bar jump from 33% to 67%.
 */

import type { LabelsProgressEvent, LabelsProgressPhase } from "../api/labels";
import type { LabelSort } from "../api/types";

/** Phases each sort emits, in the order similarity_script.py emits them. */
const SORT_PHASES: Record<LabelSort, LabelsProgressPhase[]> = {
  similarity: ["load", "sort", "neighbors"],
  suggestions: ["load", "neighbors"],
  events: ["detections"],
};

export function overallProgressPct(
  progress: LabelsProgressEvent,
  sort: LabelSort,
): number {
  const phases = SORT_PHASES[sort];
  const index = phases.indexOf(progress.phase);
  if (index < 0) return 0;
  const phaseFrac =
    progress.total > 0 ? Math.min(1, progress.done / progress.total) : 0;
  return Math.round(((index + phaseFrac) / phases.length) * 100);
}
