/**
 * The media copies option state and its two pure helpers, shared by the
 * folder-run Save step and the project image copies dialog.
 *
 * A leaf module on purpose: no API client, no DOM, so it can be unit
 * tested and imported from either place without pulling the folder-run
 * form hook along.
 */

import type {
  SaveOutputsResult,
  SeparateGroupBy,
} from "../../../api/folder-runs";

export interface SeparateState {
  enabled: boolean;
  groupBy: SeparateGroupBy;
  /** Keep a burst together: every file in an event shares one folder
   * (the event's main species). */
  groupEvents: boolean;
  /** Put the species folder inside the user's original folders instead of
   * on top (camtrapR-style ``station/species/`` layout). */
  speciesLast: boolean;
  /** Copy empty captures (no detections) too. Off = skip them. */
  copyEmpties: boolean;
  /** Media-output confidence: detections below it (unless verified)
   * are left out of the copies, drawn boxes, and blurs. The data
   * exports always contain everything regardless. */
  mediaThreshold: number;
  /** Leaf ids the user chose to include. What an empty list means
   * depends on the surface, see `EmptySelectionMeans`. The request
   * sends the complement as ``excluded_label_ids``; the data exports
   * ignore it. */
  includedLabelIds: string[];
}

export interface VisualiseState {
  enabled: boolean;
}

export interface AnonymiseState {
  enabled: boolean;
}

/** What an empty `includedLabelIds` means. The folder-run Save step
 * starts with every label ("all"); the project image copies dialog
 * starts with none picked ("none"), so copying everything is a
 * deliberate click on "Select all". */
export type EmptySelectionMeans = "all" | "none";

/** The label exclusion set sent to the backend: every leaf NOT included
 * by the user. Empty when all species are included (no filter) or the
 * tree hasn't loaded yet. Under `emptyMeans: "none"` an empty selection
 * excludes every leaf instead, so a preview honestly reads 0 files
 * while nothing is picked. The data exports ignore this entirely. */
export function excludedLabelIds(
  separate: SeparateState,
  allLabelIds: string[],
  emptyMeans: EmptySelectionMeans = "all",
): string[] {
  const { includedLabelIds } = separate;
  if (includedLabelIds.length === 0) {
    return emptyMeans === "none" ? [...allLabelIds] : [];
  }
  if (includedLabelIds.length >= allLabelIds.length) {
    return [];
  }
  const included = new Set(includedLabelIds);
  return allLabelIds.filter((id) => !included.has(id));
}

/** Roll every module's reported errors into a flat list. Returns an
 * empty array when nothing went wrong, which is the case the completion
 * screens optimise for: no banner. Deduplicated, because separation and
 * annotation both notice the same missing file in the same words. */
export function collectIssues(result: SaveOutputsResult): string[] {
  const all = [
    ...(result.separate_folders?.errors ?? []),
    ...(result.annotated_copies?.errors ?? []),
    ...(result.recognition_json?.errors ?? []),
    ...(result.csv?.errors ?? []),
    ...(result.xlsx?.errors ?? []),
    ...(result.run_readme?.errors ?? []),
  ];
  return Array.from(new Set(all));
}
