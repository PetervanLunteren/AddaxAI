/**
 * The media copies option state and its pure helpers, shared by the
 * folder-run Save step and the project image copies dialog.
 *
 * A leaf module on purpose: no API client, no DOM, so it can be unit
 * tested and imported from either place without pulling the form hook
 * along. The hook that owns the state is `useMediaOptions`.
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
  /** Leaf ids the user chose to include. Empty = every label, no
   * filter, on both surfaces. The request sends the complement as
   * ``excluded_label_ids``; the data exports ignore it. */
  includedLabelIds: string[];
  /** Which deployments to copy from. Undefined = all of them; an empty
   * list = nothing ticked in the picker, which the backend refuses and
   * the form treats as nothing in scope. Only ever set by the picker,
   * which only shows for a project with more than one deployment. */
  deploymentIds?: string[];
  /** Only the files the user marked with a heart on the Labels page. */
  likedOnly: boolean;
  /** Only the files where every box was verified. */
  verifiedOnly: boolean;
  /** Capture date range, ISO dates, inclusive on whole days. A file
   * without a capture time drops out as soon as either is set. */
  dateFrom?: string;
  dateTo?: string;
}

export interface VisualiseState {
  enabled: boolean;
}

export interface AnonymiseState {
  enabled: boolean;
}

/** The label exclusion set sent to the backend: every leaf NOT included
 * by the user. Empty when all species are included (no filter) or the
 * tree hasn't loaded yet. The data exports ignore this entirely. */
export function excludedLabelIds(
  separate: SeparateState,
  allLabelIds: string[],
): string[] {
  const { includedLabelIds } = separate;
  if (
    includedLabelIds.length === 0 ||
    includedLabelIds.length >= allLabelIds.length
  ) {
    return [];
  }
  const included = new Set(includedLabelIds);
  return allLabelIds.filter((id) => !included.has(id));
}

/** True while the form cannot copy anything: nothing ticked in the
 * deployment picker. Gates the preview query and the Start / Save
 * button, so the backend never sees an empty deployment list. An empty
 * label pick is not this case: it means every label. */
export function nothingInScope(separate: SeparateState): boolean {
  return (
    separate.deploymentIds !== undefined && separate.deploymentIds.length === 0
  );
}

/** The media half of a save request and the whole of a preview request.
 * Mirrors `MediaExportOptions` on the backend: the shared `MediaOptions`
 * fields plus `deployment_ids`. */
export interface MediaOptionsRequest {
  media_threshold: number;
  separate_group_by: SeparateGroupBy;
  group_events: boolean;
  separate_species_last: boolean;
  include_empty: boolean;
  /** Labels to leave out (LabelTaxonomy ids or raw label strings). */
  excluded_label_ids: string[];
  draw_bboxes: boolean;
  anonymise: boolean;
  name_mode: "common" | "scientific";
  favorited_only: boolean;
  verified_only: boolean;
  /** ISO dates (YYYY-MM-DD), inclusive on whole days. */
  date_from?: string;
  date_to?: string;
  /** Omitted = every deployment. Never an empty list: `nothingInScope`
   * keeps the request from being sent in that state. */
  deployment_ids?: string[];
}

/** Map the form state to the request fields, once, for the preview and
 * the save alike, so what the preview counts is what the save writes.
 * Boxes, blur, empties and the label filter are facets of the media
 * copy and only go out when the media output is on (the folder-run
 * card can be off; the dialog's is always on). */
export function mediaRequestFields(
  separate: SeparateState,
  visualise: VisualiseState,
  anonymise: AnonymiseState,
  allLabelIds: string[],
  nameMode: "common" | "scientific",
): MediaOptionsRequest {
  const on = separate.enabled;
  return {
    media_threshold: separate.mediaThreshold,
    separate_group_by: separate.groupBy,
    group_events: separate.groupEvents,
    separate_species_last: separate.speciesLast,
    include_empty: on && separate.copyEmpties,
    excluded_label_ids: on ? excludedLabelIds(separate, allLabelIds) : [],
    draw_bboxes: on && visualise.enabled,
    anonymise: on && anonymise.enabled,
    name_mode: nameMode,
    favorited_only: separate.likedOnly,
    verified_only: separate.verifiedOnly,
    date_from: separate.dateFrom,
    date_to: separate.dateTo,
    deployment_ids: separate.deploymentIds,
  };
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
