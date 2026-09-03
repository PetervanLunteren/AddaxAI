/**
 * Filter state for the Labels page, shared by both of its tabs.
 *
 * Detections and Files show different units (one card per detection, one
 * per file) but they filter the same population by the same things:
 * sites, dates, checked state, and the detection confidence floor. The
 * state lives in the URL under `lbl_*` so switching tabs keeps it, and
 * so a filtered view can be shared as a link.
 *
 * Kept in its own module rather than exported out of `LabelsTab` so the
 * contract between the two tabs is a file you can read, not a reach
 * into a 1,900-line component.
 */

import type {
  EmptyFilter,
  EventFilterParams,
  FavoritedFilter,
  FlaggedFilter,
} from "../../api/types";

export type LabelsVerification = "all" | "unverified" | "verified";

export interface LabelsFilterState {
  site_ids?: string[];
  date_from?: string;
  date_to?: string;
  labels?: string[];
  min_confidence?: number;
  max_confidence?: number;
  min_label_confidence?: number;
  max_label_confidence?: number;
  /** Default "unverified" when omitted — verified detections are usually
   *  not what the user is looking at on this tab. */
  verification?: LabelsVerification;
  /** Files tab only. Default "all" when omitted: every file, empty or
   *  not. "show_only" is the old Empties tab. */
  empty?: EmptyFilter;
  /** File-level triage marks, the Counts filters on Labels. Omitted
   *  means "all". */
  flagged?: FlaggedFilter;
  favorited?: FavoritedFilter;
}

/** Parse lbl_* params from URL. */
export function lblFiltersFromSearchParams(
  sp: URLSearchParams,
): LabelsFilterState {
  const f: LabelsFilterState = {};
  const sites = sp.get("lbl_sites");
  if (sites) f.site_ids = sites.split(",");
  const from = sp.get("lbl_from");
  if (from) f.date_from = from;
  const to = sp.get("lbl_to");
  if (to) f.date_to = to;
  const labels = sp.get("lbl_labels");
  if (labels) f.labels = labels.split(",");
  const minC = sp.get("lbl_min_confidence");
  if (minC !== null) f.min_confidence = parseFloat(minC);
  const maxC = sp.get("lbl_max_confidence");
  if (maxC !== null) f.max_confidence = parseFloat(maxC);
  const minLC = sp.get("lbl_min_label_confidence");
  if (minLC !== null) f.min_label_confidence = parseFloat(minLC);
  const maxLC = sp.get("lbl_max_label_confidence");
  if (maxLC !== null) f.max_label_confidence = parseFloat(maxLC);
  const ver = sp.get("lbl_verification");
  if (ver === "all" || ver === "unverified" || ver === "verified") {
    f.verification = ver;
  }
  const empty = sp.get("lbl_empty");
  if (empty === "all" || empty === "show_only" || empty === "hide") {
    f.empty = empty;
  }
  const flagged = sp.get("lbl_flagged");
  if (flagged === "flagged" || flagged === "not_flagged") {
    f.flagged = flagged;
  }
  const favorited = sp.get("lbl_favorited");
  if (favorited === "favorited" || favorited === "not_favorited") {
    f.favorited = favorited;
  }
  return f;
}

/** Write lbl_* params to URL, preserving non-lbl params. */
export function lblFiltersToSearchParams(
  filters: LabelsFilterState,
  current: URLSearchParams,
): URLSearchParams {
  const sp = new URLSearchParams(current);
  for (const key of [...sp.keys()]) {
    if (key.startsWith("lbl_")) sp.delete(key);
  }
  if (filters.site_ids?.length) sp.set("lbl_sites", filters.site_ids.join(","));
  if (filters.date_from) sp.set("lbl_from", filters.date_from);
  if (filters.date_to) sp.set("lbl_to", filters.date_to);
  if (filters.labels?.length) sp.set("lbl_labels", filters.labels.join(","));
  if (filters.min_confidence !== undefined)
    sp.set("lbl_min_confidence", String(filters.min_confidence));
  if (filters.max_confidence !== undefined)
    sp.set("lbl_max_confidence", String(filters.max_confidence));
  if (filters.min_label_confidence !== undefined)
    sp.set("lbl_min_label_confidence", String(filters.min_label_confidence));
  if (filters.max_label_confidence !== undefined)
    sp.set("lbl_max_label_confidence", String(filters.max_label_confidence));
  // "unverified" is the implicit default — no URL param when set to that.
  if (filters.verification && filters.verification !== "unverified") {
    sp.set("lbl_verification", filters.verification);
  }
  // "all" is the implicit default; the bar writes undefined for it.
  if (filters.empty) sp.set("lbl_empty", filters.empty);
  if (filters.flagged && filters.flagged !== "all") {
    sp.set("lbl_flagged", filters.flagged);
  }
  if (filters.favorited && filters.favorited !== "all") {
    sp.set("lbl_favorited", filters.favorited);
  }
  return sp;
}

/** Adapt to the shape `VerifyFilterBar` speaks. */
export function toFilterBarFilters(f: LabelsFilterState): EventFilterParams {
  return {
    site_ids: f.site_ids,
    date_from: f.date_from,
    date_to: f.date_to,
    labels: f.labels,
    min_confidence: f.min_confidence,
    max_confidence: f.max_confidence,
    min_label_confidence: f.min_label_confidence,
    max_label_confidence: f.max_label_confidence,
    // Raw, no default materialized: the bar resolves the resting
    // value itself and the chips must only see explicit filters.
    verification: f.verification,
    empty: f.empty,
    flagged: f.flagged,
    favorited: f.favorited,
  };
}

/** Adapt back from what `VerifyFilterBar` emits. */
export function fromFilterBarFilters(
  fp: EventFilterParams,
  current: LabelsFilterState,
): LabelsFilterState {
  return {
    ...current,
    site_ids: fp.site_ids,
    date_from: fp.date_from,
    date_to: fp.date_to,
    labels: fp.labels,
    min_confidence: fp.min_confidence,
    max_confidence: fp.max_confidence,
    min_label_confidence: fp.min_label_confidence,
    max_label_confidence: fp.max_label_confidence,
    verification: fp.verification as LabelsVerification | undefined,
    empty: fp.empty,
    flagged: fp.flagged,
    favorited: fp.favorited,
  };
}
