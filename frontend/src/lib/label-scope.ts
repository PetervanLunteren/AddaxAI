/**
 * The scope of the Labels page: where and when, apart from the selection.
 *
 * Site and date say which part of the project someone works on. Every
 * other filter (label, verification, the confidence ranges, flagged,
 * liked) says what they look at. A focused view such as the suggestions
 * review brings its own selection, so it replaces theirs, but it never
 * changes where they work: the scope goes in with them.
 *
 * The suggestions vote runs inside the scope. A label or Verified
 * filter would remove voters by the very thing they vote on; a site or
 * a date range removes them evenly across labels, so the vote holds.
 *
 * This module is the one place on the frontend that says which filters
 * are scope (`LabelFilters.scope_only` is its backend twin) and how a
 * scope is put into words where the filter bar is hidden.
 */

import type { LabelFilters } from "../api/types";
import { formatShortDate } from "./datetime";
import { NO_SITE_SENTINEL } from "./filter-url";
import { NO_SITE_LABEL } from "./site-filter-options";

export type LabelScope = Pick<LabelFilters, "site_ids" | "date_from" | "date_to">;

/** The scope part of a filter state; the selection is left behind. */
export function scopeFilters(f: LabelScope): LabelScope {
  return {
    site_ids: f.site_ids,
    date_from: f.date_from,
    date_to: f.date_to,
  };
}

function sitesText(siteIds: string[], siteNames: Record<string, string>): string {
  if (siteIds.length === 0) return "all sites";
  if (siteIds.length > 1) return `${siteIds.length} sites`;
  const [id] = siteIds;
  return id === NO_SITE_SENTINEL ? NO_SITE_LABEL : (siteNames[id] ?? "1 site");
}

function datesText(scope: LabelScope, locale?: string): string {
  const from = formatShortDate(scope.date_from, locale);
  const to = formatShortDate(scope.date_to, locale);
  if (from && to) return `${from} to ${to}`;
  if (from) return `from ${from}`;
  if (to) return `until ${to}`;
  return "all dates";
}

/**
 * The scope in words, to follow a view's name after a comma:
 * "in all sites, all dates", "in Kenai North, from 1 Jun 2026",
 * "in 3 sites, 1 Jun 2026 to 30 Sep 2026". It always names both parts,
 * so "everything" is said rather than left to be assumed.
 *
 * A folder run has no sites (`showSites` false), so only the dates are
 * named there. `locale` is for tests; the app uses the viewer's.
 */
export function scopeText(
  scope: LabelScope,
  siteNames: Record<string, string>,
  showSites: boolean,
  locale?: string,
): string {
  const dates = datesText(scope, locale);
  if (!showSites) return dates;
  return `in ${sitesText(scope.site_ids ?? [], siteNames)}, ${dates}`;
}
