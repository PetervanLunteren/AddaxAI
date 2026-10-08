/**
 * The dashboard Explore tab's filters: URL schema and the rules that turn
 * what is in the URL into a query scope.
 *
 * Labels are picked in the same taxonomy tree as on the Counts, Labels and
 * Map pages, and travel in the URL the same way the Map's do (`labels=`,
 * taxonomy leaf ids), so one selection means the same thing everywhere.
 */

import type { DashboardScope } from "../api/statistics";
import { filtersToSearchParams, type FilterSchema } from "./filter-url";

export const EXPLORE_FILTER_SCHEMA: FilterSchema = {
  labels: "string[]",
  site_ids: "string[]",
  tags: "string[]",
  date_from: "date",
  date_to: "date",
};

interface SiteWithTags {
  id: string;
  tags?: Record<string, string> | null;
}

/**
 * Combine the Sites and Site tags filters into one list of site ids.
 *
 * `undefined` means no site filter at all. An empty array means the two
 * filters together match no site, which callers must show as an empty
 * selection: passing it on as "no filter" would quietly show every site.
 * A tag pair is written "key:value"; a site matches when it has any of
 * the chosen pairs, and both filters must hold.
 */
export function foldTagsIntoSites(
  sites: SiteWithTags[],
  siteIds: string[] | undefined,
  tagPairs: string[] | undefined,
): string[] | undefined {
  if (!tagPairs || tagPairs.length === 0) {
    return siteIds && siteIds.length > 0 ? siteIds : undefined;
  }
  const picked = new Set(tagPairs);
  const tagged = new Set(
    sites
      .filter((s) =>
        Object.entries(s.tags ?? {}).some(([k, v]) => picked.has(`${k}:${v}`)),
      )
      .map((s) => s.id),
  );
  if (!siteIds || siteIds.length === 0) return [...tagged];
  return siteIds.filter((id) => tagged.has(id));
}

/** The Explore URL for a set of labels, used by the Overview chart's bars. */
export function exploreHref(projectId: string, labelIds: string[]): string {
  const params = filtersToSearchParams({ labels: labelIds }, EXPLORE_FILTER_SCHEMA);
  return `/projects/${projectId}/dashboard/explore?${params}`;
}

/** The Insights map with the same labels, sites and dates. */
export function insightsMapHref(projectId: string, scope: DashboardScope): string {
  const params = filtersToSearchParams(
    {
      labels: scope.labelTaxonomyIds,
      site_ids: scope.siteIds,
      date_from: scope.dateFrom,
      date_to: scope.dateTo,
    },
    { labels: "string[]", site_ids: "string[]", date_from: "date", date_to: "date" },
  );
  return `/projects/${projectId}/insights/map?${params}`;
}
