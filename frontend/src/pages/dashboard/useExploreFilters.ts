/**
 * The Explore tab's filter state, read from and written to the URL, and
 * turned into the one `DashboardScope` every Explore card receives.
 *
 * The URL is the only store, so a link, the back button and switching
 * between the dashboard tabs all keep the selection.
 */

import { useCallback, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";

import { sitesApi } from "../../api/sites";
import type { DashboardScope } from "../../api/statistics";
import {
  EXPLORE_FILTER_SCHEMA,
  foldTagsIntoSites,
} from "../../lib/explore-filters";
import {
  filtersFromSearchParams,
  filtersToSearchParams,
} from "../../lib/filter-url";

export interface ExploreFilters {
  labels?: string[];
  siteIds?: string[];
  tagPairs?: string[];
  dateFrom?: string;
  dateTo?: string;
}

export type ExploreFilterPatch = Partial<
  Record<keyof ExploreFilters, string | string[] | undefined>
>;

const URL_KEYS: Record<keyof ExploreFilters, string> = {
  labels: "labels",
  siteIds: "site_ids",
  tagPairs: "tags",
  dateFrom: "date_from",
  dateTo: "date_to",
};

const asString = (v: string | string[] | undefined) =>
  typeof v === "string" ? v : undefined;
const asList = (v: string | string[] | undefined) =>
  Array.isArray(v) ? v : undefined;

export function useExploreFilters(projectId: string) {
  const [searchParams, setSearchParams] = useSearchParams();
  const values = useMemo(
    () => filtersFromSearchParams(searchParams, EXPLORE_FILTER_SCHEMA),
    [searchParams],
  );

  const filters: ExploreFilters = {
    labels: asList(values.labels),
    siteIds: asList(values.site_ids),
    tagPairs: asList(values.tags),
    dateFrom: asString(values.date_from),
    dateTo: asString(values.date_to),
  };

  const update = useCallback(
    (patch: ExploreFilterPatch) => {
      const next = { ...values };
      for (const [field, value] of Object.entries(patch)) {
        next[URL_KEYS[field as keyof ExploreFilters]] = value as string | string[];
      }
      setSearchParams(filtersToSearchParams(next, EXPLORE_FILTER_SCHEMA), {
        replace: true,
      });
    },
    [values, setSearchParams],
  );

  const clearAll = useCallback(
    () => setSearchParams(new URLSearchParams(), { replace: true }),
    [setSearchParams],
  );

  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
  });

  const tagsWaiting = !!filters.tagPairs?.length && sites === undefined;
  const siteIds = tagsWaiting
    ? undefined
    : foldTagsIntoSites(sites ?? [], filters.siteIds, filters.tagPairs);

  const scope = useMemo<DashboardScope>(
    () => ({
      labelTaxonomyIds: filters.labels,
      siteIds,
      dateFrom: filters.dateFrom,
      dateTo: filters.dateTo,
    }),
    // Arrays are fresh each render; their contents are the key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [filters.labels?.join(","), siteIds?.join(","), filters.dateFrom, filters.dateTo],
  );

  return {
    filters,
    update,
    clearAll,
    sites,
    scope,
    /** False until the tag folding can be worked out. */
    ready: !tagsWaiting,
    /** The site and tag filters together match no site at all. */
    noSiteMatch: Array.isArray(siteIds) && siteIds.length === 0,
  };
}
