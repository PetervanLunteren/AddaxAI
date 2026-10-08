/**
 * Filter bar for the dashboard's Explore tab: labels, sites, site tags and
 * date range, in the card every Insights page wears.
 *
 * Labels are picked in the same taxonomy tree modal as on the Counts,
 * Labels and Map pages, and starts the same way: every label ticked. A
 * whole family, a hand-picked set like "livestock", one species, or
 * everything except people are all one control.
 *
 * Controlled: the page owns the values (in the URL) through
 * `useExploreFilters`.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { eventsApi } from "../../api/events";
import type { SiteResponse } from "../../api/types";
import { useNoSiteDeployments } from "../../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../../lib/site-filter-options";
import { speciesLabelMap } from "../../lib/species-name-mode";
import type {
  ExploreFilterPatch,
  ExploreFilters,
} from "../../pages/dashboard/useExploreFilters";
import {
  buildSiteNameMap,
  dateChips,
  InsightsFilterBarShell,
  labelChips,
  siteChips,
  type FilterChip,
} from "../plots/InsightsFilterChips";
import { LabelFilterField } from "../verify/LabelFilterField";
import { DateRangePicker } from "../ui/date-range-picker";
import { MultiSelect } from "../ui/multi-select";

interface ExploreFilterBarProps {
  projectId: string;
  filters: ExploreFilters;
  /** Sites the cards count: the picked ones, narrowed by the tags. The
   *  label tree counts the same sites, so its numbers match the cards. */
  scopeSiteIds: string[] | undefined;
  sites: SiteResponse[] | undefined;
  update: (patch: ExploreFilterPatch) => void;
  clearAll: () => void;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-muted-foreground">{label}</label>
      {children}
    </div>
  );
}

export function ExploreFilterBar({
  projectId,
  filters,
  scopeSiteIds,
  sites,
  update,
  clearAll,
}: ExploreFilterBarProps) {
  const { data: noSite } = useNoSiteDeployments(projectId);

  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId),
  });
  const names = filterOptions ? speciesLabelMap(filterOptions) : undefined;

  const siteOptions = useMemo(
    () => buildSiteOptions(sites, noSite?.count ?? 0),
    [sites, noSite],
  );
  const tagOptions = useMemo(() => {
    const pairs = new Set<string>();
    for (const s of sites ?? []) {
      for (const [k, v] of Object.entries(s.tags ?? {})) {
        if (k.trim() && String(v ?? "").trim()) pairs.add(`${k}:${v}`);
      }
    }
    return [...pairs]
      .sort((a, b) => a.localeCompare(b))
      .map((pair) => ({ value: pair, label: pair.replace(":", ": ") }));
  }, [sites]);


  const chips: FilterChip[] = [
    ...labelChips(filters.labels, names, (next) => update({ labels: next })),
    ...siteChips(filters.siteIds, buildSiteNameMap(sites), (next) =>
      update({ siteIds: next }),
    ),
    ...(filters.tagPairs ?? []).map((pair) => ({
      key: `tag-${pair}`,
      label: `Tag: ${pair.replace(":", ": ")}`,
      onRemove: () =>
        update({ tagPairs: filters.tagPairs!.filter((p) => p !== pair) }),
    })),
    ...dateChips(
      filters.dateFrom,
      filters.dateTo,
      () => update({ dateFrom: undefined }),
      () => update({ dateTo: undefined }),
    ),
  ];

  return (
    <InsightsFilterBarShell chips={chips} onClearAll={clearAll}>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="Labels">
          <LabelFilterField
            projectId={projectId}
            value={filters.labels}
            onChange={(labels) => update({ labels })}
            siteIds={scopeSiteIds}
            dateFrom={filters.dateFrom}
            dateTo={filters.dateTo}
          />
        </Field>
        <Field label="Sites">
          <MultiSelect
            options={siteOptions}
            value={filters.siteIds ?? []}
            onChange={(siteIds) => update({ siteIds })}
            placeholder="All sites"
            searchPlaceholder="Search sites..."
            emptyMessage="No sites found."
            summary={(n) => `${n} site${n > 1 ? "s" : ""}`}
          />
        </Field>
        <Field label="Site tags">
          <MultiSelect
            options={tagOptions}
            value={filters.tagPairs ?? []}
            onChange={(tagPairs) => update({ tagPairs })}
            placeholder="Any tags"
            searchPlaceholder="Search tags..."
            emptyMessage="No site tags."
            summary={(n) => `${n} tag${n > 1 ? "s" : ""}`}
          />
        </Field>
        <Field label="Date range">
          <DateRangePicker
            from={filters.dateFrom}
            to={filters.dateTo}
            onChange={({ from, to }) => update({ dateFrom: from, dateTo: to })}
          />
        </Field>
      </div>
    </InsightsFilterBarShell>
  );
}
