/**
 * Filter bar shared by the confusion matrix and classification report.
 *
 * Controlled: the parent page owns the filter values and persists them
 * to the URL. The segmented controls for rank and top-N match the
 * existing SegmentedControl usage on the Activity overlap page.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { sitesApi } from "../../api/sites";
import { useNoSiteDeployments } from "../../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../../lib/site-filter-options";
import { MultiSelect, type MultiSelectOption } from "../ui/multi-select";
import { SegmentedControl } from "../ui/segmented-control";
import type { PerformanceRank } from "../../api/performance";

export type TopN = "10" | "20" | "50" | "all";

export interface PerformancePageFilters {
  siteIds: string[];
  dateFrom: string | null;
  dateTo: string | null;
  rank: PerformanceRank;
  topN: TopN;
}

interface PerformanceFilterBarProps {
  projectId: string;
  filters: PerformancePageFilters;
  onChange: (next: PerformancePageFilters) => void;
}

const DATE_INPUT_CLASSES =
  "flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring [&::-webkit-date-and-time-value]:text-left [&::-webkit-calendar-picker-indicator]:absolute [&::-webkit-calendar-picker-indicator]:right-3 relative";

function textIcon(text: string) {
  return <span className="text-xs font-medium px-1">{text}</span>;
}

const RANK_OPTIONS = [
  { value: "class", title: "Class", icon: textIcon("class") },
  { value: "order", title: "Order", icon: textIcon("order") },
  { value: "family", title: "Family", icon: textIcon("family") },
  { value: "genus", title: "Genus", icon: textIcon("genus") },
  { value: "species", title: "Species", icon: textIcon("species") },
];

const TOP_N_OPTIONS = [
  { value: "10", title: "Top 10", icon: textIcon("10") },
  { value: "20", title: "Top 20", icon: textIcon("20") },
  { value: "50", title: "Top 50", icon: textIcon("50") },
  { value: "all", title: "All classes", icon: textIcon("all") },
];

export function PerformanceFilterBar({
  projectId,
  filters,
  onChange,
}: PerformanceFilterBarProps) {
  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId,
  });
  const { data: noSite } = useNoSiteDeployments(projectId);

  const siteOptions: MultiSelectOption[] = useMemo(
    () => buildSiteOptions(sites, noSite?.count ?? 0),
    [sites, noSite],
  );

  const update = (patch: Partial<PerformancePageFilters>) =>
    onChange({ ...filters, ...patch });

  return (
    <div className="rounded-lg border bg-card pt-2 pb-3 px-3 space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        {/* Sites */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Sites
          </label>
          <MultiSelect
            options={siteOptions}
            value={filters.siteIds}
            onChange={(siteIds) => update({ siteIds })}
            placeholder="All sites"
            searchPlaceholder="Search sites..."
            emptyMessage="No sites found."
            summary={(n) => `${n} site${n > 1 ? "s" : ""}`}
          />
        </div>

        {/* From */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            From
          </label>
          <input
            type="date"
            className={DATE_INPUT_CLASSES}
            value={filters.dateFrom ?? ""}
            max={filters.dateTo ?? undefined}
            onChange={(e) => update({ dateFrom: e.target.value || null })}
          />
        </div>

        {/* To */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            To
          </label>
          <input
            type="date"
            className={DATE_INPUT_CLASSES}
            value={filters.dateTo ?? ""}
            min={filters.dateFrom ?? undefined}
            onChange={(e) => update({ dateTo: e.target.value || null })}
          />
        </div>

        {/* Rank */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Rank
          </label>
          <SegmentedControl
            options={RANK_OPTIONS}
            value={filters.rank}
            onChange={(v) => update({ rank: v as PerformanceRank })}
          />
        </div>

        {/* Top N */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Show
          </label>
          <SegmentedControl
            options={TOP_N_OPTIONS}
            value={filters.topN}
            onChange={(v) => update({ topN: v as TopN })}
          />
        </div>
      </div>
    </div>
  );
}
