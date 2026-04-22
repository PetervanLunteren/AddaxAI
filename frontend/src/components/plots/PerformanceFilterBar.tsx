/**
 * Filter bar shared by the confusion matrix and classification report.
 *
 * Controlled: the parent page owns the filter values and persists them
 * to the URL. The Normalise toggle is matrix-specific (the report is
 * always a ratio view); it's rendered only when `showNormalise` is on.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { sitesApi } from "../../api/sites";
import { useNoSiteDeployments } from "../../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../../lib/site-filter-options";
import type { TaxonomicRank } from "../../lib/taxonomic-rank";
import { MultiSelect, type MultiSelectOption } from "../ui/multi-select";
import { SegmentedControl } from "../ui/segmented-control";
import { TaxonomicRankPicker } from "../ui/taxonomic-rank-picker";

export type TopN = "10" | "20" | "all";
export type MatrixMode = "counts" | "recall" | "precision";

export interface PerformancePageFilters {
  siteIds: string[];
  taxonomicRank: TaxonomicRank;
  topN: TopN;
  mode: MatrixMode;
}

interface PerformanceFilterBarProps {
  projectId: string;
  filters: PerformancePageFilters;
  onChange: (next: PerformancePageFilters) => void;
  /** When true, render the matrix display-mode toggle. Matrix only. */
  showModeToggle?: boolean;
}

function textIcon(text: string) {
  return <span className="text-xs font-medium px-1">{text}</span>;
}

const TOP_N_OPTIONS = [
  { value: "10", title: "Top 10 classes", icon: textIcon("top 10") },
  { value: "20", title: "Top 20 classes", icon: textIcon("top 20") },
  { value: "all", title: "All classes", icon: textIcon("all") },
];

const MODE_OPTIONS = [
  {
    value: "counts",
    title: "Absolute counts",
    icon: textIcon("counts"),
  },
  {
    value: "recall",
    title: "Recall — row-normalised (diagonal = per-class recall)",
    icon: textIcon("recall"),
  },
  {
    value: "precision",
    title: "Precision — column-normalised (diagonal = per-class precision)",
    icon: textIcon("precision"),
  },
];

export function PerformanceFilterBar({
  projectId,
  filters,
  onChange,
  showModeToggle = false,
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

  const columns = showModeToggle ? 4 : 3;

  return (
    <div className="rounded-lg border bg-card pt-2 pb-3 px-3 space-y-4">
      <div
        className="grid grid-cols-1 sm:grid-cols-2 gap-4"
        style={{
          gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
        }}
      >
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Taxonomic rank
          </label>
          <TaxonomicRankPicker
            value={filters.taxonomicRank}
            onChange={(v) => update({ taxonomicRank: v })}
          />
        </div>

        {showModeToggle && (
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              Display
            </label>
            <SegmentedControl
              options={MODE_OPTIONS}
              value={filters.mode}
              onChange={(v) => update({ mode: v as MatrixMode })}
            />
          </div>
        )}

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
      </div>
    </div>
  );
}
