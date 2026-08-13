/**
 * Filter bar for the Insights → Deployment timeline page.
 *
 * 4-column responsive layout: sites | date range | sort | view + density,
 * where the last column holds both display toggles side by side, each
 * with its own label.
 */

import { useMemo } from "react";
import {
  InsightsFilterBarShell,
  type FilterChip,
} from "./InsightsFilterChips";
import { useQuery } from "@tanstack/react-query";
import { ChartGantt, Grid3x3, Rows2, Rows4 } from "lucide-react";

import { sitesApi } from "../../api/sites";
import { useNoSiteDeployments } from "../../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../../lib/site-filter-options";
import { DateRangePicker } from "../ui/date-range-picker";
import { MultiSelect, type MultiSelectOption } from "../ui/multi-select";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { SegmentedControl } from "../ui/segmented-control";

export type TimelineSort =
  | "alpha"
  | "chrono"
  | "trap-nights"
  | "deployments"
  | "recent";

export const TIMELINE_SORT_OPTIONS: { value: TimelineSort; label: string }[] = [
  { value: "alpha", label: "Site name (alphabetically)" },
  { value: "chrono", label: "First deployment (earliest)" },
  { value: "trap-nights", label: "Trap-nights (most first)" },
  { value: "deployments", label: "Deployments (most first)" },
  { value: "recent", label: "Most recent activity" },
];

export type TimelineDensity = "normal" | "compact";

/** Bars show when a site was monitored, the heatmap how much it captured. */
export type TimelineViewMode = "bars" | "heatmap";

export interface TimelinePageFilters {
  siteIds: string[];
  dateFrom: string | null;
  dateTo: string | null;
  sort: TimelineSort;
  density: TimelineDensity;
  viewMode: TimelineViewMode;
}

interface DeploymentTimelineFilterBarProps {
  /** Active-filter chips, rendered inside the card by the shell. */
  chips: FilterChip[];
  onClearAll: () => void;
  projectId: string;
  filters: TimelinePageFilters;
  onChange: (next: TimelinePageFilters) => void;
}

export function DeploymentTimelineFilterBar({
  chips,
  onClearAll,
  projectId,
  filters,
  onChange,
}: DeploymentTimelineFilterBarProps) {
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

  const update = (patch: Partial<TimelinePageFilters>) =>
    onChange({ ...filters, ...patch });

  return (
    <InsightsFilterBarShell chips={chips} onClearAll={onClearAll}>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
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

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Date range
          </label>
          <DateRangePicker
            from={filters.dateFrom}
            to={filters.dateTo}
            onChange={({ from, to }) =>
              update({ dateFrom: from ?? null, dateTo: to ?? null })
            }
          />
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Sort
          </label>
          <Select
            value={filters.sort}
            onValueChange={(v) => update({ sort: v as TimelineSort })}
          >
            <SelectTrigger className="h-9 min-h-9">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TIMELINE_SORT_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Two controls sharing one grid cell, each keeping its own label.
            Four cells fit four columns at every width, so the row never
            wraps, and an icon-only toggle still says what it controls. */}
        <div className="flex gap-2">
          <div className="flex-1 space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              View
            </label>
            <SegmentedControl
              options={[
                {
                  value: "bars",
                  title: "Bars: when each site was monitored",
                  icon: <ChartGantt className="h-4 w-4" />,
                },
                {
                  value: "heatmap",
                  title: "Heatmap: how many files arrived each day",
                  icon: <Grid3x3 className="h-4 w-4" />,
                },
              ]}
              value={filters.viewMode}
              onChange={(v) => update({ viewMode: v as TimelineViewMode })}
            />
          </div>
          <div className="flex-1 space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              Density
            </label>
            <SegmentedControl
              options={[
                {
                  value: "normal",
                  title: "Normal: site names visible",
                  icon: <Rows2 className="h-4 w-4" />,
                },
                {
                  value: "compact",
                  title: "Compact: thin lines, no site labels",
                  icon: <Rows4 className="h-4 w-4" />,
                },
              ]}
              value={filters.density}
              onChange={(v) => update({ density: v as TimelineDensity })}
            />
          </div>
        </div>
      </div>
    </InsightsFilterBarShell>
  );
}
