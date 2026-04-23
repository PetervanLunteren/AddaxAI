/**
 * Filter bar for the Insights → Deployment timeline page.
 *
 * 4-column responsive layout: sites | from | to | sort.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Rows2, Rows4 } from "lucide-react";

import { sitesApi } from "../../api/sites";
import { useNoSiteDeployments } from "../../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../../lib/site-filter-options";
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
  { value: "alpha", label: "Site name (A → Z)" },
  { value: "chrono", label: "First deployment (earliest)" },
  { value: "trap-nights", label: "Trap-nights (most first)" },
  { value: "deployments", label: "Deployments (most first)" },
  { value: "recent", label: "Most recent activity" },
];

export type TimelineDensity = "normal" | "compact";

export interface TimelinePageFilters {
  siteIds: string[];
  dateFrom: string | null;
  dateTo: string | null;
  sort: TimelineSort;
  density: TimelineDensity;
}

interface DeploymentTimelineFilterBarProps {
  projectId: string;
  filters: TimelinePageFilters;
  onChange: (next: TimelinePageFilters) => void;
}

const DATE_INPUT_CLASSES =
  "flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring [&::-webkit-date-and-time-value]:text-left [&::-webkit-calendar-picker-indicator]:absolute [&::-webkit-calendar-picker-indicator]:right-3 relative";

export function DeploymentTimelineFilterBar({
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
    <div className="rounded-lg border bg-card pt-2 pb-3 px-3 space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
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

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Sort
          </label>
          <Select
            value={filters.sort}
            onValueChange={(v) => update({ sort: v as TimelineSort })}
          >
            <SelectTrigger className="h-9">
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

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Density
          </label>
          <SegmentedControl
            options={[
              {
                value: "normal",
                title: "Normal — site names visible",
                icon: <Rows2 className="h-4 w-4" />,
              },
              {
                value: "compact",
                title: "Compact — thin lines, no site labels",
                icon: <Rows4 className="h-4 w-4" />,
              },
            ]}
            value={filters.density}
            onChange={(v) => update({ density: v as TimelineDensity })}
          />
        </div>
      </div>
    </div>
  );
}
