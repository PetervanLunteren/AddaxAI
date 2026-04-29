/**
 * Filter bar for the Insights → Activity overlap page.
 *
 * Single-row layout mirroring MapFilterBar: six columns on xl screens,
 * collapsing responsively. Columns:
 *   Label A | Label B | Sites | From | To | Display
 *
 * The Display column stacks a clock/sun segmented control on top of
 * the twilight-bands switch so the visible row has a constant height
 * roughly matching the other columns' single controls.
 *
 * All state is controlled via props; the parent page owns the filter
 * values and persists them (URL for data filters, URL for the axis
 * and bands-visible flags).
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Clock, Sun } from "lucide-react";

import { sitesApi } from "../../api/sites";
import { useNoSiteDeployments } from "../../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../../lib/site-filter-options";
import { DateRangePicker } from "../ui/date-range-picker";
import { MultiSelect, type MultiSelectOption } from "../ui/multi-select";
import { SegmentedControl } from "../ui/segmented-control";
import { SpeciesPicker } from "./SpeciesPicker";
import { SPECIES_A_COLOR, SPECIES_B_COLOR } from "./ActivityOverlapChart";

export type TimeAxis = "clock" | "sun";

export interface ActivityOverlapPageFilters {
  speciesA: string | null;
  speciesB: string | null;
  siteIds: string[];
  dateFrom: string | null;
  dateTo: string | null;
  timeAxis: TimeAxis;
}

interface ActivityOverlapFilterBarProps {
  projectId: string;
  filters: ActivityOverlapPageFilters;
  onChange: (next: ActivityOverlapPageFilters) => void;
}

export function ActivityOverlapFilterBar({
  projectId,
  filters,
  onChange,
}: ActivityOverlapFilterBarProps) {
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

  const update = (patch: Partial<ActivityOverlapPageFilters>) =>
    onChange({ ...filters, ...patch });

  return (
    <div className="rounded-lg border bg-card pt-2 pb-3 px-3 space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {/* Label A */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Label A
          </label>
          <SpeciesPicker
            projectId={projectId}
            value={filters.speciesA}
            onChange={(value) => update({ speciesA: value })}
            placeholder="Pick a label"
            siteIds={filters.siteIds}
            dateFrom={filters.dateFrom ?? undefined}
            dateTo={filters.dateTo ?? undefined}
            excludeValue={filters.speciesB}
            swatchColor={SPECIES_A_COLOR}
          />
        </div>

        {/* Label B */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Label B
          </label>
          <SpeciesPicker
            projectId={projectId}
            value={filters.speciesB}
            onChange={(value) => update({ speciesB: value })}
            placeholder="Pick a second label"
            siteIds={filters.siteIds}
            dateFrom={filters.dateFrom ?? undefined}
            dateTo={filters.dateTo ?? undefined}
            excludeValue={filters.speciesA}
            swatchColor={SPECIES_B_COLOR}
          />
        </div>

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

        {/* Time axis */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Time axis
          </label>
          <SegmentedControl
            options={[
              { value: "clock", title: "Clock time", icon: <Clock className="h-4 w-4" /> },
              { value: "sun", title: "Sun time", icon: <Sun className="h-4 w-4" /> },
            ]}
            value={filters.timeAxis}
            onChange={(v) => update({ timeAxis: v as TimeAxis })}
          />
        </div>
      </div>
    </div>
  );
}
