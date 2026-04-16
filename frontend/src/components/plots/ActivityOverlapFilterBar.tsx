/**
 * Filter bar for the Insights → Activity overlap page.
 *
 * Single-row layout mirroring MapFilterBar: six columns on xl screens,
 * collapsing responsively. Columns:
 *   Species A | Species B | Sites | From | To | Display
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

const DATE_INPUT_CLASSES =
  "flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring [&::-webkit-date-and-time-value]:text-left [&::-webkit-calendar-picker-indicator]:absolute [&::-webkit-calendar-picker-indicator]:right-3 relative";

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

  const siteOptions: MultiSelectOption[] = useMemo(
    () => (sites ?? []).map((s) => ({ value: s.id, label: s.name })),
    [sites],
  );

  const update = (patch: Partial<ActivityOverlapPageFilters>) =>
    onChange({ ...filters, ...patch });

  return (
    <div className="rounded-lg border bg-card pt-2 pb-3 px-3 space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {/* Species A */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Species A
          </label>
          <SpeciesPicker
            projectId={projectId}
            value={filters.speciesA}
            onChange={(value) => update({ speciesA: value })}
            placeholder="Pick a species"
            siteIds={filters.siteIds}
            dateFrom={filters.dateFrom ?? undefined}
            dateTo={filters.dateTo ?? undefined}
            excludeValue={filters.speciesB}
            swatchColor={SPECIES_A_COLOR}
          />
        </div>

        {/* Species B */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Species B
          </label>
          <SpeciesPicker
            projectId={projectId}
            value={filters.speciesB}
            onChange={(value) => update({ speciesB: value })}
            placeholder="Pick a second species"
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
