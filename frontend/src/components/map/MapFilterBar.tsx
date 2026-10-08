/**
 * Filter bar for the Map page.
 *
 * 6-column responsive layout:
 *   view mode | sites | date from | date to | labels | map style
 *
 * View mode and map style are baked into the same bar as the data
 * filters for a single coherent filter row, matching the pattern on
 * the Sites and Deployments metadata pages. All state is controlled
 * via props; the parent page owns the filter values and persists
 * them (URL for data filters, localStorage for view mode + base
 * layer).
 */

import {
  InsightsFilterBarShell,
  type FilterChip,
} from "../plots/InsightsFilterChips";
import { useQuery } from "@tanstack/react-query";
import {
  Circle,
  Group,
  Hexagon,
  Map as MapIcon,
  Navigation,
  Satellite,
} from "lucide-react";

import { eventsApi } from "../../api/events";
import { sitesApi } from "../../api/sites";
import { DateRangePicker } from "../ui/date-range-picker";
import { MultiSelect, type MultiSelectOption } from "../ui/multi-select";
import { SegmentedControl } from "../ui/segmented-control";
import { LabelFilterField } from "../verify/LabelFilterField";

export type ViewMode = "hexbins" | "points" | "clusters";
export type BaseLayer = "positron" | "satellite" | "osm";

export interface MapFilters {
  site_ids?: string[];
  date_from?: string;
  date_to?: string;
  labels?: string[];
}

interface MapFilterBarProps {
  /** Active-filter chips, rendered inside the card by the shell. */
  chips: FilterChip[];
  onClearAll: () => void;
  projectId: string;
  filters: MapFilters;
  onChange: (next: MapFilters) => void;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  baseLayer: BaseLayer;
  onBaseLayerChange: (layer: BaseLayer) => void;
}

export function MapFilterBar({
  chips,
  onClearAll,
  projectId,
  filters,
  onChange,
  viewMode,
  onViewModeChange,
  baseLayer,
  onBaseLayerChange,
}: MapFilterBarProps) {

  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId,
  });

  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId),
    enabled: !!projectId,
  });


  const siteOptions: MultiSelectOption[] =
    sites?.map((s) => ({ value: s.id, label: s.name })) ?? [];


  return (
    <InsightsFilterBarShell chips={chips} onClearAll={onClearAll}>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {/* View mode */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            View mode
          </label>
          <SegmentedControl
            options={[
              { value: "hexbins", title: "Hexbins", icon: <Hexagon className="h-4 w-4" /> },
              { value: "points", title: "Points", icon: <Circle className="h-4 w-4" /> },
              { value: "clusters", title: "Clusters", icon: <Group className="h-4 w-4" /> },
            ]}
            value={viewMode}
            onChange={(v) => onViewModeChange(v as ViewMode)}
          />
        </div>

        {/* Sites */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Sites
          </label>
          <MultiSelect
            options={siteOptions}
            value={filters.site_ids ?? []}
            onChange={(v) =>
              onChange({ ...filters, site_ids: v.length ? v : undefined })
            }
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
            from={filters.date_from}
            to={filters.date_to}
            onChange={({ from, to }) =>
              onChange({ ...filters, date_from: from, date_to: to })
            }
            minDate={filterOptions?.date_range?.min}
            maxDate={filterOptions?.date_range?.max}
          />
        </div>

        {/* Labels */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Labels
          </label>
          <LabelFilterField
            projectId={projectId}
            value={filters.labels}
            onChange={(labels) => onChange({ ...filters, labels })}
            siteIds={filters.site_ids}
            dateFrom={filters.date_from}
            dateTo={filters.date_to}
          />
        </div>

        {/* Map style */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Map style
          </label>
          <SegmentedControl
            options={[
              { value: "positron", title: "Light", icon: <MapIcon className="h-4 w-4" /> },
              { value: "satellite", title: "Satellite", icon: <Satellite className="h-4 w-4" /> },
              { value: "osm", title: "OpenStreetMap", icon: <Navigation className="h-4 w-4" /> },
            ]}
            value={baseLayer}
            onChange={(v) => onBaseLayerChange(v as BaseLayer)}
          />
        </div>
      </div>
    </InsightsFilterBarShell>
  );
}
