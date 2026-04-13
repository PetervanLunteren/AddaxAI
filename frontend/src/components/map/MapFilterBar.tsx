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

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Circle,
  Group,
  Hexagon,
  ListTodo,
  Map as MapIcon,
  Navigation,
  Satellite,
} from "lucide-react";

import { eventsApi } from "../../api/events";
import { sitesApi } from "../../api/sites";
import { Button } from "../ui/button";
import { MultiSelect, type MultiSelectOption } from "../ui/multi-select";
import { LabelFilterModal } from "../verify/LabelFilterModal";
import { cn } from "../../lib/utils";

export type ViewMode = "hexbins" | "points" | "clusters";
export type BaseLayer = "positron" | "satellite" | "osm";

export interface MapFilters {
  site_ids?: string[];
  date_from?: string;
  date_to?: string;
  labels?: string[];
}

interface MapFilterBarProps {
  projectId: string;
  filters: MapFilters;
  onChange: (next: MapFilters) => void;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  baseLayer: BaseLayer;
  onBaseLayerChange: (layer: BaseLayer) => void;
}

const DATE_INPUT_CLASSES =
  "flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring [&::-webkit-date-and-time-value]:text-left [&::-webkit-calendar-picker-indicator]:absolute [&::-webkit-calendar-picker-indicator]:right-3 relative";

export function MapFilterBar({
  projectId,
  filters,
  onChange,
  viewMode,
  onViewModeChange,
  baseLayer,
  onBaseLayerChange,
}: MapFilterBarProps) {
  const [labelModalOpen, setLabelModalOpen] = useState(false);

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

  const { data: labelTree } = useQuery({
    queryKey: ["label-tree", projectId, "events"],
    queryFn: () => eventsApi.getLabelTree(projectId, "events"),
    enabled: !!projectId,
  });
  const hasTaxonomy = !!labelTree?.tree?.length;

  const siteOptions: MultiSelectOption[] =
    sites?.map((s) => ({ value: s.id, label: s.name })) ?? [];

  const labelFlatOptions: MultiSelectOption[] =
    filterOptions?.labels.map((lbl) => ({
      value: lbl,
      label: filterOptions?.display_labels?.[lbl] ?? lbl,
    })) ?? [];

  return (
    <div className="rounded-lg border bg-white pt-2 pb-3 px-3 space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
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

        {/* Date from */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            From
          </label>
          <input
            type="date"
            className={DATE_INPUT_CLASSES}
            value={filters.date_from ?? ""}
            min={
              filterOptions?.date_range
                ? filterOptions.date_range.min.slice(0, 10)
                : undefined
            }
            max={filters.date_to ?? undefined}
            onChange={(e) =>
              onChange({
                ...filters,
                date_from: e.target.value || undefined,
              })
            }
          />
        </div>

        {/* Date to */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            To
          </label>
          <input
            type="date"
            className={DATE_INPUT_CLASSES}
            value={filters.date_to ?? ""}
            min={filters.date_from ?? undefined}
            max={
              filterOptions?.date_range
                ? filterOptions.date_range.max.slice(0, 10)
                : undefined
            }
            onChange={(e) =>
              onChange({
                ...filters,
                date_to: e.target.value || undefined,
              })
            }
          />
        </div>

        {/* Labels */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Labels
          </label>
          {hasTaxonomy ? (
            <>
              <Button
                variant="outline"
                size="sm"
                className="w-full h-9 justify-start text-sm font-normal"
                onClick={() => setLabelModalOpen(true)}
              >
                <ListTodo className="h-4 w-4 mr-2 text-muted-foreground shrink-0" />
                <span className="truncate">
                  {filters.labels?.length
                    ? `${filters.labels.length} labels`
                    : "All labels"}
                </span>
              </Button>
              <LabelFilterModal
                preBuiltTree={labelTree!.tree}
                allLeafIds={labelTree!.all_leaf_ids}
                selectedLabels={filters.labels ?? []}
                onApply={(labels) => {
                  const allLeafs = labelTree!.all_leaf_ids;
                  const isAll = labels.length >= allLeafs.length;
                  onChange({
                    ...filters,
                    labels: isAll
                      ? undefined
                      : labels.length
                        ? labels
                        : undefined,
                  });
                }}
                open={labelModalOpen}
                onOpenChange={setLabelModalOpen}
                countUnit={labelTree!.count_unit}
              />
            </>
          ) : (
            <MultiSelect
              options={labelFlatOptions}
              value={filters.labels ?? []}
              onChange={(v) =>
                onChange({ ...filters, labels: v.length ? v : undefined })
              }
              placeholder="All labels"
              searchPlaceholder="Search labels..."
              emptyMessage="No labels found."
              summary={(n) => `${n} labels`}
              capitalize
            />
          )}
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
    </div>
  );
}

interface SegmentedOption {
  value: string;
  /** Shown as a native hover tooltip via the `title` attribute. */
  title: string;
  icon: React.ReactNode;
}

interface SegmentedControlProps {
  options: SegmentedOption[];
  value: string;
  onChange: (value: string) => void;
}

function SegmentedControl({ options, value, onChange }: SegmentedControlProps) {
  return (
    <div className="flex h-9 w-full rounded-md border border-input bg-background overflow-hidden">
      {options.map((opt, i) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            title={opt.title}
            aria-label={opt.title}
            onClick={() => onChange(opt.value)}
            className={cn(
              "flex-1 inline-flex items-center justify-center transition-colors",
              i > 0 && "border-l border-input",
              active
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            )}
          >
            {opt.icon}
          </button>
        );
      })}
    </div>
  );
}
