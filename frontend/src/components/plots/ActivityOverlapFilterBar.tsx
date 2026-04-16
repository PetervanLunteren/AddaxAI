/**
 * Filter bar for the Plots → Activity overlap page.
 *
 * Top row: two SpeciesPickers (A and B), inline. The page is built
 * around comparing two species so this is the primary control surface.
 *
 * Second row: sites multi-select + date range + time-axis toggle +
 * twilight-bands toggle. Sites/dates feed the API; the toggles are
 * pure presentation state managed by the parent.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Filter, X } from "lucide-react";

import { sitesApi } from "../../api/sites";
import { Button } from "../ui/button";
import { Label } from "../ui/label";
import { MultiSelect, type MultiSelectOption } from "../ui/multi-select";
import { Switch } from "../ui/switch";
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
  bandsVisible: boolean;
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
  const [popoverOpen, setPopoverOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  // Click-outside to close the sites/dates popover.
  useEffect(() => {
    const handleMouseDown = (e: MouseEvent) => {
      if (!popoverRef.current) return;
      const target = e.target as Node;
      if (popoverRef.current.contains(target)) return;
      if ((target as HTMLElement).closest?.("[data-radix-popper-content-wrapper]")) return;
      setPopoverOpen(false);
    };
    document.addEventListener("mousedown", handleMouseDown);
    return () => document.removeEventListener("mousedown", handleMouseDown);
  }, []);

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

  const activeFilterCount =
    (filters.siteIds.length > 0 ? 1 : 0) +
    (filters.dateFrom ? 1 : 0) +
    (filters.dateTo ? 1 : 0);

  const clearSitesAndDates = () => {
    update({ siteIds: [], dateFrom: null, dateTo: null });
  };

  return (
    <div className="space-y-3 rounded-lg border bg-card p-4">
      {/* Row 1: two species pickers */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label className="text-xs uppercase tracking-wide text-muted-foreground">
            Species A
          </Label>
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
        <div className="space-y-1.5">
          <Label className="text-xs uppercase tracking-wide text-muted-foreground">
            Species B (optional)
          </Label>
          <SpeciesPicker
            projectId={projectId}
            value={filters.speciesB}
            onChange={(value) => update({ speciesB: value })}
            placeholder="Pick a second species to compare"
            siteIds={filters.siteIds}
            dateFrom={filters.dateFrom ?? undefined}
            dateTo={filters.dateTo ?? undefined}
            excludeValue={filters.speciesA}
            swatchColor={SPECIES_B_COLOR}
          />
        </div>
      </div>

      {/* Row 2: sites/dates popover, time axis, bands */}
      <div className="flex flex-wrap items-center gap-3">
        <div ref={popoverRef} className="relative">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setPopoverOpen((v) => !v)}
            className="flex items-center gap-2"
          >
            <Filter className="h-4 w-4" />
            Filters
            {activeFilterCount > 0 && (
              <span className="rounded-full bg-primary px-1.5 py-0.5 text-xs text-primary-foreground">
                {activeFilterCount}
              </span>
            )}
          </Button>
          {popoverOpen && (
            <div className="absolute left-0 z-50 mt-2 w-96 space-y-4 rounded-md border bg-background p-4 shadow-lg">
              <div className="space-y-2">
                <Label className="text-sm font-medium">Sites</Label>
                <MultiSelect
                  options={siteOptions}
                  value={filters.siteIds}
                  onChange={(siteIds) => update({ siteIds })}
                  placeholder="All sites"
                  searchPlaceholder="Search sites..."
                  popoverWidth="w-[350px]"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-sm font-medium">Date range</Label>
                <div className="flex items-center gap-2">
                  <input
                    type="date"
                    value={filters.dateFrom ?? ""}
                    onChange={(e) =>
                      update({ dateFrom: e.target.value || null })
                    }
                    className="flex-1 h-9 rounded-md border border-input bg-background px-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                  <span className="text-sm text-muted-foreground">to</span>
                  <input
                    type="date"
                    value={filters.dateTo ?? ""}
                    onChange={(e) =>
                      update({ dateTo: e.target.value || null })
                    }
                    className="flex-1 h-9 rounded-md border border-input bg-background px-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>
              </div>
              {activeFilterCount > 0 && (
                <button
                  type="button"
                  onClick={clearSitesAndDates}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:underline"
                >
                  <X className="h-3 w-3" />
                  Clear sites and dates
                </button>
              )}
            </div>
          )}
        </div>

        {/* Time axis toggle (clock vs sun-time) */}
        <div className="flex items-center gap-2 rounded-md border bg-background p-1">
          <button
            type="button"
            onClick={() => update({ timeAxis: "clock" })}
            className={
              filters.timeAxis === "clock"
                ? "rounded px-2.5 py-1 text-xs font-medium bg-primary text-primary-foreground"
                : "rounded px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground"
            }
          >
            Clock time
          </button>
          <button
            type="button"
            onClick={() => update({ timeAxis: "sun" })}
            className={
              filters.timeAxis === "sun"
                ? "rounded px-2.5 py-1 text-xs font-medium bg-primary text-primary-foreground"
                : "rounded px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground"
            }
          >
            Sun time
          </button>
        </div>

        {/* Twilight bands toggle. In clock mode the bands come from a
            single-reference date; in sun mode they come from the mean
            anchor dawn / sunrise / sunset / dusk, so the toggle is
            meaningful in both axes. */}
        <div className="flex items-center gap-2">
          <Switch
            id="twilight-bands"
            checked={filters.bandsVisible}
            onCheckedChange={(checked) => update({ bandsVisible: checked })}
          />
          <Label
            htmlFor="twilight-bands"
            className="text-xs text-muted-foreground cursor-pointer"
          >
            Twilight bands
          </Label>
        </div>
      </div>
    </div>
  );
}
