/**
 * "More" popover for the verify tabs.
 *
 * Hosts the rare filters that don't justify a permanent slot in the
 * filter row: liked / flagged / empty (Events and Files only) plus the
 * detection / classification confidence ranges (every tab). Sort is not
 * a filter and lives in the second toolbar instead.
 *
 * Pattern modelled on `dashboard/DashboardFilters.tsx`: outline button
 * trigger with an active-count badge, popover content stacks the
 * controls. The chip row above the toolbar provides the canonical
 * "clear all" affordance — no duplicate inside this popover.
 */

import { useState } from "react";
import { Filter } from "lucide-react";

import { Button } from "../ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { ConfidenceRangeFilter } from "./ConfidenceRangeFilter";
import type {
  EmptyFilter,
  EventFilterParams,
  FavoritedFilter,
  FlaggedFilter,
} from "../../api/types";

interface VerifyMoreFiltersProps {
  filters: EventFilterParams;
  onChange: (next: EventFilterParams) => void;
  /** Project's detection_threshold; the det slider's resting floor. */
  detectionFloor?: number;
  /** Whether the det slider's low handle stops at the floor ("clamp",
   * Counts) or goes down the full scale ("open", Labels). */
  confidenceFloorMode?: "clamp" | "open";
  /** Reason shown while the handle rests on a clamped floor. */
  clampReason?: string;
  /** When false, the classification slider is hidden (no cls model). */
  showClassification?: boolean;
  /** Lowest classification confidence in the project (data-driven cls
   * slider clamp); null / undefined = no classifications yet. */
  minLabelConfidence?: number | null;
  /** Render the liked / flagged / empty selects. False on Observations. */
  showLikedFlaggedEmpty?: boolean;
}

export function VerifyMoreFilters({
  filters,
  onChange,
  detectionFloor = 0,
  confidenceFloorMode = "clamp",
  clampReason,
  showClassification = false,
  minLabelConfidence,
  showLikedFlaggedEmpty = true,
}: VerifyMoreFiltersProps) {
  const [open, setOpen] = useState(false);

  const detRangeActive =
    filters.min_confidence !== undefined ||
    filters.max_confidence !== undefined;
  const clsRangeActive =
    showClassification &&
    (filters.min_label_confidence !== undefined ||
      filters.max_label_confidence !== undefined);

  const activeCount =
    (showLikedFlaggedEmpty && filters.favorited && filters.favorited !== "all" ? 1 : 0) +
    (showLikedFlaggedEmpty && filters.flagged && filters.flagged !== "all" ? 1 : 0) +
    (showLikedFlaggedEmpty && filters.empty && filters.empty !== "hide" ? 1 : 0) +
    (detRangeActive ? 1 : 0) +
    (clsRangeActive ? 1 : 0);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="w-full h-9 justify-start text-sm font-normal"
        >
          <Filter className="h-4 w-4 mr-2 text-muted-foreground shrink-0" />
          <span>More filters</span>
          {activeCount > 0 && (
            <span className="ml-auto px-1.5 py-0.5 text-xs bg-primary text-primary-foreground rounded-full">
              {activeCount}
            </span>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-80 p-4 space-y-4"
        // Keep open while clicking into nested Radix portals (the Select
        // dropdowns below open in their own portal). Same guard that
        // DashboardFilters uses.
        onInteractOutside={(e) => {
          const target = e.target as HTMLElement | null;
          if (target?.closest?.("[data-radix-popper-content-wrapper]")) {
            e.preventDefault();
          }
        }}
      >
        {showLikedFlaggedEmpty && (
          <>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Liked</label>
              <Select
                value={filters.favorited ?? "all"}
                onValueChange={(v) =>
                  onChange({
                    ...filters,
                    favorited: v === "all" ? undefined : (v as FavoritedFilter),
                  })
                }
              >
                <SelectTrigger className="h-9 min-h-0 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="favorited">Liked</SelectItem>
                  <SelectItem value="not_favorited">Not liked</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Flagged</label>
              <Select
                value={filters.flagged ?? "all"}
                onValueChange={(v) =>
                  onChange({
                    ...filters,
                    flagged: v === "all" ? undefined : (v as FlaggedFilter),
                  })
                }
              >
                <SelectTrigger className="h-9 min-h-0 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="flagged">Flagged</SelectItem>
                  <SelectItem value="not_flagged">Not flagged</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Empty</label>
              <Select
                value={filters.empty ?? "hide"}
                onValueChange={(v) =>
                  onChange({
                    ...filters,
                    // Record the choice literally — "hide" is the
                    // implicit default, so "all" must be explicit
                    // (undefined would fall back to "hide").
                    empty: v as EmptyFilter,
                  })
                }
              >
                <SelectTrigger className="h-9 min-h-0 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="show_only">Show only empty</SelectItem>
                  <SelectItem value="hide">Hide empty</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </>
        )}

        <ConfidenceRangeFilter
          filters={filters}
          onChange={onChange}
          detectionFloor={detectionFloor}
          floorMode={confidenceFloorMode}
          clampReason={clampReason}
          showClassification={showClassification}
          minLabelConfidence={minLabelConfidence}
        />
      </PopoverContent>
    </Popover>
  );
}
