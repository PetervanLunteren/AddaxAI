/**
 * Single-row filter bar for the verify tabs.
 *
 * Layout:
 *
 *   [ Sites | Date range | Labels | Verified | More ]
 *
 * Each tab fills the same slots. The More popover hosts the rare
 * filters (liked / flagged / empty) plus the confidence range
 * sliders, so the bar stays a single row.
 *
 * Tab specifics:
 * - Events / Files: all five slots; `showLikedFlaggedEmpty = true`.
 * - Observations: same slots, but the More popover only contains the
 *   confidence ranges (`showLikedFlaggedEmpty = false`); Verified
 *   options swap to all / unverified / suspicious (suspicious only
 *   listed when neighbor agreement data is present).
 */

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { eventsApi } from "../../api/events";
import { sitesApi } from "../../api/sites";
import { useNoSiteDeployments } from "../../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../../lib/site-filter-options";
import type {
  EventFilterParams,
  VerificationFilter,
} from "../../api/types";
import { Button } from "../ui/button";
import { DateRangePicker } from "../ui/date-range-picker";
import { MultiSelect, type MultiSelectOption } from "../ui/multi-select";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { LabelFilterModal } from "./LabelFilterModal";
import { VerifyMoreFilters } from "./VerifyMoreFilters";

export interface VerificationOption {
  value: VerificationFilter;
  label: string;
}

const DEFAULT_VERIFICATION_OPTIONS: VerificationOption[] = [
  { value: "all", label: "All" },
  { value: "unverified", label: "Unverified" },
  { value: "verified", label: "Verified" },
];

interface VerifyFilterBarProps {
  filters: EventFilterParams;
  onChange: (next: EventFilterParams) => void;
  projectId: string;
  classificationModelId?: string | null;
  /** Project's detection_threshold; clamps the floor of the det slider. */
  detectionFloor?: number;
  /** Which unit the label-tree counts are aggregated on. */
  countBy?: "event" | "file" | "detection";
  /** Verified dropdown options; defaults to all / unverified / verified. */
  verificationOptions?: VerificationOption[];
  /** Whether the More popover renders the liked / flagged / empty
   *  selects. False on Observations (those don't apply there). */
  showLikedFlaggedEmpty?: boolean;
}

export function VerifyFilterBar({
  filters,
  onChange,
  projectId,
  classificationModelId,
  detectionFloor = 0,
  countBy,
  verificationOptions = DEFAULT_VERIFICATION_OPTIONS,
  showLikedFlaggedEmpty = true,
}: VerifyFilterBarProps) {
  const [labelModalOpen, setLabelModalOpen] = useState(false);

  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId,
  });
  const { data: noSite } = useNoSiteDeployments(projectId);

  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId),
    enabled: !!projectId,
  });

  const { data: labelTree } = useQuery({
    queryKey: ["label-tree", projectId, countBy],
    queryFn: () => eventsApi.getLabelTree(projectId, countBy),
    enabled: !!projectId,
  });
  const hasTaxonomy = !!labelTree?.tree?.length;

  const siteOptions: MultiSelectOption[] = buildSiteOptions(
    sites,
    noSite?.count ?? 0,
  );

  const labelFilterOptions: MultiSelectOption[] =
    filterOptions?.labels.map((lbl) => ({
      value: lbl,
      label: filterOptions?.display_labels?.[lbl] ?? lbl,
    })) ?? [];

  return (
    <div className="rounded-lg border bg-white px-3 py-2">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">Sites</label>
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
          <label className="text-xs font-medium text-muted-foreground">Date range</label>
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

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">Labels</label>
          {hasTaxonomy ? (
            <>
              <Button
                variant="outline"
                size="sm"
                className="w-full h-9 justify-start text-sm font-normal"
                onClick={() => setLabelModalOpen(true)}
              >
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
                    labels: isAll ? undefined : labels.length ? labels : undefined,
                  });
                }}
                open={labelModalOpen}
                onOpenChange={setLabelModalOpen}
                countUnit={labelTree!.count_unit}
              />
            </>
          ) : (
            <MultiSelect
              options={labelFilterOptions}
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

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">Verified</label>
          <Select
            value={filters.verification ?? "all"}
            onValueChange={(v) =>
              onChange({
                ...filters,
                verification: v === "all" ? undefined : (v as VerificationFilter),
              })
            }
          >
            <SelectTrigger className="h-9 min-h-0 text-sm">
              <span className="truncate">
                <SelectValue />
              </span>
            </SelectTrigger>
            <SelectContent>
              {verificationOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">&nbsp;</label>
          <VerifyMoreFilters
            filters={filters}
            onChange={onChange}
            detectionFloor={detectionFloor}
            showClassification={!!classificationModelId}
            showLikedFlaggedEmpty={showLikedFlaggedEmpty}
          />
        </div>
      </div>
    </div>
  );
}
