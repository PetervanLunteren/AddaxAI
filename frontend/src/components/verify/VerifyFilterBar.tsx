/**
 * Single-row filter bar for the verify pages.
 *
 * Layout:
 *
 *   [ Sites | Date range | Labels | Verified | More ]
 *
 * Both pages fill the same slots. The More popover hosts the rare
 * filters (liked / flagged / empty) plus the confidence range
 * sliders, so the bar stays a single row.
 *
 * Page specifics:
 * - Counts (events): all slots; `showLikedFlaggedEmpty = true`.
 * - Labels: same slots, but the More popover only contains the
 *   confidence ranges (`showLikedFlaggedEmpty = false`). Verified
 *   options match (all / unverified / verified).
 */

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { eventsApi } from "../../api/events";
import { projectsApi } from "../../api/projects";
import { sitesApi } from "../../api/sites";
import { useNoSiteDeployments } from "../../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../../lib/site-filter-options";
import { speciesLabelMap } from "../../lib/species-name-mode";
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
import { FilterChips } from "./FilterChips";
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

// The Counts page (countBy="event") filters on Event.confirmed, so it
// reads "Confirmed" / "Unconfirmed". The shared `verification` param value
// stays verified/unverified; only the display wording changes.
const EVENT_VERIFICATION_OPTIONS: VerificationOption[] = [
  { value: "all", label: "All" },
  { value: "unverified", label: "Unconfirmed" },
  { value: "verified", label: "Confirmed" },
];
const EVENT_VERIFICATION_LABELS: Record<string, string> = {
  verified: "Confirmed",
  unverified: "Unconfirmed",
};

interface VerifyFilterBarProps {
  filters: EventFilterParams;
  onChange: (next: EventFilterParams) => void;
  projectId: string;
  classificationModelId?: string | null;
  /** Project's counting_threshold; clamps the floor of the det slider. */
  detectionFloor?: number;
  /** Which unit the label-tree counts are aggregated on. */
  countBy?: "event" | "file" | "detection";
  /** Verified dropdown options; defaults to all / unverified / verified. */
  verificationOptions?: VerificationOption[];
  /** Whether the More popover renders the liked / flagged / empty
   *  selects. False on the Labels page (those don't apply there). */
  showLikedFlaggedEmpty?: boolean;
  /** Detection-confidence slider floor behaviour: "clamp" (Counts,
   *  stops at the project threshold with a reason) or "open" (Labels,
   *  full scale so the user can dig into the low-confidence tail). */
  confidenceFloorMode?: "clamp" | "open";
  /** The page's default verification value. The select rests on it
   *  when no explicit filter is set, choosing it clears the filter,
   *  and it never renders a chip. Counts defaults to "all"; the
   *  Labels page passes "unverified". */
  verificationDefault?: VerificationFilter;
}

export function VerifyFilterBar({
  filters,
  onChange,
  projectId,
  classificationModelId,
  detectionFloor = 0,
  countBy,
  verificationOptions,
  showLikedFlaggedEmpty = true,
  confidenceFloorMode = "clamp",
  verificationDefault = "all",
}: VerifyFilterBarProps) {
  const [labelModalOpen, setLabelModalOpen] = useState(false);

  // Event scope (Counts page) confirms; detection/file scope (Labels)
  // verifies. Drives the verification filter's wording.
  const isEventScope = countBy === "event";
  const verOptions =
    verificationOptions ??
    (isEventScope ? EVENT_VERIFICATION_OPTIONS : DEFAULT_VERIFICATION_OPTIONS);

  // Folder runs are a single deployment with no site, so the Sites
  // filter is meaningless there. Detect the mode from the project
  // (deduped query — VerifyView already fetches it).
  const { data: project } = useQuery({
    queryKey: ["projects", projectId],
    queryFn: () => projectsApi.get(projectId),
    enabled: !!projectId,
  });
  const showSites = project?.mode !== "folder_run";

  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId && showSites,
  });
  const { data: noSite } = useNoSiteDeployments(projectId);

  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId),
    enabled: !!projectId,
  });

  const { data: labelTree } = useQuery({
    queryKey: [
      "label-tree",
      projectId,
      countBy,
      filters.site_ids,
      filters.date_from,
      filters.date_to,
    ],
    queryFn: () =>
      eventsApi.getLabelTree(projectId, countBy, {
        siteIds: filters.site_ids,
        dateFrom: filters.date_from,
        dateTo: filters.date_to,
      }),
    enabled: !!projectId,
  });
  const hasTaxonomy = !!labelTree?.tree?.length;

  // Id → name map for the active-filter chips' site labels.
  const siteNames: Record<string, string> = {};
  for (const s of sites ?? []) siteNames[s.id] = s.name;

  const siteOptions: MultiSelectOption[] = buildSiteOptions(
    sites,
    noSite?.count ?? 0,
  );

  const labelNames = filterOptions ? speciesLabelMap(filterOptions) : {};
  const labelFilterOptions: MultiSelectOption[] =
    filterOptions?.labels.map((lbl) => ({
      value: lbl,
      label: labelNames[lbl] ?? lbl,
    })) ?? [];

  // Four controls without Sites (folder runs), five with it (projects).
  const gridCols = showSites ? "lg:grid-cols-5" : "lg:grid-cols-4";

  return (
    <div className="space-y-2 rounded-lg border bg-white px-3 py-2">
      <div className={`grid grid-cols-1 sm:grid-cols-2 ${gridCols} gap-3`}>
        {showSites && (
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
        )}

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
          <label className="text-xs font-medium text-muted-foreground">{isEventScope ? "Confirmed" : "Verified"}</label>
          <Select
            value={filters.verification ?? verificationDefault}
            onValueChange={(v) =>
              onChange({
                ...filters,
                // Choosing the page default clears the filter (a
                // default is not a filter); anything else is explicit.
                verification:
                  v === verificationDefault
                    ? undefined
                    : (v as VerificationFilter),
              })
            }
          >
            <SelectTrigger className="h-9 min-h-0 text-sm">
              <span className="truncate">
                <SelectValue />
              </span>
            </SelectTrigger>
            <SelectContent>
              {verOptions.map((opt) => (
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
            confidenceFloorMode={confidenceFloorMode}
            minLabelConfidence={filterOptions?.min_label_confidence}
            clampReason={
              `Counting starts at the project's detection threshold ` +
              `(${Math.round(detectionFloor * 100)}%). ` +
              `Adjust it in the project settings.`
            }
            showClassification={!!classificationModelId}
            showLikedFlaggedEmpty={showLikedFlaggedEmpty}
          />
        </div>
      </div>

      {/* Active-filter chips live inside the bar (one shared place for
          all three views) so the card holds both the controls and the
          chips that reflect them. Renders null when no chips. */}
      <FilterChips
        filters={filters}
        onChange={onChange}
        verificationDefault={verificationDefault}
        siteNames={siteNames}
        displayLabels={filterOptions ? speciesLabelMap(filterOptions) : undefined}
        detectionFloor={detectionFloor}
        verificationLabels={isEventScope ? EVENT_VERIFICATION_LABELS : undefined}
      />
    </div>
  );
}
