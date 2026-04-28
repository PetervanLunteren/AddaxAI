/**
 * Collapsible filter panel for the Browse & Verify page.
 *
 * Provides site, date range, label, verification status, and confidence
 * filters. All state is controlled via props (lifted to VerifyPage).
 */

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ListTodo } from "lucide-react";
import { eventsApi } from "../../api/events";
import { sitesApi } from "../../api/sites";
import { useNoSiteDeployments } from "../../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../../lib/site-filter-options";
import type { EventFilterParams, VerificationFilter } from "../../api/types";
import { Button } from "../ui/button";
import { ConfidenceRangeFilter } from "./ConfidenceRangeFilter";
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

interface FilterPanelProps {
  filters: EventFilterParams;
  onChange: (filters: EventFilterParams) => void;
  projectId: string;
  isOpen: boolean;
  onToggle: () => void;
  classificationModelId?: string | null;
  /** Project's detection_threshold; clamps the floor of the det slider in
   *  the Advanced disclosure. Pass via the parent that already has
   *  `project` data; defaults to 0 (no floor) when omitted. */
  detectionFloor?: number;
  children?: React.ReactNode;
  verificationSection?: React.ReactNode;
  verificationOptions?: { value: VerificationFilter | "all"; label: string }[];
  countBy?: "event" | "file" | "detection";
}

const DEFAULT_VERIFICATION_OPTIONS: { value: VerificationFilter | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "unverified", label: "Unverified" },
  { value: "verified", label: "Verified" },
];

export function FilterPanel({
  filters,
  onChange,
  projectId,
  isOpen,
  classificationModelId,
  detectionFloor = 0,
  children,
  verificationSection,
  verificationOptions,
  countBy,
}: FilterPanelProps) {
  const activeVerificationOptions =
    verificationOptions ?? DEFAULT_VERIFICATION_OPTIONS;
  // Fetch sites for multiselect
  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId,
  });
  const { data: noSite } = useNoSiteDeployments(projectId);

  // Fetch filter options (label list, date range)
  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId),
    enabled: !!projectId,
  });

  // Fetch pre-built label tree (taxonomy + detected labels + counts)
  const { data: labelTree } = useQuery({
    queryKey: ["label-tree", projectId, countBy],
    queryFn: () => eventsApi.getLabelTree(projectId, countBy),
    enabled: !!projectId,
  });
  const hasTaxonomy = !!labelTree?.tree?.length;

  const [labelModalOpen, setLabelModalOpen] = useState(false);

  const siteOptions: MultiSelectOption[] = buildSiteOptions(
    sites,
    noSite?.count ?? 0,
  );

  const labelFilterOptions: MultiSelectOption[] =
    filterOptions?.labels.map((lbl) => ({
      value: lbl,
      label: filterOptions?.display_labels?.[lbl] ?? lbl,
    })) ?? [];

  if (!isOpen) return null;

  const showVerification = verificationSection !== null;

  return (
    <div className="rounded-lg border bg-white pt-2 pb-3 px-3 space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {/* Sites multiselect */}
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

        {/* Label filter — taxonomy tree modal or flat multiselect fallback */}
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

        {/* Verification dropdown (single select) — swappable via verificationSection prop.
            Pass null to hide entirely, undefined (omit) for default Events dropdown. */}
        {verificationSection !== undefined ? verificationSection : (
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              Verified
            </label>
            <Select
              value={filters.verification ?? "all"}
              onValueChange={(v) =>
                onChange({
                  ...filters,
                  verification:
                    v === "all" ? undefined : (v as VerificationFilter),
                })
              }
            >
              <SelectTrigger className="h-9 min-h-0 text-sm">
                <span className="truncate">
                  <SelectValue />
                </span>
              </SelectTrigger>
              <SelectContent>
                {activeVerificationOptions.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        {/* Favorited + Flagged filters. Mirror Connect's pattern of three
            separate dropdowns alongside verification. Shown on any tab that
            renders a verification dropdown; Observations passes
            verificationSection={null} and hides both. */}
        {showVerification && (
          <>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                Liked
              </label>
              <Select
                value={filters.favorited ?? "all"}
                onValueChange={(v) =>
                  onChange({
                    ...filters,
                    favorited:
                      v === "all"
                        ? undefined
                        : (v as "favorited" | "not_favorited"),
                  })
                }
              >
                <SelectTrigger className="h-9 min-h-0 text-sm">
                  <span className="truncate">
                    <SelectValue />
                  </span>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="favorited">Liked</SelectItem>
                  <SelectItem value="not_favorited">Not liked</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                Flagged
              </label>
              <Select
                value={filters.flagged ?? "all"}
                onValueChange={(v) =>
                  onChange({
                    ...filters,
                    flagged:
                      v === "all"
                        ? undefined
                        : (v as "flagged" | "not_flagged"),
                  })
                }
              >
                <SelectTrigger className="h-9 min-h-0 text-sm">
                  <span className="truncate">
                    <SelectValue />
                  </span>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="flagged">Flagged</SelectItem>
                  <SelectItem value="not_flagged">Not flagged</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                Empty
              </label>
              <Select
                value={filters.empty ?? "all"}
                onValueChange={(v) =>
                  onChange({
                    ...filters,
                    empty:
                      v === "all"
                        ? undefined
                        : (v as "show_only" | "hide"),
                  })
                }
              >
                <SelectTrigger className="h-9 min-h-0 text-sm">
                  <span className="truncate">
                    <SelectValue />
                  </span>
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

      </div>

      <ConfidenceRangeFilter
        filters={filters}
        onChange={onChange}
        detectionFloor={detectionFloor}
        showClassification={!!classificationModelId}
      />

      {children}
    </div>
  );
}
