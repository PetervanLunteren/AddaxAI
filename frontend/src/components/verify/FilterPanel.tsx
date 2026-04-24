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
  children?: React.ReactNode;
  verificationSection?: React.ReactNode;
  verificationOptions?: { value: VerificationFilter | "all"; label: string }[];
  countBy?: "event" | "file" | "detection";
}

const DEFAULT_VERIFICATION_OPTIONS: { value: VerificationFilter | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "none_verified", label: "None verified" },
  { value: "unverified_maxn", label: "No MaxN verified" },
  { value: "some_maxn_verified", label: "Some MaxN verified" },
  { value: "all_maxn_verified", label: "All MaxN verified" },
  { value: "not_fully_verified", label: "Partially verified" },
  { value: "fully_verified", label: "Fully verified" },
];

export function FilterPanel({
  filters,
  onChange,
  projectId,
  isOpen,
  classificationModelId,
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

  return (
    <div className="rounded-lg border bg-white pt-2 pb-3 px-3 space-y-4">
      <div className={`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 ${verificationSection !== null ? "xl:grid-cols-5" : "xl:grid-cols-4"} gap-4`}>
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

        {/* Date from */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            From
          </label>
          <input
            type="date"
            className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring [&::-webkit-date-and-time-value]:text-left [&::-webkit-calendar-picker-indicator]:absolute [&::-webkit-calendar-picker-indicator]:right-3 relative"
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
            className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring [&::-webkit-date-and-time-value]:text-left [&::-webkit-calendar-picker-indicator]:absolute [&::-webkit-calendar-picker-indicator]:right-3 relative"
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
              Verification status
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

      </div>

      {children}
    </div>
  );
}
