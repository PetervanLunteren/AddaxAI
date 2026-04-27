/**
 * Collapsible filter panel for the Browse & Verify page.
 *
 * Provides site, date range, label, verification status, and confidence
 * filters. All state is controlled via props (lifted to VerifyPage).
 */

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { format, parseISO } from "date-fns";
import { CalendarIcon, ListTodo } from "lucide-react";
import { eventsApi } from "../../api/events";
import { sitesApi } from "../../api/sites";
import { useNoSiteDeployments } from "../../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../../lib/site-filter-options";
import type { EventFilterParams, VerificationFilter } from "../../api/types";
import { Button } from "../ui/button";
import { Calendar } from "../ui/calendar";
import { MultiSelect, type MultiSelectOption } from "../ui/multi-select";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
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
  { value: "unverified", label: "Unverified" },
  { value: "verified", label: "Verified" },
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
  const [dateRangeOpen, setDateRangeOpen] = useState(false);

  // Parse ISO date strings (YYYY-MM-DD) into Date objects for the calendar.
  // Keep parsing local-timezone-naive so the user sees the same calendar
  // day they typed, regardless of their browser timezone.
  const dateRange = {
    from: filters.date_from ? parseISO(filters.date_from) : undefined,
    to: filters.date_to ? parseISO(filters.date_to) : undefined,
  };
  const minDate = filterOptions?.date_range
    ? parseISO(filterOptions.date_range.min.slice(0, 10))
    : undefined;
  const maxDate = filterOptions?.date_range
    ? parseISO(filterOptions.date_range.max.slice(0, 10))
    : undefined;

  const dateRangeLabel = dateRange.from
    ? dateRange.to
      ? `${format(dateRange.from, "d MMM yyyy")} – ${format(dateRange.to, "d MMM yyyy")}`
      : format(dateRange.from, "d MMM yyyy")
    : "All dates";

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

        {/* Date range — two-month popover calendar. Replaces the old
            separate From / To inputs. Serializes the picked range back
            to YYYY-MM-DD strings for the URL. */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Date range
          </label>
          <Popover open={dateRangeOpen} onOpenChange={setDateRangeOpen}>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="w-full h-9 justify-start text-sm font-normal"
              >
                <CalendarIcon className="h-4 w-4 mr-2 text-muted-foreground shrink-0" />
                <span className="truncate">{dateRangeLabel}</span>
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="range"
                selected={dateRange}
                onSelect={(range) => {
                  onChange({
                    ...filters,
                    date_from: range?.from
                      ? format(range.from, "yyyy-MM-dd")
                      : undefined,
                    date_to: range?.to
                      ? format(range.to, "yyyy-MM-dd")
                      : undefined,
                  });
                }}
                numberOfMonths={2}
                defaultMonth={dateRange.from ?? maxDate}
                startMonth={minDate}
                endMonth={maxDate}
              />
              {(filters.date_from || filters.date_to) && (
                <div className="flex justify-end p-2 border-t">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() =>
                      onChange({
                        ...filters,
                        date_from: undefined,
                        date_to: undefined,
                      })
                    }
                  >
                    Clear
                  </Button>
                </div>
              )}
            </PopoverContent>
          </Popover>
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
          </>
        )}

      </div>

      {children}
    </div>
  );
}
