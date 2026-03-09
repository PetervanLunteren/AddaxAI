/**
 * Collapsible filter panel for the Browse & Verify page.
 *
 * Provides site, date range, species, verification status, and confidence
 * filters. All state is controlled via props (lifted to VerifyPage).
 */

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ListTodo } from "lucide-react";
import { eventsApi } from "../../api/events";
import { sitesApi } from "../../api/sites";
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
import { SpeciesFilterModal } from "./SpeciesFilterModal";

interface FilterPanelProps {
  filters: EventFilterParams;
  onChange: (filters: EventFilterParams) => void;
  projectId: string;
  isOpen: boolean;
  onToggle: () => void;
  classificationModelId?: string | null;
  children?: React.ReactNode;
  verificationSection?: React.ReactNode;
  countBy?: string;
}

const VERIFICATION_OPTIONS: { value: VerificationFilter | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "none_verified", label: "None verified" },
  { value: "not_fully_verified", label: "Partially verified" },
  { value: "unverified_representative", label: "Representative not verified" },
  { value: "verified_representative", label: "Representative verified" },
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
  countBy,
}: FilterPanelProps) {
  // Fetch sites for multiselect
  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId,
  });

  // Fetch filter options (species list, date range)
  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId),
    enabled: !!projectId,
  });

  // Fetch pre-built species tree (taxonomy + detected species + counts)
  const { data: speciesTree } = useQuery({
    queryKey: ["species-tree", projectId, countBy],
    queryFn: () => eventsApi.getSpeciesTree(projectId, countBy),
    enabled: !!projectId,
  });
  const hasTaxonomy = !!speciesTree?.tree?.length;

  const [speciesModalOpen, setSpeciesModalOpen] = useState(false);

  const siteOptions: MultiSelectOption[] =
    sites?.map((s) => ({ value: s.id, label: s.name })) ?? [];

  const speciesOptions: MultiSelectOption[] =
    filterOptions?.species.map((sp) => ({ value: sp, label: sp })) ?? [];

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

        {/* Species filter — taxonomy tree modal or flat multiselect fallback */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Species
          </label>
          {hasTaxonomy ? (
            <>
              <Button
                variant="outline"
                size="sm"
                className="w-full h-9 justify-start text-sm font-normal"
                onClick={() => setSpeciesModalOpen(true)}
              >
                <ListTodo className="h-4 w-4 mr-2 text-muted-foreground shrink-0" />
                <span className="truncate">
                  {filters.species?.length
                    ? `${filters.species.length} species`
                    : "All species"}
                </span>
              </Button>
              <SpeciesFilterModal
                preBuiltTree={speciesTree!.tree}
                allLeafIds={speciesTree!.all_leaf_ids}
                selectedSpecies={filters.species ?? []}
                onApply={(species) => {
                  const allLeafs = speciesTree!.all_leaf_ids;
                  const isAll = species.length >= allLeafs.length;
                  onChange({
                    ...filters,
                    species: isAll ? undefined : species.length ? species : undefined,
                  });
                }}
                open={speciesModalOpen}
                onOpenChange={setSpeciesModalOpen}
                countUnit={speciesTree!.count_unit}
              />
            </>
          ) : (
            <MultiSelect
              options={speciesOptions}
              value={filters.species ?? []}
              onChange={(v) =>
                onChange({ ...filters, species: v.length ? v : undefined })
              }
              placeholder="All species"
              searchPlaceholder="Search species..."
              emptyMessage="No species found."
              summary={(n) => `${n} species`}
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
                {VERIFICATION_OPTIONS.map((opt) => (
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
