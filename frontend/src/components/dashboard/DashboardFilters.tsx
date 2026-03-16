/**
 * Filter popover for dashboard: site multiselect + date range + taxonomic rank.
 *
 * Closes on click-outside. Shows badge count of active filters.
 */

import { useState, useRef, useEffect } from "react";
import { Filter } from "lucide-react";
import { Button } from "../ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { MultiSelect, type MultiSelectOption } from "../ui/multi-select";
import type { DateRange } from "./index";

const TAXONOMIC_RANKS = [
  { value: "species", label: "Species" },
  { value: "genus", label: "Genus" },
  { value: "family", label: "Family" },
  { value: "order", label: "Order" },
  { value: "class", label: "Class" },
  { value: "raw", label: "Raw label" },
];

interface DashboardFiltersProps {
  siteOptions: MultiSelectOption[];
  selectedSiteIds: string[];
  onSiteIdsChange: (ids: string[]) => void;
  dateRange: DateRange;
  onDateRangeChange: (range: DateRange) => void;
  minDate?: string | null;
  maxDate?: string | null;
  taxonomicRank: string;
  onTaxonomicRankChange: (rank: string) => void;
}

export const DashboardFilters: React.FC<DashboardFiltersProps> = ({
  siteOptions,
  selectedSiteIds,
  onSiteIdsChange,
  dateRange,
  onDateRangeChange,
  minDate,
  maxDate,
  taxonomicRank,
  onTaxonomicRankChange,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseDown = (e: MouseEvent) => {
      if (!containerRef.current) return;
      const target = e.target as Node;
      // Don't close if clicking inside the popover container
      if (containerRef.current.contains(target)) return;
      // Don't close if clicking inside a Radix portal (Select/MultiSelect dropdowns)
      if ((target as HTMLElement).closest?.("[data-radix-popper-content-wrapper]")) return;
      setIsOpen(false);
    };
    document.addEventListener("mousedown", handleMouseDown);
    return () => document.removeEventListener("mousedown", handleMouseDown);
  }, []);

  const activeCount =
    selectedSiteIds.length +
    (dateRange.startDate ? 1 : 0) +
    (dateRange.endDate ? 1 : 0) +
    (taxonomicRank !== "species" ? 1 : 0);

  const clearAll = () => {
    onSiteIdsChange([]);
    onDateRangeChange({ startDate: null, endDate: null });
    onTaxonomicRankChange("species");
  };

  return (
    <div ref={containerRef} className="relative">
      <Button
        variant="outline"
        size="sm"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2"
      >
        <Filter className="h-4 w-4" />
        Filters
        {activeCount > 0 && (
          <span className="px-1.5 py-0.5 text-xs bg-primary text-primary-foreground rounded-full">
            {activeCount}
          </span>
        )}
      </Button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-96 border rounded-md bg-background shadow-lg z-50 p-4 space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Sites</label>
            <MultiSelect
              options={siteOptions}
              value={selectedSiteIds}
              onChange={onSiteIdsChange}
              placeholder="All sites"
              searchPlaceholder="Search sites..."
              popoverWidth="w-[350px]"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Date range</label>
            <div className="flex items-center gap-2">
              <input
                type="date"
                value={dateRange.startDate || ""}
                onChange={(e) =>
                  onDateRangeChange({ ...dateRange, startDate: e.target.value || null })
                }
                min={minDate || undefined}
                max={maxDate || undefined}
                className="flex-1 h-9 rounded-md border border-input bg-background px-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              />
              <span className="text-sm text-muted-foreground">to</span>
              <input
                type="date"
                value={dateRange.endDate || ""}
                onChange={(e) =>
                  onDateRangeChange({ ...dateRange, endDate: e.target.value || null })
                }
                min={minDate || undefined}
                max={maxDate || undefined}
                className="flex-1 h-9 rounded-md border border-input bg-background px-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Taxonomic rank</label>
            <Select value={taxonomicRank} onValueChange={onTaxonomicRankChange}>
              <SelectTrigger className="w-full h-9 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TAXONOMIC_RANKS.map((r) => (
                  <SelectItem key={r.value} value={r.value}>
                    {r.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {activeCount > 0 && (
            <button
              type="button"
              onClick={clearAll}
              className="text-xs text-muted-foreground hover:underline"
            >
              Clear all filters
            </button>
          )}
        </div>
      )}
    </div>
  );
};
