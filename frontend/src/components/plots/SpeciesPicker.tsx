/**
 * Single-species searchable combobox for the Plots → Activity overlap page.
 *
 * Two instances side by side in `ActivityOverlapFilterBar` (Species A
 * and Species B). Returns the project's display-name string, which the
 * activity-overlap endpoint expects as `species_a` / `species_b`.
 *
 * Sourced from the project's species distribution (same data the
 * dashboard's `ActivityPatternChart` and `DetectionTrendChart` use).
 * That endpoint already respects the same site / date / taxonomic-rank
 * filters as the activity-overlap endpoint, so the dropdown only shows
 * species that actually have observations under the active filter set.
 */

import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, ChevronsUpDown, X } from "lucide-react";

import { statisticsApi } from "../../api/statistics";
import { cn } from "../../lib/utils";
import { normalizeLabel } from "../../utils/labels";
import { Button } from "../ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "../ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";

interface SpeciesPickerProps {
  projectId: string;
  /** Currently selected species display name, or null if none. */
  value: string | null;
  /** Called with the chosen display name (or null when cleared). */
  onChange: (value: string | null) => void;
  /** Display name to show as the placeholder when nothing is picked. */
  placeholder?: string;
  /** Optional label prefix (used as the trigger button's hint). */
  hint?: string;
  /** Disable the picker (e.g. while the API request is in-flight). */
  disabled?: boolean;
  /** Optional filter scope so the dropdown only shows species in the
   *  active site / date subset. Matches the activity-overlap filters. */
  siteIds?: string[];
  dateFrom?: string;
  dateTo?: string;
  taxonomicRank?: string;
  /** Optional value that should be hidden from the dropdown (used to
   *  prevent picking the same species in both A and B slots). */
  excludeValue?: string | null;
  /** Visual color dot next to the trigger label (the species' chart color). */
  swatchColor?: string;
}

export function SpeciesPicker({
  projectId,
  value,
  onChange,
  placeholder = "Pick a species",
  hint,
  disabled,
  siteIds,
  dateFrom,
  dateTo,
  taxonomicRank,
  excludeValue,
  swatchColor,
}: SpeciesPickerProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    if (!open) setSearch("");
  }, [open]);

  const siteIdsParam = siteIds && siteIds.length > 0 ? siteIds.join(",") : undefined;

  const { data: speciesList, isLoading } = useQuery({
    queryKey: [
      "statistics",
      "species",
      "picker",
      projectId,
      siteIdsParam,
      dateFrom,
      dateTo,
      taxonomicRank,
    ],
    queryFn: () =>
      statisticsApi.getSpeciesDistribution(
        projectId,
        siteIdsParam,
        dateFrom,
        dateTo,
        taxonomicRank,
      ),
  });

  const options = useMemo(() => {
    const list = speciesList ?? [];
    return list
      .filter((s) => s.species !== excludeValue)
      .map((s) => ({
        value: s.species,
        label: normalizeLabel(s.species),
        count: s.count,
      }));
  }, [speciesList, excludeValue]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return options;
    return options.filter(
      (opt) =>
        opt.label.toLowerCase().includes(q) ||
        opt.value.toLowerCase().includes(q),
    );
  }, [options, search]);

  const currentLabel =
    value && (options.find((o) => o.value === value)?.label ?? normalizeLabel(value));

  return (
    <div className="flex w-full items-center gap-1">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="outline"
            role="combobox"
            aria-expanded={open}
            disabled={disabled || isLoading}
            className="flex-1 min-w-0 justify-between h-9 text-sm font-normal"
          >
            <span className="flex items-center gap-2 truncate">
              {swatchColor && (
                <span
                  aria-hidden="true"
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: swatchColor }}
                />
              )}
              {hint && !value && (
                <span className="text-muted-foreground">{hint}: </span>
              )}
              <span className="truncate">{currentLabel || placeholder}</span>
            </span>
            <ChevronsUpDown className="ml-2 h-3.5 w-3.5 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[320px] p-0" align="start">
          <Command shouldFilter={false}>
            <CommandInput
              placeholder="Search species..."
              value={search}
              onValueChange={setSearch}
            />
            <CommandList>
              <CommandEmpty>
                {isLoading ? "Loading species..." : "No species found."}
              </CommandEmpty>
              <CommandGroup>
                {filtered.map((opt) => {
                  const selected = opt.value === value;
                  return (
                    <CommandItem
                      key={opt.value}
                      value={opt.value}
                      onSelect={() => {
                        onChange(opt.value);
                        setOpen(false);
                      }}
                    >
                      <Check
                        className={cn(
                          "mr-2 h-4 w-4",
                          selected ? "opacity-100" : "opacity-0",
                        )}
                      />
                      <span className="flex-1 truncate">{opt.label}</span>
                      <span className="ml-2 text-xs text-muted-foreground tabular-nums">
                        n={opt.count}
                      </span>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
      {value && !disabled && (
        <Button
          type="button"
          variant="ghost"
          size="icon"
          aria-label="Clear selection"
          className="h-9 w-9 shrink-0"
          onClick={() => onChange(null)}
        >
          <X className="h-4 w-4" />
        </Button>
      )}
    </div>
  );
}
