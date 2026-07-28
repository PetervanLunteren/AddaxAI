/**
 * Reusable multiselect combobox with checkboxes.
 *
 * Uses Popover + Command + Checkbox for a searchable, accessible dropdown
 * that supports multiple selections. Use this as the go-to multiselect
 * across the app.
 *
 * Features (mirrored from AddaxAI-Connect):
 * - Live search filter
 * - "Select all" / "Clear all" buttons that operate on the currently
 *   visible (search-filtered) options
 * - Footer showing "X of Y selected"
 */

import { useEffect, useMemo, useState } from "react";
import { ChevronsUpDown } from "lucide-react";
import { Button } from "./button";
import { Checkbox } from "./checkbox";
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "./command";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";

export interface MultiSelectOption {
  value: string;
  label: string;
}

interface MultiSelectProps {
  /** Available options. */
  options: MultiSelectOption[];
  /** Currently selected values. */
  value: string[];
  /** Called when selection changes. */
  onChange: (value: string[]) => void;
  /** Placeholder shown when nothing is selected. */
  placeholder?: string;
  /** Search input placeholder. */
  searchPlaceholder?: string;
  /** Message when search yields no results. */
  emptyMessage?: string;
  /** Summary label for the trigger button. Receives the count of selected items. */
  summary?: (count: number) => string;
  /** Popover width class (default: "w-[220px]"). */
  popoverWidth?: string;
  /** Capitalize option labels. */
  capitalize?: boolean;
}

export function MultiSelect({
  options,
  value,
  onChange,
  placeholder = "All",
  searchPlaceholder = "Search...",
  emptyMessage = "No results.",
  summary,
  popoverWidth = "w-[220px]",
  capitalize = false,
}: MultiSelectProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");

  // Reset search when the popover closes so reopening starts fresh.
  useEffect(() => {
    if (!open) setSearch("");
  }, [open]);

  const selectedSet = useMemo(() => new Set(value), [value]);

  // Manual case-insensitive substring filter — replaces cmdk's default
  // fuzzy match so the Select-all behaviour matches what the user sees.
  const filteredOptions = useMemo(() => {
    if (!search.trim()) return options;
    const q = search.toLowerCase();
    return options.filter((opt) => opt.label.toLowerCase().includes(q));
  }, [options, search]);

  const toggleOption = (optionValue: string) => {
    const next = selectedSet.has(optionValue)
      ? value.filter((v) => v !== optionValue)
      : [...value, optionValue];
    onChange(next);
  };

  /** Add every currently visible (filtered) option to the selection. */
  const selectAllVisible = () => {
    const next = [...value];
    for (const opt of filteredOptions) {
      if (!selectedSet.has(opt.value)) {
        next.push(opt.value);
      }
    }
    onChange(next);
  };

  /** Remove every currently visible (filtered) option from the selection. */
  const clearAllVisible = () => {
    const visibleSet = new Set(filteredOptions.map((o) => o.value));
    onChange(value.filter((v) => !visibleSet.has(v)));
  };

  const triggerLabel =
    value.length > 0
      ? summary
        ? summary(value.length)
        : `${value.length} selected`
      : placeholder;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between h-9 text-sm font-normal"
        >
          <span className="truncate">{triggerLabel}</span>
          <ChevronsUpDown className="ml-1 h-3.5 w-3.5 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className={`${popoverWidth} p-0`} align="start">
        <Command shouldFilter={false}>
          <CommandInput
            placeholder={searchPlaceholder}
            value={search}
            onValueChange={setSearch}
          />

          {/* Select all / Clear all bar */}
          <div className="flex items-center justify-between px-3 py-1.5 border-b">
            <button
              type="button"
              onClick={selectAllVisible}
              className="text-xs text-muted-foreground hover:underline"
              disabled={filteredOptions.length === 0}
            >
              Select all
            </button>
            <button
              type="button"
              onClick={clearAllVisible}
              className="text-xs text-muted-foreground hover:underline"
              disabled={filteredOptions.length === 0}
            >
              Clear all
            </button>
          </div>

          <CommandList>
            <CommandEmpty>{emptyMessage}</CommandEmpty>
            {filteredOptions.map((opt) => {
              const selected = selectedSet.has(opt.value);
              return (
                <CommandItem
                  key={opt.value}
                  value={opt.value}
                  onSelect={() => toggleOption(opt.value)}
                >
                  <Checkbox
                    checked={selected}
                    onCheckedChange={() => {}}
                    className="mr-2 pointer-events-none"
                  />
                  <span className={capitalize ? "capitalize" : undefined}>
                    {opt.label}
                  </span>
                </CommandItem>
              );
            })}
          </CommandList>

          {/* Footer count */}
          <div className="px-3 py-1.5 border-t text-xs text-muted-foreground">
            {value.length} of {options.length} selected
          </div>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
