/**
 * Reusable multiselect combobox with checkboxes.
 *
 * Uses Popover + Command + Checkbox for a searchable, accessible dropdown
 * that supports multiple selections. Use this as the go-to multiselect
 * across the app.
 */

import { useState } from "react";
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

  const selectedSet = new Set(value);

  const toggleOption = (optionValue: string) => {
    const next = selectedSet.has(optionValue)
      ? value.filter((v) => v !== optionValue)
      : [...value, optionValue];
    onChange(next);
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
        <Command>
          <CommandInput placeholder={searchPlaceholder} />
          <CommandList>
            <CommandEmpty>{emptyMessage}</CommandEmpty>
            {options.map((opt) => {
              const selected = selectedSet.has(opt.value);
              return (
                <CommandItem
                  key={opt.value}
                  value={opt.label}
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
        </Command>
      </PopoverContent>
    </Popover>
  );
}
