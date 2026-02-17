/**
 * Unified label picker — searchable combobox for detection labels.
 *
 * Replaces the separate category dropdown + species text input with a single
 * searchable picker. Options include species from the classification model's
 * taxonomy plus "person" and "vehicle".
 */

import { useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import { cn } from "../../lib/utils";
import { getCategoryColor } from "../../lib/detection-utils";
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
import type { LabelOption } from "../../hooks/useLabelOptions";

interface LabelPickerProps {
  value: string | null;
  onSelect: (option: LabelOption) => void;
  options: LabelOption[];
  isLoading?: boolean;
}

export function LabelPicker({
  value,
  onSelect,
  options,
  isLoading,
}: LabelPickerProps) {
  const [open, setOpen] = useState(false);

  const generalOptions = options.filter((o) => o.species === null);
  const speciesOptions = options.filter((o) => o.species !== null);

  const currentOption = options.find((o) => o.value === value);
  const displayLabel = value ?? "Select label...";
  const dotColor = currentOption
    ? getCategoryColor(currentOption.category)
    : undefined;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-1.5 gap-1 text-xs font-medium capitalize justify-start"
          onClick={(e) => e.stopPropagation()}
        >
          {dotColor && (
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: dotColor }}
            />
          )}
          <span className="truncate max-w-[120px]">{displayLabel}</span>
          <ChevronsUpDown className="h-3 w-3 opacity-50 shrink-0" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-52 p-0"
        align="start"
        onClick={(e) => e.stopPropagation()}
      >
        <Command>
          <CommandInput placeholder="Search labels..." className="h-8 text-xs" />
          <CommandList
            className="max-h-[250px]"
            onWheel={(e) => e.stopPropagation()}
          >
            <CommandEmpty>
              {isLoading ? "Loading..." : "No label found."}
            </CommandEmpty>
            <CommandGroup heading="General">
              {generalOptions.map((opt) => (
                <CommandItem
                  key={opt.value}
                  value={opt.value}
                  onSelect={() => {
                    onSelect(opt);
                    setOpen(false);
                  }}
                  className="text-xs capitalize"
                >
                  <div
                    className="w-2 h-2 rounded-full shrink-0 mr-1.5"
                    style={{ backgroundColor: getCategoryColor(opt.category) }}
                  />
                  {opt.value}
                  <Check
                    className={cn(
                      "ml-auto h-3 w-3",
                      value === opt.value ? "opacity-100" : "opacity-0"
                    )}
                  />
                </CommandItem>
              ))}
            </CommandGroup>
            {speciesOptions.length > 0 && (
              <CommandGroup heading="Species">
                {speciesOptions.map((opt) => (
                  <CommandItem
                    key={opt.value}
                    value={opt.value}
                    onSelect={() => {
                      onSelect(opt);
                      setOpen(false);
                    }}
                    className="text-xs capitalize"
                  >
                    <div
                      className="w-2 h-2 rounded-full shrink-0 mr-1.5"
                      style={{
                        backgroundColor: getCategoryColor(opt.category),
                      }}
                    />
                    {opt.value}
                    <Check
                      className={cn(
                        "ml-auto h-3 w-3",
                        value === opt.value ? "opacity-100" : "opacity-0"
                      )}
                    />
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
