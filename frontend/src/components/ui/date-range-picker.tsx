/**
 * DateRangePicker — single trigger button + popover with a two-month
 * react-day-picker calendar in range mode.
 *
 * Wraps the bespoke logic that used to live inside FilterPanel so every
 * filter bar in the app gets the same UX. Dates are exchanged with the
 * caller as ISO date strings (YYYY-MM-DD); rendering uses date-fns to
 * format the trigger label.
 */

import { useState } from "react";
import { format, parseISO } from "date-fns";
import { CalendarIcon } from "lucide-react";

import { Button } from "./button";
import { Calendar } from "./calendar";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";

interface DateRangePickerProps {
  /** ISO date string (YYYY-MM-DD), or null/undefined when unset. */
  from: string | null | undefined;
  to: string | null | undefined;
  onChange: (range: {
    from: string | undefined;
    to: string | undefined;
  }) => void;
  /** Optional bounds shown by the calendar (also ISO date strings). */
  minDate?: string | null;
  maxDate?: string | null;
  /** Label shown when no dates are picked. Defaults to "All dates". */
  placeholder?: string;
  /** Trigger button height; defaults to h-9 to match other shadcn inputs. */
  className?: string;
}

export function DateRangePicker({
  from,
  to,
  onChange,
  minDate,
  maxDate,
  placeholder = "All dates",
  className,
}: DateRangePickerProps) {
  const [open, setOpen] = useState(false);

  const range = {
    from: from ? parseISO(from) : undefined,
    to: to ? parseISO(to) : undefined,
  };
  const startMonth = minDate ? parseISO(minDate.slice(0, 10)) : undefined;
  const endMonth = maxDate ? parseISO(maxDate.slice(0, 10)) : undefined;

  const label = range.from
    ? range.to
      ? `${format(range.from, "d MMM yyyy")} – ${format(range.to, "d MMM yyyy")}`
      : format(range.from, "d MMM yyyy")
    : placeholder;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className={
            className ??
            "w-full h-9 justify-start text-sm font-normal"
          }
        >
          <CalendarIcon className="h-4 w-4 mr-2 text-muted-foreground shrink-0" />
          <span className="truncate">{label}</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0" align="start">
        <Calendar
          mode="range"
          selected={range}
          onSelect={(picked) => {
            onChange({
              from: picked?.from ? format(picked.from, "yyyy-MM-dd") : undefined,
              to: picked?.to ? format(picked.to, "yyyy-MM-dd") : undefined,
            });
          }}
          numberOfMonths={2}
          defaultMonth={range.from ?? endMonth}
          startMonth={startMonth}
          endMonth={endMonth}
        />
        {(from || to) && (
          <div className="flex justify-end p-2 border-t">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onChange({ from: undefined, to: undefined })}
            >
              Clear
            </Button>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
