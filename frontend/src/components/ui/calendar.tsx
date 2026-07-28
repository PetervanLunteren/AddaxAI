/**
 * Calendar primitive built on react-day-picker v9.
 *
 * Uses the library's default stylesheet (imported globally in index.css)
 * with two CSS variables overridden to tint the selection in our teal
 * brand. Only the chevron icons are swapped to lucide for consistency.
 *
 * Custom Tailwind class overrides on v9's `classNames` API turned out to
 * fight the library's internal class merging — the default CSS handles
 * range hover preview, focus rings, RTL, etc. correctly out of the box.
 */

import { ChevronLeft, ChevronRight } from "lucide-react";
import { DayPicker } from "react-day-picker";
import type { ComponentProps } from "react";

import { cn } from "../../lib/utils";

// react-day-picker's default stylesheet is imported once globally in
// main.tsx (BEFORE index.css), so our `.rdp-root` brand-variable
// overrides in index.css load after the defaults and win the cascade.

export type CalendarProps = ComponentProps<typeof DayPicker>;

export function Calendar({ className, ...props }: CalendarProps) {
  return (
    <DayPicker
      className={cn("p-2", className)}
      components={{
        Chevron: ({ orientation }) =>
          orientation === "left" ? (
            <ChevronLeft className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          ),
      }}
      {...props}
    />
  );
}
