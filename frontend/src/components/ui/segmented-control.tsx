/**
 * Icon-only segmented control used in filter bars.
 *
 * Each option renders as an icon button; the active one gets the
 * primary fill, the rest are muted and highlight on hover. The long
 * name goes on the `title` attribute so it surfaces as a native
 * tooltip (and is read by screen readers via aria-label).
 */

import { cn } from "../../lib/utils";

export interface SegmentedOption {
  value: string;
  /** Shown as a native hover tooltip via the `title` attribute. */
  title: string;
  icon: React.ReactNode;
}

interface SegmentedControlProps {
  options: SegmentedOption[];
  value: string;
  onChange: (value: string) => void;
}

export function SegmentedControl({ options, value, onChange }: SegmentedControlProps) {
  return (
    <div className="flex h-9 w-full rounded-md border border-input bg-background overflow-hidden">
      {options.map((opt, i) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            title={opt.title}
            aria-label={opt.title}
            onClick={() => onChange(opt.value)}
            className={cn(
              "flex-1 inline-flex items-center justify-center transition-colors",
              i > 0 && "border-l border-input",
              active
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            )}
          >
            {opt.icon}
          </button>
        );
      })}
    </div>
  );
}
