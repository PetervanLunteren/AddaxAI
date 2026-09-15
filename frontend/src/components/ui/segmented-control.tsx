/**
 * Segmented control: one choice out of a few, all visible at once.
 *
 * The active option gets the primary fill, the rest are muted and
 * highlight on hover. The filter bars use it icon-only, with the long name
 * on the `title` attribute so it surfaces as a native tooltip (and is read
 * by screen readers via aria-label). The setup forms' Data type toggle
 * passes a `label` instead, which is shown as text.
 */

import { cn } from "../../lib/utils";

export interface SegmentedOption {
  value: string;
  /** Shown as a native hover tooltip via the `title` attribute. */
  title: string;
  /** Icon-only options (filter bars). */
  icon?: React.ReactNode;
  /** Text shown on the button, beside the icon when there is one. */
  label?: string;
}

interface SegmentedControlProps {
  options: SegmentedOption[];
  value: string;
  onChange: (value: string) => void;
}

export function SegmentedControl({ options, value, onChange }: SegmentedControlProps) {
  return (
    <div
      role="radiogroup"
      className="flex h-9 w-full rounded-md border border-input bg-background overflow-hidden"
    >
      {options.map((opt, i) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={opt.title}
            aria-label={opt.title}
            onClick={() => onChange(opt.value)}
            className={cn(
              "flex-1 inline-flex items-center justify-center gap-2 text-sm transition-colors",
              i > 0 && "border-l border-input",
              active
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            )}
          >
            {opt.icon}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
