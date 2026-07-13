/**
 * A clickable "what next" row in a completion modal: icon + a short
 * action title + a one-line description of where it takes you. Used so
 * the post-analysis step is self-explanatory instead of a bare button.
 *
 * Shared by the projects-mode run modal, the folder-run save modal, and
 * the folder-run Labels step so every "pick your next move" screen offers
 * its options in the same shape. Convention: rows are the steps that take
 * the user somewhere; the "start over" action stays a footer button.
 *
 * The row has no fill of its own, which reads fine on the white dialog
 * surfaces. On a coloured page (the slate folder-run canvas) pass
 * `className="bg-card shadow-sm"` so each row reads as a raised white
 * card like the other steps' Cards, instead of a flat outline.
 */

import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

export function NextStepRow({
  icon: Icon,
  title,
  description,
  onClick,
  disabled,
  className,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  onClick: () => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "flex w-full items-start gap-3 rounded-lg border p-3 text-left transition-colors hover:bg-accent disabled:pointer-events-none disabled:opacity-50",
        className,
      )}
    >
      <Icon className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
      <div className="flex-1">
        <p className="text-sm font-medium">{title}</p>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
      <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
    </button>
  );
}
