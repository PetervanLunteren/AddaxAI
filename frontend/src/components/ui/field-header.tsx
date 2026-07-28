import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/**
 * A form field's title and its help caption, grouped tight (4px) so the
 * caption reads as part of the label instead of floating halfway to the input.
 *
 * Pass the label element you already use (a FormLabel inside a react-hook-form
 * FormItem, or a plain Label with htmlFor) so its wiring and accessibility are
 * preserved. This component only owns the title-to-caption spacing; the gap
 * down to the input still comes from the surrounding container.
 */
export function FieldHeader({
  label,
  caption,
  className,
}: {
  label: ReactNode;
  caption?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("space-y-1", className)}>
      {label}
      {caption && <p className="text-xs text-muted-foreground">{caption}</p>}
    </div>
  );
}
