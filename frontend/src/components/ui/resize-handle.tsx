import type { MouseEvent as ReactMouseEvent } from "react";

import { cn } from "@/lib/utils";

/**
 * The bar a person grabs to resize a panel below it. Pairs with
 * `useResizableHeight`, which supplies `onMouseDown`. The default look
 * is the one the Counts modal's filmstrip divider had: a thin strip
 * that lights up on hover; a dark surface passes its own classes. A
 * short grip in the middle says it can be dragged, as split views do:
 * a plain line only gave that away by its cursor, on hover.
 */
export function ResizeHandle({
  onMouseDown,
  title,
  className,
  testId = "resize-handle",
}: {
  onMouseDown: (e: ReactMouseEvent) => void;
  title: string;
  className?: string;
  /** Distinct when two handles share a page, so a test can tell them apart. */
  testId?: string;
}) {
  return (
    <div
      role="separator"
      aria-orientation="horizontal"
      onMouseDown={onMouseDown}
      title={title}
      data-testid={testId}
      className={cn(
        "flex h-1.5 shrink-0 cursor-row-resize items-center justify-center transition-colors",
        className ?? "bg-border hover:bg-primary/40",
      )}
    >
      <span className="h-0.5 w-8 rounded-full bg-muted-foreground/50" aria-hidden />
    </div>
  );
}
