/**
 * One-line text that truncates at the start, so the end stays visible:
 * the last folders of a path, the tail of a long species name. The
 * ellipsis lands on the left by laying the box out right-to-left while
 * the text itself stays left-to-right (the inner `dir`), which keeps
 * slashes and brackets in place. The full text is in the tooltip.
 */

import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface TruncateStartProps {
  children: ReactNode;
  title?: string;
  className?: string;
}

export function TruncateStart({ children, title, className }: TruncateStartProps) {
  return (
    <span
      dir="rtl"
      title={title}
      className={cn("block min-w-0 truncate text-left", className)}
    >
      <span dir="ltr">{children}</span>
    </span>
  );
}
