/**
 * Small info popover for dashboard cards. Trigger: an info icon.
 * Holds one short blurb explaining only what isn't obvious from the
 * card itself, no headers or sections.
 */

import { Info } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";

interface DashboardAboutPopoverProps {
  children: React.ReactNode;
}

export function DashboardAboutPopover({ children }: DashboardAboutPopoverProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label="About this view"
          className="text-muted-foreground hover:text-foreground transition-colors"
        >
          <Info className="h-4 w-4" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        side="bottom"
        align="end"
        sideOffset={6}
        collisionPadding={16}
        avoidCollisions
        className="w-72 max-w-[calc(100vw-2rem)] max-h-[var(--radix-popover-content-available-height,80vh)] overflow-y-auto p-3 text-sm text-muted-foreground space-y-2"
      >
        {children}
      </PopoverContent>
    </Popover>
  );
}
