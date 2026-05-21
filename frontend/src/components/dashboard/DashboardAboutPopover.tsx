/**
 * "About this view" popover for dashboard cards.
 *
 * Trigger: small info icon. Click-to-open. Shows two sub-sections —
 * "What it shows" and "How it's computed" — with the same structure
 * the insight pages use via PlotExplainer, just compressed into a
 * popover so it does not take vertical space inside the card itself.
 */

import { Info } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";

interface DashboardAboutPopoverProps {
  what: React.ReactNode;
  how: React.ReactNode;
}

export function DashboardAboutPopover({ what, how }: DashboardAboutPopoverProps) {
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
        className="w-80 max-w-[calc(100vw-2rem)] max-h-[var(--radix-popover-content-available-height,80vh)] overflow-y-auto p-4 space-y-4"
      >
        <p className="text-sm font-semibold">About this view</p>
        <section className="space-y-1.5">
          <h4 className="text-sm font-semibold">What it shows</h4>
          <div className="text-sm text-muted-foreground space-y-2">{what}</div>
        </section>
        <section className="space-y-1.5">
          <h4 className="text-sm font-semibold">How it's computed</h4>
          <div className="text-sm text-muted-foreground space-y-2">{how}</div>
        </section>
      </PopoverContent>
    </Popover>
  );
}
