/**
 * Stats toolbar for the Events tab.
 *
 * Single progress bar: "X% events verified", where an event counts as
 * verified once all its MaxN frames are verified (blank events fall
 * back to "any file verified"). Mirrors the single-bar treatment the
 * Files and Observations tabs use.
 */

import { CircleHelp } from "lucide-react";
import type { EventVerificationStats } from "../../api/types";

interface EventsStatsToolbarProps {
  stats: EventVerificationStats | undefined;
  onHelpClick: () => void;
}

export function EventsStatsToolbar({ stats, onHelpClick }: EventsStatsToolbarProps) {
  if (!stats) return null;

  const pct =
    stats.events_total > 0
      ? (stats.events_fully_verified / stats.events_total) * 100
      : 0;

  return (
    <div className="flex flex-wrap items-center gap-3 min-h-12 py-2 px-3 bg-white rounded-lg border shadow-sm">
      <button
        onClick={onHelpClick}
        className="text-muted-foreground hover:text-foreground transition-colors"
        title="Help"
      >
        <CircleHelp className="h-4 w-4" />
      </button>

      <div className="flex items-center gap-3 ml-auto">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <div className="relative h-2 w-20 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full transition-all duration-500 ease-out rounded-full"
              style={{ width: `${pct}%`, backgroundColor: "#0f6064" }}
            />
          </div>
          {Math.round(pct)}% events verified
        </div>
      </div>
    </div>
  );
}
