/**
 * Stats toolbar for the Events tab showing verification progress.
 *
 * Displays representative and file verification progress bars,
 * plus a help button. Visual style matches the Similarity toolbar.
 */

import { CircleHelp } from "lucide-react";
import type { EventVerificationStats } from "../../api/types";

interface EventsStatsToolbarProps {
  stats: EventVerificationStats | undefined;
  onHelpClick: () => void;
}

export function EventsStatsToolbar({ stats, onHelpClick }: EventsStatsToolbarProps) {
  if (!stats) return null;

  const repPct =
    stats.total_representatives > 0
      ? (stats.verified_representatives / stats.total_representatives) * 100
      : 0;
  const filePct =
    stats.total_files > 0
      ? (stats.verified_files / stats.total_files) * 100
      : 0;

  return (
    <div className="flex flex-wrap items-center gap-3 py-2 px-3 bg-white rounded-lg border shadow-sm">
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
              style={{ width: `${repPct}%`, backgroundColor: "#0f6064" }}
            />
          </div>
          {Math.round(repPct)}% representatives verified ({stats.verified_representatives}/{stats.total_representatives})
        </div>
        <div className="h-4 w-px bg-border" />
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <div className="relative h-2 w-20 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full transition-all duration-500 ease-out rounded-full"
              style={{ width: `${filePct}%`, backgroundColor: "#0f6064" }}
            />
          </div>
          {Math.round(filePct)}% files verified ({stats.verified_files}/{stats.total_files})
        </div>
      </div>
    </div>
  );
}
