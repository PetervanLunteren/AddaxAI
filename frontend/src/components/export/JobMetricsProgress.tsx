/**
 * The progress view of a job that reports `metrics` (current / total): a
 * spinner, the worker's message, the percentage and a bar. Shared by the
 * export dialog and the package import, so a counted job looks the same
 * everywhere.
 */

import { Loader2 } from "lucide-react";

import type { TqdmMetrics } from "../../hooks/useTaskProgress";
import { Progress } from "../ui/progress";

export function JobMetricsProgress({
  message,
  metrics,
  phaseProgress,
}: {
  message: string;
  metrics: TqdmMetrics | null;
  phaseProgress: number;
}) {
  const current = metrics?.current ?? 0;
  const total = metrics?.total ?? 0;
  const fraction = total > 0 ? current / total : phaseProgress;
  const pct = Math.round(fraction * 100);

  return (
    <div className="space-y-3 py-2">
      <div className="flex items-center gap-3">
        <Loader2 className="h-5 w-5 shrink-0 animate-spin text-primary-ink" />
        <span className="text-sm font-medium text-foreground">
          {message || "Starting..."}
        </span>
        {total > 0 && (
          <span className="ml-auto font-mono text-xs text-muted-foreground">
            {pct}%
          </span>
        )}
      </div>
      {total > 0 && <Progress value={pct} className="h-2" />}
    </div>
  );
}
