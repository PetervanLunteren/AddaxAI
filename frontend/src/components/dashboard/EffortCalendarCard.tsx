/**
 * Survey effort at a glance: when the cameras were running, as a calendar
 * of weeks (columns) by years (rows), darker for more cameras.
 *
 * Every rate per 100 trap nights rests on this effort, and a gap in it is
 * the first thing to see before reading any trend. The data is the
 * deployment timeline's own cameras-running series, fetched without the
 * per-site daily file counts it does not use (megabytes on a big
 * project), and the whole card opens the timeline page for the detail.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { timelineApi } from "../../api/timeline";
import {
  effortLevel,
  WEEKS_PER_YEAR,
  weeklyCameras,
} from "../../lib/effort-calendar";
import { cn } from "../../lib/utils";
import { Card, CardContent } from "../ui/card";
import { DashboardCardHeader } from "./DashboardCardHeader";

/** Teal ink at four strengths, so it reads in both themes. */
const LEVEL_FILL = [
  undefined,
  "color-mix(in srgb, var(--primary-ink) 30%, transparent)",
  "color-mix(in srgb, var(--primary-ink) 55%, transparent)",
  "color-mix(in srgb, var(--primary-ink) 80%, transparent)",
  "var(--primary-ink)",
];

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** Week column each month starts in (non-leap year; a day off is invisible). */
const MONTH_WEEK = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334].map(
  (dayOfYear) => Math.floor(dayOfYear / 7),
);

const weekStartLabel = (year: number, week: number) =>
  new Date(Date.UTC(year, 0, 1 + week * 7)).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });

interface EffortCalendarCardProps {
  projectId: string;
  className?: string;
}

export function EffortCalendarCard({ projectId, className }: EffortCalendarCardProps) {
  const { data, isLoading } = useQuery({
    queryKey: ["timeline", projectId, "effort-calendar"],
    queryFn: () => timelineApi.get(projectId, { withoutHeatmap: true }),
  });

  const rows = useMemo(
    () => weeklyCameras(data?.concurrent_cameras ?? []),
    [data],
  );
  const max = Math.max(0, ...rows.flatMap((r) => r.weeks));
  const grid = { gridTemplateColumns: `repeat(${WEEKS_PER_YEAR}, minmax(0, 1fr))` };

  return (
    // The whole card is the link, so the week tooltips stay reachable.
    <Link
      to={`/projects/${projectId}/insights/timeline`}
      aria-label="Open the deployment timeline"
      className={cn("block", className)}
    >
    <Card className="flex h-full flex-col transition-shadow hover:shadow-md">
      <DashboardCardHeader title="Survey effort" caption="Cameras running per week" />
      <CardContent className="flex flex-1 flex-col">
        {isLoading ? (
          <div className="h-20 animate-pulse rounded-md bg-muted" />
        ) : rows.length === 0 ? (
          <p className="py-6 text-center text-sm text-muted-foreground">
            No capture dates to show effort for
          </p>
        ) : (
          <div className="space-y-1 text-[10px] text-muted-foreground">
            <div className="flex gap-2">
              <span className="w-8 shrink-0" />
              <div className="grid flex-1 gap-[2px]" style={grid}>
                {MONTHS.map((m, i) => (
                  <span key={m} style={{ gridColumnStart: MONTH_WEEK[i] + 1 }}>
                    {m}
                  </span>
                ))}
              </div>
            </div>
            {rows.map(({ year, weeks }) => (
              <div key={year} className="flex items-center gap-2">
                <span className="w-8 shrink-0 tabular-nums">{year}</span>
                <div className="grid flex-1 gap-[2px]" style={grid}>
                  {weeks.map((cameras, w) => {
                    const level = effortLevel(cameras, max);
                    return (
                      <span
                        key={w}
                        className={cn("aspect-square rounded-[2px]", level === 0 && "bg-muted")}
                        style={{ backgroundColor: LEVEL_FILL[level] }}
                        title={`Week of ${weekStartLabel(year, w)}: ${cameras} camera${cameras === 1 ? "" : "s"}`}
                      />
                    );
                  })}
                </div>
              </div>
            ))}
            <div className="flex items-center justify-end gap-1 pt-1">
              <span className="mr-1">None</span>
              {LEVEL_FILL.map((fill, level) => (
                <span
                  key={level}
                  className={cn("h-2.5 w-2.5 rounded-[2px]", level === 0 && "bg-muted")}
                  style={{ backgroundColor: fill }}
                />
              ))}
              <span className="ml-1">{max} camera{max === 1 ? "" : "s"}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
    </Link>
  );
}
