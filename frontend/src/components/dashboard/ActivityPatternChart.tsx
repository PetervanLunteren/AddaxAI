/**
 * Activity pattern: 24-hour clock face of observation counts.
 *
 * Pure SVG clock ported from AddaxAI-Connect: 24 radial bars around
 * a circle with hour 0 at the top, color-coded by time of day.
 * Hovering over any bar snaps to the nearest hour and shows the
 * count in the center. Counts are normalized to "per 100 trap
 * nights" when the parent supplies a trapNights prop.
 */

import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Filter } from "lucide-react";
import { cn } from "../../lib/utils";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "../ui/select";
import { DashboardAboutPopover } from "./DashboardAboutPopover";
import { statisticsApi, type SunBands } from "../../api/statistics";
import { normalizeLabel } from "../../utils/labels";
import type { DateRange } from "./index";

// Time-of-day color bands. Fed by project-specific sunrise/sunset
// (astral, backend) so the dawn/day/dusk/night boundaries reflect
// the real sun position at the project's site lat/lon for the
// filter date range.
const NIGHT_COLOR = "#0f6064";
const TWILIGHT_COLOR = "#ff8945";
const DAY_COLOR = "#71b7ba";

/**
 * Pick a color for a given hour based on the project's sun bands.
 *
 * Each bar represents the one-hour span `[hour, hour + 1)`. We
 * classify by overlap, not by point-membership, because at low
 * latitudes civil twilight is only ~20 minutes wide and would
 * otherwise never intersect an integer hour. Twilight takes
 * priority when an hour touches both a twilight window and
 * day/night, so crepuscular activity stays visually obvious.
 *
 * Falls back to the neutral day color when sun bands aren't
 * available (no sites in the project, astral declined to compute,
 * etc.).
 */
function getHourColor(hour: number, sunBands: SunBands | null): string {
  if (!sunBands) return DAY_COLOR;
  const { dawn, sunrise, sunset, dusk } = sunBands;
  const start = hour;
  const end = hour + 1;

  // Overlap with either twilight window?
  const overlapsMorningTwilight = start < sunrise && end > dawn;
  const overlapsEveningTwilight = start < dusk && end > sunset;
  if (overlapsMorningTwilight || overlapsEveningTwilight) {
    return TWILIGHT_COLOR;
  }

  // Overlap with the daylight window?
  if (start < sunset && end > sunrise) return DAY_COLOR;

  return NIGHT_COLOR;
}

interface ActivityClockProps {
  hours: { hour: number; count: number }[];
  /** If non-null, counts are already normalized and suffix the center tooltip. */
  normalized: boolean;
  sunBands: SunBands | null;
}

function ActivityClock({ hours, normalized, sunBands }: ActivityClockProps) {
  // 200x200 viewBox centered at (100, 100). Inner empty circle gives the
  // center tooltip room; outer ring leaves space for hour labels.
  const cx = 100;
  const cy = 100;
  const innerR = 30;
  const outerR = 82;
  const labelR = 92;
  const barWidth = 5;
  const maxBarLength = outerR - innerR;

  const maxCount = Math.max(1, ...hours.map((h) => h.count));

  // hour 0 -> top of the circle: subtract 90 degrees from the standard
  // SVG angle (0 deg = right, increasing clockwise).
  const hourAngle = (hour: number) => ((hour * 15 - 90) * Math.PI) / 180;

  const [hoveredHour, setHoveredHour] = useState<number | null>(null);
  const hoveredEntry =
    hoveredHour !== null ? hours.find((h) => h.hour === hoveredHour) ?? null : null;

  // Snap cursor to the nearest 15-degree slot. Anywhere inside the
  // outer circle counts as a hover; outside clears it.
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 200;
    const y = ((e.clientY - rect.top) / rect.height) * 200;
    const dx = x - cx;
    const dy = y - cy;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist > outerR + 6) {
      setHoveredHour(null);
      return;
    }
    // atan2 returns angle in [-pi, pi] with 0 at the right side.
    // Rotate by +90 deg so 0 is at the top, then snap to nearest 15 deg slot.
    let deg = (Math.atan2(dy, dx) * 180) / Math.PI;
    deg = (deg + 90 + 360) % 360;
    setHoveredHour(Math.round(deg / 15) % 24);
  };

  return (
    <svg
      viewBox="0 0 200 200"
      className="w-full h-full"
      role="img"
      aria-label="Hourly activity clock"
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setHoveredHour(null)}
    >
      {/* Faint guide circles for visual scale */}
      <circle
        cx={cx}
        cy={cy}
        r={innerR + maxBarLength * 0.5}
        fill="none"
        stroke="rgba(0, 0, 0, 0.08)"
        strokeWidth={0.5}
      />
      <circle
        cx={cx}
        cy={cy}
        r={outerR}
        fill="none"
        stroke="rgba(0, 0, 0, 0.08)"
        strokeWidth={0.5}
      />

      {/* 24 radial bars */}
      {hours.map(({ hour, count }) => {
        const angle = hourAngle(hour);
        const length = (count / maxCount) * maxBarLength;
        const x1 = cx + innerR * Math.cos(angle);
        const y1 = cy + innerR * Math.sin(angle);
        const x2 = cx + (innerR + length) * Math.cos(angle);
        const y2 = cy + (innerR + length) * Math.sin(angle);
        const isHovered = hoveredHour === hour;
        return (
          <line
            key={hour}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={getHourColor(hour, sunBands)}
            strokeWidth={isHovered ? barWidth + 2 : barWidth}
            strokeLinecap="round"
          />
        );
      })}

      {/* 8 hour labels every 3 hours, around the rim */}
      {[0, 3, 6, 9, 12, 15, 18, 21].map((hour) => {
        const angle = hourAngle(hour);
        const x = cx + labelR * Math.cos(angle);
        const y = cy + labelR * Math.sin(angle);
        return (
          <text
            key={hour}
            x={x}
            y={y}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={9}
            fill="currentColor"
            className="text-muted-foreground"
          >
            {hour}
          </text>
        );
      })}

      {/* Hover details in the center of the clock */}
      {hoveredEntry && (
        <g style={{ pointerEvents: "none" }}>
          <rect
            x={cx - 38}
            y={cy - 16}
            width={76}
            height={32}
            rx={3}
            fillOpacity={0.85}
            style={{
              fill: "hsl(var(--card))",
              stroke: "hsl(var(--border))",
              strokeWidth: 0.5,
            }}
          />
          <text
            x={cx}
            y={cy - 4}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={10}
            fontWeight="bold"
            fill="currentColor"
            className="text-foreground"
          >
            {`${hoveredEntry.hour.toString().padStart(2, "0")}:00`}
          </text>
          <text
            x={cx}
            y={cy + 7}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={7}
            fill="currentColor"
            className="text-muted-foreground"
          >
            {normalized
              ? `${hoveredEntry.count.toFixed(2)} / 100 nights`
              : `${hoveredEntry.count} observation${hoveredEntry.count === 1 ? "" : "s"}`}
          </text>
        </g>
      )}
    </svg>
  );
}

interface ActivityPatternChartProps {
  dateRange: DateRange;
  projectId: string;
  siteIds?: string;
  trapNights?: number;
  taxonomicRank?: string;
}

export const ActivityPatternChart: React.FC<ActivityPatternChartProps> = ({
  dateRange,
  projectId,
  siteIds,
  trapNights,
  taxonomicRank,
}) => {
  const [selectedSpecies, setSelectedSpecies] = useState("all");

  // Fetch species list for the selector
  const { data: speciesList } = useQuery({
    queryKey: [
      "statistics",
      "species",
      projectId,
      siteIds,
      dateRange.startDate,
      dateRange.endDate,
      taxonomicRank,
    ],
    queryFn: () =>
      statisticsApi.getSpeciesDistribution(
        projectId,
        siteIds,
        dateRange.startDate || undefined,
        dateRange.endDate || undefined,
        taxonomicRank,
      ),
  });

  // Reset species selection when species list changes
  useEffect(() => {
    setSelectedSpecies("all");
  }, [speciesList]);

  // Fetch activity pattern data
  const { data: activityData, isLoading } = useQuery({
    queryKey: [
      "statistics",
      "activity-pattern",
      projectId,
      selectedSpecies,
      siteIds,
      dateRange.startDate,
      dateRange.endDate,
      taxonomicRank,
    ],
    queryFn: () =>
      statisticsApi.getActivityPattern(projectId, {
        species: selectedSpecies === "all" ? undefined : selectedSpecies,
        siteIds,
        dateFrom: dateRange.startDate || undefined,
        dateTo: dateRange.endDate || undefined,
        taxonomicRank,
      }),
  });

  const normalized = !!trapNights && trapNights > 0;
  const norm = (n: number) =>
    normalized ? +((n / (trapNights as number)) * 100).toFixed(2) : n;

  // Build a full 24-hour array, filling missing hours with 0, applying
  // the per-100-trap-nights normalization if requested.
  const hours = useMemo(() => {
    const hourMap = new Map<number, number>();
    activityData?.hours.forEach((h) => hourMap.set(h.hour, h.count));
    const result: { hour: number; count: number }[] = [];
    for (let h = 0; h < 24; h++) {
      result.push({ hour: h, count: norm(hourMap.get(h) ?? 0) });
    }
    return result;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activityData, trapNights]);

  const hasActivity = hours.some((h) => h.count > 0);
  const sunBands: SunBands | null = activityData?.sun_bands ?? null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <div>
            <div className="flex items-center gap-1.5">
              <CardTitle className="text-lg">Activity pattern</CardTitle>
              <DashboardAboutPopover
                what={
                  <p>
                    Observation counts by hour of day for the filtered
                    project view. The coloured bands behind the bars
                    show night, twilight, and day at the project's
                    averaged camera location. Use the species filter to
                    compare patterns between taxa.
                  </p>
                }
                how={
                  <p>
                    For each hour 0 to 23, sums MaxN across every event
                    whose start time falls in that hour. The bands come
                    from python-astral applied to the project's averaged
                    site coordinates and IANA timezone for a date drawn
                    from the filter range midpoint. When the project has
                    no sites with coordinates, the bands are hidden.
                  </p>
                }
              />
            </div>
            <p className="text-sm text-muted-foreground">
              {selectedSpecies === "all"
                ? "Observations by hour of day"
                : `Observations by hour of day · ${normalizeLabel(selectedSpecies)}`}
            </p>
          </div>
          <Select value={selectedSpecies} onValueChange={setSelectedSpecies}>
            <SelectTrigger
              aria-label="Filter by species"
              className={cn(
                "w-auto shrink-0 px-2 gap-1 hover:bg-accent",
                selectedSpecies !== "all" && "bg-accent text-primary"
              )}
            >
              <Filter className="h-4 w-4" />
            </SelectTrigger>
            <SelectContent align="end">
              <SelectItem value="all">All</SelectItem>
              {speciesList?.map((s) => (
                <SelectItem key={s.species} value={s.species}>
                  {normalizeLabel(s.species)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent>
        <div className="aspect-square w-full max-h-80 mx-auto">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">Loading...</p>
            </div>
          ) : hasActivity ? (
            <ActivityClock
              hours={hours}
              normalized={normalized}
              sunBands={sunBands}
            />
          ) : (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">No activity data available</p>
            </div>
          )}
        </div>
        {sunBands && hasActivity && (
          <div className="flex items-center justify-center gap-6 mt-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-3 w-3 rounded-full"
                style={{ backgroundColor: NIGHT_COLOR }}
              />
              Night
            </span>
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-3 w-3 rounded-full"
                style={{ backgroundColor: TWILIGHT_COLOR }}
              />
              Dawn / Dusk
            </span>
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-3 w-3 rounded-full"
                style={{ backgroundColor: DAY_COLOR }}
              />
              Day
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
