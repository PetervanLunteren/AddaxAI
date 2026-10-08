/**
 * Line chart showing observations over time with gradient fill.
 *
 * Supports day/week/month granularity and picks one from the span on load.
 * What it counts comes from the page: the Explore tab's taxon, sites and
 * dates, passed in as one `scope`, so this card can never disagree with
 * the cards beside it.
 */

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
  type ChartOptions,
} from "chart.js";
import { Card, CardContent } from "../ui/card";
import { DashboardCardHeader } from "./DashboardCardHeader";
import { useChartColors } from "../../lib/theme";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { MissingDatesIcon } from "./MissingDatesWarning";
import { statisticsApi } from "../../api/statistics";
import type { DetectionTrendPoint, DashboardScope } from "../../api/statistics";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

type Granularity = "day" | "week" | "month";

// --- Grouping helpers ---

function getWeekKey(dateStr: string): string {
  const d = new Date(dateStr);
  // ISO week number calculation
  const temp = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  temp.setDate(temp.getDate() + 3 - ((temp.getDay() + 6) % 7));
  const yearStart = new Date(temp.getFullYear(), 0, 1);
  const weekNum = Math.ceil((((temp.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
  return `${temp.getFullYear()}-W${String(weekNum).padStart(2, "0")}`;
}

function getMonthKey(dateStr: string): string {
  return dateStr.slice(0, 7); // "YYYY-MM"
}

/**
 * Generate every bucket key between `from` and `to` inclusive, at the
 * given granularity. Iterates day-by-day and dedupes so week and month
 * keys come out without gaps even though the calendar inside them is
 * irregular (ISO weeks don't line up with 7-day increments starting
 * mid-week; months have variable length).
 */
function denseRangeKeys(
  from: Date,
  to: Date,
  keyOf: (dateStr: string) => string,
): string[] {
  const keys: string[] = [];
  const seen = new Set<string>();
  const cursor = new Date(from);
  // UTC-based advance to avoid DST edge cases shifting the cursor.
  while (cursor.getTime() <= to.getTime()) {
    const iso = cursor.toISOString().slice(0, 10);
    const k = keyOf(iso);
    if (!seen.has(k)) {
      seen.add(k);
      keys.push(k);
    }
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return keys;
}

function groupData(
  points: DetectionTrendPoint[],
  granularity: Granularity,
  rangeStart: string | null,
  rangeEnd: string | null,
): { labels: string[]; values: number[] } {
  const keyOf: (d: string) => string =
    granularity === "day"
      ? (d) => d
      : granularity === "week"
        ? getWeekKey
        : getMonthKey;

  // Bucket observed counts by key (multiple point.date values can
  // collapse to the same week or month key).
  const bucketed = new Map<string, number>();
  for (const point of points) {
    const k = keyOf(point.date);
    bucketed.set(k, (bucketed.get(k) ?? 0) + point.count);
  }

  // Inclusive range bounds: user's filter wins; otherwise fall back to
  // the first and last observed dates in the data.
  const from = rangeStart ?? points[0]?.date;
  const to = rangeEnd ?? points[points.length - 1]?.date;
  if (!from || !to) return { labels: [], values: [] };

  // Dense list of bucket keys across the whole range, then zero-fill.
  const labels = denseRangeKeys(new Date(from), new Date(to), keyOf);
  const values = labels.map((k) => bucketed.get(k) ?? 0);
  return { labels, values };
}

/**
 * Pick a sensible default granularity based on the span of the range
 * in days. Days for short surveys, weeks for quarters, months for
 * multi-year projects.
 */
function pickGranularity(days: number): Granularity {
  if (days > 365) return "month";
  if (days > 90) return "week";
  return "day";
}

/**
 * Rolling-average window per granularity, chosen so the window maps to a
 * natural cycle (a week, a month, a year). Keeps the smoothed line
 * comparable across granularities.
 */
function getSmoothingWindow(
  granularity: Granularity,
): { size: number; label: string } {
  if (granularity === "day") return { size: 7, label: "7-day average" };
  if (granularity === "week") return { size: 4, label: "4-week average" };
  return { size: 12, label: "12-month average" };
}

/**
 * Trailing simple moving average. Same length as the input; the first
 * (window - 1) slots are null so the chart leaves a warm-up gap instead
 * of drawing a partial average.
 */
function trailingMovingAverage(
  values: number[],
  window: number,
): (number | null)[] {
  if (values.length < window) return values.map(() => null);
  const out: (number | null)[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    out.push(i >= window - 1 ? +(sum / window).toFixed(2) : null);
  }
  return out;
}

interface DetectionTrendChartProps {
  projectId: string;
  scope: DashboardScope;
  /** Trap nights in the same scope, so the line reads per 100 nights. */
  trapNights?: number;
}

export const DetectionTrendChart: React.FC<DetectionTrendChartProps> = ({
  projectId,
  scope,
  trapNights,
}) => {
  const [granularity, setGranularity] = useState<Granularity>("day");
  const chartRef = useRef<ChartJS<"line"> | null>(null);
  const rangeStart = scope.dateFrom ?? null;
  const rangeEnd = scope.dateTo ?? null;

  const norm = (n: number) => trapNights && trapNights > 0 ? +(n / trapNights * 100).toFixed(2) : n;

  const { data: trendData, isLoading } = useQuery({
    queryKey: ["statistics", "detection-trend", projectId, scope],
    queryFn: () => statisticsApi.getDetectionTrend(projectId, scope),
  });

  // Auto-select optimal granularity when data arrives. Keyed on the
  // span of the chart's range in days (user filter wins, else the first
  // and last observed dates), not the point count — after zero-filling
  // empty days the point count is roughly equal to the span anyway.
  useEffect(() => {
    if (!trendData || trendData.length === 0) return;
    const first = rangeStart ?? trendData[0].date;
    const last = rangeEnd ?? trendData[trendData.length - 1].date;
    const days =
      Math.round(
        (new Date(last).getTime() - new Date(first).getTime()) / 86400000,
      ) + 1;
    setGranularity(pickGranularity(days));
  }, [trendData, rangeStart, rangeEnd]);

  const { labels, values } = useMemo(
    () =>
      trendData
        ? groupData(trendData, granularity, rangeStart, rangeEnd)
        : { labels: [], values: [] },
    [trendData, granularity, rangeStart, rangeEnd],
  );

  const normalizedValues = useMemo(() => values.map(norm), [values, trapNights]);

  // Trailing moving average over the rate series, window tied to the
  // granularity. Reveals the trend under the bursty raw line.
  const { size: smoothingWindow, label: smoothingLabel } =
    getSmoothingWindow(granularity);
  const movingAverage = useMemo(
    () => trailingMovingAverage(normalizedValues, smoothingWindow),
    [normalizedValues, smoothingWindow],
  );

  // One series at a time, so the color carries no information: always
  // teal ink from the theme (lighter in dark, where raw brand teal
  // disappears into the card). Species colors from species-colors.ts
  // are for charts where several species appear together.
  const colors = useChartColors();
  const lineColor = colors.primaryInk;

  // Build gradient fill for the line
  const createGradient = useCallback(
    (ctx: CanvasRenderingContext2D, chartArea: { top: number; bottom: number }) => {
      const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
      gradient.addColorStop(0, colors.withAlpha(lineColor, 0.4));
      gradient.addColorStop(1, colors.withAlpha(lineColor, 0.02));
      return gradient;
    },
    [colors, lineColor],
  );

  const chartData = useMemo(
    () => ({
      labels,
      datasets: [
        {
          label: "Observations",
          data: normalizedValues,
          borderColor: lineColor,
          backgroundColor: (context: { chart: ChartJS }) => {
            const { chart } = context;
            if (!chart.chartArea) return colors.withAlpha(lineColor, 0.2);
            return createGradient(chart.ctx, chart.chartArea);
          },
          fill: true,
          tension: 0.3,
          pointRadius: normalizedValues.length > 60 ? 0 : 3,
          pointHoverRadius: 5,
        },
        {
          label: smoothingLabel,
          data: movingAverage,
          borderColor: lineColor,
          backgroundColor: "transparent",
          borderDash: [6, 4],
          fill: false,
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 0,
        },
      ],
    }),
    [
      labels,
      normalizedValues,
      movingAverage,
      smoothingLabel,
      colors,
      lineColor,
      createGradient,
    ],
  );

  const chartOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        display: true,
        position: "top",
        align: "end",
        labels: { boxWidth: 24, usePointStyle: false, color: colors.axis },
      },
      tooltip: {
        // Drop the moving-average's warm-up null points from the tooltip.
        filter: (item) => item.parsed.y != null,
        callbacks: {
          label: (context) => {
            const y = context.parsed.y;
            if (y == null) return "";
            return `${context.dataset.label}: ${y.toLocaleString()} per 100 trap nights`;
          },
        },
      },
    },
    scales: {
      x: {
        ticks: {
          maxTicksLimit: 12,
          maxRotation: 45,
          color: colors.axis,
        },
        grid: { display: false },
      },
      y: {
        beginAtZero: true,
        title: { display: true, text: "Per 100 trap nights", color: colors.axis },
        ticks: {
          callback: (value) => Number(value).toLocaleString(),
          color: colors.axis,
        },
        grid: { color: colors.grid },
      },
    },
  };

  return (
    <Card>
      <DashboardCardHeader
        title="Observation trend"
        caption="Observations over time"
        marker={<MissingDatesIcon projectId={projectId} />}
        info={
          <>
            <p>
              Observations per bin (each event's confirmed count, or
              the AI's count where not yet confirmed), not
              detections, per 100 trap nights, of the labels, sites
              and dates chosen above. Bin by day, week, or month.
            </p>
          </>
        }
        actions={
          <>
            <Select value={granularity} onValueChange={(v) => setGranularity(v as Granularity)}>
              <SelectTrigger className="w-28 h-9 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="day">Daily</SelectItem>
                <SelectItem value="week">Weekly</SelectItem>
                <SelectItem value="month">Monthly</SelectItem>
              </SelectContent>
            </Select>
          </>
        }
      />
      <CardContent>
        <div className="h-80">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">Loading...</p>
            </div>
          ) : values.length > 0 ? (
            <Line ref={chartRef} data={chartData} options={chartOptions} />
          ) : (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">No observation data available</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
