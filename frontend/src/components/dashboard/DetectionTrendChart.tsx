/**
 * Line chart showing detection trends over time with gradient fill.
 *
 * Supports day/week/month granularity and species filtering.
 * Auto-selects the most observed species and optimal granularity on load.
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
import { Info } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Tooltip as UITooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";
import { statisticsApi } from "../../api/statistics";
import { normalizeLabel } from "../../utils/labels";
import { getSpeciesColor, getSpeciesColorWithAlpha } from "../../utils/species-colors";
import type { DateRange } from "./index";
import type { DetectionTrendPoint } from "../../api/statistics";

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

interface DetectionTrendChartProps {
  dateRange: DateRange;
  projectId: string;
  siteIds?: string;
  trapNights?: number;
  taxonomicRank?: string;
}

export const DetectionTrendChart: React.FC<DetectionTrendChartProps> = ({
  dateRange,
  projectId,
  siteIds,
  trapNights,
  taxonomicRank,
}) => {
  const [granularity, setGranularity] = useState<Granularity>("day");
  const [selectedSpecies, setSelectedSpecies] = useState("all");
  const chartRef = useRef<ChartJS<"line"> | null>(null);

  const norm = (n: number) => trapNights && trapNights > 0 ? +(n / trapNights * 100).toFixed(2) : n;

  // Fetch species list for the selector
  const { data: speciesList } = useQuery({
    queryKey: ["statistics", "species", projectId, siteIds, dateRange.startDate, dateRange.endDate, taxonomicRank],
    queryFn: () =>
      statisticsApi.getSpeciesDistribution(
        projectId,
        siteIds,
        dateRange.startDate || undefined,
        dateRange.endDate || undefined,
        taxonomicRank,
      ),
  });

  // Auto-select the most observed species on load and when species list changes
  useEffect(() => {
    if (speciesList && speciesList.length > 0) {
      const top = speciesList.reduce((best, s) => (s.count > best.count ? s : best), speciesList[0]);
      setSelectedSpecies(top.species);
    }
  }, [speciesList]);

  // Fetch trend data
  const { data: trendData, isLoading } = useQuery({
    queryKey: [
      "statistics",
      "detection-trend",
      projectId,
      selectedSpecies,
      siteIds,
      dateRange.startDate,
      dateRange.endDate,
      taxonomicRank,
    ],
    queryFn: () =>
      statisticsApi.getDetectionTrend(projectId, {
        species: selectedSpecies === "all" ? undefined : selectedSpecies,
        siteIds,
        dateFrom: dateRange.startDate || undefined,
        dateTo: dateRange.endDate || undefined,
        taxonomicRank,
      }),
  });

  // Auto-select optimal granularity when data arrives. Keyed on the
  // span of the chart's range in days (user filter wins, else the first
  // and last observed dates), not the point count — after zero-filling
  // empty days the point count is roughly equal to the span anyway.
  useEffect(() => {
    if (!trendData || trendData.length === 0) return;
    const first = dateRange.startDate ?? trendData[0].date;
    const last =
      dateRange.endDate ?? trendData[trendData.length - 1].date;
    const days =
      Math.round(
        (new Date(last).getTime() - new Date(first).getTime()) / 86400000,
      ) + 1;
    setGranularity(pickGranularity(days));
  }, [trendData, dateRange.startDate, dateRange.endDate]);

  const { labels, values } = useMemo(
    () =>
      trendData
        ? groupData(
            trendData,
            granularity,
            dateRange.startDate,
            dateRange.endDate,
          )
        : { labels: [], values: [] },
    [trendData, granularity, dateRange.startDate, dateRange.endDate],
  );

  const normalizedValues = useMemo(() => values.map(norm), [values, trapNights]);

  // Derive line color from selected species
  const lineColor = selectedSpecies === "all" ? "#0f6064" : getSpeciesColor(selectedSpecies);

  // Build gradient fill for the line
  const createGradient = useCallback(
    (ctx: CanvasRenderingContext2D, chartArea: { top: number; bottom: number }) => {
      const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
      if (selectedSpecies === "all") {
        gradient.addColorStop(0, "rgba(15, 96, 100, 0.4)");
        gradient.addColorStop(1, "rgba(15, 96, 100, 0.02)");
      } else {
        gradient.addColorStop(0, getSpeciesColorWithAlpha(selectedSpecies, 0.4));
        gradient.addColorStop(1, getSpeciesColorWithAlpha(selectedSpecies, 0.02));
      }
      return gradient;
    },
    [selectedSpecies],
  );

  const chartData = useMemo(
    () => ({
      labels,
      datasets: [
        {
          label: "Detections",
          data: normalizedValues,
          borderColor: lineColor,
          backgroundColor: (context: { chart: ChartJS }) => {
            const { chart } = context;
            if (!chart.chartArea) return getSpeciesColorWithAlpha(selectedSpecies === "all" ? "" : selectedSpecies, 0.2);
            return createGradient(chart.ctx, chart.chartArea);
          },
          fill: true,
          tension: 0.3,
          pointRadius: normalizedValues.length > 60 ? 0 : 3,
          pointHoverRadius: 5,
        },
      ],
    }),
    [labels, normalizedValues, lineColor, createGradient, selectedSpecies],
  );

  const chartOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (context) => {
            return `${context.parsed.y.toLocaleString()} per 100 trap nights`;
          },
        },
      },
    },
    scales: {
      x: {
        ticks: {
          maxTicksLimit: 12,
          maxRotation: 45,
        },
        grid: { display: false },
      },
      y: {
        beginAtZero: true,
        title: { display: true, text: "Per 100 trap nights" },
        ticks: {
          callback: (value) => Number(value).toLocaleString(),
        },
      },
    },
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-1.5">
              <CardTitle className="text-lg">Detection trend</CardTitle>
              <TooltipProvider delayDuration={200}>
                <UITooltip>
                  <TooltipTrigger asChild>
                    <Info className="h-4 w-4 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent side="bottom" className="max-w-sm p-3">
                    <p>Shows the total number of observations per day over the survey period. Use the species filter to see trends for a single species.</p>
                  </TooltipContent>
                </UITooltip>
              </TooltipProvider>
            </div>
            <p className="text-sm text-muted-foreground">
              Observations over time
            </p>
          </div>
          <div className="flex items-center gap-2">
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
            <Select value={selectedSpecies} onValueChange={setSelectedSpecies}>
              <SelectTrigger className="w-44 h-9 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                {speciesList?.map((s) => (
                  <SelectItem key={s.species} value={s.species}>
                    {normalizeLabel(s.species)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
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
              <p className="text-muted-foreground">No detection data available</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
