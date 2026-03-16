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
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
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

function groupData(
  points: DetectionTrendPoint[],
  granularity: Granularity,
): { labels: string[]; values: number[] } {
  if (granularity === "day") {
    return {
      labels: points.map((p) => p.date),
      values: points.map((p) => p.count),
    };
  }

  const grouped = new Map<string, number>();
  for (const point of points) {
    const key = granularity === "week" ? getWeekKey(point.date) : getMonthKey(point.date);
    grouped.set(key, (grouped.get(key) ?? 0) + point.count);
  }

  const sortedKeys = [...grouped.keys()].sort();
  return {
    labels: sortedKeys,
    values: sortedKeys.map((k) => grouped.get(k)!),
  };
}

/**
 * Pick a sensible default granularity based on the number of raw data points.
 */
function pickGranularity(pointCount: number): Granularity {
  if (pointCount > 180) return "month";
  if (pointCount > 60) return "week";
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

  // Auto-select optimal granularity when data arrives
  useEffect(() => {
    if (trendData) {
      setGranularity(pickGranularity(trendData.length));
    }
  }, [trendData]);

  const { labels, values } = useMemo(
    () => (trendData ? groupData(trendData, granularity) : { labels: [], values: [] }),
    [trendData, granularity],
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
            <CardTitle className="text-lg">Detection trend</CardTitle>
            <p className="text-sm text-muted-foreground">
              Detections over time
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
