/**
 * Polar area chart showing detection counts by hour of day.
 *
 * Hours are color-coded: Night (teal), Dawn/Dusk (orange), Day (light teal).
 * Includes a species selector to filter detections.
 */

import { useState, useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { PolarArea } from "react-chartjs-2";
import {
  Chart as ChartJS,
  RadialLinearScale,
  ArcElement,
  Tooltip,
  Legend,
  type ChartOptions,
} from "chart.js";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { statisticsApi } from "../../api/statistics";
import { normalizeLabel } from "../../utils/labels";
import type { DateRange } from "./index";

ChartJS.register(RadialLinearScale, ArcElement, Tooltip, Legend);

// Hour color bands
const NIGHT_COLOR = "#0f6064";
const TWILIGHT_COLOR = "#ff8945";
const DAY_COLOR = "#71b7ba";

function getHourColor(hour: number): string {
  if (hour >= 7 && hour < 17) return DAY_COLOR;
  if ((hour >= 5 && hour < 7) || (hour >= 17 && hour < 21)) return TWILIGHT_COLOR;
  return NIGHT_COLOR;
}

function formatHourLabel(hour: number): string {
  const suffix = hour >= 12 ? "PM" : "AM";
  const display = hour === 0 ? 12 : hour > 12 ? hour - 12 : hour;
  return `${display} ${suffix}`;
}

interface ActivityPatternChartProps {
  dateRange: DateRange;
  projectId: string;
  siteIds?: string;
}

export const ActivityPatternChart: React.FC<ActivityPatternChartProps> = ({
  dateRange,
  projectId,
  siteIds,
}) => {
  const [selectedSpecies, setSelectedSpecies] = useState("all");

  // Fetch species list for the selector
  const { data: speciesList } = useQuery({
    queryKey: ["statistics", "species", projectId, siteIds, dateRange.startDate, dateRange.endDate],
    queryFn: () =>
      statisticsApi.getSpeciesDistribution(
        projectId,
        siteIds,
        dateRange.startDate || undefined,
        dateRange.endDate || undefined,
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
    ],
    queryFn: () =>
      statisticsApi.getActivityPattern(projectId, {
        species: selectedSpecies === "all" ? undefined : selectedSpecies,
        siteIds,
        dateFrom: dateRange.startDate || undefined,
        dateTo: dateRange.endDate || undefined,
      }),
  });

  const chartData = useMemo(() => {
    // Build a full 24-hour array, filling missing hours with 0
    const hourMap = new Map<number, number>();
    activityData?.hours.forEach((h) => hourMap.set(h.hour, h.count));

    const labels: string[] = [];
    const values: number[] = [];
    const colors: string[] = [];

    for (let h = 0; h < 24; h++) {
      labels.push(formatHourLabel(h));
      values.push(hourMap.get(h) ?? 0);
      colors.push(getHourColor(h));
    }

    return {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: colors.map((c) => `${c}99`),
          borderColor: colors,
          borderWidth: 1,
        },
      ],
    };
  }, [activityData]);

  const chartOptions: ChartOptions<"polarArea"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (context) => `${context.label}: ${context.parsed.r} detections`,
        },
      },
    },
    scales: {
      r: {
        beginAtZero: true,
        ticks: { display: false },
        grid: { color: "rgba(0,0,0,0.06)" },
      },
    },
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">Activity pattern</CardTitle>
            <p className="text-sm text-muted-foreground">
              {activityData
                ? `${activityData.total_detections.toLocaleString()} total detections`
                : "Detections by hour of day"}
            </p>
          </div>
          <Select value={selectedSpecies} onValueChange={setSelectedSpecies}>
            <SelectTrigger className="w-44 h-9 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All species</SelectItem>
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
        <div className="h-80">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">Loading...</p>
            </div>
          ) : (
            <PolarArea data={chartData} options={chartOptions} />
          )}
        </div>
        {/* Color legend */}
        <div className="flex items-center justify-center gap-6 mt-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: NIGHT_COLOR }} />
            Night (9PM-5AM)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: TWILIGHT_COLOR }} />
            Dawn/Dusk
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: DAY_COLOR }} />
            Day (7AM-5PM)
          </span>
        </div>
      </CardContent>
    </Card>
  );
};
