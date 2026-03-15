/**
 * Multi-line chart comparing detection trends across species.
 *
 * Uses toggleable species chips. Auto-selects top 3 species on load.
 * Fetches each selected species trend in parallel via useQueries.
 */

import { useState, useEffect, useMemo, useCallback } from "react";
import { useQuery, useQueries } from "@tanstack/react-query";
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
import { statisticsApi } from "../../api/statistics";
import { normalizeLabel } from "../../utils/labels";
import {
  setSpeciesContext,
  getSpeciesColor,
  getSpeciesColorWithAlpha,
} from "../../utils/species-colors";
import type { DateRange } from "./index";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const MAX_SELECTED = 5;

interface SpeciesComparisonChartProps {
  dateRange: DateRange;
  projectId: string;
  siteIds?: string;
}

export const SpeciesComparisonChart: React.FC<SpeciesComparisonChartProps> = ({
  dateRange,
  projectId,
  siteIds,
}) => {
  const [selectedSpecies, setSelectedSpecies] = useState<string[]>([]);

  // Fetch species list
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

  // Auto-select top 3 species when list first loads
  useEffect(() => {
    if (speciesList && speciesList.length > 0 && selectedSpecies.length === 0) {
      const topSpecies = [...speciesList]
        .sort((a, b) => b.count - a.count)
        .slice(0, 3)
        .map((s) => s.species);
      setSelectedSpecies(topSpecies);
    }
  }, [speciesList, selectedSpecies.length]);

  // Set color context whenever species list changes
  useEffect(() => {
    if (speciesList) {
      setSpeciesContext(speciesList.map((s) => s.species));
    }
  }, [speciesList]);

  // Sorted species names for chip display
  const sortedSpeciesNames = useMemo(
    () => (speciesList ? [...speciesList].map((s) => s.species).sort() : []),
    [speciesList],
  );

  const toggleSpecies = (species: string) => {
    setSelectedSpecies((prev) => {
      if (prev.includes(species)) {
        return prev.filter((s) => s !== species);
      }
      if (prev.length >= MAX_SELECTED) return prev;
      return [...prev, species];
    });
  };

  // Fetch trend data for each selected species in parallel
  const speciesQueries = useQueries({
    queries: selectedSpecies.map((species) => ({
      queryKey: [
        "statistics",
        "detection-trend",
        projectId,
        species,
        siteIds,
        dateRange.startDate,
        dateRange.endDate,
      ],
      queryFn: () =>
        statisticsApi.getDetectionTrend(projectId, {
          species,
          siteIds,
          dateFrom: dateRange.startDate || undefined,
          dateTo: dateRange.endDate || undefined,
        }),
    })),
  });

  const isLoading = speciesQueries.some((q) => q.isLoading);

  // Build gradient for a given species color
  const createGradient = useCallback(
    (ctx: CanvasRenderingContext2D, chartArea: { top: number; bottom: number }, species: string) => {
      const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
      gradient.addColorStop(0, getSpeciesColorWithAlpha(species, 0.3));
      gradient.addColorStop(1, getSpeciesColorWithAlpha(species, 0.02));
      return gradient;
    },
    [],
  );

  // Merge all species datasets onto a shared date axis
  const chartData = useMemo(() => {
    // Collect all unique dates across species
    const allDates = new Set<string>();
    speciesQueries.forEach((q) => {
      q.data?.forEach((p) => allDates.add(p.date));
    });
    const labels = [...allDates].sort();

    const datasets = selectedSpecies.map((species, idx) => {
      const points = speciesQueries[idx]?.data ?? [];
      const dateMap = new Map(points.map((p) => [p.date, p.count]));

      return {
        label: normalizeLabel(species),
        data: labels.map((d) => dateMap.get(d) ?? 0),
        borderColor: getSpeciesColor(species),
        backgroundColor: (context: { chart: ChartJS }) => {
          const { chart } = context;
          if (!chart.chartArea) return getSpeciesColorWithAlpha(species, 0.2);
          return createGradient(chart.ctx, chart.chartArea, species);
        },
        fill: true,
        tension: 0.3,
        pointRadius: labels.length > 60 ? 0 : 3,
        pointHoverRadius: 5,
      };
    });

    return { labels, datasets };
  }, [selectedSpecies, speciesQueries, createGradient]);

  const chartOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: {
        ticks: { maxTicksLimit: 12, maxRotation: 45 },
        grid: { display: false },
      },
      y: {
        beginAtZero: true,
        ticks: {
          callback: (value) => Number(value).toLocaleString(),
        },
      },
    },
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Species comparison</CardTitle>
        <p className="text-sm text-muted-foreground">
          Select up to {MAX_SELECTED} species to compare trends
        </p>
      </CardHeader>
      <CardContent>
        {/* Species chips */}
        <div className="flex flex-wrap gap-2 mb-4">
          {sortedSpeciesNames.map((species) => {
            const isSelected = selectedSpecies.includes(species);
            const color = getSpeciesColor(species);
            return (
              <button
                key={species}
                type="button"
                onClick={() => toggleSpecies(species)}
                className={
                  "px-3 py-1 rounded-full text-xs font-medium border transition-colors " +
                  (isSelected
                    ? "text-white border-transparent"
                    : "text-muted-foreground border-input hover:border-primary/50")
                }
                style={isSelected ? { backgroundColor: color, borderColor: color } : undefined}
                disabled={!isSelected && selectedSpecies.length >= MAX_SELECTED}
              >
                {normalizeLabel(species)}
              </button>
            );
          })}
        </div>

        {/* Chart */}
        <div className="h-80">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">Loading...</p>
            </div>
          ) : selectedSpecies.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">Select at least one species</p>
            </div>
          ) : (
            <Line data={chartData} options={chartOptions} />
          )}
        </div>
      </CardContent>
    </Card>
  );
};
