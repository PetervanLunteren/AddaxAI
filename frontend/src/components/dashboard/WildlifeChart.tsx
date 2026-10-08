/**
 * "Wildlife detected": the project's top ten wildlife taxa as one-colour
 * horizontal bars, per 100 trap nights. The Overview never has a pick, so
 * under the dashboard's one rule it shows wildlife.
 *
 * Two ways to count (frequency: independent events; abundance: summed
 * observations) and a rank to group by, both chosen in the card. The
 * rank is what answers "how many mammals against birds": at class level
 * the bars are classes.
 *
 * A bar opens Explore for that taxon, so the overview leads straight into
 * the detail. It lives on the Overview only: the same chart on both tabs
 * made them read as one. A bar without taxonomy (a detector-only class
 * such as "shark") has no ids to filter by and is not clickable.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Bar } from "react-chartjs-2";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  LinearScale,
  Tooltip,
  type ChartOptions,
} from "chart.js";

import { statisticsApi } from "../../api/statistics";
import { exploreHref } from "../../lib/explore-filters";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import {
  DEFAULT_TAXONOMIC_RANK,
  RANK_OPTIONS,
} from "../../lib/taxonomic-rank";
import { useChartColors } from "../../lib/theme";
import { Card, CardContent } from "../ui/card";
import { DashboardCardHeader } from "./DashboardCardHeader";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

const TOP_N = 10;

type CountMode = "events" | "max_n";

interface WildlifeChartProps {
  projectId: string;
  /** Project trap nights, so bars read per 100 nights. */
  trapNights: number;
}

export function WildlifeChart({ projectId, trapNights }: WildlifeChartProps) {
  const navigate = useNavigate();
  const colors = useChartColors();
  const [rank, setRank] = useState<string>(DEFAULT_TAXONOMIC_RANK);
  const [countMode, setCountMode] = useState<CountMode>("max_n");

  const { data: species, isLoading } = useQuery({
    queryKey: ["statistics", "species", "wildlife", projectId, rank, countMode],
    queryFn: () =>
      statisticsApi.getSpeciesDistribution(
        projectId,
        undefined,
        undefined,
        undefined,
        rank,
        countMode,
        true,
      ),
  });

  const top = (species ?? []).slice(0, TOP_N);
  const perHundred = (n: number) =>
    trapNights > 0 ? +((n / trapNights) * 100).toFixed(2) : n;
  const axisLabel =
    countMode === "events"
      ? "Independent events per 100 trap nights"
      : "Observations per 100 trap nights";
  const clickable = (index: number) =>
    (top[index]?.label_taxonomy_ids.length ?? 0) > 0;

  const options: ChartOptions<"bar"> = {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    onClick: (_evt, elements) => {
      const index = elements[0]?.index;
      if (index === undefined || !clickable(index)) return;
      navigate(exploreHref(projectId, top[index].label_taxonomy_ids));
    },
    onHover: (evt, elements) => {
      const target = evt.native?.target as HTMLElement | undefined;
      if (!target) return;
      const index = elements[0]?.index;
      target.style.cursor =
        index !== undefined && clickable(index) ? "pointer" : "default";
    },
    plugins: { legend: { display: false } },
    scales: {
      x: {
        beginAtZero: true,
        title: { display: true, text: axisLabel, color: colors.axis },
        ticks: { color: colors.axis },
        grid: { color: colors.grid },
      },
      y: { ticks: { color: colors.axis }, grid: { display: false } },
    },
  };

  const data = {
    labels: top.map((s) =>
      resolveSpeciesName({ scientific_name: s.species, common_name: s.common_name }),
    ),
    datasets: [
      {
        label: axisLabel,
        data: top.map((s) => perHundred(s.count)),
        // One colour: the name is already on the axis, so per-bar
        // colours would only suggest a meaning they do not have.
        backgroundColor: colors.withAlpha(colors.primaryInk, 0.18),
        borderColor: colors.primaryInk,
        borderWidth: 1.25,
        borderRadius: 4,
        barPercentage: 0.75,
      },
    ],
  };

  return (
    <Card>
      <DashboardCardHeader
        title="Wildlife detected"
        caption={
          countMode === "events"
            ? "Top 10 by number of independent events"
            : "Top 10 by total observations"
        }
        info={
          <>
            <p>
              Top 10 wildlife taxa. People, vehicles, and labels like
              blank or false detection are left out. Frequency counts
              the independent events containing the taxon (the basis
              for RAI). Abundance counts individuals, using each
              event's confirmed count, or the AI's count where not yet
              confirmed.
            </p>
            <p>
              The AI's count is the most individuals visible in a
              single frame, so the same animals aren't counted twice
              from frame to frame. Click a bar to explore that taxon.
              Pick a higher rank to compare groups, such as mammals
              against birds at class level.
            </p>
          </>
        }
        actions={
          <>
            <Select value={rank} onValueChange={setRank}>
              <SelectTrigger className="h-9 w-[140px] text-sm" aria-label="Taxonomic rank">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {RANK_OPTIONS.map((o) => (
                  <SelectItem key={o.value} value={o.value}>
                    {o.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select
              value={countMode}
              onValueChange={(v) => setCountMode(v as CountMode)}
            >
              <SelectTrigger className="h-9 w-[120px] text-sm" aria-label="Count by">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="events">Frequency</SelectItem>
                <SelectItem value="max_n">Abundance</SelectItem>
              </SelectContent>
            </Select>
          </>
        }
      />
      <CardContent>
        <div className="h-72">
          {isLoading ? (
            <div className="flex h-full items-center justify-center">
              <p className="text-muted-foreground">Loading...</p>
            </div>
          ) : top.length > 0 ? (
            <Bar data={data} options={options} />
          ) : (
            <div className="flex h-full items-center justify-center">
              <p className="text-muted-foreground">No wildlife detected</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
