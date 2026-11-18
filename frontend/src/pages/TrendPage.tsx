/**
 * Trend page: the observation rate over time, with its uncertainty.
 *
 * The site is the sampling unit. Each site gets its own rate, and the
 * chart draws the mean of those rates per week or month with a 95%
 * confidence band (backend: `get_rate_trend` in `crud/rate_trend.py`).
 * The same statistic per species is the Species rates page.
 *
 * The frame, the filters and the download are `RatePageLayout`. The
 * period size is this page's display control, in the filter bar like the
 * view mode on the Map.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";

import { statisticsApi, type TrendPeriodSize } from "../api/statistics";
import { FilterField } from "../components/dashboard/ExploreFilterBar";
import { PlotExplainer } from "../components/plots/PlotExplainer";
import { RatePageLayout } from "../components/plots/RatePageLayout";
import { RateTrendChart } from "../components/plots/RateTrendChart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import { ROVERO_MARSHALL_2009, SWEETAPPLE_HICKLING_2024 } from "../lib/rate-references";
import { DATA_CSV, IMAGE_PNG } from "../components/plots/DownloadMenu";
import { saveChartPng, useDrawnChart } from "../lib/chart-image";
import {
  downloadFileName,
  saveCsv,
  trendPeriodsCsv,
  type FileDates,
  type RateCsvScope,
} from "../lib/insights-csv";
import { useChartColors } from "../lib/theme";
import { useExploreFilters } from "./dashboard/useExploreFilters";

export function TrendPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const id = projectId as string;
  const explore = useExploreFilters(id);
  const { scope } = explore;

  // Until the user picks one, the backend chooses week or month from
  // the span of the data.
  const [picked, setPicked] = useState<TrendPeriodSize>();

  const { data, isLoading } = useQuery({
    queryKey: ["statistics", "rate-trend", id, scope, picked],
    queryFn: () => statisticsApi.getRateTrend(id, scope, picked),
    enabled: explore.ready && !explore.noSiteMatch,
  });
  // The chart while one is drawn, so "Image (PNG)" is offered only then.
  const [chart, chartRef] = useDrawnChart();
  const chartColors = useChartColors();

  const periodSize: TrendPeriodSize =
    picked ?? (data?.granularity === "month" ? "month" : "week");

  return (
    <RatePageLayout
      projectId={id}
      title="Trend"
      subtitle="Observation rate over time, with its uncertainty"
      explore={explore}
      dates={data && { first: data.first_date, last: data.last_date }}
      downloads={[
        ...(data?.periods.length
          ? [
              {
                label: DATA_CSV,
                save: (fileScope: RateCsvScope, dates: FileDates) =>
                  saveCsv(`trend-${periodSize}`, trendPeriodsCsv(data.periods, fileScope), dates),
              },
            ]
          : []),
        // Only while a chart is drawn: not on the single-week or empty states.
        ...(chart
          ? [
              {
                label: IMAGE_PNG,
                save: (_scope: RateCsvScope, dates: FileDates) =>
                  saveChartPng(
                    chart,
                    chartColors.card,
                    downloadFileName(`trend-${periodSize}`, "png", dates),
                  ),
              },
            ]
          : []),
      ]}
      filterExtra={
        <FilterField label="Period">
          <Select value={periodSize} onValueChange={(v) => setPicked(v as TrendPeriodSize)}>
            {/* The filter bar's other controls are h-9; the shared trigger
                defaults to min-h-10. Same fix as the Deployment timeline. */}
            <SelectTrigger className="h-9 min-h-9">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="week">Weekly</SelectItem>
              <SelectItem value="month">Monthly</SelectItem>
            </SelectContent>
          </Select>
        </FilterField>
      }
      explainer={
        <PlotExplainer
          plotKey="trend"
          what={
            <p>
              Each site gets its own observation rate. The line is the average
              of those site rates per week or month, and the band shows how sure
              that average is: with more sites, or sites that agree, the band is
              narrower. Hover a point for its dates, its 95% confidence interval
              and the number of sites behind it. An observation counts
              individuals: each event's confirmed count, or the AI's count where
              not yet confirmed. A trap night is one day a camera was active.
              The average of the site rates differs from the single rate on the
              dashboard when sites ran for different lengths of time: here every
              site weighs the same. The rate is an index of how many animals
              there are, not a count of them: it can track density closely
              (Rovero & Marshall 2009), but it also rises and falls with how
              active animals are, for example between seasons.
            </p>
          }
          how={
            <p>
              Per site, rate = observations / trap_nights, shown in the unit
              chosen under View &gt; Rate unit. A site with effort and no
              observations is a rate of zero. Then mean and standard deviation
              across the n sites, and 95% confidence interval = mean ± t × SD /
              √n, with t from Student's distribution for n − 1 degrees of
              freedom, the method of New Zealand's camera trap catch index
              (Sweetapple & Hickling 2024). The lower bound stops at zero. A site counts in a week or
              month when it was active on at least 80% of its days, so a camera
              that started or stopped partway does not distort the period. A
              period without such a site is a gap, not a zero, and so is a week
              or month the date range cuts short. Deployments without a site are
              left out. With few sites the interval is wide and rough. Read the
              number of sites next to it.
            </p>
          }
          references={[ROVERO_MARSHALL_2009, SWEETAPPLE_HICKLING_2024]}
        />
      }
    >
      <RateTrendChart
        data={data}
        isLoading={isLoading}
        periodSize={periodSize}
        chartRef={chartRef}
      />
    </RatePageLayout>
  );
}

export default TrendPage;
