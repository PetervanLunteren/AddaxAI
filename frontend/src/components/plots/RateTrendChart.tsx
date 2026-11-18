/**
 * The Trend page's chart: the mean of the site rates per week or month,
 * with a 95% confidence band, in the plain box an Insights page puts its
 * one chart in. The page header says what it is and "About this view"
 * says how it is computed, so the box carries no title of its own.
 */

import type { RateTrendResponse, TrendPeriod, TrendPeriodSize } from "../../api/statistics";
import { exceedsChartYears, yearSpan } from "../../lib/date-span";
import { formatRate, rateFromPer100, rateLabel } from "../../lib/trap-night-rate";
import { WideSpanNotice } from "../dashboard/WideSpanNotice";
import type { DrawnChartRef } from "../../lib/chart-image";
import { TrendLineChart } from "./TrendLineChart";

interface RateTrendChartProps {
  data: RateTrendResponse | undefined;
  isLoading: boolean;
  periodSize: TrendPeriodSize;
  /** The Chart.js instance while a chart is drawn, for the PNG download. */
  chartRef?: DrawnChartRef;
}

/** The backend's rates are per 100 trap nights; the chart is in the
 *  user's rate unit. A missing value stays missing. */
const scaled = (value: number | null) => (value === null ? null : rateFromPer100(value));

function tooltipLines(period: TrendPeriod): string[] {
  const sites = `${period.n_sites} site${period.n_sites === 1 ? "" : "s"}`;
  if (period.mean === null) return [];
  const lines = [`Mean: ${formatRate(rateFromPer100(period.mean))} ${rateLabel()}`];
  if (period.ci_low !== null && period.ci_high !== null) {
    lines.push(
      `95% CI: ${formatRate(rateFromPer100(period.ci_low))} to ${formatRate(rateFromPer100(period.ci_high))}`,
    );
  }
  return [...lines, sites];
}

function Message({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-center h-full">
      <p className="max-w-md text-center text-muted-foreground">{children}</p>
    </div>
  );
}

export function RateTrendChart({ data, isLoading, periodSize, chartRef }: RateTrendChartProps) {
  const periods = data?.periods ?? [];
  const span =
    data?.first_date && data.last_date
      ? yearSpan(data.first_date, data.last_date)
      : null;
  const tooWide = span !== null && exceedsChartYears(span);
  const hasMean = periods.some((p) => p.mean !== null);

  return (
    <div className="h-[28rem] rounded-lg border bg-card p-4">
      {isLoading ? (
        <Message>Loading...</Message>
      ) : tooWide ? (
        <div className="flex items-center justify-center h-full">
          <WideSpanNotice span={span} className="max-w-md" />
        </div>
      ) : periods.length === 0 ? (
        <Message>No site with camera effort in this selection</Message>
      ) : periods.length === 1 ? (
        <Message>
          This date range is a single {periodSize}, too short to draw a
          trend. Species rates covers it.
        </Message>
      ) : !hasMean ? (
        <Message>
          No site was active for 80% of any {periodSize} in this
          selection. Species rates still counts every site.
        </Message>
      ) : (
        <TrendLineChart
          chartRef={chartRef}
          labels={periods.map((p) => p.period)}
          spans={periods}
          line={{ label: "Mean rate", values: periods.map((p) => scaled(p.mean)) }}
          band={{
            label: "95% confidence interval",
            low: periods.map((p) => scaled(p.ci_low)),
            high: periods.map((p) => scaled(p.ci_high)),
          }}
          yTitle={`Observations ${rateLabel()}`}
          tooltip={(_series, index) => tooltipLines(periods[index])}
        />
      )}
    </div>
  );
}
