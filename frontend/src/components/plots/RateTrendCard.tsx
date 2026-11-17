/**
 * The Trend page's chart: the mean of the site rates per week or month,
 * with a 95% confidence band.
 *
 * Presentational. The page fetches one response for this card and the
 * species table, so the two are always the same scope.
 */

import type { RateTrendResponse, TrendPeriod } from "../../api/statistics";
import { exceedsChartYears, yearSpan } from "../../lib/date-span";
import { formatRate, rateFromPer100, rateLabel } from "../../lib/trap-night-rate";
import { DashboardCardHeader } from "../dashboard/DashboardCardHeader";
import { MissingDatesIcon } from "../dashboard/MissingDatesWarning";
import { WideSpanNotice } from "../dashboard/WideSpanNotice";
import { Card, CardContent } from "../ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { DownloadCsvButton } from "./DownloadCsvButton";
import { TrendLineChart } from "./TrendLineChart";

export type TrendPeriodSize = "week" | "month";

interface RateTrendCardProps {
  projectId: string;
  data: RateTrendResponse | undefined;
  isLoading: boolean;
  periodSize: TrendPeriodSize;
  onPeriodSizeChange: (size: TrendPeriodSize) => void;
  onDownload: () => void;
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

export function RateTrendCard({
  projectId,
  data,
  isLoading,
  periodSize,
  onPeriodSizeChange,
  onDownload,
}: RateTrendCardProps) {
  const periods = data?.periods ?? [];
  const span =
    data?.first_date && data.last_date
      ? yearSpan(data.first_date, data.last_date)
      : null;
  const tooWide = span !== null && exceedsChartYears(span);
  const hasMean = periods.some((p) => p.mean !== null);

  return (
    <Card>
      <DashboardCardHeader
        title="Rate over time"
        caption="The mean of the site rates, with a 95% confidence band"
        marker={<MissingDatesIcon projectId={projectId} />}
        info={
          <>
            <p>
              Each site gets its own observation rate per {periodSize}.
              The line is the average of those site rates, and the band
              shows how sure that average is: with more sites, or sites
              that agree, the band is narrower.
            </p>
            <p>
              A site counts in a {periodSize} when it was active on at
              least 80% of its days. A {periodSize} without such a site
              is a gap in the line.
            </p>
          </>
        }
        actions={
          <>
            <Select
              value={periodSize}
              onValueChange={(v) => onPeriodSizeChange(v as TrendPeriodSize)}
            >
              <SelectTrigger className="w-28 h-9 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="week">Weekly</SelectItem>
                <SelectItem value="month">Monthly</SelectItem>
              </SelectContent>
            </Select>
            <DownloadCsvButton onClick={onDownload} disabled={periods.length === 0} />
          </>
        }
      />
      <CardContent>
        <div className="h-96">
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
              trend. The table below covers it.
            </Message>
          ) : !hasMean ? (
            <Message>
              No site was active for 80% of any {periodSize} in this
              selection. The table below still counts every site.
            </Message>
          ) : (
            <TrendLineChart
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
      </CardContent>
    </Card>
  );
}
