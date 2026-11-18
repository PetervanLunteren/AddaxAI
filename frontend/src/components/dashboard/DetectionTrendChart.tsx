/**
 * The Explore tab's "Observation trend": observations per day, week or
 * month, with a dashed moving average.
 *
 * A count, not a rate. The dashboard has to be quick, and effort per
 * period means reading every file row of the scope; the Trend page has
 * the rate and its uncertainty, one click away under "Open trend". What
 * it counts comes from the page: the Explore tab's taxon, sites and
 * dates, passed in as one `scope`, so this card can never disagree with
 * the cards beside it. The backend cuts the time into periods
 * (`crud/rate_trend.py`).
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { statisticsApi } from "../../api/statistics";
import type { DashboardScope, TrendGranularity } from "../../api/statistics";
import { wideSpan } from "../../lib/date-span";
import { TrendLineChart } from "../plots/TrendLineChart";
import { Card, CardContent } from "../ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { CARD_HOVER_PILL } from "./card-hover-pill";
import { DashboardCardHeader } from "./DashboardCardHeader";
import { MissingDatesIcon } from "./MissingDatesWarning";
import { WideSpanNotice } from "./WideSpanNotice";

/**
 * Moving-average window per granularity, chosen so the window maps to a
 * natural cycle (a week, a month, a year). Keeps the smoothed line
 * comparable across granularities.
 */
const SMOOTHING: Record<TrendGranularity, { size: number; label: string }> = {
  day: { size: 7, label: "7-day average" },
  week: { size: 4, label: "4-week average" },
  month: { size: 12, label: "12-month average" },
};

/**
 * Trailing simple moving average. Same length as the input; the first
 * (window - 1) slots are null so the chart leaves a warm-up gap instead
 * of drawing a partial average.
 */
function trailingMean(values: number[], window: number): (number | null)[] {
  let sum = 0;
  return values.map((value, i) => {
    sum += value - (i >= window ? values[i - window] : 0);
    return i >= window - 1 ? +(sum / window).toFixed(2) : null;
  });
}

interface DetectionTrendChartProps {
  projectId: string;
  scope: DashboardScope;
  /** The Trend page with the same labels, sites and dates. */
  trendHref: string;
}

export const DetectionTrendChart: React.FC<DetectionTrendChartProps> = ({
  projectId,
  scope,
  trendHref,
}) => {
  // Until the user picks one, the backend chooses the period size from
  // the span: days for a short survey, months for years.
  const [picked, setPicked] = useState<TrendGranularity>();

  const { data, isLoading } = useQuery({
    queryKey: ["statistics", "detection-trend", projectId, scope, picked],
    queryFn: () => statisticsApi.getDetectionTrend(projectId, scope, picked),
  });

  const granularity = picked ?? data?.granularity ?? "day";
  const points = data?.points ?? [];

  // A corrupt capture year makes the range millennia wide; the backend
  // sends the two dates and no points, and the card says why.
  const tooWide = wideSpan(data?.first_date, data?.last_date);

  const smoothing = SMOOTHING[granularity];
  const counts = points.map((p) => p.count);

  return (
    <Card className="group">
      <DashboardCardHeader
        title="Observation trend"
        caption="Observations over time"
        marker={<MissingDatesIcon projectId={projectId} />}
        info={
          <>
            <p>
              Observations per day, week or month (each event's confirmed
              count, or the AI's count where not yet confirmed), not
              detections, of the labels, sites and dates chosen above.
            </p>
            <p>
              A count, not a rate: it rises when more cameras run. For the
              rate against trap nights, with its uncertainty, use Open
              trend. It opens Trend with the same labels, sites and dates.
            </p>
          </>
        }
        actions={
          <>
            <Link to={trendHref} className={CARD_HOVER_PILL}>
              Open trend
            </Link>
            <Select value={granularity} onValueChange={(v) => setPicked(v as TrendGranularity)}>
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
          ) : tooWide ? (
            <div className="flex items-center justify-center h-full">
              <WideSpanNotice span={tooWide} className="max-w-md" />
            </div>
          ) : points.length > 0 ? (
            <TrendLineChart
              labels={points.map((p) => p.period)}
              spans={points}
              line={{ label: "Observations", values: counts }}
              dashed={{ label: smoothing.label, values: trailingMean(counts, smoothing.size) }}
              yTitle="Observations"
              tooltip={(series, _index, value) =>
                `${series === "line" ? "Observations" : smoothing.label}: ${value.toLocaleString()}`
              }
            />
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
