/**
 * The line chart both trend cards draw: a rate per period over time.
 *
 * Presentational: it gets ready values and knows nothing about where
 * they come from. The Explore card gives it a pooled rate with a dashed
 * moving average; the Trend page gives it the mean of the site rates
 * with a shaded 95% confidence band. One component, so the two cannot
 * drift apart in colour, axis styling or tooltip behaviour.
 *
 * A null value is a gap in the line: a period without effort, or one no
 * site ran long enough to count in. It is never drawn as zero.
 */

import { useCallback, useMemo } from "react";
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
  type ChartDataset,
  type ChartOptions,
} from "chart.js";

import { formatCameraDate } from "../../lib/datetime";
import type { DrawnChartRef } from "../../lib/chart-image";
import { useChartColors } from "../../lib/theme";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

type Values = (number | null)[];

/** Which drawn line a tooltip row is for. The band has no rows. */
export type TrendSeries = "line" | "dashed";

/** The days a period covers, for the tooltip title: "15 Jan 2024" for
 *  a day, "15 Jan 2024 to 21 Jan 2024" for a longer period. */
export interface PeriodSpan {
  start: string;
  end: string;
}

function spanTitle({ start, end }: PeriodSpan): string {
  const first = formatCameraDate(start);
  return start === end ? first : `${first} to ${formatCameraDate(end)}`;
}

interface TrendLineChartProps {
  /** One label per period, on the x axis. */
  labels: string[];
  /** The days of each period, so the tooltip says which dates a point
   *  covers instead of a week number. */
  spans: PeriodSpan[];
  /** The solid line. */
  line: { label: string; values: Values };
  /** A dashed second line, e.g. a moving average. */
  dashed?: { label: string; values: Values };
  /** A shaded band around the line, e.g. a confidence interval. Without
   *  a band the area under the line is filled instead. */
  band?: { label: string; low: Values; high: Values };
  yTitle: string;
  /** Tooltip text for a point; an array is one row per entry. */
  tooltip: (series: TrendSeries, index: number, value: number) => string | string[];
  /** The drawn chart, for a page that saves it as an image. */
  chartRef?: DrawnChartRef;
}

export function TrendLineChart({
  labels,
  spans,
  line,
  dashed,
  band,
  yTitle,
  tooltip,
  chartRef,
}: TrendLineChartProps) {
  // One series at a time, so the color carries no information: always
  // teal ink from the theme (lighter in dark, where raw brand teal
  // disappears into the card). Species colors from species-colors.ts
  // are for charts where several species appear together.
  const colors = useChartColors();
  const lineColor = colors.primaryInk;

  const createGradient = useCallback(
    (ctx: CanvasRenderingContext2D, chartArea: { top: number; bottom: number }) => {
      const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
      gradient.addColorStop(0, colors.withAlpha(lineColor, 0.4));
      gradient.addColorStop(1, colors.withAlpha(lineColor, 0.02));
      return gradient;
    },
    [colors, lineColor],
  );

  // Each dataset says which series it is, so the tooltip can ask the
  // caller for its text and skip the band's two edges.
  const { datasets, seriesOf } = useMemo(() => {
    const out: ChartDataset<"line", Values>[] = [];
    const roles: (TrendSeries | null)[] = [];
    if (band) {
      // The lower edge is invisible; the upper edge fills down to it.
      out.push(
        {
          label: band.label,
          data: band.low,
          borderWidth: 0,
          pointRadius: 0,
          pointHoverRadius: 0,
          tension: 0.3,
          fill: false,
        },
        {
          label: band.label,
          data: band.high,
          borderWidth: 0,
          pointRadius: 0,
          pointHoverRadius: 0,
          tension: 0.3,
          backgroundColor: colors.withAlpha(lineColor, 0.18),
          fill: "-1",
        },
      );
      roles.push(null, null);
    }
    out.push({
      label: line.label,
      data: line.values,
      borderColor: lineColor,
      backgroundColor: band
        ? "transparent"
        : (context: { chart: ChartJS }) => {
            const { chart } = context;
            if (!chart.chartArea) return colors.withAlpha(lineColor, 0.2);
            return createGradient(chart.ctx, chart.chartArea);
          },
      fill: !band,
      tension: 0.3,
      pointRadius: line.values.length > 60 ? 0 : 3,
      pointHoverRadius: 5,
    });
    roles.push("line");
    if (dashed) {
      out.push({
        label: dashed.label,
        data: dashed.values,
        borderColor: lineColor,
        backgroundColor: "transparent",
        borderDash: [6, 4],
        fill: false,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 0,
      });
      roles.push("dashed");
    }
    return { datasets: out, seriesOf: roles };
  }, [line, dashed, band, colors, lineColor, createGradient]);

  const options: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        display: true,
        position: "top",
        align: "end",
        labels: {
          boxWidth: 24,
          usePointStyle: false,
          color: colors.axis,
          // The band is two datasets with one name: show the filled one.
          filter: (item) => !band || item.datasetIndex !== 0,
        },
      },
      tooltip: {
        // No rows for the band's edges, a gap, or a moving average that
        // is still warming up.
        filter: (item) =>
          seriesOf[item.datasetIndex] !== null && item.parsed.y != null,
        callbacks: {
          title: (items) => (items.length ? spanTitle(spans[items[0].dataIndex]) : ""),
          label: (context) => {
            const series = seriesOf[context.datasetIndex];
            const y = context.parsed.y;
            if (series === null || y == null) return "";
            return tooltip(series, context.dataIndex, y);
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
        title: { display: true, text: yTitle, color: colors.axis },
        ticks: {
          callback: (value) => Number(value).toLocaleString(),
          color: colors.axis,
        },
        grid: { color: colors.grid },
      },
    },
  };

  return <Line ref={chartRef} data={{ labels, datasets }} options={options} />;
}
