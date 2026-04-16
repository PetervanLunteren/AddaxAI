/**
 * Activity overlap chart — page-wide Cartesian KDE comparison of 1 or 2
 * species' temporal activity patterns.
 *
 * The visual conventions follow the R `overlap` package and Wildlife
 * Insights' activity view:
 *   - 0..24h x-axis (clock or sun-anchored, controlled by the parent)
 *   - one smooth von Mises KDE curve per species (the backend ships
 *     a pre-fit 240-point density grid, so the chart just plots it)
 *   - shaded overlap region = pointwise min(species_a, species_b)
 *   - twilight bands (dawn/sunrise/sunset/dusk) drawn as a background
 *     plugin in clock mode; auto-suppressed in sun-time mode (which
 *     anchors to mean sunrise/sunset by definition)
 *   - rug ticks under the curves showing raw detection times
 *
 * The math (KDE fit, sun-band computation) lives server-side in
 * `app.ml.activity_analysis` so this file is purely visual. See
 * DEVELOPERS.md "Datetime conventions" for why hours are camera local.
 */

import { useMemo } from "react";
import { Line } from "react-chartjs-2";
import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  type ChartData,
  type ChartOptions,
  type Plugin,
} from "chart.js";

import type {
  ActivityOverlapResponse,
  SunBands,
} from "../../api/statistics";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
);

// Species colors pinned to slots A and B so the picker swatches, the
// chart curves, and the legend badges always agree. Picked to match
// the AddaxAI palette (teal + accent orange).
export const SPECIES_A_COLOR = "#0f6064";
export const SPECIES_B_COLOR = "#ff8945";
const OVERLAP_FILL = "rgba(120, 120, 120, 0.28)";
const RUG_HEIGHT_PX = 6;

/**
 * Background plugin that paints dawn/dusk twilight bands and the
 * night region across the chart area. Reads `sunBands` from the chart
 * options' plugin block. Disabled when `sunBands` is null (e.g. in
 * sun-time mode, where the bands are redundant).
 */
const twilightBandsPlugin: Plugin<"line"> = {
  id: "twilightBands",
  beforeDatasetsDraw(chart, _args, options) {
    const opts = options as { sunBands?: SunBands | null; visible?: boolean };
    if (!opts.visible || !opts.sunBands) return;
    const { dawn, sunrise, sunset, dusk } = opts.sunBands;
    const { ctx, chartArea, scales } = chart;
    const xScale = scales.x;
    if (!xScale) return;
    const top = chartArea.top;
    const bottom = chartArea.bottom;

    const xAt = (h: number) => xScale.getPixelForValue(h);

    ctx.save();

    // Night bands: 0..dawn and dusk..24
    ctx.fillStyle = "rgba(30, 41, 59, 0.06)";
    ctx.fillRect(chartArea.left, top, xAt(dawn) - chartArea.left, bottom - top);
    ctx.fillRect(xAt(dusk), top, chartArea.right - xAt(dusk), bottom - top);

    // Twilight bands: dawn..sunrise and sunset..dusk
    ctx.fillStyle = "rgba(255, 165, 0, 0.10)";
    ctx.fillRect(xAt(dawn), top, xAt(sunrise) - xAt(dawn), bottom - top);
    ctx.fillRect(xAt(sunset), top, xAt(dusk) - xAt(sunset), bottom - top);

    ctx.restore();
  },
};

/**
 * Foreground plugin that draws short vertical "rug" ticks at the bottom
 * of the chart area for each species' raw detection times. Uses two
 * vertical bands (species A on top, species B just below) so each
 * tick is associated with its species without visual overlap.
 */
const rugTicksPlugin: Plugin<"line"> = {
  id: "rugTicks",
  afterDatasetsDraw(chart, _args, options) {
    const opts = options as {
      speciesA?: number[];
      speciesB?: number[];
      colorA?: string;
      colorB?: string;
    };
    const { ctx, chartArea, scales } = chart;
    const xScale = scales.x;
    if (!xScale) return;

    const drawRug = (times: number[] | undefined, y: number, color: string) => {
      if (!times || times.length === 0) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.6;
      for (const t of times) {
        const x = xScale.getPixelForValue(t);
        if (x < chartArea.left || x > chartArea.right) continue;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x, y + RUG_HEIGHT_PX);
        ctx.stroke();
      }
      ctx.restore();
    };

    drawRug(
      opts.speciesA,
      chartArea.bottom - 2 * RUG_HEIGHT_PX - 2,
      opts.colorA ?? SPECIES_A_COLOR,
    );
    drawRug(
      opts.speciesB,
      chartArea.bottom - RUG_HEIGHT_PX,
      opts.colorB ?? SPECIES_B_COLOR,
    );
  },
};

ChartJS.register(twilightBandsPlugin, rugTicksPlugin);

interface ActivityOverlapChartProps {
  data: ActivityOverlapResponse;
  /** "clock" = 0..24h camera local time, "sun" = sun-anchored. */
  timeAxis: "clock" | "sun";
  /** When false, hide the dawn/dusk overlay even in clock mode. */
  bandsVisible: boolean;
}

const SAMPLES = 240;

/** 240-point grid in [0..24) hours, evenly spaced. Matches the backend KDE grid. */
const GRID_HOURS: number[] = Array.from(
  { length: SAMPLES },
  (_, i) => (24 * i) / SAMPLES,
);

export function ActivityOverlapChart({
  data,
  timeAxis,
  bandsVisible,
}: ActivityOverlapChartProps) {
  const overlapMin = useMemo(() => {
    if (!data.species_b) return null;
    const a = data.species_a.kde_density;
    const b = data.species_b.kde_density;
    return a.map((v, i) => Math.min(v, b[i] ?? 0));
  }, [data]);

  const chartData: ChartData<"line"> = useMemo(() => {
    const datasets: ChartData<"line">["datasets"] = [];

    // Shaded overlap region (drawn first so curves render on top)
    if (overlapMin && data.species_b) {
      datasets.push({
        label: "Overlap",
        data: overlapMin,
        borderColor: "transparent",
        backgroundColor: OVERLAP_FILL,
        fill: "origin",
        pointRadius: 0,
        tension: 0.4,
        order: 3,
      });
    }

    datasets.push({
      label: data.species_a.label,
      data: data.species_a.kde_density,
      borderColor: SPECIES_A_COLOR,
      backgroundColor: "transparent",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.4,
      fill: false,
      order: 1,
    });

    if (data.species_b) {
      datasets.push({
        label: data.species_b.label,
        data: data.species_b.kde_density,
        borderColor: SPECIES_B_COLOR,
        backgroundColor: "transparent",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.4,
        fill: false,
        order: 2,
      });
    }

    return {
      labels: GRID_HOURS,
      datasets,
    };
  }, [data, overlapMin]);

  const options: ChartOptions<"line"> = useMemo(() => {
    const xTitle =
      timeAxis === "clock"
        ? "Hour of day (camera local)"
        : "Hour of day (relative to mean sunrise/sunset)";
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          type: "linear",
          min: 0,
          max: 24,
          ticks: {
            stepSize: 3,
            callback: (value) => `${String(value).padStart(2, "0")}:00`,
          },
          title: { display: true, text: xTitle },
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: "Activity density" },
          ticks: {
            callback: (value) => Number(value).toFixed(2),
          },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: {
            filter: (item) => item.text !== "Overlap",
          },
        },
        tooltip: {
          callbacks: {
            title: (items) => {
              const hour = Number(items[0]?.label ?? 0);
              const h = Math.floor(hour);
              const m = Math.round((hour - h) * 60);
              return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
            },
            label: (item) => {
              if (item.dataset.label === "Overlap") return "";
              return `${item.dataset.label}: ${item.parsed.y.toFixed(3)}`;
            },
          },
        },
        // @ts-expect-error custom plugin options aren't in Chart.js's typings
        twilightBands: {
          sunBands: timeAxis === "clock" && bandsVisible ? data.sun_bands : null,
          visible: timeAxis === "clock" && bandsVisible,
        },
        // @ts-expect-error custom plugin options aren't in Chart.js's typings
        rugTicks: {
          speciesA: data.species_a.raw_detection_times,
          speciesB: data.species_b?.raw_detection_times,
          colorA: SPECIES_A_COLOR,
          colorB: SPECIES_B_COLOR,
        },
      },
    };
  }, [data, timeAxis, bandsVisible]);

  return (
    <div className="h-[420px] w-full">
      <Line data={chartData} options={options} />
    </div>
  );
}
