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
 *     plugin. In clock mode they come from a single reference date's
 *     dawn/sunrise/sunset/dusk; in sun mode they come from the mean
 *     anchor bands across the observation set.
 *   - rug ticks under the curves showing raw detection times
 *
 * The math (KDE fit, sun-band computation) lives server-side in
 * `app.ml.activity_analysis` so this file is purely visual. See
 * DEVELOPERS.md "Datetime conventions" for why hours are camera local.
 */

import { useMemo } from "react";
import { Line } from "react-chartjs-2";

import { useChartColors, useTheme } from "../../lib/theme";
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
import { inSunBand } from "../../lib/sun-bands";

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
// chart curves, and the legend badges always agree. Slot A is the
// theme's teal ink (lighter in dark); slot B is the accent orange,
// which reads on both themes. One hook so every consumer stays in
// sync with the theme.
export function useSpeciesSlotColors(): { a: string; b: string } {
  const colors = useChartColors();
  return { a: colors.primaryInk, b: "#ff8945" };
}
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
    const opts = options as {
      sunBands?: SunBands | null;
      visible?: boolean;
      nightFill?: string;
    };
    if (!opts.visible || !opts.sunBands) return;
    const { dawn, sunrise, sunset, dusk } = opts.sunBands;
    const { ctx, chartArea, scales } = chart;
    const xScale = scales.x;
    if (!xScale) return;
    const top = chartArea.top;
    const bottom = chartArea.bottom;

    const xAt = (h: number) => xScale.getPixelForValue(h);
    // A band may cross the axis edge (dusk at 24.5), so paint it one
    // day either side too and let the clip keep what is on the axis.
    const paintBand = (start: number, end: number) => {
      for (const shift of [-24, 0, 24]) {
        const x = xAt(start + shift);
        ctx.fillRect(x, top, xAt(end + shift) - x, bottom - top);
      }
    };

    ctx.save();
    ctx.beginPath();
    ctx.rect(chartArea.left, top, chartArea.right - chartArea.left, bottom - top);
    ctx.clip();

    ctx.fillStyle = opts.nightFill ?? "rgba(30, 41, 59, 0.06)";
    paintBand(dusk, dawn + 24);

    ctx.fillStyle = "rgba(255, 165, 0, 0.10)";
    paintBand(dawn, sunrise);
    paintBand(sunset, dusk);

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

    // Colours always come through the options block (injected per
    // render from the theme); no module-level fallback, so a theme
    // flip can never leave a stale closure colour behind.
    if (opts.colorA) {
      drawRug(opts.speciesA, chartArea.bottom - 2 * RUG_HEIGHT_PX - 2, opts.colorA);
    }
    if (opts.colorB) {
      drawRug(opts.speciesB, chartArea.bottom - RUG_HEIGHT_PX, opts.colorB);
    }
  },
};

ChartJS.register(twilightBandsPlugin, rugTicksPlugin);

interface ActivityOverlapChartProps {
  data: ActivityOverlapResponse;
  /** Pre-resolved dataset labels (common or scientific per the active
   *  display preference). Fall back to the response's scientific key. */
  speciesAName?: string;
  speciesBName?: string;
}

const SAMPLES = 240;

/** 240-point grid in [0..24) hours, evenly spaced. Matches the backend KDE grid. */
const GRID_HOURS: number[] = Array.from(
  { length: SAMPLES },
  (_, i) => (24 * i) / SAMPLES,
);

export function ActivityOverlapChart({
  data,
  speciesAName,
  speciesBName,
}: ActivityOverlapChartProps) {
  // Use the effective axis the backend actually delivered, not the
  // user's toggle. Sun mode can silently downgrade to clock when a
  // project has no site coordinates or every observation's date is
  // polar; in that case the x-axis title + bands should reflect clock.
  const timeAxis = data.time_axis;

  // In sun mode we shift the x-axis so the anchor sunrise lands at 0
  // and the anchor sunset lands at +day_length. Axis range becomes
  // [-anchor_sunrise, 24 - anchor_sunrise]. Shifted data is monotonic
  // within a "day that starts 6 h before sunrise and ends 18 h after
  // sunrise", which is low-density for most ecological plots, so the
  // midnight wrap cut is visually clean. Falls back to 0 shift when
  // the backend downgraded to clock or bands are missing.
  const sunShift =
    timeAxis === "sun" && data.anchor_sun_bands
      ? data.anchor_sun_bands.sunrise
      : 0;

  const gridX: number[] = useMemo(
    () => GRID_HOURS.map((h) => h - sunShift),
    [sunShift],
  );

  const overlapMin = useMemo(() => {
    if (!data.species_b) return null;
    const a = data.species_a.kde_density;
    const b = data.species_b.kde_density;
    return a.map((v, i) => Math.min(v, b[i] ?? 0));
  }, [data]);

  const colors = useChartColors();
  const slotColors = useSpeciesSlotColors();
  const { resolvedTheme } = useTheme();

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
      label: speciesAName ?? data.species_a.label,
      data: data.species_a.kde_density,
      borderColor: slotColors.a,
      backgroundColor: "transparent",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.4,
      fill: false,
      order: 1,
    });

    if (data.species_b) {
      datasets.push({
        label: speciesBName ?? data.species_b.label,
        data: data.species_b.kde_density,
        borderColor: slotColors.b,
        backgroundColor: "transparent",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.4,
        fill: false,
        order: 2,
      });
    }

    return {
      labels: gridX,
      datasets,
    };
  }, [data, overlapMin, gridX, speciesAName, speciesBName, slotColors]);

  const options: ChartOptions<"line"> = useMemo(() => {
    const isSun = timeAxis === "sun";
    const xTitle = isSun ? "" : "Hour of day (camera local)";
    // In sun mode use the mean-anchor bands; in clock mode use the
    // single-reference bands. Both are shifted by `sunShift` so they
    // land correctly on the axis.
    const rawBands = isSun ? data.anchor_sun_bands : data.sun_bands;
    const bandsForMode = rawBands
      ? {
          dawn: rawBands.dawn - sunShift,
          sunrise: rawBands.sunrise - sunShift,
          sunset: rawBands.sunset - sunShift,
          dusk: rawBands.dusk - sunShift,
        }
      : null;
    // Axis range: clock mode stays [0, 24). Sun mode is [−sunrise, 24−sunrise]
    // so sunrise sits at 0 and the cut points are the opposing midnight.
    const xMin = 0 - sunShift;
    const xMax = 24 - sunShift;
    // Phase tick positions in the shifted sun-time frame. In sun mode
    // we show only these (no numeric hour labels), because hours are
    // fictional on the synthetic anchor day; the sun events are the
    // only markers that carry biological meaning.
    const dayLength = rawBands ? rawBands.sunset - rawBands.sunrise : 0;
    const dawnPos = bandsForMode?.dawn ?? 0;
    const sunrisePos = 0;
    const noonPos = dayLength / 2;
    const sunsetPos = dayLength;
    const duskPos = bandsForMode?.dusk ?? 0;
    const TOL = 0.01;
    const fmtTick = (value: number): string => {
      if (!isSun) return `${String(value).padStart(2, "0")}:00`;
      if (Math.abs(value - xMin) < TOL || Math.abs(value - xMax) < TOL) return "midnight";
      if (Math.abs(value - dawnPos) < TOL) return "dawn";
      if (Math.abs(value - sunrisePos) < TOL) return "sunrise";
      if (Math.abs(value - noonPos) < TOL) return "noon";
      if (Math.abs(value - sunsetPos) < TOL) return "sunset";
      if (Math.abs(value - duskPos) < TOL) return "dusk";
      return "";
    };
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          type: "linear",
          min: xMin,
          max: xMax,
          ticks: {
            color: colors.axis,
            ...(isSun
              ? {
                  autoSkip: false,
                  callback: (v) => fmtTick(Number(v)),
                }
              : {
                  stepSize: 3,
                  callback: (v) => fmtTick(Number(v)),
                }),
          },
          grid: { color: colors.grid },
          afterBuildTicks: isSun && rawBands
            ? (scale) => {
                // dawn and dusk ticks dropped: they sit so close to
                // sunrise / sunset that their labels always collide.
                // The twilight bands themselves still mark those
                // transitions visually. A sunset past midnight (Nome,
                // Alaska in May) lies beyond the axis and is dropped;
                // its band still paints next to the midnight tick.
                scale.ticks = [
                  { value: xMin },
                  { value: sunrisePos },
                  { value: noonPos },
                  { value: sunsetPos },
                  { value: xMax },
                ].filter((t) => t.value <= xMax);
              }
            : undefined,
          title: { display: !!xTitle, text: xTitle, color: colors.axis },
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: "Activity density", color: colors.axis },
          ticks: {
            callback: (value) => Number(value).toFixed(2),
            color: colors.axis,
          },
          grid: { color: colors.grid },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: {
            filter: (item) => item.text !== "Overlap",
            color: colors.axis,
          },
        },
        tooltip: {
          callbacks: {
            title: (items) => {
              const x = Number(items[0]?.label ?? 0);
              if (isSun) {
                if (inSunBand(x, dawnPos, sunrisePos)) return "dawn";
                if (inSunBand(x, sunrisePos, sunsetPos)) return "day";
                if (inSunBand(x, sunsetPos, duskPos)) return "dusk";
                return "night";
              }
              const h = Math.floor(x);
              const m = Math.round((x - h) * 60);
              return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
            },
            label: (item) => {
              if (item.dataset.label === "Overlap") return "";
              return `${item.dataset.label}: ${(item.parsed.y ?? 0).toFixed(3)}`;
            },
          },
        },
        twilightBands: {
          sunBands: bandsForMode,
          visible: bandsForMode !== null,
          // On dark the night region darkens instead of greying out.
          nightFill: resolvedTheme === "dark"
            ? "rgba(0, 0, 0, 0.30)"
            : "rgba(30, 41, 59, 0.06)",
        },
        rugTicks: {
          speciesA: data.species_a.raw_detection_times.map((t) => t - sunShift),
          speciesB: data.species_b?.raw_detection_times.map((t) => t - sunShift),
          colorA: slotColors.a,
          colorB: slotColors.b,
        },
      },
    };
  }, [data, timeAxis, sunShift, colors, slotColors, resolvedTheme]);

  return (
    <div className="h-full w-full">
      <Line data={chartData} options={options} />
    </div>
  );
}
