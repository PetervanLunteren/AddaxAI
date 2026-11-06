/**
 * One label's calibration curve: the chance the AI label is right per
 * score, with its 95% band. Drawn only over the answered score range;
 * the backend never extrapolates and neither does this.
 */

import { useMemo } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  Filler,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  type ChartData,
  type ChartOptions,
} from "chart.js";

import type { CalibrationLabel } from "../../api/calibration";
import { useChartColors } from "../../lib/theme";

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Filler);

const PCT = (v: number) => `${Math.round(v * 100)}%`;

export function CalibrationCurve({ row }: { row: CalibrationLabel }) {
  const colors = useChartColors();
  const curve = useMemo(() => row.curve ?? [], [row.curve]);

  const data: ChartData<"line"> = useMemo(
    () => ({
      datasets: [
        {
          label: "Upper",
          data: curve.map((pt) => ({ x: pt.score, y: pt.upper })),
          borderColor: "transparent",
          pointRadius: 0,
        },
        {
          // Filled down to the dataset above, so the band sits between them.
          label: "Lower",
          data: curve.map((pt) => ({ x: pt.score, y: pt.lower })),
          borderColor: "transparent",
          backgroundColor: colors.withAlpha(colors.primaryInk, 0.15),
          fill: "-1",
          pointRadius: 0,
        },
        {
          label: "Right",
          data: curve.map((pt) => ({ x: pt.score, y: pt.p })),
          borderColor: colors.primaryInk,
          borderWidth: 2,
          pointRadius: 0,
        },
      ],
    }),
    [curve, colors],
  );

  const options: ChartOptions<"line"> = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: "index",
          intersect: false,
          filter: (item) => item.dataset.label === "Right",
          callbacks: {
            title: (items) => `Score ${Number(items[0].parsed.x).toFixed(3)}`,
            label: (item) => {
              const pt = curve[item.dataIndex];
              return `${PCT(pt.p)} right (95% band ${PCT(pt.lower)} to ${PCT(pt.upper)})`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "AI score", color: colors.axis },
          ticks: { color: colors.axis },
          grid: { color: colors.grid },
        },
        y: {
          min: 0,
          max: 1,
          title: { display: true, text: "Labels right", color: colors.axis },
          ticks: { color: colors.axis, callback: (v) => PCT(Number(v)) },
          grid: { color: colors.grid },
        },
      },
    }),
    [curve, colors],
  );

  return (
    <div className="h-56">
      <Line data={data} options={options} />
    </div>
  );
}
