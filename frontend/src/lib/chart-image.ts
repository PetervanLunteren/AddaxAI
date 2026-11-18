/**
 * "Image (PNG)" on the Insights pages that draw one Chart.js chart: the
 * chart as it is on screen, saved as a PNG.
 *
 * A chart canvas is transparent, so pasted into a document it would show
 * whatever is behind it. The image is therefore the chart on a filled
 * background: the caller passes the card colour (`useChartColors().card`),
 * which makes the file look like the screen, in light and in dark theme.
 * It has the canvas's own pixel size, so it is sharp on a Retina screen.
 */

import { useCallback, useState } from "react";

import { downloadBlob } from "./download";

/** All this needs of a Chart.js chart: the canvas it drew on. */
export interface DrawnChart {
  canvas: HTMLCanvasElement;
}

/** The ref a chart component passes to react-chartjs-2's `<Line ref>`. */
export type DrawnChartRef = (chart: DrawnChart | null | undefined) => void;

/**
 * The chart a page currently draws, or null while it draws none (loading,
 * an empty state), and the ref to hand the chart component. A page offers
 * "Image (PNG)" only while there is a chart. The ref is stable, so React
 * does not detach and reattach it on every render.
 */
export function useDrawnChart(): [DrawnChart | null, DrawnChartRef] {
  const [chart, setChart] = useState<DrawnChart | null>(null);
  const ref = useCallback<DrawnChartRef>((drawn) => setChart(drawn ?? null), []);
  return [chart, ref];
}

export function saveChartPng(chart: DrawnChart, background: string, fileName: string): void {
  const source = chart.canvas;
  const image = document.createElement("canvas");
  image.width = source.width;
  image.height = source.height;
  const ctx = image.getContext("2d");
  if (!ctx) throw new Error("No 2D canvas context to draw the image on");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, image.width, image.height);
  ctx.drawImage(source, 0, 0);
  image.toBlob((blob) => {
    if (!blob) throw new Error("The chart could not be written as a PNG");
    downloadBlob(blob, fileName);
  }, "image/png");
}
