/**
 * Deployment timeline chart — row-per-site Gantt + concurrent-cameras strip.
 *
 * One `<svg>` hosts three regions stacked vertically:
 *   - Top: x-axis with month / year ticks.
 *   - Middle: one row per site, outer light-teal bar per deployment,
 *             inner primary-teal bars per trap-night interval.
 *   - Bottom: step-function area chart of concurrent active cameras.
 *
 * Dates arrive as `YYYY-MM-DD` strings. They are parsed via
 * `Date.UTC(...)` to stay tz-agnostic (same reasoning as frontend/src/lib/datetime.ts).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";

import type {
  TimelineDeployment,
  TimelineResponse,
  TimelineSite,
  TrapNightInterval,
} from "../../api/timeline";
import { NO_SITE_SENTINEL } from "../../lib/filter-url";

const BAR_FILL = "#0f6064";
const CONCURRENT_FILL = "rgba(15, 96, 100, 0.18)";
const CONCURRENT_STROKE = "#0f6064";
const GRID_STROKE = "rgba(0, 0, 0, 0.06)";
const CONNECTOR_STROKE = "rgba(100, 116, 139, 0.55)";

type Density = "normal" | "compact";

interface DensityConfig {
  /** Height per track (one parallel subfolder = one track).
   *  Clean deployments use exactly one track per site row. */
  trackHeight: number;
  /** Vertical gap between site rows (not between tracks inside a row). */
  rowGap: number;
  barHeight: number;
  labelWidth: number;
  showLabels: boolean;
  barRadius: number;
}

const DENSITY: Record<Density, DensityConfig> = {
  normal: {
    trackHeight: 22,
    rowGap: 4,
    barHeight: 12,
    labelWidth: 160,
    showLabels: true,
    barRadius: 2,
  },
  compact: {
    trackHeight: 6,
    rowGap: 1,
    barHeight: 4,
    labelWidth: 24,
    showLabels: false,
    barRadius: 0,
  },
};

const CONCURRENT_HEIGHT = 70;
const AXIS_HEIGHT = 28;
const RIGHT_PADDING = 16;
const TOP_PADDING = 4;
const SECTION_GAP = 14;
const CONCURRENT_Y_LABEL_WIDTH = 28;

/** Pixel margin between the vertical branch connector and the bar's edge.
 *  Keeps the "split" and "merge" points visibly offset from the bar, so the
 *  branch reads as a clear L-shape rather than sitting flush against the bar. */
const BRANCH_MARGIN = 5;

const MS_PER_DAY = 86_400_000;

function parseDate(s: string): number {
  // s is YYYY-MM-DD. Date.UTC gives tz-agnostic ms.
  const [y, m, d] = s.split("-").map(Number);
  return Date.UTC(y, m - 1, d);
}

function formatYMD(ms: number): string {
  const d = new Date(ms);
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function formatDateLabel(s: string): string {
  // s is "YYYY-MM-DD". Render in the browser's locale with
  // day-short-month-year ordering, UTC-anchored so the date doesn't
  // shift by the viewer's timezone. Example outputs:
  //   en-GB  → "4 Apr 2024"
  //   en-US  → "Apr 4, 2024"
  //   nl-NL  → "4 apr. 2024"
  const [y, m, d] = s.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

interface MonthTick {
  ms: number;
  label: string;
  major: boolean;
}

function generateMonthTicks(xMinMs: number, xMaxMs: number): MonthTick[] {
  const ticks: MonthTick[] = [];
  const start = new Date(xMinMs);
  let y = start.getUTCFullYear();
  let m = start.getUTCMonth();
  if (start.getUTCDate() > 1) m += 1;
  while (true) {
    const tickMs = Date.UTC(y, m, 1);
    if (tickMs > xMaxMs) break;
    const monthName = new Date(tickMs).toLocaleString("en", {
      month: "short",
      timeZone: "UTC",
    });
    const major = m === 0;
    ticks.push({
      ms: tickMs,
      label: major ? `${monthName} ${y}` : monthName,
      major,
    });
    m += 1;
    if (m > 11) {
      m = 0;
      y += 1;
    }
  }
  return ticks;
}

/**
 * Greedy interval-graph colouring: returns a track index per interval
 * such that intervals sharing a track never overlap. The minimum number
 * of tracks equals the largest set of intervals active at one time,
 * which is exactly "how many parallel subfolders need to stack" for
 * this deployment. Disjoint sequential intervals collapse onto track 0.
 *
 * Input intervals come from the backend sorted by start; we preserve
 * their original order in the output so the caller can zip back.
 */
function assignTracks(
  intervals: TrapNightInterval[],
): { trackByIndex: number[]; trackCount: number } {
  const lastEnds: string[] = []; // per track, the latest assigned end (YMD)
  const trackByIndex = new Array<number>(intervals.length);
  const sortedIdx = intervals
    .map((_, i) => i)
    .sort((a, b) => intervals[a].start.localeCompare(intervals[b].start));
  for (const i of sortedIdx) {
    const iv = intervals[i];
    let placed = -1;
    for (let t = 0; t < lastEnds.length; t++) {
      if (lastEnds[t] < iv.start) {
        lastEnds[t] = iv.end;
        placed = t;
        break;
      }
    }
    if (placed < 0) {
      lastEnds.push(iv.end);
      placed = lastEnds.length - 1;
    }
    trackByIndex[i] = placed;
  }
  return { trackByIndex, trackCount: lastEnds.length };
}

/** Decide which ticks to label given available pixel width. */
function thinLabels(ticks: MonthTick[], plotWidth: number): Set<number> {
  const MIN_PX = 56;
  const span = ticks.length > 1 ? ticks.length : 1;
  const spacingPx = plotWidth / span;
  const keep = new Set<number>();
  if (ticks.length === 0) return keep;
  let step = 1;
  while (spacingPx * step < MIN_PX) step += 1;
  for (let i = 0; i < ticks.length; i += step) keep.add(i);
  // Always keep majors (January) if we can.
  ticks.forEach((t, i) => {
    if (t.major) keep.add(i);
  });
  return keep;
}

interface DeploymentTimelineChartProps {
  data: TimelineResponse | undefined;
  loading: boolean;
  projectId: string;
  density?: Density;
  /** Called with YYYY-MM-DD strings when the user finishes a
   *  drag-to-zoom on the date axis. Parent page is expected to
   *  write these into the dateFrom / dateTo filter so the timeline
   *  refetches at the narrower range. */
  onZoom?: (from: string, to: string) => void;
}

/** Minimum drag distance in pixels before we treat a click on the
 *  axis as a zoom intent. Below this we ignore the gesture so a
 *  stray click doesn't accidentally clobber the filters. */
const ZOOM_DRAG_THRESHOLD_PX = 4;

export function DeploymentTimelineChart({
  data,
  loading,
  projectId,
  density = "normal",
  onZoom,
}: DeploymentTimelineChartProps) {
  const densityConfig = DENSITY[density];
  const navigate = useNavigate();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(960);
  const [hover, setHover] = useState<{
    x: number;
    dateMs: number;
    count: number;
  } | null>(null);
  const [drag, setDrag] = useState<{ startX: number; currentX: number } | null>(
    null,
  );

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = Math.max(320, Math.floor(entry.contentRect.width));
        setWidth(w);
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const geometry = useMemo(() => {
    if (!data || data.sites.length === 0) return null;
    const xMin = data.date_range_from ? parseDate(data.date_range_from) : null;
    const xMax = data.date_range_to ? parseDate(data.date_range_to) : null;
    if (xMin === null || xMax === null || xMax <= xMin) return null;

    const plotLeft = densityConfig.labelWidth;
    const plotRight = width - RIGHT_PADDING - CONCURRENT_Y_LABEL_WIDTH;
    const plotWidth = Math.max(100, plotRight - plotLeft);

    // Per-site geometry: row height = max parallel-subfolder count across the
    // site's deployments, using greedy track assignment per deployment so
    // disjoint-sequential intervals collapse to one track. No cap: if a
    // deployment bundles 20 cameras, the row grows to accommodate all 20.
    const axisY = TOP_PADDING + AXIS_HEIGHT;
    interface SiteGeom {
      trackCount: number;
      rowTop: number;
      rowHeight: number;
      depTracks: Map<string, { trackByIndex: number[]; trackCount: number }>;
    }
    const siteGeoms: SiteGeom[] = [];
    let cursorY = axisY;
    for (const site of data.sites) {
      const depTracks = new Map<
        string,
        { trackByIndex: number[]; trackCount: number }
      >();
      let maxDepTrackCount = 1;
      for (const dep of site.deployments) {
        const assigned = assignTracks(dep.intervals);
        depTracks.set(dep.deployment_id, assigned);
        maxDepTrackCount = Math.max(maxDepTrackCount, assigned.trackCount);
      }
      const trackCount = maxDepTrackCount;
      const rowHeight = trackCount * densityConfig.trackHeight;
      siteGeoms.push({
        trackCount,
        rowTop: cursorY,
        rowHeight,
        depTracks,
      });
      cursorY += rowHeight + densityConfig.rowGap;
    }
    const ganttHeight = cursorY - axisY;
    const totalHeight =
      TOP_PADDING
      + AXIS_HEIGHT
      + ganttHeight
      + SECTION_GAP
      + CONCURRENT_HEIGHT
      + 18;

    const ganttBottom = axisY + ganttHeight;
    const concurrentTop = ganttBottom + SECTION_GAP;
    const concurrentBottom = concurrentTop + CONCURRENT_HEIGHT;

    const dateToX = (ms: number) =>
      plotLeft + ((ms - xMin) / (xMax - xMin)) * plotWidth;
    const yMsToX = (ms: number) => {
      if (ms <= xMin) return plotLeft;
      if (ms >= xMax) return plotRight;
      return dateToX(ms);
    };

    const monthTicks = generateMonthTicks(xMin, xMax);
    const labelledIdx = thinLabels(monthTicks, plotWidth);

    return {
      xMin,
      xMax,
      plotLeft,
      plotRight,
      plotWidth,
      siteGeoms,
      totalHeight,
      axisY,
      ganttBottom,
      concurrentTop,
      concurrentBottom,
      dateToX,
      yMsToX,
      monthTicks,
      labelledIdx,
    };
  }, [data, width, density, densityConfig]);

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center rounded-lg border bg-card p-16">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">
          Computing timeline...
        </span>
      </div>
    );
  }

  if (!data || data.sites.length === 0) {
    return (
      <div className="rounded-lg border bg-card p-8 text-center space-y-2">
        <div className="text-sm font-medium text-foreground">
          No deployments to show
        </div>
        <div className="text-sm text-muted-foreground max-w-xl mx-auto">
          Analyse a folder on the Analyses page to populate the timeline.
          Deployments appear here once their files have been processed.
        </div>
      </div>
    );
  }

  if (!geometry) {
    return (
      <div className="rounded-lg border bg-card p-8 text-center text-sm text-muted-foreground">
        Cannot render the timeline: deployments have no date range yet.
      </div>
    );
  }

  const maxConc = data.metrics.max_concurrent_cameras;
  const concurrentY = (count: number) => {
    if (maxConc <= 0) return geometry.concurrentBottom;
    const frac = count / maxConc;
    return geometry.concurrentBottom - frac * CONCURRENT_HEIGHT;
  };

  // Build step-function path for concurrent-cameras chart.
  const concurrentPathD = (() => {
    if (data.concurrent_cameras.length === 0) return "";
    const parts: string[] = [];
    let prevY = geometry.concurrentBottom;
    parts.push(`M ${geometry.plotLeft} ${prevY}`);
    for (const p of data.concurrent_cameras) {
      const x = geometry.yMsToX(parseDate(p.date));
      parts.push(`L ${x} ${prevY}`);
      prevY = concurrentY(p.count);
      parts.push(`L ${x} ${prevY}`);
    }
    parts.push(`L ${geometry.plotRight} ${prevY}`);
    return parts.join(" ");
  })();

  const concurrentFillD =
    concurrentPathD
      ? `${concurrentPathD} L ${geometry.plotRight} ${geometry.concurrentBottom} L ${geometry.plotLeft} ${geometry.concurrentBottom} Z`
      : "";

  const handleSiteClick = (site: TimelineSite) => {
    const param = site.site_id ?? NO_SITE_SENTINEL;
    navigate(`/projects/${projectId}/deployments?site_ids=${param}`);
  };

  const handleDeploymentClick = (dep: TimelineDeployment) => {
    navigate(`/projects/${projectId}/deployments?info=${dep.deployment_id}`);
  };

  return (
    <div ref={containerRef} className="w-full">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 pb-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span
            aria-hidden
            className="inline-block h-3 w-4 rounded-sm"
            style={{ backgroundColor: BAR_FILL }}
          />
          Trap nights
        </span>
      </div>
      <svg
        width={width}
        height={geometry.totalHeight}
        role="img"
        aria-label="Deployment timeline"
      >
        {/* Month gridlines across both Gantt and concurrent regions */}
        {geometry.monthTicks.map((tick, i) => {
          const x = geometry.dateToX(tick.ms);
          return (
            <line
              key={`grid-${i}`}
              x1={x}
              x2={x}
              y1={geometry.axisY}
              y2={geometry.concurrentBottom}
              stroke={GRID_STROKE}
              strokeWidth={1}
              shapeRendering="crispEdges"
            />
          );
        })}

        {/* Axis labels */}
        {geometry.monthTicks.map((tick, i) => {
          if (!geometry.labelledIdx.has(i)) return null;
          const x = geometry.dateToX(tick.ms);
          return (
            <text
              key={`label-${i}`}
              x={x}
              y={TOP_PADDING + 14}
              fontSize={11}
              fill="var(--muted-foreground, #64748b)"
              textAnchor="middle"
            >
              {tick.label}
            </text>
          );
        })}

        {/* Axis baseline */}
        <line
          x1={geometry.plotLeft}
          x2={geometry.plotRight}
          y1={geometry.axisY}
          y2={geometry.axisY}
          stroke={GRID_STROKE}
          shapeRendering="crispEdges"
        />

        {/* Drag-to-zoom overlay on the axis strip. Sits above the
            month labels and below the bars, so dragging across the
            top picks a new date range without blocking clicks on
            site / deployment bars below. */}
        {onZoom && (
          <rect
            x={geometry.plotLeft}
            y={TOP_PADDING}
            width={geometry.plotRight - geometry.plotLeft}
            height={AXIS_HEIGHT}
            fill="transparent"
            style={{ cursor: "col-resize" }}
            onMouseDown={(e) => {
              const bbox = (
                e.currentTarget as SVGRectElement
              ).getBoundingClientRect();
              const svgX = geometry.plotLeft + (e.clientX - bbox.left);
              setDrag({ startX: svgX, currentX: svgX });
            }}
            onMouseMove={(e) => {
              if (!drag) return;
              const bbox = (
                e.currentTarget as SVGRectElement
              ).getBoundingClientRect();
              const svgX = geometry.plotLeft + (e.clientX - bbox.left);
              const clamped = Math.max(
                geometry.plotLeft,
                Math.min(geometry.plotRight, svgX),
              );
              setDrag({ ...drag, currentX: clamped });
            }}
            onMouseUp={() => {
              if (!drag) return;
              const { startX, currentX } = drag;
              setDrag(null);
              if (Math.abs(currentX - startX) < ZOOM_DRAG_THRESHOLD_PX) return;
              const xToMs = (x: number): number => {
                const frac =
                  (x - geometry.plotLeft) / geometry.plotWidth;
                return geometry.xMin + frac * (geometry.xMax - geometry.xMin);
              };
              const a = xToMs(Math.min(startX, currentX));
              const b = xToMs(Math.max(startX, currentX));
              onZoom(formatYMD(a), formatYMD(b));
            }}
            onMouseLeave={() => {
              // Cancel a drag that left the strip; the user can
              // re-grab from inside if they want to retry.
              if (drag) setDrag(null);
            }}
          />
        )}

        {/* Drag selection band. Spans the full plot height so the
            user sees exactly which deployments fall in the range. */}
        {drag && drag.currentX !== drag.startX && (
          <rect
            x={Math.min(drag.startX, drag.currentX)}
            y={geometry.axisY}
            width={Math.abs(drag.currentX - drag.startX)}
            height={geometry.concurrentBottom - geometry.axisY}
            fill={BAR_FILL}
            opacity={0.12}
            pointerEvents="none"
          />
        )}

        {/* Gantt rows */}
        {data.sites.map((site, rowIdx) => {
          const siteGeom = geometry.siteGeoms[rowIdx];
          const { rowTop, rowHeight, depTracks } = siteGeom;
          const rowCenterY = rowTop + rowHeight / 2;
          // Per-deployment local-track → y coordinate: position tracks
          // symmetrically around the row centre, spaced by trackHeight.
          // A deployment with trackCount=1 places its bar on the centre.
          const depTrackCenterY = (depTrackCount: number, localTrack: number) =>
            rowCenterY
            + (localTrack - (depTrackCount - 1) / 2) * densityConfig.trackHeight;
          return (
            <g key={site.site_id ?? "no-site"}>
              {densityConfig.showLabels && (
                <>
                  <text
                    x={densityConfig.labelWidth - 12}
                    y={rowCenterY + 4}
                    fontSize={12}
                    fill="var(--foreground, #0f172a)"
                    textAnchor="end"
                    style={{ cursor: "pointer" }}
                    onClick={() => handleSiteClick(site)}
                  >
                    <title>
                      {site.site_name} — click to open Deployments filtered to
                      this site
                    </title>
                    {site.site_name}
                  </text>
                </>
              )}

              {/* Horizontal spine at row centre, label → plot end. This is
                  the "main line" the user referred to. Single-interval
                  deployments sit on it; parallel-interval deployments fan
                  off it via per-interval vertical connectors drawn below. */}
              <line
                x1={densityConfig.labelWidth - 10}
                x2={geometry.plotRight}
                y1={rowCenterY}
                y2={rowCenterY}
                stroke={CONNECTOR_STROKE}
                strokeWidth={1}
                shapeRendering="crispEdges"
              />

              {/* Branch connectors: for every interval whose track sits off
                  the spine, draw an L-shape on each side — vertical split
                  BRANCH_MARGIN px before the bar, short horizontal feeler
                  into the bar, then mirror after the bar. This gives the
                  branch a visible offset from the bar instead of sitting
                  flush against it. */}
              {site.deployments.map((dep) => {
                const depInfo = depTracks.get(dep.deployment_id);
                if (!depInfo || depInfo.trackCount <= 1) return null;
                return (
                  <g key={`conn-${dep.deployment_id}`}>
                    {dep.intervals.map((iv, j) => {
                      const t = depInfo.trackByIndex[j];
                      const y = depTrackCenterY(depInfo.trackCount, t);
                      if (y === rowCenterY) return null;
                      const barStart = geometry.yMsToX(parseDate(iv.start));
                      const barEnd = geometry.yMsToX(
                        parseDate(iv.end) + MS_PER_DAY,
                      );
                      const splitX = barStart - BRANCH_MARGIN;
                      const mergeX = barEnd + BRANCH_MARGIN;
                      return (
                        <g key={`conn-${dep.deployment_id}-${j}`}>
                          {/* split: vertical from spine to track */}
                          <line
                            x1={splitX}
                            x2={splitX}
                            y1={rowCenterY}
                            y2={y}
                            stroke={CONNECTOR_STROKE}
                            strokeWidth={1}
                            shapeRendering="crispEdges"
                          />
                          {/* feeler into the bar */}
                          <line
                            x1={splitX}
                            x2={barStart}
                            y1={y}
                            y2={y}
                            stroke={CONNECTOR_STROKE}
                            strokeWidth={1}
                            shapeRendering="crispEdges"
                          />
                          {/* feeler out of the bar */}
                          <line
                            x1={barEnd}
                            x2={mergeX}
                            y1={y}
                            y2={y}
                            stroke={CONNECTOR_STROKE}
                            strokeWidth={1}
                            shapeRendering="crispEdges"
                          />
                          {/* merge: vertical from track back to spine */}
                          <line
                            x1={mergeX}
                            x2={mergeX}
                            y1={rowCenterY}
                            y2={y}
                            stroke={CONNECTOR_STROKE}
                            strokeWidth={1}
                            shapeRendering="crispEdges"
                          />
                        </g>
                      );
                    })}
                  </g>
                );
              })}

              {/* Bars. Each interval sits on its assigned track y. Single-
                  interval deployments stay on rowCenterY (spine). */}
              {site.deployments.map((dep) => {
                const depInfo = depTracks.get(dep.deployment_id);
                const cameraLine = dep.camera_model
                  ? `\nCamera: ${dep.camera_model}`
                  : "";
                return (
                  <g key={dep.deployment_id}>
                    {dep.intervals.map((iv, j) => {
                      const t = depInfo ? depInfo.trackByIndex[j] : 0;
                      const depTrackCount = depInfo ? depInfo.trackCount : 1;
                      const trackY = depTrackCenterY(depTrackCount, t);
                      const barY = trackY - densityConfig.barHeight / 2;
                      const ivStart = geometry.yMsToX(parseDate(iv.start));
                      const ivEnd = geometry.yMsToX(
                        parseDate(iv.end) + MS_PER_DAY,
                      );
                      const ivW = Math.max(1, ivEnd - ivStart);
                      return (
                        <rect
                          key={`${dep.deployment_id}-${j}`}
                          x={ivStart}
                          y={barY}
                          width={ivW}
                          height={densityConfig.barHeight}
                          fill={BAR_FILL}
                          rx={densityConfig.barRadius}
                          ry={densityConfig.barRadius}
                          style={{ cursor: "pointer" }}
                          onClick={() => handleDeploymentClick(dep)}
                        >
                          <title>
                            {site.site_name} — {dep.deployment_label}
                            {cameraLine}
                            {"\n"}Period:{" "}
                            {formatDateLabel(iv.start)} →{" "}
                            {formatDateLabel(iv.end)}
                            {"\n"}Trap-nights: {iv.trap_nights}
                            {"\n"}Files: {dep.file_count.toLocaleString()}
                          </title>
                        </rect>
                      );
                    })}
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* Concurrent-cameras fill + line */}
        {concurrentFillD && (
          <path
            d={concurrentFillD}
            fill={CONCURRENT_FILL}
            stroke="none"
          />
        )}
        {concurrentPathD && (
          <path
            d={concurrentPathD}
            fill="none"
            stroke={CONCURRENT_STROKE}
            strokeWidth={1.5}
          />
        )}
        <line
          x1={geometry.plotLeft}
          x2={geometry.plotRight}
          y1={geometry.concurrentBottom}
          y2={geometry.concurrentBottom}
          stroke={GRID_STROKE}
          shapeRendering="crispEdges"
        />
        {maxConc > 0 && (
          <>
            <line
              x1={geometry.plotLeft}
              x2={geometry.plotRight}
              y1={concurrentY(maxConc)}
              y2={concurrentY(maxConc)}
              stroke={GRID_STROKE}
              strokeDasharray="2 2"
              shapeRendering="crispEdges"
            />
            <text
              x={geometry.plotRight + 4}
              y={concurrentY(maxConc) + 4}
              fontSize={11}
              fill="var(--muted-foreground, #64748b)"
            >
              {maxConc}
            </text>
            <text
              x={geometry.plotRight + 4}
              y={geometry.concurrentBottom + 4}
              fontSize={11}
              fill="var(--muted-foreground, #64748b)"
            >
              0
            </text>
          </>
        )}
        {/* Hover overlay + crosshair for the concurrent strip */}
        <rect
          x={geometry.plotLeft}
          y={geometry.concurrentTop}
          width={geometry.plotRight - geometry.plotLeft}
          height={CONCURRENT_HEIGHT}
          fill="transparent"
          style={{ cursor: "crosshair" }}
          onMouseMove={(e) => {
            const bbox = (e.currentTarget as SVGRectElement).getBoundingClientRect();
            const offsetX = e.clientX - bbox.left;
            const svgX = geometry.plotLeft + offsetX;
            const ms =
              geometry.xMin
              + (offsetX / geometry.plotWidth) * (geometry.xMax - geometry.xMin);
            let count = 0;
            for (const p of data.concurrent_cameras) {
              if (parseDate(p.date) > ms) break;
              count = p.count;
            }
            setHover({ x: svgX, dateMs: ms, count });
          }}
          onMouseLeave={() => setHover(null)}
        />
        {hover && (() => {
          const y = concurrentY(hover.count);
          const label = `${formatDateLabel(formatYMD(hover.dateMs))} — ${hover.count} camera${hover.count === 1 ? "" : "s"}`;
          const labelW = Math.max(120, label.length * 6.5);
          const labelH = 20;
          const labelY = y - 8 - labelH < geometry.concurrentTop
            ? y + 8
            : y - 8 - labelH;
          const labelX = Math.max(
            geometry.plotLeft,
            Math.min(geometry.plotRight - labelW, hover.x - labelW / 2),
          );
          return (
            <g pointerEvents="none">
              <line
                x1={hover.x}
                x2={hover.x}
                y1={geometry.concurrentTop}
                y2={geometry.concurrentBottom}
                stroke={CONNECTOR_STROKE}
                strokeWidth={1}
                shapeRendering="crispEdges"
              />
              <circle
                cx={hover.x}
                cy={y}
                r={3}
                fill={CONCURRENT_STROKE}
              />
              <rect
                x={labelX}
                y={labelY}
                width={labelW}
                height={labelH}
                rx={3}
                ry={3}
                fill="white"
                stroke={CONNECTOR_STROKE}
              />
              <text
                x={labelX + labelW / 2}
                y={labelY + 14}
                fontSize={11}
                fill="var(--foreground, #0f172a)"
                textAnchor="middle"
              >
                {label}
              </text>
            </g>
          );
        })()}

        {densityConfig.showLabels && (
          <text
            x={densityConfig.labelWidth - 8}
            y={concurrentY(maxConc) + 4}
            fontSize={11}
            fill="var(--muted-foreground, #64748b)"
            textAnchor="end"
          >
            Concurrent cameras
          </text>
        )}

      </svg>
    </div>
  );
}
