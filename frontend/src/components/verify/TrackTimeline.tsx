/**
 * The tracks of a clip over its length, under the video player.
 *
 * One group per species, and inside a group as many rows as its animals
 * need to sit beside each other (`packTrackRows`). One bar per animal in
 * the species colour, faint while the animal's card is unverified and
 * solid once it is; a rejected animal is a verified one on the "False
 * detection" row after the species, in that class's grey. A bar has a
 * hole where the detector missed the animal on a sample (`trackGaps`):
 * the tracker carries a track through up to two seconds without a box,
 * the still and the player show nothing of the animal there, and until
 * 2026-11-03 the bar alone said it was present, so forty bars crossed
 * the playhead on a frame that drew 33 boxes. The hole is a thin line
 * through the bar, a second layer of its own background
 * (`barBackground`), one element per animal still, which keeps the bar
 * one thing to click. A box is drawn
 * half a step either side of its frame, so the playhead on a sampled
 * frame sits in the middle of colour or of a hole, never on an edge. The species
 * is named on the group's first row, with a MaxN button beside it when
 * the row has a marker: the frame its MaxN was counted on, which the
 * Counts modal takes from the event's rows and the Files viewer from
 * the file itself (`fileMaxNMarkers`). A playhead follows the player.
 * Click a bar to select that animal and jump to it; hover the ruler
 * above the rows for a moment's time and frame and click to go there
 * (`TimeRuler`), or click the lane elsewhere. Drag the top edge to make the lane taller;
 * rows past the height scroll, the height is remembered per page
 * (`TimelineSurface`: the two modals lay the clip out differently), and
 * selecting an animal scrolls its row into view. The verdict shading is
 * the Labels viewer's; the Counts page shows every bar solid.
 *
 * **Zoom.** Each slider step halves the visible span, and the window
 * stays centred on the playhead, clamped at the clip's ends. So the
 * playhead is the scroll position: clicking in the lane moves it,
 * playback scrolls the window along, and a sideways swipe or a
 * shift-scroll over the lane pans by moving it. No second scroll
 * position to keep in step. The pan is previewed here and handed to
 * the player once the gesture settles, because in frame mode a seek
 * decodes a frame from the clip and a swipe is dozens of wheel ticks.
 * The slider is absent when there is nothing to zoom into, a clip of
 * `MIN_WINDOW_SECONDS` or less, so a one-frame video still renders the
 * single row it always did. It exists because packing alone does not
 * make a long clip clickable: on an 81 minute drop the median track is
 * one pixel wide and the busiest 90 seconds is 16 pixels of lane
 * holding 39 animals.
 *
 * **Look.** On the app's own light surface, under the dark picture, as
 * annotation tools lay it out: the species palette is made for light
 * backgrounds, and on near-black the dark ones (fish is a dark red) at
 * 40% were close to invisible (2026-11-05). An unchecked bar is its
 * colour at half strength, a checked one solid. Not an outline: drawn
 * round the whole bar it would bridge the gaps the bar breaks at.
 *
 * **Activity.** The ruler is also a graph of the most boxes at once in
 * each slice of the window (`activityPeaks`), the clip-wide peak as its
 * full height, so a busy moment stands out even where its bars are a
 * pixel wide. The same idea as DIVE's detection count timeline and
 * YouTube's "most replayed" graph above the seek bar: one quiet graph
 * on the strip a person already clicks, not a panel of its own. All
 * species with nothing selected; with an animal selected, its species
 * alone, in its colour, so the tallest peak is that species' MaxN
 * marker: MaxN is counted per species, and in a total a rare species
 * vanishes under a common one. Not a stacked graph per species: in a
 * strip this thin a few colours are noise, and the peak of all species
 * together is not the sum of each one's peak.
 *
 * Rendering only: which track is selected and where the playhead is
 * belong to the player and its modal. No keyboard handling here.
 */

import { memo, useEffect, useMemo, useRef, useState } from "react";
import { ZoomIn, ZoomOut } from "lucide-react";

import { useSpeciesColorsVersion } from "../../utils/species-colors";
import { cn } from "../../lib/utils";
import type { TrackRow } from "../../lib/track-utils";
import {
  activityPeaks,
  barBackground,
  clipDurationFrames,
  fileSamplingStep,
  frameBoxCounts,
  packTrackRows,
  snapToSample,
  trackCards,
  trackRows,
} from "../../lib/track-utils";
import { formatClipPosition } from "../../lib/datetime";
import { useResizableHeight } from "../../hooks/useResizableHeight";
import { ResizeHandle } from "../ui/resize-handle";
import type { FileWithDetections } from "../../api/types";

/**
 * Where the timeline is shown. The two pages lay the clip out
 * differently, so each keeps its own dragged height, and only the
 * Labels viewer shades bars by verdict: checking is its work, while the
 * Counts page has no verdict buttons and shows every bar solid.
 */
export type TimelineSurface = "labels" | "counts";

export interface TimelineMarker {
  frame: number;
  /** The species key, as `speciesKey` spells it. */
  label: string;
  /** The count made on that frame, for the button's tooltip. */
  count?: number;
}

interface TrackTimelineProps {
  rows: TrackRow[];
  /** Frames in the whole clip; the bars are fractions of it. */
  durationFrames: number;
  currentFrame: number;
  selectedTrackId: string | null;
  markers?: TimelineMarker[];
  /** The clip's frame rate, for the zoom floor and the span text. */
  frameRate?: number | null;
  /** Frames between two sampled frames, for the ruler's snap and label;
   *  0 when unknown, and the ruler then names the exact frame. */
  sampleStep?: number;
  /** Boxes drawn per frame (`frameBoxCounts`), for the ruler's activity
   *  graph; without it the ruler is a plain strip. */
  frameCounts?: Map<number, number>;
  /** Whose boxes `frameCounts` holds: a species' name and colour, or
   *  undefined for all of them. */
  activitySpecies?: { name: string; color: string };
  surface?: TimelineSurface;
  /** The zoom, held by the modal when given: the timeline is mounted
   *  afresh when a clip switches between its still and the player, and
   *  the zoom must survive that, as the selection does. */
  zoomStep?: number;
  onZoomStep?: (step: number) => void;
  onSelectTrack: (trackId: string) => void;
  onSeek: (frame: number) => void;
}

const ROW_HEIGHT_PX = 16;
/** The ruler above the rows: hover for a moment's time and frame,
 *  click to go there. Sticky inside the scrolling lane, so it keeps the
 *  lane's own width (the scrollbar sits outside it, as for the
 *  playhead) and stays while the rows scroll. */
const RULER_HEIGHT_PX = 24;
/** Slices of the window the activity graph is drawn in: about two
 *  pixels each on a typical lane, fine enough to show a lone spike. */
const ACTIVITY_BINS = 400;
/** The smallest lane a drag can leave: the ruler and two rows. */
const MIN_LANE_PX = RULER_HEIGHT_PX + 2 * ROW_HEIGHT_PX;
/** The lane's height until someone drags it: six rows, the old cap. */
const DEFAULT_ROWS = 6;
/** The species column, named once for the rows and the playhead. */
const LABEL_COLUMN_REM = 10;
/** The most the slider zooms in. Below this a step stops being a useful
 *  amount of footage to look at. */
const MIN_WINDOW_SECONDS = 5;
/** A one-frame track is still worth a mark at the widest zoom. */
const MIN_BAR_PX = 2;
/** How much of its colour an unchecked bar keeps, on the light lane. */
const UNVERIFIED_FILL_PERCENT = 50;
/** How long after the last wheel tick a pan is handed to the player. */
const PAN_COMMIT_MS = 150;

const clamp = (value: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, value));
const halfTheViewport = () => window.innerHeight * 0.5;

export function TrackTimeline({
  rows,
  durationFrames,
  currentFrame,
  selectedTrackId,
  markers = [],
  frameRate,
  sampleStep = 0,
  frameCounts,
  activitySpecies,
  surface = "labels",
  zoomStep: zoomStepProp,
  onZoomStep,
  onSelectTrack,
  onSeek,
}: TrackTimelineProps) {
  useSpeciesColorsVersion();
  const [ownZoomStep, setOwnZoomStep] = useState(0);
  const zoomStep = zoomStepProp ?? ownZoomStep;
  const setZoomStep = onZoomStep ?? setOwnZoomStep;
  const showVerdicts = surface === "labels";
  const laneRef = useRef<HTMLDivElement | null>(null);
  const { height, startDrag } = useResizableHeight({
    storageKey: `addaxai:timelineHeight:${surface}`,
    defaultHeight: RULER_HEIGHT_PX + DEFAULT_ROWS * ROW_HEIGHT_PX,
    minHeight: MIN_LANE_PX,
    maxHeight: halfTheViewport,
  });

  // A pan in progress: the playhead the wheel has moved to, previewed
  // here until the gesture settles and `onSeek` makes it the player's.
  // Tied to the frame it started from, so a frame set elsewhere in the
  // meantime (a bar click, playback) drops the preview.
  const [pan, setPan] = useState<{ base: number; frame: number } | null>(null);
  const shownFrame = pan && pan.base === currentFrame ? pan.frame : currentFrame;

  const groups = useMemo(() => packTrackRows(rows), [rows]);
  const totalRows = groups.reduce((n, group) => n + group.rows.length, 0);

  // Each step halves the span; the window is centred on the playhead and
  // sticks to the clip's ends, so it needs no scroll position of its own.
  const minWindow = MIN_WINDOW_SECONDS * (frameRate || 30);
  const maxZoomStep = Math.max(
    0,
    Math.ceil(Math.log2(Math.max(1, durationFrames / minWindow))),
  );
  const step = clamp(zoomStep, 0, maxZoomStep);
  const windowFrames = durationFrames / 2 ** step;
  const windowStart = clamp(
    shownFrame - windowFrames / 2,
    0,
    Math.max(0, durationFrames - windowFrames),
  );

  // What the wheel handler reads, refreshed after every render, so the
  // listener is bound once and never sees stale props.
  const live = useRef({ step, windowFrames, durationFrames, currentFrame, pan, onSeek });
  useEffect(() => {
    live.current = { step, windowFrames, durationFrames, currentFrame, pan, onSeek };
  });

  // Pan with the wheel. Native and non-passive, because React's own
  // onWheel is passive and preventDefault there does nothing: a
  // sideways swipe would become the browser's back gesture. Horizontal
  // deltas and shift-scroll pan; a plain vertical scroll keeps scrolling
  // the rows. Bound once per lane, re-bound when the lane appears.
  const commitTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    const lane = laneRef.current;
    if (!lane) return;
    const onWheel = (e: WheelEvent) => {
      const horizontal = Math.abs(e.deltaX) > Math.abs(e.deltaY);
      if (!horizontal && !e.shiftKey) return;
      e.preventDefault();
      const state = live.current;
      if (state.step === 0) return;
      // The bars' column: the scroll box minus the species column.
      const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      const laneWidth = Math.max(1, lane.clientWidth - LABEL_COLUMN_REM * rem);
      const delta = horizontal ? e.deltaX : e.deltaY;
      const from =
        state.pan && state.pan.base === state.currentFrame
          ? state.pan.frame
          : state.currentFrame;
      const frame = Math.round(
        clamp(from + (delta * state.windowFrames) / laneWidth, 0, state.durationFrames),
      );
      const next = { base: state.currentFrame, frame };
      // Written here too, so the next tick builds on this one before
      // React has re-rendered with it.
      live.current = { ...state, pan: next };
      setPan(next);
      if (commitTimer.current) clearTimeout(commitTimer.current);
      commitTimer.current = setTimeout(() => state.onSeek(frame), PAN_COMMIT_MS);
    };
    lane.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      lane.removeEventListener("wheel", onWheel);
      if (commitTimer.current) clearTimeout(commitTimer.current);
    };
  }, [laneRef, totalRows]);

  // Which row the selected animal sits on, so a selection made elsewhere
  // (a card in the grid) is never left below the fold.
  const selectedRow = useMemo(() => {
    if (!selectedTrackId) return -1;
    let index = 0;
    for (const group of groups) {
      for (const packed of group.rows) {
        if (packed.some((row) => row.track.id === selectedTrackId)) return index;
        index += 1;
      }
    }
    return -1;
  }, [groups, selectedTrackId]);

  useEffect(() => {
    const lane = laneRef.current;
    if (!lane || selectedRow < 0) return;
    // The sticky ruler covers the first RULER_HEIGHT_PX of the view.
    const top = RULER_HEIGHT_PX + selectedRow * ROW_HEIGHT_PX;
    if (top < lane.scrollTop + RULER_HEIGHT_PX) lane.scrollTop = top - RULER_HEIGHT_PX;
    else if (top + ROW_HEIGHT_PX > lane.scrollTop + lane.clientHeight) {
      lane.scrollTop = top + ROW_HEIGHT_PX - lane.clientHeight;
    }
  }, [laneRef, selectedRow]);

  if (totalRows === 0 || durationFrames <= 0) return null;

  const pct = (frame: number) => ((frame - windowStart) / windowFrames) * 100;
  const windowEnd = windowStart + windowFrames;
  const inWindow = (frame: number) => frame >= windowStart && frame <= windowEnd;

  const seekAt = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fraction = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    onSeek(Math.round(clamp(windowStart + fraction * windowFrames, 0, durationFrames)));
  };

  const spanText =
    step === 0
      ? "whole clip"
      : frameRate
        ? `showing ${formatClipPosition(windowFrames / frameRate)}`
        : `showing ${Math.round(windowFrames)} frames`;

  const maxNTitle = (name: string, m: TimelineMarker) =>
    `The most ${name} at once${m.count != null ? `: ${m.count}` : ""}, at ${
      frameRate ? formatClipPosition(m.frame / frameRate) : `frame ${m.frame}`
    }`;

  return (
    <div
      className="w-full shrink-0 bg-background text-[11px] text-foreground/80"
      data-testid="track-timeline"
    >
      {/* The top edge is always the same strip, so the timeline looks
          the same whether or not its rows fit. It is a handle only while
          some rows are hidden: the lane never grows past its rows, so on
          a clip whose rows all show a drag would move nothing. */}
      {RULER_HEIGHT_PX + totalRows * ROW_HEIGHT_PX > height ? (
        <ResizeHandle
          onMouseDown={startDrag}
          title="Drag to show more rows"
          testId="timeline-resize-handle"
        />
      ) : (
        <div className="h-1.5 shrink-0 bg-border" aria-hidden />
      )}
      <div
        ref={laneRef}
        // Horizontal is hidden, not auto: a bar at the very end of the
        // clip is held to MIN_BAR_PX and so reaches past the lane,
        // which otherwise grows a horizontal scrollbar and eats a row's
        // worth of height.
        // Relative so the playhead below is laid out against this box's
        // padding area, which the lane rows fill: the vertical scrollbar
        // the rows grow past `height` sits outside it. Positioned against
        // the surface instead, the playhead drifted right of the marker
        // and the bars by the scrollbar's width times the clip position.
        className="relative overflow-y-auto overflow-x-hidden"
        style={{ maxHeight: height }}
      >
        <div className="sticky top-0 z-10 flex items-stretch bg-background" style={{ height: RULER_HEIGHT_PX }}>
          {/* The zoom, in the label column above the species names: the
              control for the strip beside the strip, a slider between the
              two magnifiers every map and photo viewer uses, each a button
              for one step, never dimmed: at either end a click does nothing,
              and the slider's knob already says where the range ends. Absent
              when the clip is too short to zoom into.
              The span it shows is the tooltip, since the ruler names the
              time under the cursor. */}
          <div className="flex shrink-0 items-center gap-1 px-2" style={{ width: `${LABEL_COLUMN_REM}rem` }}>
            {maxZoomStep > 0 && (
              <>
                <button
                  type="button"
                  aria-label="Zoom out"
                  title="Zoom out"
                  onClick={() => setZoomStep(Math.max(0, step - 1))}
                  className="shrink-0 text-muted-foreground hover:text-foreground"
                >
                  <ZoomOut className="h-3 w-3" />
                </button>
                <input
                  type="range"
                  min={0}
                  max={maxZoomStep}
                  step={1}
                  value={step}
                  onChange={(e) => setZoomStep(Number(e.target.value))}
                  aria-label="Zoom the timeline"
                  title={spanText}
                  data-testid="timeline-zoom"
                  className="h-1 min-w-0 flex-1 cursor-pointer accent-primary"
                />
                <button
                  type="button"
                  aria-label="Zoom in"
                  title="Zoom in"
                  onClick={() => setZoomStep(Math.min(maxZoomStep, step + 1))}
                  className="shrink-0 text-muted-foreground hover:text-foreground"
                >
                  <ZoomIn className="h-3 w-3" />
                </button>
              </>
            )}
          </div>
          <TimeRuler
            windowStart={windowStart}
            windowFrames={windowFrames}
            durationFrames={durationFrames}
            frameRate={frameRate}
            sampleStep={sampleStep}
            frameCounts={frameCounts}
            activitySpecies={activitySpecies}
            onSeek={onSeek}
          />
        </div>
        {groups.map((group, groupIndex) => {
          const maxN = markers.find((m) => m.label === group.label);
          return group.rows.map((packed, rowIndex) => (
            <div
              key={`${group.label}-${rowIndex}`}
              // A line where one species' rows end and the next begin
              // (the swimlane divider), drawn inside the row so its
              // height is unchanged. None between the rows of one
              // species: those only split animals that overlapped.
              className={cn(
                "flex items-stretch",
                groupIndex > 0 && rowIndex === 0 && "border-t border-border",
              )}
              style={{ height: ROW_HEIGHT_PX }}
            >
              <div
                className="flex shrink-0 items-center gap-1 px-2"
                style={{ width: `${LABEL_COLUMN_REM}rem` }}
              >
                {rowIndex === 0 && (
                  <>
                    <span className="min-w-0 truncate leading-4" title={group.displayLabel}>
                      {group.displayLabel}
                    </span>
                    {maxN && (
                      <button
                        type="button"
                        data-testid="maxn-button"
                        title={maxNTitle(group.displayLabel, maxN)}
                        onClick={() => onSeek(maxN.frame)}
                        className={cn(
                          "shrink-0 rounded border border-border bg-background px-1",
                          "text-[9px] leading-[13px] text-foreground hover:border-foreground/50",
                          "hover:bg-muted focus-visible:outline-none",
                          "focus-visible:ring-2 focus-visible:ring-ring",
                        )}
                      >
                        MaxN
                      </button>
                    )}
                  </>
                )}
              </div>
              <div
                className="relative flex-1 cursor-pointer"
                onClick={seekAt}
                data-testid="track-lane"
              >
                {packed
                  .filter(
                    (row) =>
                      row.track.end_frame >= windowStart &&
                      row.track.start_frame <= windowEnd,
                  )
                  .map((row) => {
                    const selected = row.track.id === selectedTrackId;
                    // With nothing selected, the animals on the frame shown
                    // are what the frame verify acts on; outline their bars
                    // so the lane says "these N" while the still does. A
                    // bar has a box on the frame where the playhead sits on
                    // a run, not on a gap's thin line. The Labels viewer's
                    // alone, like the verdict shading, and never a rejected
                    // animal: neither is in the N the button counts.
                    const onFrame =
                      showVerdicts &&
                      row.verdict !== "rejected" &&
                      selectedTrackId == null &&
                      currentFrame >= row.startFrame &&
                      currentFrame < row.endFrame &&
                      !row.gaps.some(([a, b]) => a <= currentFrame && currentFrame < b);
                    // Cut to the window, and the holes with it.
                    const from = Math.max(row.startFrame, windowStart);
                    const to = Math.min(row.endFrame, windowEnd);
                    const left = pct(from);
                    return (
                      <button
                        key={row.track.id}
                        type="button"
                        // The verdict in words, so the first hover teaches
                        // what faint and solid mean; not where no verdict shows.
                        title={`${group.displayLabel}${showVerdicts ? `, ${row.verdict}` : ""}, frames ${row.track.start_frame} to ${row.track.end_frame}`}
                        data-testid="track-bar"
                        data-track-id={row.track.id}
                        data-verdict={row.verdict}
                        data-on-frame={onFrame || undefined}
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectTrack(row.track.id);
                        }}
                        className={cn(
                          "absolute top-[3px] h-[10px] rounded-sm",
                          selected && "ring-2 ring-foreground",
                          onFrame && "ring-1 ring-foreground/50",
                        )}
                        style={{
                          left: `${left}%`,
                          width: `${Math.max(0, pct(to) - left)}%`,
                          minWidth: MIN_BAR_PX,
                          ...barStyle(row, from, to, showVerdicts),
                        }}
                      />
                    );
                  })}
                {rowIndex === 0 && maxN && inWindow(maxN.frame) && (
                  <span
                    title={`MaxN at frame ${maxN.frame}`}
                    data-testid="maxn-marker"
                    className="pointer-events-none absolute -top-px h-0 w-0 -translate-x-1/2 border-x-[4px] border-t-[6px] border-x-transparent border-t-foreground"
                    style={{ left: `${pct(maxN.frame)}%` }}
                  />
                )}
              </div>
            </div>
          ));
        })}
        {/* Playhead over every row, in the lane's column. */}
        <div
          className="pointer-events-none absolute top-0 w-px bg-foreground"
          data-testid="playhead"
          style={{
            left: `calc(${LABEL_COLUMN_REM}rem + (100% - ${LABEL_COLUMN_REM}rem) * ${clamp((shownFrame - windowStart) / windowFrames, 0, 1)})`,
            height: RULER_HEIGHT_PX + totalRows * ROW_HEIGHT_PX,
          }}
        />
      </div>
    </div>
  );
}

/**
 * The strip above the rows where a person picks a moment: a hairline
 * and "12:34 · frame 37020" follow the cursor, a click seeks there. The
 * moment is the nearest sampled frame (`snapToSample`), so the label
 * names the frame the click will show and a bar under the playhead
 * means a box on screen. Its hover state lives here and nowhere else,
 * so moving the mouse re-renders this strip and not the lane's five
 * thousand bars. Until 2026-11-03 a moment could only be picked by
 * clicking empty lane, with no feedback, and a dense drop has no empty
 * lane: every pixel is a bar, and a bar click selects an animal.
 *
 * Behind the label, the activity graph (header): one SVG path, scaled to
 * the clip's busiest frame so zooming into a quiet stretch shows it
 * quiet rather than blowing it up to full height.
 */
function TimeRuler({
  windowStart,
  windowFrames,
  durationFrames,
  frameRate,
  sampleStep,
  frameCounts,
  activitySpecies,
  onSeek,
}: {
  windowStart: number;
  windowFrames: number;
  durationFrames: number;
  frameRate?: number | null;
  sampleStep: number;
  frameCounts?: Map<number, number>;
  activitySpecies?: { name: string; color: string };
  onSeek: (frame: number) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const clipPeak = useMemo(
    () => (frameCounts ? Math.max(0, ...frameCounts.values()) : 0),
    [frameCounts],
  );
  const activityPath = useMemo(() => {
    if (!frameCounts || clipPeak === 0) return null;
    const peaks = activityPeaks(
      frameCounts, windowStart, windowStart + windowFrames, ACTIVITY_BINS,
    );
    // A step outline across the slices, closed along the bottom.
    let d = `M0 ${clipPeak}`;
    peaks.forEach((n, i) => {
      d += ` V${clipPeak - n} H${i + 1}`;
    });
    return `${d} V${clipPeak} Z`;
  }, [frameCounts, clipPeak, windowStart, windowFrames]);
  const frameAt = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fraction = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    return Math.min(
      snapToSample(windowStart + fraction * windowFrames, sampleStep),
      Math.max(0, durationFrames - 1),
    );
  };
  const fraction = hover == null ? 0 : clamp((hover - windowStart) / windowFrames, 0, 1);
  const label =
    hover == null
      ? ""
      : `${frameRate ? `${formatClipPosition(hover / frameRate)} · ` : ""}frame ${hover}`;
  return (
    <div
      className="relative flex-1 cursor-pointer border-b border-border"
      data-testid="time-ruler"
      onMouseMove={(e) => setHover(frameAt(e))}
      onMouseLeave={() => setHover(null)}
      onClick={(e) => onSeek(frameAt(e))}
    >
      {activityPath && (
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox={`0 0 ${ACTIVITY_BINS} ${clipPeak}`}
          preserveAspectRatio="none"
          data-testid="activity-graph"
          aria-hidden
        >
          <path
            d={activityPath}
            className={activitySpecies ? undefined : "fill-primary/60"}
            style={activitySpecies && { fill: activitySpecies.color, fillOpacity: 0.6 }}
          />
        </svg>
      )}
      {frameCounts && (
        <span
          className="pointer-events-none absolute bottom-0 left-1 text-[9px] leading-3 text-muted-foreground"
          data-testid="activity-label"
        >
          {activitySpecies?.name ?? "All"}
        </span>
      )}
      {hover != null && (
        <>
          <div
            className="pointer-events-none absolute inset-y-0 w-px bg-foreground/60"
            style={{ left: `${fraction * 100}%` }}
          />
          <span
            className="pointer-events-none absolute top-0 whitespace-nowrap rounded bg-background/80 px-0.5 text-[10px] leading-4 text-foreground"
            data-testid="time-ruler-label"
            style={
              fraction < 0.5
                ? { left: `calc(${fraction * 100}% + 4px)` }
                : { right: `calc(${(1 - fraction) * 100}% + 4px)` }
            }
          >
            {label}
          </span>
        </>
      )}
    </div>
  );
}

/**
 * The timeline of one clip, from the file itself: rows and clip length
 * derived here, so the player (video mode) and the two modals (frame
 * mode) mount the same thing and nothing is computed twice in code.
 * Nothing is rendered for a clip without tracks. `durationFrames` is
 * the player's own measured length once it has one; frame mode passes
 * nothing and the stored length (or the last track) is used.
 *
 * Memoised because the player around it now re-renders at
 * animation-frame rate: its overlay interpolates between the frames the
 * detector sampled and so runs off fractional time, while this only
 * ever needs the whole frame the playhead sits on. Without the memo
 * every track's bars were rebuilt sixty times a second to move a
 * one-pixel line.
 */
export const ClipTimeline = memo(function ClipTimeline({
  file,
  detectionThreshold,
  durationFrames,
  currentFrame,
  selectedTrackId,
  markers,
  surface,
  zoomStep,
  onZoomStep,
  onSelectTrack,
  onSeek,
}: {
  file: FileWithDetections;
  /** The counting threshold: a bar is drawn for a track whose card is
   *  in scope at it, the same set the player draws and the tiles
   *  count. A rejected track above it keeps its bar, on the False
   *  detection row. */
  detectionThreshold: number;
  durationFrames?: number;
  currentFrame: number;
  selectedTrackId: string | null;
  markers?: TimelineMarker[];
  surface?: TimelineSurface;
  zoomStep?: number;
  onZoomStep?: (step: number) => void;
  onSelectTrack: (trackId: string) => void;
  onSeek: (frame: number) => void;
}) {
  const rows = useMemo(() => {
    const tracks = file.tracks ?? [];
    const onShow = trackCards(tracks, file.detections, detectionThreshold);
    return trackRows(
      tracks.filter((t) => onShow.has(t.id)),
      file.detections,
    );
  }, [file.tracks, file.detections, detectionThreshold]);
  const sampleStep = useMemo(() => fileSamplingStep(file), [file]);
  // The selected animal's species, unless it was rejected: a false
  // detection draws no box, so its graph would be empty.
  const selectedRow = rows.find(
    (row) => row.track.id === selectedTrackId && row.verdict !== "rejected",
  );
  const species = selectedRow?.label;
  const frameCounts = useMemo(
    () => frameBoxCounts(file, detectionThreshold, species),
    [file, detectionThreshold, species],
  );
  if (rows.length === 0) return null;
  return (
    <TrackTimeline
      rows={rows}
      durationFrames={durationFrames || clipDurationFrames(file)}
      currentFrame={currentFrame}
      selectedTrackId={selectedTrackId}
      markers={markers}
      frameRate={file.frame_rate}
      sampleStep={sampleStep}
      frameCounts={frameCounts}
      activitySpecies={
        selectedRow && { name: selectedRow.displayLabel, color: selectedRow.color }
      }
      surface={surface}
      zoomStep={zoomStep}
      onZoomStep={onZoomStep}
      onSelectTrack={onSelectTrack}
      onSeek={onSeek}
    />
  );
});

/**
 * Where in the clip one track was: a bar over the clip's length and the
 * numbers. The card modal shows it under the frame line. Without a
 * known clip length (a video analysed before the length was stored)
 * only the text is shown.
 */
export function TrackSpan({
  track,
  frameRate,
  durationSeconds,
  color,
  bar = { start: track.start_frame, end: track.end_frame + 1, gaps: [] },
}: {
  track: { start_frame: number; end_frame: number; frame_count: number };
  frameRate: number | null;
  durationSeconds: number | null;
  color: string;
  /** The bar's extent and its holes, as the timeline draws them
   *  (`trackGaps`); without it the bar runs solid to the last box. */
  bar?: { start: number; end: number; gaps: [number, number][] };
}) {
  const fps = frameRate || null;
  const start = fps ? track.start_frame / fps : null;
  const end = fps ? track.end_frame / fps : null;
  const seconds = start != null && end != null ? Math.round(end - start) : 0;
  const text = [
    start != null && end != null
      ? `from ${formatClipPosition(start)} to ${formatClipPosition(end)}`
      : `frames ${track.start_frame} to ${track.end_frame}`,
    seconds >= 1 ? `${seconds} s` : null,
    `${track.frame_count} ${track.frame_count === 1 ? "frame" : "frames"}`,
  ]
    .filter(Boolean)
    .join(" · ");
  const total = durationSeconds && fps ? durationSeconds * fps : null;
  return (
    <div className="space-y-1" data-testid="track-span">
      {total != null && total > 0 && (
        <div className="relative h-1.5 w-full rounded bg-muted">
          <div
            className="absolute top-0 h-full rounded"
            style={{
              left: `${(bar.start / total) * 100}%`,
              width: `${Math.max(0.5, ((bar.end - bar.start) / total) * 100)}%`,
              ...barBackground(color, bar.gaps, bar.start, bar.end),
            }}
          />
        </div>
      )}
      <div>{text}</div>
    </div>
  );
}

/** A bar's paint: solid once the animal's card is verified (a rejected
 *  animal is verified, in its class's grey from the colour map), at
 *  half strength until then. Mixed into the colour rather than set as
 *  opacity, so the gap line through the bar fades with it. The Counts
 *  page shows every bar solid. */
function barStyle(
  row: TrackRow,
  from: number,
  to: number,
  showVerdicts: boolean,
): React.CSSProperties {
  if (!showVerdicts || row.verdict !== "unverified") {
    return barBackground(row.color, row.gaps, from, to);
  }
  const fill = `color-mix(in srgb, ${row.color} ${UNVERIFIED_FILL_PERCENT}%, transparent)`;
  return barBackground(fill, row.gaps, from, to);
}
