/**
 * The tracks of a clip over its length, under the video player.
 *
 * One group per species, and inside a group as many rows as its animals
 * need to sit beside each other (`packTrackRows`). One bar per animal in
 * the species colour, faint while the animal's card is unverified and
 * solid once it is; a rejected animal is a verified one on the "False
 * detection" row after the species, in that class's grey. The species
 * is named on the group's first row, with a MaxN button beside it when
 * the row has a marker: the frame its MaxN was counted on, which the
 * Counts modal takes from the event's rows and the Files viewer from
 * the file itself (`fileMaxNMarkers`). A playhead follows the player.
 * Click a bar to select that animal and jump to it; click the lane
 * elsewhere to seek there. Drag the top edge to make the lane taller;
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
 * Rendering only: which track is selected and where the playhead is
 * belong to the player and its modal. No keyboard handling here.
 */

import { memo, useEffect, useMemo, useRef, useState } from "react";

import { useSpeciesColorsVersion } from "../../utils/species-colors";
import { cn } from "../../lib/utils";
import type { TrackRow } from "../../lib/track-utils";
import {
  clipDurationFrames,
  packTrackRows,
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
/** The lane's height until someone drags it: six rows, the old cap. */
const DEFAULT_ROWS = 6;
/** The species column, named once for the rows and the playhead. */
const LABEL_COLUMN_REM = 10;
/** The most the slider zooms in. Below this a step stops being a useful
 *  amount of footage to look at. */
const MIN_WINDOW_SECONDS = 5;
/** A one-frame track is still worth a mark at the widest zoom. */
const MIN_BAR_PX = 2;
/** An unverified bar, so a checked one reads solid beside it. */
const UNVERIFIED_OPACITY = 0.4;
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
    defaultHeight: DEFAULT_ROWS * ROW_HEIGHT_PX,
    minHeight: 2 * ROW_HEIGHT_PX,
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
    const top = selectedRow * ROW_HEIGHT_PX;
    if (top < lane.scrollTop) lane.scrollTop = top;
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
      className="w-full shrink-0 border-t border-white/10 bg-black/80 text-[11px] text-white/80"
      data-testid="track-timeline"
    >
      <ResizeHandle
        onMouseDown={startDrag}
        title="Drag to resize the timeline"
        testId="timeline-resize-handle"
        className="bg-white/10 hover:bg-white/40"
      />
      {maxZoomStep > 0 && (
        <div className="flex items-center gap-2 px-2 py-[3px] text-[10px] text-white/60">
          <input
            type="range"
            min={0}
            max={maxZoomStep}
            step={1}
            value={step}
            onChange={(e) => setZoomStep(Number(e.target.value))}
            aria-label="Zoom the timeline"
            data-testid="timeline-zoom"
            className="h-1 w-24 cursor-pointer accent-white/80"
          />
          <span data-testid="timeline-span">{spanText}</span>
        </div>
      )}
      <div
        ref={laneRef}
        // Horizontal is hidden, not auto: a bar at the very end of the
        // clip is held to MIN_BAR_PX and so reaches past the lane,
        // which otherwise grows a horizontal scrollbar and eats a row's
        // worth of height.
        className="overflow-y-auto overflow-x-hidden"
        style={{ maxHeight: height }}
      >
        {groups.map((group) => {
          const maxN = markers.find((m) => m.label === group.label);
          return group.rows.map((packed, rowIndex) => (
            <div
              key={`${group.label}-${rowIndex}`}
              className="flex items-stretch"
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
                          "shrink-0 rounded border border-white/30 bg-black/60 px-1",
                          "text-[9px] leading-[13px] text-white hover:border-white/70",
                          "hover:bg-black/85 focus-visible:outline-none",
                          "focus-visible:ring-2 focus-visible:ring-white/70",
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
                    const left = Math.max(0, pct(row.track.start_frame));
                    const right = Math.min(100, pct(row.track.end_frame + 1));
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
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectTrack(row.track.id);
                        }}
                        className={cn(
                          "absolute top-[3px] h-[10px] rounded-sm",
                          selected && "ring-2 ring-white",
                        )}
                        style={{
                          left: `${left}%`,
                          width: `${Math.max(0, right - left)}%`,
                          minWidth: MIN_BAR_PX,
                          ...barStyle(row.color, showVerdicts ? row.verdict : "verified"),
                        }}
                      />
                    );
                  })}
                {rowIndex === 0 && maxN && inWindow(maxN.frame) && (
                  <span
                    title={`MaxN at frame ${maxN.frame}`}
                    data-testid="maxn-marker"
                    className="pointer-events-none absolute -top-px h-0 w-0 -translate-x-1/2 border-x-[4px] border-t-[6px] border-x-transparent border-t-white"
                    style={{ left: `${pct(maxN.frame)}%` }}
                  />
                )}
              </div>
            </div>
          ));
        })}
      </div>
      {/* Playhead over every row, in the lane's column. */}
      <div className="relative h-0">
        <div
          className="pointer-events-none absolute bottom-0 w-px bg-white"
          data-testid="playhead"
          style={{
            left: `calc(${LABEL_COLUMN_REM}rem + (100% - ${LABEL_COLUMN_REM}rem) * ${clamp((shownFrame - windowStart) / windowFrames, 0, 1)})`,
            height: Math.min(totalRows * ROW_HEIGHT_PX, height),
          }}
        />
      </div>
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
  if (rows.length === 0) return null;
  return (
    <TrackTimeline
      rows={rows}
      durationFrames={durationFrames || clipDurationFrames(file)}
      currentFrame={currentFrame}
      selectedTrackId={selectedTrackId}
      markers={markers}
      frameRate={file.frame_rate}
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
}: {
  track: { start_frame: number; end_frame: number; frame_count: number };
  frameRate: number | null;
  durationSeconds: number | null;
  color: string;
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
              left: `${(track.start_frame / total) * 100}%`,
              width: `${Math.max(0.5, ((track.end_frame - track.start_frame + 1) / total) * 100)}%`,
              background: color,
            }}
          />
        </div>
      )}
      <div>{text}</div>
    </div>
  );
}

/** Faint until the animal's card is verified, solid after; a rejected
 *  animal is verified, in its class's grey from the colour map. */
function barStyle(color: string, verdict: TrackRow["verdict"]): React.CSSProperties {
  return { background: color, opacity: verdict === "unverified" ? UNVERIFIED_OPACITY : 1 };
}
