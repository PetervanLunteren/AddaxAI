/**
 * The tracks of a clip over its length, under the video player.
 *
 * One group per species, and inside a group as many rows as its animals
 * need to sit beside each other (`packTrackRows`). One bar per animal in
 * the species colour, styled by verdict: solid when every box is
 * verified, hatched while unverified, hollow when rejected. The species
 * is named on the group's first row only. A playhead follows the player,
 * a marker per species sits at the frame its MaxN was counted on (the
 * Counts modal passes those; the Files viewer passes none). Click a bar
 * to select that animal and jump to it; click the lane elsewhere to seek
 * there. Rows past the height cap scroll, and selecting an animal
 * scrolls its row into view.
 *
 * **Zoom.** Each slider step halves the visible span, and the window
 * stays centred on the playhead, clamped at the clip's ends. So the
 * playhead is the scroll position: clicking in the lane moves it, and
 * playback scrolls the window along. No panning and no horizontal
 * scrollbar, which is one piece of state rather than two kept in step.
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
import { clipDurationFrames, packTrackRows, trackRows } from "../../lib/track-utils";
import { formatClipPosition } from "../../lib/datetime";
import type { FileWithDetections } from "../../api/types";

export interface TimelineMarker {
  frame: number;
  label: string;
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
  onSelectTrack: (trackId: string) => void;
  onSeek: (frame: number) => void;
}

/** Rows visible before the lane scrolls. */
const ROW_CAP = 6;
const ROW_HEIGHT_PX = 16;
/** The most the slider zooms in. Below this a step stops being a useful
 *  amount of footage to look at. */
const MIN_WINDOW_SECONDS = 5;
/** A one-frame track is still worth a mark at the widest zoom. */
const MIN_BAR_PX = 2;

const clamp = (value: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, value));

export function TrackTimeline({
  rows,
  durationFrames,
  currentFrame,
  selectedTrackId,
  markers = [],
  frameRate,
  onSelectTrack,
  onSeek,
}: TrackTimelineProps) {
  useSpeciesColorsVersion();
  const [zoomStep, setZoomStep] = useState(0);
  const laneRef = useRef<HTMLDivElement>(null);

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
    currentFrame - windowFrames / 2,
    0,
    Math.max(0, durationFrames - windowFrames),
  );

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
  }, [selectedRow]);

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

  return (
    <div
      className="w-full shrink-0 border-t border-white/10 bg-black/80 text-[11px] text-white/80"
      data-testid="track-timeline"
    >
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
        style={{ maxHeight: ROW_CAP * ROW_HEIGHT_PX }}
      >
        {groups.map((group) =>
          group.rows.map((packed, rowIndex) => (
            <div
              key={`${group.label}-${rowIndex}`}
              className="flex items-stretch"
              style={{ height: ROW_HEIGHT_PX }}
            >
              <div
                className="w-28 shrink-0 truncate px-2 leading-4"
                title={rowIndex === 0 ? group.displayLabel : undefined}
              >
                {rowIndex === 0 ? group.displayLabel : ""}
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
                        title={`${group.displayLabel}, frames ${row.track.start_frame} to ${row.track.end_frame}`}
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
                          ...barStyle(row.color, row.verdict),
                        }}
                      />
                    );
                  })}
                {rowIndex === 0 &&
                  markers
                    .filter((m) => m.label === group.label && inWindow(m.frame))
                    .map((m) => (
                      <span
                        key={`${m.label}-${m.frame}`}
                        title={`MaxN at frame ${m.frame}`}
                        data-testid="maxn-marker"
                        className="pointer-events-none absolute -top-px h-0 w-0 -translate-x-1/2 border-x-[4px] border-t-[6px] border-x-transparent border-t-white"
                        style={{ left: `${pct(m.frame)}%` }}
                      />
                    ))}
              </div>
            </div>
          )),
        )}
      </div>
      {/* Playhead over every row, in the lane's column (the label
          column is 7rem wide). */}
      <div className="relative h-0">
        <div
          className="pointer-events-none absolute bottom-0 w-px bg-white"
          data-testid="playhead"
          style={{
            left: `calc(7rem + (100% - 7rem) * ${clamp((currentFrame - windowStart) / windowFrames, 0, 1)})`,
            height: Math.min(totalRows, ROW_CAP) * ROW_HEIGHT_PX,
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
  durationFrames,
  currentFrame,
  selectedTrackId,
  markers,
  onSelectTrack,
  onSeek,
}: {
  file: FileWithDetections;
  durationFrames?: number;
  currentFrame: number;
  selectedTrackId: string | null;
  markers?: TimelineMarker[];
  onSelectTrack: (trackId: string) => void;
  onSeek: (frame: number) => void;
}) {
  const rows = useMemo(
    () => trackRows(file.tracks ?? [], file.detections),
    [file.tracks, file.detections],
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

/** Solid, hatched or hollow, in the species colour. */
function barStyle(color: string, verdict: TrackRow["verdict"]): React.CSSProperties {
  if (verdict === "verified") return { background: color };
  if (verdict === "rejected") return { border: `1px solid ${color}`, background: "transparent" };
  return {
    background: `repeating-linear-gradient(45deg, ${color} 0 3px, transparent 3px 6px)`,
    border: `1px solid ${color}`,
  };
}
