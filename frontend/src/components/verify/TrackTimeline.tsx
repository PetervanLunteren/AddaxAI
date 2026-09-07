/**
 * The tracks of a clip over its length, under the video player.
 *
 * One row per species, one bar per track in the species colour, styled
 * by verdict: solid when every box is verified, hatched while
 * unverified, hollow when rejected. A playhead follows the player, a
 * marker per species sits at the frame its MaxN was counted on (the
 * Counts modal passes those; the Files viewer passes none). Click a bar
 * to select that animal and jump to it; click the lane elsewhere to
 * seek there. Rows past the height cap scroll.
 *
 * Rendering only: which track is selected and where the playhead is
 * belong to the player and its modal. No keyboard handling here.
 */

import { useSpeciesColorsVersion } from "../../utils/species-colors";
import { cn } from "../../lib/utils";
import type { TrackRow } from "../../lib/track-utils";
import { formatClipPosition } from "../../lib/datetime";

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
  onSelectTrack: (trackId: string) => void;
  onSeek: (frame: number) => void;
}

/** Rows visible before the lane scrolls. */
const ROW_CAP = 6;
const ROW_HEIGHT_PX = 16;

export function TrackTimeline({
  rows,
  durationFrames,
  currentFrame,
  selectedTrackId,
  markers = [],
  onSelectTrack,
  onSeek,
}: TrackTimelineProps) {
  useSpeciesColorsVersion();
  if (rows.length === 0 || durationFrames <= 0) return null;

  const species: { label: string; displayLabel: string }[] = [];
  for (const row of rows) {
    if (!species.some((sp) => sp.label === row.label)) {
      species.push({ label: row.label, displayLabel: row.displayLabel });
    }
  }
  const pct = (frame: number) =>
    `${Math.min(100, Math.max(0, (frame / durationFrames) * 100))}%`;

  const seekAt = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fraction = (e.clientX - rect.left) / rect.width;
    onSeek(Math.round(Math.min(1, Math.max(0, fraction)) * durationFrames));
  };

  return (
    <div
      className="w-full shrink-0 border-t border-white/10 bg-black/80 text-[11px] text-white/80"
      data-testid="track-timeline"
    >
      <div
        className="overflow-y-auto"
        style={{ maxHeight: ROW_CAP * ROW_HEIGHT_PX }}
      >
        {species.map(({ label, displayLabel }) => (
          <div key={label} className="flex items-stretch" style={{ height: ROW_HEIGHT_PX }}>
            <div className="w-28 shrink-0 truncate px-2 leading-4" title={displayLabel}>
              {displayLabel}
            </div>
            <div
              className="relative flex-1 cursor-pointer"
              onClick={seekAt}
              data-testid="track-lane"
            >
              {rows
                .filter((row) => row.label === label)
                .map((row) => {
                  const selected = row.track.id === selectedTrackId;
                  const width = Math.max(
                    0.3,
                    ((row.track.end_frame - row.track.start_frame + 1) / durationFrames) * 100,
                  );
                  return (
                    <button
                      key={row.track.id}
                      type="button"
                      title={`${displayLabel}, frames ${row.track.start_frame} to ${row.track.end_frame}`}
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
                        left: pct(row.track.start_frame),
                        width: `${width}%`,
                        ...barStyle(row.color, row.verdict),
                      }}
                    />
                  );
                })}
              {markers
                .filter((m) => m.label === label)
                .map((m) => (
                  <span
                    key={`${m.label}-${m.frame}`}
                    title={`MaxN at frame ${m.frame}`}
                    data-testid="maxn-marker"
                    className="pointer-events-none absolute -top-px h-0 w-0 -translate-x-1/2 border-x-[4px] border-t-[6px] border-x-transparent border-t-white"
                    style={{ left: pct(m.frame) }}
                  />
                ))}
            </div>
          </div>
        ))}
      </div>
      {/* Playhead over every row, in the lane's column (the label
          column is 7rem wide). */}
      <div className="relative h-0">
        <div
          className="pointer-events-none absolute bottom-0 w-px bg-white"
          data-testid="playhead"
          style={{
            left: `calc(7rem + (100% - 7rem) * ${Math.min(1, Math.max(0, currentFrame / durationFrames))})`,
            height: Math.min(species.length, ROW_CAP) * ROW_HEIGHT_PX,
          }}
        />
      </div>
    </div>
  );
}

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
