/**
 * EventFilmstrip - the resizable wrapping-grid filmstrip under the focus
 * image in the Counts modal. Shows every frame of the event; click a tile
 * to make it the focus. The grid auto-wraps to fill the width at the chosen
 * S/M/L thumbnail size, so dragging the divider taller turns it into a full
 * scanning grid. Thumbnails mirror the focus's box overlay and image filter.
 */

import { Play } from "lucide-react";

import { cn } from "../../lib/utils";
import { formatTimeOffset } from "../../lib/datetime";
import { describeEventMedia } from "../../lib/event-media";
import { FrameThumbnail } from "./FrameThumbnail";
import { TileSizeToggle } from "./TileSizeToggle";
import type { TileSize } from "./CropGrid";
import type { FileWithDetections } from "../../api/types";

// Minimum tile width per size; the grid fits as many columns as the region
// width allows.
const TILE_MIN_WIDTH: Record<TileSize, number> = { S: 96, M: 140, L: 200 };

interface EventFilmstripProps {
  files: FileWithDetections[];
  selectedIndex: number;
  onSelectFile: (index: number) => void;
  detectionThreshold: number;
  showBoxes: boolean;
  imageFilter?: string;
  tileSize: TileSize;
  onTileSizeChange: (v: TileSize) => void;
}

export function EventFilmstrip({
  files,
  selectedIndex,
  onSelectFile,
  detectionThreshold,
  showBoxes,
  imageFilter,
  tileSize,
  onTileSizeChange,
}: EventFilmstripProps) {
  // Each tile shows the time gap since the previous frame, so an unusually
  // large gap (a pause, or really two encounters) jumps out. The first
  // frame and any file without a timestamp show no label.
  const times = files.map((f) =>
    f.captured_at_local ? new Date(f.captured_at_local).getTime() : null,
  );

  return (
    <div className="flex h-full flex-col bg-muted/30">
      <div className="flex items-center justify-between gap-2 px-3 py-1.5 border-b shrink-0">
        <span className="text-xs text-muted-foreground">
          {describeEventMedia(files)}
        </span>
        <TileSizeToggle
          value={tileSize}
          onChange={onTileSizeChange}
          className="w-28"
        />
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        <div
          className="grid gap-2"
          style={{
            gridTemplateColumns: `repeat(auto-fill, minmax(${TILE_MIN_WIDTH[tileSize]}px, 1fr))`,
          }}
        >
          {files.map((file, index) => {
            const t = times[index];
            const prev = index > 0 ? times[index - 1] : null;
            // First frame is the reference point ("start"); the rest show the
            // gap since the previous frame. "+0" would be wrong — there's no
            // previous frame to measure from.
            const offset =
              index === 0
                ? "start"
                : t != null && prev != null
                  ? formatTimeOffset((t - prev) / 1000)
                  : null;
            return (
            <button
              key={file.id}
              type="button"
              onClick={() => onSelectFile(index)}
              className={cn(
                "relative aspect-[4/3] overflow-hidden rounded-md border-2 transition-colors",
                index === selectedIndex
                  ? "border-primary ring-2 ring-primary/30"
                  : "border-transparent opacity-90 hover:border-primary/50 hover:opacity-100",
              )}
            >
              <FrameThumbnail
                fileId={file.id}
                file={file}
                detectionThreshold={detectionThreshold}
                showBoxes={showBoxes}
                imageFilter={imageFilter}
              />
              {offset != null && (
                <span className="pointer-events-none absolute top-1 left-1 rounded bg-black/55 px-1.5 py-0.5 text-[11px] leading-none tabular-nums text-white/90">
                  {offset}
                </span>
              )}
              {file.file_type === "video" && (
                <span className="pointer-events-none absolute bottom-1 left-1 flex items-center justify-center rounded-full bg-black/60 p-1">
                  <Play className="h-3 w-3 fill-white text-white" />
                </span>
              )}
            </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
