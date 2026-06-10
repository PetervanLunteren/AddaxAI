/**
 * EventFilmstrip - the resizable wrapping-grid filmstrip under the focus
 * image in the Counts modal. Shows every frame of the event; click a tile
 * to make it the focus. The grid auto-wraps to fill the width at the chosen
 * S/M/L thumbnail size, so dragging the divider taller turns it into a full
 * scanning grid. Thumbnails mirror the focus's box overlay and image filter.
 */

import { Play } from "lucide-react";

import { cn } from "../../lib/utils";
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
  return (
    <div className="flex h-full flex-col bg-muted/30">
      <div className="flex items-center justify-between gap-2 px-3 py-1.5 border-b shrink-0">
        <span className="text-xs text-muted-foreground">
          {files.length} frame{files.length !== 1 ? "s" : ""}
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
          {files.map((file, index) => (
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
              {file.file_type === "video" && (
                <span className="pointer-events-none absolute bottom-1 left-1 flex items-center justify-center rounded-full bg-black/60 p-1">
                  <Play className="h-3 w-3 fill-white text-white" />
                </span>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
