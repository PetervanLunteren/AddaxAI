/**
 * EmptiesGrid - paged grid of empty-photo tiles.
 *
 * The Empties half of the Labels page. One tile per photo rather than
 * per detection, so it is a plain thumbnail: an empty photo has no box
 * worth cropping to, and drawing the sub-threshold ones on 48 tiles at
 * once is noise on a wall of vegetation. They are shown, dimmed and
 * scored, when a photo is opened full size, which is where they say
 * something useful.
 *
 * Deliberately not `CropGrid`. That one is window-virtualized with a
 * selection store and divider rows because it carries tens of thousands
 * of cards; this carries one page of 48. Reusing it would mean making
 * the most performance-sensitive component in the app generic over two
 * different tile types to save about a hundred lines.
 */

import { memo, useLayoutEffect, useRef, useState } from "react";
import { Check, ImageOff, Play } from "lucide-react";

import { API_BASE_URL } from "../../lib/api-client";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { basename } from "../../lib/path-utils";
import { reportMissingMedia } from "../../hooks/useBrokenDeployments";
import { cn } from "../../lib/utils";
import { columnsForWidth, useWideModeValue } from "./wide-mode";
import type { EmptyFileItem } from "../../api/types";
import type { TileSize } from "./CropGrid";

const GAP = 12;
// Wider than the crop grid's tiles at every step (crops run 80 / 124 /
// 290), because the two show completely different amounts of picture. A
// crop is a tight box around the animal, so the subject fills it. Here
// the subject is a speck in a landscape: measured across 1,470 animal
// boxes in a real deployment, the median animal covers 0.41% of the
// frame. At the old default width of 193px that is a 9x9 pixel blob,
// which is not something a person can judge. These give 193 / 295 / 398
// at a typical window: 5, 3 and 2 columns. Even at the largest, the
// median animal is only about 38 pixels across, which is why the
// full-size viewer still exists.
//
// The top is set by the thumbnail, not by taste: `/image?size=thumb`
// serves 768px wide (`_THUMB_MAX_WIDTH`), so a tile past that upscales
// and goes soft. L is deliberately under it.
const MIN_TILE: Record<TileSize, number> = { S: 220, M: 320, L: 460 };

interface EmptiesGridProps {
  items: EmptyFileItem[];
  selectedIds: Set<string>;
  onSelect: (fileId: string, e: React.MouseEvent) => void;
  onOpen: (item: EmptyFileItem) => void;
  /** Clicking the gap between tiles clears, as in the crop grid. */
  onBackgroundClick?: () => void;
  tileSize?: TileSize;
  /** Applied to the grid container, e.g. to dim held-over tiles. */
  className?: string;
}

export function EmptiesGrid({
  items,
  selectedIds,
  onSelect,
  onOpen,
  onBackgroundClick,
  tileSize = "M",
  className,
}: EmptiesGridProps) {
  const wide = useWideModeValue();
  const containerRef = useRef<HTMLDivElement>(null);
  const [columns, setColumns] = useState(4);

  // Measure rather than guess at breakpoints: wide mode changes the
  // container width without changing the viewport, so a media-query
  // grid would keep the same column count in both.
  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () =>
      setColumns(columnsForWidth(el.clientWidth, MIN_TILE[tileSize], GAP));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [tileSize, wide]);

  return (
    <div
      ref={containerRef}
      className={cn("grid", className)}
      onClick={(e) => {
        if (e.target === e.currentTarget) onBackgroundClick?.();
      }}
      style={{
        gap: GAP,
        gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
      }}
    >
      {items.map((item) => (
        <EmptyTile
          key={item.id}
          item={item}
          selected={selectedIds.has(item.id)}
          onSelect={onSelect}
          onOpen={onOpen}
          tileSize={tileSize}
        />
      ))}
    </div>
  );
}

interface EmptyTileProps {
  item: EmptyFileItem;
  selected: boolean;
  onSelect: (fileId: string, e: React.MouseEvent) => void;
  onOpen: (item: EmptyFileItem) => void;
  tileSize: TileSize;
}

const EmptyTile = memo(function EmptyTile({
  item,
  selected,
  onSelect,
  onOpen,
  tileSize,
}: EmptyTileProps) {
  const [imageFailed, setImageFailed] = useState(false);
  const isSmall = tileSize === "S";

  return (
    <div
      className={cn(
        "relative group cursor-pointer rounded-lg border bg-card text-card-foreground",
        "transition-[box-shadow,transform] duration-150",
        "hover:-translate-y-0.5 hover:shadow-md",
        selected && "ring-2 ring-offset-2 ring-[#0f6064]",
      )}
      onClick={(e) => onSelect(item.id, e)}
      onDoubleClick={(e) => {
        e.stopPropagation();
        onOpen(item);
      }}
      title={basename(item.file_path)}
    >
      {/* 4:3 and `object-contain`, so the whole frame is visible and
          nothing is cropped away. This matters more than it looks: 74%
          of the files in a real database are 4:3 and 24% are 16:9, so a
          fixed 16:9 tile with `object-cover` was cutting a quarter of
          the height off three quarters of the photos, top and bottom.
          That is exactly where an animal walking into shot appears, and
          this grid exists to find those. A 16:9 photo letterboxes here
          instead, which costs a little space and hides nothing. */}
      <div
        className={cn(
          "aspect-[4/3] bg-muted relative overflow-hidden",
          // S hides the caption, so the image is the bottom of the card
          // and has to carry the card's rounding itself.
          isSmall ? "rounded-lg" : "rounded-t-lg",
        )}
      >
        {imageFailed ? (
          <div className="absolute inset-0 flex items-center justify-center bg-neutral-200">
            <ImageOff
              className={cn(
                "text-neutral-400",
                isSmall ? "h-4 w-4" : "h-6 w-6",
              )}
            />
          </div>
        ) : (
          <img
            src={`${API_BASE_URL}/api/files/${item.id}/image?size=thumb`}
            alt={basename(item.file_path)}
            loading="lazy"
            className="w-full h-full object-contain"
            onError={() => {
              setImageFailed(true);
              reportMissingMedia(item.deployment_id);
            }}
          />
        )}
        {/* A video reads as a photo here, because the tile is the best
            frame. Without this the person scanning a wall of tiles has
            no way to tell which of them are clips they are seeing one
            frame of. Same pill as the event-context tiles in the
            Detections viewer, so it is not a new thing to learn. It is
            a marker, not a play button: the viewer shows the frame, and
            the note there says so. */}
        {item.file_type === "video" && (
          <span
            className="pointer-events-none absolute bottom-1 left-1 flex items-center justify-center rounded-full bg-black/60 p-1"
            title="Video, shown as one frame"
          >
            <Play className="h-3 w-3 fill-white text-white" />
          </span>
        )}
        {item.verified && (
          <div
            className="absolute top-1 right-1 rounded-full p-0.5 text-white"
            style={{ backgroundColor: "#0f6064" }}
            title="Verified"
          >
            <Check className="h-3 w-3" />
          </div>
        )}
      </div>
      {!isSmall && (
        <div className="px-2 py-1 text-[10px] text-muted-foreground truncate">
          {item.captured_at_local
            ? `${formatCameraDate(item.captured_at_local)} ${formatCameraTime(
                item.captured_at_local,
              )}`
            : basename(item.file_path)}
        </div>
      )}
    </div>
  );
});
