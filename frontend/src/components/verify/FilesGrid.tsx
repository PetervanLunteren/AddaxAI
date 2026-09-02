/**
 * FilesGrid - paged grid of file tiles for the Files tab.
 *
 * One tile per file rather than per detection: the whole frame with its
 * visible boxes drawn on it (`FrameThumbnail`, the same overlay the
 * Counts filmstrip uses), so a file with boxes is recognisable at a
 * glance and an empty one is a plain photo. Each tile fetches the file
 * detail for its boxes, the same query the viewer opens, so one cache
 * entry serves both (`EventCollage` does the same).
 *
 * Deliberately not `CropGrid`. That one is window-virtualized with a
 * selection store and divider rows because it carries tens of thousands
 * of cards; this carries one page of 48. Reusing it would mean making
 * the most performance-sensitive component in the app generic over two
 * different tile types to save about a hundred lines.
 */

import { memo, useLayoutEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check } from "lucide-react";

import { filesApi } from "../../api/files";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { basename } from "../../lib/path-utils";
import { cn } from "../../lib/utils";
import { FrameThumbnail } from "./FrameThumbnail";
import { columnsForWidth, useWideModeValue } from "./wide-mode";
import type { LabelsFileItem } from "../../api/types";
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

interface FilesGridProps {
  items: LabelsFileItem[];
  selectedIds: Set<string>;
  onSelect: (fileId: string, e: React.MouseEvent) => void;
  onOpen: (item: LabelsFileItem) => void;
  /** Clicking the gap between tiles clears, as in the crop grid. */
  onBackgroundClick?: () => void;
  /** The project's counting threshold: which boxes the tiles draw. */
  detectionThreshold: number;
  tileSize?: TileSize;
  /** Applied to the grid container, e.g. to dim held-over tiles. */
  className?: string;
}

export function FilesGrid({
  items,
  selectedIds,
  onSelect,
  onOpen,
  onBackgroundClick,
  detectionThreshold,
  tileSize = "M",
  className,
}: FilesGridProps) {
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
        <FileTile
          key={item.id}
          item={item}
          selected={selectedIds.has(item.id)}
          onSelect={onSelect}
          onOpen={onOpen}
          detectionThreshold={detectionThreshold}
          tileSize={tileSize}
        />
      ))}
    </div>
  );
}

interface FileTileProps {
  item: LabelsFileItem;
  selected: boolean;
  onSelect: (fileId: string, e: React.MouseEvent) => void;
  onOpen: (item: LabelsFileItem) => void;
  detectionThreshold: number;
  tileSize: TileSize;
}

const FileTile = memo(function FileTile({
  item,
  selected,
  onSelect,
  onOpen,
  detectionThreshold,
  tileSize,
}: FileTileProps) {
  const isSmall = tileSize === "S";
  // The same key the viewer uses, so opening a tile costs no request.
  const { data: file } = useQuery({
    queryKey: ["file", item.id],
    queryFn: ({ signal }) => filesApi.get(item.id, { signal }),
  });

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
      {/* 4:3 and `contain`, so the whole frame is visible and nothing is
          cropped away. This matters more than it looks: 74% of the files
          in a real database are 4:3 and 24% are 16:9, so a fixed 16:9
          tile with `cover` was cutting a quarter of the height off three
          quarters of the photos, top and bottom. That is exactly where an
          animal walking into shot appears, and this grid exists to find
          those. A 16:9 photo letterboxes here instead, which costs a
          little space and hides nothing. */}
      <div
        className={cn(
          "aspect-[4/3] relative overflow-hidden",
          // S hides the caption, so the image is the bottom of the card
          // and has to carry the card's rounding itself.
          isSmall ? "rounded-lg" : "rounded-t-lg",
        )}
      >
        <FrameThumbnail
          fileId={item.id}
          file={file}
          detectionThreshold={detectionThreshold}
          fit="contain"
        />
        {/* A video reads as a photo here, because the tile is the best
            frame. Without this, someone scanning a wall of tiles has no
            way to tell which of them are clips they are seeing one frame
            of.

            Words rather than a play triangle, and that is the whole
            point. A filled triangle in a circle means "press me to
            watch" everywhere else on the internet, and this page
            deliberately offers no playback, so the glyph invited a click
            that did nothing (the badge is click-through, so it selected
            the tile instead). It also said the wrong thing about the
            feature: the frame is all there is.

            Top left, not bottom left. Tiles are 4:3 with `contain` and
            camera videos are 16:9, so the picture letterboxes and
            anything pinned to the bottom floats on the background strip
            below it, reading as a control under the photo rather than a
            mark on it. The top edge has no such gap, and `Check` already
            owns the top right. */}
        {item.file_type === "video" && (
          <span
            className="pointer-events-none absolute top-1 left-1 rounded bg-black/60 px-1 py-0.5 text-[10px] leading-none text-white"
          >
            Video · one frame
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
