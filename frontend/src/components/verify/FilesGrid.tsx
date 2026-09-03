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

import { memo, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { Fragment } from "react";

import { filesApi } from "../../api/files";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { getDetectionColor, shouldDrawBbox } from "../../lib/detection-utils";
import { basename } from "../../lib/path-utils";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import { cn } from "../../lib/utils";
import {
  getContrastTextColor,
  useSpeciesColorsVersion,
} from "../../utils/species-colors";
import { FrameThumbnail } from "./FrameThumbnail";
import { StatusBadgeCluster } from "./StatusBadgeCluster";
import { columnsForWidth, useWideModeValue } from "./wide-mode";
import type { LabelsFileItem } from "../../api/types";
import type { TileSize } from "./CropGrid";

const GAP = 12;
// Wider than the crop grid's tiles at every step (crops run 80 / 124 /
// 290), because the two show completely different amounts of picture. A
// crop is a tight box around the animal, so the subject fills it. Here
// the subject is a speck in a landscape: measured across 1,470 animal
// boxes in a real deployment, the median animal covers 0.41% of the
// frame, a blob a few pixels across at small tile widths, which is not
// something a person can judge. At the capped page width (max-w-7xl,
// 1216px of grid) these give 4, 3 and 2 columns, tiles of 295 / 397 /
// 602. Even at the largest, the median animal is only about 38 pixels
// across, which is why the full-size viewer still exists. Wide mode
// adds columns instead of stretching tiles.
//
// The top is set by the thumbnail, not by taste: `/image?size=thumb`
// serves 768px wide (`_THUMB_MAX_WIDTH`), so a tile past that upscales
// and goes soft. L is deliberately under it.
const MIN_TILE: Record<TileSize, number> = { S: 260, M: 320, L: 460 };

// Species chips shown per tile before "+n more". Distinct species per
// file is nearly always 1 or 2, so the cap is a guard, not a feature.
const MAX_CHIPS = 3;

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
  /** Divider rows per event (the events sort): the page's items are
   *  grouped by consecutive `event_id`, each group headed by its first
   *  file's capture time, a count and a Select link, mirroring the
   *  Detections tab's event dividers. */
  groupByEvent?: boolean;
  /** Replace the selection with one event's files (the divider link). */
  onSelectEvent?: (fileIds: string[]) => void;
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
  groupByEvent = false,
  onSelectEvent,
  className,
}: FilesGridProps) {
  const wide = useWideModeValue();

  // One tile shape per page: the majority aspect ratio of the files on
  // it. Camera traps shoot 4:3 and 16:9 in one project (a real one held
  // 4,102 against 2,143), and a fixed 4:3 frame letterboxed every photo
  // of a 16:9 camera. The default sort groups by folder, so a page is
  // usually one camera and fills its tiles edge to edge; on a mixed
  // page the minority keeps its neutral bars (`fit="contain"`), which
  // is the cost of rows that stay aligned. Per-row shapes were the
  // alternative and lose that: row membership changes with every
  // resize, so tiles would change shape under the cursor.
  const aspect = useMemo(() => {
    const counts = new Map<number, number>();
    for (const it of items) {
      if (!it.width_px || !it.height_px) continue;
      const r = Math.round((it.width_px / it.height_px) * 100) / 100;
      counts.set(r, (counts.get(r) ?? 0) + 1);
    }
    let best = 4 / 3;
    let bestN = 0;
    for (const [r, n] of counts) {
      if (n > bestN) {
        best = r;
        bestN = n;
      }
    }
    return best;
  }, [items]);
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

  // Consecutive runs of one event_id; the list arrives ordered by
  // event, so runs are whole events (an event split across a page
  // boundary repeats its divider on the next page, which is honest).
  const groups = useMemo(() => {
    if (!groupByEvent) return [{ key: "all", items }];
    const out: { key: string; items: LabelsFileItem[] }[] = [];
    for (const item of items) {
      const key = item.event_id ?? "none";
      const last = out[out.length - 1];
      if (last && last.key === key) last.items.push(item);
      else out.push({ key, items: [item] });
    }
    return out;
  }, [groupByEvent, items]);

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
      {groups.map((group, gi) => (
        <Fragment key={`${group.key}-${gi}`}>
          {groupByEvent && (
            <div
              className="flex items-center gap-2 px-1 pt-1"
              style={{ gridColumn: "1 / -1" }}
              data-event-divider={group.key}
            >
              <span className="text-xs text-muted-foreground font-medium whitespace-nowrap">
                {group.key === "none"
                  ? "No event"
                  : group.items[0].captured_at_local
                    ? `${formatCameraDate(group.items[0].captured_at_local)} · ${formatCameraTime(group.items[0].captured_at_local)}`
                    : "Event"}{" "}
                ({group.items.length})
              </span>
              <div className="h-px flex-1 bg-border" />
              {onSelectEvent && (
                <button
                  type="button"
                  onClick={(e) => {
                    // Don't bubble to the grid's background handler,
                    // which would clear the selection we just set.
                    e.stopPropagation();
                    onSelectEvent(group.items.map((i) => i.id));
                  }}
                  className="text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline whitespace-nowrap"
                >
                  Select
                </button>
              )}
            </div>
          )}
          {group.items.map((item) => (
            <FileTile
              key={item.id}
              item={item}
              selected={selectedIds.has(item.id)}
              onSelect={onSelect}
              onOpen={onOpen}
              detectionThreshold={detectionThreshold}
              tileSize={tileSize}
              aspect={aspect}
            />
          ))}
        </Fragment>
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
  /** The page's shared width/height ratio (see the grid's useMemo). */
  aspect: number;
}

const FileTile = memo(function FileTile({
  item,
  selected,
  onSelect,
  onOpen,
  detectionThreshold,
  tileSize,
  aspect,
}: FileTileProps) {
  const isSmall = tileSize === "S";
  // The same key the viewer uses, so opening a tile costs no request.
  const { data: file } = useQuery({
    queryKey: ["file", item.id],
    queryFn: ({ signal }) => filesApi.get(item.id, { signal }),
  });
  // Chip colours resolve from the project colour map, which loads after
  // the tiles; repaint when it lands.
  useSpeciesColorsVersion();

  // One chip per species over the boxes the tile draws (same filter as
  // FrameThumbnail, so a chip never names an invisible box). Ordered by
  // how many boxes carry the label, so the dominant species comes first
  // when the cap cuts the list.
  const chips: { key: string; name: string; color: string; boxes: number }[] =
    [];
  if (file) {
    for (const d of file.detections) {
      if (!shouldDrawBbox(d, file, detectionThreshold)) continue;
      const key = (d.label_taxonomy_id || d.label || d.category).toLowerCase();
      const existing = chips.find((c) => c.key === key);
      if (existing) {
        existing.boxes += 1;
      } else {
        chips.push({
          key,
          name: resolveSpeciesName(d),
          color: getDetectionColor(d),
          boxes: 1,
        });
      }
    }
    chips.sort((a, b) => b.boxes - a.boxes);
  }

  return (
    <div
      className={cn(
        "relative group cursor-pointer rounded-lg border bg-card text-card-foreground",
        "transition-[box-shadow,transform] duration-150",
        "hover:-translate-y-0.5 hover:shadow-md",
        selected && "ring-2 ring-offset-2 ring-[#0f6064]",
      )}
      data-file-id={item.id}
      onClick={(e) => onSelect(item.id, e)}
      onDoubleClick={(e) => {
        e.stopPropagation();
        onOpen(item);
      }}
      title={basename(item.file_path)}
    >
      {/* Corner badges — the Counts cards' cluster: verified check,
          like, flag. Read from the tile's live file query (the same row
          `FrameThumbnail` draws from), so a flag set in the viewer
          shows up here without a list refetch; the list row is the
          fallback while that query is in flight. */}
      <StatusBadgeCluster
        confirmed={file?.verified ?? item.verified}
        favorited={file?.favorited ?? false}
        flagged={file?.flagged ?? false}
      />
      {/* The page's majority ratio and `contain`, so the whole frame is
          visible and nothing is cropped away. `cover` was the original
          sin here: a fixed 16:9 tile cut a quarter of the height off
          every 4:3 photo, top and bottom, exactly where an animal
          walking into shot appears, and this grid exists to find those.
          A minority-ratio photo letterboxes instead, which costs a
          little space and hides nothing. */}
      <div
        className={cn(
          "relative overflow-hidden",
          // S hides the caption, so the image is the bottom of the card
          // and has to carry the card's rounding itself.
          isSmall ? "rounded-lg" : "rounded-t-lg",
        )}
        style={{ aspectRatio: aspect }}
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
            mark on it. The top edge has no such gap, and the badge
            cluster already owns the top right.

            The species chips stack under it: what the AI thinks is in
            the picture, so "verify as is" is an informed action without
            opening the file. One chip per species in the species
            colour, text picked for contrast against it. */}
        <div className="pointer-events-none absolute top-1 left-1 flex max-w-[calc(100%-2rem)] flex-col items-start gap-0.5">
          {item.file_type === "video" && (
            <span className="rounded bg-black/60 px-1 py-0.5 text-[10px] leading-none text-white">
              Video · one frame
            </span>
          )}
          {chips.slice(0, MAX_CHIPS).map((chip) => (
            <span
              key={chip.key}
              className="max-w-full truncate rounded px-1.5 py-0.5 text-xs leading-tight"
              style={{
                backgroundColor: chip.color,
                color: getContrastTextColor(chip.color),
              }}
            >
              {chip.boxes > 1 ? `${chip.boxes}× ${chip.name}` : chip.name}
            </span>
          ))}
          {chips.length > MAX_CHIPS && (
            <span className="rounded bg-black/60 px-1.5 py-0.5 text-xs leading-tight text-white">
              +{chips.length - MAX_CHIPS} more
            </span>
          )}
        </div>
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
