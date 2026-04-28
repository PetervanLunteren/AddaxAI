/**
 * CropGrid - virtualized grid of detection crop thumbnails.
 *
 * Uses @tanstack/react-virtual for efficient rendering of large detection sets.
 * Responsive columns: 4 (sm), 6 (md), 8 (lg), 10 (xl).
 * Supports optional label divider rows between label groups.
 */

import { memo, useCallback, useRef, useMemo, useEffect, useLayoutEffect, useState, useSyncExternalStore } from "react";
import { useWindowVirtualizer } from "@tanstack/react-virtual";
import { Ban, Search, Tag } from "lucide-react";
import { CropCard } from "./CropCard";
import {
  ContextMenu,
  ContextMenuTrigger,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
} from "../ui/context-menu";
import type { DetectionSummary } from "../../api/types";

/**
 * Lightweight pub/sub store that lets individual GridCells subscribe to
 * their own selection state without re-rendering the entire CropGrid.
 */
/**
 * Lightweight pub/sub store that lets individual GridCells subscribe to
 * their own selection state without re-rendering the entire CropGrid.
 */
class SelectionStore {
  ids: Set<string> = new Set();
  private listeners = new Set<() => void>();

  getSnapshot = () => this.ids;

  subscribe = (cb: () => void) => {
    this.listeners.add(cb);
    return () => { this.listeners.delete(cb); };
  };

  /** Update data and notify subscribers. Safe to call from useLayoutEffect. */
  update(next: Set<string>) {
    if (next === this.ids) return;
    this.ids = next;
    for (const cb of this.listeners) cb();
  }
}

export type TileSize = "S" | "M" | "L";

type GridRow =
  | { type: "cards"; detections: DetectionSummary[] }
  | { type: "divider"; label: string; count: number };

interface CropGridProps {
  detections: DetectionSummary[];
  selectedIds: Set<string>;
  onSelect: (detectionId: string, e: React.MouseEvent) => void;
  onDoubleClick?: (detection: DetectionSummary) => void;
  onFindSimilar?: (detectionId: string) => void;
  onRelabel?: (detectionId: string, label: string, category: string) => void;
  onMarkFalse?: (detectionId: string) => void;
  onBackgroundClick?: () => void;
  tileSize?: TileSize;
  showLabelDividers?: boolean;
}

const COLUMN_PRESETS: Record<TileSize, [number, number, number, number]> = {
  S: [6, 8, 12, 14],
  M: [3, 5, 7, 9],
  L: [2, 3, 3, 4],
};

const ESTIMATE_SIZE: Record<TileSize, number> = {
  S: 140,
  M: 180,
  L: 380,
};

const DIVIDER_HEIGHT = 32;

function useColumns(tileSize: TileSize = "M"): number {
  const [cols, setCols] = useState(COLUMN_PRESETS[tileSize][1]);
  useEffect(() => {
    const preset = COLUMN_PRESETS[tileSize];
    function update() {
      const w = window.innerWidth;
      if (w < 640) setCols(preset[0]);
      else if (w < 1024) setCols(preset[1]);
      else if (w < 1280) setCols(preset[2]);
      else setCols(preset[3]);
    }
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [tileSize]);
  return cols;
}

const GAP_CLASS: Record<TileSize, string> = {
  S: "gap-2 pb-2",
  M: "gap-3 pb-3",
  L: "gap-4 pb-4",
};

interface GridCellProps {
  detection: DetectionSummary;
  selectionStore: SelectionStore;
  tileSize: TileSize;
  onSelect: (detectionId: string, e: React.MouseEvent) => void;
  onDoubleClick?: (detection: DetectionSummary) => void;
  onFindSimilar?: (detectionId: string) => void;
  onRelabel?: (detectionId: string, label: string, category: string) => void;
  onMarkFalse?: (detectionId: string) => void;
}

const GridCell = memo(function GridCell({
  detection,
  selectionStore,
  tileSize,
  onSelect,
  onDoubleClick,
  onFindSimilar,
  onRelabel,
  onMarkFalse,
}: GridCellProps) {
  const selected = useSyncExternalStore(
    selectionStore.subscribe,
    () => selectionStore.getSnapshot().has(detection.detection_id),
  );
  const showRelabel =
    onRelabel &&
    detection.neighbor_top_label &&
    detection.neighbor_top_label !== detection.label;

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <div data-crop-card>
          <CropCard
            detection={detection}
            selected={selected}
            tileSize={tileSize}
            onSelect={onSelect}
            onDoubleClick={onDoubleClick}
          />
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent>
        {onFindSimilar && (
          <ContextMenuItem
            onClick={() => onFindSimilar(detection.detection_id)}
          >
            <Search className="h-4 w-4" />
            Find similar
          </ContextMenuItem>
        )}
        {showRelabel && (
          <>
            <ContextMenuSeparator />
            <ContextMenuItem
              onClick={() =>
                onRelabel(
                  detection.detection_id,
                  detection.neighbor_top_label!,
                  detection.category
                )
              }
            >
              <Tag className="h-4 w-4" />
              Relabel to {detection.neighbor_top_label}
            </ContextMenuItem>
          </>
        )}
        {onMarkFalse && (
          <>
            <ContextMenuSeparator />
            <ContextMenuItem
              onClick={() => onMarkFalse(detection.detection_id)}
            >
              <Ban className="h-4 w-4" />
              Mark as false detection
            </ContextMenuItem>
          </>
        )}
      </ContextMenuContent>
    </ContextMenu>
  );
});

export function CropGrid({
  detections,
  selectedIds,
  onSelect,
  onDoubleClick,
  onFindSimilar,
  onRelabel,
  onMarkFalse,
  onBackgroundClick,
  tileSize = "M",
  showLabelDividers = false,
}: CropGridProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const columns = useColumns(tileSize);

  // Selection store — individual GridCells subscribe to their own selection
  // state via useSyncExternalStore, avoiding full grid re-renders.
  const [selectionStore] = useState(() => new SelectionStore());
  // useLayoutEffect fires synchronously after DOM mutation but before paint,
  // so cells see updated selection before the browser paints the frame.
  useLayoutEffect(() => { selectionStore.update(selectedIds); }, [selectedIds, selectionStore]);

  // Build rows: optionally insert divider rows at label transitions
  const rows = useMemo((): GridRow[] => {
    if (!showLabelDividers) {
      const result: GridRow[] = [];
      for (let i = 0; i < detections.length; i += columns) {
        result.push({ type: "cards", detections: detections.slice(i, i + columns) });
      }
      return result;
    }

    // Walk detections, group by label, insert dividers
    const result: GridRow[] = [];
    let i = 0;
    while (i < detections.length) {
      const currentLabel = detections[i].label || detections[i].category;
      // Count how many consecutive detections share this label
      let j = i;
      while (
        j < detections.length &&
        (detections[j].label || detections[j].category) === currentLabel
      ) {
        j++;
      }
      const count = j - i;
      result.push({ type: "divider", label: currentLabel, count });
      // Chunk this label group into card rows
      for (let k = i; k < j; k += columns) {
        result.push({ type: "cards", detections: detections.slice(k, Math.min(k + columns, j)) });
      }
      i = j;
    }
    return result;
  }, [detections, columns, showLabelDividers]);

  const cardHeight = ESTIMATE_SIZE[tileSize];

  const virtualizer = useWindowVirtualizer({
    count: rows.length,
    estimateSize: (index) =>
      rows[index].type === "divider" ? DIVIDER_HEIGHT : cardHeight,
    overscan: 5,
    scrollMargin: listRef.current?.offsetTop ?? 0,
    measureElement: (el) => el.getBoundingClientRect().height,
  });

  return (
    <div
      ref={listRef}
      style={{
        height: `${virtualizer.getTotalSize()}px`,
        width: "100%",
        position: "relative",
      }}
      onClick={(e) => {
        if (
          onBackgroundClick &&
          !(e.target as HTMLElement).closest("[data-crop-card]")
        ) {
          onBackgroundClick();
        }
      }}
    >
      {virtualizer.getVirtualItems().map((virtualRow) => {
        const row = rows[virtualRow.index];

        if (row.type === "divider") {
          return (
            <div
              key={`divider-${virtualRow.index}`}
              data-index={virtualRow.index}
              ref={virtualizer.measureElement}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                transform: `translateY(${virtualRow.start - virtualizer.options.scrollMargin}px)`,
              }}
            >
              <div className="flex items-center gap-2 px-1 h-full">
                <div className="h-px flex-1 bg-border" />
                <span className="text-xs text-muted-foreground font-medium capitalize whitespace-nowrap">
                  {row.label} ({row.count})
                </span>
                <div className="h-px flex-1 bg-border" />
              </div>
            </div>
          );
        }

        return (
          <div
            key={virtualRow.index}
            data-index={virtualRow.index}
            ref={virtualizer.measureElement}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              transform: `translateY(${virtualRow.start - virtualizer.options.scrollMargin}px)`,
            }}
          >
              <div
                className={`grid px-1 ${GAP_CLASS[tileSize]}`}
                style={{
                  gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                }}
              >
                {row.detections.map((det) => (
                  <GridCell
                    key={det.detection_id}
                    detection={det}
                    selectionStore={selectionStore}
                    tileSize={tileSize}
                    onSelect={onSelect}
                    onDoubleClick={onDoubleClick}
                    onFindSimilar={onFindSimilar}
                    onRelabel={onRelabel}
                    onMarkFalse={onMarkFalse}
                  />
                ))}
              </div>
            </div>
          );
        })}
    </div>
  );
}
