/**
 * CropGrid - virtualized grid of detection crop thumbnails.
 *
 * Uses @tanstack/react-virtual for efficient rendering of large detection sets.
 * Responsive columns: 4 (sm), 6 (md), 8 (lg), 10 (xl).
 * Supports optional species divider rows between species groups.
 */

import { useRef, useMemo, useEffect, useState } from "react";
import { useWindowVirtualizer } from "@tanstack/react-virtual";
import { Search, Tag } from "lucide-react";
import { CropCard } from "./CropCard";
import {
  ContextMenu,
  ContextMenuTrigger,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
} from "../ui/context-menu";
import type { DetectionSummary } from "../../api/types";

export type TileSize = "S" | "M" | "L";

type GridRow =
  | { type: "cards"; detections: DetectionSummary[] }
  | { type: "divider"; species: string; count: number };

interface CropGridProps {
  detections: DetectionSummary[];
  selectedIds: Set<string>;
  onSelect: (detectionId: string, e: React.MouseEvent) => void;
  onCardClick: (detection: DetectionSummary) => void;
  onFindSimilar?: (detectionId: string) => void;
  onRelabel?: (detectionId: string, species: string, category: string) => void;
  tileSize?: TileSize;
  showSpeciesDividers?: boolean;
}

const COLUMN_PRESETS: Record<TileSize, [number, number, number, number]> = {
  S: [6, 8, 12, 14],
  M: [4, 6, 8, 10],
  L: [3, 4, 6, 8],
};

const ESTIMATE_SIZE: Record<TileSize, number> = {
  S: 140,
  M: 200,
  L: 280,
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

export function CropGrid({
  detections,
  selectedIds,
  onSelect,
  onCardClick,
  onFindSimilar,
  onRelabel,
  tileSize = "M",
  showSpeciesDividers = false,
}: CropGridProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const columns = useColumns(tileSize);

  // Build rows: optionally insert divider rows at species transitions
  const rows = useMemo((): GridRow[] => {
    if (!showSpeciesDividers) {
      const result: GridRow[] = [];
      for (let i = 0; i < detections.length; i += columns) {
        result.push({ type: "cards", detections: detections.slice(i, i + columns) });
      }
      return result;
    }

    // Walk detections, group by species, insert dividers
    const result: GridRow[] = [];
    let i = 0;
    while (i < detections.length) {
      const currentSpecies = detections[i].species || detections[i].category;
      // Count how many consecutive detections share this species
      let j = i;
      while (
        j < detections.length &&
        (detections[j].species || detections[j].category) === currentSpecies
      ) {
        j++;
      }
      const count = j - i;
      result.push({ type: "divider", species: currentSpecies, count });
      // Chunk this species group into card rows
      for (let k = i; k < j; k += columns) {
        result.push({ type: "cards", detections: detections.slice(k, Math.min(k + columns, j)) });
      }
      i = j;
    }
    return result;
  }, [detections, columns, showSpeciesDividers]);

  const cardHeight = ESTIMATE_SIZE[tileSize];

  const virtualizer = useWindowVirtualizer({
    count: rows.length,
    estimateSize: (index) =>
      rows[index].type === "divider" ? DIVIDER_HEIGHT : cardHeight,
    overscan: 5,
    scrollMargin: listRef.current?.offsetTop ?? 0,
  });

  return (
    <div
      ref={listRef}
      style={{
        height: `${virtualizer.getTotalSize()}px`,
        width: "100%",
        position: "relative",
      }}
    >
      {virtualizer.getVirtualItems().map((virtualRow) => {
        const row = rows[virtualRow.index];

        if (row.type === "divider") {
          return (
            <div
              key={`divider-${virtualRow.index}`}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start - virtualizer.options.scrollMargin}px)`,
              }}
            >
              <div className="flex items-center gap-2 px-1 h-full">
                <div className="h-px flex-1 bg-border" />
                <span className="text-xs text-muted-foreground font-medium capitalize whitespace-nowrap">
                  {row.species} ({row.count})
                </span>
                <div className="h-px flex-1 bg-border" />
              </div>
            </div>
          );
        }

        return (
          <div
            key={virtualRow.index}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: `${virtualRow.size}px`,
              transform: `translateY(${virtualRow.start - virtualizer.options.scrollMargin}px)`,
            }}
          >
              <div
                className="grid gap-2 px-1"
                style={{
                  gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                }}
              >
                {row.detections.map((det) => {
                  const showRelabel =
                    onRelabel &&
                    det.neighbor_top_label &&
                    det.neighbor_top_label !== det.species;

                  return (
                    <ContextMenu key={det.detection_id}>
                      <ContextMenuTrigger asChild>
                        <div>
                          <CropCard
                            detection={det}
                            selected={selectedIds.has(det.detection_id)}
                            onClick={(e) => {
                              if (e.ctrlKey || e.metaKey || e.shiftKey || selectedIds.size > 0) {
                                onSelect(det.detection_id, e);
                              } else {
                                onCardClick(det);
                              }
                            }}
                          />
                        </div>
                      </ContextMenuTrigger>
                      <ContextMenuContent>
                        {onFindSimilar && (
                          <ContextMenuItem
                            onClick={() => onFindSimilar(det.detection_id)}
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
                                  det.detection_id,
                                  det.neighbor_top_label!,
                                  det.category
                                )
                              }
                            >
                              <Tag className="h-4 w-4" />
                              Relabel to {det.neighbor_top_label}
                            </ContextMenuItem>
                          </>
                        )}
                      </ContextMenuContent>
                    </ContextMenu>
                  );
                })}
              </div>
            </div>
          );
        })}
    </div>
  );
}
