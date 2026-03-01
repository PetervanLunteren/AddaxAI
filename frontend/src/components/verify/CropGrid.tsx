/**
 * CropGrid - virtualized grid of detection crop thumbnails.
 *
 * Uses @tanstack/react-virtual for efficient rendering of large detection sets.
 * Responsive columns: 4 (sm), 6 (md), 8 (lg), 10 (xl).
 */

import { useRef, useMemo, useEffect, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
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

interface CropGridProps {
  detections: DetectionSummary[];
  selectedIds: Set<string>;
  onSelect: (detectionId: string, e: React.MouseEvent) => void;
  onCardClick: (detection: DetectionSummary) => void;
  onFindSimilar?: (detectionId: string) => void;
  onRelabel?: (detectionId: string, species: string, category: string) => void;
}

function useColumns(): number {
  const [cols, setCols] = useState(6);
  useEffect(() => {
    function update() {
      const w = window.innerWidth;
      if (w < 640) setCols(4);
      else if (w < 1024) setCols(6);
      else if (w < 1280) setCols(8);
      else setCols(10);
    }
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  return cols;
}

export function CropGrid({
  detections,
  selectedIds,
  onSelect,
  onCardClick,
  onFindSimilar,
  onRelabel,
}: CropGridProps) {
  const parentRef = useRef<HTMLDivElement>(null);
  const columns = useColumns();

  // Build rows: chunks of cards
  const rows = useMemo(() => {
    const result: DetectionSummary[][] = [];
    for (let i = 0; i < detections.length; i += columns) {
      result.push(detections.slice(i, i + columns));
    }
    return result;
  }, [detections, columns]);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 200,
    overscan: 5,
  });

  return (
    <div
      ref={parentRef}
      className="h-[calc(100vh-280px)] overflow-auto"
    >
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: "100%",
          position: "relative",
        }}
      >
        {virtualizer.getVirtualItems().map((virtualRow) => {
          const row = rows[virtualRow.index];
          return (
            <div
              key={virtualRow.index}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              <div
                className="grid gap-2 px-1"
                style={{
                  gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                }}
              >
                {row.map((det) => {
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
    </div>
  );
}
