/**
 * Horizontal filmstrip for navigating files within an event.
 *
 * Shows small thumbnails with selection highlight and verified status.
 */

import { useEffect, useMemo, useRef } from "react";
import { Check } from "lucide-react";
import { cn } from "../../lib/utils";
import { API_BASE_URL } from "../../lib/api-client";
import { getDetectionColor } from "../../lib/detection-utils";
import { getSpeciesColor, getSpeciesTextColor } from "../../utils/species-colors";
import type { FileWithDetections, MaxNFrame } from "../../api/types";

const THUMB_W = 96;
const THUMB_H = 64;

interface EventFilmstripProps {
  files: FileWithDetections[];
  selectedIndex: number;
  onSelectIndex: (index: number, shiftKey: boolean) => void;
  detectionThreshold: number;
  maxNFrames: MaxNFrame[];
  bulkSelection?: Set<number>;
}

export function EventFilmstrip({
  files,
  selectedIndex,
  onSelectIndex,
  detectionThreshold,
  maxNFrames,
  bulkSelection,
}: EventFilmstripProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const selectedRef = useRef<HTMLButtonElement>(null);

  // Build lookup: file_id -> list of MaxN labels for that file
  const maxNByFile = useMemo(() => {
    const map = new Map<string, MaxNFrame[]>();
    for (const frame of maxNFrames) {
      const existing = map.get(frame.file_id);
      if (existing) existing.push(frame);
      else map.set(frame.file_id, [frame]);
    }
    return map;
  }, [maxNFrames]);

  // Count video File rows in this event. Post-2026-05 each video is
  // its own File row (no separate per-frame rows), so this is just a
  // type filter.
  const videoCount = useMemo(
    () => files.filter((f) => f.file_type === "video").length,
    [files],
  );

  // Scroll selected thumbnail into view
  useEffect(() => {
    selectedRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
      inline: "center",
    });
  }, [selectedIndex]);

  return (
    <div className="border-t bg-white shrink-0">
      <div
        ref={scrollRef}
        className="flex items-center gap-1.5 px-4 py-2 overflow-x-auto"
      >
        {files.map((file, index) => {
          const thumbnailUrl = `${API_BASE_URL}/api/files/${file.id}/image?size=thumb`;
          const isSelected = index === selectedIndex;
          const fileMaxNFrames = maxNByFile.get(file.id);

          return (
            <button
              key={file.id}
              ref={isSelected ? selectedRef : undefined}
              onClick={(e) => onSelectIndex(index, e.shiftKey)}
              className={cn(
                "relative shrink-0 w-24 h-16 rounded border-2 transition-all",
                isSelected
                  ? "border-primary ring-2 ring-primary/30"
                  : bulkSelection?.has(index)
                    ? "border-primary/60 ring-1 ring-primary/20"
                    : "border-transparent hover:border-gray-300 opacity-75"
              )}
            >
              <img
                src={thumbnailUrl}
                alt={`File ${index + 1}`}
                className="w-full h-full object-cover rounded-sm"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = "none";
                }}
              />
              {/* Spotlight + detection overlay. Rendered for every
                  filmstrip thumbnail, including empties: the dim
                  layer keeps brightness uniform across the strip.
                  When there are no detections the mask has no holes,
                  so the layer covers the full image. */}
              {(() => {
                let dets = file.detections.filter(
                  (d) => d.confidence >= detectionThreshold
                );
                // For videos, only show detections from the best frame
                if (file.file_type === "video" && file.best_frame_number != null) {
                  dets = dets.filter((d) => d.frame_number === file.best_frame_number);
                }
                const imgW = file.width_px || 1;
                const imgH = file.height_px || 1;
                // Compute object-cover transform
                const scale = Math.max(THUMB_W / imgW, THUMB_H / imgH);
                const dw = imgW * scale;
                const dh = imgH * scale;
                const ox = (THUMB_W - dw) / 2;
                const oy = (THUMB_H - dh) / 2;
                const maskId = `m-film-${file.id}`;
                const boxes = dets.map((det) => {
                  const bx = ox + det.bbox_x * dw;
                  const by = oy + det.bbox_y * dh;
                  const bw = det.bbox_width * dw;
                  const bh = det.bbox_height * dh;
                  const color = getDetectionColor(det);
                  return { bx, by, bw, bh, color };
                });
                return (
                  <svg
                    className="absolute inset-0 w-full h-full pointer-events-none"
                    viewBox={`0 0 ${THUMB_W} ${THUMB_H}`}
                  >
                    <defs>
                      <mask id={maskId}>
                        <rect width={THUMB_W} height={THUMB_H} fill="white" />
                        {boxes.map((b, i) => (
                          <rect key={i} x={b.bx} y={b.by} width={b.bw} height={b.bh} rx={4} fill="black" />
                        ))}
                      </mask>
                    </defs>
                    <rect width={THUMB_W} height={THUMB_H} fill="rgba(0,0,0,0.55)" mask={`url(#${maskId})`} />
                    {boxes.map((b, i) => (
                      <rect
                        key={i}
                        x={b.bx}
                        y={b.by}
                        width={b.bw}
                        height={b.bh}
                        rx={4}
                        fill="none"
                        stroke={b.color}
                        strokeWidth={2}
                        opacity={1}
                      />
                    ))}
                  </svg>
                );
              })()}
              {/* Verified badge */}
              {file.verified && (
                <div className="absolute -top-1.5 -right-1.5 bg-primary rounded-full p-0.5">
                  <Check className="h-2.5 w-2.5 text-white" />
                </div>
              )}
              {/* MaxN badges */}
              {fileMaxNFrames && (
                <div className="absolute top-0.5 left-0.5 flex flex-col gap-0.5">
                  {fileMaxNFrames.map((frame) => (
                    <span
                      key={frame.label}
                      className="text-[9px] leading-none font-semibold px-1 py-0.5 rounded-sm shadow-sm"
                      style={{ backgroundColor: getSpeciesColor(frame.label_taxonomy_id || frame.label || ""), color: getSpeciesTextColor(frame.label_taxonomy_id || frame.label || "") }}
                    >
                      MaxN
                    </span>
                  ))}
                </div>
              )}
              {/* Bulk selection overlay */}
              {bulkSelection?.has(index) && !isSelected && (
                <div className="absolute inset-0 bg-primary/15 rounded-sm pointer-events-none" />
              )}
            </button>
          );
        })}
      </div>
      <div className="text-center text-xs text-muted-foreground pb-1">
        {bulkSelection && bulkSelection.size > 1
          ? `${bulkSelection.size} files selected`
          : videoCount === 0
            ? `Image ${selectedIndex + 1} of ${files.length}`
            : `Frame ${selectedIndex + 1} of ${files.length} · ${videoCount} video${videoCount !== 1 ? "s" : ""}`}
      </div>
    </div>
  );
}
