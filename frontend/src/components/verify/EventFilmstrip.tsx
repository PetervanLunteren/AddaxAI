/**
 * Horizontal filmstrip for navigating files within an event.
 *
 * Shows small thumbnails with selection highlight and verified status.
 */

import { useEffect, useRef } from "react";
import { Check } from "lucide-react";
import { cn } from "../../lib/utils";
import { API_BASE_URL } from "../../lib/api-client";
import { getCategoryColor } from "../../lib/detection-utils";
import type { FileWithDetections } from "../../api/types";

const THUMB_W = 96;
const THUMB_H = 64;

interface EventFilmstripProps {
  files: FileWithDetections[];
  selectedIndex: number;
  onSelectIndex: (index: number) => void;
  detectionThreshold: number;
  representativeFileId: string | null;
}

export function EventFilmstrip({
  files,
  selectedIndex,
  onSelectIndex,
  detectionThreshold,
  representativeFileId,
}: EventFilmstripProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const selectedRef = useRef<HTMLButtonElement>(null);

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
          const thumbnailUrl = `${API_BASE_URL}/api/files/${file.id}/image`;
          const isSelected = index === selectedIndex;
          const isRepresentative = file.id === representativeFileId;

          return (
            <button
              key={file.id}
              ref={isSelected ? selectedRef : undefined}
              onClick={() => onSelectIndex(index)}
              className={cn(
                "relative shrink-0 w-24 h-16 rounded border-2 transition-all",
                isSelected
                  ? "border-primary ring-2 ring-primary/30"
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
              {/* Detection overlay */}
              {(() => {
                const dets = file.detections.filter(
                  (d) => d.confidence >= detectionThreshold
                );
                if (dets.length === 0) return null;
                const imgW = file.width_px || 1;
                const imgH = file.height_px || 1;
                // Compute object-cover transform
                const scale = Math.max(THUMB_W / imgW, THUMB_H / imgH);
                const dw = imgW * scale;
                const dh = imgH * scale;
                const ox = (THUMB_W - dw) / 2;
                const oy = (THUMB_H - dh) / 2;
                // Build evenodd path: outer rect + hole per detection
                let d = `M0,0H${THUMB_W}V${THUMB_H}H0Z`;
                const boxes = dets.map((det) => {
                  const bx = ox + det.bbox_x * dw;
                  const by = oy + det.bbox_y * dh;
                  const bw = det.bbox_width * dw;
                  const bh = det.bbox_height * dh;
                  const color = getCategoryColor(det.category);
                  d += `M${bx},${by}h${bw}v${bh}h${-bw}Z`;
                  return { bx, by, bw, bh, color };
                });
                return (
                  <svg
                    className="absolute inset-0 w-full h-full pointer-events-none"
                    viewBox={`0 0 ${THUMB_W} ${THUMB_H}`}
                  >
                    <path
                      fillRule="evenodd"
                      d={d}
                      fill="rgba(0,0,0,0.35)"
                    />
                    {boxes.map((b, i) => (
                      <rect
                        key={i}
                        x={b.bx}
                        y={b.by}
                        width={b.bw}
                        height={b.bh}
                        rx={2}
                        fill="none"
                        stroke={b.color}
                        strokeWidth={1}
                        opacity={0.5}
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
              {/* Representative chip */}
              {isRepresentative && (
                <span className="absolute top-0.5 left-0.5 bg-primary text-white text-[10px] leading-none font-medium px-1 py-0.5 rounded-sm">
                  Representative
                </span>
              )}
            </button>
          );
        })}
      </div>
      <div className="text-center text-xs text-muted-foreground pb-1">
        Image {selectedIndex + 1} of {files.length}
      </div>
    </div>
  );
}
