/**
 * FileCard - single-file tile for the Files verify tab.
 *
 * Shows the file's thumbnail (best_frame for videos via /api/files/{id}/image),
 * detection overlay, observation_type + label badges, date/site, and the
 * status badge cluster (verified / favorited / flagged) clipped to the
 * card's top-right corner. Clicking opens FileDetailModal.
 */

import { Video as VideoIcon } from "lucide-react";
import { API_BASE_URL } from "../../lib/api-client";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { getDetectionColor, getObservationBadge } from "../../lib/detection-utils";
import { getSpeciesColor, getSpeciesTextColor } from "../../utils/species-colors";
import { Badge } from "../ui/badge";
import { Card, CardContent } from "../ui/card";
import { StatusBadgeCluster } from "./StatusBadgeCluster";
import type { FileSummary } from "../../api/types";

interface FileCardProps {
  file: FileSummary;
  detectionThreshold: number;
  onClick: () => void;
}

export function FileCard({ file, detectionThreshold, onClick }: FileCardProps) {
  const dateStr = formatCameraDate(file.captured_at_local, {
    month: "short",
    day: "numeric",
  });
  const timeStr = formatCameraTime(file.captured_at_local);

  const thumbnailUrl = `${API_BASE_URL}/api/files/${file.id}/image`;
  const dets = file.detections.filter((d) => d.confidence >= detectionThreshold);

  // Drop species chips whose display name duplicates an observation
  // badge that's already on this tile (Person / Vehicle). Without this,
  // a vehicle file shows two "Vehicle" chips: the observation badge and
  // the redundant species chip.
  const observationBadgeNames = new Set(
    file.observation_types
      .filter((t) => t === "human" || t === "vehicle")
      .map((t) => (t === "human" ? "person" : t)),
  );
  const speciesLabels = file.labels.filter((sp) => {
    const display = (file.display_labels?.[sp] || sp).toLowerCase();
    return !observationBadgeNames.has(display);
  });

  return (
    <Card
      className="relative hover:shadow-lg transition-shadow cursor-pointer"
      onClick={onClick}
    >
      <StatusBadgeCluster
        verified={file.verified}
        favorited={file.favorited}
        flagged={file.flagged}
      />
      <div className="aspect-video bg-muted relative overflow-hidden rounded-t-lg">
        <img
          src={thumbnailUrl}
          alt="File thumbnail"
          className="w-full h-full object-cover"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />

        {/* Video-frame marker — this still was extracted from a video clip */}
        {file.source_video_id != null && (
          <div className="absolute top-2 left-2 bg-black/55 text-white rounded px-1.5 py-0.5 text-[10px] flex items-center gap-1">
            <VideoIcon className="h-3 w-3" />
            Video frame
          </div>
        )}

        {/* Spotlight + detection overlay. Rendered for every tile,
            including empties: the dim layer keeps brightness uniform
            across the grid. When there are no detections the mask has
            no holes, so the layer covers the full image. */}
        {(() => {
          const imgW = file.width_px || 1;
          const imgH = file.height_px || 1;
          const VW = 320;
          const VH = 180;
          const scale = Math.max(VW / imgW, VH / imgH);
          const dw = imgW * scale;
          const dh = imgH * scale;
          const ox = (VW - dw) / 2;
          const oy = (VH - dh) / 2;
          const maskId = `m-file-card-${file.id}`;
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
              viewBox={`0 0 ${VW} ${VH}`}
            >
              <defs>
                <mask id={maskId}>
                  <rect width={VW} height={VH} fill="white" />
                  {boxes.map((b, i) => (
                    <rect key={i} x={b.bx} y={b.by} width={b.bw} height={b.bh} rx={4} fill="black" />
                  ))}
                </mask>
              </defs>
              <rect width={VW} height={VH} fill="rgba(0,0,0,0.55)" mask={`url(#${maskId})`} />
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
                  strokeWidth={2.5}
                />
              ))}
            </svg>
          );
        })()}

      </div>

      <CardContent className="p-3 space-y-1.5">
        {/* Label chips — moved out of the thumbnail so the image is
            unobstructed. Observation-type chips first (human/vehicle),
            then the top two species labels. "+N" when more. "Empty"
            when there are no labels on this file. [&>*]:rounded-sm
            overrides the Badge default rounded-full so the chips read
            as rectangles, not pills. */}
        <div className="flex flex-wrap gap-1 [&>*]:rounded-sm">
          {file.observation_types
            .filter((t) => t === "human" || t === "vehicle")
            .map((t) => {
              const badge = getObservationBadge(t);
              return (
                <Badge
                  key={t}
                  variant="outline"
                  className={`text-[10px] px-1.5 py-0.5 ${badge.className}`}
                  style={badge.style}
                >
                  {badge.label}
                </Badge>
              );
            })}
          {speciesLabels.length > 0 ? (
            <>
              {speciesLabels.slice(0, 2).map((sp) => (
                <Badge
                  key={sp}
                  variant="default"
                  className="text-[10px] px-1.5 py-0.5 max-w-[100px]"
                  style={{
                    backgroundColor: getSpeciesColor(sp),
                    color: getSpeciesTextColor(sp),
                  }}
                >
                  <span className="truncate">
                    {file.display_labels?.[sp] || sp.charAt(0).toUpperCase() + sp.slice(1)}
                  </span>
                </Badge>
              ))}
              {speciesLabels.length > 2 && (
                <Badge variant="default" className="text-[10px] px-1.5 py-0.5">
                  +{speciesLabels.length - 2}
                </Badge>
              )}
            </>
          ) : (
            observationBadgeNames.size === 0 && (
              <Badge
                variant="outline"
                className="text-[10px] px-1.5 py-0.5 border-muted-foreground/40 text-muted-foreground"
              >
                Empty
              </Badge>
            )
          )}
        </div>

        {file.site_name && (
          <div className="text-sm font-medium truncate">{file.site_name}</div>
        )}
        <div className="text-xs text-muted-foreground">
          {dateStr} · {timeStr}
        </div>
      </CardContent>
    </Card>
  );
}
