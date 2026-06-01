/**
 * FileCard - single-file tile for the Files verify tab.
 *
 * Shows the file's thumbnail (best_frame for videos via /api/files/{id}/image),
 * detection overlay, observation_type + label badges, date/site, and the
 * status badge cluster (verified / favorited / flagged) clipped to the
 * card's top-right corner. Clicking opens FileDetailModal.
 */

import { Image as ImageIcon, Video as VideoIcon } from "lucide-react";
import { API_BASE_URL } from "../../lib/api-client";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { getCategoryColor, getCategoryTextColor, getDetectionColor, shouldDrawBbox } from "../../lib/detection-utils";
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

  const thumbnailUrl = `${API_BASE_URL}/api/files/${file.id}/image?size=thumb`;
  const dets = file.detections.filter((d) =>
    shouldDrawBbox(d, file, detectionThreshold),
  );

  // Observation-level "verified" for the badge: the file reads as
  // verified when every reviewable detection on it is verified, so the
  // badge agrees with the project-wide "% observations verified" metric.
  // Empty files have no observations to verify, so they fall back to the
  // file's own verified flag (set by a whole-frame review).
  const reviewable = file.detections.filter(
    (d) => d.confidence >= detectionThreshold || d.verified,
  );
  const observationsVerified =
    reviewable.length > 0
      ? reviewable.every((d) => d.verified)
      : file.verified;

  // Chips mirror the boxes drawn above (`dets`), so what's labelled
  // matches what's outlined — including unclassified animals and the
  // person / vehicle boxes that never carry a species label. A box with
  // a species label becomes a species chip; a box without one becomes an
  // observation badge for its category (Animal / Person / Vehicle).
  const drawnSpecies = [
    ...new Set(dets.filter((d) => d.label).map((d) => d.label as string)),
  ];
  const drawnCategories = [
    ...new Set(dets.filter((d) => !d.label).map((d) => d.category)),
  ];

  return (
    <Card
      className="relative hover:shadow-lg transition-shadow cursor-pointer"
      onClick={onClick}
    >
      <StatusBadgeCluster
        verified={observationsVerified}
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
        {/* Label chips mirror the boxes drawn above: a category badge
            (Animal / Person / Vehicle) for each unlabelled box, then the
            top two species labels with "+N" for the rest. "Empty" only
            when nothing is boxed. [&>*]:rounded-sm overrides the Badge
            default rounded-full so the chips read as rectangles. */}
        <div className="flex flex-wrap gap-1 [&>*]:rounded-sm">
          {drawnCategories.map((cat) => (
            <Badge
              key={cat}
              variant="default"
              className="text-[10px] px-1.5 py-0.5"
              style={{
                backgroundColor: getCategoryColor(cat),
                color: getCategoryTextColor(cat),
              }}
            >
              {cat.charAt(0).toUpperCase() + cat.slice(1)}
            </Badge>
          ))}
          {drawnSpecies.slice(0, 2).map((sp) => (
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
                {file.display_labels?.[sp] ||
                  sp.charAt(0).toUpperCase() + sp.slice(1)}
              </span>
            </Badge>
          ))}
          {drawnSpecies.length > 2 && (
            <Badge variant="default" className="text-[10px] px-1.5 py-0.5">
              +{drawnSpecies.length - 2}
            </Badge>
          )}
          {dets.length === 0 && (
            <Badge
              variant="outline"
              className="text-[10px] px-1.5 py-0.5 border-muted-foreground/40 text-muted-foreground"
            >
              Empty
            </Badge>
          )}
        </div>

        {file.site_name && (
          <div className="text-sm font-medium truncate">{file.site_name}</div>
        )}
        <div className="flex items-center justify-between gap-1.5 text-xs text-muted-foreground">
          <span>
            {dateStr} · {timeStr}
          </span>
          <span className="inline-flex shrink-0 items-center gap-1 rounded-sm border border-muted-foreground/40 px-1.5 py-0.5 text-[10px] font-medium">
            {file.file_type === "video" ? (
              <>
                <VideoIcon className="h-3 w-3" />
                Video
              </>
            ) : (
              <>
                <ImageIcon className="h-3 w-3" />
                Image
              </>
            )}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
