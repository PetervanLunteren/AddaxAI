/**
 * CropCard - detection crop thumbnail with bbox overlay and metadata.
 *
 * Shows the label pill, the verified badge (top-right), and the
 * selection state. The crop is expanded to show context around the
 * detection, with an SVG overlay highlighting the bbox. The "neighbours
 * disagree" signal is workflow-driven, not decorative: it surfaces via
 * the suggestions sort + cohort divider (bulk-accept descendant
 * promotions) and the right-click "Relabel to X" context menu in
 * CropGrid, rather than passively decorating the tile.
 */

import { memo, useState } from "react";
import { Check, ImageOff } from "lucide-react";
import { API_BASE_URL } from "../../lib/api-client";
import {
  getDetectionColor,
  getDetectionDisplayName,
  isNonLabel,
} from "../../lib/detection-utils";
import { getContrastTextColor } from "../../utils/species-colors";
import {
  BBOX_STROKE_WIDTH,
  BBOX_OPACITY,
  BBOX_CORNER_RADIUS,
  svgRoundedRectPath,
} from "../../lib/detection-overlay";
import { cn } from "../../lib/utils";
import { reportMissingMedia } from "../../hooks/useBrokenDeployments";
import { Badge } from "../ui/badge";
import type { DetectionSummary } from "../../api/types";
import { useSpeciesColorsVersion } from "../../utils/species-colors";

type TileSize = "S" | "M" | "L";

interface CropCardProps {
  detection: DetectionSummary;
  selected: boolean;
  onSelect: (detectionId: string, e: React.MouseEvent) => void;
  onDoubleClick?: (detection: DetectionSummary) => void;
  tileSize?: TileSize;
}

export const CropCard = memo(function CropCard({ detection, selected, onSelect, onDoubleClick, tileSize = "M" }: CropCardProps) {
  // Repaint when the project's colour map lands or changes.
  useSpeciesColorsVersion();
  // The shared rule, not the one string: pressing X writes
  // "false detection", but the same tile should read the same way for
  // any of the six "nothing here" labels a person can apply.
  const isFalseDetection = isNonLabel(detection.label) && detection.verified;

  const [imageFailed, setImageFailed] = useState(false);
  const isSmall = tileSize === "S";
  const pillSize = isSmall
    ? "text-[7px] px-0.5 py-0 rounded-sm"
    : "text-[10px] px-1.5 py-0.5 rounded-sm";

  return (
    <div
      className={cn(
        "relative group cursor-pointer rounded-lg border bg-card text-card-foreground transition-[box-shadow,transform] duration-150",
        "hover:-translate-y-0.5 hover:shadow-md",
        selected && "ring-2 ring-offset-2 ring-[#0f6064]"
      )}
      onClick={(e) => onSelect(detection.detection_id, e)}
      onDoubleClick={(e) => { e.stopPropagation(); onDoubleClick?.(detection); }}
    >
      {/* Crop image */}
      <div className="aspect-square bg-muted relative overflow-hidden rounded-t-lg">
        {imageFailed ? (
          // The shimmer below is a *loading* signal and it used to survive a
          // failed image, so a tile whose photo had gone missing pulsed for
          // ever and read as "still loading". Users waited instead of
          // reconnecting the folder. Say it is not coming instead. The bbox
          // overlay goes too: its dark mask over an empty tile looks like a
          // very dark photo rather than an absent one.
          <div className="absolute inset-0 flex items-center justify-center bg-neutral-200 dark:bg-neutral-800">
            <ImageOff
              className={cn(
                "text-neutral-400 dark:text-neutral-500",
                isSmall ? "h-4 w-4" : "h-6 w-6",
              )}
            />
          </div>
        ) : (
          <>
            <img
              src={`${API_BASE_URL}${detection.crop_url}`}
              alt={getDetectionDisplayName(detection)}
              loading="lazy"
              className="w-full h-full object-cover"
              onError={() => {
                setImageFailed(true);
                reportMissingMedia(detection.deployment_id);
              }}
            />
            {/* Bbox overlay */}
            {detection.crop_bbox && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox="0 0 200 200"
              >
                <path
                  fillRule="evenodd"
                  d={`M0,0H200V200H0Z` + svgRoundedRectPath(
                    detection.crop_bbox.x * 200,
                    detection.crop_bbox.y * 200,
                    detection.crop_bbox.w * 200,
                    detection.crop_bbox.h * 200,
                    BBOX_CORNER_RADIUS
                  )}
                  fill="rgba(0, 0, 0, 0.6)"
                />
                <rect
                  x={detection.crop_bbox.x * 200}
                  y={detection.crop_bbox.y * 200}
                  width={detection.crop_bbox.w * 200}
                  height={detection.crop_bbox.h * 200}
                  rx={BBOX_CORNER_RADIUS}
                  fill="none"
                  stroke={getDetectionColor(detection)}
                  strokeWidth={BBOX_STROKE_WIDTH}
                  opacity={BBOX_OPACITY}
                />
              </svg>
            )}
            {/* Loading shimmer placeholder */}
            <div className="absolute inset-0 bg-gradient-to-r from-muted via-muted-foreground/5 to-muted animate-pulse -z-10" />
          </>
        )}
      </div>

      {/* Verified badge — overflows top-right corner */}
      {detection.verified && (
        <div className="absolute -top-1.5 -right-1.5 z-10 bg-primary rounded-full p-0.5 shadow-sm">
          <Check className="h-3 w-3 text-primary-foreground" />
        </div>
      )}

      {/* Info bar — pill labels */}
      <div className="px-2 py-1.5">
        <div className="flex items-center justify-center gap-0.5 min-w-0">
          <Badge
            variant={isFalseDetection ? "secondary" : "default"}
            className={cn(
              pillSize, "capitalize max-w-full",
              isFalseDetection && "bg-muted text-muted-foreground hover:bg-muted",
            )}
            style={(() => {
              if (isFalseDetection) return undefined;
              // Derive the foreground from the ACTUAL background so a
              // bright chip never gets unreadable white text. The bg
              // here uses (label_taxonomy_id || label || category) via
              // getDetectionColor; the text now follows that exactly.
              const bg = getDetectionColor(detection);
              return { backgroundColor: bg, color: getContrastTextColor(bg) };
            })()}
          >
            <span className="truncate">{getDetectionDisplayName(detection)}</span>
          </Badge>
        </div>
      </div>
    </div>
  );
});
