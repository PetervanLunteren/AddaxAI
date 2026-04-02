/**
 * CropCard - detection crop thumbnail with bbox overlay and metadata.
 *
 * Shows label pill, label suggestion pill,
 * verified badge, and selection state. The crop is expanded to show
 * context around the detection, with an SVG overlay highlighting the bbox.
 */

import { memo } from "react";
import { Check } from "lucide-react";
import { API_BASE_URL } from "../../lib/api-client";
import { getDetectionColor, getDetectionDisplayName } from "../../lib/detection-utils";
import { getSpeciesTextColor } from "../../utils/species-colors";
import {
  BBOX_STROKE_WIDTH,
  BBOX_OPACITY,
  BBOX_CORNER_RADIUS,
  svgRoundedRectPath,
} from "../../lib/detection-overlay";
import { cn } from "../../lib/utils";
import { Badge } from "../ui/badge";
import type { DetectionSummary } from "../../api/types";

type TileSize = "S" | "M" | "L";

interface CropCardProps {
  detection: DetectionSummary;
  selected: boolean;
  onSelect: (detectionId: string, e: React.MouseEvent) => void;
  onDoubleClick?: (detection: DetectionSummary) => void;
  tileSize?: TileSize;
}

export const CropCard = memo(function CropCard({ detection, selected, onSelect, onDoubleClick, tileSize = "M" }: CropCardProps) {
  const isFalseDetection =
    detection.label === "false detection" && detection.verified;
  const isSuspicious =
    !isFalseDetection &&
    !detection.verified &&
    detection.neighbor_agreement != null &&
    detection.neighbor_agreement < 0.7;
  const hasSuggestion =
    !detection.verified &&
    detection.neighbor_top_label &&
    detection.neighbor_top_label !== detection.label;

  const isSmall = tileSize === "S";
  const pillSize = isSmall ? "text-[8px] px-1 py-0" : "text-[10px] px-1.5 py-0.5";
  const arrowSize = isSmall ? "text-[8px]" : "text-[10px]";

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
        <img
          src={`${API_BASE_URL}${detection.crop_url}`}
          alt={getDetectionDisplayName(detection)}
          loading="lazy"
          className="w-full h-full object-cover"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
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
      </div>

      {/* Verified badge — overflows top-right corner */}
      {detection.verified && (
        <div className="absolute -top-1.5 -right-1.5 z-10 bg-primary rounded-full p-0.5 shadow-sm">
          <Check className="h-3 w-3 text-primary-foreground" />
        </div>
      )}

      {/* Info bar — pill labels */}
      <div className="px-1.5 py-1">
        <div className="flex items-center justify-center gap-0.5 min-w-0">
          <Badge
            variant={isFalseDetection ? "secondary" : isSuspicious ? "outline" : "default"}
            className={cn(
              pillSize, "capitalize",
              hasSuggestion ? "max-w-[50%]" : "max-w-full",
              isFalseDetection && "bg-muted text-muted-foreground hover:bg-muted",
              isSuspicious && "border-transparent bg-[#882000] text-white hover:bg-[#882000]"
            )}
            style={
              !isFalseDetection && !isSuspicious
                ? { backgroundColor: getDetectionColor(detection), color: getSpeciesTextColor(detection.label || detection.category) }
                : undefined
            }
          >
            <span className="truncate">{getDetectionDisplayName(detection)}</span>
          </Badge>
          {hasSuggestion && (
            <>
              <span className={cn(arrowSize, "text-muted-foreground shrink-0")}>→</span>
              <Badge
                variant="secondary"
                className={cn(pillSize, "capitalize max-w-[50%]")}
              >
                <span className="truncate">{detection.neighbor_top_label}</span>
              </Badge>
            </>
          )}
        </div>
      </div>
    </div>
  );
});
