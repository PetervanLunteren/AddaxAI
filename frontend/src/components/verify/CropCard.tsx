/**
 * CropCard - detection crop thumbnail with bbox overlay and metadata.
 *
 * Shows species, confidence, similarity score,
 * verified badge, and selection state. The crop is expanded to show
 * context around the detection, with an SVG overlay highlighting the bbox.
 */

import { Check, Circle } from "lucide-react";
import { API_BASE_URL } from "../../lib/api-client";
import { getCategoryColor } from "../../lib/detection-utils";
import {
  BBOX_STROKE_WIDTH,
  BBOX_OPACITY,
  BBOX_CORNER_RADIUS,
  svgRoundedRectPath,
} from "../../lib/detection-overlay";
import { cn } from "../../lib/utils";
import type { DetectionSummary } from "../../api/types";

interface CropCardProps {
  detection: DetectionSummary;
  selected: boolean;
  onClick: (e: React.MouseEvent) => void;
}

export function CropCard({ detection, selected, onClick }: CropCardProps) {
  const score =
    detection.similarity != null
      ? `${(detection.similarity * 100).toFixed(0)}%`
      : null;

  const agreement = detection.neighbor_agreement;
  const agreementStyle =
    agreement != null
      ? agreement >= 0.5
        ? selected
          ? { boxShadow: "0 0 0 2px #0f6064, 0 2px 10px rgba(15,96,100,0.35)", background: "rgba(15,96,100,0.10)" }
          : { boxShadow: "0 2px 10px rgba(15,96,100,0.35)", background: "rgba(15,96,100,0.10)" }
        : selected
          ? { boxShadow: "0 0 0 2px #0f6064, 0 2px 10px rgba(136,32,0,0.35)", background: "rgba(136,32,0,0.10)" }
          : { boxShadow: "0 2px 10px rgba(136,32,0,0.35)", background: "rgba(136,32,0,0.10)" }
      : undefined;

  return (
    <div
      className={cn(
        "relative group cursor-pointer rounded-lg overflow-hidden bg-muted transition-all",
        "hover:-translate-y-0.5",
        !agreementStyle && "hover:shadow-md",
        selected && !agreementStyle && "ring-2 ring-offset-2 ring-[#0f6064]"
      )}
      style={agreementStyle}
      onClick={onClick}
    >
      {/* Crop image */}
      <div className="aspect-square bg-muted relative">
        <img
          src={`${API_BASE_URL}${detection.crop_url}`}
          alt={detection.species || detection.category}
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
              stroke={getCategoryColor(detection.category)}
              strokeWidth={BBOX_STROKE_WIDTH}
              opacity={BBOX_OPACITY}
            />
          </svg>
        )}
        {/* Loading shimmer placeholder */}
        <div className="absolute inset-0 bg-gradient-to-r from-muted via-muted-foreground/5 to-muted animate-pulse -z-10" />
      </div>

      {/* Verified badge */}
      <div className="absolute top-1.5 right-1.5">
        {detection.verified ? (
          <div className="bg-primary rounded-full p-0.5 shadow-sm">
            <Check className="h-3 w-3 text-primary-foreground" />
          </div>
        ) : (
          <Circle className="h-4 w-4 text-muted-foreground/40" />
        )}
      </div>

      {/* Info bar */}
      <div className="px-2 py-1.5 space-y-0.5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium truncate capitalize max-w-[100px]">
            {detection.species || detection.category}
          </span>
          <span className="text-[10px] text-muted-foreground">
            {(detection.confidence * 100).toFixed(0)}%
          </span>
        </div>
        {detection.neighbor_top_label &&
          detection.neighbor_top_label !== detection.species && (
            <div className="text-[10px] text-amber-600 dark:text-amber-400 truncate capitalize">
              → {detection.neighbor_top_label}
            </div>
          )}
        {score && (
          <div className="text-[10px] text-muted-foreground">{score}</div>
        )}
      </div>
    </div>
  );
}
