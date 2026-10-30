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
import { ChevronRight, Film, ImageOff } from "lucide-react";
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
import { StatusBadgeCluster } from "./StatusBadgeCluster";
import type { DetectionSummary } from "../../api/types";
import { useSpeciesColorsVersion } from "../../utils/species-colors";

type TileSize = "S" | "M" | "L";

interface CropCardProps {
  detection: DetectionSummary;
  selected: boolean;
  onSelect: (detectionId: string, e: React.MouseEvent) => void;
  onDoubleClick?: (detection: DetectionSummary) => void;
  /** Open this card's track. Absent while a track is already open, so
   *  its frames carry no badge: opening one would be opening what you
   *  are already looking at. */
  onOpenTrack?: (detection: DetectionSummary) => void;
  tileSize?: TileSize;
}

export const CropCard = memo(function CropCard({ detection, selected, onSelect, onDoubleClick, onOpenTrack, tileSize = "M" }: CropCardProps) {
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
        {/* The way into the frames the AI followed this animal through.
            A photo has no track and shows nothing; a one-frame track has
            nothing to open.

            No number: the tracker's count includes boxes too weak to be
            worth checking, which the frames view leaves out, so it would
            promise more cards than open. Counting the ones that do open
            here would copy the scope rule into the grid's SQL.

            stopPropagation on both handlers: the card root selects on
            click, so without it opening a track would also select the
            card, and shift-open would draw a range. */}
        {onOpenTrack && (detection.track_frames ?? 0) > 1 && (
          <button
            type="button"
            data-testid="open-track"
            title="Open the frames the AI followed (T)."
            onClick={(e) => {
              e.stopPropagation();
              onOpenTrack(detection);
            }}
            onDoubleClick={(e) => e.stopPropagation()}
            className={cn(
              "absolute bottom-1 left-1 flex items-center gap-1 rounded border",
              "border-white/30 bg-black/60 text-white hover:border-white/70",
              "hover:bg-black/85 focus-visible:outline-none",
              "focus-visible:ring-2 focus-visible:ring-white/70",
              isSmall ? "px-1 py-0.5 text-[7px]" : "px-1.5 py-0.5 text-[10px]",
            )}
          >
            <Film className={isSmall ? "h-2 w-2" : "h-2.5 w-2.5"} />
            {/* The word makes it read as a control. The smallest tiles
                keep the icon and chevron only: no room. */}
            {!isSmall && "Frames"}
            <ChevronRight className={isSmall ? "h-2 w-2" : "h-2.5 w-2.5"} />
          </button>
        )}
      </div>

      {/* Corner badges — the Counts cards' cluster: verified check,
          like, flag. The marks are the file's, so every crop of a
          flagged file carries the flag. */}
      <StatusBadgeCluster
        confirmed={detection.verified}
        favorited={detection.file_favorited}
        flagged={detection.file_flagged}
      />

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
