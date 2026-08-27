/**
 * FrameThumbnail - a file's thumbnail with the shared spotlight + bbox
 * overlay. One renderer for the event collage (event cards) and the event
 * filmstrip (Counts modal).
 *
 * Draws the 512px thumbnail (?size=thumb) and, when `showBoxes` is on and the
 * file's detections have loaded, a dimming spotlight with colored box
 * outlines. The SVG draws in the image's own pixel space with
 * preserveAspectRatio="slice" (SVG's object-cover), so the boxes line up with
 * the object-cover image at any tile aspect, with no manual fitting math.
 */

import { useState } from "react";
import { ImageOff } from "lucide-react";
import { API_BASE_URL } from "../../lib/api-client";
import { getDetectionColor, shouldDrawBbox } from "../../lib/detection-utils";
import { reportMissingMedia } from "../../hooks/useBrokenDeployments";
import { SpotlightDim } from "./SpotlightDim";
import type { FileWithDetections } from "../../api/types";
import { useSpeciesColorsVersion } from "../../utils/species-colors";

interface FrameThumbnailProps {
  fileId: string;
  /** Detections source. Undefined while the file detail is still loading;
   *  the image renders immediately and boxes appear once it arrives. */
  file: FileWithDetections | undefined;
  detectionThreshold: number;
  /** Draw the spotlight + boxes. Default true. */
  showBoxes?: boolean;
  /** CSS filter applied to the image (brightness / contrast). */
  imageFilter?: string;
  className?: string;
}

export function FrameThumbnail({
  fileId,
  file,
  detectionThreshold,
  showBoxes = true,
  imageFilter,
  className,
}: FrameThumbnailProps) {
  // Repaint when the project's colour map lands or changes.
  useSpeciesColorsVersion();
  // A thumbnail that 404s leaves the neutral grey behind, and boxes drawn
  // on that grey read as a very dark photo with animals in it rather than
  // as an absent one. Say the picture is gone instead of decorating a
  // tile that has nothing under it.
  const [imageFailed, setImageFailed] = useState(false);

  const dets =
    file && showBoxes && !imageFailed
      ? file.detections.filter((d) =>
          shouldDrawBbox(d, file, detectionThreshold),
        )
      : [];

  // Draw in the image's pixel space; the SVG slices it to the tile exactly
  // like the image's object-cover.
  const imgW = file?.width_px || 1;
  const imgH = file?.height_px || 1;
  const rx = Math.round(Math.min(imgW, imgH) * 0.02);

  return (
    <div
      // Neutral grey fallback, not bg-muted: the muted token is hue 210 (a
      // blue-grey), which shows through as a blue "cast" on tiles whose
      // thumbnail image hasn't painted yet (a slow load, or a file whose
      // image is gone and shows the ImageOff tile below).
      className={`relative overflow-hidden bg-neutral-200 dark:bg-neutral-800 h-full w-full ${className ?? ""}`}
    >
      {imageFailed ? (
        <div className="absolute inset-0 flex items-center justify-center">
          <ImageOff className="h-6 w-6 text-neutral-400 dark:text-neutral-500" />
        </div>
      ) : (
        <img
          src={`${API_BASE_URL}/api/files/${fileId}/image?size=thumb`}
          alt=""
          className="w-full h-full object-cover"
          style={imageFilter ? { filter: imageFilter } : undefined}
          onError={() => {
            setImageFailed(true);
            reportMissingMedia(file?.deployment_id);
          }}
        />
      )}
      {/* Spotlight + outlines. Rendered once `file` has loaded (and boxes
          are on) so empty frames dim uniformly. */}
      {file && showBoxes && !imageFailed && (
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox={`0 0 ${imgW} ${imgH}`}
          preserveAspectRatio="xMidYMid slice"
        >
          <SpotlightDim
            width={imgW}
            height={imgH}
            rx={rx}
            fill="rgba(0,0,0,0.55)"
            boxes={dets.map((d) => ({
              x: d.bbox_x * imgW,
              y: d.bbox_y * imgH,
              width: d.bbox_width * imgW,
              height: d.bbox_height * imgH,
            }))}
          />
          {dets.map((d) => (
            <rect
              key={d.id}
              x={d.bbox_x * imgW}
              y={d.bbox_y * imgH}
              width={d.bbox_width * imgW}
              height={d.bbox_height * imgH}
              rx={rx}
              fill="none"
              stroke={getDetectionColor(d)}
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      )}
    </div>
  );
}
