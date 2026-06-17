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

import { API_BASE_URL } from "../../lib/api-client";
import { getDetectionColor, shouldDrawBbox } from "../../lib/detection-utils";
import { SpotlightDim } from "./SpotlightDim";
import type { FileWithDetections } from "../../api/types";

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
  const dets =
    file && showBoxes
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
      className={`relative overflow-hidden bg-muted h-full w-full ${className ?? ""}`}
    >
      <img
        src={`${API_BASE_URL}/api/files/${fileId}/image?size=thumb`}
        alt=""
        className="w-full h-full object-cover"
        style={imageFilter ? { filter: imageFilter } : undefined}
        onError={(e) => {
          (e.target as HTMLImageElement).style.display = "none";
        }}
      />
      {/* Spotlight + outlines. Rendered once `file` has loaded (and boxes
          are on) so empty frames dim uniformly. */}
      {file && showBoxes && (
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
