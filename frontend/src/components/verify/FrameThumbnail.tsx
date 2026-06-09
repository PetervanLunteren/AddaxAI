/**
 * FrameThumbnail - a file's thumbnail with the shared spotlight + bbox
 * overlay. One renderer for the event collage (event cards) and the event
 * filmstrip (Counts modal).
 *
 * Draws the 512px thumbnail (?size=thumb) and, when `showBoxes` is on and
 * the file's detections have loaded, a dimming spotlight with colored box
 * outlines (the SVG-mask pattern shared across the verify views). The
 * `viewBox` aspect must match the tile's rendered aspect so the boxes stay
 * aligned under object-cover.
 */

import { API_BASE_URL } from "../../lib/api-client";
import { getDetectionColor, shouldDrawBbox } from "../../lib/detection-utils";
import type { FileWithDetections } from "../../api/types";

interface FrameThumbnailProps {
  fileId: string;
  /** Detections source. Undefined while the file detail is still loading;
   *  the image renders immediately and boxes appear once it arrives. */
  file: FileWithDetections | undefined;
  detectionThreshold: number;
  /** Draw the spotlight + boxes. Default true. */
  showBoxes?: boolean;
  /** SVG viewBox; its aspect must match the tile's rendered aspect. */
  viewBox: { width: number; height: number };
  /** CSS filter applied to the image (brightness / contrast). */
  imageFilter?: string;
  className?: string;
}

export function FrameThumbnail({
  fileId,
  file,
  detectionThreshold,
  showBoxes = true,
  viewBox,
  imageFilter,
  className,
}: FrameThumbnailProps) {
  const VW = viewBox.width;
  const VH = viewBox.height;

  const dets =
    file && showBoxes
      ? file.detections.filter((d) =>
          shouldDrawBbox(d, file, detectionThreshold),
        )
      : [];

  let boxes: Array<{
    id: string;
    bx: number;
    by: number;
    bw: number;
    bh: number;
    color: string;
  }> = [];
  if (file && dets.length > 0) {
    const imgW = file.width_px || 1;
    const imgH = file.height_px || 1;
    const scale = Math.max(VW / imgW, VH / imgH);
    const dw = imgW * scale;
    const dh = imgH * scale;
    const ox = (VW - dw) / 2;
    const oy = (VH - dh) / 2;
    boxes = dets.map((det) => ({
      id: det.id,
      bx: ox + det.bbox_x * dw,
      by: oy + det.bbox_y * dh,
      bw: det.bbox_width * dw,
      bh: det.bbox_height * dh,
      color: getDetectionColor(det),
    }));
  }

  const maskId = `m-frame-${fileId}`;

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
          are on) so empty frames dim uniformly, the same baseline used by
          the event-card collage. */}
      {file && showBoxes && (
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox={`0 0 ${VW} ${VH}`}
        >
          <defs>
            <mask id={maskId}>
              <rect width={VW} height={VH} fill="white" />
              {boxes.map((b) => (
                <rect
                  key={b.id}
                  x={b.bx}
                  y={b.by}
                  width={b.bw}
                  height={b.bh}
                  rx={4}
                  fill="black"
                />
              ))}
            </mask>
          </defs>
          <rect
            width={VW}
            height={VH}
            fill="rgba(0,0,0,0.55)"
            mask={`url(#${maskId})`}
          />
          {boxes.map((b) => (
            <rect
              key={b.id}
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
      )}
    </div>
  );
}
