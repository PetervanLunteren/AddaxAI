/**
 * EventCollage - thumbnail area for event cards on the Verify page.
 *
 * Renders 1 to 4 representative frames inside a fixed 16:9 area:
 *   1: full card
 *   2: two equal halves, left/right
 *   3: 2 top tiles + 1 wide bottom
 *   4: 2x2
 *
 * Each tile fetches its file detail via the shared ["file", fileId]
 * query so detection bboxes can be drawn with the same SVG mask
 * pattern used by FileCard. The viewBox of each tile matches the
 * tile's pixel aspect ratio so bboxes stay aligned in every layout.
 */

import { useQuery } from "@tanstack/react-query";
import { Layers } from "lucide-react";

import { filesApi } from "../../api/files";
import { API_BASE_URL } from "../../lib/api-client";
import { getDetectionColor } from "../../lib/detection-utils";

interface EventCollageProps {
  /** Up to four file IDs from EventSummary.collage_file_ids. Empty
   *  renders the same placeholder as today's empty-thumbnail card. */
  fileIds: string[];
  detectionThreshold: number;
}

const FRAME_CLASSES =
  "aspect-video bg-muted relative overflow-hidden rounded-t-lg";

export function EventCollage({ fileIds, detectionThreshold }: EventCollageProps) {
  if (fileIds.length === 0) {
    return (
      <div className={FRAME_CLASSES}>
        <div className="flex items-center justify-center h-full">
          <Layers className="h-8 w-8 text-muted-foreground/30" />
        </div>
      </div>
    );
  }

  if (fileIds.length === 1) {
    return (
      <div className={FRAME_CLASSES}>
        <EventCollageTile
          fileId={fileIds[0]}
          detectionThreshold={detectionThreshold}
          viewBox={{ width: 320, height: 180 }}
        />
      </div>
    );
  }

  if (fileIds.length === 2) {
    return (
      <div className={`${FRAME_CLASSES} grid grid-cols-2 gap-0.5`}>
        {fileIds.map((id) => (
          <EventCollageTile
            key={id}
            fileId={id}
            detectionThreshold={detectionThreshold}
            viewBox={{ width: 160, height: 180 }}
          />
        ))}
      </div>
    );
  }

  if (fileIds.length === 3) {
    return (
      <div className={`${FRAME_CLASSES} grid grid-cols-2 grid-rows-2 gap-0.5`}>
        <EventCollageTile
          fileId={fileIds[0]}
          detectionThreshold={detectionThreshold}
          viewBox={{ width: 160, height: 90 }}
        />
        <EventCollageTile
          fileId={fileIds[1]}
          detectionThreshold={detectionThreshold}
          viewBox={{ width: 160, height: 90 }}
        />
        <EventCollageTile
          fileId={fileIds[2]}
          detectionThreshold={detectionThreshold}
          viewBox={{ width: 320, height: 90 }}
          className="col-span-2"
        />
      </div>
    );
  }

  return (
    <div className={`${FRAME_CLASSES} grid grid-cols-2 grid-rows-2 gap-0.5`}>
      {fileIds.slice(0, 4).map((id) => (
        <EventCollageTile
          key={id}
          fileId={id}
          detectionThreshold={detectionThreshold}
          viewBox={{ width: 160, height: 90 }}
        />
      ))}
    </div>
  );
}

interface EventCollageTileProps {
  fileId: string;
  detectionThreshold: number;
  viewBox: { width: number; height: number };
  className?: string;
}

function EventCollageTile({
  fileId,
  detectionThreshold,
  viewBox,
  className,
}: EventCollageTileProps) {
  const { data: file } = useQuery({
    queryKey: ["file", fileId],
    queryFn: ({ signal }) => filesApi.get(fileId, { signal }),
  });

  const VW = viewBox.width;
  const VH = viewBox.height;

  const dets = file
    ? file.detections.filter((d) => d.confidence >= detectionThreshold)
    : [];

  let boxes: Array<{
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
      bx: ox + det.bbox_x * dw,
      by: oy + det.bbox_y * dh,
      bw: det.bbox_width * dw,
      bh: det.bbox_height * dh,
      color: getDetectionColor(det),
    }));
  }

  const maskId = `m-tile-${fileId}`;

  return (
    <div className={`relative overflow-hidden bg-muted ${className ?? ""}`}>
      <img
        src={`${API_BASE_URL}/api/files/${fileId}/image`}
        alt=""
        className="w-full h-full object-cover"
        onError={(e) => {
          (e.target as HTMLImageElement).style.display = "none";
        }}
      />
      {/* Spotlight + bbox outlines. Always rendered once `file` has
          loaded so the dim layer keeps every tile in the collage at
          the same brightness baseline, even when a tile is empty. */}
      {file && (
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox={`0 0 ${VW} ${VH}`}
        >
          <defs>
            <mask id={maskId}>
              <rect width={VW} height={VH} fill="white" />
              {boxes.map((b, i) => (
                <rect
                  key={i}
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
      )}
    </div>
  );
}
