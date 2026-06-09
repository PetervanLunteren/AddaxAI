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
import { FrameThumbnail } from "./FrameThumbnail";

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

  return (
    <FrameThumbnail
      fileId={fileId}
      file={file}
      detectionThreshold={detectionThreshold}
      viewBox={viewBox}
      className={className}
    />
  );
}
