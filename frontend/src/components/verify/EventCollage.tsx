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
 * query and hands it to FrameThumbnail, which draws the bboxes in image
 * space and slices them to the tile so they stay aligned in every layout.
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
        />
        <EventCollageTile
          fileId={fileIds[1]}
          detectionThreshold={detectionThreshold}
        />
        <EventCollageTile
          fileId={fileIds[2]}
          detectionThreshold={detectionThreshold}
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
        />
      ))}
    </div>
  );
}

interface EventCollageTileProps {
  fileId: string;
  detectionThreshold: number;
  className?: string;
}

function EventCollageTile({
  fileId,
  detectionThreshold,
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
      className={className}
    />
  );
}
