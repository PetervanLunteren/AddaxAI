/**
 * One random animal photo: the detection's crop, its name, where and when.
 *
 * Shared by the Overview's "Gracious random animal" and the Explore photo
 * wall, so both look and behave alike. Clicking opens the file in the
 * Labels page's Files view (the `lbl_file` deep link FilesTab consumes),
 * where the label can be checked or corrected: looking turns into
 * verifying.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { ImageOff } from "lucide-react";

import type { AnimalPhoto } from "../../api/statistics";
import { API_BASE_URL } from "../../lib/api-client";
import { formatCameraDate } from "../../lib/datetime";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import { cn } from "../../lib/utils";

/**
 * The crop each size asks the shared crop service for. The tile has the
 * same shape as the crop, so CSS never cuts it: the wall is square like
 * the Labels grid, and the hero is a wide frame that the service fills
 * with the photo around the animal (`aspect`), long side sharp enough for
 * a retina screen.
 */
const CROPS = {
  large: { query: "size=1024&aspect=1.6", frame: "aspect-[16/10]" },
  small: { query: "size=512", frame: "aspect-square" },
} as const;

interface AnimalPhotoTileProps {
  projectId: string;
  photo: AnimalPhoto;
  /** Big caption for the hero tile, a one-liner for the wall. */
  size: "large" | "small";
  className?: string;
}

export function AnimalPhotoTile({
  projectId,
  photo,
  size,
  className,
}: AnimalPhotoTileProps) {
  const [failed, setFailed] = useState(false);
  const name = resolveSpeciesName(photo);
  // captured_date is a bare camera date; the time part only makes it a
  // value formatCameraDate can parse.
  const date = photo.captured_date
    ? formatCameraDate(`${photo.captured_date}T00:00:00`)
    : null;
  const crop = CROPS[size];
  const cropUrl = `${API_BASE_URL}/api/detections/${photo.detection_id}/crop?${crop.query}`;
  const details = [photo.site_name, date].filter(Boolean).join(" · ");

  return (
    <Link
      to={`/projects/${projectId}/labels?view=files&lbl_file=${photo.file_id}`}
      title={`${name}. Open this file on the Labels page`}
      className={cn(
        "group relative block overflow-hidden rounded-md border bg-muted",
        crop.frame,
        className,
      )}
    >
      {failed ? (
        <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
          <ImageOff className="h-6 w-6" />
        </div>
      ) : (
        <img
          src={cropUrl}
          alt={name}
          className="absolute inset-0 h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]"
          onError={() => setFailed(true)}
        />
      )}
      {/* White on a dark gradient over the photo: theme-independent by
          design (FRONTEND_CONVENTIONS, "whites over photos"). */}
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/75 via-black/40 to-transparent px-3 pb-2 pt-8 text-white">
        <p
          className={cn(
            "truncate font-semibold",
            size === "large" ? "text-lg" : "text-xs",
          )}
        >
          {name}
        </p>
        {size === "large" && (
          <p className="truncate text-xs text-white/80">{details}</p>
        )}
      </div>
    </Link>
  );
}
