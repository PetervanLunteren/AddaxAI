/**
 * Video filmstrip: a 3x3 gallery of evenly-spaced (in time) low-res frames
 * for the counts modal, so a video shows how the clip progresses instead of
 * just its single best frame. Frames are decoded on demand by the backend
 * (nothing persisted) and cached by react-query.
 *
 * Renders the grid only; the modal's existing play-button overlay sits on
 * top. When the backend returns no frames (undecodable video) this renders
 * nothing so the parent can fall back to the best-frame still.
 */

import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { filesApi } from "../../api/files";

/** Seconds -> "m:ss", or null when the video has no known frame rate. */
function formatClock(seconds: number | null): string | null {
  if (seconds == null) return null;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

interface VideoFilmstripProps {
  fileId: string;
}

export function VideoFilmstrip({ fileId }: VideoFilmstripProps) {
  const { data, isLoading } = useQuery({
    queryKey: ["filmstrip", fileId],
    queryFn: () => filesApi.getFilmstrip(fileId),
    // A filmstrip is immutable per video, so never refetch within a session.
    staleTime: Infinity,
  });

  if (isLoading) {
    return (
      <div className="grid h-full w-full grid-cols-3 grid-rows-3 gap-1 p-2">
        {Array.from({ length: 9 }).map((_, i) => (
          <div
            key={i}
            className="flex items-center justify-center rounded bg-white/5"
          >
            <Loader2 className="h-5 w-5 animate-spin text-white/40" />
          </div>
        ))}
      </div>
    );
  }

  const frames = data?.frames ?? [];
  if (frames.length === 0) return null; // parent falls back to the best frame

  return (
    <div
      className="grid h-full w-full grid-cols-3 gap-1 p-2"
      style={{ gridAutoRows: "1fr" }}
    >
      {frames.map((frame) => {
        const clock = formatClock(frame.time_seconds);
        return (
          <div
            key={frame.frame_number}
            className="relative overflow-hidden rounded bg-black"
          >
            <img
              src={frame.image}
              alt=""
              className="h-full w-full object-cover"
            />
            {clock && (
              <span className="absolute left-1 top-1 rounded bg-black/60 px-1 text-[10px] leading-tight text-white/90">
                {clock}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
