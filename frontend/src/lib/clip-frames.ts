/**
 * A clip's frames as pictures: the one URL every viewer asks for a frame
 * by, and the warm-up that asks for the next ones before they are needed.
 *
 * One URL builder on purpose. The browser caches a frame by its URL, and
 * the backend keeps the decoded frame too, so a frame warmed here is free
 * on screen only when the canvas and the card modal ask for it in exactly
 * the same form. The cover is the stored picture (`/image`, no frame), so
 * it is never warmed: it is already on disk.
 *
 * The warm-up loads one frame at a time and the latest call wins. Every
 * frame is a decode from the clip on the backend (at most four at once),
 * and a request the browser drops does not stop that decode, so firing a
 * handful at once would queue behind each other and in front of the
 * frame a person asks for next. One in flight, the rest dropped when the
 * person has already moved on: the same rule CVAT's chunk preload follows
 * for stepping, the opposite of loading frames on hover, which a sweep of
 * the ruler would turn into hundreds of decodes (2026-11-05).
 */

import type { FileWithDetections } from "../api/types";

/** The URL of one frame of a file, the cover when `frame` is omitted. */
export function frameImageUrl(fileId: string, frame?: number | null): string {
  return frame == null
    ? `/api/files/${fileId}/image`
    : `/api/files/${fileId}/image?frame=${frame}`;
}

let warmToken = 0;

/**
 * Ask for `frames` of a clip in the background, one after the other.
 * The cover and repeats are skipped; a later call stops what is left of
 * an earlier one.
 */
export function warmClipFrames(file: FileWithDetections, frames: (number | undefined)[]): void {
  const token = ++warmToken;
  const queue = [...new Set(frames)].filter(
    (f): f is number => f != null && f !== file.best_frame_number,
  );
  const next = () => {
    const frame = queue.shift();
    if (frame == null || token !== warmToken) return;
    const img = new Image();
    img.onload = next;
    img.onerror = next;
    img.src = frameImageUrl(file.id, frame);
  };
  next();
}
