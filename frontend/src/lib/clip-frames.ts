/**
 * A clip's frames as pictures: the one URL every viewer asks for a frame
 * by, and the warm-up that asks for the next ones before they are needed.
 *
 * One URL builder and one image maker on purpose. The browser caches a
 * picture by its URL and by its CORS mode: the canvas loads with
 * `crossOrigin` set (an annotated export reads its pixels back), and a
 * plain `Image()` of the same URL is a separate cache entry that the
 * canvas never hits (measured 2026-11-05: every warmed frame downloaded
 * again). So every load meant for the canvas goes through
 * `canvasImage`. The backend keeps the decoded frame too, which also
 * helps the card modal's plain `<img>`. The cover is the stored picture
 * (`/image`, no frame), so it is never warmed: it is already on disk.
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

/** An image loaded the way the canvas loads it, so a warm-up and the
 *  canvas share one browser cache entry. Set `src` after the handlers. */
export function canvasImage(): HTMLImageElement {
  const img = new Image();
  img.crossOrigin = "anonymous";
  return img;
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
    const img = canvasImage();
    img.onload = next;
    img.onerror = next;
    img.src = frameImageUrl(file.id, frame);
  };
  next();
}
