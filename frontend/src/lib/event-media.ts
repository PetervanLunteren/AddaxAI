/**
 * Human description of what an event's files consist of:
 * "1 video", "12 images", or "2 videos and 2 images".
 *
 * Counts files, not frames (the event filmstrip used to mislabel its file
 * count as "frames"). Everything that isn't a video is counted as an image.
 */
export function describeEventMedia(files: { file_type: string }[]): string {
  const videos = files.filter((f) => f.file_type === "video").length;
  const images = files.length - videos;
  const parts: string[] = [];
  if (videos > 0) parts.push(`${videos} video${videos === 1 ? "" : "s"}`);
  if (images > 0) parts.push(`${images} image${images === 1 ? "" : "s"}`);
  return (
    parts.join(" and ") ||
    `${files.length} file${files.length === 1 ? "" : "s"}`
  );
}
