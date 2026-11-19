/**
 * The "Include the frames table" choice, shared by the project
 * Spreadsheet export and the folder-run Save step so both say the same
 * thing. The Frames table is one row per box per video frame: the
 * largest table by far and only needed for per-frame work, so it is
 * opt-in and off by default. Excel never carries it (too many rows for
 * a sheet), so with Excel picked the box is shown greyed out with the
 * reason rather than hidden, which would leave people looking for it.
 */

export const FRAMES_TABLE_LABEL = "Include the frames table";

export function framesTableCaption(excel: boolean): string {
  return excel
    ? "Not available in Excel, it has too many rows for a sheet. Pick CSV to include it."
    : "One row per box per video frame, for movement and per-frame analysis. Large. Only written when there are videos.";
}

const KEY = "addaxai.export.includeFrames";

/** The project dialog's last choice. Off when nothing is stored. */
export function loadIncludeFrames(): boolean {
  try {
    return localStorage.getItem(KEY) === "true";
  } catch {
    return false;
  }
}

export function saveIncludeFrames(value: boolean): void {
  try {
    localStorage.setItem(KEY, String(value));
  } catch {
    // Convenience only; ignore storage failures.
  }
}
