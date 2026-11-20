/**
 * Export choices the app remembers between exports, in this browser
 * profile: whether to add the frames table, and the folder the last
 * export went to. Convenience only, so a storage failure just means the
 * defaults (frames off, no folder).
 *
 * The frames wording lives here too, so the project Spreadsheet export
 * and the folder-run Save step say the same thing. The Frames table is
 * one row per box per video frame: the largest table by far and only
 * needed for per-frame work, so it is opt-in. It works in every format.
 * The caption leaves Excel's row limit out: an export that passes it
 * stops with the shared message that names CSV (`XlsxRowLimitError`),
 * which is where that advice is true and needed.
 */

export const FRAMES_TABLE_LABEL = "Include the frames table";

export const FRAMES_TABLE_CAPTION =
  "One row per box per video frame. Can grow large.";

const FRAMES_KEY = "addaxai.export.includeFrames";
const FOLDER_KEY = "addaxai.export.lastFolder";

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Convenience only; ignore storage failures.
  }
}

/** The Spreadsheet dialog's last frames choice. Off when nothing is stored. */
export const loadIncludeFrames = (): boolean => read(FRAMES_KEY) === "true";
export const saveIncludeFrames = (value: boolean): void => write(FRAMES_KEY, String(value));

/** Where the last export went, to start the next one there. Empty when
 *  nothing is stored. One folder for every export and project: names
 *  carry the project and the date, and nothing is ever overwritten. */
export const loadLastExportFolder = (): string => read(FOLDER_KEY) ?? "";
export const saveLastExportFolder = (folder: string): void => write(FOLDER_KEY, folder);
