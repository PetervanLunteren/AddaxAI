/**
 * Reveal a file in the OS file explorer (Electron only).
 *
 * Post-2026-05 there are no `file_type="frame"` rows: images and videos
 * are the only file types and both already point at the user's actual
 * on-disk source, so we just hand `file_path` straight to Electron.
 */

import { useCallback } from "react";

interface RevealableFile {
  file_path: string;
}

export function useRevealInFolder() {
  return useCallback(async (file: RevealableFile) => {
    if (!window.electronAPI) return;
    await window.electronAPI.showItemInFolder(file.file_path);
  }, []);
}
