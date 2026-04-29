/**
 * Reveal a file in the OS file explorer (Electron only).
 *
 * Frames (file_type="frame") are extracted thumbnails stored under a
 * hidden `.addaxai/video_frames/` directory. Revealing those is useless
 * to the user, so for frames we fetch the source video record and
 * reveal the original video file instead. The lookup goes through the
 * shared ["file", id] react-query cache so a second click is free.
 */

import { useQueryClient } from "@tanstack/react-query";
import { useCallback } from "react";

import { filesApi } from "../api/files";

interface RevealableFile {
  file_type: string;
  file_path: string;
  source_video_id: string | null;
}

export function useRevealInFolder() {
  const queryClient = useQueryClient();
  return useCallback(
    async (file: RevealableFile) => {
      if (!window.electronAPI) return;
      let path = file.file_path;
      if (file.file_type === "frame" && file.source_video_id) {
        const source = await queryClient.fetchQuery({
          queryKey: ["file", file.source_video_id],
          queryFn: () => filesApi.get(file.source_video_id!),
        });
        path = source.file_path;
      }
      await window.electronAPI.showItemInFolder(path);
    },
    [queryClient],
  );
}
