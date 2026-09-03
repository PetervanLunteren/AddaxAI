/**
 * State hooks behind the shared `ViewerToolRail` (see that file for the
 * design). Separate module because a component file must export only
 * components for fast refresh to work.
 */

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { filesApi } from "../../api/files";

/** What the rail needs to know about the focused file. */
export interface RailFile {
  id: string;
  file_path: string;
  flagged: boolean;
  favorited: boolean;
}

/** Brightness/contrast state plus the CSS filter they mean. */
export function useImageAdjust() {
  const [brightness, setBrightness] = useState(50);
  const [contrast, setContrast] = useState(50);
  const imageFilter =
    brightness !== 50 || contrast !== 50
      ? `brightness(${brightness / 50}) contrast(${contrast / 50})`
      : undefined;
  return { brightness, setBrightness, contrast, setContrast, imageFilter };
}

export interface FileTriage {
  toggleFlag: (file: RailFile) => void;
  toggleLike: (file: RailFile) => void;
  pending: boolean;
}

/**
 * The flag and like writes. The `["file"]` invalidation is common to
 * every host; `onChanged` carries a host's own extras (the Counts modal
 * refreshes its event caches, the grids their liked/flagged filters).
 * Instantiated by the host, not the rail, so a keyboard shortcut (F)
 * and the rail's buttons share one mutation and cannot drift.
 */
export function useFileTriage(onChanged?: () => void): FileTriage {
  const queryClient = useQueryClient();
  const done = () => {
    queryClient.invalidateQueries({ queryKey: ["file"] });
    onChanged?.();
  };
  const failed = (err: Error) => toast.error(err.message);
  const flagMutation = useMutation({
    mutationFn: (f: RailFile) => filesApi.update(f.id, { flagged: !f.flagged }),
    onSuccess: done,
    onError: failed,
  });
  const likeMutation = useMutation({
    mutationFn: (f: RailFile) =>
      filesApi.update(f.id, { favorited: !f.favorited }),
    onSuccess: done,
    onError: failed,
  });
  // One toggle at a time. The new value is computed from the cached row
  // (`!f.flagged`), so a second press before the refetch reads the old
  // value and re-sends the same write: two fast F presses left a file
  // flagged. The rail's buttons are disabled while pending; this makes
  // the keyboard path equally safe.
  const pending = flagMutation.isPending || likeMutation.isPending;
  return {
    toggleFlag: (f) => {
      if (!pending) flagMutation.mutate(f);
    },
    toggleLike: (f) => {
      if (!pending) likeMutation.mutate(f);
    },
    pending,
  };
}
