/**
 * State hooks behind the shared `ViewerToolRail` (see that file for the
 * design). Separate module because a component file must export only
 * components for fast refresh to work.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { create } from "zustand";

import { filesApi } from "../../api/files";
import { readLabelsSettings, persistLabelsSetting } from "./labels-settings";

/** What the rail needs to know about the focused file. */
export interface RailFile {
  id: string;
  file_path: string;
  flagged: boolean;
  favorited: boolean;
}

/** A persisted slider value; anything odd falls back to neutral (50). */
function clampAdjust(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 100
    ? v
    : 50;
}

interface ImageAdjustState {
  brightness: number;
  contrast: number;
  setBrightness: (v: number) => void;
  setContrast: (v: number) => void;
}

/**
 * One store for the whole app, initialised from localStorage, so the grids,
 * the detail modals and the video player always show the same adjustment
 * and it survives a restart. View-only CSS filters; stored data never
 * changes.
 */
const useImageAdjustStore = create<ImageAdjustState>((set) => ({
  brightness: clampAdjust(readLabelsSettings().brightness),
  contrast: clampAdjust(readLabelsSettings().contrast),
  setBrightness: (v) => {
    set({ brightness: v });
    persistLabelsSetting("brightness", v);
  },
  setContrast: (v) => {
    set({ contrast: v });
    persistLabelsSetting("contrast", v);
  },
}));

function toFilter(brightness: number, contrast: number): string | undefined {
  return brightness !== 50 || contrast !== 50
    ? `brightness(${brightness / 50}) contrast(${contrast / 50})`
    : undefined;
}

/** Brightness/contrast state plus the CSS filter they mean. */
export function useImageAdjust() {
  const { brightness, contrast, setBrightness, setContrast } =
    useImageAdjustStore();
  return {
    brightness,
    setBrightness,
    contrast,
    setContrast,
    imageFilter: toFilter(brightness, contrast),
  };
}

/** Just the CSS filter, for tiles that only display it. */
export function useImageFilter(): string | undefined {
  return useImageAdjustStore((s) => toFilter(s.brightness, s.contrast));
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
