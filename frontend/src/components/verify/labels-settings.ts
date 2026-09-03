/**
 * View preferences for the Labels page, shared by both of its tabs.
 *
 * Small things the user sets once and expects to stay put: the sort they
 * like, how big the tiles are. Kept in localStorage rather than the URL
 * because they describe how someone likes to work, not what they are
 * looking at, so they should survive a fresh link and not travel with a
 * shared one.
 *
 * One module owns the key, so Detections and Files read and write the same
 * store instead of each keeping its own idea of what "large tiles"
 * means. Tile size in particular is one preference, not two: a person
 * who wants big tiles wants them in both tabs, even though the two map
 * S / M / L to different pixel widths because a crop and a whole camera
 * frame are different shapes.
 */

import { useCallback, useMemo, useState } from "react";

import type { TileSize } from "./CropGrid";

const LS_KEY = "addaxai:labelsSettings";

export function readLabelsSettings(): Record<string, unknown> {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) || "{}");
  } catch {
    return {};
  }
}

/** Read-modify-write, so two settings written from different tabs cannot
 *  clobber each other. */
export function persistLabelsSetting(key: string, value: unknown): void {
  try {
    const current = readLabelsSettings();
    current[key] = value;
    localStorage.setItem(LS_KEY, JSON.stringify(current));
  } catch {
    /* A full or blocked localStorage is not worth failing a render over. */
  }
}

const TILE_SIZES: TileSize[] = ["S", "M", "L"];

/** The Files tab's sort modes; mirrors `LabelsFilesSort` on the wire. */
export type FilesSort = "path" | "events" | "newest" | "oldest" | "random";
const FILES_SORTS: FilesSort[] = [
  "path",
  "events",
  "newest",
  "oldest",
  "random",
];

/** The Files tab's sort, persisted like the Detections sort so a person
 *  who works by event opens on event sort every time. "random" is a
 *  one-off shuffle, not a way of working, and its seed is session
 *  state, so it is never written: leaving in random reopens on the
 *  last persisted sort. */
export function useFilesSort(): [FilesSort, (v: FilesSort) => void] {
  const saved = useMemo(() => readLabelsSettings().filesSort, []);
  const [sort, setLocal] = useState<FilesSort>(
    FILES_SORTS.includes(saved as FilesSort) && saved !== "random"
      ? (saved as FilesSort)
      : "path",
  );
  const setSort = useCallback((v: FilesSort) => {
    setLocal(v);
    if (v !== "random") persistLabelsSetting("filesSort", v);
  }, []);
  return [sort, setSort];
}

/** Tile size, shared by both grids and persisted. */
export function useTileSize(): [TileSize, (v: TileSize) => void] {
  const saved = useMemo(() => readLabelsSettings().tileSize, []);
  const [tileSize, setLocal] = useState<TileSize>(
    TILE_SIZES.includes(saved as TileSize) ? (saved as TileSize) : "M",
  );
  const setTileSize = useCallback((v: TileSize) => {
    setLocal(v);
    persistLabelsSetting("tileSize", v);
  }, []);
  return [tileSize, setTileSize];
}
