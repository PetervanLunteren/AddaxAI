/**
 * Wide (full-width) mode for the Labels and Counts grids.
 *
 * Opt-in per user, persisted to localStorage, default off. The normal
 * capped view (`max-w-7xl`) is the default and is left untouched; wide
 * mode removes the page's width cap so the grid fills the content area,
 * and the grids switch their column count from viewport breakpoints to
 * the measured container width so extra width means more columns.
 *
 * The flag is set on the page (LabelsPage / CountsPage) and read deep in
 * the grid via `WideModeContext`, so it doesn't have to thread through
 * every intermediate component.
 */

import { createContext, useCallback, useContext, useMemo, useState } from "react";

export interface WideModeState {
  wide: boolean;
  toggle: () => void;
}

export const WideModeContext = createContext<WideModeState>({
  wide: false,
  toggle: () => {},
});

/** Just the flag — for the grids, which only need to know the mode. */
export const useWideModeValue = (): boolean => useContext(WideModeContext).wide;

/** Flag + toggle — for the toolbar button that lives inside the view. */
export const useWideModeControls = (): WideModeState =>
  useContext(WideModeContext);

/**
 * Host-level wide-mode state. The page/step owns it, applies the width
 * change to its own container, and passes the returned object into
 * `WideModeContext.Provider` so the in-view toolbar button and the grid
 * can read it.
 *
 * Deliberately NOT persisted: it resets to the normal capped view on
 * every mount (page visit / run), so the default is always the
 * consistent look and wide only appears right after a user clicks it.
 */
export function useWideMode(): WideModeState {
  const [wide, setWide] = useState(false);
  const toggle = useCallback(() => setWide((prev) => !prev), []);
  return useMemo(() => ({ wide, toggle }), [wide, toggle]);
}

/**
 * How many equal columns fit `width` px given a minimum tile width and
 * inter-tile gap. Floors so tiles never shrink below `minTile`; never
 * returns less than 1.
 */
export function columnsForWidth(
  width: number,
  minTile: number,
  gap: number,
): number {
  if (width <= 0) return 1;
  return Math.max(1, Math.floor((width + gap) / (minTile + gap)));
}
