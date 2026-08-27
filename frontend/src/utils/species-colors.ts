/**
 * Species colours.
 *
 * The backend assigns them (`backend/app/api/crud/label_colors.py`):
 * species present in the project are sorted by taxonomy and walk a
 * palette ordered so consecutive entries contrast most, which gives
 * look-alike siblings the most different colours. This file only holds
 * the map that `useLabelColors` fetches per project and answers lookups
 * from it. There is deliberately no colour algorithm on this side: one
 * implementation means the annotated JPEG export and the grid can never
 * disagree.
 *
 * Keys are `label_taxonomy_id` or the label name, matched
 * case-insensitively. A key the map does not know renders neutral grey
 * so a missing map is visible rather than silently plausible.
 */
import { useSyncExternalStore } from "react";
import chroma from "chroma-js";

/** Colour for a label the project map does not know (map not loaded
 * yet, or a label outside the project's counting threshold). */
export const UNKNOWN_SPECIES_COLOR = "#6b7280";

let labelColors: Record<string, string> = {};
let version = 0;
const listeners = new Set<() => void>();

/** Replace the active map. Called by `useLabelColors` whenever the
 * project's colour query resolves. Bumps a version so subscribed
 * components repaint. */
export function setLabelColors(colors: Record<string, string>): void {
  labelColors = colors;
  version += 1;
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getVersion(): number {
  return version;
}

/** Re-render the calling component when the colour map changes. Needed
 * by anything that colours during render and is memoised, or that
 * paints onto a canvas inside an effect (add the returned version to
 * that effect's deps). */
export function useSpeciesColorsVersion(): number {
  return useSyncExternalStore(subscribe, getVersion, getVersion);
}

export function getSpeciesColor(species: string): string {
  return labelColors[species.toLowerCase()] ?? UNKNOWN_SPECIES_COLOR;
}

/** Pick a foreground color (white or near-black) that meets WCAG AA
 * contrast against the given background.
 *
 * Pass the ACTUAL rendered background colour (hex string) rather than a
 * species key, so the text can never be derived from a different key
 * than the background was. Threshold is 4.5 (AA, normal text); chips
 * below that get dark text.
 */
export function getContrastTextColor(bgHex: string): string {
  return chroma.contrast(bgHex, "white") >= 4.5 ? "white" : "#1f2937";
}
