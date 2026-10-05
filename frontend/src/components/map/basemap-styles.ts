/**
 * The one place that knows where basemaps come from.
 *
 * Street-map styles are OpenFreeMap vector styles (positron for light,
 * dark for dark), drawn by MapLibre GL. They replaced the visually
 * identical CARTO tiles when CARTO started watermarking keyless
 * requests (August 2026) - the same switch AddaxAI Connect made.
 * OpenFreeMap needs no key and has no usage limits.
 *
 * Attributions are hardcoded on purpose: Leaflet's attribution control
 * writes them with innerHTML and sanitises nothing, so the string must
 * never come from the style server.
 */

export function openFreeMapStyleUrl(dark: boolean): string {
  return dark
    ? "https://tiles.openfreemap.org/styles/dark"
    : "https://tiles.openfreemap.org/styles/positron";
}

// What openfreemap.org asks for, minus the part they call optional
// (their own name): OpenMapTiles built the tiles, OpenStreetMap is the
// data.
export const OPENFREEMAP_ATTRIBUTION =
  '<a href="https://www.openmaptiles.org/" target="_blank">&copy; OpenMapTiles</a> ' +
  'Data from <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>';

export const OSM_LAYER = {
  url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
};

/**
 * MapLibre needs WebGL2; the handful of machines without it get OSM
 * raster tiles instead of a blank map.
 */
let hasWebGL2: boolean | null = null;

export function supportsWebGL2(): boolean {
  if (hasWebGL2 === null) {
    try {
      hasWebGL2 = !!document.createElement("canvas").getContext("webgl2");
    } catch {
      hasWebGL2 = false;
    }
  }
  return hasWebGL2;
}
