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
 * The backgrounds the Map page offers, in the order of its layers
 * control. `name` is what the control shows, and also how Leaflet tells
 * the layers apart, so names are unique. An entry without `url` is the
 * street map (`StreetBaseLayer`). The ids are stored per device, so
 * they do not change.
 */
export const BASE_LAYERS = [
  { id: "positron", name: "Light" },
  {
    id: "satellite",
    name: "Satellite",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution:
      "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
  },
  { id: "osm", name: "OpenStreetMap", ...OSM_LAYER },
  {
    id: "topo",
    name: "Topographic",
    url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    // The credit line opentopomap.org asks for.
    attribution:
      'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
    // Its tiles stop one level before the map does; the last level is
    // enlarged instead of left blank.
    maxNativeZoom: 17,
  },
] as const;

export type BaseLayer = (typeof BASE_LAYERS)[number]["id"];

export function isBaseLayer(value: unknown): value is BaseLayer {
  return BASE_LAYERS.some((layer) => layer.id === value);
}

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
