/**
 * The street basemap every map in the app draws by default: the
 * OpenFreeMap vector style for the active theme, or OSM raster tiles on
 * the few machines without WebGL2 (see basemap-styles.ts). One component
 * so the Map page, the site preview and the dashboard mini map cannot
 * drift apart.
 */

import { TileLayer } from "react-leaflet";

import { useTheme } from "../../lib/theme";
import {
  OPENFREEMAP_ATTRIBUTION,
  OSM_LAYER,
  openFreeMapStyleUrl,
  supportsWebGL2,
} from "./basemap-styles";
import MapLibreGLLayer from "./MapLibreGLLayer";

export function StreetBaseLayer() {
  const { resolvedTheme } = useTheme();
  if (!supportsWebGL2()) {
    return (
      <TileLayer
        key={OSM_LAYER.url}
        url={OSM_LAYER.url}
        attribution={OSM_LAYER.attribution}
      />
    );
  }
  const styleUrl = openFreeMapStyleUrl(resolvedTheme === "dark");
  return (
    <MapLibreGLLayer
      key={styleUrl}
      styleUrl={styleUrl}
      attribution={OPENFREEMAP_ATTRIBUTION}
    />
  );
}
