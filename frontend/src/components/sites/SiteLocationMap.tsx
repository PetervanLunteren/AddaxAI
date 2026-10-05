/**
 * Small single-marker Leaflet preview for the Site info sheet.
 *
 * Non-interactive by default (no scroll-wheel zoom, no double-click
 * zoom), just enough to show where the site is. Uses the same
 * OpenFreeMap street style as the Map page so the visual style
 * matches, with the same OSM raster fallback without WebGL2.
 */

import { CircleMarker, MapContainer, TileLayer } from "react-leaflet";

import { useTheme } from "../../lib/theme";
import {
  OPENFREEMAP_ATTRIBUTION,
  OSM_LAYER,
  openFreeMapStyleUrl,
  supportsWebGL2,
} from "../map/basemap-styles";
import MapLibreGLLayer from "../map/MapLibreGLLayer";

interface SiteLocationMapProps {
  latitude: number;
  longitude: number;
  zoom?: number;
}

export function SiteLocationMap({
  latitude,
  longitude,
  zoom = 12,
}: SiteLocationMapProps) {
  const { resolvedTheme } = useTheme();
  const styleUrl = openFreeMapStyleUrl(resolvedTheme === "dark");
  return (
    <div className="h-[180px] w-full overflow-hidden rounded-md border">
      <MapContainer
        center={[latitude, longitude]}
        zoom={zoom}
        maxZoom={18}
        style={{ height: "100%", width: "100%" }}
        scrollWheelZoom={false}
        doubleClickZoom={false}
        zoomControl={false}
      >
        {supportsWebGL2() ? (
          <MapLibreGLLayer
            key={styleUrl}
            styleUrl={styleUrl}
            attribution={OPENFREEMAP_ATTRIBUTION}
          />
        ) : (
          <TileLayer url={OSM_LAYER.url} attribution={OSM_LAYER.attribution} />
        )}
        <CircleMarker
          center={[latitude, longitude]}
          radius={7}
          pathOptions={{
            color: "#0f6064",
            fillColor: "#0f6064",
            fillOpacity: 0.8,
            weight: 2,
          }}
        />
      </MapContainer>
    </div>
  );
}
