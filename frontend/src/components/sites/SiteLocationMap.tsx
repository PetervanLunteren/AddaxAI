/**
 * Small single-marker Leaflet preview for the Site info sheet.
 *
 * Non-interactive by default (no scroll-wheel zoom, no double-click
 * zoom), just enough to show where the site is. Reuses the same
 * positron base layer the Map page uses so the visual style matches.
 */

import { CircleMarker, MapContainer, TileLayer } from "react-leaflet";

import { useTheme } from "../../lib/theme";

interface SiteLocationMapProps {
  latitude: number;
  longitude: number;
  zoom?: number;
}

const CARTO_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>';

function positronUrl(dark: boolean): string {
  return dark
    ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
    : "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png";
}

export function SiteLocationMap({
  latitude,
  longitude,
  zoom = 12,
}: SiteLocationMapProps) {
  const { resolvedTheme } = useTheme();
  const url = positronUrl(resolvedTheme === "dark");
  return (
    <div className="h-[180px] w-full overflow-hidden rounded-md border">
      <MapContainer
        center={[latitude, longitude]}
        zoom={zoom}
        style={{ height: "100%", width: "100%" }}
        scrollWheelZoom={false}
        doubleClickZoom={false}
        zoomControl={false}
      >
        <TileLayer key={url} url={url} attribution={CARTO_ATTRIBUTION} />
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
