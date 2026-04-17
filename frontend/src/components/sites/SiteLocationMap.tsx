/**
 * Small single-marker Leaflet preview for the Site info sheet.
 *
 * Non-interactive by default (no scroll-wheel zoom, no double-click
 * zoom), just enough to show where the site is. Reuses the same
 * positron base layer the Map page uses so the visual style matches.
 */

import { CircleMarker, MapContainer, TileLayer } from "react-leaflet";

interface SiteLocationMapProps {
  latitude: number;
  longitude: number;
  zoom?: number;
}

const POSITRON = {
  url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
};

export function SiteLocationMap({
  latitude,
  longitude,
  zoom = 12,
}: SiteLocationMapProps) {
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
        <TileLayer url={POSITRON.url} attribution={POSITRON.attribution} />
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
