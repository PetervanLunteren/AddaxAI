/**
 * Site Map Component
 *
 * Interactive map for selecting site locations.
 * - Shows existing project sites as markers
 * - Click map to place new site marker
 * - Auto-zooms to fit existing markers
 * - Displays lat/lon coordinates
 */

import { useState, useCallback, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Map, Marker, NavigationControl } from "react-map-gl/maplibre";
import { MapPin } from "lucide-react";
import { sitesApi } from "@/api/sites";
import "maplibre-gl/dist/maplibre-gl.css";

interface SiteMapProps {
  projectId: string;
  selectedLocation: { lat: number; lon: number } | null;
  onLocationSelect: (lat: number, lon: number) => void;
}

export function SiteMap({ projectId, selectedLocation, onLocationSelect }: SiteMapProps) {
  // Fetch existing sites
  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
  });

  // Map viewport state
  const [viewState, setViewState] = useState({
    longitude: 0,
    latitude: 20,
    zoom: 1.5,
  });

  // Auto-zoom to fit sites on load
  useEffect(() => {
    if (!sites || sites.length === 0) return;

    // Calculate bounds to fit all sites
    const validSites = sites.filter((s) => s.latitude != null && s.longitude != null);
    if (validSites.length === 0) return;

    const lats = validSites.map((s) => s.latitude!);
    const lons = validSites.map((s) => s.longitude!);

    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);

    const centerLat = (minLat + maxLat) / 2;
    const centerLon = (minLon + maxLon) / 2;

    // Calculate zoom level based on bounds
    const latDiff = maxLat - minLat;
    const lonDiff = maxLon - minLon;
    const maxDiff = Math.max(latDiff, lonDiff);

    let zoom = 1.5;
    if (maxDiff < 0.1) zoom = 12;
    else if (maxDiff < 1) zoom = 8;
    else if (maxDiff < 5) zoom = 6;
    else if (maxDiff < 20) zoom = 4;
    else zoom = 2;

    setViewState({
      longitude: centerLon,
      latitude: centerLat,
      zoom,
    });
  }, [sites]);

  // Handle map click
  const handleMapClick = useCallback(
    (event: any) => {
      const { lngLat } = event;
      onLocationSelect(lngLat.lat, lngLat.lng);
    },
    [onLocationSelect]
  );

  return (
    <div className="relative h-[400px] w-full rounded-lg overflow-hidden border border-gray-200">
      <Map
        {...viewState}
        onMove={(evt) => setViewState(evt.viewState)}
        onClick={handleMapClick}
        mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
        style={{ width: "100%", height: "100%" }}
      >
        <NavigationControl position="top-right" />

        {/* Existing sites as markers */}
        {sites?.map(
          (site) =>
            site.latitude != null &&
            site.longitude != null && (
              <Marker
                key={site.id}
                longitude={site.longitude}
                latitude={site.latitude}
                anchor="bottom"
              >
                <div className="relative group">
                  <MapPin className="h-6 w-6 text-blue-600 drop-shadow-lg" fill="currentColor" />
                  {/* Tooltip */}
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                    {site.name}
                  </div>
                </div>
              </Marker>
            )
        )}

        {/* New site marker (user selection) */}
        {selectedLocation && (
          <Marker
            longitude={selectedLocation.lon}
            latitude={selectedLocation.lat}
            anchor="bottom"
          >
            <div className="relative animate-bounce">
              <MapPin className="h-8 w-8 text-red-600 drop-shadow-lg" fill="currentColor" />
              {/* Pulse effect */}
              <div className="absolute inset-0 h-8 w-8 animate-ping">
                <MapPin className="h-8 w-8 text-red-600 opacity-75" fill="currentColor" />
              </div>
            </div>
          </Marker>
        )}
      </Map>

      {/* Coordinates display */}
      {selectedLocation && (
        <div className="absolute bottom-2 left-2 bg-white px-3 py-1.5 rounded shadow-md text-xs font-mono">
          {selectedLocation.lat.toFixed(6)}, {selectedLocation.lon.toFixed(6)}
        </div>
      )}

      {/* Help text */}
      <div className="absolute top-2 left-2 bg-white px-3 py-1.5 rounded shadow-md text-xs text-gray-600 max-w-[200px]">
        Click anywhere on the map to place your site marker
      </div>
    </div>
  );
}
