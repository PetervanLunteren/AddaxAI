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
import { MapPin, MapIcon, Satellite } from "lucide-react";
import { sitesApi } from "@/api/sites";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import "maplibre-gl/dist/maplibre-gl.css";

// Map style options
const MAP_STYLES = {
  streets: {
    name: "Streets",
    icon: MapIcon,
    url: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  },
  satellite: {
    name: "Satellite",
    icon: Satellite,
    url: {
      version: 8,
      sources: {
        satellite: {
          type: "raster",
          tiles: [
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
          ],
          tileSize: 256,
          minzoom: 0,
          maxzoom: 22,
        },
      },
      layers: [
        {
          id: "satellite",
          type: "raster",
          source: "satellite",
        },
      ],
    },
  },
} as const;

type MapStyleKey = keyof typeof MAP_STYLES;

interface SiteMapProps {
  projectId: string;
  selectedLocation: { lat: number; lon: number } | null;
  onLocationSelect: (lat: number, lon: number) => void;
  onMapError?: () => void;
}

export function SiteMap({ projectId, selectedLocation, onLocationSelect, onMapError }: SiteMapProps) {
  // Fetch existing sites
  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
  });

  // Map style state
  const [mapStyle, setMapStyle] = useState<MapStyleKey>("satellite");

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

    // Add padding to bounds (20% on each side)
    const latDiff = maxLat - minLat;
    const lonDiff = maxLon - minLon;
    const latPadding = Math.max(latDiff * 0.3, 0.01); // At least 0.01 degrees padding
    const lonPadding = Math.max(lonDiff * 0.3, 0.01);

    const centerLat = (minLat + maxLat) / 2;
    const centerLon = (minLon + maxLon) / 2;

    // Calculate zoom level based on padded bounds
    const paddedLatDiff = latDiff + (latPadding * 2);
    const paddedLonDiff = lonDiff + (lonPadding * 2);
    const maxDiff = Math.max(paddedLatDiff, paddedLonDiff);

    let zoom = 1.5;
    if (maxDiff < 0.1) zoom = 11;
    else if (maxDiff < 1) zoom = 7;
    else if (maxDiff < 5) zoom = 5;
    else if (maxDiff < 20) zoom = 3;
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
        mapStyle={MAP_STYLES[mapStyle].url}
        onError={onMapError}
        style={{ width: "100%", height: "100%" }}
      >
        <NavigationControl position="top-right" />

        {/* Map style selector */}
        <div className="absolute top-2 left-2 z-10">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="secondary" size="sm" className="shadow-md">
                {(() => {
                  const Icon = MAP_STYLES[mapStyle].icon;
                  return <Icon className="h-4 w-4 mr-2" />;
                })()}
                {MAP_STYLES[mapStyle].name}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              {(Object.keys(MAP_STYLES) as MapStyleKey[]).map((key) => {
                const style = MAP_STYLES[key];
                const Icon = style.icon;
                return (
                  <DropdownMenuItem key={key} onClick={() => setMapStyle(key)}>
                    <Icon className="h-4 w-4 mr-2" />
                    {style.name}
                  </DropdownMenuItem>
                );
              })}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

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
      <div className="absolute bottom-14 left-2 bg-white px-3 py-1.5 rounded shadow-md text-xs text-gray-600 max-w-[200px]">
        Click anywhere on the map to place your site marker
      </div>
    </div>
  );
}
