/**
 * Observation rate map: shows sites colored by their observation rate.
 * Colours come from the backend's rate per 100 trap nights; they are
 * normalised against the visible sites, so the rate unit cannot change
 * them.
 *
 * A site can have multiple deployments over time. Each marker
 * aggregates across the site's deployments that pass the active
 * filters. The three view modes (hexbins / points / clusters) and the
 * base layer are controlled via props so they can live in the
 * MapPage's filter bar alongside the data filters. All state lifted
 * to the parent: the data (the page fetches it, so its Download CSV
 * saves the same sites), view mode, base layer.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import L, { latLngBounds } from "leaflet";
import { Info } from "lucide-react";
import { LayersControl, MapContainer, TileLayer, useMap, useMapEvents } from "react-leaflet";

import { BASE_LAYERS, isBaseLayer, type BaseLayer } from "./basemap-styles";
import { StreetBaseLayer } from "./StreetBaseLayer";

import type {
  ObservationRateMapFeature,
  ObservationRateMapResponse,
} from "../../api/statistics";
import { calculateRateDomain, getRateColor, type RateRamp } from "../../lib/heat-color-scale";
import { ClusterLayer } from "./ClusterLayer";
import { FullscreenControl } from "./FullscreenControl";
import { HexbinLayer } from "./HexbinLayer";
import { MapLegend } from "./MapLegend";
import { SiteMarker } from "./SiteMarker";
import type { ViewMode } from "./MapFilterBar";

interface ObservationRateMapProps {
  data: ObservationRateMapResponse | undefined;
  isLoading: boolean;
  error: Error | null;
  viewMode: ViewMode;
  ramp: RateRamp;
}

const BASE_LAYER_KEY = "addaxai:map-base-layer";

function readStoredBaseLayer(): BaseLayer {
  const saved = localStorage.getItem(BASE_LAYER_KEY);
  return isBaseLayer(saved) ? saved : "positron";
}

/** Reports the background picked in the layers control. */
function BaseLayerListener({ onChange }: { onChange: (layer: BaseLayer) => void }) {
  useMapEvents({
    baselayerchange: (event) => {
      const picked = BASE_LAYERS.find((layer) => layer.name === event.name);
      if (picked) onChange(picked.id);
    },
  });
  return null;
}

/**
 * Tracks zoom/pan events with a debounce so hex regeneration and
 * viewport filtering don't thrash. Special-cases zoomend because
 * Leaflet's zoom animation also fires moveend, which we want to
 * suppress within 5 seconds of a zoom.
 */
function MapEventHandler({
  onZoomChange,
  onBoundsChange,
}: {
  onZoomChange: (zoom: number) => void;
  onBoundsChange: (bounds: L.LatLngBounds) => void;
}) {
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const lastZoomTimeRef = useRef<number>(0);

  const map = useMapEvents({
    zoomend: (e) => {
      const zoom = e.target.getZoom();
      if (debounceRef.current) clearTimeout(debounceRef.current);
      lastZoomTimeRef.current = Date.now();
      debounceRef.current = setTimeout(() => {
        onZoomChange(zoom);
        onBoundsChange(e.target.getBounds());
      }, 300);
    },
    moveend: (e) => {
      if (Date.now() - lastZoomTimeRef.current < 5000) return;
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        onBoundsChange(e.target.getBounds());
      }, 300);
    },
  });

  useEffect(() => {
    onZoomChange(map.getZoom());
    onBoundsChange(map.getBounds());
  }, [map, onZoomChange, onBoundsChange]);

  return null;
}

function FitBounds({ points }: { points: [number, number][] }) {
  const map = useMap();
  const fitted = useRef(false);
  useEffect(() => {
    if (points.length === 0 || fitted.current) return;
    map.fitBounds(latLngBounds(points), { padding: [20, 20] });
    fitted.current = true;
  }, [points, map]);
  return null;
}

export function ObservationRateMap({
  data,
  isLoading,
  error,
  viewMode,
  ramp,
}: ObservationRateMapProps) {
  // The background is the map's own choice, made in its layers control
  // and remembered per device. The state follows the control: the
  // street map is rebuilt when the theme changes, and a stale "checked"
  // would put it back on top of the chosen background.
  const [baseLayer, setBaseLayer] = useState<BaseLayer>(readStoredBaseLayer);
  const handleBaseLayerChange = useCallback((layer: BaseLayer) => {
    setBaseLayer(layer);
    localStorage.setItem(BASE_LAYER_KEY, layer);
  }, []);

  const [zoomLevel, setZoomLevel] = useState(12);
  const [mapBounds, setMapBounds] = useState<L.LatLngBounds | null>(null);

  const handleZoomChange = useCallback((zoom: number) => setZoomLevel(zoom), []);
  const handleBoundsChange = useCallback(
    (bounds: L.LatLngBounds) => setMapBounds(bounds),
    []
  );

  const sites: ObservationRateMapFeature[] = data?.features ?? [];

  const visibleSites = useMemo(() => {
    if (!mapBounds) return sites;
    return sites.filter((f) => mapBounds.contains([f.latitude, f.longitude]));
  }, [sites, mapBounds]);

  const colorDomain = useMemo(() => {
    if (visibleSites.length === 0) {
      return { max: 0, cap: 0 };
    }
    return calculateRateDomain(visibleSites.map((f) => f.rate_per_100));
  }, [visibleSites]);

  const mapCenter = useMemo<[number, number]>(() => {
    if (sites.length === 0) return [52.0, 5.0];
    const avgLat =
      sites.reduce((sum, f) => sum + f.latitude, 0) / sites.length;
    const avgLon =
      sites.reduce((sum, f) => sum + f.longitude, 0) / sites.length;
    return [avgLat, avgLon];
  }, [sites]);

  const fitBoundsPoints = useMemo<[number, number][]>(
    () => sites.map((f) => [f.latitude, f.longitude] as [number, number]),
    [sites]
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-[600px] rounded-lg border bg-card">
        <div className="text-muted-foreground">Loading map data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-[600px] rounded-lg border bg-card">
        <div className="text-destructive">
          Failed to load map:{" "}
          {error instanceof Error ? error.message : "unknown error"}
        </div>
      </div>
    );
  }

  if (sites.length === 0) {
    return (
      <div className="rounded-lg border bg-card p-8 text-center space-y-2">
        <div className="text-sm font-medium text-foreground">
          No sites to show
        </div>
        <div className="text-sm text-muted-foreground max-w-xl mx-auto">
          No sites match these filters. Try clearing them, or run an analysis
          on the Analyses page to populate the map.
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-[600px] flex-col overflow-hidden rounded-lg border bg-card">
      <MapContainer
        center={mapCenter}
        zoom={12}
        // Set on the container, not only the layers: the zoom range
        // must be bounded before FitBounds runs on a single point, or
        // Leaflet zooms to Infinity and MapLibre crashes on init.
        maxZoom={18}
        style={{ width: "100%" }}
        className="min-h-0 flex-1"
      >
        {/* The backgrounds: the shared street map (OpenFreeMap vector,
            OSM raster without WebGL2) and the raster ones. */}
        <LayersControl position="topright">
          {BASE_LAYERS.map((layer) => (
            <LayersControl.BaseLayer
              key={layer.id}
              name={layer.name}
              checked={layer.id === baseLayer}
            >
              {"url" in layer ? (
                <TileLayer
                  url={layer.url}
                  attribution={layer.attribution}
                  maxNativeZoom={"maxNativeZoom" in layer ? layer.maxNativeZoom : undefined}
                />
              ) : (
                <StreetBaseLayer />
              )}
            </LayersControl.BaseLayer>
          ))}
        </LayersControl>
        <BaseLayerListener onChange={handleBaseLayerChange} />

        <MapEventHandler
          onZoomChange={handleZoomChange}
          onBoundsChange={handleBoundsChange}
        />
        <FitBounds points={fitBoundsPoints} />

        {viewMode === "points" &&
          visibleSites.map((feature) => (
            <SiteMarker
              key={feature.site_id}
              feature={feature}
              color={getRateColor(feature.rate_per_100, colorDomain.cap, ramp)}
            />
          ))}

        {viewMode === "clusters" && (
          <ClusterLayer
            sites={visibleSites}
            maxRate={colorDomain.cap}
            ramp={ramp}
          />
        )}

        {viewMode === "hexbins" && mapBounds && (
          <HexbinLayer
            sites={visibleSites}
            zoomLevel={zoomLevel}
            viewportBounds={mapBounds}
            maxRate={colorDomain.cap}
            ramp={ramp}
          />
        )}

        <MapLegend domain={colorDomain} ramp={ramp} />
        <FullscreenControl />
      </MapContainer>

      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t px-4 py-3 text-xs text-muted-foreground">
        <Info className="h-3.5 w-3.5" />
        <span>
          {visibleSites.length} site{visibleSites.length === 1 ? "" : "s"} shown
        </span>
      </div>
    </div>
  );
}
