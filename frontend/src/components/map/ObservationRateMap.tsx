/**
 * Observation rate map — shows deployments colored by their
 * "observations per 100 trap nights" rate.
 *
 * The three view modes (hexbins / points / clusters) and the base
 * layer are controlled via props so they can live in the MapPage's
 * filter bar alongside the data filters. All state lifted to the
 * parent: filters, view mode, base layer.
 *
 * Ported from AddaxAI-Connect's DetectionRateMap.tsx with these
 * changes for WebUI:
 *   - Flat ObservationRateMapFeature shape (no GeoJSON geometry)
 *   - Counts observations (MaxN per event), not raw detections
 *   - No internal controls: view mode / base layer / filters are
 *     all props from the parent
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import L, { latLngBounds } from "leaflet";
import { Info } from "lucide-react";
import { MapContainer, TileLayer, useMap, useMapEvents } from "react-leaflet";

import { statisticsApi } from "../../api/statistics";
import type {
  ObservationRateMapFeature,
  ObservationRateMapFilters,
} from "../../api/statistics";
import { calculateRateDomain, getRateColor } from "../../lib/heat-color-scale";
import { ClusterLayer } from "./ClusterLayer";
import { DeploymentMarker } from "./DeploymentMarker";
import { FullscreenControl } from "./FullscreenControl";
import { HexbinLayer } from "./HexbinLayer";
import { MapLegend } from "./MapLegend";
import type { BaseLayer, ViewMode } from "./MapFilterBar";

interface ObservationRateMapProps {
  projectId: string;
  filters: ObservationRateMapFilters;
  viewMode: ViewMode;
  baseLayer: BaseLayer;
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
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
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

function getTileLayer(base: BaseLayer) {
  switch (base) {
    case "satellite":
      return {
        url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution:
          "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
      };
    case "osm":
      return {
        url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      };
    case "positron":
    default:
      return {
        url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      };
  }
}

export function ObservationRateMap({
  projectId,
  filters,
  viewMode,
  baseLayer,
}: ObservationRateMapProps) {
  const [zoomLevel, setZoomLevel] = useState(12);
  const [mapBounds, setMapBounds] = useState<L.LatLngBounds | null>(null);

  const handleZoomChange = useCallback((zoom: number) => setZoomLevel(zoom), []);
  const handleBoundsChange = useCallback(
    (bounds: L.LatLngBounds) => setMapBounds(bounds),
    []
  );

  const { data, isLoading, error } = useQuery({
    queryKey: ["observation-rate-map", projectId, filters],
    queryFn: () => statisticsApi.getObservationRateMap(projectId, filters),
    enabled: !!projectId,
  });

  const features: ObservationRateMapFeature[] = data?.features ?? [];

  const visibleDeployments = useMemo(() => {
    if (!mapBounds) return features;
    return features.filter((f) => mapBounds.contains([f.latitude, f.longitude]));
  }, [features, mapBounds]);

  const colorDomain = useMemo(() => {
    if (visibleDeployments.length === 0) {
      return { min: 0, max: 0, p33: 0, p66: 0 };
    }
    return calculateRateDomain(visibleDeployments.map((f) => f.rate_per_100));
  }, [visibleDeployments]);

  const mapCenter = useMemo<[number, number]>(() => {
    if (features.length === 0) return [52.0, 5.0];
    const avgLat =
      features.reduce((sum, f) => sum + f.latitude, 0) / features.length;
    const avgLon =
      features.reduce((sum, f) => sum + f.longitude, 0) / features.length;
    return [avgLat, avgLon];
  }, [features]);

  const fitBoundsPoints = useMemo<[number, number][]>(
    () => features.map((f) => [f.latitude, f.longitude] as [number, number]),
    [features]
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

  if (features.length === 0) {
    return (
      <div className="rounded-lg border bg-card p-8 text-center space-y-2">
        <div className="text-sm font-medium text-foreground">
          No deployments to show
        </div>
        <div className="text-sm text-muted-foreground max-w-xl mx-auto">
          No deployments match these filters. Try clearing them, or run an
          analysis on the Analyses page to populate the map.
        </div>
      </div>
    );
  }

  const tile = getTileLayer(baseLayer);

  return (
    <div className="flex h-[600px] flex-col overflow-hidden rounded-lg border bg-card">
      <MapContainer
        center={mapCenter}
        zoom={12}
        style={{ width: "100%" }}
        className="min-h-0 flex-1"
      >
        <TileLayer
          key={baseLayer}
          attribution={tile.attribution}
          url={tile.url}
        />

        <MapEventHandler
          onZoomChange={handleZoomChange}
          onBoundsChange={handleBoundsChange}
        />
        <FitBounds points={fitBoundsPoints} />

        {viewMode === "points" &&
          visibleDeployments.map((feature) => (
            <DeploymentMarker
              key={feature.deployment_id}
              feature={feature}
              color={getRateColor(feature.rate_per_100, colorDomain.p66)}
            />
          ))}

        {viewMode === "clusters" && (
          <ClusterLayer
            deployments={visibleDeployments}
            maxRate={colorDomain.p66}
          />
        )}

        {viewMode === "hexbins" && mapBounds && (
          <HexbinLayer
            deployments={visibleDeployments}
            zoomLevel={zoomLevel}
            viewportBounds={mapBounds}
            maxRate={colorDomain.p66}
          />
        )}

        <MapLegend domain={colorDomain} />
        <FullscreenControl />
      </MapContainer>

      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t px-4 py-3 text-xs text-muted-foreground">
        <Info className="h-3.5 w-3.5" />
        <span>
          {visibleDeployments.length} deployment
          {visibleDeployments.length === 1 ? "" : "s"} shown
        </span>
        <span aria-hidden="true">·</span>
        <span>Rate is observations (MaxN per event) per 100 trap nights</span>
      </div>
    </div>
  );
}
