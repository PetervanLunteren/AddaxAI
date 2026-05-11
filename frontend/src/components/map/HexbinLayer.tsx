/**
 * Hexbin layer for the observation rate map.
 *
 * Generates a viewport-filling hex grid directly in Mercator pixel
 * space, aggregates sites into cells, and renders the result as a
 * Leaflet GeoJSON layer with a gradient fill per cell.
 */

import { useCallback, useMemo } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { GeoJSON, useMap } from "react-leaflet";
import { featureCollection } from "@turf/helpers";
import type { Feature, FeatureCollection, Polygon } from "geojson";
import type { LatLngBounds, Layer } from "leaflet";

import type { ObservationRateMapFeature } from "../../api/statistics";
import { getRateColor } from "../../lib/heat-color-scale";
import {
  aggregateSitesToHexes,
  generateHexGrid,
  getHexRadiusPx,
  type HexCell,
} from "../../lib/hex-grid";
import { HexPopup } from "./HexPopup";

interface HexbinLayerProps {
  sites: ObservationRateMapFeature[];
  zoomLevel: number;
  /**
   * Current viewport bounds. The Leaflet LatLngBounds object is
   * cached in the parent so its identity change triggers regeneration.
   */
  viewportBounds: LatLngBounds;
  maxRate?: number;
}

interface HexFeatureProperties {
  hexCell: HexCell;
  color: string;
}

export function HexbinLayer({
  sites,
  zoomLevel,
  viewportBounds,
  maxRate,
}: HexbinLayerProps) {
  const map = useMap();

  const hexCells = useMemo(() => {
    if (sites.length === 0) return [];
    const radiusPx = getHexRadiusPx(zoomLevel);
    const grid = generateHexGrid(map, viewportBounds, radiusPx);
    return aggregateSitesToHexes(sites, grid);
  }, [sites, zoomLevel, viewportBounds, map]);

  const effectiveMax = useMemo(() => {
    if (maxRate !== undefined) return maxRate;
    if (hexCells.length === 0) return 0;
    return Math.max(...hexCells.map((c) => c.rate_per_100));
  }, [hexCells, maxRate]);

  const collection = useMemo<
    FeatureCollection<Polygon, HexFeatureProperties>
  >(() => {
    const features = hexCells.map((cell) => ({
      ...cell.hex,
      properties: {
        hexCell: cell,
        color: getRateColor(cell.rate_per_100, effectiveMax),
      },
    }));
    return featureCollection(features) as FeatureCollection<
      Polygon,
      HexFeatureProperties
    >;
  }, [hexCells, effectiveMax]);

  const styleFunction = useCallback(
    (feature: Feature<Polygon, HexFeatureProperties> | undefined) => {
      const props = feature?.properties;
      if (!props) return {};
      const isEmpty = props.hexCell.observation_count === 0;
      return {
        fillColor: props.color,
        // Empty hexes render as outline-only so "effort but no
        // observations" reads differently from "low rate".
        fillOpacity: isEmpty ? 0 : 0.8,
        color: "#555555",
        weight: 1,
        opacity: 0.8,
      };
    },
    []
  );

  const onEachFeature = useCallback(
    (feature: Feature<Polygon, HexFeatureProperties>, layer: Layer) => {
      const html = renderToStaticMarkup(<HexPopup hexCell={feature.properties.hexCell} />);
      layer.bindPopup(html);
    },
    []
  );

  if (hexCells.length === 0) return null;

  return (
    <GeoJSON
      key={`hexbin-${zoomLevel}-${hexCells.length}`}
      data={collection}
      style={styleFunction}
      onEachFeature={onEachFeature}
    />
  );
}
