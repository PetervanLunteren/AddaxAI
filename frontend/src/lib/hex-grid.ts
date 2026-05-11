/**
 * Hexagonal grid utilities for the observation rate map.
 *
 * We compute hex vertices directly in Mercator PIXEL coordinates
 * (via Leaflet's `project` / `unproject`) instead of asking turf for
 * a degree-based grid. This is the only reliable way to get
 * visually-symmetric hexes at any latitude — turf's hex-grid uses a
 * single bbox-center latitude to convert km↔degrees and then applies
 * the same degree dimensions to every hex in the grid, so hexes
 * above and below the center end up stretched differently under
 * Mercator. Working in pixel space sidesteps the problem entirely.
 */

import { featureCollection, point } from "@turf/helpers";
import { pointsWithinPolygon } from "@turf/points-within-polygon";
import type { BBox, Feature, FeatureCollection, Polygon } from "geojson";
import type { LatLngBounds, Map as LeafletMap } from "leaflet";

import type { ObservationRateMapFeature } from "../api/statistics";

export interface HexCell {
  hex: Feature<Polygon>;
  sites: ObservationRateMapFeature[];
  trap_nights: number;
  observation_count: number;
  rate_per_100: number;
  site_count: number;
}

/**
 * On-screen hex radius (center to vertex, in pixels) for a given zoom.
 * Constant pixel size across zooms gives the map its even look.
 */
export function getHexRadiusPx(zoomLevel: number): number {
  // Reference values chosen so zoom=10 gets ~30px hexes.
  const referenceZoom = 10;
  const referenceRadiusPx = 30;
  const scale = Math.pow(2, zoomLevel - referenceZoom);
  // Clamp to sensible bounds so very high zooms don't explode.
  return Math.max(10, Math.min(120, referenceRadiusPx * Math.max(0.4, Math.min(2.5, scale))));
}

/**
 * Hard cap on the number of hexes the grid can contain. If the loop
 * would exceed this (some weird combination of radius and viewport),
 * we bail out cleanly instead of locking the browser.
 */
const MAX_HEXES = 20_000;

/**
 * Generate a hex grid that tiles the given map viewport in pixel
 * space, then un-projects each hex to lat/lon so it can be rendered
 * by Leaflet's GeoJSON layer.
 *
 * @param map         The live Leaflet map instance
 * @param bounds      The current viewport bounds
 * @param radiusPx    Hex radius in screen pixels (center to vertex)
 */
export function generateHexGrid(
  map: LeafletMap,
  bounds: LatLngBounds,
  radiusPx: number
): FeatureCollection<Polygon> {
  // Use the same zoom throughout so radius and projection stay in sync.
  const zoom = map.getZoom();

  // Defensive: if the radius or zoom are degenerate, bail early.
  if (!Number.isFinite(zoom) || !Number.isFinite(radiusPx) || radiusPx < 4) {
    return featureCollection([]);
  }

  // Flat-top hex geometry:
  //   width  (point to point, horizontal)  = 2 * r
  //   height (flat to flat, vertical)      = sqrt(3) * r
  // Tiling steps:
  //   x_step = 3/4 * width = 1.5 * r   (adjacent columns overlap)
  //   y_step = height = sqrt(3) * r    (adjacent rows in the same column)
  //   alternate columns are offset vertically by height / 2
  const r = radiusPx;
  const width = 2 * r;
  const height = Math.sqrt(3) * r;
  const xStep = 1.5 * r;
  const yStep = height;

  // Project viewport corners to pixel space (at the current zoom).
  const nw = map.project(bounds.getNorthWest(), zoom);
  const se = map.project(bounds.getSouthEast(), zoom);

  if (
    !Number.isFinite(nw.x) ||
    !Number.isFinite(nw.y) ||
    !Number.isFinite(se.x) ||
    !Number.isFinite(se.y)
  ) {
    return featureCollection([]);
  }

  // Pad by 1.5 × width / height so hexes clipped by edges are still
  // rendered (their centers sit outside the viewport but their body
  // overlaps it).
  const minX = Math.floor(nw.x - width * 1.5);
  const maxX = Math.ceil(se.x + width * 1.5);
  const minY = Math.floor(nw.y - height * 1.5);
  const maxY = Math.ceil(se.y + height * 1.5);

  // Anchor column index 0 at the first column <= minX.
  const firstCol = Math.floor(minX / xStep) - 1;
  const lastCol = Math.ceil(maxX / xStep) + 1;
  const firstRow = Math.floor(minY / yStep) - 1;
  const lastRow = Math.ceil(maxY / yStep) + 1;

  // Sanity: reject grids that would allocate too many hexes. A typical
  // viewport should have a few hundred to a few thousand. Anything
  // above MAX_HEXES is almost certainly a bug (bad bounds, zoom mismatch).
  const estimatedHexes = (lastCol - firstCol + 1) * (lastRow - firstRow + 1);
  if (estimatedHexes <= 0 || estimatedHexes > MAX_HEXES) {
    // eslint-disable-next-line no-console
    console.warn(
      `[hex-grid] Skipping grid: would allocate ${estimatedHexes} hexes ` +
        `(cols=${firstCol}..${lastCol}, rows=${firstRow}..${lastRow}, ` +
        `radiusPx=${radiusPx}, zoom=${zoom})`
    );
    return featureCollection([]);
  }

  const features: Feature<Polygon>[] = [];
  for (let col = firstCol; col <= lastCol; col++) {
    const cx = col * xStep;
    const isOdd = ((col % 2) + 2) % 2 === 1;
    const yOffset = isOdd ? yStep / 2 : 0;
    for (let row = firstRow; row <= lastRow; row++) {
      const cy = row * yStep + yOffset;

      const ring: [number, number][] = [];
      for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 3) * i; // flat-top: first vertex at the right
        const px = cx + r * Math.cos(angle);
        const py = cy + r * Math.sin(angle);
        const latlng = map.unproject([px, py], zoom);
        ring.push([latlng.lng, latlng.lat]);
      }
      ring.push(ring[0]); // close

      features.push({
        type: "Feature",
        properties: {},
        geometry: {
          type: "Polygon",
          coordinates: [ring],
        },
      });
    }
  }

  return featureCollection(features);
}

/**
 * Aggregate sites into the hex cells they fall inside. See module doc
 * for why the aggregation still happens in geographic coordinates —
 * the hex polygons are valid lat/lon after un-projection so turf's
 * point-in-polygon works unchanged.
 */
export function aggregateSitesToHexes(
  sites: ObservationRateMapFeature[],
  hexGridCollection: FeatureCollection<Polygon>
): HexCell[] {
  const sitePoints = featureCollection(
    sites.map((s) =>
      point([s.longitude, s.latitude], { site_id: s.site_id })
    )
  );

  const bySiteId = new Map<string, ObservationRateMapFeature>();
  for (const s of sites) bySiteId.set(s.site_id, s);

  const cells: HexCell[] = [];
  for (const hex of hexGridCollection.features) {
    const within = pointsWithinPolygon(sitePoints, hex);
    if (within.features.length === 0) continue;

    const members: ObservationRateMapFeature[] = [];
    let trapNights = 0;
    let observationCount = 0;

    for (const pt of within.features) {
      const id = (pt.properties as { site_id: string }).site_id;
      const site = bySiteId.get(id);
      if (!site) continue;
      members.push(site);
      trapNights += site.trap_nights;
      observationCount += site.observation_count;
    }

    const ratePer100 =
      trapNights > 0 ? (observationCount / trapNights) * 100 : 0;

    cells.push({
      hex,
      sites: members,
      trap_nights: trapNights,
      observation_count: observationCount,
      rate_per_100: ratePer100,
      site_count: members.length,
    });
  }

  return cells;
}

/**
 * Bounding box of all site points, in [minLon, minLat, maxLon, maxLat].
 * Returns world bounds if the input is empty.
 */
export function getSitesBounds(
  sites: ObservationRateMapFeature[]
): BBox {
  if (sites.length === 0) {
    return [-180, -90, 180, 90];
  }
  let minLon = Infinity;
  let maxLon = -Infinity;
  let minLat = Infinity;
  let maxLat = -Infinity;
  for (const s of sites) {
    if (s.longitude < minLon) minLon = s.longitude;
    if (s.longitude > maxLon) maxLon = s.longitude;
    if (s.latitude < minLat) minLat = s.latitude;
    if (s.latitude > maxLat) maxLat = s.latitude;
  }
  return [minLon, minLat, maxLon, maxLat];
}
