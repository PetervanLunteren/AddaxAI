/**
 * Cluster layer for the observation rate map.
 *
 * Groups nearby sites into single bubbles colored by the
 * effort-weighted rate across the cluster. Clicking a cluster
 * zooms in to reveal the individual site markers.
 */

import L from "leaflet";
import MarkerClusterGroup from "react-leaflet-cluster";

import type { ObservationRateMapFeature } from "../../api/statistics";
import { getRateColor } from "../../lib/heat-color-scale";
import { SiteMarker } from "./SiteMarker";

interface ClusterLayerProps {
  sites: ObservationRateMapFeature[];
  maxRate: number;
}

export function ClusterLayer({ sites, maxRate }: ClusterLayerProps) {
  // Index by coordinate so the cluster icon can look up member features.
  const coordsToFeature = new Map<string, ObservationRateMapFeature>();
  for (const s of sites) {
    coordsToFeature.set(`${s.latitude},${s.longitude}`, s);
  }

  const createClusterCustomIcon = (cluster: L.MarkerCluster) => {
    const markers = cluster.getAllChildMarkers();
    let totalObs = 0;
    let totalNights = 0;

    for (const marker of markers) {
      const latlng = (marker as L.CircleMarker).getLatLng();
      const feature = coordsToFeature.get(`${latlng.lat},${latlng.lng}`);
      if (feature) {
        totalObs += feature.observation_count;
        totalNights += feature.trap_nights;
      }
    }

    const overallRate = totalNights > 0 ? (totalObs / totalNights) * 100 : 0;
    const color = getRateColor(overallRate, maxRate);
    // Clusters made entirely of zero-observation sites render hollow
    // (transparent fill, dashed border, grey text) to stay consistent
    // with the hex + point rendering.
    const isEmpty = totalObs === 0;

    const background = isEmpty ? "transparent" : color;
    const textColor = isEmpty ? "#555555" : "white";
    const textShadow = isEmpty ? "none" : "0 0 2px rgba(0,0,0,0.5)";

    return L.divIcon({
      html: `<div style="
        background-color: ${background};
        width: 40px;
        height: 40px;
        border-radius: 50%;
        border: 2px solid #555555;
        display: flex;
        align-items: center;
        justify-content: center;
        color: ${textColor};
        font-weight: bold;
        font-size: 14px;
        text-shadow: ${textShadow};
      ">${Math.round(overallRate)}</div>`,
      className: "custom-cluster-icon",
      iconSize: L.point(40, 40, true),
    });
  };

  return (
    <MarkerClusterGroup
      iconCreateFunction={createClusterCustomIcon}
      maxClusterRadius={50}
      spiderfyOnMaxZoom={true}
      showCoverageOnHover={false}
      zoomToBoundsOnClick={true}
    >
      {sites.map((feature) => (
        <SiteMarker
          key={feature.site_id}
          feature={feature}
          color={getRateColor(feature.rate_per_100, maxRate)}
        />
      ))}
    </MarkerClusterGroup>
  );
}
