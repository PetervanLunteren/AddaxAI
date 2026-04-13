/**
 * Single-deployment marker for the observation rate map.
 *
 * Renders a CircleMarker colored by rate, with a hollow/dashed look
 * for deployments that have effort but zero observations so users can
 * still see them on the map.
 */

import { CircleMarker, Popup } from "react-leaflet";

import type { ObservationRateMapFeature } from "../../api/statistics";
import { DeploymentPopup } from "./DeploymentPopup";

interface DeploymentMarkerProps {
  feature: ObservationRateMapFeature;
  color: string;
}

export function DeploymentMarker({ feature, color }: DeploymentMarkerProps) {
  const isZero = feature.observation_count === 0;

  return (
    <CircleMarker
      center={[feature.latitude, feature.longitude]}
      radius={8}
      pathOptions={{
        fillColor: color,
        // Zero-observation deployments render as hollow outline
        // circles to make "effort but no observations" visually
        // distinct from low-rate deployments.
        fillOpacity: isZero ? 0 : 0.7,
        color: "#555555",
        weight: 1,
        opacity: 1,
      }}
    >
      <Popup>
        <DeploymentPopup feature={feature} />
      </Popup>
    </CircleMarker>
  );
}
