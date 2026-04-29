/**
 * Popup shown when a user clicks a single-deployment marker.
 *
 * Shows site name + start date as the primary identifier (biologists
 * think in those terms, not in "camera 3 deployment 2"), effort and
 * rate metrics, and a per-species chip list colored via
 * getSpeciesColor so chips match the rest of the dashboard.
 */

import type { ObservationRateMapFeature } from "../../api/statistics";
import {
  getSpeciesColor,
  getSpeciesTextColor,
} from "../../utils/species-colors";

interface DeploymentPopupProps {
  feature: ObservationRateMapFeature;
}

export function DeploymentPopup({ feature }: DeploymentPopupProps) {
  const {
    site_name,
    start_date,
    end_date,
    trap_nights,
    observation_count,
    rate_per_100,
    species_breakdown,
  } = feature;

  const isActive = end_date === null;

  return (
    <div className="p-1 min-w-[220px]">
      <div className="font-semibold text-sm mb-1">{site_name}</div>
      <div className="text-xs text-gray-600 mb-2">
        {start_date} {end_date ? `to ${end_date}` : "(active)"}
      </div>

      <div className="space-y-1 text-xs">
        <div className="flex justify-between">
          <span className="text-gray-600">Trap nights</span>
          <span className="font-medium">{trap_nights}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-600">Observations</span>
          <span className="font-medium">{observation_count}</span>
        </div>
        <div className="flex justify-between border-t pt-1 mt-1">
          <span className="text-gray-600">Rate</span>
          <span className="font-semibold">
            {rate_per_100.toFixed(2)} / 100 trap nights
          </span>
        </div>
      </div>

      {species_breakdown.length > 0 && (
        <div className="mt-3 pt-2 border-t">
          <div className="text-xs font-semibold text-gray-700 mb-1">
            Top species
          </div>
          <div className="flex flex-wrap gap-1">
            {species_breakdown.slice(0, 8).map((sp) => {
              const bg = getSpeciesColor(sp.label_taxonomy_id ?? sp.label);
              const fg = getSpeciesTextColor(
                sp.label_taxonomy_id ?? sp.label
              );
              return (
                <span
                  key={`${sp.label_taxonomy_id ?? sp.label}`}
                  className="rounded px-1.5 py-0.5 text-[10px] font-medium"
                  style={{ backgroundColor: bg, color: fg }}
                >
                  {sp.label} · {sp.count}
                </span>
              );
            })}
          </div>
        </div>
      )}

      {isActive && (
        <div className="mt-2 text-[10px] text-blue-700 bg-blue-50 px-2 py-1 rounded">
          Currently active
        </div>
      )}
    </div>
  );
}
