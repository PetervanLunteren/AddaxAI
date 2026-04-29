/**
 * Popup shown when a user clicks a hexagon. Aggregates all
 * deployments within the hex and lists them with their individual
 * rates.
 */

import type { HexCell } from "../../lib/hex-grid";

interface HexPopupProps {
  hexCell: HexCell;
}

export function HexPopup({ hexCell }: HexPopupProps) {
  const {
    trap_nights,
    observation_count,
    rate_per_100,
    deployment_count,
    deployments,
  } = hexCell;

  return (
    <div className="p-2 min-w-[280px] max-w-[400px]">
      <div className="mb-3 pb-2 border-b border-gray-200">
        <div className="font-semibold text-gray-900 mb-1 text-sm">
          Aggregated metrics
        </div>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-600">Deployments</span>
            <span className="font-medium">{deployment_count}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Total trap nights</span>
            <span className="font-medium">{trap_nights}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Total observations</span>
            <span className="font-medium">{observation_count}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Rate</span>
            <span className="font-medium">
              {rate_per_100.toFixed(2)} / 100 trap nights
            </span>
          </div>
        </div>
      </div>

      <div>
        <div className="font-semibold text-gray-900 mb-2 text-sm">
          {deployments.length === 1 ? "Deployment" : "Deployments"} (
          {deployments.length})
        </div>
        <div className="max-h-[200px] overflow-y-auto space-y-2">
          {deployments.map((dep) => {
            const isZero = dep.observation_count === 0;
            return (
              <div
                key={dep.deployment_id}
                className="p-2 bg-gray-50 rounded text-[11px] space-y-0.5"
              >
                <div className="font-medium text-gray-900">{dep.site_name}</div>
                <div className="text-gray-600">
                  {dep.start_date_local}
                  {dep.end_date_local ? ` to ${dep.end_date_local}` : " (active)"}
                </div>
                <div className="flex justify-between text-gray-700">
                  <span>Trap nights: {dep.trap_nights}</span>
                  <span>
                    Obs: {dep.observation_count}
                    {isZero && (
                      <span className="text-gray-500 ml-1">(empty)</span>
                    )}
                  </span>
                </div>
                {!isZero && (
                  <div className="text-gray-700">
                    Rate: {dep.rate_per_100.toFixed(2)} / 100 trap nights
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
