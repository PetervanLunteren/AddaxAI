/**
 * Popup shown when a user clicks a hexagon. Aggregates all sites
 * within the hex and lists them alphabetically with their individual
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
    site_count,
    sites,
  } = hexCell;

  const orderedSites = [...sites].sort((a, b) =>
    a.site_name.localeCompare(b.site_name, undefined, { sensitivity: "base" })
  );

  return (
    <div className="p-2 min-w-[280px] max-w-[400px]">
      <div className="mb-3 pb-2 border-b border-border">
        <div className="font-semibold text-foreground mb-1 text-sm">
          Aggregated metrics
        </div>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Sites</span>
            <span className="font-medium">{site_count}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Total trap nights</span>
            <span className="font-medium">{trap_nights}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Total observations</span>
            <span className="font-medium">{observation_count}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Rate</span>
            <span className="font-medium">
              {rate_per_100.toFixed(2)} / 100 trap nights
            </span>
          </div>
        </div>
      </div>

      <div>
        <div className="font-semibold text-foreground mb-2 text-sm">
          {orderedSites.length === 1 ? "Site" : "Sites"} ({orderedSites.length})
        </div>
        <div className="max-h-[200px] overflow-y-auto space-y-2">
          {orderedSites.map((site) => {
            const isZero = site.observation_count === 0;
            const depSuffix =
              site.deployment_count > 1
                ? ` (${site.deployment_count} deployments)`
                : "";
            return (
              <div
                key={site.site_id}
                className="p-2 bg-muted rounded text-[11px] space-y-0.5"
              >
                <div className="font-medium text-foreground">
                  {site.site_name}
                  <span className="text-muted-foreground font-normal">{depSuffix}</span>
                </div>
                <div className="flex justify-between text-muted-foreground">
                  <span>Trap nights: {site.trap_nights}</span>
                  <span>
                    Obs: {site.observation_count}
                    {isZero && (
                      <span className="text-muted-foreground ml-1">(empty)</span>
                    )}
                  </span>
                </div>
                {!isZero && (
                  <div className="text-muted-foreground">
                    Rate: {site.rate_per_100.toFixed(2)} / 100 trap nights
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
