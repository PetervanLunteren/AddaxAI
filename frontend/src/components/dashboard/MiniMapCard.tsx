/**
 * A small, still map of where the Explore selection was observed, ported
 * from AddaxAI Connect.
 *
 * Every site is a soft blob coloured by observations per 100 trap nights,
 * from the same endpoint and colour scale as the Insights map, with the
 * same labels, sites and dates, so the two can never disagree. It is deliberately dead as an instrument: no zoom, no
 * pan, no popups. The whole card opens the full map with the same labels,
 * sites and dates.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { latLngBounds } from "leaflet";
import { CircleMarker, MapContainer } from "react-leaflet";
import { Link } from "react-router-dom";

import { statisticsApi, type DashboardScope } from "../../api/statistics";
import { calculateRateDomain, getRateColor } from "../../lib/heat-color-scale";
import { cn } from "../../lib/utils";
import { StreetBaseLayer } from "../map/StreetBaseLayer";
import { Card, CardContent } from "../ui/card";
import { DashboardCardHeader } from "./DashboardCardHeader";

interface MiniMapCardProps {
  projectId: string;
  scope: DashboardScope;
  /** The full map with the same filters. */
  mapHref: string;
  className?: string;
}

export function MiniMapCard({
  projectId,
  scope,
  mapHref,
  className,
}: MiniMapCardProps) {
  const { data, isLoading } = useQuery({
    queryKey: ["statistics", "observation-rate-map", projectId, scope],
    queryFn: () => statisticsApi.getObservationRateMap(projectId, scope),
  });

  const features = useMemo(() => data?.features ?? [], [data]);
  const maxRate = useMemo(
    () => calculateRateDomain(features.map((f) => f.rate_per_100)).p66,
    [features],
  );
  const bounds = useMemo(
    () =>
      features.length > 0
        ? latLngBounds(features.map((f) => [f.latitude, f.longitude]))
        : null,
    [features],
  );

  return (
    <Card className={cn("group relative flex flex-col", className)}>
      <DashboardCardHeader
        title="Where"
        caption="Darker sites have more observations per 100 trap nights"
      />
      <CardContent className="flex flex-1 flex-col">
        {isLoading ? (
          <div className="min-h-56 flex-1 animate-pulse rounded-md bg-muted" />
        ) : bounds === null ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No sites with a location in this selection
          </p>
        ) : (
          <div className="min-h-56 flex-1 overflow-hidden rounded-md border">
            <MapContainer
              // Bounds only apply on mount, so a new set of sites
              // remounts the map to fit them.
              key={features.map((f) => f.site_id).join(",")}
              bounds={bounds}
              boundsOptions={{ padding: [24, 24], maxZoom: 12 }}
              maxZoom={18}
              style={{ height: "100%", width: "100%", minHeight: "14rem" }}
              zoomControl={false}
              dragging={false}
              scrollWheelZoom={false}
              doubleClickZoom={false}
              touchZoom={false}
              boxZoom={false}
              keyboard={false}
            >
              <StreetBaseLayer />
              {features.map((f) => {
                const color = getRateColor(f.rate_per_100, maxRate);
                return [16, 5].map((radius) => (
                  <CircleMarker
                    key={`${f.site_id}-${radius}`}
                    center={[f.latitude, f.longitude]}
                    radius={radius}
                    interactive={false}
                    pathOptions={{
                      stroke: false,
                      fillColor: color,
                      fillOpacity: radius === 16 ? 0.3 : 0.85,
                    }}
                  />
                ));
              })}
            </MapContainer>
          </div>
        )}
        {/* Without coordinates a deployment cannot be drawn, so its
            observations are in the tiles but not on the map. Say so
            rather than let the two disagree silently. */}
        {!isLoading && (data?.deployments_without_site ?? 0) > 0 && (
          <p className="mt-2 text-xs text-muted-foreground">
            {data!.deployments_without_site === 1
              ? "1 deployment has no site and is not on the map."
              : `${data!.deployments_without_site} deployments have no site and are not on the map.`}
          </p>
        )}
      </CardContent>
      {bounds !== null && (
        <Link
          to={mapHref}
          aria-label="Open the full map"
          className="absolute inset-0 z-[500] rounded-lg"
        >
          <span className="absolute bottom-8 right-8 rounded-full border bg-background/90 px-3 py-1 text-xs opacity-0 shadow-sm transition-opacity group-hover:opacity-100">
            Open the full map
          </span>
        </Link>
      )}
    </Card>
  );
}
