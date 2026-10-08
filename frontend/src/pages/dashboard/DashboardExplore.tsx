/**
 * Dashboard Explore: every label, or the labels you pick, at the sites
 * and dates the filter bar says.
 *
 * Every card gets the same `scope` from `useExploreFilters`, so one filter
 * drives all of them and no card can quietly count something else. Leads
 * with photographs, because once a taxon is named, looking at it is the
 * fastest check of whether the model is right, and every photo opens the
 * file on the Labels page where the label can be corrected.
 */

import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";

import { statisticsApi } from "../../api/statistics";
import { ActivityPatternChart } from "../../components/dashboard/ActivityPatternChart";
import { AnimalPhotosCard } from "../../components/dashboard/AnimalPhotosCard";
import { DemographicsCard } from "../../components/dashboard/DemographicsCard";
import { DetectionTrendChart } from "../../components/dashboard/DetectionTrendChart";
import { ExploreFilterBar } from "../../components/dashboard/ExploreFilterBar";
import { MiniMapCard } from "../../components/dashboard/MiniMapCard";
import { StatTile } from "../../components/dashboard/StatTile";
import { Callout } from "../../components/ui/callout";
import { insightsMapHref } from "../../lib/explore-filters";
import { NO_SITE_SENTINEL } from "../../lib/filter-url";
import { useExploreFilters } from "./useExploreFilters";

export default function DashboardExplore() {
  const { projectId } = useParams<{ projectId: string }>();
  const id = projectId as string;
  const explore = useExploreFilters(id);
  const { scope, ready } = explore;
  const showCards = ready && !explore.noSiteMatch;

  const { data: summary, isLoading } = useQuery({
    queryKey: ["statistics", "summary", id, scope],
    queryFn: () => statisticsApi.getDashboardSummary(id, scope),
    enabled: showCards,
  });
  const trapNights = summary?.trap_nights ?? 0;
  // Each count says what it is out of: the effort behind it.
  // No note without effort: zero trap nights has several causes (no
  // files, no capture dates, a date range with no cameras running), and
  // the Trap nights tile beside it already shows the 0.
  const perHundred = (n: number) =>
    trapNights > 0
      ? `${((n / trapNights) * 100).toLocaleString(undefined, { maximumFractionDigits: 1 })} per 100 trap nights`
      : undefined;
  // Sites in view: the picked ones, else all of the project's.
  const sitesInView =
    scope.siteIds?.filter((s) => s !== NO_SITE_SENTINEL).length ??
    explore.sites?.length ??
    0;

  return (
    <div className="space-y-6">
      <ExploreFilterBar
        projectId={id}
        filters={explore.filters}
        scopeSiteIds={scope.siteIds}
        sites={explore.sites}
        update={explore.update}
        clearAll={explore.clearAll}
      />

      {explore.noSiteMatch ? (
        <Callout variant="info">
          No site has these tags. Pick other sites or tags to see results.
        </Callout>
      ) : !ready ? (
        <p className="text-sm text-muted-foreground">Loading...</p>
      ) : (
        // Two columns that each flow on their own, so a short card lets
        // the one below it rise instead of leaving a gap.
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="flex flex-col gap-6">
            <AnimalPhotosCard projectId={id} variant="wall" scope={scope} />
            <ActivityPatternChart
              projectId={id}
              scope={scope}
              trapNights={trapNights}
            />
            <DemographicsCard projectId={id} scope={scope} />
          </div>
          <div className="flex flex-col gap-6">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <StatTile
                label="Events"
                value={(summary?.events ?? 0).toLocaleString()}
                note={summary ? perHundred(summary.events) : undefined}
                loading={isLoading}
              />
              <StatTile
                label="Observations"
                value={(summary?.observations ?? 0).toLocaleString()}
                note={summary ? perHundred(summary.observations) : undefined}
                loading={isLoading}
              />
              <StatTile
                label="Trap nights"
                value={trapNights.toLocaleString()}
                // A project without sites has nothing to say "of".
                note={
                  summary && sitesInView > 0
                    ? `Detected at ${summary.sites_with_detections.toLocaleString()} of ${sitesInView.toLocaleString()} sites`
                    : undefined
                }
                loading={isLoading}
              />
            </div>
            <DetectionTrendChart
              projectId={id}
              scope={scope}
              trapNights={trapNights}
            />
            <MiniMapCard
              projectId={id}
              scope={scope}
              mapHref={insightsMapHref(id, scope)}
              className="flex-1"
            />
          </div>
        </div>
      )}
    </div>
  );
}
