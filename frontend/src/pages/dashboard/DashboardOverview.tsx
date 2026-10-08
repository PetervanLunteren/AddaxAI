/**
 * Dashboard Overview: the whole project, no filters.
 *
 * A filter is a promise every card has to keep, and a filter nobody can
 * see is worse than none, so this tab has none at all. It holds what is
 * about the project as a whole: a random animal for the eye, how much was
 * recorded (files) and the effort behind it (trap nights, and when the
 * cameras ran), what lives there, and how much a person has checked.
 * Event and observation totals depend on which labels you count, so they
 * live on the Explore tab, next to the filter that decides that.
 */

import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";

import { statisticsApi } from "../../api/statistics";
import { AnimalPhotosCard } from "../../components/dashboard/AnimalPhotosCard";
import { EffortCalendarCard } from "../../components/dashboard/EffortCalendarCard";
import { MissingDatesBanner } from "../../components/dashboard/MissingDatesWarning";
import { StatTile } from "../../components/dashboard/StatTile";
import { VerificationProgressChart } from "../../components/dashboard/VerificationProgressChart";
import { WildlifeChart } from "../../components/dashboard/WildlifeChart";
import { formatCameraDate } from "../../lib/datetime";
import { shareNote } from "../../lib/share";

const plural = (n: number, word: string) =>
  `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
// The overview sends "YYYY-MM-DD HH:MM:SS"; the T makes it ISO.
const monthYear = (stamp: string) =>
  formatCameraDate(stamp.replace(" ", "T"), { month: "short", year: "numeric" });

export default function DashboardOverview() {
  const { projectId } = useParams<{ projectId: string }>();
  const id = projectId as string;

  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: ["statistics", "overview", id],
    queryFn: () => statisticsApi.getOverview(id),
  });
  const { data: categories } = useQuery({
    queryKey: ["statistics", "categories", id],
    queryFn: () => statisticsApi.getDetectionCategories(id),
  });

  // From the overview, which already works trap nights out; asking the
  // summary as well computed them twice and slowed big projects down.
  const trapNights = overview?.trap_nights ?? 0;
  const span =
    overview?.first_file_date && overview.last_file_date
      ? `${monthYear(overview.first_file_date)} to ${monthYear(overview.last_file_date)}`
      : null;

  return (
    <div className="space-y-6">
      <MissingDatesBanner projectId={id} />

      {/* Two columns that each flow on their own, as on Explore, so no
          card is stretched to a neighbour's height. Verification fills
          the right column to end level with the left; its list scrolls. */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="flex flex-col gap-6">
          {/* Hidden when nothing is confident enough. */}
          <AnimalPhotosCard projectId={id} variant="hero" />
          <WildlifeChart projectId={id} trapNights={trapNights} />
        </div>
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <StatTile
              label="Files"
              value={(overview?.total_files ?? 0).toLocaleString()}
              note={
                categories
                  ? shareNote([
                      { label: "animals", count: categories.animal_count },
                      { label: "people", count: categories.person_count },
                      { label: "vehicles", count: categories.vehicle_count },
                      { label: "empty", count: categories.empty_count },
                    ])
                  : undefined
              }
              loading={overviewLoading}
            />
            <StatTile
              label="Trap nights"
              value={trapNights.toLocaleString()}
              note={
                overview
                  ? [
                      // Only what exists: "0 sites" is noise.
                      overview.total_sites > 0 && plural(overview.total_sites, "site"),
                      overview.total_deployments > 0 &&
                        plural(overview.total_deployments, "deployment"),
                      span,
                    ]
                      .filter(Boolean)
                      .join(" · ")
                  : undefined
              }
              loading={overviewLoading}
            />
          </div>
          <EffortCalendarCard projectId={id} />
          <VerificationProgressChart projectId={id} className="flex-1" />
        </div>
      </div>
    </div>
  );
}
