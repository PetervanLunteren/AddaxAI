/**
 * Warnings for media files with no capture date.
 *
 * Folder mode is data-agnostic: images with no EXIF capture date are
 * ingested but can't appear in time-based views (activity, trends,
 * trap-night rates). Two surfaces share one self-fetched count:
 * - `MissingDatesBanner`: page-wide amber notice for the systemic
 *   impact (trap-nights feed almost every rate), used on the dashboard
 *   and the time-based insights pages.
 * - `MissingDatesIcon`: a small amber icon for a single purely
 *   time-axis card, marking that its data excludes the date-less files.
 *
 * Both render nothing when every media file has a capture date.
 */
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";

import { statisticsApi } from "../../api/statistics";
import { Callout } from "../ui/callout";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";

function useCaptureDateCoverage(projectId: string | undefined) {
  return useQuery({
    queryKey: ["capture-date-coverage", projectId],
    queryFn: () => statisticsApi.getCaptureDateCoverage(projectId!),
    enabled: !!projectId,
    staleTime: 60_000,
  });
}

export function MissingDatesBanner({
  projectId,
}: {
  projectId: string | undefined;
}) {
  const { data } = useCaptureDateCoverage(projectId);
  if (!data || data.without_date === 0) return null;
  const dated = data.total - data.without_date;
  return (
    <Callout variant="warning">
      {data.without_date.toLocaleString()} of {data.total.toLocaleString()}{" "}
      images have no capture date. Time-based metrics (activity, trends,
      trap-night rates) cover only the {dated.toLocaleString()} dated ones.
    </Callout>
  );
}

export function MissingDatesIcon({
  projectId,
}: {
  projectId: string | undefined;
}) {
  const { data } = useCaptureDateCoverage(projectId);
  if (!data || data.without_date === 0) return null;
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex">
            <AlertTriangle className="h-4 w-4 text-yellow-600" />
          </span>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">
          Excludes {data.without_date.toLocaleString()} image
          {data.without_date === 1 ? "" : "s"} with no capture date.
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
