/**
 * One progress bar for project-wide observations verified, plus a
 * per-label breakdown below. The three-row design (Events / Media /
 * Observations) is gone: every Verify-page surface now reports the
 * same "percent observations verified" number, so showing three
 * different counts here was misleading.
 */

import { useQuery } from "@tanstack/react-query";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import { Separator } from "../ui/separator";
import { DashboardAboutPopover } from "./DashboardAboutPopover";
import { eventsApi } from "../../api/events";
import { statisticsApi } from "../../api/statistics";
import { resolveSpeciesName } from "../../lib/species-name-mode";

interface VerificationProgressChartProps {
  projectId: string;
  siteIds?: string;
  dateFrom?: string;
  dateTo?: string;
}

interface BarRow {
  label: string;
  verified: number;
  total: number;
}

function SlimProgressRow({ label, verified, total }: BarRow) {
  const pct = total > 0 ? (verified / total) * 100 : 0;

  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-baseline text-sm">
        <span className="truncate pr-2">{label}</span>
        <span className="text-muted-foreground tabular-nums shrink-0">
          {verified.toLocaleString()} of {total.toLocaleString()}
          {total > 0 && ` (${Math.round(pct)}%)`}
        </span>
      </div>
      <div className="h-2 rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${pct}%`, backgroundColor: "#0f6064" }}
        />
      </div>
    </div>
  );
}

export const VerificationProgressChart: React.FC<VerificationProgressChartProps> = ({
  projectId,
  siteIds,
  dateFrom,
  dateTo,
}) => {
  // One source of truth: the events stats endpoint provides the
  // `verified_detections / total_detections` counts that every Verify
  // pill reads. Filter args narrow the population to the user's
  // current site / date scope on the dashboard.
  const filterArgs = {
    site_ids: siteIds ? siteIds.split(",") : undefined,
    date_from: dateFrom,
    date_to: dateTo,
  };

  const { data: eventStats, isLoading } = useQuery({
    queryKey: ["statistics", "verification-progress", "events", projectId, siteIds, dateFrom, dateTo],
    queryFn: () => eventsApi.verificationStats(projectId, filterArgs),
  });
  const { data: labelStats } = useQuery({
    queryKey: ["statistics", "verification-progress", "by-label", projectId, siteIds, dateFrom, dateTo],
    queryFn: () =>
      statisticsApi.getVerificationProgressByLabel(projectId, siteIds, dateFrom, dateTo),
  });

  const overallRow: BarRow | null = eventStats
    ? {
        label: "Total observations",
        verified: eventStats.verified_detections,
        total: eventStats.total_detections,
      }
    : null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div>
          <div className="flex items-center gap-1.5">
            <CardTitle className="text-lg">Verification</CardTitle>
            <DashboardAboutPopover
              what={
                <>
                  <p>
                    Percent of observations a person has verified. The
                    same number is shown on every Verify view (Events,
                    Media, Observations), so progress reads the same
                    wherever you are.
                  </p>
                  <p>
                    The list below shows verified vs total observations
                    per label, sorted by support.
                  </p>
                </>
              }
              how={
                <>
                  <p>
                    You generally do not need to verify every detection.
                    The Events and Media modals walk through MaxN frames
                    first (the peak-count frames per species in an
                    event); verifying those covers what statistics need.
                  </p>
                  <p>
                    Per-class rows count observations that pass the
                    project threshold or are verified, and skip false
                    detections.
                  </p>
                </>
              }
            />
          </div>
          <p className="text-sm text-muted-foreground">
            Progress overall and per label
          </p>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <p className="text-muted-foreground">Loading...</p>
          </div>
        ) : overallRow ? (
          <div className="rounded-lg bg-muted/50 p-3 max-h-80 overflow-y-auto">
            <div className="flex flex-col gap-3">
              <SlimProgressRow {...overallRow} />
              {labelStats && labelStats.rows.length > 0 && (
                <>
                  <Separator className="my-1" />
                  {labelStats.rows.map((row) => (
                    <SlimProgressRow
                      key={row.label_taxonomy_id ?? row.scientific_name}
                      label={resolveSpeciesName(row)}
                      verified={row.verified}
                      total={row.total}
                    />
                  ))}
                </>
              )}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-40">
            <p className="text-muted-foreground">No data to verify</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
