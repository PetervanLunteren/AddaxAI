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

  // Two jobs, two bars: Labels (per-detection label cleanup, detection-
  // level) and Counts (event species + count sign-off, event-level). They
  // are separate pages and complete independently, so one number can't
  // represent both. The per-label breakdown below belongs to Labels.
  const labelsRow: BarRow | null = eventStats
    ? {
        label: "Labels verified",
        verified: eventStats.verified_detections,
        total: eventStats.total_detections,
      }
    : null;
  const countsRow: BarRow | null = eventStats
    ? {
        label: "Counts confirmed",
        verified: eventStats.events_confirmed,
        total: eventStats.events_total,
      }
    : null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div>
          <div className="flex items-center gap-1.5">
            <CardTitle className="text-lg">Verification</CardTitle>
            <DashboardAboutPopover>
              <p>
                Two jobs. "Labels verified" is the percent of detections
                checked on the Labels page. "Counts confirmed" is the
                percent of events signed off on the Counts page. The list
                below is label verification per taxon.
              </p>
            </DashboardAboutPopover>
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
        ) : labelsRow && countsRow ? (
          <div className="rounded-lg bg-muted/50 p-3 max-h-80 overflow-y-auto">
            <div className="flex flex-col gap-3">
              <SlimProgressRow {...labelsRow} />
              <SlimProgressRow {...countsRow} />
              {labelStats && labelStats.rows.length > 0 && (
                <>
                  <Separator className="my-1" />
                  {/* These break down "Labels verified" (detection-level),
                      not "Counts confirmed", so label the group. */}
                  <div className="text-xs font-medium text-muted-foreground">
                    Labels verified per taxon
                  </div>
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
