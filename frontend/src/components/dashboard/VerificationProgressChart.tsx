/**
 * Three stacked progress bars showing verification progress for the
 * three verify-tab units: events, captures, and observations.
 */

import { useQuery } from "@tanstack/react-query";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import { Separator } from "../ui/separator";
import { DashboardAboutPopover } from "./DashboardAboutPopover";
import { eventsApi } from "../../api/events";
import { filesApi } from "../../api/files";
import { statisticsApi } from "../../api/statistics";

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
  // Events + Observations rows come from the events stats endpoint, which
  // scopes to events with at least one above-threshold detection (the same
  // scope the Events tab uses). The Captures row reads the per-capture
  // verification stats so its total matches the Captures tab even when an
  // event is otherwise blank — captures are still verifiable items.
  const filterArgs = {
    site_ids: siteIds ? siteIds.split(",") : undefined,
    date_from: dateFrom,
    date_to: dateTo,
  };

  const { data: eventStats, isLoading: eventsLoading } = useQuery({
    queryKey: ["statistics", "verification-progress", "events", projectId, siteIds, dateFrom, dateTo],
    queryFn: () => eventsApi.verificationStats(projectId, filterArgs),
  });
  const { data: captureStats, isLoading: capturesLoading } = useQuery({
    queryKey: ["statistics", "verification-progress", "captures", projectId, siteIds, dateFrom, dateTo],
    queryFn: () => filesApi.verificationStats(projectId, filterArgs),
  });
  const { data: labelStats } = useQuery({
    queryKey: ["statistics", "verification-progress", "by-label", projectId, siteIds, dateFrom, dateTo],
    queryFn: () =>
      statisticsApi.getVerificationProgressByLabel(projectId, siteIds, dateFrom, dateTo),
  });

  const isLoading = eventsLoading || capturesLoading;
  const rows: BarRow[] =
    eventStats && captureStats
      ? [
          { label: "Events", verified: eventStats.events_fully_verified, total: eventStats.events_total },
          { label: "Captures", verified: captureStats.verified_files, total: captureStats.total_files },
          { label: "Observations", verified: eventStats.verified_detections, total: eventStats.total_detections },
        ]
      : [];

  return (
    <Card>
      <CardHeader className="pb-2">
        <div>
          <div className="flex items-center gap-1.5">
            <CardTitle className="text-lg">Verification</CardTitle>
            <DashboardAboutPopover
              what={
                <>
                  <p>Verification progress per unit:</p>
                  <ul className="list-disc pl-5 space-y-0.5">
                    <li>Events: files grouped by time.</li>
                    <li>Captures: stills and video frames.</li>
                    <li>Observations: AI detections.</li>
                  </ul>
                  <p>
                    The list below shows verified vs total detections per
                    label, sorted by support.
                  </p>
                </>
              }
              how={
                <>
                  <p>
                    An event is verified when all its MaxN frames are. A
                    capture is verified when you mark it. A detection is
                    verified when you confirm or correct its label. The
                    three rows are independent.
                  </p>
                  <p>
                    Per-class rows count detections that pass the project
                    threshold or are verified, and skip false detections.
                  </p>
                </>
              }
            />
          </div>
          <p className="text-sm text-muted-foreground">
            Progress by unit and label
          </p>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <p className="text-muted-foreground">Loading...</p>
          </div>
        ) : rows.length > 0 ? (
          <div className="rounded-lg bg-muted/50 p-3 max-h-80 overflow-y-auto">
            <div className="flex flex-col gap-3">
              {rows.map((row) => (
                <SlimProgressRow key={row.label} {...row} />
              ))}
              {labelStats && labelStats.rows.length > 0 && (
                <>
                  <Separator className="my-1" />
                  {labelStats.rows.map((row) => (
                    <SlimProgressRow
                      key={row.label_taxonomy_id ?? row.display_name}
                      label={row.display_name}
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
