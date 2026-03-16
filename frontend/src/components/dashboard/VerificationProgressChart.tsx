/**
 * Three stacked progress bars showing verification progress
 * for files, representatives, and detections.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import { eventsApi } from "../../api/events";

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

function ProgressBar({ label, verified, total }: BarRow) {
  const pct = total > 0 ? (verified / total) * 100 : 0;

  return (
    <div className="p-3 rounded-lg bg-muted/50 space-y-2">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="text-muted-foreground">
          {verified.toLocaleString()} of {total.toLocaleString()}
          {total > 0 && ` (${Math.round(pct)}%)`}
        </span>
      </div>
      <div className="h-2.5 rounded-full bg-muted overflow-hidden">
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
  const [showExplanation, setShowExplanation] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: ["statistics", "verification-progress", projectId, siteIds, dateFrom, dateTo],
    queryFn: () =>
      eventsApi.verificationStats(projectId, {
        site_ids: siteIds ? siteIds.split(",") : undefined,
        date_from: dateFrom,
        date_to: dateTo,
      }),
  });

  const rows: BarRow[] = data
    ? [
        { label: "Event representatives", verified: data.verified_representatives, total: data.total_representatives },
        { label: "Files", verified: data.verified_files, total: data.total_files },
        { label: "Detections", verified: data.verified_detections, total: data.total_detections },
      ]
    : [];

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Verification</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <p className="text-muted-foreground">Loading...</p>
          </div>
        ) : rows.length > 0 ? (
          <div className="flex flex-col gap-3">
            {rows.map((row) => (
              <ProgressBar key={row.label} {...row} />
            ))}

            <button
              type="button"
              onClick={() => setShowExplanation((v) => !v)}
              className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground transition-colors"
            >
              What's the difference?
            </button>

            {showExplanation && (
              <div className="text-xs text-muted-foreground space-y-2 pt-1">
                <p><strong>Files</strong> are individual images or video frames captured by the camera.</p>
                <p><strong>Event representatives</strong> are one file per event, used for quick review.</p>
                <p><strong>Detections</strong> are individual animal, person, or vehicle bounding boxes within files.</p>
                <p>
                  With <strong>similarity verification</strong>, you can verify detections quickly by
                  comparing similar crops. However, it can't catch false negatives (missed animals)
                  because you only see the bounding boxes the model found.
                </p>
                <p>
                  <strong>Event verification</strong> shows the full file with all objects in context,
                  which is the only way to spot missed detections. Use it to verify files and event
                  representatives.
                </p>
              </div>
            )}
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
