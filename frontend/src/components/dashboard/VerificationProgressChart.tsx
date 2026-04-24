/**
 * Three stacked progress bars showing verification progress
 * for MaxN frames, files, and detections.
 */

import { useQuery } from "@tanstack/react-query";
import { Info } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import { Tooltip as UITooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";
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
        { label: "MaxN frames", verified: data.verified_max_n_frames, total: data.total_max_n_frames },
        { label: "Files", verified: data.verified_files, total: data.total_files },
        { label: "Observations", verified: data.verified_detections, total: data.total_observations },
      ]
    : [];

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-1.5">
          <CardTitle className="text-lg">Verification</CardTitle>
          <TooltipProvider delayDuration={200}>
            <UITooltip>
              <TooltipTrigger asChild>
                <Info className="h-4 w-4 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent side="bottom" className="max-w-sm p-3 space-y-2">
                <p><span className="font-semibold">Files</span> are individual images or video frames captured by the camera.</p>
                <p><span className="font-semibold">MaxN frames</span> are the images where the peak count for each species was observed.</p>
                <p><span className="font-semibold">Observations</span> are individual animal, person, or vehicle detections within files.</p>
                <p>The <span className="font-semibold">Observations tab</span> verifies individual detection crops quickly by grouping visually similar ones together. It can&apos;t catch false negatives (missed animals) because you only see what the model found.</p>
                <p>The <span className="font-semibold">Events tab</span> shows the full file with every object in context, which is the only way to spot missed detections. Use it to verify files and MaxN frames.</p>
                <p>The <span className="font-semibold">Files tab</span> sits between the two: one tile per image or video, with the full frame and its detection overlay.</p>
              </TooltipContent>
            </UITooltip>
          </TooltipProvider>
        </div>
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
