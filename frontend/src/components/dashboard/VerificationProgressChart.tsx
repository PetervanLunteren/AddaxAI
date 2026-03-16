/**
 * Doughnut chart showing verification progress.
 *
 * Supports three verification units: files, event representatives, and detections.
 */

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  type ChartOptions,
} from "chart.js";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { eventsApi } from "../../api/events";

ChartJS.register(ArcElement, Tooltip, Legend);

type VerificationUnit = "files" | "representatives" | "detections";

const UNIT_OPTIONS: { value: VerificationUnit; label: string }[] = [
  { value: "files", label: "Files" },
  { value: "representatives", label: "Representatives" },
  { value: "detections", label: "Detections" },
];

interface VerificationProgressChartProps {
  projectId: string;
  siteIds?: string;
  dateFrom?: string;
  dateTo?: string;
}

export const VerificationProgressChart: React.FC<VerificationProgressChartProps> = ({
  projectId,
  siteIds,
  dateFrom,
  dateTo,
}) => {
  const [unit, setUnit] = useState<VerificationUnit>("files");

  const { data, isLoading } = useQuery({
    queryKey: ["statistics", "verification-progress", projectId, siteIds, dateFrom, dateTo],
    queryFn: () =>
      eventsApi.verificationStats(projectId, {
        site_ids: siteIds ? siteIds.split(",") : undefined,
        date_from: dateFrom,
        date_to: dateTo,
      }),
  });

  const { verified, total } = useMemo(() => {
    if (!data) return { verified: 0, total: 0 };
    switch (unit) {
      case "representatives":
        return { verified: data.verified_representatives, total: data.total_representatives };
      case "detections":
        return { verified: data.verified_detections, total: data.total_detections };
      default:
        return { verified: data.verified_files, total: data.total_files };
    }
  }, [data, unit]);

  const unverified = total - verified;
  const pct = total > 0 ? Math.round((verified / total) * 100) : 0;

  const chartData = {
    labels: ["Verified", "Unverified"],
    datasets: [
      {
        data: [verified, unverified],
        backgroundColor: ["#0f6064", "#71b7ba"],
        borderWidth: 0,
      },
    ],
  };

  const chartOptions: ChartOptions<"doughnut"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: "bottom" },
    },
  };

  const unitLabel = UNIT_OPTIONS.find((o) => o.value === unit)?.label.toLowerCase() ?? unit;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">Verification</CardTitle>
            <p className="text-sm text-muted-foreground">
              {data
                ? `${pct}% verified (${verified.toLocaleString()} of ${total.toLocaleString()} ${unitLabel})`
                : "Verification status"}
            </p>
          </div>
          <Select value={unit} onValueChange={(v) => setUnit(v as VerificationUnit)}>
            <SelectTrigger className="w-40 h-9 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {UNIT_OPTIONS.map((o) => (
                <SelectItem key={o.value} value={o.value}>
                  {o.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-72">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">Loading...</p>
            </div>
          ) : total > 0 ? (
            <Doughnut data={chartData} options={chartOptions} />
          ) : (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">No {unitLabel} to verify</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
