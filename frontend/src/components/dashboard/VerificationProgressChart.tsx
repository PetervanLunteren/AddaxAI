/**
 * Doughnut chart showing verified vs unverified files.
 */

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
import { statisticsApi } from "../../api/statistics";

ChartJS.register(ArcElement, Tooltip, Legend);

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
  const { data, isLoading } = useQuery({
    queryKey: ["statistics", "verification-progress", projectId, siteIds, dateFrom, dateTo],
    queryFn: () => statisticsApi.getVerificationProgress(projectId, siteIds, dateFrom, dateTo),
  });

  const verified = data?.verified_files ?? 0;
  const unverified = (data?.total_files ?? 0) - verified;
  const pct =
    data && data.total_files > 0
      ? Math.round((verified / data.total_files) * 100)
      : 0;

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

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Verification progress</CardTitle>
        <p className="text-sm text-muted-foreground">
          {data
            ? `${pct}% verified (${verified.toLocaleString()} of ${data.total_files.toLocaleString()} files)`
            : "File verification status"}
        </p>
      </CardHeader>
      <CardContent>
        <div className="h-72">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">Loading...</p>
            </div>
          ) : data && data.total_files > 0 ? (
            <Doughnut data={chartData} options={chartOptions} />
          ) : (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">No files to verify</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
