/**
 * Review page - placeholder for AI-assisted review workflow
 */

import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { filesApi } from "../api/files";

export default function ReviewPage() {
  const { projectId } = useParams<{ projectId: string }>();

  const { data: stats, isLoading } = useQuery({
    queryKey: ["observation-type-stats", projectId],
    queryFn: () => filesApi.getObservationTypeStats(projectId!),
    enabled: !!projectId,
  });

  const statItems = [
    { key: "animal", label: "Animal", color: "bg-green-500" },
    { key: "human", label: "Human", color: "bg-red-500" },
    { key: "vehicle", label: "Vehicle", color: "bg-blue-500" },
    { key: "blank", label: "Blank", color: "bg-gray-400" },
    { key: "unknown", label: "Unknown", color: "bg-yellow-500" },
    { key: "unclassified", label: "Unclassified", color: "bg-gray-300" },
  ];

  const totalFiles = stats
    ? Object.values(stats).reduce((sum, count) => sum + count, 0)
    : 0;

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Review</h1>
        <p className="text-muted-foreground mt-2">
          Review and verify AI detections
        </p>
      </div>

      {/* Observation type summary */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        {statItems.map((item) => (
          <Card key={item.key}>
            <CardContent className="p-4 text-center">
              <div className="flex items-center justify-center gap-2 mb-2">
                <div className={`w-3 h-3 rounded-full ${item.color}`} />
                <span className="text-sm font-medium text-muted-foreground">
                  {item.label}
                </span>
              </div>
              <p className="text-2xl font-bold">
                {isLoading ? "..." : (stats?.[item.key] ?? 0)}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Coming soon */}
      <Card>
        <CardHeader>
          <CardTitle>AI-Assisted Review</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12 space-y-4">
            <p className="text-lg text-muted-foreground">
              Coming soon
            </p>
            <div className="max-w-lg mx-auto text-sm text-muted-foreground space-y-2">
              <p>
                The review workflow will use a two-threshold system to efficiently
                verify AI predictions:
              </p>
              <ul className="text-left list-disc list-inside space-y-1">
                <li>
                  <strong>High-confidence detections</strong> are auto-accepted
                </li>
                <li>
                  <strong>Low-confidence detections</strong> are auto-rejected
                </li>
                <li>
                  <strong>Mid-range detections</strong> are flagged for human review
                </li>
              </ul>
              <p>
                {totalFiles > 0
                  ? `You have ${totalFiles} files ready for review.`
                  : "Run an analysis to get started."}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
