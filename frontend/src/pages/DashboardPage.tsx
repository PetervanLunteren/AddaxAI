/**
 * Dashboard page with detection statistics
 */

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { X } from "lucide-react";
import { api } from "../lib/api-client";

export default function DashboardPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [showModelAlert, setShowModelAlert] = useState(false);

  // Fetch detection statistics
  const { data: stats, isLoading } = useQuery({
    queryKey: ["detection-stats", projectId],
    queryFn: () =>
      api.get<Record<string, number>>(`/api/projects/${projectId}/detection-stats`),
  });

  // Fetch model updates
  const { data: updates } = useQuery({
    queryKey: ["model-updates"],
    queryFn: () => api.get<{ new_models: string[]; checked_at: string | null }>("/api/ml/updates"),
    staleTime: Infinity, // Only check once per session
  });

  // Show alert if new models found
  useEffect(() => {
    if (updates?.new_models && updates.new_models.length > 0) {
      setShowModelAlert(true);
    }
  }, [updates]);

  // Transform data for pie chart
  const chartData = stats
    ? Object.entries(stats).map(([category, count]) => ({
        name: category.charAt(0).toUpperCase() + category.slice(1),
        value: count,
      }))
    : [];

  // Colors for categories
  const COLORS: Record<string, string> = {
    Animal: "#22c55e", // green
    Person: "#ef4444", // red
    Vehicle: "#3b82f6", // blue
  };

  const totalDetections = chartData.reduce((sum, item) => sum + item.value, 0);

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Overview of detection statistics
        </p>
      </div>

      {/* New models alert */}
      {showModelAlert && updates?.new_models && updates.new_models.length > 0 && (
        <Alert className="mb-6">
          <AlertTitle className="flex items-center justify-between">
            <span>New models available</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowModelAlert(false)}
              className="h-6 w-6 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </AlertTitle>
          <AlertDescription>
            {updates.new_models.length} new model{updates.new_models.length > 1 ? "s" : ""} added to your catalog
          </AlertDescription>
        </Alert>
      )}

      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-muted-foreground">Loading statistics...</div>
        </div>
      ) : totalDetections === 0 ? (
        <Card>
          <CardContent className="py-12">
            <div className="text-center text-muted-foreground">
              <p className="text-lg">No detections yet</p>
              <p className="text-sm mt-2">
                Run detection on a deployment to see statistics here
              </p>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Pie Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Detections by Category</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value, percent }) =>
                      `${name}: ${value} (${(percent * 100).toFixed(0)}%)`
                    }
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={COLORS[entry.name] || "#9ca3af"}
                      />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Summary Stats */}
          <Card>
            <CardHeader>
              <CardTitle>Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
                  <span className="text-lg font-semibold">Total Detections</span>
                  <span className="text-2xl font-bold">{totalDetections}</span>
                </div>

                {chartData.map((item) => (
                  <div
                    key={item.name}
                    className="flex items-center justify-between p-3 border rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: COLORS[item.name] || "#9ca3af" }}
                      />
                      <span className="font-medium">{item.name}</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="text-muted-foreground">
                        {((item.value / totalDetections) * 100).toFixed(1)}%
                      </span>
                      <span className="font-semibold text-lg">{item.value}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
