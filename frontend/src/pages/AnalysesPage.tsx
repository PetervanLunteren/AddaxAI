/**
 * Analyses Page - ML Analysis Queue Management
 *
 * Manages the queue of deployments waiting for ML analysis.
 */

import { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, Play, Trash2, FolderOpen, AlertCircle } from "lucide-react";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { AddDeploymentDialog } from "../components/deployments/AddDeploymentDialog";
import { jobsApi } from "../api/jobs";
import type { JobResponse, DeploymentAnalysisPayload } from "../api/types";

export function AnalysesPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [dialogOpen, setDialogOpen] = useState(false);
  const queryClient = useQueryClient();

  // Fetch deployment_analysis jobs for this project
  const { data: jobs, isLoading: jobsLoading } = useQuery({
    queryKey: ["jobs", { type: "deployment_analysis", project_id: projectId }],
    queryFn: () => jobsApi.list({ type: "deployment_analysis", project_id: projectId }),
    enabled: !!projectId,
  });

  // Delete job mutation
  const deleteJobMutation = useMutation({
    mutationFn: (jobId: string) => jobsApi.delete(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  // Run queue mutation
  const runQueueMutation = useMutation({
    mutationFn: () => jobsApi.runQueue(projectId),
    onSuccess: (data) => {
      // Show success message (you can add toast notifications later)
      alert(`${data.message}\n\n${data.jobs_started} jobs in queue.`);
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  // Helper to parse payload as DeploymentAnalysisPayload
  const getPayload = (job: JobResponse): DeploymentAnalysisPayload | null => {
    try {
      return job.payload as unknown as DeploymentAnalysisPayload;
    } catch {
      return null;
    }
  };

  // Split jobs into pending/running and completed/failed
  const queuedJobs = jobs?.filter(
    (j) => j.status === "pending" || j.status === "running"
  ) || [];
  const completedJobs = jobs?.filter(
    (j) => j.status === "completed" || j.status === "failed" || j.status === "cancelled"
  ) || [];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "pending":
        return "text-gray-600 bg-gray-50";
      case "running":
        return "text-blue-600 bg-blue-50";
      case "completed":
        return "text-green-600 bg-green-50";
      case "failed":
        return "text-red-600 bg-red-50";
      case "cancelled":
        return "text-yellow-600 bg-yellow-50";
      default:
        return "text-gray-600 bg-gray-50";
    }
  };

  if (!projectId) {
    return (
      <div className="p-8">
        <p className="text-destructive">Project ID missing</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">
                Analysis Queue
              </h1>
              <p className="text-sm text-muted-foreground">
                Manage deployments waiting for ML analysis
              </p>
            </div>
            <div className="flex gap-2">
              {queuedJobs.length > 0 && (
                <Button
                  onClick={() => runQueueMutation.mutate()}
                  disabled={runQueueMutation.isPending}
                  variant="default"
                >
                  <Play className="h-4 w-4 mr-2" />
                  {runQueueMutation.isPending ? "Starting..." : "Run Queue"}
                </Button>
              )}
              <Button onClick={() => setDialogOpen(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Add Deployment
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {jobsLoading ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground">Loading queue...</p>
          </div>
        ) : (
          <div className="space-y-8">
            {/* Queued Jobs Section */}
            <div>
              <h2 className="text-lg font-semibold mb-4">
                Queue ({queuedJobs.length})
              </h2>
              {queuedJobs.length > 0 ? (
                <div className="space-y-4">
                  {queuedJobs.map((job) => {
                    const payload = getPayload(job);
                    if (!payload) return null;

                    return (
                      <Card
                        key={job.id}
                        className="transition-shadow hover:shadow-lg"
                      >
                        <CardHeader>
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center gap-3">
                                <FolderOpen className="h-5 w-5 text-muted-foreground" />
                                <CardTitle className="text-base font-mono">
                                  {payload.folder_path.split("/").pop() || payload.folder_path}
                                </CardTitle>
                                <span
                                  className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-medium ${getStatusColor(
                                    job.status
                                  )}`}
                                >
                                  {job.status.charAt(0).toUpperCase() +
                                    job.status.slice(1)}
                                </span>
                              </div>
                              <CardDescription className="mt-2 font-mono text-xs text-muted-foreground">
                                {payload.folder_path}
                              </CardDescription>
                            </div>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => deleteJobMutation.mutate(job.id)}
                              disabled={deleteJobMutation.isPending}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <div className="flex items-center gap-6 text-sm text-muted-foreground">
                            <div>
                              <span className="font-medium">Detection:</span>{" "}
                              {payload.detection_model}
                            </div>
                            <div>
                              <span className="font-medium">
                                Classification:
                              </span>{" "}
                              {payload.classification_model}
                            </div>
                            <div className="ml-auto text-xs">
                              Added {new Date(job.created_at).toLocaleString()}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              ) : (
                <Card>
                  <CardContent className="flex flex-col items-center justify-center py-12">
                    <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
                    <p className="text-muted-foreground mb-4">
                      No deployments in queue. Add your first deployment to
                      start analyzing camera trap images.
                    </p>
                    <Button onClick={() => setDialogOpen(true)}>
                      <Plus className="h-4 w-4 mr-2" />
                      Add Deployment
                    </Button>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Completed Jobs Section */}
            {completedJobs.length > 0 && (
              <div>
                <h2 className="text-lg font-semibold mb-4">
                  Completed ({completedJobs.length})
                </h2>
                <div className="space-y-4">
                  {completedJobs.map((job) => {
                    const payload = getPayload(job);
                    if (!payload) return null;

                    return (
                      <Card key={job.id} className="opacity-75">
                        <CardHeader>
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center gap-3">
                                <FolderOpen className="h-5 w-5 text-muted-foreground" />
                                <CardTitle className="text-base font-mono">
                                  {payload.folder_path.split("/").pop() || payload.folder_path}
                                </CardTitle>
                                <span
                                  className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-medium ${getStatusColor(
                                    job.status
                                  )}`}
                                >
                                  {job.status.charAt(0).toUpperCase() +
                                    job.status.slice(1)}
                                </span>
                              </div>
                              <CardDescription className="mt-2 font-mono text-xs text-muted-foreground">
                                {payload.folder_path}
                              </CardDescription>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <div className="flex items-center gap-6 text-sm text-muted-foreground">
                            <div>
                              <span className="font-medium">Detection:</span>{" "}
                              {payload.detection_model}
                            </div>
                            <div>
                              <span className="font-medium">
                                Classification:
                              </span>{" "}
                              {payload.classification_model}
                            </div>
                            {job.completed_at && (
                              <div className="ml-auto text-xs">
                                Completed{" "}
                                {new Date(job.completed_at).toLocaleString()}
                              </div>
                            )}
                          </div>
                          {job.error && (
                            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-800">
                              <span className="font-medium">Error:</span>{" "}
                              {job.error}
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Add Deployment Dialog */}
      <AddDeploymentDialog open={dialogOpen} onOpenChange={setDialogOpen} />
    </div>
  );
}
