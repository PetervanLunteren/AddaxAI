/**
 * Queue Card Component
 *
 * Displays list of deployment queue entries.
 * Shows count and "Run queue" button at bottom.
 * Simple vertical list layout (not kanban).
 */

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Play, Loader2, ListTodo } from "lucide-react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { deploymentQueueApi } from "@/api/deployment-queue";
import { QueueItem } from "./QueueItem";
import { RunQueueModal } from "./RunQueueModal";

interface QueueCardProps {
  projectId: string;
}

export function QueueCard({ projectId }: QueueCardProps) {
  const queryClient = useQueryClient();
  const [showRunModal, setShowRunModal] = useState(false);
  const [jobIds, setJobIds] = useState<string[]>([]);
  const [processingCount, setProcessingCount] = useState(0);

  // Fetch queue entries
  const { data: entries, isLoading } = useQuery({
    queryKey: ["deployment-queue", projectId],
    queryFn: () => deploymentQueueApi.list(projectId),
    // Only poll when there are deployments being processed
    refetchInterval: (query) => {
      const hasProcessing = query.state.data?.some(
        (entry: any) => entry.status === "processing"
      );
      return hasProcessing ? 5000 : false;
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => deploymentQueueApi.remove(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deployment-queue", projectId] });
    },
  });

  // Process queue mutation
  const processQueueMutation = useMutation({
    mutationFn: () => deploymentQueueApi.process({ project_id: projectId }),
    onSuccess: (data) => {
      // Fire-and-forget invalidation so we don't block the mutation resolve
      void queryClient.invalidateQueries({ queryKey: ["deployment-queue", projectId] });
      return data;
    },
  });

  const handleDelete = async (id: string) => {
    await deleteMutation.mutateAsync(id);
  };

  const handleRunQueue = async () => {
    const pendingCount = entries?.filter((e) => e.status === "pending").length || 0;
    if (pendingCount === 0) {
      alert("No pending deployments to process");
      return;
    }

    try {
      console.log("[QueueCard] Starting queue processing...");
      setProcessingCount(pendingCount);

      console.log("[QueueCard] Calling API...");
      const result = await processQueueMutation.mutateAsync();
      console.log("[QueueCard] API returned:", result);

      setJobIds(result.job_ids);
      setShowRunModal(true);
      console.log("[QueueCard] Modal opened with job IDs:", result.job_ids);
    } catch (error) {
      console.error("[QueueCard] Failed to process queue:", error);
      const message = error instanceof Error ? error.message : String(error);
      alert(`Failed to start processing: ${message}`);
    }
  };

  const pendingCount = entries?.filter((e) => e.status === "pending").length || 0;
  const hasPending = pendingCount > 0;

  if (isLoading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 text-gray-400 animate-spin" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>
            Analysis queue
          </CardTitle>
          <CardDescription>
            {entries && entries.length > 0 ? (
              <span>
                {pendingCount} {pendingCount === 1 ? "deployment" : "deployments"} pending
              </span>
            ) : (
              <span>No deployments in queue yet</span>
            )}
          </CardDescription>
        </CardHeader>

        <CardContent>
          {entries && entries.length > 0 ? (
            <div className="space-y-3 max-h-[500px] overflow-y-auto border border-gray-200 rounded-lg p-3">
              {entries.map((entry) => (
                <QueueItem key={entry.id} entry={entry} onDelete={handleDelete} />
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <ListTodo className="h-12 w-12 mx-auto mb-3 text-gray-300" />
              <p className="text-sm">No deployments in queue</p>
              <p className="text-xs mt-1">Add deployments using the form on the left</p>
            </div>
          )}
        </CardContent>

        <CardFooter>
          <Button
            onClick={handleRunQueue}
            disabled={!hasPending}
            className="w-full"
            size="lg"
          >
            <Play className="h-4 w-4 mr-2" />
            Run queue ({pendingCount})
          </Button>
        </CardFooter>
      </Card>

      {/* Run queue modal */}
      <RunQueueModal
        open={showRunModal}
        onOpenChange={setShowRunModal}
        queueCount={processingCount}
        jobIds={jobIds}
      />
    </>
  );
}
