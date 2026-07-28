/**
 * React Query hooks for deployment queue operations.
 *
 * Following DEVELOPERS.MD principles:
 * - Type hints everywhere
 * - Explicit operations
 * - Simple, clear API
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deploymentQueueApi,
  type DeploymentQueueCreate,
  type ProcessQueueRequest,
} from "@/api/deployment-queue";

/**
 * Fetch deployment queue for a project
 */
export function useDeploymentQueue(projectId: string, status?: string) {
  return useQuery({
    queryKey: ["deployment-queue", projectId, status],
    queryFn: () => deploymentQueueApi.list(projectId, status),
    enabled: !!projectId,
  });
}

/**
 * Add entry to deployment queue
 */
export function useAddToQueue() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: DeploymentQueueCreate) =>
      deploymentQueueApi.create(data),
    onSuccess: (_, variables) => {
      // Invalidate queue list for this project
      queryClient.invalidateQueries({
        queryKey: ["deployment-queue", variables.project_id],
      });
    },
  });
}

/**
 * Remove entry from queue
 */
export function useRemoveFromQueue() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => deploymentQueueApi.remove(id),
    onSuccess: () => {
      // Invalidate all deployment queue queries
      queryClient.invalidateQueries({
        queryKey: ["deployment-queue"],
      });
    },
  });
}

/**
 * Process deployment queue
 */
export function useProcessQueue() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: ProcessQueueRequest) => deploymentQueueApi.process(data),
    onSuccess: (_, variables) => {
      // Invalidate queue list for this project
      queryClient.invalidateQueries({
        queryKey: ["deployment-queue", variables.project_id],
      });
    },
  });
}
