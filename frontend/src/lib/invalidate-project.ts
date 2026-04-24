/**
 * Fire-and-forget blanket invalidation of every project-scoped query.
 *
 * Use this only for cascade events that touch multiple entity types:
 * analysis completion, re-embedding, postprocessing reprocess,
 * deployment delete/split, site delete. For narrow edits (rename,
 * notes, tags) keep the targeted `invalidateQueries` calls.
 */

import type { QueryClient } from "@tanstack/react-query";

export function invalidateProjectData(
  queryClient: QueryClient,
  projectId: string,
): void {
  const keys: unknown[][] = [
    ["files", projectId],
    ["file"],
    ["detection-stats", projectId],
    ["label-stats", projectId],
    ["observation-type-stats", projectId],
    ["projects", projectId],
    ["deployments", projectId],
    ["deployment-stats", projectId],
    ["deployment-queue", projectId],
    ["sites", projectId],
    ["events"],
    ["event-count"],
    ["event-count-filtered"],
    ["statistics"],
    ["label-tree"],
    ["project-label-stats"],
    ["observations-stats", projectId],
    ["files-for-verify", projectId],
    ["files-count-for-verify", projectId],
    ["files-verification-stats", projectId],
  ];
  for (const queryKey of keys) {
    void queryClient.invalidateQueries({ queryKey });
  }
}
