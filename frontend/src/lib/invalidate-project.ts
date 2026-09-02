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
    ["sites-with-stats", projectId],
    ["events"],
    ["event-count"],
    ["event-count-filtered"],
    ["statistics"],
    ["label-tree"],
    ["label-colors"],
    ["project-label-stats"],
    ["observations-stats", projectId],
    ["observation-rate-map", projectId],
    ["labels-unprocessed", projectId],
    // The Labels page's photo-level progress bar, and the files list
    // itself: a reprocess or a threshold change moves which photos count
    // as empty, so both go stale on exactly these cascade events.
    ["labels-progress", projectId],
    ["labels-files", projectId],
  ];
  for (const queryKey of keys) {
    void queryClient.invalidateQueries({ queryKey });
  }
}
