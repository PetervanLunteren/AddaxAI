/**
 * Shared before/after statistics for the reprocess "how the DB changed"
 * summary. Both surfaces that change reprocess-triggering settings use
 * these: the project Settings page and the folder-run "Refine results"
 * slideout. Keeping the fetch + the SaveResults builder here (one source
 * of truth) means the two summaries can never drift.
 */

import { projectsApi } from "../api/projects";
import type {
  SaveResults,
  StatSnapshot,
} from "../components/projects/SaveResultsModal";

export interface ProjectStats {
  observations: StatSnapshot;
  independent_observations: StatSnapshot;
  events: StatSnapshot;
}

/** Fetch observation and event snapshots for the given project.
 *
 * `threshold` filters the raw detection counts (the Detections card).
 * The observation/event snapshots come from the materialized event
 * observations, which already bake in the interval and honour human
 * counts, so they take no threshold/interval. */
export async function fetchStats(
  projectId: string,
  threshold: number,
): Promise<ProjectStats> {
  const [detectionCount, labelStats, indepObsStats, eventStats] = await Promise.all([
    projectsApi.getDetectionCount(projectId, threshold),
    projectsApi.getLabelStats(projectId, threshold),
    projectsApi.getIndependentObservationStats(projectId),
    projectsApi.getIndependentEventStats(projectId),
  ]);
  return {
    observations: { total: detectionCount.count, labels: labelStats },
    independent_observations: {
      total: indepObsStats.total,
      labels: indepObsStats.labels,
    },
    events: { total: eventStats.total, labels: eventStats.labels },
  };
}

/** Pair a before- and after-snapshot into the SaveResults the modal wants. */
export function buildSaveResults(
  before: ProjectStats,
  after: ProjectStats,
): SaveResults {
  return {
    observations: {
      before: before.observations,
      after: after.observations,
    },
    independent_observations: {
      before: before.independent_observations,
      after: after.independent_observations,
    },
    events: {
      before: before.events,
      after: after.events,
    },
  };
}
