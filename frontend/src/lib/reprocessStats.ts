/**
 * Shared before/after statistics for the reprocess "how the DB changed"
 * summary. Both surfaces that change reprocess-triggering settings use
 * these: the project Settings page and the folder-run "Refine results"
 * slideout. Keeping the fetch + the SaveResults builder here (one source
 * of truth) means the two summaries can never drift.
 */

import { eventsApi } from "../api/events";
import { projectsApi } from "../api/projects";
import type {
  Protection,
  SaveResults,
  StatSnapshot,
} from "../components/projects/SaveResultsModal";

export interface ProjectStats {
  /** Per-label detection counts (the "Labels" card). */
  observations: StatSnapshot;
  /** Per-species effective_count, human count where set (the "Counts" card). */
  independent_observations: StatSnapshot;
}

/** Fetch the label and count snapshots for the given project.
 *
 * `threshold` filters the raw detection counts (the Labels card). The
 * count snapshot comes from the materialized event observations, which
 * already bake in the interval and honour human counts, so it takes no
 * threshold/interval. */
export async function fetchStats(
  projectId: string,
  threshold: number,
): Promise<ProjectStats> {
  const [detectionCount, labelStats, indepObsStats] = await Promise.all([
    projectsApi.getDetectionCount(projectId, threshold),
    projectsApi.getLabelStats(projectId, threshold),
    projectsApi.getIndependentObservationStats(projectId),
  ]);
  return {
    observations: { total: detectionCount.count, labels: labelStats },
    independent_observations: {
      total: indepObsStats.total,
      labels: indepObsStats.labels,
    },
  };
}

/** Verified-labels / confirmed-counts share, for the modal's footer lines. */
export async function fetchProtection(projectId: string): Promise<Protection> {
  const s = await eventsApi.verificationStats(projectId);
  return {
    verifiedLabels: s.verified_detections,
    totalLabels: s.total_detections,
    confirmedCounts: s.events_confirmed,
    totalCounts: s.events_total,
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
  };
}
