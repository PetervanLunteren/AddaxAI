/**
 * Shared machinery for the retroactive analysis settings (smoothing,
 * rollup, independence interval, species exclusion). Changing any of
 * them does not re-run the models: the backend re-reads the raw
 * results.json and re-applies the transforms (a "reprocess" job).
 *
 * One source of truth for both surfaces that apply these settings:
 * the project Settings page and the folder-run Labels step's analysis
 * panel. Keep the trigger list, the diff check, and the job kick-off
 * here so the two can never drift.
 */

import { projectsApi } from "../api/projects";

/** Settings whose change requires a reprocess job to take effect. */
export const REPROCESS_TRIGGER_FIELDS = [
  "event_smoothing",
  "smoothing_strength",
  "taxonomic_rollup",
  "independence_interval",
  "excluded_classes",
] as const;

/** True when any reprocess-triggering setting differs between two form
 * snapshots. Fields absent from both snapshots are skipped: not every
 * form carries every trigger field (the folder-run panel has no
 * excluded_classes). */
export function hasReprocessChanges(
  before: Record<string, unknown>,
  after: Record<string, unknown>,
): boolean {
  for (const key of REPROCESS_TRIGGER_FIELDS) {
    if (!(key in before) && !(key in after)) continue;
    const a = before[key];
    const b = after[key];
    if (Array.isArray(a) && Array.isArray(b)) {
      if (a.length !== b.length || a.some((v, i) => v !== b[i])) return true;
    } else if (a !== b) {
      return true;
    }
  }
  return false;
}

/**
 * Kick off a reprocess job when the project has classifications to
 * reprocess. Returns the job id to track, or null when there is
 * nothing to do (no classifier output yet — the PATCHed settings
 * simply apply to the next analysis).
 */
export async function startReprocessIfNeeded(
  projectId: string,
): Promise<string | null> {
  const status = await projectsApi.getPostprocessingStatus(projectId);
  if (!status.has_classifications) return null;
  const result = await projectsApi.reprocess(projectId);
  return result.job_id;
}
