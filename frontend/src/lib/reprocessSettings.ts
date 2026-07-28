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

/** Seconds → a tidy minutes label, e.g. 300 → "5 min", matching the
 * IntervalControl's minute-based presets. */
export function formatInterval(seconds: number): string {
  if (seconds <= 0) return "disabled";
  const mins =
    seconds % 60 === 0
      ? String(seconds / 60)
      : String(Number((seconds / 60).toFixed(4)));
  return `${mins} min`;
}

export interface RegroupExample {
  time_range: string | null;
  observations: { label: string; count: number }[];
  /** How many new events this event's files land in (1 = merged, >1 = split). */
  maps_to: number;
}

export interface RegroupImpact {
  confirmed_at_risk: number;
  counts_at_risk: number;
  total_confirmed: number;
  example: RegroupExample | null;
}

/**
 * When the independence interval changed, return how much confirmed count
 * work a regroup would reset, but only if any is actually at risk. Returns
 * null when the interval is unchanged or nothing would be lost, i.e. when
 * no warning is needed. Shared by both apply surfaces so the gate can't drift.
 */
export async function fetchRegroupImpact(
  projectId: string,
  oldInterval: number,
  newInterval: number,
): Promise<RegroupImpact | null> {
  if (newInterval === oldInterval) return null;
  const impact = await projectsApi.regroupPreview(projectId, newInterval);
  if (impact.confirmed_at_risk === 0 && impact.counts_at_risk === 0) {
    return null;
  }
  return impact;
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
