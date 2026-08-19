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

import { toast } from "sonner";

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

/** One reason the reprocess could not touch a folder, as the job reports
 *  it: how many folders hit this, and one of them to name. */
interface SkippedCause {
  count: number;
  path: string;
}

/** One paragraph per cause. `count` folders hit it, `path` is one of
 *  them, spelled out in full and on its own line so the user can read it
 *  and go look. Never "starting with": the backend reports whichever
 *  folder the query happened to return first, so any claim of order
 *  would send people to the wrong one. */
const SKIP_TEXT: Record<string, (c: SkippedCause) => string> = {
  // The folder itself is gone. Reconnecting it is the fix, and the
  // Deployments page already has that flow.
  folder_missing: ({ count, path }) =>
    count === 1
      ? `AddaxAI cannot find this folder:\n${path}\nIt may have been ` +
        `moved, renamed or unplugged. Reconnect it on the Deployments ` +
        `page, then apply your settings again.`
      : `AddaxAI cannot find ${count} folders, for example:\n${path}\n` +
        `They may have been moved, renamed or unplugged. Reconnect them ` +
        `on the Deployments page, then apply your settings again.`,
  // The folder is there but its AI results are not. Only a new analysis
  // can bring them back, so say what is missing and where it lived.
  no_results: ({ count, path }) =>
    count === 1
      ? `The raw AI results are missing for this folder:\n${path}\n` +
        `AddaxAI keeps them in a hidden .addaxai subfolder inside it. ` +
        `Copying or cleaning up a folder often leaves hidden files ` +
        `behind. Analyse this folder again to apply the new settings.`
      : `The raw AI results are missing for ${count} folders, for ` +
        `example:\n${path}\nAddaxAI keeps them in a hidden .addaxai ` +
        `subfolder inside each one. Copying or cleaning up a folder often ` +
        `leaves hidden files behind. Analyse these folders again to apply ` +
        `the new settings.`,
  // The results are there but could not be opened or parsed: a locked
  // folder, a half-written file, a disk owned by someone else.
  unreadable: ({ count, path }) =>
    count === 1
      ? `AddaxAI could not read the AI results for this folder:\n${path}\n` +
        `The file may be damaged, or the folder may be locked. Analyse ` +
        `this folder again to rebuild it.`
      : `AddaxAI could not read the AI results for ${count} folders, for ` +
        `example:\n${path}\nThe files may be damaged, or the folders may ` +
        `be locked. Analyse these folders again to rebuild them.`,
  // No folder on the deployment row at all. Cannot happen in a folder
  // run, which is why this one says "deployment".
  no_folder: ({ count }) =>
    count === 1
      ? `1 deployment has no folder set, so there is nothing to ` +
        `reprocess for it. Open it on the Deployments page and pick its ` +
        `folder.`
      : `${count} deployments have no folder set, so there is nothing to ` +
        `reprocess for them. Open them on the Deployments page and pick ` +
        `their folders.`,
};

/**
 * Warn when a finished reprocess job could not reach every folder.
 *
 * Those folders keep the labels from the run that made them, so the
 * settings the user just applied are not what those labels reflect.
 * Without this the job reports success and the difference is invisible.
 *
 * Call from the job's `onComplete` with its `data` payload. Returns true
 * when nothing at all was applied, so the caller can drop the "settings
 * saved" summary: there is no change to show, and claiming a save next
 * to this warning would contradict it.
 */
export function warnIfDeploymentsSkipped(
  data?: Record<string, unknown>,
): boolean {
  const skipped = (data?.skipped ?? {}) as Record<string, SkippedCause>;
  let total = 0;
  const paragraphs: string[] = [];

  for (const [cause, entry] of Object.entries(skipped)) {
    if (!entry?.count) continue;
    total += entry.count;
    const body = SKIP_TEXT[cause];
    // Unknown cause from a newer backend: counted, but nothing to say.
    if (body) paragraphs.push(body(entry));
  }

  // One toast for every cause, not one each. Toasts stack collapsed, so a
  // second one would sit behind the first and be invisible unless the user
  // happens to hover the pile.
  if (paragraphs.length) {
    toast.warning(
      total === 1
        ? "Settings not applied to 1 folder"
        : `Settings not applied to ${total} folders`,
      {
        description: paragraphs.join("\n\n"),
        duration: Infinity,
        // Both surfaces keep their primary button (Save changes, Continue)
        // in the bottom-right corner, and this toast stays until dismissed,
        // so down there it sits on top of the thing the user needs next.
        position: "top-center",
        // Renders the newlines that put each path on its own line.
        style: { whiteSpace: "pre-line" },
      },
    );
  }

  return total > 0 && total >= Number(data?.deployments_processed ?? 0);
}
