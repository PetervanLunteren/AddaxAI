/**
 * Deployments whose files can no longer be found.
 *
 * One definition of "broken", shared by the startup toast and the sidebar
 * dot so the two can never disagree about what they are reporting.
 *
 * Reuses the ["deployments", projectId] query the deployments page and the
 * toast already run, so mounting this costs no extra request. That key is
 * invalidated by every relink and edit path, so consumers clear themselves
 * as soon as the user reconnects a folder.
 */

import { useQuery } from "@tanstack/react-query";
import { deploymentsApi } from "../api/deployments";
import { queryClient } from "../lib/query-client";
import type { DeploymentResponse } from "../api/types";

export const isBrokenDeployment = (d: DeploymentResponse): boolean =>
  d.folder_status === "needs_relink";

export function useBrokenDeployments(projectId: string | undefined) {
  const { data } = useQuery({
    queryKey: ["deployments", projectId],
    queryFn: () => deploymentsApi.list({ projectId: projectId! }),
    enabled: !!projectId,
  });
  return (data ?? []).filter(isBrokenDeployment);
}

/**
 * When each deployment was last re-checked, so a grid of 500 tiles off one
 * missing folder fires one request rather than 500.
 *
 * Rate-limited rather than once-only. A plain "already asked" set muted a
 * deployment for the rest of the page session, so a folder that broke,
 * was reconnected, and then broke again was never noticed a second time:
 * the tiles went grey, nothing re-checked, and every surface kept saying
 * the deployment was fine. Ejecting a drive twice is enough to hit it.
 * A minute is far longer than the burst of failures one screenful
 * produces, and far shorter than a user takes to break a folder twice.
 */
const RECHECK_INTERVAL_MS = 60_000;
const lastCheckedAt = new Map<string, number>();

/**
 * Report that a crop or frame image could not be loaded.
 *
 * `folder_status` is otherwise only refreshed by the startup sweep, so a
 * drive unplugged or a folder renamed mid-session left every surface
 * saying the deployment was fine while its pictures quietly turned grey.
 * The browser already knows the image 404'd, and `/check-folder` already
 * knows how to re-stat one deployment, so this connects the two: the app
 * finds out at the moment the user sees it.
 *
 * Fire-and-forget. A failed check is not worth a toast; the images are
 * already missing and the user has that in front of them.
 *
 * Note this also fires for a legitimately absent crop, most often a
 * verified video detection sitting off the best frame (see DEVELOPERS.md,
 * "The best frame is the only frame a video detection can be shown on").
 * That costs one ten-file check per deployment per minute and reports
 * "valid", which is the right answer.
 *
 * Invalidates the whole `["deployments"]` prefix rather than one project's
 * key, so callers deep in the tile tree need no project id.
 */
export function reportMissingMedia(deploymentId: string | null | undefined): void {
  if (!deploymentId) return;
  const last = lastCheckedAt.get(deploymentId);
  if (last !== undefined && Date.now() - last < RECHECK_INTERVAL_MS) return;
  lastCheckedAt.set(deploymentId, Date.now());

  void deploymentsApi
    .checkFolder(deploymentId)
    .then(() => queryClient.invalidateQueries({ queryKey: ["deployments"] }))
    .catch(() => {
      /* already visible as a missing image */
    });
}
