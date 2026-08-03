/**
 * Health toast for deployments with broken folder links.
 *
 * Mounted inside AppLayout so it activates for any project route. Fires a
 * warning toast when the project holds any deployment whose files can no
 * longer be found, with a "View" action that goes to the deployments page,
 * where the relink banners live.
 *
 * The toast does not auto-close. A missing folder is a state that persists
 * until the user fixes it, and it silently breaks crop thumbnails and video
 * playback everywhere in the app, so a notice that vanishes after ten
 * seconds is worse than useless: it teaches the user that nothing is wrong.
 * Only the close button takes it away.
 *
 * Shown once per project per session, so dismissing it silences it for the
 * rest of the session but a still-missing folder announces itself again on
 * the next app start (sessionStorage dies with the window).
 *
 * The sidebar dot (see `useBrokenDeployments`) reports the same state
 * quietly and permanently. This toast is the one that grabs attention.
 */

import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { useBrokenDeployments } from "../../hooks/useBrokenDeployments";

const SESSION_STORAGE_PREFIX = "addaxai:deployment-health-toast-shown:";

const toastIdFor = (projectId: string) => `deployment-health-${projectId}`;

export function DeploymentHealthToast() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const count = useBrokenDeployments(projectId).length;

  // The user reconnected the folder while the toast was still up. That is a
  // normal route now, because the sidebar dot leads to the deployments page
  // without touching the toast. A toast that never expires on its own must
  // not be left asserting something that is no longer true. Dismissing an
  // id that is not showing is a no-op, so this needs no bookkeeping.
  useEffect(() => {
    if (projectId && count === 0) toast.dismiss(toastIdFor(projectId));
  }, [projectId, count]);

  useEffect(() => {
    if (!projectId || count === 0) return;

    const sessionKey = `${SESSION_STORAGE_PREFIX}${projectId}`;
    if (sessionStorage.getItem(sessionKey)) return;
    sessionStorage.setItem(sessionKey, "1");

    toast.warning(
      count === 1
        ? "1 deployment folder couldn't be found"
        : `${count} deployment folders couldn't be found`,
      {
        // A stable id keeps a re-run of this effect from stacking a second
        // copy of a toast that now never expires on its own.
        id: toastIdFor(projectId),
        description:
          "They may have been moved, renamed, or unmounted. We'll help reconnect them.",
        duration: Infinity,
        action: {
          label: "View",
          onClick: () => navigate(`/projects/${projectId}/deployments`),
        },
      },
    );
  }, [projectId, count, navigate]);

  return null;
}
