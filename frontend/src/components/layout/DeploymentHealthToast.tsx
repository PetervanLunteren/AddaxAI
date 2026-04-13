/**
 * Startup health toast for deployments with broken folder links.
 *
 * Mounted inside AppLayout so it activates for any project route.
 * On first mount per project per session, queries the deployments
 * list (reusing the cache that the deployments page populates) and
 * checks for any deployment with folder_status === "needs_relink".
 *
 * If at least one is found, fires a sonner warning toast with a
 * "View" action that navigates to the deployments page with the
 * status filter pre-applied.
 *
 * Shown once per project per session via sessionStorage so it
 * doesn't nag users on every navigation.
 */

import { useEffect, useRef } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { deploymentsApi } from "../../api/deployments";

const SESSION_STORAGE_PREFIX = "addaxai:deployment-health-toast-shown:";

export function DeploymentHealthToast() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const firedRef = useRef(false);

  const { data: deployments } = useQuery({
    queryKey: ["deployments", projectId],
    queryFn: () => deploymentsApi.list({ projectId: projectId! }),
    enabled: !!projectId,
  });

  useEffect(() => {
    if (!projectId || !deployments || firedRef.current) return;

    const sessionKey = `${SESSION_STORAGE_PREFIX}${projectId}`;
    if (sessionStorage.getItem(sessionKey)) return;

    const broken = deployments.filter((d) => d.folder_status === "needs_relink");
    if (broken.length === 0) return;

    firedRef.current = true;
    sessionStorage.setItem(sessionKey, "1");

    toast.warning(
      broken.length === 1
        ? "1 deployment folder couldn't be found"
        : `${broken.length} deployment folders couldn't be found`,
      {
        description:
          "They may have been moved, renamed, or unmounted. We'll help reconnect them.",
        action: {
          label: "View",
          onClick: () => navigate(`/projects/${projectId}/deployments`),
        },
        duration: 10000,
      }
    );
  }, [projectId, deployments, navigate]);

  return null;
}
