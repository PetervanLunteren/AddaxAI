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
