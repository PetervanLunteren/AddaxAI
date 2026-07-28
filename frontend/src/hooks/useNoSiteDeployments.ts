/**
 * Shared hook for GPS-dependent pages.
 *
 * Returns the count of deployments in the project that have no camera
 * site assigned. The consuming page renders `<NoSiteBanner>` when the
 * count is non-zero.
 */

import { useQuery } from "@tanstack/react-query";
import { projectsApi, type DeploymentsWithoutSiteResponse } from "../api/projects";

export function useNoSiteDeployments(projectId: string | undefined) {
  return useQuery<DeploymentsWithoutSiteResponse>({
    queryKey: ["deployments-without-site", projectId],
    queryFn: () => projectsApi.getDeploymentsWithoutSite(projectId!),
    enabled: !!projectId,
    staleTime: 30_000,
  });
}
