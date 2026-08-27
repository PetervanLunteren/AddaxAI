/**
 * Fetch the project's species colour map and make it the active one.
 *
 * Mounted once per project-scoped layout (`AppLayout`, `FolderRunLayout`).
 * Components read colours through `getSpeciesColor`; the ones that
 * need to repaint when the map lands subscribe with
 * `useSpeciesColorsVersion`. Invalidated by `invalidateLabelQueries`
 * whenever labels change, because a species that first appears through
 * a relabel shifts the slots of the species after it.
 */

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { projectsApi } from "../api/projects";
import { setLabelColors } from "../utils/species-colors";

export function useLabelColors(projectId: string | undefined): void {
  const { data } = useQuery({
    queryKey: ["label-colors", projectId],
    queryFn: () => projectsApi.getLabelColors(projectId!),
    enabled: !!projectId,
  });

  useEffect(() => {
    if (data) setLabelColors(data);
  }, [data]);

  // Leaving the project clears the map so the next one never paints
  // with a stale set of slots while its own query is in flight.
  useEffect(() => {
    return () => setLabelColors({});
  }, [projectId]);
}
