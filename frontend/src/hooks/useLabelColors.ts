/**
 * Make the project's species colour map the active one.
 *
 * The map comes with the filter options, the one request about the
 * labels a project has, which the filter bars share. Mounted once per
 * project-scoped layout (`AppLayout`, `FolderRunLayout`). Components read
 * colours through `getSpeciesColor`; the ones that need to repaint when
 * the map lands subscribe with `useSpeciesColorsVersion`. Refreshed by
 * `invalidateLabelQueries` whenever labels change, because a species that
 * first appears through a relabel shifts the slots of the species after
 * it.
 */

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { eventsApi } from "../api/events";
import { setLabelColors } from "../utils/species-colors";

export function useLabelColors(projectId: string | undefined): void {
  const { data } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId!),
    enabled: !!projectId,
  });
  const colors = data?.label_colors;

  useEffect(() => {
    if (colors) setLabelColors(colors);
  }, [colors]);

  // Leaving the project clears the map so the next one never paints
  // with a stale set of slots while its own query is in flight.
  useEffect(() => {
    return () => setLabelColors({});
  }, [projectId]);
}
