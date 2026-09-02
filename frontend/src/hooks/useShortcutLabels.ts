/**
 * The project's saved labels, the 1 to 5 slots of the Labels page.
 *
 * Stored on the project (`Project.shortcut_labels`), so the Detections
 * grid and the Files viewer see the same five. One hook owns the read
 * and the write, so neither surface can drift on how a slot is parsed
 * or persisted. The project query is the single source: an update
 * writes the new slots into that cache entry first, so every reader
 * sees them at once, then saves them.
 */

import { useCallback, useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { projectsApi } from "../api/projects";
import type { ProjectResponse } from "../api/types";
import type { LabelOption } from "./useLabelOptions";

export type ShortcutLabels = Record<number, LabelOption>;

export function useShortcutLabels(projectId: string) {
  const queryClient = useQueryClient();
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId),
  });

  const shortcutLabels = useMemo<ShortcutLabels>(() => {
    const parsed: ShortcutLabels = {};
    for (const [k, v] of Object.entries(project?.shortcut_labels ?? {})) {
      parsed[Number(k)] = v as LabelOption;
    }
    return parsed;
  }, [project?.shortcut_labels]);

  /** Update the slots in the project cache and persist them. */
  const updateShortcutLabels = useCallback(
    (updater: (prev: ShortcutLabels) => ShortcutLabels) => {
      const next = updater(shortcutLabels);
      queryClient.setQueryData<ProjectResponse>(["project", projectId], (p) =>
        p ? { ...p, shortcut_labels: next } : p,
      );
      projectsApi.update(projectId, { shortcut_labels: next });
    },
    [projectId, queryClient, shortcutLabels],
  );

  return { shortcutLabels, updateShortcutLabels };
}
