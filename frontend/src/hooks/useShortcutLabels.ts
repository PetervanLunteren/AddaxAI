/**
 * The project's saved labels, the number-key slots of the Labels page.
 *
 * Stored on the project (`Project.shortcut_labels`), so the Detections
 * grid, the Files grid and the file viewer see the same slots. One hook
 * owns the read and the write, so no surface can drift on how a slot
 * is parsed or persisted. The project query is the single source: an
 * update writes the new slots into that cache entry first, so every
 * reader sees them at once, then saves them.
 *
 * `SHORTCUT_SLOTS` is the one list of slot numbers; every key handler,
 * picker row and free-slot search reads it, so the count lives in one
 * place. It was five, hard coded in five places; a keypad user asked
 * for nine.
 */

import { useCallback, useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { projectsApi } from "../api/projects";
import type { ProjectResponse } from "../api/types";
import type { LabelOption } from "./useLabelOptions";

export type ShortcutLabels = Record<number, LabelOption>;

export const SHORTCUT_SLOTS = [1, 2, 3, 4, 5, 6, 7, 8, 9] as const;

/** The slot a plain number key stands for, or null for any other key. */
export function shortcutSlotFromKey(e: KeyboardEvent): number | null {
  if (e.ctrlKey || e.metaKey || e.altKey) return null;
  const slot = Number(e.key);
  return (SHORTCUT_SLOTS as readonly number[]).includes(slot) ? slot : null;
}

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
