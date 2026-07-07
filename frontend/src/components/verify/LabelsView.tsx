/**
 * Labels view — the Labels page body, decoupled from page chrome.
 *
 * Thin wrapper that resolves the project's classification model and
 * mounts `LabelsTab` (the embedding-driven crop grid: similarity sort,
 * cohort relabel, per-detection label cleanup). Mounted by
 * `pages/LabelsPage.tsx` (research projects) and
 * `pages/folder-run/FolderRunLabelsStep.tsx` (folder runs), mirroring
 * how `VerifyView` backs the Counts page.
 */

import type { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { projectsApi } from "../../api/projects";
import { LabelsTab } from "./LabelsTab";

export interface LabelsViewProps {
  projectId: string;
  /** Forwarded to LabelsTab so a host page (folder-run Labels step) can
   *  hide its sticky nav while a bulk selection is live. */
  onSelectionChange?: (count: number) => void;
  /** Forwarded to LabelsTab's toolbar slot (the folder-run Labels step
   *  puts its "Analysis settings" button there). */
  toolbarExtra?: ReactNode;
}

export function LabelsView({
  projectId,
  onSelectionChange,
  toolbarExtra,
}: LabelsViewProps) {
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId),
  });

  return (
    <LabelsTab
      projectId={projectId}
      classificationModelId={project?.classification_model_id ?? null}
      onSelectionChange={onSelectionChange}
      toolbarExtra={toolbarExtra}
    />
  );
}
