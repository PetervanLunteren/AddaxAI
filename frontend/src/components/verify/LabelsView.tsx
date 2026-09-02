/**
 * Labels view — the Labels page body, decoupled from page chrome.
 *
 * Owns the Detections / Files switch and mounts one of the two tabs.
 * Detections is `LabelsTab`, the embedding-driven crop grid: one card per
 * detection above the threshold. Files is `FilesTab`: one card per file,
 * boxes and all. Two views of the same work, see `ViewModeToggle`.
 *
 * The switch rides into each tab's toolbar through the `toolbarExtra`
 * slot that already existed for the folder-run step's settings button,
 * so neither tab needs to know the other exists.
 *
 * Mounted by `pages/LabelsPage.tsx` (research projects) and
 * `pages/folder-run/FolderRunLabelsStep.tsx` (folder runs), mirroring
 * how `VerifyView` backs the Counts page. Both get the switch.
 */

import type { ReactNode } from "react";
import { useCallback, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";

import { projectsApi } from "../../api/projects";
import { FilesTab } from "./FilesTab";
import { LabelsTab } from "./LabelsTab";
import { lblFiltersFromSearchParams } from "./labels-filters";
import { useLabelsProgress } from "./useLabelsProgress";
import { ViewModeToggle, type LabelsViewMode } from "./ViewModeToggle";

export interface LabelsViewProps {
  projectId: string;
  /** Forwarded to the active tab so a host page (folder-run Labels
   *  step) can hide its sticky nav while a bulk selection is live. */
  onSelectionChange?: (count: number) => void;
  /** Forwarded to the active tab's toolbar slot (the folder-run Labels
   *  step puts its "Analysis settings" button there), after the mode
   *  switch. */
  toolbarExtra?: ReactNode;
  /** Forwarded to LabelsTab: bump to force a re-sort onto refreshed
   *  labels (e.g. after a reprocess). */
  refreshSignal?: number;
  /** Take the user to where the detection threshold is set. The control
   *  is in a different place per mode, a route in projects mode and a
   *  slideout in a folder run, so the host supplies the trip rather than
   *  the grid guessing. Omitted, the threshold note just names it. */
  onEditThreshold?: () => void;
}

export function LabelsView({
  projectId,
  onSelectionChange,
  toolbarExtra,
  refreshSignal,
  onEditThreshold,
}: LabelsViewProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId),
  });

  const mode: LabelsViewMode =
    searchParams.get("view") === "files" ? "files" : "crops";

  // Drives the chips on the toggle and the pointer each tab shows when
  // its grid runs out, so neither half of the work can hide behind the
  // other. Same query the tabs use, so it costs no extra request.
  const filters = useMemo(
    () => lblFiltersFromSearchParams(searchParams),
    [searchParams],
  );
  const progress = useLabelsProgress(projectId, filters);

  const setMode = useCallback(
    (next: LabelsViewMode) => {
      setSearchParams(
        (prev) => {
          const sp = new URLSearchParams(prev);
          // "crops" is the implicit default, so it leaves no param.
          if (next === "files") sp.set("view", "files");
          else sp.delete("view");
          return sp;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const toolbar = (
    <>
      <ViewModeToggle
        value={mode}
        onChange={setMode}
        cropsLeft={progress.cropsLeft}
        filesLeft={progress.filesLeft}
      />
      {toolbarExtra}
    </>
  );

  if (mode === "files") {
    return (
      <FilesTab
        projectId={projectId}
        toolbarExtra={toolbar}
        onSelectionChange={onSelectionChange}
        otherTabLeft={progress.cropsLeft}
        thisTabLeft={progress.filesLeft}
        totalLabels={progress.total}
        onSwitchTab={() => setMode("crops")}
        onEditThreshold={onEditThreshold}
      />
    );
  }

  return (
    <LabelsTab
      projectId={projectId}
      classificationModelId={project?.classification_model_id ?? null}
      onSelectionChange={onSelectionChange}
      toolbarExtra={toolbar}
      refreshSignal={refreshSignal}
      otherTabLeft={progress.filesLeft}
      thisTabLeft={progress.cropsLeft}
      totalLabels={progress.total}
      onSwitchTab={() => setMode("files")}
    />
  );
}
