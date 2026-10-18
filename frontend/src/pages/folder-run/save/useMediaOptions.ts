/**
 * The media copies form: one hook for the folder-run Save step and the
 * project image copies dialog.
 *
 * Owns the option state (seeded from the last-used choices), the lookups
 * the rows need (label tree, deployment count, date bounds), the request
 * fields both the preview and the save send, and the preview query
 * itself. The two surfaces compose this with their own output folder and
 * submit; neither holds media state of its own, so they cannot drift.
 */

import { useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { deploymentsApi } from "../../../api/deployments";
import { eventsApi } from "../../../api/events";
import { exportApi } from "../../../api/export";
import type { OutputPreview } from "../../../api/folder-runs";
import type { LabelTreeResponse } from "../../../api/types";
import {
  FILTER_DEBOUNCE_MS,
  useDebouncedValue,
} from "../../../hooks/useDebouncedValue";
import { DEFAULT_COUNTING_THRESHOLD } from "../../../lib/confidence";
import {
  loadLastUsedSaveOutputs,
  saveLastUsedSaveOutputs,
} from "../../../lib/folderRunSettings";
import { getSpeciesNameMode } from "../../../lib/species-name-mode";
import {
  mediaRequestFields,
  nothingInScope as computeNothingInScope,
  type AnonymiseState,
  type MediaOptionsRequest,
  type SeparateState,
  type VisualiseState,
} from "./media-options";

export interface UseMediaOptionsParams {
  /** The project the copies come from. A folder run passes its run id,
   * which is its project id. */
  projectId: string;
  /** The dialog has no master toggle for the media card: the copies are
   * the whole point of it. The folder-run card can be off. */
  alwaysEnabled?: boolean;
}

/** What the media rows and the preview panel read. */
export interface MediaFormHandles {
  /** For the rows that fetch for themselves (the deployment picker). */
  projectId: string;
  separate: SeparateState;
  setSeparate: (s: SeparateState) => void;
  visualise: VisualiseState;
  setVisualise: (s: VisualiseState) => void;
  anonymise: AnonymiseState;
  setAnonymise: (s: AnonymiseState) => void;
  /** The project's label tree for the label picker, or null when it
   * has no taxonomy (the row is hidden). */
  labelTree: LabelTreeResponse | null;
  /** Only a project with more than one deployment has something to
   * pick; a folder run has one and never shows the row. */
  showDeploymentPicker: boolean;
  /** First and last capture date in the project, for the date picker. */
  dateBounds: { min: string; max: string } | null;
  /** Nothing ticked in the deployment picker, or no label picked where
   * that means "nothing yet". The preview is not asked and the Start /
   * Save button stays off. */
  nothingInScope: boolean;
  /** The media fields the save sends, with the live slider value. */
  requestFields: MediaOptionsRequest;
  /** The preview for the current options (debounced). The last one
   * stays on screen while the next loads. */
  preview: OutputPreview | undefined;
  /** True until the preview matches the form as set: the debounce has
   * not fired yet or the request is in flight. The panel shows its
   * loading state only before the first result. */
  previewLoading: boolean;
  /** The options as set copy at least one file. Gates Start / Save on
   * both surfaces: a run with nothing to copy would only wipe the
   * previous addaxai-media. */
  hasFilesToCopy: boolean;
  /** Persist the layout and look choices for the next save, here or on
   * the other surface. Call it when the user commits. */
  remember: () => void;
}

export function useMediaOptions({
  projectId,
  alwaysEnabled = false,
}: UseMediaOptionsParams): MediaFormHandles {
  // Seed from the last-used choices (persisted on save), falling back to
  // the defaults. Loaded once on mount. The filters and the picks are
  // not persisted: they describe one export.
  const [persisted] = useState(loadLastUsedSaveOutputs);

  const [separate, setSeparate] = useState<SeparateState>(() => ({
    enabled: alwaysEnabled || (persisted?.mediaEnabled ?? false),
    groupBy: persisted?.groupBy ?? "flat",
    groupEvents: persisted?.groupEvents ?? true,
    speciesLast: persisted?.speciesLast ?? false,
    copyEmpties: persisted?.copyEmpties ?? false,
    mediaThreshold: persisted?.mediaThreshold ?? DEFAULT_COUNTING_THRESHOLD,
    includedLabelIds: [],
    likedOnly: false,
    verifiedOnly: false,
  }));
  const [visualise, setVisualise] = useState<VisualiseState>(() => ({
    enabled: persisted?.drawBoxes ?? false,
  }));
  const [anonymise, setAnonymise] = useState<AnonymiseState>(() => ({
    enabled: persisted?.blur ?? false,
  }));

  // Label tree counted by file so the picker reads "N files" per
  // species. Project-wide: a species absent from the picked deployments
  // simply yields no files.
  const { data: labelTree } = useQuery({
    queryKey: ["label-tree", projectId, "file"],
    queryFn: () => eventsApi.getLabelTree(projectId, "file"),
    enabled: !!projectId,
  });
  // Same key as the picker itself, so the count costs no extra request.
  const { data: deployments } = useQuery({
    queryKey: ["deployments", projectId],
    queryFn: () => deploymentsApi.list({ projectId }),
    enabled: !!projectId,
  });
  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId),
    enabled: !!projectId,
  });

  const allLabelIds = labelTree?.all_leaf_ids ?? [];
  const nothingInScope = computeNothingInScope(separate);
  // The name toggle is read per render so the previewed leaf folders
  // match what the save will write.
  const requestFields = mediaRequestFields(
    separate,
    visualise,
    anonymise,
    allLabelIds,
    getSpeciesNameMode(),
  );

  // One preview call per burst of changes: a slider drag, a quick run
  // of toggles. The controls stay live on the state; the query waits
  // until the form has been quiet for FILTER_DEBOUNCE_MS (the preview
  // does 3 SELECTs + a per-file loop, 1.6 s on 40k files).
  const previewFields = useDebouncedValue(requestFields, FILTER_DEBOUNCE_MS);
  // Every field is in the key, so any change refetches and the preview
  // stays exact.
  const { data, isFetching, isPlaceholderData, isError } = useQuery({
    queryKey: ["media-preview", projectId, previewFields],
    queryFn: () => exportApi.previewImageCopies(projectId, previewFields),
    enabled: !!projectId && !nothingInScope,
    staleTime: 30_000,
    placeholderData: keepPreviousData,
  });
  // A failed request must not leave the previous tree standing as if it
  // were the answer: the panel then shows its error state and nothing
  // can be started. While a new preview loads, the previous one stays
  // on screen as placeholder data, which is not loaded either.
  const preview = isError ? undefined : data;
  // The debounced copy is a parse of the serialised fields, so compare
  // by content, as the debounce itself does.
  const previewLoading =
    isFetching ||
    isPlaceholderData ||
    JSON.stringify(previewFields) !== JSON.stringify(requestFields);
  const hasFilesToCopy =
    !nothingInScope &&
    !previewLoading &&
    (preview?.in_scope_files ?? 0) > 0;

  const remember = () => {
    saveLastUsedSaveOutputs({
      // The dialog's card is always on; writing that would switch the
      // folder-run card on for the next run.
      ...(alwaysEnabled ? {} : { mediaEnabled: separate.enabled }),
      groupBy: separate.groupBy,
      groupEvents: separate.groupEvents,
      speciesLast: separate.speciesLast,
      copyEmpties: separate.copyEmpties,
      mediaThreshold: separate.mediaThreshold,
      drawBoxes: visualise.enabled,
      blur: anonymise.enabled,
    });
  };

  return {
    projectId,
    separate,
    setSeparate,
    visualise,
    setVisualise,
    anonymise,
    setAnonymise,
    labelTree: labelTree ?? null,
    showDeploymentPicker: (deployments?.length ?? 0) > 1,
    dateBounds: filterOptions?.date_range ?? null,
    nothingInScope,
    requestFields,
    preview,
    previewLoading,
    hasFilesToCopy,
    remember,
  };
}
