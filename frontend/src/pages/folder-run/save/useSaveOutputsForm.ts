/**
 * Form state + submit logic for the Save outputs step.
 *
 * Owns the option state (groups + output dir + filter), spawns the
 * save-outputs background job, and holds the final result payload
 * the JobProgressModal hands back on completion. The page component
 * is responsible for the WebSocket subscription (``useTaskProgress``)
 * and the modal rendering — this hook just exposes the handles.
 *
 * The output dir defaults to the source folder itself. That is safe
 * because the backend writes media copies into an ``addaxai-media``
 * subfolder (with a scan-skip marker) and the loose data files carry
 * the ``addaxai-`` prefix, so originals are never overwritten and the
 * recognition JSON lands where its source-relative paths resolve
 * (what Timelapse needs).
 */

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  folderRunsApi,
  type SaveOutputsRequest,
  type SaveOutputsResult,
  type SeparateGroupBy,
} from "../../../api/folder-runs";
import { eventsApi } from "../../../api/events";
import type { LabelTreeResponse } from "../../../api/types";
import { isElectron } from "../../../lib/platform";
import { getSpeciesNameMode } from "../../../lib/species-name-mode";
import {
  loadLastUsedSaveOutputs,
  saveLastUsedSaveOutputs,
} from "../../../lib/folderRunSettings";

export interface SeparateState {
  enabled: boolean;
  groupBy: SeparateGroupBy;
  /** Keep a burst together: every file in an event shares one folder
   * (the event's main species). */
  groupEvents: boolean;
  /** Put the species folder inside the user's original folders instead of
   * on top (camtrapR-style ``station/species/`` layout). */
  speciesLast: boolean;
  /** Copy empty captures (no detections) too. Off = skip them. */
  copyEmpties: boolean;
  /** Media-output confidence: detections below it (unless verified)
   * are left out of the copies, drawn boxes, and blurs. The data
   * exports always contain everything regardless. */
  mediaConfidence: number;
  /** Leaf ids the user chose to include (empty = all species). The
   * request sends the complement as ``excluded_label_ids``; the data
   * exports ignore it. */
  includedLabelIds: string[];
}

export interface VisualiseState {
  enabled: boolean;
}

export interface AnonymiseState {
  enabled: boolean;
}

export interface ExportState {
  enabled: boolean;
  csv: boolean;
  xlsx: boolean;
  recognitionJson: boolean;
}

function buildRequest(
  outputDir: string,
  separate: SeparateState,
  visualise: VisualiseState,
  anonymise: AnonymiseState,
  exportOpts: ExportState,
  allLabelIds: string[],
): SaveOutputsRequest {
  return {
    output_dir: outputDir,
    media_confidence: separate.mediaConfidence,
    separate_folders: separate.enabled,
    separate_group_by: separate.groupBy,
    group_events: separate.groupEvents,
    separate_species_last: separate.speciesLast,
    // Visualise / anonymise / copy-empties / species-filter are facets
    // of the media copy: only emit them when the media output is on.
    draw_bboxes: separate.enabled && visualise.enabled,
    anonymise: separate.enabled && anonymise.enabled,
    include_empty: separate.enabled && separate.copyEmpties,
    excluded_label_ids: separate.enabled
      ? excludedLabelIds(separate, allLabelIds)
      : [],
    csv: exportOpts.enabled && exportOpts.csv,
    xlsx: exportOpts.enabled && exportOpts.xlsx,
    recognition_json: exportOpts.enabled && exportOpts.recognitionJson,
    // Burn the user's current name preference into the visualised images
    // (EXIF still carries both names regardless).
    name_mode: getSpeciesNameMode(),
  };
}

/** The label exclusion set sent to the backend: every leaf NOT included
 * by the user. Empty when all species are included (no filter) or the
 * tree hasn't loaded yet. The data exports ignore this entirely. */
export function excludedLabelIds(
  separate: SeparateState,
  allLabelIds: string[],
): string[] {
  const { includedLabelIds } = separate;
  if (
    includedLabelIds.length === 0 ||
    includedLabelIds.length >= allLabelIds.length
  ) {
    return [];
  }
  const included = new Set(includedLabelIds);
  return allLabelIds.filter((id) => !included.has(id));
}

export interface UseSaveOutputsFormParams {
  runId: string;
  sourceFolder: string | undefined;
}

export interface UseSaveOutputsFormResult {
  outputDir: string;
  setOutputDir: (v: string) => void;
  /** The dir the form will submit. Defaults to the source folder
   * itself; equal to ``outputDir``. */
  effectiveOutputDir: string;

  separate: SeparateState;
  setSeparate: (s: SeparateState) => void;
  /** The run's label tree for the species filter modal, or null when the
   * run has no taxonomy (filter row hidden). */
  labelTree: LabelTreeResponse | null;
  visualise: VisualiseState;
  setVisualise: (s: VisualiseState) => void;
  anonymise: AnonymiseState;
  setAnonymise: (s: AnonymiseState) => void;
  exportOpts: ExportState;
  setExportOpts: (s: ExportState) => void;

  promoteOpen: boolean;
  setPromoteOpen: (v: boolean) => void;

  /** Job currently running (null when idle or after completion). */
  jobId: string | null;
  /** Final result payload from the job's complete event. */
  result: SaveOutputsResult | null;
  /** Set during the brief HTTP roundtrip that spawns the job. */
  isSpawning: boolean;
  saveError: Error | null;

  /** Spawn the save-outputs job. */
  saveAll: () => void;
  /** Called by the modal when the job's complete event arrives. */
  onJobComplete: (data: SaveOutputsResult) => void;
  /** Called by the modal on job error. */
  onJobError: (message: string) => void;
  /** Called by the modal on user-cancellation. */
  onJobCancelled: (message: string) => void;
  /** Reset the completion screen back to the form. */
  clearResult: () => void;

  handleBrowse: () => Promise<void>;
  handleOpenResults: () => Promise<void>;

  exportPicked: boolean;
  canSave: boolean;
}

export function useSaveOutputsForm({
  runId,
  sourceFolder,
}: UseSaveOutputsFormParams): UseSaveOutputsFormResult {
  const [outputDir, setOutputDir] = useState("");
  const effectiveOutputDir = outputDir;

  // Seed a sensible default once the source folder is known: the
  // source folder itself. The recognition JSON only resolves its
  // source-relative paths there (Timelapse requirement), the data
  // files carry the addaxai- prefix, and media copies go into an
  // addaxai-media subfolder with a scan-skip marker — so nothing
  // collides with the originals. Fires once, and only while the field
  // is still empty, so it never clobbers a user-picked path.
  const hasSeededOutputRef = useRef(false);
  useEffect(() => {
    if (hasSeededOutputRef.current || !sourceFolder) return;
    hasSeededOutputRef.current = true;
    // One-time seed of a default from an async-loaded prop, guarded by
    // the ref so it can't loop.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setOutputDir((cur) => (cur ? cur : sourceFolder));
  }, [sourceFolder]);

  // Seed the option state from the last-used choices (persisted on
  // Save), falling back to the defaults. Loaded once on mount.
  const [persisted] = useState(loadLastUsedSaveOutputs);

  const [separate, setSeparate] = useState<SeparateState>(() => ({
    enabled: persisted?.mediaEnabled ?? false,
    groupBy: persisted?.groupBy ?? "flat",
    groupEvents: persisted?.groupEvents ?? true,
    speciesLast: persisted?.speciesLast ?? false,
    copyEmpties: persisted?.copyEmpties ?? false,
    mediaConfidence: persisted?.mediaConfidence ?? 0.2,
    includedLabelIds: [],
  }));
  const [visualise, setVisualise] = useState<VisualiseState>(() => ({
    enabled: persisted?.drawBoxes ?? false,
  }));
  const [anonymise, setAnonymise] = useState<AnonymiseState>(() => ({
    enabled: persisted?.blur ?? false,
  }));
  const [exportOpts, setExportOpts] = useState<ExportState>(() => ({
    enabled: persisted?.exportEnabled ?? true,
    csv: persisted?.csv ?? true,
    xlsx: persisted?.xlsx ?? false,
    recognitionJson: persisted?.recognitionJson ?? true,
  }));

  // Label tree for the species filter. Counts by file so the modal
  // shows "N files" per species. null when the run has no taxonomy.
  const { data: labelTree } = useQuery({
    queryKey: ["label-tree", runId, "file"],
    queryFn: () => eventsApi.getLabelTree(runId, "file"),
    enabled: !!runId,
  });
  const allLabelIds = labelTree?.all_leaf_ids ?? [];

  const [promoteOpen, setPromoteOpen] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<SaveOutputsResult | null>(null);
  const [saveError, setSaveError] = useState<Error | null>(null);

  const spawn = useMutation({
    mutationFn: (payload: SaveOutputsRequest) =>
      folderRunsApi.saveOutputs(runId, payload),
    onSuccess: (resp) => setJobId(resp.job_id),
    onError: (e) =>
      setSaveError(e instanceof Error ? e : new Error("unknown")),
  });

  const runSpawn = () => {
    // Remember these choices for the next run's Save step (not the
    // output folder, which is derived per run from the source).
    saveLastUsedSaveOutputs({
      exportEnabled: exportOpts.enabled,
      csv: exportOpts.csv,
      xlsx: exportOpts.xlsx,
      recognitionJson: exportOpts.recognitionJson,
      mediaEnabled: separate.enabled,
      groupBy: separate.groupBy,
      groupEvents: separate.groupEvents,
      speciesLast: separate.speciesLast,
      copyEmpties: separate.copyEmpties,
      mediaConfidence: separate.mediaConfidence,
      drawBoxes: visualise.enabled,
      blur: anonymise.enabled,
    });
    spawn.mutate(
      buildRequest(
        effectiveOutputDir,
        separate,
        visualise,
        anonymise,
        exportOpts,
        allLabelIds,
      ),
    );
  };

  const saveAll = () => {
    setSaveError(null);
    setResult(null);
    runSpawn();
  };

  const onJobComplete = (data: SaveOutputsResult) => {
    setResult(data);
    setJobId(null);
  };

  const onJobError = (message: string) => {
    setSaveError(new Error(message));
    setJobId(null);
  };

  const onJobCancelled = () => {
    setJobId(null);
  };

  const clearResult = () => setResult(null);

  const handleBrowse = async () => {
    if (!isElectron() || !window.electronAPI?.selectFolder) return;
    const folder = await window.electronAPI.selectFolder();
    if (folder) setOutputDir(folder);
  };

  const handleOpenResults = async () => {
    if (!result || !window.electronAPI?.openPath) return;
    await window.electronAPI.openPath(result.output_dir);
  };

  const exportPicked =
    exportOpts.enabled &&
    (exportOpts.csv || exportOpts.xlsx || exportOpts.recognitionJson);
  const canSave =
    !!effectiveOutputDir &&
    !spawn.isPending &&
    jobId === null &&
    (separate.enabled || exportPicked);

  return {
    outputDir,
    setOutputDir,
    effectiveOutputDir,
    separate,
    setSeparate,
    labelTree: labelTree ?? null,
    visualise,
    setVisualise,
    anonymise,
    setAnonymise,
    exportOpts,
    setExportOpts,
    promoteOpen,
    setPromoteOpen,
    jobId,
    result,
    isSpawning: spawn.isPending,
    saveError,
    saveAll,
    onJobComplete,
    onJobError,
    onJobCancelled,
    clearResult,
    handleBrowse,
    handleOpenResults,
    exportPicked,
    canSave,
  };
}
