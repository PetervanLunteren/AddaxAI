/**
 * Form state + submit logic for the Save outputs step.
 *
 * Owns the option state (groups + output dir + filter), spawns the
 * save-outputs background job, and holds the final result payload
 * the JobProgressModal hands back on completion. The page component
 * is responsible for the WebSocket subscription (``useTaskProgress``)
 * and the modal rendering — this hook just exposes the handles.
 *
 * Two safety nets live here:
 * - ``sourceFolderConflict``: blocks Save when the chosen output
 *   folder is the source folder or sits inside it. Writing copies at
 *   the root with original names would overwrite the source.
 * - ``pendingMoveConfirm``: when ``separate.method === "move"`` the
 *   click on Save flips this flag instead of spawning immediately, so
 *   the page can render a confirm dialog (Move removes files from the
 *   source folder; explicit acknowledgement required).
 */

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  folderRunsApi,
  type SaveOutputsRequest,
  type SaveOutputsResult,
  type SeparateGroupBy,
  type SeparateMode,
} from "../../../api/folder-runs";
import { isElectron } from "../../../lib/platform";

export interface SeparateState {
  enabled: boolean;
  method: SeparateMode;
  groupBy: SeparateGroupBy;
}

export interface VisualiseState {
  enabled: boolean;
}

export interface AnonymiseState {
  enabled: boolean;
}

/** Label-exclusion filter — list of LabelTaxonomy ids and / or raw
 * label strings to remove from every output. */
export type ExcludedLabelIds = readonly string[];

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
  excludedLabelIds: ExcludedLabelIds,
): SaveOutputsRequest {
  return {
    output_dir: outputDir,
    separate_folders: separate.enabled,
    separate_method: separate.method,
    separate_group_by: separate.groupBy,
    excluded_label_ids: [...excludedLabelIds],
    draw_bboxes: visualise.enabled,
    anonymise: anonymise.enabled,
    csv: exportOpts.enabled && exportOpts.csv,
    xlsx: exportOpts.enabled && exportOpts.xlsx,
    recognition_json: exportOpts.enabled && exportOpts.recognitionJson,
  };
}

/** True when ``output`` equals ``source`` or sits inside it. Handles
 * both POSIX and Windows separators so the check works regardless of
 * which one the OS picker emitted. */
function outputConflictsWithSource(
  output: string,
  source: string | undefined,
): boolean {
  if (!output || !source) return false;
  const norm = (p: string) => p.replace(/[\\/]+$/, "").replace(/\\/g, "/");
  const o = norm(output);
  const s = norm(source);
  if (!s) return false;
  return o === s || o.startsWith(s + "/");
}

export interface UseSaveOutputsFormParams {
  runId: string;
  sourceFolder: string | undefined;
}

export interface UseSaveOutputsFormResult {
  outputDir: string;
  setOutputDir: (v: string) => void;
  /** The dir the form will submit. There is no default — the user must
   * pick one explicitly. Equal to ``outputDir``. */
  effectiveOutputDir: string;
  /** True when the chosen output dir is the source folder or
   * nested inside it. Save is disabled in this state. */
  sourceFolderConflict: boolean;

  separate: SeparateState;
  setSeparate: (s: SeparateState) => void;
  visualise: VisualiseState;
  setVisualise: (s: VisualiseState) => void;
  anonymise: AnonymiseState;
  setAnonymise: (s: AnonymiseState) => void;
  exportOpts: ExportState;
  setExportOpts: (s: ExportState) => void;
  excludedLabelIds: ExcludedLabelIds;
  setExcludedLabelIds: (v: ExcludedLabelIds) => void;

  promoteOpen: boolean;
  setPromoteOpen: (v: boolean) => void;

  /** Job currently running (null when idle or after completion). */
  jobId: string | null;
  /** Final result payload from the job's complete event. */
  result: SaveOutputsResult | null;
  /** Set during the brief HTTP roundtrip that spawns the job. */
  isSpawning: boolean;
  saveError: Error | null;

  /** True while waiting for the user to confirm a Move-mode save in
   * the dialog. The page renders the dialog off this flag. */
  pendingMoveConfirm: boolean;
  /** Dismiss the Move-confirm dialog without saving. */
  cancelMoveConfirm: () => void;
  /** Confirm the Move-mode save: clears the flag and spawns the job. */
  confirmMoveAndSave: () => void;

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
  const sourceFolderConflict = outputConflictsWithSource(
    effectiveOutputDir,
    sourceFolder,
  );

  const [separate, setSeparate] = useState<SeparateState>({
    enabled: true,
    method: "copy",
    groupBy: "taxonomic",
  });
  const [visualise, setVisualise] = useState<VisualiseState>({
    enabled: false,
  });
  const [anonymise, setAnonymise] = useState<AnonymiseState>({
    enabled: false,
  });
  const [exportOpts, setExportOpts] = useState<ExportState>({
    enabled: false,
    csv: true,
    xlsx: false,
    recognitionJson: false,
  });

  const [excludedLabelIds, setExcludedLabelIdsState] = useState<
    readonly string[]
  >([]);
  const setExcludedLabelIds = (v: ExcludedLabelIds) =>
    setExcludedLabelIdsState([...v]);

  const [promoteOpen, setPromoteOpen] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<SaveOutputsResult | null>(null);
  const [saveError, setSaveError] = useState<Error | null>(null);
  const [pendingMoveConfirm, setPendingMoveConfirm] = useState(false);

  const spawn = useMutation({
    mutationFn: (payload: SaveOutputsRequest) =>
      folderRunsApi.saveOutputs(runId, payload),
    onSuccess: (resp) => setJobId(resp.job_id),
    onError: (e) =>
      setSaveError(e instanceof Error ? e : new Error("unknown")),
  });

  const runSpawn = () => {
    spawn.mutate(
      buildRequest(
        effectiveOutputDir,
        separate,
        visualise,
        anonymise,
        exportOpts,
        excludedLabelIds,
      ),
    );
  };

  const saveAll = () => {
    setSaveError(null);
    setResult(null);
    if (separate.enabled && separate.method === "move") {
      setPendingMoveConfirm(true);
      return;
    }
    runSpawn();
  };

  const cancelMoveConfirm = () => setPendingMoveConfirm(false);
  const confirmMoveAndSave = () => {
    setPendingMoveConfirm(false);
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

  const onJobCancelled = (_message: string) => {
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
    !sourceFolderConflict &&
    !spawn.isPending &&
    jobId === null &&
    (separate.enabled ||
      visualise.enabled ||
      anonymise.enabled ||
      exportPicked);

  return {
    outputDir,
    setOutputDir,
    effectiveOutputDir,
    sourceFolderConflict,
    separate,
    setSeparate,
    visualise,
    setVisualise,
    anonymise,
    setAnonymise,
    exportOpts,
    setExportOpts,
    excludedLabelIds,
    setExcludedLabelIds,
    promoteOpen,
    setPromoteOpen,
    jobId,
    result,
    isSpawning: spawn.isPending,
    saveError,
    pendingMoveConfirm,
    cancelMoveConfirm,
    confirmMoveAndSave,
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
