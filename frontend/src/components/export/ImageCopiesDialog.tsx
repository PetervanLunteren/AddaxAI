/**
 * Image copies: the folder-run media options for a project, in one
 * dialog on the Export page.
 *
 * Pick where the copies go, which files (liked, verified, a date
 * range), which labels, and the same folder / boxes / blur options a
 * folder run offers. A summary line from the preview endpoint says how
 * many files and bytes will be written before anything is. Starting
 * runs the same save-outputs job a folder run uses; progress and the
 * result screen reuse its pieces.
 *
 * The label picker starts with nothing picked on purpose: a project can
 * hold a million files, so copying all of them is a deliberate click on
 * "Select all", not the default.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, FolderOpen } from "lucide-react";

import { eventsApi } from "../../api/events";
import {
  exportApi,
  type ExportScope,
  type ImageCopiesOptions,
} from "../../api/export";
import type { SaveOutputsResult } from "../../api/folder-runs";
import {
  FILTER_DEBOUNCE_MS,
  useDebouncedValue,
} from "../../hooks/useDebouncedValue";
import { DEFAULT_COUNTING_THRESHOLD } from "../../lib/confidence";
import { getSpeciesNameMode } from "../../lib/species-name-mode";
import {
  collectIssues,
  excludedLabelIds,
  type AnonymiseState,
  type SeparateState,
  type VisualiseState,
} from "../../pages/folder-run/save/media-options";
import {
  CaptionedCheckbox,
  IssuesPanel,
  MediaBody,
} from "../../pages/folder-run/save/SaveShared";
import { FolderSelector } from "../analyses/FolderSelector";
import { JobProgressModal } from "../folder-run/JobProgressModal";
import { SaveOutputsProgress } from "../folder-run/SaveOutputsProgress";
import { useSaveOutputsProgress } from "../folder-run/useSaveOutputsProgress";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import { DateRangePicker } from "../ui/date-range-picker";
import { formatBytes } from "../ui/info-sheet-parts";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { NextStepRow } from "../ui/next-step-row";

interface ImageCopiesDialogProps {
  projectId: string;
  /** Undefined = whole project. The row is disabled when the picker
   *  reports an empty list, so the dialog never sees one. */
  scope: ExportScope | undefined;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const INITIAL_SEPARATE: SeparateState = {
  enabled: true,
  groupBy: "flat",
  groupEvents: true,
  speciesLast: false,
  copyEmpties: false,
  mediaThreshold: DEFAULT_COUNTING_THRESHOLD,
  includedLabelIds: [],
};

export function ImageCopiesDialog({
  projectId,
  scope,
  open,
  onOpenChange,
}: ImageCopiesDialogProps) {
  const [outputDir, setOutputDir] = useState("");
  const [separate, setSeparate] = useState<SeparateState>(INITIAL_SEPARATE);
  const [visualise, setVisualise] = useState<VisualiseState>({ enabled: false });
  const [anonymise, setAnonymise] = useState<AnonymiseState>({ enabled: false });
  const [likedOnly, setLikedOnly] = useState(false);
  const [verifiedOnly, setVerifiedOnly] = useState(false);
  const [dateFrom, setDateFrom] = useState<string | undefined>(undefined);
  const [dateTo, setDateTo] = useState<string | undefined>(undefined);

  const [jobId, setJobId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [result, setResult] = useState<SaveOutputsResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Project-wide label tree, counted by file so the modal reads "N
  // files" per species. Not narrowed to the picked deployments: a
  // species absent from them simply yields no files.
  const { data: labelTree } = useQuery({
    queryKey: ["label-tree", projectId, "file"],
    queryFn: () => eventsApi.getLabelTree(projectId, "file"),
    enabled: open,
  });
  const allLabelIds = labelTree?.all_leaf_ids ?? [];

  // Date bounds for the picker, the same source the Labels page uses.
  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId),
    enabled: open,
  });

  const nameMode = getSpeciesNameMode();
  const nothingPicked = separate.includedLabelIds.length === 0;
  // A drag on the slider fires one preview call after it settles.
  const mediaThreshold = useDebouncedValue(
    separate.mediaThreshold,
    FILTER_DEBOUNCE_MS,
  );

  const options: ImageCopiesOptions = {
    media_threshold: mediaThreshold,
    separate_group_by: separate.groupBy,
    group_events: separate.groupEvents,
    separate_species_last: separate.speciesLast,
    include_empty: separate.copyEmpties,
    // "none": while nothing is picked every label is excluded, so the
    // summary honestly reads 0 files instead of promising everything.
    excluded_label_ids: excludedLabelIds(separate, allLabelIds, "none"),
    draw_bboxes: visualise.enabled,
    anonymise: anonymise.enabled,
    name_mode: nameMode,
    deployment_ids: scope?.deploymentIds,
    favorited_only: likedOnly,
    verified_only: verifiedOnly,
    date_from: dateFrom,
    date_to: dateTo,
  };

  const { data: preview, isFetching: previewLoading } = useQuery({
    queryKey: ["image-copies-preview", projectId, options],
    queryFn: () => exportApi.previewImageCopies(projectId, options),
    enabled: open && !nothingPicked,
    staleTime: 30_000,
  });

  const progress = useSaveOutputsProgress({
    jobId,
    onComplete: (data) => {
      setResult(data);
      setJobId(null);
    },
    onError: (msg) => {
      setError(msg);
      setJobId(null);
    },
    onCancelled: () => setJobId(null),
  });

  const running = jobId !== null;
  const inScope = preview?.in_scope_files ?? 0;
  const canStart =
    !!outputDir &&
    !nothingPicked &&
    inScope > 0 &&
    !previewLoading &&
    !starting &&
    !running;

  const close = (next: boolean) => {
    if (running) return;
    if (!next) {
      setResult(null);
      setError(null);
    }
    onOpenChange(next);
  };

  const start = async () => {
    setError(null);
    setStarting(true);
    try {
      const { job_id } = await exportApi.prepareImageCopies(projectId, {
        ...options,
        // Not the debounced value: what the slider shows is what runs.
        media_threshold: separate.mediaThreshold,
        output_dir: outputDir,
      });
      setJobId(job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setStarting(false);
    }
  };

  const openResults = async () => {
    if (!result || !window.electronAPI?.openPath) return;
    await window.electronAPI.openPath(result.output_dir);
  };

  // What landed on disk: separation always runs for this export and
  // counts one placement per copied file, with the label filter and the
  // empties toggle already applied. `source_file_count` is the scope
  // before those, so it overstates.
  const copied = result?.separate_folders?.written_count ?? 0;

  const summary = nothingPicked
    ? "Pick at least one label to copy."
    : previewLoading || !preview
      ? "Counting..."
      : inScope === 0
        ? "Nothing to copy with these filters."
        : `${inScope.toLocaleString()} ${inScope === 1 ? "file" : "files"}, ${formatBytes(
            preview.in_scope_bytes,
          )}`;

  return (
    <>
      <Dialog open={open && !running} onOpenChange={close}>
        <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
          {result ? (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-primary-ink" />
                  Copies written
                </DialogTitle>
                <DialogDescription>
                  {copied.toLocaleString()} {copied === 1 ? "file" : "files"}{" "}
                  copied into addaxai-media.
                </DialogDescription>
              </DialogHeader>
              <div className="rounded-md border bg-muted/30 p-3 text-xs">
                <p className="mb-1 text-muted-foreground">Saved to</p>
                <code
                  className="block truncate text-left font-mono [direction:rtl]"
                  title={result.output_dir}
                >
                  <bdi>{result.output_dir}</bdi>
                </code>
              </div>
              {collectIssues(result).length > 0 && (
                <IssuesPanel issues={collectIssues(result)} />
              )}
              {window.electronAPI?.openPath && (
                <NextStepRow
                  icon={FolderOpen}
                  title="Open output folder"
                  description="Show the copies AddaxAI wrote."
                  onClick={() => void openResults()}
                />
              )}
              <DialogFooter>
                <Button type="button" onClick={() => close(false)}>
                  Close
                </Button>
              </DialogFooter>
            </>
          ) : (
            <>
              <DialogHeader>
                <DialogTitle>Image copies</DialogTitle>
                <DialogDescription>
                  Copies of your images in species folders, with your labels
                  in the image metadata. Originals are never touched.
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-5 text-sm">
                <div className="space-y-2">
                  <span className="block font-medium">Output folder</span>
                  <FolderSelector
                    value={outputDir || null}
                    onChange={setOutputDir}
                    hideLabel
                    hideScanResult
                    noScan
                  />
                  <p className="text-xs text-muted-foreground">
                    Copies go into an addaxai-media folder inside the folder
                    you pick, as species / camera folder / your subfolders.
                    Exporting again into the same folder replaces it.
                  </p>
                </div>

                <div className="space-y-1">
                  <span className="block font-medium">Which files</span>
                  <div className="divide-y [&>*:last-child]:pb-0">
                    <div className="grid grid-cols-[2fr_1fr] items-center gap-3 py-3">
                      <span>
                        Capture dates
                        <span className="mt-0.5 block text-xs text-muted-foreground">
                          Files without a capture time are left out once a
                          date is set
                        </span>
                      </span>
                      <DateRangePicker
                        from={dateFrom}
                        to={dateTo}
                        onChange={(r) => {
                          setDateFrom(r.from);
                          setDateTo(r.to);
                        }}
                        minDate={filterOptions?.date_range?.min}
                        maxDate={filterOptions?.date_range?.max}
                      />
                    </div>
                    <CaptionedCheckbox
                      checked={likedOnly}
                      onChange={setLikedOnly}
                      label="Only liked files"
                      caption="The files you marked with a heart on the Labels page"
                    />
                    <CaptionedCheckbox
                      checked={verifiedOnly}
                      onChange={setVerifiedOnly}
                      label="Only verified files"
                      caption="Files where every box was verified"
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <span className="block font-medium">Copies</span>
                  <MediaBody
                    form={{
                      separate,
                      setSeparate,
                      labelTree: labelTree ?? null,
                      visualise,
                      setVisualise,
                      anonymise,
                      setAnonymise,
                    }}
                    emptyMeans="none"
                  />
                </div>

                {error && (
                  <Callout variant="error" size="compact">
                    {error}
                  </Callout>
                )}
              </div>

              <DialogFooter className="items-center gap-3 sm:justify-between">
                <span className="text-sm text-muted-foreground">{summary}</span>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => close(false)}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="button"
                    onClick={() => void start()}
                    disabled={!canStart}
                  >
                    {starting ? "Starting..." : "Start"}
                  </Button>
                </div>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>

      <JobProgressModal
        open={running}
        title="Copying images"
        isCancelling={progress.isCancelling}
        onCancel={progress.cancel}
      >
        <SaveOutputsProgress
          modules={progress.modules}
          currentModule={progress.currentModule}
          moduleIndex={progress.moduleIndex}
          totalModules={progress.totalModules}
          message={progress.message}
          phaseProgress={progress.phaseProgress}
        />
      </JobProgressModal>
    </>
  );
}
