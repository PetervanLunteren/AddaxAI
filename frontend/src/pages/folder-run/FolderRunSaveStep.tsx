/**
 * Step 5: Save outputs.
 *
 * Two-column layout on wide screens: options on the left (four
 * GroupCards: Separate / Visualise / Write EXIF / Export), live
 * folder preview on the right. The preview reads from
 * `/api/folder-runs/{id}/output-preview` and updates reactively as
 * the user ticks options, so the disk impact of each choice is
 * visible before the user hits Save. On narrower viewports the
 * preview stacks below the options column.
 *
 * Shared state, submit logic, and the small building blocks live in
 * `save/useSaveOutputsForm` and `save/SaveShared` so this file stays
 * focused on layout.
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { Card, CardContent } from "../../components/ui/card";
import {
  BackSaveBar,
  CompletionScreen,
  ExportBody,
  OutputFolderField,
  SaveErrorLine,
  SeparateBody,
} from "./save/SaveShared";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../../components/ui/dialog";
import { Button } from "../../components/ui/button";
import { OutputFilterCard } from "./save/OutputFilterCard";
import { OutputPreviewPanel } from "./save/OutputPreviewPanel";
import { useSaveOutputsForm } from "./save/useSaveOutputsForm";
import { useFolderRun } from "./FolderRunLayout";
import { JobProgressModal } from "../../components/folder-run/JobProgressModal";
import { SaveOutputsProgress } from "../../components/folder-run/SaveOutputsProgress";
import { StepHeader } from "../../components/folder-run/StepHeader";
import {
  folderRunsApi,
  type SaveOutputsResult,
} from "../../api/folder-runs";
import { useTaskProgress } from "../../hooks/useTaskProgress";

export function FolderRunSaveStep() {
  const navigate = useNavigate();
  const { runId, run, isLoading } = useFolderRun();

  const form = useSaveOutputsForm({
    runId: runId ?? "",
    sourceFolder: run?.queue_entry?.folder_path ?? undefined,
  });

  // Job-progress state driven by the worker's WebSocket events.
  // Each event carries the ordered module list, the active module
  // (or null at start / end), and the index into the list — enough
  // for SaveOutputsProgress to render the checklist.
  const [modules, setModules] = useState<string[]>([]);
  const [currentModule, setCurrentModule] = useState<string | null>(
    null,
  );
  const [moduleIndex, setModuleIndex] = useState(0);
  const [totalModules, setTotalModules] = useState(0);
  const [isCancelling, setIsCancelling] = useState(false);

  const progress = useTaskProgress({
    taskId: form.jobId,
    onProgress: (msg) => {
      const d = msg.data ?? {};
      if (Array.isArray(d.modules)) setModules(d.modules as string[]);
      if ("current_module" in d) {
        setCurrentModule(d.current_module as string | null);
      }
      if (typeof d.module_index === "number") {
        setModuleIndex(d.module_index);
      }
      if (typeof d.total_modules === "number") {
        setTotalModules(d.total_modules);
      }
    },
    onComplete: (data) => {
      form.onJobComplete(data as unknown as SaveOutputsResult);
      setIsCancelling(false);
    },
    onError: (msg) => {
      form.onJobError(msg);
      setIsCancelling(false);
    },
    onCancelled: (msg) => {
      form.onJobCancelled(msg);
      setIsCancelling(false);
    },
  });

  // Sorted exclusion list as part of the query key so React Query
  // refetches the preview whenever the filter changes. Sort first
  // so {dog, wolf} and {wolf, dog} map to the same cache entry.
  const excludedKey = [...form.excludedLabelIds].sort();
  const { data: preview, isLoading: previewLoading } = useQuery({
    queryKey: ["folder-run-output-preview", runId, excludedKey],
    queryFn: () =>
      folderRunsApi.getOutputPreview(runId!, {
        excluded_label_ids: excludedKey,
      }),
    enabled: !!runId,
    staleTime: 30_000,
  });

  if (!runId) {
    navigate("/folder-runs/new", { replace: true });
    return null;
  }

  if (isLoading || !run) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-sm text-muted-foreground">
          Loading run...
        </CardContent>
      </Card>
    );
  }

  if (form.result) {
    return (
      <CompletionScreen
        runId={runId}
        runName={run.project.name}
        form={form}
      />
    );
  }

  return (
    <>
      <StepHeader
        title="Save outputs"
        caption="Pick what to write to disk and where to save it."
      />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px] lg:items-start">
      <div className="space-y-6">
        <OutputFolderField form={form} />

        <OutputFilterCard form={form} projectId={runId} />

        <GroupCard
          title="Separate files into subfolders"
          caption="Group files by species so you can browse them in your file manager."
          enabled={form.separate.enabled}
          onEnabledChange={(v) =>
            form.setSeparate({ ...form.separate, enabled: v })
          }
        >
          <SeparateBody form={form} />
        </GroupCard>

        <GroupCard
          title="Visualise detections"
          caption="Save a copy of each image with detection boxes drawn on top."
          enabled={form.visualise.enabled}
          onEnabledChange={(v) =>
            form.setVisualise({ ...form.visualise, enabled: v })
          }
        />

        <GroupCard
          title="Anonymise people and vehicles"
          caption="Save a copy of each image with people and vehicles blurred."
          enabled={form.anonymise.enabled}
          onEnabledChange={(v) =>
            form.setAnonymise({ ...form.anonymise, enabled: v })
          }
        />

        <GroupCard
          title="Export results and metadata"
          caption="Save flat observation tables and the recognition JSON."
          enabled={form.exportOpts.enabled}
          onEnabledChange={(v) =>
            form.setExportOpts({ ...form.exportOpts, enabled: v })
          }
        >
          <ExportBody form={form} />
        </GroupCard>

        <SaveErrorLine error={form.saveError} />
        <BackSaveBar runId={runId} form={form} />
      </div>

      <OutputPreviewPanel
        form={form}
        preview={preview}
        runName={run.project.name}
        isLoading={previewLoading}
      />

      <JobProgressModal
        open={!!form.jobId}
        title="Saving outputs"
        isCancelling={isCancelling}
        onCancel={() => {
          setIsCancelling(true);
          progress.cancel();
        }}
      >
        <SaveOutputsProgress
          modules={modules}
          currentModule={currentModule}
          moduleIndex={moduleIndex}
          totalModules={totalModules}
          message={progress.message}
          progress={progress.progress}
        />
      </JobProgressModal>

      <ConfirmMoveDialog
        open={form.pendingMoveConfirm}
        onCancel={form.cancelMoveConfirm}
        onConfirm={form.confirmMoveAndSave}
      />
      </div>
    </>
  );
}

function ConfirmMoveDialog({
  open,
  onCancel,
  onConfirm,
}: {
  open: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={(v) => (v ? null : onCancel())}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Move files from the source folder?</DialogTitle>
          <DialogDescription>
            Move places each file under the output folder and removes
            it from its original location. The source folder will no
            longer contain those files after the run. Copy is the
            safer option if you want to keep the originals in place.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={onConfirm}>
            Move and save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function GroupCard({
  title,
  caption,
  enabled,
  onEnabledChange,
  children,
}: {
  title: string;
  /** One-line description of what this card does. Lives directly
   * under the title so the user can decide without ticking the box. */
  caption: string;
  enabled: boolean;
  onEnabledChange: (v: boolean) => void;
  children?: React.ReactNode;
}) {
  return (
    <Card>
      <CardContent className="space-y-4 p-6">
        <label className="flex cursor-pointer items-start gap-2">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => onEnabledChange(e.target.checked)}
            className="mt-0.5 h-4 w-4 accent-primary"
          />
          <span>
            <span className="block text-sm font-semibold">{title}</span>
            <span className="mt-0.5 block text-xs text-muted-foreground">
              {caption}
            </span>
          </span>
        </label>
        {enabled && children && <div className="pl-6">{children}</div>}
      </CardContent>
    </Card>
  );
}
