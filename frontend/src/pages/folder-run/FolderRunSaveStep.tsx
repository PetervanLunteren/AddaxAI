/**
 * Step 5: Save outputs.
 *
 * Two-column layout on wide screens: options on the left (two output
 * groups — "Export results" on by default, and an opt-in "Save copies
 * of your media"), live folder preview on the right. The preview comes
 * from the shared media hook and updates reactively as the user ticks
 * options, so the disk impact of each choice is visible before the
 * user hits Save. On narrower viewports the preview stacks below the
 * options column.
 *
 * Shared state, submit logic, and the small building blocks live in
 * `save/useSaveOutputsForm`, `save/useMediaOptions` and
 * `save/SaveShared` so this file stays focused on layout. The media
 * card and the preview are the same pieces the project image copies
 * dialog renders.
 */

import { useNavigate } from "react-router-dom";

import { Card, CardContent } from "../../components/ui/card";
import {
  BackSaveBar,
  CompletionDialog,
  ExportBody,
  MediaBody,
  OutputFolderField,
} from "./save/SaveShared";
import { OutputPreviewPanel } from "./save/OutputPreviewPanel";
import { useSaveOutputsForm } from "./save/useSaveOutputsForm";
import { useFolderRun } from "./FolderRunLayout";
import { JobProgressModal } from "../../components/folder-run/JobProgressModal";
import { SaveOutputsProgress } from "../../components/folder-run/SaveOutputsProgress";
import { StepHeader } from "../../components/folder-run/StepHeader";
import { useSaveOutputsProgress } from "../../components/folder-run/useSaveOutputsProgress";

export function FolderRunSaveStep() {
  const navigate = useNavigate();
  const { runId, run, isLoading } = useFolderRun();

  const form = useSaveOutputsForm({
    runId: runId ?? "",
    sourceFolder: run?.queue_entry?.folder_path ?? undefined,
    projectThreshold: run?.project?.counting_threshold,
  });

  // Job progress driven by the worker's WebSocket events, decoded for
  // the per-module checklist by the hook the image copies dialog shares.
  const progress = useSaveOutputsProgress({
    jobId: form.jobId,
    onComplete: form.onJobComplete,
    onError: form.onJobError,
    onCancelled: form.onJobCancelled,
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

  return (
    <>
      <StepHeader
        title="Save outputs"
        caption="Pick what to write to disk and where to save it."
      />
      {/* pb-6 puts the same 24px between the last card and the sticky bar
          as the cards keep between each other. Without it the bar sat flush
          against the media card; the Labels step's pb-24 read as a hole
          here, because this page is a stack of cards and that one is a
          full-height grid. */}
      <div className="grid gap-6 pb-6 lg:grid-cols-[minmax(0,1fr)_360px] lg:items-start">
      <div className="space-y-6">
        {/* "images and videos", not "originals": saving again DOES
            overwrite the addaxai-* data files from a previous save. The
            promise only holds for the user's own media, which is copied,
            never modified in place. */}
        <OutputFolderField
          value={form.outputDir}
          onChange={form.setOutputDir}
          caption="Where everything gets written. Defaults to the folder you analysed. Your original images and videos are never overwritten."
        />

        {/* Data exports: lightweight, non-destructive, on by default. */}
        <GroupCard
          title="Export results"
          caption="Data tables and the recognition file. Your media is untouched."
          enabled={form.exportOpts.enabled}
          onEnabledChange={(v) =>
            form.setExportOpts({ ...form.exportOpts, enabled: v })
          }
        >
          <ExportBody form={form} />
        </GroupCard>

        {/* Media copies: one feature with folder structure + render
            options. Off by default (opt-in), so the common run is just
            data exports. */}
        <GroupCard
          title="Save copies of your media"
          caption="Your images and videos sorted into folders. Copies only, your originals stay where they are."
          enabled={form.separate.enabled}
          onEnabledChange={(v) =>
            form.setSeparate({ ...form.separate, enabled: v })
          }
        >
          <MediaBody form={form} />
        </GroupCard>
      </div>

      <OutputPreviewPanel
        media={form}
        exportOpts={form.exportOpts}
        outputDir={form.effectiveOutputDir}
        fallbackName={run.project.name}
        preview={form.preview}
        isLoading={form.previewLoading}
      />

      <JobProgressModal
        open={!!form.jobId}
        title="Saving outputs"
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

      <CompletionDialog
        runId={runId}
        runName={run.project.name}
        form={form}
      />
      </div>

      {/* Outside the grid on purpose: the bar bleeds to the page edges, so
          it has to sit at the step root rather than inside the options
          column, where the bleed would align to the column instead. */}
      <BackSaveBar runId={runId} form={form} />
    </>
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
        {enabled && children && (
          <>
            {/* Full-bleed rule separating the master toggle from its
                options. Only shown when open, so a collapsed card stays
                a single clean row. The options sit flush at the card
                padding, not indented under the title, so the media
                panels land at the same edge as in the Image copies
                dialog, which has no master toggle. */}
            <div className="-mx-6 border-t" />
            {children}
          </>
        )}
      </CardContent>
    </Card>
  );
}
