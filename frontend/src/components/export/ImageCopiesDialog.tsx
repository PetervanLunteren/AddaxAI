/**
 * Image copies: the folder-run media options and their tree preview, on
 * the shared export dialog. The options, the preview and the layout
 * memory come from `useMediaOptions`, the hook the folder-run Save step
 * composes too, so a user learns one form; starting runs the same
 * save-outputs job a folder run uses. The deployment picker stays in
 * the form's "Which files" block, next to the other filters, so this
 * dialog passes no scope to the shell.
 */

import { exportApi } from "../../api/export";
import type { SaveOutputsResult } from "../../api/folder-runs";
import { collectIssues } from "../../pages/folder-run/save/media-options";
import { OutputPreviewPanel } from "../../pages/folder-run/save/OutputPreviewPanel";
import { IssuesPanel, MediaBody } from "../../pages/folder-run/save/SaveShared";
import { useMediaOptions } from "../../pages/folder-run/save/useMediaOptions";
import { Card, CardContent } from "../ui/card";
import { ExportDialog, WIDE_COLUMNS, type ExportResultView } from "./ExportDialog";
import type { ExportDialogComponentProps } from "./FileExportDialogs";

function describeResult(data: unknown): ExportResultView {
  const result = data as SaveOutputsResult;
  // Separation always runs for this export and counts one placement per
  // copied file, with the label filter and the empties toggle applied.
  const copied = result.separate_folders?.written_count ?? 0;
  const issues = collectIssues(result);
  return {
    title: copied > 0 ? `${copied.toLocaleString()} ${copied === 1 ? "copy" : "copies"} written` : "Nothing copied",
    folder: result.output_dir,
    details: (
      <>
        <p className="text-sm text-muted-foreground">
          The copies are in the addaxai-media folder.
        </p>
        {issues.length > 0 && <IssuesPanel issues={issues} />}
      </>
    ),
  };
}

export function ImageCopiesDialog({ projectId, onClose }: ExportDialogComponentProps) {
  const media = useMediaOptions({ projectId, alwaysEnabled: true });
  return (
    <ExportDialog
      projectId={projectId}
      title="Image copies"
      description="Copies of your images in species folders, with your labels in the image metadata. Originals are never touched."
      onClose={onClose}
      canStart={media.hasFilesToCopy}
      start={async (outputDir) => {
        media.remember();
        const { job_id } = await exportApi.prepareImageCopies(projectId, {
          ...media.requestFields,
          output_dir: outputDir,
        });
        return { jobId: job_id };
      }}
      describeResult={describeResult}
      folderCaption="Copies go into an addaxai-media folder inside the folder you pick. Exporting again replaces it. Your originals are never touched."
      cancellable
      wide
    >
      {(outputDir) => (
        <div className={WIDE_COLUMNS}>
          <Card>
            <CardContent className="p-6">
              <MediaBody form={media} />
            </CardContent>
          </Card>
          <OutputPreviewPanel
            media={media}
            outputDir={outputDir}
            fallbackName="output-folder"
            preview={media.preview}
            isLoading={media.previewLoading}
            emptyMessage="Pick at least one deployment to see the preview"
          />
        </div>
      )}
    </ExportDialog>
  );
}
