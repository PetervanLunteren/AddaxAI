/**
 * Recognition file: writes `addaxai-recognitions.json` into the folder of
 * every selected deployment, the file Timelapse reads. The one export
 * without a Save in: the file belongs next to the photos.
 */

import { useState } from "react";

import {
  exportApi,
  type ExportScope,
  type RecognitionJsonExportResult,
} from "../../api/export";
import { basename } from "../../lib/path-utils";
import { Callout } from "../ui/callout";
import { ExportDialog, type ExportResultView } from "./ExportDialog";
import type { ExportDialogComponentProps } from "./FileExportDialogs";

const folders = (n: number) => `${n.toLocaleString()} ${n === 1 ? "folder" : "folders"}`;

function describeResult(data: unknown): ExportResultView {
  const { written, skipped } = data as RecognitionJsonExportResult;
  return {
    title: `Written into ${folders(written.length)}`,
    details:
      skipped.length > 0 ? (
        <Callout variant="warning" size="compact">
          <p>Skipped {folders(skipped.length)}:</p>
          <ul className="mt-1 list-disc pl-5">
            {skipped.map((s) => (
              <li key={s.deployment_id}>
                {s.folder_path ? basename(s.folder_path) : "(no folder)"}: {s.reason}
              </li>
            ))}
          </ul>
        </Callout>
      ) : undefined,
  };
}

export function RecognitionJsonExportDialog({
  projectId,
  onClose,
}: ExportDialogComponentProps) {
  const [scope, setScope] = useState<ExportScope | undefined>();
  return (
    <ExportDialog
      projectId={projectId}
      title="Recognition file"
      description="The file Timelapse reads, in the MegaDetector output format, with your labels and corrections."
      onClose={onClose}
      scope={scope}
      onScopeChange={setScope}
      needsFolder={false}
      start={async () => {
        const { job_id } = await exportApi.prepareRecognitionJson(projectId, scope);
        return { jobId: job_id };
      }}
      describeResult={describeResult}
    >
      <p className="text-sm text-muted-foreground">
        AddaxAI writes <code>addaxai-recognitions.json</code> into the folder of every
        selected deployment, with paths relative to that folder. A file already there is
        replaced. Open Timelapse on a deployment folder to import it.
      </p>
    </ExportDialog>
  );
}
