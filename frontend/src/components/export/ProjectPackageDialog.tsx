/**
 * AddaxAI bundle (a project package in the code): the selected deployments, with or without their
 * photos, as a folder another AddaxAI imports. On the shared export
 * dialog; the body is the contents choice and a size line.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { exportApi, type ExportScope } from "../../api/export";
import type { PackageExportResult } from "../../api/packages";
import { CaptionedRadioGroup } from "../ui/captioned-radio-group";
import { formatBytes } from "../ui/info-sheet-parts";
import { ExportDialog, ExportSection, type ExportResultView } from "./ExportDialog";
import type { ExportDialogComponentProps } from "./FileExportDialogs";

type Contents = "data" | "media";

const CONTENTS_OPTIONS = [
  {
    value: "data" as const,
    label: "Data only",
    caption:
      "Labels, corrections and counts. The photos and videos must already be on the other computer, where you reconnect each folder once.",
  },
  {
    value: "media" as const,
    label: "With photos and videos",
    caption:
      "Self-contained: the deployment folders are copied into the bundle. As large as those folders. Keep the bundle where it is after importing.",
  },
];

const plural = (n: number, one: string, many: string) =>
  `${n.toLocaleString()} ${n === 1 ? one : many}`;

function describeResult(data: unknown): ExportResultView {
  const result = data as PackageExportResult;
  return {
    title: "Bundle written",
    folder: result.package_dir,
    details: (
      <p className="text-sm text-muted-foreground">
        {plural(result.deployments, "deployment", "deployments")},{" "}
        {plural(result.files, "file", "files")}
        {result.includes_media ? ", photos and videos included." : ", data only."} On the
        other computer, choose {"File > Import bundle…"}. Importing a later bundle
        of the same project adds new deployments and can replace the ones already there.
      </p>
    ),
  };
}

export function ProjectPackageDialog({ projectId, onClose }: ExportDialogComponentProps) {
  const [scope, setScope] = useState<ExportScope | undefined>();
  const [contents, setContents] = useState<Contents>("data");
  const nothingSelected = scope?.deploymentIds?.length === 0;

  const { data: size } = useQuery({
    queryKey: ["package-size", projectId, scope?.deploymentIds ?? "all"],
    queryFn: () => exportApi.packageSize(projectId, scope),
    enabled: !nothingSelected,
  });

  const bytes = size ? size.data_bytes + (contents === "media" ? size.media_bytes : 0) : 0;
  const sizeLine = size
    ? `${plural(size.deployments, "deployment", "deployments")}, ${plural(size.files, "file", "files")}, about ${formatBytes(bytes)}.`
    : "";

  return (
    <ExportDialog
      projectId={projectId}
      title="AddaxAI bundle"
      description="Your labels and corrections for the selected deployments or the whole project, to open in another AddaxAI: to continue on another computer, or to let a team member check them and send them back."
      onClose={onClose}
      scope={scope}
      onScopeChange={setScope}
      folderCaption="The bundle is written as a new folder inside the folder you pick, named after the project."
      start={async (outputDir) => {
        const { job_id } = await exportApi.preparePackage(projectId, {
          output_dir: outputDir,
          deployment_ids: scope?.deploymentIds,
          include_media: contents === "media",
        });
        return { jobId: job_id };
      }}
      describeResult={describeResult}
      cancellable
    >
      <ExportSection title="Contents">
        <CaptionedRadioGroup
          name="package-contents"
          value={contents}
          onChange={setContents}
          options={CONTENTS_OPTIONS}
        />
        <p className="text-xs text-muted-foreground">{sizeLine}</p>
      </ExportSection>
    </ExportDialog>
  );
}
