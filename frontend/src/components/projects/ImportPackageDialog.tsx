/**
 * Import a project package: pick the folder, see what the import would
 * do, confirm, watch it run, land on the project.
 *
 * The preview comes from the backend and is the whole decision surface.
 * A package whose project is not on this computer creates it; one whose
 * project is here adds its deployments, and the ones already here are
 * skipped or replaced, the user's choice. Replacing deletes this
 * computer's version of those deployments, so it goes through the
 * type-to-confirm gate every other such delete uses.
 */

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

import {
  packagesApi,
  type PackageImportResult,
  type PackagePreview,
} from "../../api/packages";
import { useTaskProgress } from "../../hooks/useTaskProgress";
import { ApiError } from "../../lib/api-client";
import { invalidateProjectData } from "../../lib/invalidate-project";
import { FolderSelector } from "../analyses/FolderSelector";
import { JobMetricsProgress } from "../export/JobMetricsProgress";
import { JobProgressModal } from "../folder-run/JobProgressModal";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import { CaptionedRadioGroup } from "../ui/captioned-radio-group";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { TypeToConfirmDialog } from "../ui/type-to-confirm-dialog";

type Existing = "skip" | "replace";

const EXISTING_OPTIONS = [
  {
    value: "replace" as const,
    label: "Replace them with the bundle's version",
    caption:
      "For work coming back from a colleague. Their labels, corrections and counts win; what this computer had for those deployments is deleted.",
  },
  {
    value: "skip" as const,
    label: "Leave them as they are",
    caption: "Only the deployments that are not here yet are added.",
  },
];

interface ImportPackageDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function plural(n: number, one: string, many: string): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`;
}

export function ImportPackageDialog({ open, onOpenChange }: ImportPackageDialogProps) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [path, setPath] = useState("");
  const [existing, setExisting] = useState<Existing>("replace");
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const preview = useQuery({
    queryKey: ["package-preview", path],
    queryFn: () => packagesApi.preview(path),
    enabled: open && path.length > 0,
    retry: false,
    // Never from cache: the same package reads differently after it was
    // imported once, and a stale "create" preview must not hide the
    // deployments a Replace would delete.
    staleTime: 0,
    gcTime: 0,
  });
  const data: PackagePreview | undefined = preview.data;
  const previewError =
    preview.error instanceof ApiError
      ? preview.error.message
      : preview.error
        ? "Could not read the bundle."
        : null;

  const existingCount = data ? data.deployments.filter((d) => d.exists).length : 0;
  const newCount = data ? data.deployments.length - existingCount : 0;
  const replacing = existing === "replace" && existingCount > 0;
  const running = jobId !== null;
  const canImport =
    !!data && data.problems.length === 0 && !starting && (newCount > 0 || replacing);

  const progress = useTaskProgress({
    taskId: jobId,
    onComplete: (payload) => {
      const result = payload as unknown as PackageImportResult;
      setJobId(null);
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
      invalidateProjectData(queryClient, result.project_id);
      reset();
      onOpenChange(false);
      toast.success(
        result.project_created
          ? `Project created with ${plural(result.deployments_created, "deployment", "deployments")}`
          : `${plural(result.deployments_created, "deployment", "deployments")} added, ${result.deployments_replaced.toLocaleString()} replaced, ${result.deployments_skipped.toLocaleString()} left as they were`,
      );
      navigate(`/projects/${result.project_id}/dashboard`);
    },
    onError: (msg) => {
      setJobId(null);
      setError(msg);
    },
  });

  const reset = () => {
    setPath("");
    setExisting("replace");
    setError(null);
    setConfirmOpen(false);
  };

  const close = (next: boolean) => {
    if (running) return;
    if (!next) reset();
    onOpenChange(next);
  };

  const start = async () => {
    setError(null);
    setStarting(true);
    try {
      // `replacing`, not the radio: the request may only replace what
      // this preview showed and the confirm gate covered.
      const { job_id } = await packagesApi.importPackage(path, replacing);
      setConfirmOpen(false);
      setJobId(job_id);
    } catch (err) {
      setConfirmOpen(false);
      setError(err instanceof Error ? err.message : "Import failed");
    } finally {
      setStarting(false);
    }
  };

  const onImportClick = () => {
    if (replacing) setConfirmOpen(true);
    else void start();
  };

  const headline = (() => {
    if (!data) return null;
    if (!data.project_exists) {
      return `Create project "${data.project_name}" with ${plural(data.deployments.length, "deployment", "deployments")}.`;
    }
    const parts = [`Add to "${data.project_name}": ${plural(newCount, "new deployment", "new deployments")}`];
    if (existingCount > 0) parts.push(`${existingCount.toLocaleString()} already on this computer`);
    return `${parts.join(", ")}.`;
  })();

  const details: string[] = [];
  if (data) {
    const files = data.deployments.reduce((n, d) => n + d.files, 0);
    details.push(plural(files, "file", "files"));
    details.push(
      data.includes_media ? "photos and videos included" : "data only, no photos",
    );
  }

  return (
    <>
      <Dialog open={open && !running && !confirmOpen} onOpenChange={close}>
        <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Import bundle</DialogTitle>
            <DialogDescription>
              An AddaxAI bundle is a folder written by the Export page of
              another AddaxAI, its name ending in .addaxai-bundle. It opens as
              a new project, or adds to the same project when that is already
              here.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <FolderSelector
              value={path || null}
              onChange={(p) => {
                setPath(p);
                setError(null);
              }}
              hideLabel
              hideScanResult
              noScan
              pickerTitle="Choose an AddaxAI bundle"
            />

            {preview.isFetching && (
              <p className="text-sm text-muted-foreground">Reading the bundle...</p>
            )}

            {previewError && (
              <Callout variant="error" size="compact">
                {previewError}
              </Callout>
            )}

            {data && (
              <div className="space-y-3">
                <p className="text-sm font-medium">{headline}</p>
                <p className="text-xs text-muted-foreground">
                  {details.join(", ")}. Made with AddaxAI {data.app_version}.
                </p>

                {data.problems.map((p) => (
                  <Callout key={p} variant="error" size="compact">
                    {p}
                  </Callout>
                ))}
                {data.warnings.map((w) => (
                  <Callout key={w} variant="warning" size="compact">
                    {w}
                  </Callout>
                ))}
                {data.includes_media && data.problems.length === 0 && (
                  <Callout variant="info" size="compact">
                    The photos stay inside the bundle. Keep it where it is,
                    or move it to its final place before importing.
                  </Callout>
                )}

                {existingCount > 0 && data.problems.length === 0 && (
                  <CaptionedRadioGroup
                    name="package-existing"
                    value={existing}
                    onChange={setExisting}
                    options={EXISTING_OPTIONS}
                  />
                )}

                {data.problems.length === 0 && newCount === 0 && !replacing && (
                  <p className="text-sm text-muted-foreground">
                    Nothing to import: every deployment in this bundle is
                    already here. Choose Replace to take the bundle's version.
                  </p>
                )}
              </div>
            )}

            {error && (
              <Callout variant="error" size="compact">
                {error}
              </Callout>
            )}
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => close(false)}>
              Cancel
            </Button>
            <Button type="button" onClick={onImportClick} disabled={!canImport}>
              {starting ? "Starting..." : "Import"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <TypeToConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        title="Replace deployments?"
        confirmWord="REPLACE"
        confirmLabel="Replace and import"
        pendingLabel="Starting..."
        onConfirm={() => void start()}
        isPending={starting}
        error={error}
        variant="destructive"
      >
        <Callout variant="warning">
          This replaces {plural(existingCount, "deployment", "deployments")} on
          this computer with the bundle's version, including any labels,
          corrections and counts made here. A backup of the database is taken
          first.
        </Callout>
      </TypeToConfirmDialog>

      <JobProgressModal open={running} title="Importing bundle" isCancelling={false}>
        <JobMetricsProgress
          message={progress.message}
          metrics={progress.metrics}
          phaseProgress={progress.progress}
        />
      </JobProgressModal>
    </>
  );
}
