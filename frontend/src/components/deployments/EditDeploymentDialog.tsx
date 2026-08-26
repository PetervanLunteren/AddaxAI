/**
 * Edit deployment dialog.
 *
 * Metadata-only: site, datetime offset, paired cameras, notes, tags. Folder linking
 * is handled by the per-group RelinkGroupBanner at the top of the
 * deployments page.
 */

import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Pencil } from "lucide-react";
import { deploymentsApi } from "../../api/deployments";
import type { DeploymentResponse } from "../../api/types";
import { useFolderScan } from "../../hooks/useFolderScan";
import { useTaskProgress } from "../../hooks/useTaskProgress";
import { invalidateProjectData } from "../../lib/invalidate-project";
import {
  startReprocessIfNeeded,
  warnIfDeploymentsSkipped,
} from "../../lib/reprocessSettings";
import { formatOffset } from "../../lib/utils";
import { ApplySettingsModal } from "../settings/ApplySettingsModal";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Label } from "../ui/label";
import { Textarea } from "../ui/textarea";
import { TagsEditor } from "../ui/tags-editor";
import { AddSiteModal } from "../analyses/AddSiteModal";
import { DatetimeOffsetModal } from "../analyses/DatetimeOffsetModal";
import { PairedCamerasCheckbox } from "../analyses/PairedCamerasCheckbox";
import { SiteSelector } from "../analyses/SiteSelector";

interface EditDeploymentDialogProps {
  deployment: DeploymentResponse;
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Opens the Split dialog for the same deployment. Parent is expected to
   * close this Edit dialog before opening the Split one. */
  onSplit?: () => void;
}

export function EditDeploymentDialog({
  deployment,
  projectId,
  open,
  onOpenChange,
  onSplit,
}: EditDeploymentDialogProps) {
  const queryClient = useQueryClient();

  const [siteId, setSiteId] = useState<string | null>(deployment.site_id);
  const [datetimeOffset, setDatetimeOffset] = useState<number>(
    deployment.datetime_offset_seconds ?? 0
  );
  const [pairedCameras, setPairedCameras] = useState<boolean>(
    deployment.paired_cameras
  );
  const [notes, setNotes] = useState<string>(deployment.notes ?? "");
  const [tags, setTags] = useState<Record<string, string>>(deployment.tags ?? {});
  const [offsetModalOpen, setOffsetModalOpen] = useState(false);
  const [showAddSiteModal, setShowAddSiteModal] = useState(false);
  // A paired-cameras change regroups the deployment's events in the
  // PATCH and then needs a reprocess so smoothing runs on the new
  // grouping. Same job and same modal as saving analysis settings.
  const [reprocessJobId, setReprocessJobId] = useState<string | null>(null);

  // Reset all state when the modal opens with a different deployment
  useEffect(() => {
    if (open) {
      setSiteId(deployment.site_id);
      setDatetimeOffset(deployment.datetime_offset_seconds ?? 0);
      setPairedCameras(deployment.paired_cameras);
      setNotes(deployment.notes ?? "");
      setTags(deployment.tags ?? {});
    }
  }, [open, deployment]);

  // Only scan the folder when the dialog is open AND the folder is
  // known to be valid — otherwise the scan just errors out.
  const canAdjustDates =
    open && deployment.folder_status === "valid" && !!deployment.folder_path;
  const { data: scanResult } = useFolderScan(
    canAdjustDates ? deployment.folder_path : null
  );

  const pairedChanged = pairedCameras !== deployment.paired_cameras;

  const invalidateAfterSave = () => {
    queryClient.invalidateQueries({ queryKey: ["deployments", projectId] });
    queryClient.invalidateQueries({ queryKey: ["deployment-stats", projectId] });
    queryClient.invalidateQueries({ queryKey: ["sites-with-stats", projectId] });
    // Slideout uses ["deployments", deploymentId, "info"], not the
    // project-scoped key above, so it needs its own invalidation.
    queryClient.invalidateQueries({
      queryKey: ["deployments", deployment.id, "info"],
    });
    // Offset changes shift every File.captured_at_local in the
    // deployment, which feeds events, dashboard charts, the
    // verification grid, and timeline insights. Invalidate broadly
    // so views that read off file timestamps refresh too.
    queryClient.invalidateQueries({ queryKey: ["events"] });
    queryClient.invalidateQueries({ queryKey: ["files"] });
    queryClient.invalidateQueries({ queryKey: ["statistics"] });
  };

  const updateMutation = useMutation({
    mutationFn: async () => {
      await deploymentsApi.update(deployment.id, {
        site_id: siteId,
        datetime_offset_seconds: datetimeOffset || null,
        paired_cameras: pairedCameras,
        notes: notes.trim() || null,
        tags,
      });
      // The PATCH already regrouped the events; the reprocess re-runs
      // smoothing on them. Null when the project has no classifications.
      return pairedChanged ? await startReprocessIfNeeded(projectId) : null;
    },
    onSuccess: (jobId) => {
      if (jobId) {
        setReprocessJobId(jobId);
        return; // modal takes over; dialog closes in onComplete
      }
      invalidateAfterSave();
      onOpenChange(false);
    },
  });

  const reprocessProgress = useTaskProgress({
    taskId: reprocessJobId,
    onComplete: (data) => {
      setReprocessJobId(null);
      warnIfDeploymentsSkipped(data);
      // The reprocess rewrote labels and events project-wide.
      invalidateProjectData(queryClient, projectId);
      invalidateAfterSave();
      onOpenChange(false);
    },
    onError: () => {
      setReprocessJobId(null);
      invalidateAfterSave();
    },
  });

  const offsetButtonDisabled =
    deployment.folder_status !== "valid" || !scanResult;
  const offsetLabel =
    datetimeOffset === 0 ? "No offset applied" : formatOffset(datetimeOffset);

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit deployment</DialogTitle>
            <DialogDescription>
              Update this deployment's metadata
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            {/* Site picker. Optional: leaving it blank marks the
                deployment as site-less (batch or unknown location).
                Reuses the analyses SiteSelector so the [+] new-site
                affordance is consistent across the app. */}
            <div>
              <SiteSelector
                projectId={projectId}
                value={siteId}
                onChange={setSiteId}
                onAddNew={() => setShowAddSiteModal(true)}
              />
              {onSplit && (
                <p className="mt-1 text-xs text-muted-foreground">
                  If this deployment contains files from multiple sites,{" "}
                  <button
                    type="button"
                    onClick={onSplit}
                    className="text-primary underline underline-offset-2"
                  >
                    split
                  </button>
                  {" "}it first, then assign each part its site.
                </p>
              )}
            </div>

            {/* Datetime offset (opens DatetimeOffsetModal on click) */}
            <div className="space-y-2">
              <Label>Datetime offset</Label>
              <Button
                type="button"
                variant="outline"
                className="w-full justify-between font-normal"
                disabled={offsetButtonDisabled}
                onClick={() => setOffsetModalOpen(true)}
                title={
                  deployment.folder_status !== "valid"
                    ? "Relink the folder first to adjust dates"
                    : undefined
                }
              >
                <span
                  className={
                    datetimeOffset === 0 ? "text-muted-foreground" : ""
                  }
                >
                  {offsetLabel}
                </span>
                <Pencil className="h-4 w-4 opacity-50" />
              </Button>
            </div>

            {/* Paired cameras. Changing it regroups this deployment's
                events on save, so say what that costs before the click. */}
            <div className="space-y-3">
              <PairedCamerasCheckbox
                checked={pairedCameras}
                onChange={setPairedCameras}
              />
              {pairedChanged && (
                <Callout variant="warning" size="compact">
                  Saving regroups this deployment's events. Confirmed
                  counts stay only on events whose files stay together.
                </Callout>
              )}
            </div>

            {/* Notes */}
            <div className="space-y-2">
              <Label htmlFor="edit-deployment-notes">Notes</Label>
              <p className="text-xs text-muted-foreground">
                Free-text for your own records. Shown on the deployment's info
                panel.
              </p>
              <Textarea
                id="edit-deployment-notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                maxLength={1000}
                placeholder="e.g., Lens was covered in baboon fingerprints"
              />
            </div>

            {/* Tags */}
            <TagsEditor
              value={tags}
              onChange={setTags}
              keyPlaceholder="e.g., season"
              valuePlaceholder="e.g., wet"
              description="Labels to group and filter deployments later."
            />

            {updateMutation.isError && (
              <p className="text-sm font-medium text-destructive">
                Failed to save:{" "}
                {updateMutation.error instanceof Error
                  ? updateMutation.error.message
                  : String(updateMutation.error)}
              </p>
            )}
          </div>

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => updateMutation.mutate()}
              disabled={updateMutation.isPending || !!reprocessJobId}
            >
              {updateMutation.isPending || reprocessJobId
                ? "Saving..."
                : "Save changes"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Reprocess progress after a paired-cameras change (shared with
          the Settings page and the folder-run analysis panel) */}
      <ApplySettingsModal
        open={!!reprocessJobId}
        message={reprocessProgress.message}
        progress={reprocessProgress.progress}
        fallbackMessage="Regrouping..."
      />

      {/* Datetime offset modal: only render when folder is valid + scanned */}
      {scanResult && deployment.folder_path && (
        <DatetimeOffsetModal
          open={offsetModalOpen}
          onOpenChange={setOffsetModalOpen}
          sampleFiles={scanResult.sample_files}
          folderPath={deployment.folder_path}
          currentOffsetSeconds={datetimeOffset}
          onApply={setDatetimeOffset}
          // The queue entry's mtime opt-in is gone after analysis, so
          // derive it: when the scan finds no metadata dates, any dates
          // in the DB can only have come from file modification times,
          // and those are the reference the offset must be computed
          // against.
          useFileMtimeFallback={scanResult.missing_datetime}
        />
      )}

      {/* Add-site modal triggered by the [+] button in SiteSelector.
          Selects the newly-created site automatically. */}
      <AddSiteModal
        projectId={projectId}
        open={showAddSiteModal}
        onOpenChange={setShowAddSiteModal}
        onSiteCreated={(newSiteId) => setSiteId(newSiteId)}
      />
    </>
  );
}
