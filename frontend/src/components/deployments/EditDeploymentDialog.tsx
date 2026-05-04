/**
 * Edit deployment dialog.
 *
 * Metadata-only: site, datetime offset, notes, tags. Folder linking
 * is handled by the per-group RelinkGroupBanner at the top of the
 * deployments page.
 */

import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Pencil } from "lucide-react";
import { deploymentsApi } from "../../api/deployments";
import type { DeploymentResponse } from "../../api/types";
import { useFolderScan } from "../../hooks/useFolderScan";
import { formatOffset } from "../../lib/utils";
import { Button } from "../ui/button";
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
  const [notes, setNotes] = useState<string>(deployment.notes ?? "");
  const [tags, setTags] = useState<Record<string, string>>(deployment.tags ?? {});
  const [offsetModalOpen, setOffsetModalOpen] = useState(false);
  const [showAddSiteModal, setShowAddSiteModal] = useState(false);

  // Reset all state when the modal opens with a different deployment
  useEffect(() => {
    if (open) {
      setSiteId(deployment.site_id);
      setDatetimeOffset(deployment.datetime_offset_seconds ?? 0);
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

  const updateMutation = useMutation({
    mutationFn: () =>
      deploymentsApi.update(deployment.id, {
        site_id: siteId,
        datetime_offset_seconds: datetimeOffset || null,
        notes: notes.trim() || null,
        tags,
      }),
    onSuccess: () => {
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
      onOpenChange(false);
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
                allowEmpty
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

            {/* Notes */}
            <div className="space-y-2">
              <Label htmlFor="edit-deployment-notes">Notes</Label>
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
              disabled={updateMutation.isPending}
            >
              {updateMutation.isPending ? "Saving..." : "Save changes"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Datetime offset modal: only render when folder is valid + scanned */}
      {scanResult && deployment.folder_path && (
        <DatetimeOffsetModal
          open={offsetModalOpen}
          onOpenChange={setOffsetModalOpen}
          sampleFiles={scanResult.sample_files}
          folderPath={deployment.folder_path}
          currentOffsetSeconds={datetimeOffset}
          onApply={setDatetimeOffset}
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
