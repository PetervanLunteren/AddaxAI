/**
 * Edit deployment dialog.
 *
 * Metadata-only: site, datetime offset, notes, tags. Folder linking
 * is handled by the per-group RelinkGroupBanner at the top of the
 * deployments page.
 */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil } from "lucide-react";
import { deploymentsApi } from "../../api/deployments";
import { sitesApi } from "../../api/sites";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Textarea } from "../ui/textarea";
import { TagsEditor } from "../ui/tags-editor";
import { DatetimeOffsetModal } from "../analyses/DatetimeOffsetModal";

interface EditDeploymentDialogProps {
  deployment: DeploymentResponse;
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function EditDeploymentDialog({
  deployment,
  projectId,
  open,
  onOpenChange,
}: EditDeploymentDialogProps) {
  const queryClient = useQueryClient();

  const [siteId, setSiteId] = useState<string>(deployment.site_id);
  const [datetimeOffset, setDatetimeOffset] = useState<number>(
    deployment.datetime_offset_seconds ?? 0
  );
  const [notes, setNotes] = useState<string>(deployment.notes ?? "");
  const [tags, setTags] = useState<Record<string, string>>(deployment.tags ?? {});
  const [offsetModalOpen, setOffsetModalOpen] = useState(false);

  // Reset all state when the modal opens with a different deployment
  useEffect(() => {
    if (open) {
      setSiteId(deployment.site_id);
      setDatetimeOffset(deployment.datetime_offset_seconds ?? 0);
      setNotes(deployment.notes ?? "");
      setTags(deployment.tags ?? {});
    }
  }, [open, deployment]);

  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId && open,
  });

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
            {/* Site picker */}
            <div className="space-y-2">
              <Label htmlFor="edit-deployment-site">Site</Label>
              <Select value={siteId} onValueChange={setSiteId}>
                <SelectTrigger id="edit-deployment-site">
                  <SelectValue placeholder="Select a site" />
                </SelectTrigger>
                <SelectContent>
                  {(sites ?? []).map((s) => (
                    <SelectItem key={s.id} value={s.id}>
                      {s.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
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
                placeholder="e.g., Camera angled slightly left to avoid sun glare"
              />
            </div>

            {/* Tags */}
            <TagsEditor value={tags} onChange={setTags} />

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
              disabled={updateMutation.isPending || !siteId}
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
    </>
  );
}
