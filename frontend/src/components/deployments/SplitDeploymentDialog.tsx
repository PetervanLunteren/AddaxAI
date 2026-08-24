/**
 * Split a deployment along its folder hierarchy.
 *
 * Users pick a descent depth; the backend returns the subfolders at that
 * depth along with per-branch image/video counts. On OK we call POST
 * /api/deployments/{id}/split which creates one child deployment per
 * non-empty subfolder, slices the .addaxai artifacts, and removes the
 * original row.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { toast } from "sonner";

import { deploymentsApi } from "../../api/deployments";
import { ApiError } from "../../lib/api-client";
import { invalidateProjectData } from "../../lib/invalidate-project";
import { Button } from "../ui/button";
import { StartTruncatedPath } from "../ui/start-truncated-path";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

export interface SplitDeploymentTarget {
  id: string;
  folder_path: string | null;
}

interface SplitDeploymentDialogProps {
  deployment: SplitDeploymentTarget | null;
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SplitDeploymentDialog({
  deployment,
  projectId,
  open,
  onOpenChange,
}: SplitDeploymentDialogProps) {
  const queryClient = useQueryClient();
  const [depth, setDepth] = useState(1);

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      setDepth(1);
    }
    onOpenChange(nextOpen);
  };

  const deploymentId = deployment?.id ?? null;
  const folderPath = deployment?.folder_path ?? null;

  const previewQuery = useQuery({
    queryKey: ["split-preview", deploymentId, depth],
    queryFn: () => deploymentsApi.getSplitPreview(deploymentId!, depth),
    enabled: open && !!deploymentId,
    staleTime: 5_000,
  });

  const splitMutation = useMutation({
    mutationFn: () => deploymentsApi.split(deploymentId!, depth),
    onSuccess: (data) => {
      // Split creates new deployment rows and re-partitions files,
      // detections, and events. Blanket invalidate so every page picks
      // up the new structure.
      invalidateProjectData(queryClient, projectId);
      queryClient.invalidateQueries({
        queryKey: ["sites-with-stats", projectId],
      });
      queryClient.invalidateQueries({ queryKey: ["split-preview"] });
      toast.success(data.message);
      handleOpenChange(false);
    },
    onError: (error: unknown) => {
      const msg =
        error instanceof ApiError
          ? String(error.message)
          : error instanceof Error
            ? error.message
            : "Split failed";
      toast.error(msg);
    },
  });

  const preview = previewQuery.data;
  const blocked = preview?.blocked_reason ?? null;
  const canDecrease = depth > 1 && (preview?.can_decrease ?? false);
  const canIncrease = preview?.can_increase ?? false;
  const okDisabled =
    !preview ||
    blocked != null ||
    preview.targets.length <= 1 ||
    splitMutation.isPending;

  if (!deployment) return null;

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Split deployment</DialogTitle>
          <DialogDescription>
            A deployment is one camera at one location recording continuously,
            basically what you collect on one SD card pickup. If your folder
            mixes several of those, split it along the folder hierarchy so
            each child is a clean single deployment. The original images and
            videos stay where they are on disk.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="text-sm">
            <p className="text-muted-foreground">Original folder</p>
            <StartTruncatedPath
              className="font-mono"
              path={folderPath}
              emptyLabel="No folder path set"
            />
          </div>

          <div className="flex items-center gap-3">
            <p className="text-sm text-muted-foreground">Split depth</p>
            <div className="flex items-center gap-1">
              <Button
                type="button"
                size="icon"
                variant="outline"
                className="h-8 w-8"
                disabled={!canDecrease}
                onClick={() => setDepth((d) => Math.max(1, d - 1))}
                aria-label="Decrease depth"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="w-8 text-center tabular-nums text-sm">
                {depth}
              </span>
              <Button
                type="button"
                size="icon"
                variant="outline"
                className="h-8 w-8"
                disabled={!canIncrease}
                onClick={() => setDepth((d) => d + 1)}
                aria-label="Increase depth"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {blocked && (
            <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
              {blocked}
            </div>
          )}

          {previewQuery.isLoading && (
            <p className="text-sm text-muted-foreground">Loading preview...</p>
          )}

          {preview && !blocked && preview.targets.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No subfolders with media at this depth.
            </p>
          )}

          {preview &&
            !blocked &&
            preview.targets.length === 1 &&
            !canIncrease && (
              <p className="text-sm text-muted-foreground">
                Only one folder holds files here, so there is nothing to
                split. Splitting needs at least two subfolders that each hold
                files.
              </p>
            )}

          {preview && !blocked && preview.targets.length > 0 && (
            <div>
              <p className="text-sm mb-2">
                Will create {preview.targets.length}{" "}
                {preview.targets.length === 1 ? "deployment" : "deployments"}:
              </p>
              <ul className="max-h-[50vh] overflow-y-auto rounded-lg border divide-y text-sm">
                {preview.targets.map((t) => (
                  <li
                    key={t.folder_path}
                    className="flex items-baseline gap-4 px-3 py-2"
                  >
                    <StartTruncatedPath
                      className="font-mono flex-1"
                      path={t.folder_path}
                    />
                    <span className="tabular-nums text-muted-foreground shrink-0">
                      {t.image_count} img, {t.video_count} vid
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => handleOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            type="button"
            onClick={() => splitMutation.mutate()}
            disabled={okDisabled}
          >
            {splitMutation.isPending ? "Splitting..." : "OK"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
