/**
 * Delete custom label dialog.
 *
 * Custom labels are global (shared across all of AddaxAI), so deleting one
 * removes it everywhere and unlabels its detections in every project. The
 * confirm shows that blast radius by fetching the usage count first.
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { projectsApi } from "../../api/projects";
import type { CustomLabelResponse } from "../../api/types";
import { invalidateAfterLabelEdit } from "../../lib/invalidate-label-queries";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../ui/alert-dialog";

interface DeleteCustomLabelDialogProps {
  label: CustomLabelResponse | null;
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function DeleteCustomLabelDialog({
  label,
  projectId,
  open,
  onOpenChange,
}: DeleteCustomLabelDialogProps) {
  const queryClient = useQueryClient();

  const { data: usage } = useQuery({
    queryKey: ["custom-label-usage", label?.id],
    queryFn: () => projectsApi.getCustomLabelUsage(projectId, label!.id),
    enabled: open && !!label,
  });

  const deleteMutation = useMutation({
    mutationFn: () => projectsApi.deleteCustomLabel(projectId, label!.id),
    onSuccess: () => {
      invalidateAfterLabelEdit(queryClient, projectId);
      onOpenChange(false);
    },
    onError: (e: Error) =>
      toast.error("Could not delete the label", { description: e.message }),
  });

  if (!label) return null;

  const n = usage?.detection_count ?? 0;
  const m = usage?.project_count ?? 0;

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete "{label.name}"?</AlertDialogTitle>
          <AlertDialogDescription>
            {usage === undefined
              ? "Checking where this label is used..."
              : n === 0
                ? "This label is not used on any detection yet. It is shared across all of AddaxAI and will be removed everywhere."
                : `This label is used on ${n} detection${n === 1 ? "" : "s"} across ${m} project${m === 1 ? "" : "s"}. Those boxes go back to unverified and lose the label, ready to re-label. Custom labels are shared across all of AddaxAI, so this removes it everywhere.`}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={(e) => {
              e.preventDefault();
              deleteMutation.mutate();
            }}
            disabled={deleteMutation.isPending}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            {deleteMutation.isPending ? "Deleting..." : "Delete label"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
