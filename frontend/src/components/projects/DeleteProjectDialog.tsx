/**
 * Delete Project Dialog with GitHub-style confirmation.
 */

import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";
import { projectsApi, type ProjectResponse } from "../../api/projects";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Input } from "../ui/input";
import { Label } from "../ui/label";

interface DeleteProjectDialogProps {
  project: ProjectResponse | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function DeleteProjectDialog({
  project,
  open,
  onOpenChange,
}: DeleteProjectDialogProps) {
  const queryClient = useQueryClient();
  const [confirmText, setConfirmText] = useState("");

  // Reset confirmation text when dialog opens/closes
  useEffect(() => {
    if (!open) {
      setConfirmText("");
    }
  }, [open]);

  const deleteMutation = useMutation({
    mutationFn: () => projectsApi.delete(project!.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      onOpenChange(false);
    },
    onError: (error: Error) => {
      console.error("Failed to delete project:", error);
    },
  });

  const handleDelete = () => {
    if (confirmText === project?.name) {
      deleteMutation.mutate();
    }
  };

  if (!project) return null;

  const isConfirmValid = confirmText === project.name;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-destructive" />
            Delete project
          </DialogTitle>
          <DialogDescription>
            This action cannot be undone. This will permanently delete the project and all associated data.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
            <p className="text-sm font-medium text-destructive mb-2">
              Warning - This will permanently delete
            </p>
            <ul className="text-sm text-destructive/90 list-disc list-inside space-y-1">
              <li>All sites and deployment information</li>
              <li>All AI predictions and classifications</li>
              <li>All detection results and analyses</li>
              <li>All metadata and database records</li>
            </ul>
          </div>

          <div className="rounded-lg border bg-blue-50 border-blue-200 p-4">
            <p className="text-sm text-blue-900">
              <strong>Note:</strong> Your original images and videos will not be deleted. AddaxAI never touches the source files.
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="confirm-delete">
              Please type <span className="font-mono font-semibold bg-muted px-1.5 py-0.5 rounded">{project.name}</span> to confirm
            </Label>
            <Input
              id="confirm-delete"
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              placeholder={project.name}
              autoComplete="off"
            />
          </div>
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
            variant="destructive"
            onClick={handleDelete}
            disabled={!isConfirmValid || deleteMutation.isPending}
          >
            {deleteMutation.isPending ? "Deleting..." : "Delete project"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
