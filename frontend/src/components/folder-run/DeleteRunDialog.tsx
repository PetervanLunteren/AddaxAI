/**
 * The confirm behind every "Delete this run": one window, one text, one
 * delete call, shared by the Recent runs list and the "Outputs saved"
 * window so the two cannot drift.
 *
 * Delete is irreversible: it takes the run's DB rows and its on-disk
 * `.addaxai` cache, so the confirm names the folder. The run then drops out
 * of Recent runs by itself, because that list is read from the same rows.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";

import { folderRunsApi } from "../../api/folder-runs";
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

interface DeleteRunDialogProps {
  /** The run to delete, or null when the window is closed. */
  run: { id: string; folder_path: string } | null;
  onClose: () => void;
  /** Called once the run is gone, before the window closes. */
  onDeleted?: () => void;
}

export function DeleteRunDialog({ run, onClose, onDeleted }: DeleteRunDialogProps) {
  const queryClient = useQueryClient();

  const remove = useMutation({
    mutationFn: (runId: string) => folderRunsApi.remove(runId),
    onSuccess: () => {
      // The lookup too: it is what tells the folder picker "already
      // analysed", and it would keep saying so for a run that is gone.
      queryClient.invalidateQueries({ queryKey: ["folder-runs"] });
      queryClient.invalidateQueries({ queryKey: ["folder-run-lookup"] });
      onDeleted?.();
      onClose();
    },
  });

  const close = () => {
    // A failed attempt must not greet the next run that opens this window.
    remove.reset();
    onClose();
  };

  return (
    // Escape cannot close it while the delete runs: the request still lands,
    // so closing would read as a cancel that then deletes anyway.
    <AlertDialog
      open={run !== null}
      onOpenChange={(o) => !o && !remove.isPending && close()}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete this run?</AlertDialogTitle>
          <AlertDialogDescription>
            Deletes the analysis of{" "}
            <span className="font-medium text-foreground">
              {run?.folder_path}
            </span>
            , including any labels you verified. Your images, videos and
            saved outputs are not touched. This cannot be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        {remove.error && (
          <p className="break-words text-sm text-destructive-ink">
            Could not delete the run: {remove.error.message}
          </p>
        )}
        <AlertDialogFooter>
          <AlertDialogCancel disabled={remove.isPending}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction
            disabled={remove.isPending}
            onClick={(e) => {
              // Keep the dialog up until the delete resolves.
              e.preventDefault();
              if (run) remove.mutate(run.id);
            }}
          >
            {remove.isPending ? "Deleting…" : "Delete run"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
