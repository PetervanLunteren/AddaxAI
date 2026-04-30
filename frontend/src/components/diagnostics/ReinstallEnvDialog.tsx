/**
 * Reinstall analysis environment dialog.
 *
 * Triggered from the app menu's "Re-run setup wizard" item. Deletes
 * env-addaxai-base on disk and redirects to /setup so the user can
 * trigger the install themselves. Lower friction than the full Reset
 * dialog: no type-to-confirm, just an explicit confirmation.
 */

import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { AlertTriangle } from "lucide-react";
import { api } from "../../lib/api-client";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

interface ReinstallEnvDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ReinstallEnvDialog({
  open,
  onOpenChange,
}: ReinstallEnvDialogProps) {
  const navigate = useNavigate();

  const reinstall = useMutation({
    mutationFn: () => api.post("/api/setup/reinstall-env", {}),
    onSuccess: () => {
      onOpenChange(false);
      navigate("/setup");
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-destructive" />
            Re-run setup wizard
          </DialogTitle>
          <DialogDescription>
            Remove the analysis environment and run the setup wizard
            again from scratch.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
            <p className="text-sm text-destructive/90">
              The current env-addaxai-base directory (~2 GB) will be
              deleted. After confirming you'll be taken to the setup
              wizard where you can start the reinstall when ready.
              Reinstalling typically takes 10 to 30 minutes and
              requires an internet connection.
            </p>
          </div>
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={reinstall.isPending}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            onClick={() => reinstall.mutate()}
            disabled={reinstall.isPending}
          >
            {reinstall.isPending ? "Removing..." : "Remove and re-run wizard"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
