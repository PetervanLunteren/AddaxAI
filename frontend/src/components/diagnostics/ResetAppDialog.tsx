/**
 * Reset application dialog with type-to-confirm.
 *
 * Wipes user data (logs, envs, models, bin, thumbnails, crash-dumps,
 * sentinels) and optionally the SQLite DB. After the backend confirms
 * the wipe, asks Electron to quit so the next launch runs the setup
 * wizard and rebuilds everything from scratch.
 *
 * Layout matches DeleteProjectDialog: AlertTriangle title, destructive
 * warning panel listing what will be removed, info panel for what's
 * preserved, type-to-confirm input.
 */

import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";
import { diagnosticsApi } from "../../api/diagnostics";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import { Checkbox } from "../ui/checkbox";
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

interface ResetAppDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ResetAppDialog({ open, onOpenChange }: ResetAppDialogProps) {
  const [confirmText, setConfirmText] = useState("");
  const [wipeDatabase, setWipeDatabase] = useState(false);

  // Reset state when the dialog closes so the next open is clean.
  useEffect(() => {
    if (!open) {
      setConfirmText("");
      setWipeDatabase(false);
    }
  }, [open]);

  const reset = useMutation({
    mutationFn: () => diagnosticsApi.resetApplication(wipeDatabase),
    onSuccess: async () => {
      // Tell Electron to close. On the next launch the setup wizard
      // runs again and the env / models reinstall from scratch.
      // In a browser (dev) we can't call quitApp; the user closes the
      // tab manually after seeing the confirmation toast in the UI.
      if (typeof window !== "undefined" && window.electronAPI?.quitApp) {
        await window.electronAPI.quitApp();
      } else {
        onOpenChange(false);
      }
    },
  });

  const isConfirmValid = confirmText === "RESET";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-destructive" />
            Reset application
          </DialogTitle>
          <DialogDescription>
            Wipes installed environments, models, logs, and other AddaxAI
            files. The app closes after the wipe; relaunch to start fresh.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
            <p className="text-sm font-medium text-destructive mb-2">
              Warning - This will permanently delete
            </p>
            <ul className="text-sm text-destructive/90 list-disc list-inside space-y-1">
              <li>Installed analysis environments</li>
              <li>Installed model weights</li>
              <li>All log files and crash dumps</li>
              <li>Cached thumbnails and the bundled micromamba binary</li>
            </ul>
          </div>

          <Callout variant="info">
            <strong>Your original images and videos are never touched.</strong>{" "}
            AddaxAI only writes to its own data directory; your files on disk
            are read-only as far as this app is concerned.
          </Callout>

          <div className="flex items-start gap-2">
            <div className="mt-0.5">
              <Checkbox
                checked={wipeDatabase}
                onCheckedChange={(c) => setWipeDatabase(c)}
              />
            </div>
            <div>
              <Label
                onClick={() => setWipeDatabase(!wipeDatabase)}
                className="cursor-pointer"
              >
                Also remove the project database (irreversible)
              </Label>
              <p className="text-xs text-muted-foreground mt-1">
                Deletes addaxai.db. All projects, sites, deployments, and
                detection records are permanently lost. Only check this if
                the database itself is corrupted or you want to start
                completely fresh.
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="confirm-reset">
              Please type{" "}
              <span className="font-mono font-semibold bg-muted px-1.5 py-0.5 rounded">
                RESET
              </span>{" "}
              to confirm
            </Label>
            <Input
              id="confirm-reset"
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              placeholder="RESET"
              autoComplete="off"
            />
          </div>
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={reset.isPending}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            onClick={() => reset.mutate()}
            disabled={!isConfirmValid || reset.isPending}
          >
            {reset.isPending ? "Resetting..." : "Reset and quit"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
