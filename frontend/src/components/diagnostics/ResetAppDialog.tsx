/**
 * Reset application dialog with type-to-confirm.
 *
 * Wipes user data (logs, envs, models, bin, thumbnails, crash-dumps,
 * sentinels) and optionally the SQLite DB. After the backend confirms
 * the wipe, asks Electron to quit so the next launch runs the setup
 * wizard and rebuilds everything from scratch.
 */

import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { diagnosticsApi } from "../../api/diagnostics";
import { Callout } from "../ui/callout";
import { Checkbox } from "../ui/checkbox";
import { Label } from "../ui/label";
import { TypeToConfirmDialog } from "../ui/type-to-confirm-dialog";

interface ResetAppDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ResetAppDialog({ open, onOpenChange }: ResetAppDialogProps) {
  const [wipeDatabase, setWipeDatabase] = useState(false);

  // The typed word resets inside TypeToConfirmDialog; reset the checkbox too.
  useEffect(() => {
    if (!open) setWipeDatabase(false);
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

  return (
    <TypeToConfirmDialog
      open={open}
      onOpenChange={onOpenChange}
      title="Reset application"
      description="Wipes installed environments, models, logs, and other AddaxAI files. The app closes after the wipe; relaunch to start fresh."
      confirmWord="RESET"
      confirmLabel="Reset and quit"
      pendingLabel="Resetting..."
      onConfirm={() => reset.mutate()}
      isPending={reset.isPending}
      error={reset.error?.message ?? null}
      variant="destructive"
    >
      <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
        <p className="text-sm font-medium text-destructive mb-2">
          Warning - This will permanently delete
        </p>
        <ul className="text-sm text-destructive/90 list-disc list-outside pl-5 space-y-1">
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
    </TypeToConfirmDialog>
  );
}
