/**
 * Restore database from backup.
 *
 * The user picks a `.db` file via Electron's file picker. The backend
 * validates it (PRAGMA integrity_check) and writes the
 * .restore-on-next-launch marker. We then ask Electron to relaunch;
 * the new process swaps the file in before init_db runs and a safety
 * snapshot of the current DB lands in the ring buffer.
 *
 * Type-to-confirm `RESTORE` mirrors the Reset dialog's UX. The DB
 * swap is destructive in spirit even though the safety snapshot makes
 * it recoverable. Reset still uses quitApp because its intent is
 * "wipe and walk away"; Restore relaunches because the user is
 * waiting for their old data to come back.
 */

import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { AlertTriangle, FileSearch } from "lucide-react";
import { backupApi } from "../../api/backup";
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
import { Input } from "../ui/input";
import { Label } from "../ui/label";

interface RestoreBackupDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const FILE_FILTERS = [{ name: "SQLite database", extensions: ["db"] }];

export function RestoreBackupDialog({
  open,
  onOpenChange,
}: RestoreBackupDialogProps) {
  const [sourcePath, setSourcePath] = useState<string | null>(null);
  const [confirmText, setConfirmText] = useState("");
  const [pickError, setPickError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setSourcePath(null);
      setConfirmText("");
      setPickError(null);
    }
  }, [open]);

  const pick = async () => {
    setPickError(null);
    if (!window.electronAPI?.openFile) {
      setPickError("File picker is only available in the desktop app.");
      return;
    }
    const path = await window.electronAPI.openFile({
      title: "Select database backup",
      filters: FILE_FILTERS,
    });
    if (path) setSourcePath(path);
  };

  const restore = useMutation({
    mutationFn: () => {
      if (!sourcePath) throw new Error("No backup file selected.");
      return backupApi.restore(sourcePath);
    },
    onSuccess: async () => {
      // Backend has written the marker. Relaunch the desktop app so
      // the lifespan in the new process consumes the marker and swaps
      // the DB in before init_db. In dev (browser) we just close the
      // dialog; the user has to restart the backend manually.
      if (window.electronAPI?.relaunchApp) {
        await window.electronAPI.relaunchApp();
      } else {
        onOpenChange(false);
      }
    },
  });

  const isConfirmValid = confirmText === "RESTORE" && sourcePath !== null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-destructive" />
            Restore database from backup
          </DialogTitle>
          <DialogDescription>
            Replaces the current project database with the contents of a
            backup file. AddaxAI restarts to finish the swap. A safety
            snapshot of the current database is saved to the backups
            folder before the swap, so this is reversible.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <Callout variant="info">
            <strong>Your original images and videos are never touched.</strong>{" "}
            Only the SQLite database file is swapped.
          </Callout>

          <div className="space-y-2">
            <Label>Backup file</Label>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={pick}
                disabled={restore.isPending}
              >
                <FileSearch className="h-4 w-4 mr-2" />
                Choose backup file…
              </Button>
              {sourcePath && (
                <span className="text-xs text-muted-foreground truncate max-w-[18rem]">
                  {sourcePath}
                </span>
              )}
            </div>
            {pickError && (
              <p className="text-sm text-destructive">{pickError}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="confirm-restore">
              Please type{" "}
              <span className="font-mono font-semibold bg-muted px-1.5 py-0.5 rounded">
                RESTORE
              </span>{" "}
              to confirm
            </Label>
            <Input
              id="confirm-restore"
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              placeholder="RESTORE"
              autoComplete="off"
              disabled={!sourcePath || restore.isPending}
            />
          </div>

          {restore.isError && (
            <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
              {(restore.error as Error)?.message ??
                "Restore failed. The backup file may be corrupt."}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={restore.isPending}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            onClick={() => restore.mutate()}
            disabled={!isConfirmValid || restore.isPending}
          >
            {restore.isPending ? "Restoring…" : "Restore and restart"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
