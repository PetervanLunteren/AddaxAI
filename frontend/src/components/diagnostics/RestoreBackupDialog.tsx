/**
 * Restore database from backup.
 *
 * The user picks a `.db` file via Electron's file picker. The backend
 * validates it (PRAGMA integrity_check) and writes the
 * .restore-on-next-launch marker. We then ask Electron to relaunch;
 * the new process swaps the file in before init_db runs and a safety
 * snapshot of the current DB lands in the ring buffer.
 *
 * Type-to-confirm `RESTORE`, gated on a backup file being chosen first.
 */

import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { FileSearch } from "lucide-react";
import { backupApi } from "../../api/backup";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import { Label } from "../ui/label";
import { TypeToConfirmDialog } from "../ui/type-to-confirm-dialog";

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
  const [pickError, setPickError] = useState<string | null>(null);

  // The typed word resets inside TypeToConfirmDialog; reset the rest here.
  useEffect(() => {
    if (!open) {
      setSourcePath(null);
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

  return (
    <TypeToConfirmDialog
      open={open}
      onOpenChange={onOpenChange}
      title="Restore database from backup"
      description="Replaces the current project database with the contents of a backup file. AddaxAI restarts to finish the swap. A safety snapshot of the current database is saved to the backups folder before the swap, so this is reversible."
      confirmWord="RESTORE"
      confirmLabel="Restore and restart"
      pendingLabel="Restoring…"
      onConfirm={() => restore.mutate()}
      isPending={restore.isPending}
      disabled={sourcePath === null}
      variant="destructive"
    >
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
        {pickError && <p className="text-sm text-destructive">{pickError}</p>}
      </div>

      {restore.isError && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
          {(restore.error as Error)?.message ??
            "Restore failed. The backup file may be corrupt."}
        </div>
      )}
    </TypeToConfirmDialog>
  );
}
