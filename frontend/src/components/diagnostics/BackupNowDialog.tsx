/**
 * Manual database backup dialog.
 *
 * Two paths:
 * - Save to the AddaxAI backups folder (~/AddaxAI/backups/). Force-
 *   writes a daily-format file even on the same UTC day, so users can
 *   trigger an extra snapshot before doing something risky.
 * - Save to a folder of the user's choosing. Uses the existing
 *   selectFolder IPC; the file is named with the same daily timestamp
 *   pattern as ring-buffer entries.
 *
 * On success we show a toast with a "Reveal" action that opens the
 * file in the OS file manager.
 */

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Database, FolderOpen, HardDrive } from "lucide-react";
import { toast } from "sonner";
import { backupApi } from "../../api/backup";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

interface BackupNowDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function revealAfterSave(savedPath: string): void {
  if (window.electronAPI?.showItemInFolder) {
    void window.electronAPI.showItemInFolder(savedPath);
  }
}

export function BackupNowDialog({ open, onOpenChange }: BackupNowDialogProps) {
  const [busy, setBusy] = useState<"ring" | "folder" | null>(null);

  const ringMut = useMutation({
    mutationFn: () => backupApi.snapshotToRingBuffer(),
    onMutate: () => setBusy("ring"),
    onSettled: () => setBusy(null),
    onSuccess: (data) => {
      toast.success("Backup saved", {
        description: data.path,
        action: window.electronAPI?.showItemInFolder
          ? { label: "Reveal", onClick: () => revealAfterSave(data.path) }
          : undefined,
      });
      onOpenChange(false);
    },
    onError: (err: Error) => toast.error(`Backup failed: ${err.message}`),
  });

  const folderMut = useMutation({
    mutationFn: async () => {
      if (!window.electronAPI?.selectFolder) {
        throw new Error("Folder picker only available in the desktop app.");
      }
      const dir = await window.electronAPI.selectFolder();
      if (!dir) return null; // user cancelled
      return await backupApi.snapshotToFolder(dir);
    },
    onMutate: () => setBusy("folder"),
    onSettled: () => setBusy(null),
    onSuccess: (data) => {
      if (!data) return; // user cancelled the folder picker
      toast.success("Backup saved", {
        description: data.path,
        action: window.electronAPI?.showItemInFolder
          ? { label: "Reveal", onClick: () => revealAfterSave(data.path) }
          : undefined,
      });
      onOpenChange(false);
    },
    onError: (err: Error) => toast.error(`Backup failed: ${err.message}`),
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Back up database
          </DialogTitle>
          <DialogDescription>
            Take a snapshot of the project database. Backups are
            consolidated <code className="text-xs">.db</code> files; your
            original images and videos are not touched.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <button
            type="button"
            onClick={() => ringMut.mutate()}
            disabled={busy !== null}
            className="w-full flex items-start gap-3 rounded-lg border p-4 text-left transition-colors hover:bg-accent disabled:opacity-60"
          >
            <HardDrive className="h-5 w-5 mt-0.5" />
            <div className="flex-1">
              <div className="font-medium">Save to backups folder</div>
              <p className="text-sm text-muted-foreground mt-0.5">
                Adds a snapshot to{" "}
                <code className="text-xs">~/AddaxAI/backups/</code>. The
                folder keeps the five most recent daily snapshots
                automatically.
              </p>
            </div>
            {busy === "ring" && (
              <span className="text-xs text-muted-foreground">Working…</span>
            )}
          </button>

          <button
            type="button"
            onClick={() => folderMut.mutate()}
            disabled={busy !== null || !window.electronAPI?.selectFolder}
            className="w-full flex items-start gap-3 rounded-lg border p-4 text-left transition-colors hover:bg-accent disabled:opacity-60"
          >
            <FolderOpen className="h-5 w-5 mt-0.5" />
            <div className="flex-1">
              <div className="font-medium">Save to chosen folder…</div>
              <p className="text-sm text-muted-foreground mt-0.5">
                Pick any folder. Useful if you want to keep a copy on an
                external drive or alongside your image archive.
              </p>
              {!window.electronAPI?.selectFolder && (
                <p className="text-xs text-muted-foreground mt-1 italic">
                  Only available in the desktop app.
                </p>
              )}
            </div>
            {busy === "folder" && (
              <span className="text-xs text-muted-foreground">Working…</span>
            )}
          </button>
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={busy !== null}
          >
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
