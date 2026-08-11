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
 *
 * An optional note is slugged into the backup's filename (both paths)
 * and shown again on the card in the restore picker.
 */

import { useEffect, useState } from "react";
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
import { FieldHeader } from "../ui/field-header";
import { Input } from "../ui/input";
import { Label } from "../ui/label";

/**
 * Live mirror of the backend's `_slugify_note` (`app/db/backup.py`), so
 * the field shows exactly what ends up in the filename. Same accepted
 * mirror pattern as confidence.py / confidence.ts. One deliberate
 * difference: no edge-hyphen stripping here, or typing the space in
 * "camera trap" would be eaten mid-word; the backend strips edges on
 * save.
 */
function normalizeNote(raw: string): string {
  return raw
    .normalize("NFKD")
    .replace(/[^\x00-\x7F]/g, "") // ascii-fold, like encode("ascii", "ignore")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .slice(0, 40);
}

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
  const [note, setNote] = useState("");

  // Reset on close so last session's note can't silently attach to the
  // next backup.
  useEffect(() => {
    if (!open) setNote("");
  }, [open]);

  const ringMut = useMutation({
    mutationFn: () => backupApi.snapshotToRingBuffer(note),
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
      return await backupApi.snapshotToFolder(dir, note);
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

        <div className="space-y-2">
          <FieldHeader
            label={<Label htmlFor="backup-note">Note</Label>}
            caption="Optional. Becomes part of the file name so you can recognise this backup when restoring."
          />
          <Input
            id="backup-note"
            value={note}
            onChange={(e) => {
              // Normalizing rewrites the typed character (uppercase,
              // space), and a controlled input then drops the caret to
              // the end when editing mid-string. Map the caret through
              // the transform (normalize the prefix before it) and put
              // both value and caret back on the DOM synchronously,
              // which also covers React bailing out of a re-render
              // when the normalized value is unchanged.
              const el = e.target;
              const pos = el.selectionStart ?? el.value.length;
              const next = normalizeNote(el.value);
              const caret = Math.min(
                normalizeNote(el.value.slice(0, pos)).length,
                next.length,
              );
              setNote(next);
              el.value = next;
              el.setSelectionRange(caret, caret);
            }}
            placeholder="e.g. before the big run"
            disabled={busy !== null}
          />
        </div>

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
                Adds a snapshot to the{" "}
                <code className="text-xs">backups</code> folder inside the
                app's data folder. The folder keeps the five most recent
                daily snapshots automatically.
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
