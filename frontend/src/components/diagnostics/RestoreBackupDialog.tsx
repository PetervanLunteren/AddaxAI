/**
 * Restore database from backup.
 *
 * Lists the app's own backups (daily / before-update / before-restore /
 * manual) as dated cards, newest first, so the user picks a restore point
 * instead of hunting the filesystem. An escape hatch ("Restore from a
 * file…") still opens Electron's picker for backups saved to a custom
 * folder. The chosen source is validated by the backend, which writes the
 * .restore-on-next-launch marker; we then ask Electron to relaunch and the
 * next process swaps the file in (snapshotting the current DB first).
 *
 * Type-to-confirm `RESTORE`, gated on a backup being selected first.
 */

import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Check, FileSearch } from "lucide-react";
import { backupApi, type BackupEntry, type BackupKind } from "../../api/backup";
import { basename } from "../../lib/path-utils";
import { cn } from "../../lib/utils";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import { TypeToConfirmDialog } from "../ui/type-to-confirm-dialog";

interface RestoreBackupDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const FILE_FILTERS = [{ name: "SQLite database", extensions: ["db"] }];

interface Flavour {
  /** Badge text: the coarse human-vs-machine distinction. */
  label: string;
  /** Sub-line: the specific reason this snapshot exists. */
  sub: string;
  /** Badge colour classes. */
  badge: string;
}

const BADGE_AUTO = "text-[#0f6064] bg-[#0f6064]/10 border-[#0f6064]/30";
const BADGE_MANUAL = "text-muted-foreground bg-transparent border-border";

// The badge stays coarse (Automatic vs Manual); the sub-line carries the
// specific flavour so the user can read the detail without a rainbow of
// badges.
const FLAVOUR: Record<BackupKind, Flavour> = {
  daily: { label: "Automatic", sub: "automatic daily backup", badge: BADGE_AUTO },
  "pre-upgrade": {
    label: "Automatic",
    sub: "saved before an app update",
    badge: BADGE_AUTO,
  },
  "pre-restore": {
    label: "Automatic",
    sub: "saved before a restore",
    badge: BADGE_AUTO,
  },
  manual: { label: "Manual", sub: "you saved this", badge: BADGE_MANUAL },
};

function formatBytes(n: number): string {
  if (n < 1024) return `${Math.round(n)} B`;
  const kb = n / 1024;
  if (kb < 999.5) return `${Math.round(kb)} KB`;
  const mb = kb / 1024;
  if (mb < 999.95) return `${mb.toFixed(1)} MB`;
  return `${(mb / 1024).toFixed(1)} GB`;
}

/** Human "when" for an audit timestamp (backups are absolute server-local
 * moments, so they render in the viewer's own timezone). Returns a relative
 * headline ("Today, 14:03" / "Yesterday, …" / "3 days ago") plus the full
 * date as a secondary line. */
function formatWhen(iso: string): { rel: string; abs: string } {
  const d = new Date(iso);
  const time = d.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
  const abs = d.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });

  const startOfDay = (x: Date) =>
    new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round(
    (startOfDay(new Date()) - startOfDay(d)) / 86_400_000,
  );

  let rel: string;
  if (days <= 0) rel = `Today, ${time}`;
  else if (days === 1) rel = `Yesterday, ${time}`;
  else if (days < 7) rel = `${days} days ago`;
  else rel = abs;
  return { rel, abs };
}

export function RestoreBackupDialog({
  open,
  onOpenChange,
}: RestoreBackupDialogProps) {
  // A card selection (a backups-folder path) or a custom file the user
  // browsed to. Exactly one is active; the custom pick clears the card.
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [customPath, setCustomPath] = useState<string | null>(null);
  const [pickError, setPickError] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["backups"],
    queryFn: backupApi.list,
    enabled: open,
  });
  const entries: BackupEntry[] = data?.entries ?? [];

  useEffect(() => {
    if (!open) {
      setSelectedPath(null);
      setCustomPath(null);
      setPickError(null);
    }
  }, [open]);

  // Pre-select the newest backup once the list loads (nothing chosen yet).
  useEffect(() => {
    if (open && selectedPath === null && customPath === null && entries.length) {
      setSelectedPath(entries[0].path);
    }
  }, [open, entries, selectedPath, customPath]);

  const sourcePath = customPath ?? selectedPath;

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
    if (path) {
      setCustomPath(path);
      setSelectedPath(null);
    }
  };

  const restore = useMutation({
    mutationFn: () => {
      if (!sourcePath) throw new Error("No backup selected.");
      return backupApi.restore(sourcePath);
    },
    onSuccess: async () => {
      // Marker written; relaunch so the next process swaps the DB in
      // before init_db. In the browser (dev) we just close.
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
      description="Pick a point to roll back to. AddaxAI restarts to finish the swap. A snapshot of the current database is saved first, so this is reversible."
      confirmWord="RESTORE"
      confirmLabel="Restore and restart"
      pendingLabel="Restoring…"
      onConfirm={() => restore.mutate()}
      isPending={restore.isPending}
      disabled={sourcePath === null}
      variant="destructive"
    >
      <Callout variant="info">
        Your images and videos are never touched. Only the database is
        swapped.
      </Callout>

      <div className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Backups in AddaxAI
        </p>
        <div className="max-h-64 space-y-1.5 overflow-y-auto rounded-lg border bg-muted/30 p-2">
          {isLoading ? (
            <p className="py-6 text-center text-sm text-muted-foreground">
              Loading backups…
            </p>
          ) : entries.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">
              No automatic or manual backups yet. You can still restore from a
              file below.
            </p>
          ) : (
            entries.map((e) => {
              const f = FLAVOUR[e.kind];
              const when = formatWhen(e.created_utc);
              const isSel = !customPath && selectedPath === e.path;
              return (
                <button
                  key={e.path}
                  type="button"
                  onClick={() => {
                    setSelectedPath(e.path);
                    setCustomPath(null);
                  }}
                  className={cn(
                    "flex w-full items-center gap-3 rounded-lg border p-2.5 text-left transition-colors",
                    isSel
                      ? "border-[#0f6064] bg-[#0f6064]/10"
                      : "hover:border-border hover:bg-muted/50",
                  )}
                >
                  <span
                    className={cn(
                      "grid h-4 w-4 shrink-0 place-items-center rounded-full border-2",
                      isSel ? "border-[#0f6064]" : "border-muted-foreground/40",
                    )}
                  >
                    {isSel && (
                      <span className="h-2 w-2 rounded-full bg-[#0f6064]" />
                    )}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-semibold">
                      {when.rel}
                    </span>
                    <span className="block truncate text-xs text-muted-foreground tabular-nums">
                      {when.abs} · {f.sub}
                    </span>
                  </span>
                  <span className="flex shrink-0 flex-col items-end gap-1">
                    <span
                      className={cn(
                        "rounded-full border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide",
                        f.badge,
                      )}
                    >
                      {f.label}
                    </span>
                    <span className="text-[11px] text-muted-foreground tabular-nums">
                      {formatBytes(e.size_bytes)}
                    </span>
                  </span>
                </button>
              );
            })
          )}
        </div>

        {/* Escape hatch: a backup the user saved to their own folder. */}
        <button
          type="button"
          onClick={pick}
          disabled={restore.isPending}
          className="mt-1 flex w-full items-center gap-3 rounded-lg border border-dashed p-2.5 text-left transition-colors hover:border-[#0f6064]/40 hover:bg-[#0f6064]/5 disabled:opacity-50"
        >
          <span className="grid h-7 w-7 shrink-0 place-items-center rounded-md bg-[#0f6064]/10 text-[#0f6064]">
            <FileSearch className="h-4 w-4" />
          </span>
          <span className="min-w-0 flex-1">
            <span className="block text-sm font-semibold">
              Restore from a file…
            </span>
            <span className="block truncate text-xs text-muted-foreground">
              {customPath
                ? basename(customPath)
                : "Choose a .db backup from another folder or drive"}
            </span>
          </span>
          {customPath && (
            <Check className="h-4 w-4 shrink-0 text-[#0f6064]" />
          )}
        </button>
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
