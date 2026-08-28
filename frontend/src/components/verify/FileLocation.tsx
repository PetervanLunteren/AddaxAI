/**
 * Where a file lives, for the Image card of the large views.
 *
 * The filename alone did not tell a user with many subfolders where to
 * look when they want the raw photo outside the app (Grant Hiebert,
 * 2026-08-25), so the folder is shown under it on one line, cut at the
 * start so the nearest subfolders stay readable, full path on hover. The
 * "Show in folder" button hands the file to the OS file explorer and
 * only renders inside Electron, where that is possible; the event view
 * offers the same action in its menu through the same hook.
 *
 * Shared by the Detections and Empties large views so the two cards
 * cannot drift.
 */

import type { ReactNode } from "react";
import { FolderOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRevealInFolder } from "@/lib/file-reveal";
import { basename, dirname } from "@/lib/path-utils";
import { TruncateStart } from "@/components/ui/truncate-start";

interface FileLocationProps {
  filePath: string;
  /** Rendered after the filename, e.g. the frame number of a video. */
  suffix?: ReactNode;
}

export function FileLocation({ filePath, suffix }: FileLocationProps) {
  const revealInFolder = useRevealInFolder();
  const folder = dirname(filePath);
  return (
    <>
      <div className="truncate">
        {basename(filePath)}
        {suffix}
      </div>
      {folder && (
        <div className="flex items-center gap-1">
          <TruncateStart
            title={folder}
            className="flex-1 text-[11px] text-muted-foreground/70"
          >
            {folder}
          </TruncateStart>
          {window.electronAPI && (
            <Button
              variant="ghost"
              size="icon"
              className="h-5 w-5 shrink-0"
              title="Show in folder"
              onClick={() => void revealInFolder({ file_path: filePath })}
            >
              <FolderOpen className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      )}
    </>
  );
}
