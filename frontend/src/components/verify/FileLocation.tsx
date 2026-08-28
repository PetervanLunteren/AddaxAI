/**
 * The file's name in the Image card, with the way to it: the full path
 * on hover, and a small folder button that opens the file in the OS
 * file explorer. A user with many subfolders could not tell where to
 * look for the raw photo outside the app (Grant Hiebert, 2026-08-25).
 * The button only renders inside Electron, where opening the explorer
 * is possible; the event view offers the same action in its menu
 * through the same hook.
 *
 * Shared by the Detections and Empties large views so the two cards
 * cannot drift.
 */

import type { ReactNode } from "react";
import { FolderOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRevealInFolder } from "@/lib/file-reveal";
import { basename } from "@/lib/path-utils";

interface FileLocationProps {
  filePath: string;
  /** Rendered after the filename, e.g. the frame number of a video. */
  suffix?: ReactNode;
}

export function FileLocation({ filePath, suffix }: FileLocationProps) {
  const revealInFolder = useRevealInFolder();
  return (
    <div className="flex items-center gap-1">
      <span className="min-w-0 truncate" title={filePath}>
        {basename(filePath)}
        {suffix}
      </span>
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
  );
}
