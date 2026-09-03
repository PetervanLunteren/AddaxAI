/**
 * The left tool rail every large viewer shares: the Counts event modal,
 * the Detections modal and the Files viewer. One mental model: the rail
 * is how you LOOK at the picture (brightness/contrast, the box toggle,
 * flag, like, download and open in file explorer, each a plain icon),
 * identical everywhere; what you DECIDE lives in each modal's right
 * column and legitimately differs per surface. `children` holds a
 * host's own looking-tools (the Counts modal's event loop) so the rail
 * stays one component without pretending the hosts are identical.
 *
 * The state behind it lives in `viewer-tools.ts` (`useImageAdjust`,
 * `useFileTriage`), instantiated by the host: the host also consumes
 * them (the filter lands on its image, the F key on its keydown
 * handler), and a hook and the rail sharing one mutation is what keeps
 * the key and the button from drifting.
 */

import type { ReactNode } from "react";
import {
  Download,
  Eye,
  EyeOff,
  Flag,
  FolderOpen,
  Heart,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { useRevealInFolder } from "@/lib/file-reveal";
import { Button } from "../ui/button";
import { ViewControls } from "./ViewControls";
import type { FileTriage, RailFile } from "./viewer-tools";

interface ViewerToolRailProps {
  brightness: number;
  onBrightnessChange: (v: number) => void;
  contrast: number;
  onContrastChange: (v: number) => void;
  boxesHidden: boolean;
  onToggleBoxes: () => void;
  /** The focused file; the flag, like and explorer buttons wait for it. */
  file: RailFile | null | undefined;
  triage: FileTriage;
  /** "Download video" / "Download image" / "Download crop". */
  downloadLabel: string;
  onDownload: () => void;
  /** Extra rail tools between the box toggle and the flag (the Counts
   *  modal's event loop). */
  children?: ReactNode;
}

export function ViewerToolRail({
  brightness,
  onBrightnessChange,
  contrast,
  onContrastChange,
  boxesHidden,
  onToggleBoxes,
  file,
  triage,
  downloadLabel,
  onDownload,
  children,
}: ViewerToolRailProps) {
  const revealInFolder = useRevealInFolder();

  return (
    <>
      {/* Image: brightness / contrast (seeing a dark IR animal). */}
      <ViewControls
        brightness={brightness}
        onBrightnessChange={onBrightnessChange}
        contrast={contrast}
        onContrastChange={onContrastChange}
      />
      {/* Show / hide the AI boxes — toggle off to see the scene without
          the AI's boxes anchoring you. */}
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={onToggleBoxes}
        title={boxesHidden ? "Show AI boxes (B)" : "Hide AI boxes (B)"}
      >
        {boxesHidden ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
      </Button>
      {children}
      {/* Flag for review — the one triage action worth its own key. */}
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={() => file && triage.toggleFlag(file)}
        disabled={!file || triage.pending}
        title={file?.flagged ? "Remove flag" : "Flag for review (F)"}
      >
        <Flag
          className={cn("h-4 w-4", file?.flagged && "fill-[#71b7ba] text-[#71b7ba]")}
        />
      </Button>
      {/* Like — flag's sibling triage mark, so it gets the same direct
          click. There used to be a kebab menu here holding like,
          download and reveal, inherited from a Counts-era frequency
          call; a hidden feature is an unused feature, and the rail has
          room, so everything is a plain icon now. */}
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={() => file && triage.toggleLike(file)}
        disabled={!file || triage.pending}
        title={file?.favorited ? "Unlike" : "Like"}
      >
        <Heart
          className={cn(
            "h-4 w-4",
            file?.favorited && "fill-[#882000] text-[#882000]",
          )}
        />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={onDownload}
        title={downloadLabel}
      >
        <Download className="h-4 w-4" />
      </Button>
      {window.electronAPI && (
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          onClick={() => file && void revealInFolder({ file_path: file.file_path })}
          disabled={!file}
          title="Open in file explorer"
        >
          <FolderOpen className="h-4 w-4" />
        </Button>
      )}
    </>
  );
}
