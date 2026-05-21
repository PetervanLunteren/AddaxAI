/**
 * Help sheet with comprehensive guide for the Media tab and modal.
 * Slides in from the left, covering the toolbar area. Mirrors the
 * structure of HelpSheet.tsx (the Events guide) so users can transfer
 * what they learn between tabs.
 */

import {
  Pencil,
  ChevronsUpDown,
  SquarePlus,
  Play,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Scale,
  Sun,
  Contrast,
  Heart,
  Flag,
  Download,
  FolderOpen,
} from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "../ui/sheet";
import { Separator } from "../ui/separator";

interface MediaHelpSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function ToolRow({
  icon,
  children,
}: {
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="flex gap-2.5 text-sm text-muted-foreground">
      <div className="shrink-0 mt-0.5 text-foreground">{icon}</div>
      <p>{children}</p>
    </div>
  );
}

export function MediaHelpSheet({ open, onOpenChange }: MediaHelpSheetProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="left" className="w-full sm:max-w-2xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Media verification guide</SheetTitle>
          <SheetDescription>
            How to review and verify observations file by file
          </SheetDescription>
        </SheetHeader>

        <div className="space-y-5 mt-4">
          {/* Overview */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Overview</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Each tile in this tab is one file: a still photo or a
                video. The AI detects and labels animals, people, and
                vehicles automatically. Browse here and verify by
                confirming or correcting the observations.
              </p>
              <p>
                This tab verifies at the file level. A file counts as
                verified when you
                press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> on
                it in the modal: the file and all its observations are
                marked verified. The progress bar in the toolbar shows
                how many files are verified.
              </p>
              <p>
                Want to work by event (a group of files captured close
                together in time)? Use the Events tab. Want one
                observation at a time across the whole project? Use the
                Observations tab.
              </p>
            </div>
          </section>

          {/* Workflow */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Verification workflow</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Click a tile to open it. The label pill on each box shows
                what the AI thinks. If the labels look right,
                press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> to
                verify and jump to the next unverified file. You only
                need to correct what the AI got wrong.
              </p>
              <p>
                Add missing observations
                with <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">A</code>,
                or draw one yourself
                with <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">D</code>.
                Remove false positives
                with <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Del</code>.
                Mark a file as empty
                with <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">E</code>:
                this wipes the observations, marks the file verified, and
                jumps to the next.
              </p>
              <p>
                Quick workflow: review the labels.
                Use <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">↓</code> and <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Tab</code> to
                fix mistakes.
                Press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Del</code> to
                remove false positives.
                Hit <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> to
                verify and move on.
              </p>
              <p>
                Files that came from a video have
                a <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Video</code> badge
                on the tile. In the modal,
                press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">P</code> to
                play the source clip. The box overlays follow playback.
                Press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">P</code> again
                to return to the still frame.
              </p>
              <p>
                Click <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Show keyboard shortcuts</code> at
                the bottom of the sidebar to see every shortcut. You can
                assign labels to
                keys <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">1</code> to <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">5</code> to
                relabel every observation in a file with one key.
              </p>
            </div>
          </section>

          {/* Toolbar */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Toolbar</h3>
            <Separator className="mb-3" />
            <div className="space-y-3">
              <ToolRow icon={<Pencil className="h-4 w-4" />}>
                Draw a new observation box manually.
              </ToolRow>
              <ToolRow icon={<ChevronsUpDown className="h-4 w-4" />}>
                Choose the label for drawn boxes. Appears in the toolbar
                when draw mode is active.
              </ToolRow>
              <ToolRow icon={<SquarePlus className="h-4 w-4" />}>
                Promote the highest-confidence below-threshold AI box into
                a confirmed observation.
              </ToolRow>
              <ToolRow icon={<Play className="h-4 w-4" />}>
                Toggle between the still frame and video playback (only
                for files that came from a video).
              </ToolRow>
              <ToolRow icon={<ZoomIn className="h-4 w-4" />}>
                Zoom in.
              </ToolRow>
              <ToolRow icon={<ZoomOut className="h-4 w-4" />}>
                Zoom out.
              </ToolRow>
              <ToolRow icon={<RotateCcw className="h-4 w-4" />}>
                Reset zoom to fit.
              </ToolRow>
              <ToolRow icon={<Scale className="h-4 w-4" />}>
                Adjust the view threshold for this file. Local override;
                it does not change the project setting. Lowering it shows
                lower-confidence observations in the modal.
              </ToolRow>
              <ToolRow icon={<Sun className="h-4 w-4" />}>
                Adjust brightness for dark images.
              </ToolRow>
              <ToolRow icon={<Contrast className="h-4 w-4" />}>
                Adjust contrast for washed-out images.
              </ToolRow>
              <ToolRow icon={<Heart className="h-4 w-4" />}>
                Like.
              </ToolRow>
              <ToolRow icon={<Flag className="h-4 w-4" />}>
                Flag for follow-up review.
              </ToolRow>
              <ToolRow icon={<Download className="h-4 w-4" />}>
                Download the current view with annotations.
              </ToolRow>
              <ToolRow icon={<FolderOpen className="h-4 w-4" />}>
                Open the source file in your file explorer.
              </ToolRow>
            </div>
          </section>
        </div>
      </SheetContent>
    </Sheet>
  );
}
