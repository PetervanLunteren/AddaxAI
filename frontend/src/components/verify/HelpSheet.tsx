/**
 * Help sheet with comprehensive guide for the event detail modal.
 * Slides in from the left, covering the toolbar area.
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

interface HelpSheetProps {
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

export function HelpSheet({ open, onOpenChange }: HelpSheetProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="left" className="w-full sm:max-w-2xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Event verification guide</SheetTitle>
          <SheetDescription>
            How to review and verify detections event by event
          </SheetDescription>
        </SheetHeader>

        <div className="space-y-5 mt-4">
          {/* Overview */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Overview</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                The AI detects and labels animals, people, and vehicles
                automatically. Browse your events here and verify them by
                confirming or correcting the detections.
              </p>
              <p>
                This tab verifies at the event level. An event counts as
                verified when all its MaxN frames are verified: the
                representative frames, one per species. Blank events have no
                MaxN frames, so marking any file verified counts as verifying
                the event. The progress bar in the toolbar shows how many
                events are verified.
              </p>
              <p>
                Verification still works file by file. When you press Enter
                on a file in the modal, that file and its detections are
                marked verified. The event status updates once all MaxN
                frames are covered.
              </p>
              <p>
                Want to work file by file instead of by event? Use the
                Media tab. Want one detection at a time across the whole
                project? Use the Observations tab.
              </p>
            </div>
          </section>

          {/* Workflow */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Verification workflow</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Each event groups images and videos captured close together
                in time. The event opens to its first unverified MaxN frame:
                the image where the peak count for each species was observed.
                MaxN frames carry
                a <code className="bg-primary text-white px-1 py-0.5 rounded-sm text-xs">MaxN</code> badge
                in the filmstrip, tinted with the species colour. You do not
                need to verify every file. Verifying the MaxN frames is
                usually enough for accurate statistics.
                Use <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Shift + ← / →</code> to
                walk every frame in the event when you want to.
              </p>
              <p>
                For video files, verification works at the frame level set by
                the project's "Video frame rate" setting. The MaxN frame is
                shown first; you can still view and verify the other
                analysed frames.
                Press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">P</code> to
                toggle between frame view and video playback.
              </p>
              <p>
                Add missing detections
                with <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">A</code>,
                or draw one yourself
                with <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">D</code>.
                If all labels look right,
                press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> to
                verify and move on. You only need to correct what the AI got
                wrong.
              </p>
              <p>
                Quick workflow: review the labels.
                Use <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">↓</code> and <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Tab</code> to
                fix mistakes.
                Press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Del</code> to
                remove false positives.
                Hit <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> to
                verify and move on.
                Use <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Shift + ← / →</code> to
                walk every frame in the event. To verify multiple files at
                once, <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Shift + Click</code> thumbnails
                in the filmstrip to select a range,
                or <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">{navigator.platform.includes("Mac") ? "Cmd" : "Ctrl"} + A</code> to
                select all. Then
                press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> to
                verify them all.
              </p>
              <p>
                Click <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Show keyboard shortcuts</code> at
                the bottom of the sidebar to see every shortcut. You can
                assign labels to
                keys <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">1</code> to <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">5</code> to
                relabel every detection in a file with one key.
              </p>
            </div>
          </section>

          {/* Toolbar */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Toolbar</h3>
            <Separator className="mb-3" />
            <div className="space-y-3">
              <ToolRow icon={<Pencil className="h-4 w-4" />}>
                Draw a new detection box manually.
              </ToolRow>
              <ToolRow icon={<ChevronsUpDown className="h-4 w-4" />}>
                Choose the label for drawn boxes. Appears in the toolbar when
                draw mode is active.
              </ToolRow>
              <ToolRow icon={<SquarePlus className="h-4 w-4" />}>
                Promote the highest-confidence below-threshold AI box into a
                confirmed detection.
              </ToolRow>
              <ToolRow icon={<Play className="h-4 w-4" />}>
                Toggle between frame view and video playback (for video files).
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
                Adjust the view threshold for this event. Local override; it
                does not change the project setting. Lowering it shows
                lower-confidence detections in the modal.
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
