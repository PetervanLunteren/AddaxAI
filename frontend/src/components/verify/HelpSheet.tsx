/**
 * Help sheet with comprehensive guide for the event detail modal.
 * Slides in from the left, covering the toolbar area.
 */

import {
  Pencil,
  SquarePlus,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Scale,
  Sun,
  Contrast,
  Heart,
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
          <SheetTitle>Verification guide</SheetTitle>
          <SheetDescription>
            How to review and verify detections
          </SheetDescription>
        </SheetHeader>

        <div className="space-y-5 mt-4">
          {/* Overview */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Overview</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                The AI detects and labels animals automatically. You can browse
                your events here, or verify them by confirming or correcting
                the detections in each image.
              </p>
            </div>
          </section>

          {/* Workflow */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Verification workflow</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Each event groups images captured close together in time. The
                event opens to its representative image: the clearest image with
                the most individuals. You don't need to verify every image,
                verifying the representative image is generally enough to get
                accurate statistics. If you wish to verify all images, you can
                by setting the navigation dropdown to <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">file</code>.
              </p>
              <p>
                You can add missing detections
                with <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">A</code> or
                draw them manually
                with <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">D</code>.
                If the labels look right, press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> to
                verify and advance to the next unverified event. You only need to
                correct what the AI got wrong.
              </p>
              <p>
                A quick workflow is to review the labels,
                use <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">↓</code> and <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Tab</code> to
                correct any mistakes,
                press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">X</code> to
                remove false positives, then
                hit <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> to
                verify and move to the next event.
                Use <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Shift + ← / →</code> to
                flip through files within an event without changing scope.
                To verify multiple files at once, <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Shift + Click</code> thumbnails
                in the filmstrip to select a range, then press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> to
                verify them all.
              </p>
              <p>
                Click <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Show keyboard shortcuts</code> at
                the bottom of the sidebar to see all shortcuts. You can assign
                species to
                keys <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">1</code> to <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">5</code>,
                so you can relabel every detection in an image with one key.
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
              <ToolRow icon={<SquarePlus className="h-4 w-4" />}>
                Promote the next below-threshold AI detection into a new
                detection.
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
                Adjust the detection confidence threshold. Lowering it reveals
                lower-confidence detections.
              </ToolRow>
              <ToolRow icon={<Sun className="h-4 w-4" />}>
                Adjust brightness for dark images.
              </ToolRow>
              <ToolRow icon={<Contrast className="h-4 w-4" />}>
                Adjust contrast for washed-out images.
              </ToolRow>
              <ToolRow icon={<Heart className="h-4 w-4" />}>
                Mark as favorite.
              </ToolRow>
              <ToolRow icon={<Download className="h-4 w-4" />}>
                Download the image with annotations.
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
