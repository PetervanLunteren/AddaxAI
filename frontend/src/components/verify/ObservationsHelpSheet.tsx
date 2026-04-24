/**
 * Help sheet for the Observations verify tab. Slides in from the left,
 * matching the pattern of HelpSheet.tsx.
 */

import { Check } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "../ui/sheet";
import { Separator } from "../ui/separator";
import { Badge } from "../ui/badge";

interface ObservationsHelpSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const mod = navigator.platform.includes("Mac") ? "Cmd" : "Ctrl";

export function ObservationsHelpSheet({ open, onOpenChange }: ObservationsHelpSheetProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="left" className="w-full sm:max-w-2xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Observations verification guide</SheetTitle>
          <SheetDescription>
            How to review and verify observations using visual similarity
          </SheetDescription>
        </SheetHeader>

        <div className="space-y-5 mt-4">
          {/* Overview */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Overview</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <div>
                The AI embeds each detection crop into a vector based on visual
                similarity, then sorts detections so similar-looking ones appear
                next to each other. This makes it easy to spot mislabels and
                verify in bulk. Each card shows the label (e.g.{" "}
                <Badge variant="default" className="text-[9px] px-1 py-0 leading-tight capitalize">
                  zebra
                </Badge>
                ) below the crop.
              </div>
              <p>
                This tab verifies at the detection level. Each
                individual detection crop can be verified independently, which
                is useful for bulk-verifying labels across your entire
                dataset rather than going file by file. It cannot verify at the
                file level because it only shows individual crops, not the full
                image, so you cannot know whether the file is missing detections
                that should be there.
              </p>
            </div>
          </section>

          {/* Sort mode */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Sort mode</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Switch between{" "}
                <span className="inline-flex rounded-md bg-muted p-0.5 align-middle">
                  <span className="px-1.5 py-0 text-[9px] leading-tight font-medium rounded bg-background text-foreground shadow-sm">Sort</span>
                  <span className="px-1.5 py-0 text-[9px] leading-tight font-medium text-muted-foreground">Search</span>
                </span>
                {" "}modes using the toggle in the toolbar. The default view is
                Sort, where detections are arranged using a nearest-neighbor
                walk so visually similar crops sit side by side. Sorting updates
                automatically when filters change.
              </p>
              <p>
                "Noise first" reverses the order so outliers (unusual
                detections that don't look like their neighbors) appear at the top.
              </p>
              <p>
                Use the filter in the toolbar to switch between{" "}
                <span className="inline-flex rounded-md bg-muted p-0.5 align-middle">
                  <span className="px-1.5 py-0 text-[9px] leading-tight font-medium text-muted-foreground">All</span>
                  <span className="px-1.5 py-0 text-[9px] leading-tight font-medium rounded bg-background text-foreground shadow-sm">Unverified</span>
                  <span className="px-1.5 py-0 text-[9px] leading-tight font-medium text-muted-foreground">Suspicious</span>
                </span>
                {" "}detections. The counts update as you verify.
              </p>
            </div>
          </section>

          {/* Search mode */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Search mode</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Right-click a detection and select "Find similar", or use the
                button in the detail window, to search for
                visually similar detections across the dataset. Results are
                ranked by similarity score. Use the threshold slider to filter
                results.
              </p>
            </div>
          </section>

          {/* Suspicious labels */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Suspicious labels</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <div>
                The system checks each detection's 10 nearest embedding
                neighbors. If fewer than 7 share the same label, the detection
                is flagged as suspicious with a red label:{" "}
                <Badge
                  variant="outline"
                  className="text-[9px] px-1 py-0 leading-tight capitalize border-transparent bg-[#882000] text-white hover:bg-[#882000]"
                >
                  impala
                </Badge>
              </div>
              <div>
                If the majority of neighbors have a different label, that label
                is shown as a suggestion on the card:{" "}
                <span className="inline-flex items-center gap-0.5 align-middle">
                  <Badge
                    variant="outline"
                    className="text-[9px] px-1 py-0 leading-tight capitalize border-transparent bg-[#882000] text-white hover:bg-[#882000]"
                  >
                    impala
                  </Badge>
                  <span className="text-[9px] text-muted-foreground">→</span>
                  <Badge variant="secondary" className="text-[9px] px-1 py-0 leading-tight capitalize">
                    gazelle
                  </Badge>
                </span>
              </div>
              <p>
                Use the{" "}
                <span className="inline-flex items-center gap-0.5 align-middle px-1 py-0 rounded-md bg-muted text-[9px] leading-tight font-medium text-foreground">
                  <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: "#882000" }} />
                  Suspicious
                </span>
                {" "}filter to focus on these. Once you verify a detection, it is
                no longer flagged as suspicious, even if its neighbors disagree.
              </p>
            </div>
          </section>

          {/* Detection detail */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Detection detail</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Double-click a card to open the detail window. For unverified
                detections with neighbor data, you'll see a label agreement
                section with a progress bar showing how many of the 10
                neighbors agree:
              </p>
              <div className="flex items-center gap-2 max-w-xs">
                <div className="relative h-2 w-full overflow-hidden rounded-full flex">
                  <div style={{ width: "60%", backgroundColor: "#0f6064" }} className="h-full" />
                  <div style={{ width: "40%", backgroundColor: "#882000" }} className="h-full" />
                </div>
                <span className="text-[11px] whitespace-nowrap">6/10 agree</span>
              </div>
              <p>
                If there's a suggestion, you can accept it with a button.
                Below the bar, neighbor thumbnails appear with colored borders:{" "}
                <span className="inline-block h-3 w-3 rounded border-2 align-middle" style={{ borderColor: "#0f6064", background: "#f4f4f5" }} /> agrees,{" "}
                <span className="inline-block h-3 w-3 rounded border-2 align-middle" style={{ borderColor: "#882000", background: "#f4f4f5" }} /> disagrees.
              </p>
            </div>
          </section>

          {/* Selection and bulk actions */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Selection and bulk actions</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Click a card to select it.{" "}
                <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Shift+Click</code> to
                select a range from the last clicked card.{" "}
                <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">{mod}+Click</code> to
                toggle individual cards without clearing the selection.
                A floating bar appears with: Verify, Relabel,
                Find similar, Deselect, and Accept suggestions
                (when suggestions are available).
              </p>
              <p>
                Double-click any card to open the detail window.
              </p>
              <p>
                Verified detections show a{" "}
                <span className="inline-flex items-center align-middle bg-primary rounded-full p-px">
                  <Check className="h-2.5 w-2.5 text-primary-foreground" />
                </span>
                {" "}badge on the card.
              </p>
              <p>
                Keyboard shortcuts:
              </p>
              <ul className="list-disc list-inside space-y-1 ml-1">
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> verify selected</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">X</code> mark as false detection</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">R</code> relabel selected</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">F</code> find similar</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">A</code> accept suggestions for selected</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">{mod}+A</code> select all</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Escape</code> deselect</li>
              </ul>
            </div>
          </section>

          {/* Settings */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Settings</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                The gear icon opens settings:{" "}
                <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Noise first</code> (reverse sort
                order), <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Label dividers</code> (group headers between labels), and tile
                size (<code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">S</code> / <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">M</code> / <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">L</code>).
              </p>
            </div>
          </section>
        </div>
      </SheetContent>
    </Sheet>
  );
}
