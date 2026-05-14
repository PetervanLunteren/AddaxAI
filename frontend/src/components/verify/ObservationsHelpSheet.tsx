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
                The AI turns each detection crop into a vector and groups
                similar-looking crops next to each other. This makes
                mislabels easy to spot and easy to fix in bulk. Each card
                shows its label (e.g.{" "}
                <Badge variant="default" className="text-[9px] px-1 py-0 leading-tight capitalize">
                  zebra
                </Badge>
                ) below the crop.
              </div>
              <p>
                This tab verifies at the detection level. Each crop is its
                own unit. This is the fastest way to sweep mislabels across
                the whole project. It does not verify whole files: you only
                see the crop, so you cannot tell if the original image is
                missing other detections.
              </p>
              <p>
                Want to work by event or by file instead? Use the Events
                or Media tab.
              </p>
            </div>
          </section>

          {/* Sort */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Sort</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                The grid sorts by visual similarity by default. Use the
                sort selector in the toolbar to choose another order:
              </p>
              <ul className="list-disc list-inside space-y-1 ml-1">
                <li>Similarity (typical first): greedy nearest-neighbor walk; similar crops sit side by side.</li>
                <li>Similarity (outliers first): same walk in reverse; unusual crops show up at the top.</li>
                <li>Newest / Oldest: by file date.</li>
                <li>Lowest classifier confidence: hardest cases first.</li>
              </ul>
              <p>
                Sorting refreshes automatically when filters change.
              </p>
              <p>
                The Verified filter in the filter bar lets you scope to
                "All", "Unverified" (default), or "Suspicious". Counts
                update as you verify.
              </p>
            </div>
          </section>

          {/* Search mode */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Search mode</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Select a detection and press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">F</code>,
                or click <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Find similar</code> in
                the selection bar or in the detail window. The grid switches
                to search mode against that anchor crop. Results are ranked
                by similarity score; the threshold slider above the grid
                filters them.
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
                Pick "Suspicious" in the Verified filter to focus on these.
                Once you verify a detection, it is no longer flagged as
                suspicious, even if its neighbours disagree.
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
                Click a card to select it.
                Use <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Shift + Click</code> to
                select a range from the last clicked card.
                Use <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">{mod} + Click</code> to
                toggle a card on or off without clearing the rest. With
                one or more cards selected, a floating bar appears with:
                Verify, Mark false, Relabel, Find similar, Deselect, and
                Accept suggestions (when suggestions exist).
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
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> verify the selection</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">X</code> mark the selection as false detection</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">R</code> relabel the selection</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">F</code> find similar to the selection</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">A</code> accept neighbour suggestions for the selection</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">1</code> to <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">5</code> apply your shortcut labels to the selection</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">{mod} + A</code> select all visible</li>
                <li><code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Esc</code> deselect</li>
              </ul>
            </div>
          </section>

          {/* View options */}
          <section>
            <h3 className="text-sm font-semibold mb-2">View options</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                The view options icon opens:
              </p>
              <ul className="list-disc list-inside space-y-1 ml-1">
                <li>Label dividers: show a header between groups of the same label. Only available when sorting by similarity.</li>
                <li>Tile size: <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">S</code> / <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">M</code> / <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">L</code>.</li>
              </ul>
            </div>
          </section>
        </div>
      </SheetContent>
    </Sheet>
  );
}
