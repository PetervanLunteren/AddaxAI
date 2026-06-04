/**
 * Single help sheet for the whole Verify page. Replaces the per-view
 * HelpSheet / MediaHelpSheet / ObservationsHelpSheet trio that lingered
 * from the old tabbed layout.
 *
 * Deliberately short. The detail modal has its own in-context keyboard
 * shortcut sheet, and every toolbar icon has a tooltip, so this sheet
 * does not try to be a full reference. It explains the page, the three
 * views, the cascade rule, and the Observations-grid shortcuts (which
 * are not discoverable from the UI).
 */

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "../ui/sheet";
import { Separator } from "../ui/separator";

interface VerifyHelpSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">{children}</code>
  );
}

export function VerifyHelpSheet({ open, onOpenChange }: VerifyHelpSheetProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="left" className="w-full sm:max-w-2xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Edit guide</SheetTitle>
          <SheetDescription>
            Review and fix the AI's predictions
          </SheetDescription>
        </SheetHeader>

        <div className="space-y-5 mt-4">
          {/* Overview */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Overview</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                The Edit page shows the AI's predictions so you can
                confirm or fix them. The same data appears in three views.
                Events groups files captured close together in time, best
                for reviewing one animal visit in context. Media shows one
                tile per file, good for going through files one by one.
                Observations shows one tile per detection crop, good for
                sweeping wrong labels across the whole project. Switch
                view from the "View as" dropdown in the filter bar.
              </p>
            </div>
          </section>

          {/* What verified means */}
          <section>
            <h3 className="text-sm font-semibold mb-2">What verified means</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Verified means a person has checked it. A small
                checkmark appears in the top-right corner of a verified
                file or detection.
              </p>
              <p>
                You can verify everything, or only one species, or just
                a random sample. It depends on what your project needs.
              </p>
              <p>
                For trustworthy statistics, you do not need to verify
                every detection. The most useful files to check are the
                ones where a species has its peak count in an event.
                These are called MaxN frames. The Events view walks you
                through them first when you open an event. An event with
                two species has two MaxN frames, one per species.
                Verifying those is enough for statistics. Verify more if
                you want, but it is not required.
              </p>
            </div>
          </section>

          {/* Events */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Events</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Each card is one event. Click it to open. The window
                lands on the first unverified MaxN frame.
                Press <Kbd>Enter</Kbd> to verify the file and jump to the
                next MaxN frame. When all MaxN frames in the event are
                verified, the event is done.
              </p>
            </div>
          </section>

          {/* Media */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Media</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Each card is one file (image or video). Click it to
                open. Press <Kbd>Enter</Kbd> to mark the file verified
                and jump to the next unverified file.
              </p>
            </div>
          </section>

          {/* Observations */}
          <section>
            <h3 className="text-sm font-semibold mb-2">Observations</h3>
            <Separator className="mb-3" />
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                Each card is one detection crop, with no file context.
                The grid sorts by visual similarity by default, so
                look-alike crops sit next to each other and wrong labels
                stand out.
              </p>
              <p>
                The Sparkles pill in the toolbar counts crops where most
                of the nearest neighbour crops carry a different label,
                either a more specific one (for example "canis"
                surrounded by "domestic dog") or a sibling species at
                the same rank (for example "grey fox" surrounded by
                "coyote"). Click Review and the grid groups them into
                cohorts. Each cohort has an Accept button that relabels
                and verifies the whole group in one click. Broader-rank
                suggestions are filtered out, so the suggestion never
                walks back up the taxonomy.
              </p>
              <p>
                If a cohort is wrong, for example a mix of different
                animals rather than one species, click Dismiss. That
                hides the suggestion without changing any labels and
                leaves the crops unverified, so you can relabel them in
                the normal sort whenever you like.
              </p>
            </div>
          </section>

        </div>
      </SheetContent>
    </Sheet>
  );
}
