/**
 * Help sheet for the two verification steps, Labels and Counts. The same (?)
 * trigger opens it on both, so it takes a `step` prop and shows the guidance
 * for that step, with a short pointer to the other one.
 *
 * Both steps are optional: the AI already produced labels and counts, this is
 * where a person checks them. Every toolbar icon has a tooltip, the Labels
 * keyboard popover and the event modal's own shortcut list cover the keys, so
 * this sheet explains the workflow rather than being an exhaustive reference.
 */

import { Sparkles } from "lucide-react";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "../ui/sheet";
import { Separator } from "../ui/separator";

/** Inline replica of the toolbar's suggestions pill, so the help text can
 *  point at what it actually looks like instead of describing it. */
function SuggestionsPillExample() {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-muted/30 px-2 py-0.5 align-middle text-xs">
      <Sparkles className="h-3.5 w-3.5 text-muted-foreground" />
      <span className="text-muted-foreground">
        <span className="font-medium text-foreground">12</span> suggestions
      </span>
      <span className="rounded bg-primary px-1.5 py-0.5 text-[10px] font-medium text-primary-foreground">
        Review
      </span>
    </span>
  );
}

/** Inline replica of one row of the count panel, so the help text can show
 *  what the species + count control looks like instead of describing it. */
function CountRowExample() {
  return (
    <span className="inline-flex items-center gap-2 rounded border bg-white px-2 py-1 align-middle text-xs">
      <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm bg-[#0f6064]" />
      <span>Red deer</span>
      <span className="ml-1 inline-flex items-center gap-1 text-muted-foreground">
        <span className="inline-flex h-4 w-4 items-center justify-center rounded border">
          −
        </span>
        <span className="w-5 rounded border text-center tabular-nums text-foreground">
          3
        </span>
        <span className="inline-flex h-4 w-4 items-center justify-center rounded border">
          +
        </span>
      </span>
    </span>
  );
}

interface VerifyHelpSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Which step's guidance to show. */
  step: "labels" | "counts";
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">{children}</code>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <h3 className="text-sm font-semibold mb-2">{title}</h3>
      <Separator className="mb-3" />
      <div className="space-y-2 text-sm text-muted-foreground">{children}</div>
    </section>
  );
}

export function VerifyHelpSheet({
  open,
  onOpenChange,
  step,
}: VerifyHelpSheetProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="left" className="w-full sm:max-w-2xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle>
            {step === "labels"
              ? "Check the AI's labels"
              : "Check the AI's counts"}
          </SheetTitle>
          <SheetDescription>
            {step === "labels"
              ? "The AI labelled every detection. Fix the ones it got wrong."
              : "The AI counted the species in each event. Confirm or adjust them."}
          </SheetDescription>
        </SheetHeader>

        <div className="space-y-5 mt-4">
          {step === "labels" ? <LabelsGuide /> : <CountsGuide />}
        </div>
      </SheetContent>
    </Sheet>
  );
}

function LabelsGuide() {
  return (
    <>
      <Section title="What this step is">
        <p>
          The AI already gave every detection a label. Here you check those
          labels and fix the wrong ones. It is optional: the AI is good but
          makes mistakes, so a pass over the doubtful ones makes your data more
          reliable.
        </p>
      </Section>

      <Section title="How it works">
        <p>
          Each tile is one animal. Double-click a tile to open it and see the
          file context. By default the grid sorts by visual similarity, so
          look-alike crops sit next to each other and wrong labels stand out.
        </p>
        <p>
          Click a tile to select it, <Kbd>Shift</Kbd>-click another to select
          the range between, then verify or relabel the whole selection at
          once.
        </p>
        <p>
          Keys: <Kbd>Enter</Kbd> verify, <Kbd>R</Kbd> relabel, <Kbd>X</Kbd> mark
          as a false detection, <Kbd>U</Kbd> mark as unknown when you can't
          identify it, <Kbd>M</Kbd> set the whole selection to its most common
          label, <Kbd>1</Kbd>-<Kbd>5</Kbd> your saved labels. The keyboard icon
          in the toolbar lists them all and lets you set the{" "}
          <Kbd>1</Kbd>-<Kbd>5</Kbd> slots.
        </p>
      </Section>

      <Section title="Sorting">
        <p>
          The sort menu reorders the grid. <Kbd>Similarity</Kbd> (the default)
          puts look-alike crops next to each other. <Kbd>By event</Kbd> groups
          the crops from each visit together; each event header has a{" "}
          <Kbd>Select</Kbd> link, or press <Kbd>E</Kbd> to grab the next event
          that still needs checking, then verify or relabel it in one go.{" "}
          <Kbd>Lowest confidence first</Kbd> leads with the labels the AI was
          least sure about.
        </p>
      </Section>

      <Section title="Suggestions">
        <p>
          When the toolbar shows <SuggestionsPillExample />, the AI found crops
          whose look-alikes mostly carry a different label (for example "canis"
          surrounded by "domestic dog"). Click <Kbd>Review</Kbd> to group them
          into cohorts: <Kbd>Accept</Kbd> relabels and verifies a whole cohort
          in one click, <Kbd>Dismiss</Kbd> hides the suggestion without changing
          anything.
        </p>
      </Section>

    </>
  );
}

function CountsGuide() {
  return (
    <>
      <Section title="What this step is">
        <p>
          The AI already counted each species. It is good at counting the
          animals in a single frame, but it cannot tell whether the animals
          elsewhere in the event are the same individuals or new ones. So it can
          undercount when separate animals keep walking in and out of frame.
          That is where you come in: the AI gives a starting count, and you
          raise it when you can tell the individuals apart. It is optional, but
          confirming the events that matter makes your data more reliable.
        </p>
      </Section>

      <Section title="How it works">
        <p>
          Each card is one event: the photos and videos from a single visit,
          defined by the <Kbd>Independence interval</Kbd> setting. Click a card
          to open it. It opens on the frame with the most animals visible, so
          you can check the species and count in one look.
        </p>
        <p>
          The strip below the image is the rest of the event. For a video it
          shows frames spread across the clip, so you can see the animal come
          and go. Click any tile to inspect it, press <Kbd>play</Kbd> to watch
          the video, or press <Kbd>Space</Kbd> to loop through the event
          automatically.
        </p>
      </Section>

      <Section title="Confirming counts">
        <p>
          The panel on the right lists each species the AI found and its count,
          for example <CountRowExample />. The count is the MaxN: the most
          individuals of that species visible in a single frame, taken across
          the whole event. Adjust a count if it is wrong, or add a species the
          AI missed.
        </p>
        <p>
          Keys: <Kbd>↑</Kbd> <Kbd>↓</Kbd> pick a species, <Kbd>0</Kbd>-
          <Kbd>9</Kbd> set its count (type fast for numbers like 12),{" "}
          <Kbd>+</Kbd> <Kbd>−</Kbd> nudge by one, <Kbd>A</Kbd> add a species.
          Press <Kbd>Enter</Kbd> to confirm the event and jump to the next
          unconfirmed one.
        </p>
      </Section>

    </>
  );
}
