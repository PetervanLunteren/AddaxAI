/**
 * What a Labels grid says when it has nothing to show.
 *
 * There are several reasons a grid can be empty and they mean different
 * things, so one message for all of them was misleading: "no labels
 * match your filters" reads as failure when you have in fact just
 * finished, and it says nothing about the work waiting in the other tab.
 *
 * Rather than a message per case, this composes two slots from two
 * numbers, so both tabs read the same way and the rule stays one rule:
 *
 *   headline   why this grid is empty: nothing of this kind exists, you
 *              finished what the filters selected, or nothing matched
 *   remainder  what is left, nearest first: work in this tab hidden
 *              behind the filters, else work in the other tab, and if
 *              neither then the job is genuinely done and it says so
 *
 * "Nearest first" and one line only: the count chip on the tab toggle
 * already keeps the other tab visible while you work, so this does not
 * have to carry both and can stay calm.
 */

import { Check, Layers } from "lucide-react";

import { Button } from "../ui/button";

interface GridEmptyStateProps {
  /** Unverified labels in this tab, within the site and date scope.
   *  Zero means this tab is finished. */
  thisTabLeft: number;
  /** The same for the other tab. */
  otherTabLeft: number;
  /** Every label in scope, both tabs. Guards the "all done" message on
   *  a project that has nothing in it yet, where nothing was done. */
  totalLabels: number;
  /** True when the current view matched items and every one of them is
   *  verified. False when nothing matched at all. The difference is the
   *  whole point of the headline. */
  viewFinished: boolean;
  /** How many the view held, shown only when `viewFinished`. */
  viewCount: number;
  /** True when this tab holds nothing of its kind at all. */
  tabHasNothing: boolean;
  /** "labels" / "empty files", for this tab and the other one. */
  noun: string;
  otherNoun: string;
  otherTabName: string;
  onClearFilters: () => void;
  onSwitchTab?: () => void;
}

export function GridEmptyState({
  thisTabLeft,
  otherTabLeft,
  totalLabels,
  viewFinished,
  viewCount,
  tabHasNothing,
  noun,
  otherNoun,
  otherTabName,
  onClearFilters,
  onSwitchTab,
}: GridEmptyStateProps) {
  const allDone =
    totalLabels > 0 && thisTabLeft === 0 && otherTabLeft === 0;

  if (allDone) {
    return (
      <Wrapper icon="check">
        <p className="text-sm font-medium">All done. Every label is verified.</p>
      </Wrapper>
    );
  }

  // The fallback has to be true whether the filters matched nothing or
  // matched only things already verified. On a fresh load the server
  // never returns the verified ones, so "nothing matches your filters"
  // was flatly misleading: the filter had matched four labels and the
  // user had finished them. `viewFinished` is only knowable while the
  // sort result is still in memory, so it sharpens the wording when that
  // is free rather than earning a request of its own.
  const headline = tabHasNothing
    ? `There are no ${noun} in this view.`
    : viewFinished
      ? `You've verified all ${viewCount.toLocaleString()} ${noun} in this view.`
      : "Nothing left to verify in this view.";

  return (
    <Wrapper icon={viewFinished ? "check" : "layers"}>
      <p className="text-sm font-medium">{headline}</p>

      {thisTabLeft > 0 ? (
        <>
          <p className="mt-1 text-sm text-muted-foreground">
            {thisTabLeft.toLocaleString()} more {noun} are outside your
            filters.
          </p>
          <Button
            variant="outline"
            size="sm"
            className="mt-4"
            onClick={onClearFilters}
          >
            Clear filters
          </Button>
        </>
      ) : otherTabLeft > 0 && onSwitchTab ? (
        <>
          <p className="mt-1 text-sm text-muted-foreground">
            There are also {otherTabLeft.toLocaleString()} {otherNoun} in{" "}
            {otherTabName}, if you want a look.
          </p>
          <Button
            variant="outline"
            size="sm"
            className="mt-4"
            onClick={onSwitchTab}
          >
            Go to {otherTabName}
          </Button>
        </>
      ) : null}
    </Wrapper>
  );
}

function Wrapper({
  icon,
  children,
}: {
  icon: "check" | "layers";
  children: React.ReactNode;
}) {
  const Icon = icon === "check" ? Check : Layers;
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground">
      <Icon className="mb-3 h-8 w-8 text-muted-foreground/60" />
      {children}
    </div>
  );
}
