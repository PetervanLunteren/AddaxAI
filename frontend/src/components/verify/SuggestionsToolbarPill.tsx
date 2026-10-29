/**
 * SuggestionsToolbarPill: the way into the suggestions review.
 *
 * Lives inside the Labels VerifyToolbar. Reads the cohort count (sum of
 * cohort sizes) from the cohorts endpoint and offers a Review button.
 * The review itself is a focused view with its own bar and way back
 * (`FocusedViewBar`), and the toolbar is hidden while it is open, so
 * this pill only ever offers the way in.
 *
 * Renders nothing when there are no suggestions to review, so the
 * toolbar stays compact in the common case where the dataset is clean
 * or the user has finished reviewing.
 */

import { useQuery } from "@tanstack/react-query";
import { Sparkles } from "lucide-react";

import { labelsApi } from "../../api/labels";
import { Button } from "../ui/button";

interface SuggestionsToolbarPillProps {
  projectId: string;
  onEnter: () => void;
}

export function SuggestionsToolbarPill({
  projectId,
  onEnter,
}: SuggestionsToolbarPillProps) {
  const { data } = useQuery({
    queryKey: ["cohorts", projectId],
    queryFn: () => labelsApi.cohorts(projectId),
    // The same expensive ML pass that powers the suggestions sort.
    // Cache for 5 minutes so the toolbar doesn't re-fetch as the user
    // moves around the page; bulk-relabel invalidates the key.
    staleTime: 5 * 60_000,
  });

  const total = (data?.cohorts ?? []).reduce((sum, c) => sum + c.count, 0);

  if (total === 0) return null;

  return (
    <div className="flex items-center gap-2 rounded-md border border-border bg-muted/30 px-2 py-1 text-xs">
      <Sparkles className="h-3.5 w-3.5 text-muted-foreground" />
      <span className="text-muted-foreground">
        <span className="font-medium text-foreground">{total}</span>{" "}
        suggestion{total === 1 ? "" : "s"}
      </span>
      <Button
        type="button"
        variant="default"
        size="sm"
        onClick={onEnter}
        className="h-6 px-2 text-xs"
      >
        Review
      </Button>
    </div>
  );
}
