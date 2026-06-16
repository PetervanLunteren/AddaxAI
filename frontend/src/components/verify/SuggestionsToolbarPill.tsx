/**
 * SuggestionsToolbarPill — entry point for the suggestions review mode.
 *
 * Lives inside the Observations VerifyToolbar. Reads the cohort count
 * (sum of cohort sizes) from the cohorts endpoint and offers a Review
 * button. When the user is already in suggestions mode the pill flips
 * to an Exit button.
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
  /** True when the parent's current sort is "suggestions". */
  isActive: boolean;
  onEnter: () => void;
  onExit: () => void;
}

export function SuggestionsToolbarPill({
  projectId,
  isActive,
  onEnter,
  onExit,
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

  // Show the exit pill while the user is reviewing, even if the local
  // count has drained to zero (the optimistic patch removes detections
  // before the cache invalidates). Otherwise hide entirely when there
  // is nothing to surface.
  if (!isActive && total === 0) return null;

  if (isActive) {
    return (
      <div className="flex items-center gap-2 rounded-md border border-border bg-muted/30 px-2 py-1 text-xs">
        <Sparkles className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-muted-foreground">Reviewing suggestions</span>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onExit}
          className="h-6 px-2 text-xs"
        >
          Exit
        </Button>
      </div>
    );
  }

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
