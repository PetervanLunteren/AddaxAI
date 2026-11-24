/**
 * Refetch everything that describes "which labels exist in this project"
 * after a relabel or a custom label edit.
 *
 * The filter options (labels, names, colours, the filter tree), the label
 * counts and the dashboard statistics all derive from the set of present
 * species, so they go stale together. One helper so no call site can
 * refresh one and forget the others.
 *
 * `cancelRefetch: false`: a caller that also refreshes the event queries
 * has already started the filter options, after the same write. Starting
 * them again would make the backend read the project's boxes twice.
 */

import type { Query, QueryClient } from "@tanstack/react-query";

export function invalidateLabelQueries(queryClient: QueryClient): void {
  for (const key of ["event-filter-options", "label-counts", "statistics"]) {
    void queryClient.invalidateQueries({ queryKey: [key] }, { cancelRefetch: false });
  }
}

/**
 * Every event-family query: the event list and count (["events", ...]),
 * one event (["event", id]), the filtered count
 * (["event-count-filtered", ...]) and the filter options
 * (["event-filter-options", ...]). A plain ["events"] prefix
 * misses the singular keys, which once left the Counts page stale after a
 * relabel.
 */
export function isEventFamilyQuery(query: Query): boolean {
  const key = query.queryKey[0];
  return typeof key === "string" && key.startsWith("event");
}

/**
 * What an action on detections changed: only their verified flag
 * ("verify", which includes unverify), or their label too ("change").
 */
export type DetectionActionKind = "verify" | "change";

/** How long the counts wait after the last verify before they refetch. */
export const VERIFY_SETTLE_MS = 1000;

let verifySettleTimer: ReturnType<typeof setTimeout> | undefined;

/**
 * Refresh what a verify or unverify changes, once the clicks stop.
 *
 * A verify moves no box into or out of a species, so the filter options
 * (labels, colours, tree) and the dashboard statistics are not refetched.
 * (One rare case does change them: unverifying a box below the threshold
 * hides it, and with it a species that had no other box. Those lists
 * catch up on their next refetch.) A verify does move the progress bar,
 * the cohorts, and the label counts when the Verified filter is on (its default is "Unverified"). On a large
 * project each of those is a pass over every box, so they refetch once,
 * `VERIFY_SETTLE_MS` after the last verify, instead of once per click.
 * The event-family queries only go stale: the Counts page refetches them
 * when it is next opened, and the Labels page has nothing to redraw.
 */
export function invalidateAfterVerify(
  queryClient: QueryClient,
  projectId: string,
  verifiedFilterOn: boolean,
): void {
  void queryClient.invalidateQueries({
    predicate: isEventFamilyQuery,
    refetchType: "none",
  });
  clearTimeout(verifySettleTimer);
  verifySettleTimer = setTimeout(() => {
    void queryClient.invalidateQueries({ queryKey: ["labels-progress", projectId] });
    void queryClient.invalidateQueries({ queryKey: ["cohorts", projectId] });
    if (verifiedFilterOn) {
      void queryClient.invalidateQueries({ queryKey: ["label-counts", projectId] });
    }
  }, VERIFY_SETTLE_MS);
}

/**
 * Refetch everything a custom-label create / edit / delete touches.
 *
 * Renaming or re-ranking a custom label rewrites the stored name, scientific
 * name and colour of every detection that carries it, so the surfaces that
 * draw those (the label picker, the filter tree, the colour map, and the
 * canvases in the detail modals, which read their boxes from the file /
 * event queries) all go stale together. One helper so an edit can never
 * refresh some of them and leave the canvas showing the old label.
 *
 * It does not reach the Labels grid, which renders from a sort mutation
 * rather than a query; that surface refreshes on its next action.
 */
export function invalidateAfterLabelEdit(
  queryClient: QueryClient,
  projectId: string,
): void {
  void queryClient.invalidateQueries({ queryKey: ["custom-labels", projectId] });
  void queryClient.invalidateQueries({
    queryKey: ["label-taxonomy-map", projectId],
  });
  invalidateLabelQueries(queryClient);
  // The detail-modal canvases read their boxes from these, so a rename or
  // rank change shows on the picture without a hard reload.
  void queryClient.invalidateQueries({ queryKey: ["file"] });
  void queryClient.invalidateQueries({ queryKey: ["event"] });
  void queryClient.invalidateQueries({ queryKey: ["events"] });
}
