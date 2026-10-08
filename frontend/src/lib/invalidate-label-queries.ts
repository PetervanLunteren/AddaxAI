/**
 * Refetch everything that describes "which labels exist in this project"
 * after a relabel, a verify, or a custom label edit.
 *
 * The label tree, the species colour map and the dashboard statistics
 * all derive from the set of present species, so they go stale together.
 * One helper so no call site can refresh one and forget the others.
 */

import type { QueryClient } from "@tanstack/react-query";

export function invalidateLabelQueries(queryClient: QueryClient): void {
  void queryClient.invalidateQueries({ queryKey: ["label-tree"] });
  void queryClient.invalidateQueries({ queryKey: ["label-colors"] });
  void queryClient.invalidateQueries({ queryKey: ["statistics"] });
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
