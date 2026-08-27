/**
 * Refetch everything that describes "which labels exist in this project"
 * after a relabel, a verify, or a custom label edit.
 *
 * The label tree and the species colour map both derive from the set of
 * present species, so they go stale together. One helper so no call
 * site can refresh one and forget the other.
 */

import type { QueryClient } from "@tanstack/react-query";

export function invalidateLabelQueries(queryClient: QueryClient): void {
  void queryClient.invalidateQueries({ queryKey: ["label-tree"] });
  void queryClient.invalidateQueries({ queryKey: ["label-colors"] });
}
