/**
 * The Labels page's progress, shared by both tabs and the dashboard.
 *
 * Counted in labels, where a label is one call a person has to make: a
 * detection above the threshold, which is one card in Detections, or a file
 * with nothing above it, one "nothing here" call. The two never overlap,
 * so 100% means every one has been looked at.
 *
 * The tab chips count in their own units: boxes left for Detections,
 * files not signed off for Files. Those two overlap, which is the point
 * of having two views of the work.
 *
 * One hook so the two tabs cannot show different numbers for the same
 * work. Switching tabs must not move the bar.
 *
 * Follows the site and date filters, which narrow what the user is
 * working on. It deliberately ignores the verified filter: a bar whose
 * denominator moves with the thing it measures can only ever read 0% or
 * 100%.
 */

import { useQuery } from "@tanstack/react-query";

import { labelsApi } from "../../api/labels";
import {
  labelsProgressValue,
  type LabelsProgressValue,
} from "../../lib/verified-progress";
import type { LabelsFilterState } from "./labels-filters";

export function useLabelsProgress(
  projectId: string,
  filters: LabelsFilterState,
): LabelsProgressValue {
  const params = {
    site_ids: filters.site_ids,
    date_from: filters.date_from,
    date_to: filters.date_to,
    // The confidence slider too, so the chips on the tab switch cannot
    // contradict the grid beside them. At 1% the empties chip read "220"
    // above a grid header saying "68 files".
    min_confidence: filters.min_confidence,
  };
  const { data } = useQuery({
    queryKey: ["labels-progress", projectId, params],
    queryFn: () => labelsApi.progress(projectId, params),
  });
  return labelsProgressValue(data);
}
