/**
 * The Labels page's progress, shared by both tabs and the dashboard.
 *
 * Counted in labels, where a label is one call a person has to make: a
 * detection above the threshold, which is one card in Detections, or a file
 * with nothing above it, which is one card in Empties carrying the
 * label "nothing here". The two never overlap, so the total is exactly
 * the number of cards across both tabs and 100% means every one has
 * been looked at.
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
import type { LabelsFilterState } from "./labels-filters";

export interface LabelsProgressValue {
  pct: number;
  verified: number;
  total: number;
  /** Native tooltip for the pill, so the raw counts are reachable. */
  title: string;
  /** Labels left in each tab, for the toggle chips and the nudge shown
   *  when a grid runs out. Filter-scoped, unlike `pct`: the site and
   *  date filters carry across the tab switch, so a count that ignored
   *  them would promise more than the user is about to see. */
  cropsLeft: number;
  emptiesLeft: number;
}

export function useLabelsProgress(
  projectId: string,
  filters: LabelsFilterState,
): LabelsProgressValue {
  const params = {
    site_ids: filters.site_ids,
    date_from: filters.date_from,
    date_to: filters.date_to,
    // The confidence slider too, so the chips on the tab switch cannot
    // contradict the grid beside them. At 1% the chip read "Empties 220"
    // above a grid header saying "68 files".
    min_confidence: filters.min_confidence,
  };
  const { data } = useQuery({
    queryKey: ["labels-progress", projectId, params],
    queryFn: () => labelsApi.progress(projectId, params),
  });

  const total = data?.total_labels ?? 0;
  const verified = data?.verified_labels ?? 0;
  return {
    pct: total ? (verified / total) * 100 : 0,
    verified,
    total,
    cropsLeft: (data?.crop_labels ?? 0) - (data?.crop_labels_verified ?? 0),
    emptiesLeft: (data?.empty_labels ?? 0) - (data?.empty_labels_verified ?? 0),
    title:
      `${verified.toLocaleString()} of ${total.toLocaleString()} labels ` +
      `verified: every box above your detection threshold, plus one for ` +
      `each file the AI found nothing in`,
  };
}
