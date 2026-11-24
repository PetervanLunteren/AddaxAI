/**
 * What the Labels progress pill and tab chips show for one
 * `labels/progress` response, or for none yet. Kept apart from the hook
 * that fetches it so it can be tested without a browser.
 */

import type { LabelsProgress } from "../api/types";

export interface LabelsProgressValue {
  /** Null until the count is back, so the pill says "counting…" rather
   *  than "0% verified": on a big project the count takes 20 s or more. */
  pct: number | null;
  verified: number;
  total: number;
  /** Native tooltip for the pill, so the raw counts are reachable. */
  title: string;
  /** Work left in each tab, for the toggle chips and the nudge shown
   *  when a grid runs out. Filter-scoped, unlike `pct`: the site and
   *  date filters carry across the tab switch, so a count that ignored
   *  them would promise more than the user is about to see. A count of 0
   *  hides a chip, so these need no "not yet" value. */
  cropsLeft: number;
  filesLeft: number;
}

export function labelsProgressValue(
  data: LabelsProgress | undefined,
): LabelsProgressValue {
  const total = data?.total_labels ?? 0;
  const verified = data?.verified_labels ?? 0;
  return {
    pct: data === undefined ? null : total ? (verified / total) * 100 : 0,
    verified,
    total,
    cropsLeft: (data?.crop_labels ?? 0) - (data?.crop_labels_verified ?? 0),
    filesLeft: (data?.files ?? 0) - (data?.files_verified ?? 0),
    title:
      data === undefined
        ? "Counting the labels to verify"
        : `${verified.toLocaleString()} of ${total.toLocaleString()} labels ` +
          `verified: every box above your detection threshold, plus one for ` +
          `each file the AI found nothing in`,
  };
}
