/**
 * Group classification models by region for the dropdown picker.
 *
 * The backend already returns `ModelInfo[]` sorted by (region, friendly_name)
 * so all this helper does is walk the list once and split adjacent
 * same-region entries into a single group per region. That keeps the
 * dropdown rendering trivial and lets the backend stay the source of
 * truth for sort order.
 */

import type { ModelInfo } from "../api/types";

export type ClsRegion =
  | "global"
  | "africa"
  | "americas"
  | "asia"
  | "europe"
  | "oceania"
  | "other";

export interface ClsModelGroup {
  region: ClsRegion;
  label: string;
  models: ModelInfo[];
}

/** User-facing label for each region. Title case is intentional here:
 *  these are headers, not body copy (CONVENTIONS rule 12 capitalises
 *  the first letter of headers). */
const REGION_LABELS: Record<ClsRegion, string> = {
  global: "Global",
  africa: "Africa",
  americas: "Americas",
  asia: "Asia",
  europe: "Europe",
  oceania: "Oceania",
  other: "Other",
};

/** Group a flat ModelInfo[] into one ClsModelGroup per region.
 *
 *  The input is expected to be already sorted by region + friendly_name
 *  (the backend does this); this function just walks it once and
 *  collects adjacent entries into groups. Models without a region land
 *  in the synthetic "Other" group at the end so the UI never silently
 *  drops anything.
 */
export function groupClassificationModels(
  models: ModelInfo[],
): ClsModelGroup[] {
  const groups = new Map<ClsRegion, ModelInfo[]>();
  for (const m of models) {
    const region: ClsRegion = (m.region as ClsRegion) ?? "other";
    if (!groups.has(region)) groups.set(region, []);
    groups.get(region)!.push(m);
  }
  return [...groups.entries()].map(([region, list]) => ({
    region,
    label: REGION_LABELS[region],
    models: list,
  }));
}
