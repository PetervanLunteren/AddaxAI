/**
 * The metrics of the reprocess "how the DB changed" summary, named for
 * the two things users already verify: Labels (the Labels step) and
 * Counts (the Counts step). Kept out of the modal component so it can be
 * shared with the hook without breaking fast-refresh.
 */

import type { SaveResults } from "../components/projects/SaveResultsModal";

/** Which cards to show. The value maps each to the SaveResults field it
 * reads (the field names are the older, looser wording). */
export type SaveMetric = "labels" | "counts";

export const METRIC_FIELD: Record<SaveMetric, keyof SaveResults> = {
  labels: "observations",
  counts: "independent_observations",
};

export const METRIC_META: Record<SaveMetric, { title: string; subtitle: string }> = {
  labels: {
    title: "Labels",
    subtitle: "Detections grouped by species label.",
  },
  counts: {
    title: "Counts",
    subtitle: "Individuals counted per species.",
  },
};

export const ALL_METRICS: SaveMetric[] = ["labels", "counts"];
