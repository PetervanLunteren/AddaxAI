/**
 * The three metrics of the reprocess "how the DB changed" summary, named
 * to match the app-wide stat vocabulary (dashboard cards, site/deployment
 * sheets, statistics.py). Kept out of the modal component so it can be
 * shared with the hook without breaking fast-refresh.
 */

import type { SaveResults } from "../components/projects/SaveResultsModal";

/** Which cards to show. The value maps each to the SaveResults field it
 * reads (the field names are the older, looser wording). */
export type SaveMetric = "detections" | "observations" | "events";

export const METRIC_FIELD: Record<SaveMetric, keyof SaveResults> = {
  detections: "observations",
  observations: "independent_observations",
  events: "events",
};

export const METRIC_META: Record<SaveMetric, { title: string; subtitle: string }> = {
  detections: {
    title: "Detections",
    subtitle:
      "Every detection above the confidence threshold. The same animal photographed several times counts several times.",
  },
  observations: {
    title: "Observations",
    subtitle:
      "The most individuals seen at once in an event, summed across events. A burst of the same animal counts once.",
  },
  events: {
    title: "Events",
    subtitle:
      "Separate visits to a camera. Detections close together in time count as one event.",
  },
};

export const ALL_METRICS: SaveMetric[] = ["detections", "observations", "events"];
