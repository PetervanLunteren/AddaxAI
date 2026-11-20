/**
 * The metrics of the reprocess "how the DB changed" summary, named for
 * the two things users already verify: Labels (the Labels step) and
 * Counts (the Counts step). Kept out of the modal component so it can be
 * shared with the hook without breaking fast-refresh.
 */

import type {
  LabelCount,
  SaveResults,
} from "../components/projects/SaveResultsModal";

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

/** Compute per-label diffs from two snapshots, sorted by absolute change. */
export function computeLabelDiff(
  before: LabelCount[],
  after: LabelCount[],
): { label: string; before: number; after: number }[] {
  const beforeMap = new Map(before.map((s) => [s.label, s.count]));
  const afterMap = new Map(after.map((s) => [s.label, s.count]));
  const allLabels = new Set([...beforeMap.keys(), ...afterMap.keys()]);

  const diff: { label: string; before: number; after: number }[] = [];
  for (const lbl of allLabels) {
    const b = beforeMap.get(lbl) ?? 0;
    const a = afterMap.get(lbl) ?? 0;
    if (b !== a) diff.push({ label: lbl, before: b, after: a });
  }
  return diff.sort((a, b) => Math.abs(b.after - b.before) - Math.abs(a.after - a.before));
}

/** Whether any shown card has a number that moved. Without one, the
 * summary has nothing to show and the save gets a plain toast. */
export function hasChanges(results: SaveResults, metrics: SaveMetric[]): boolean {
  return metrics.some((m) => {
    const { before, after } = results[METRIC_FIELD[m]];
    return (
      before.total !== after.total ||
      computeLabelDiff(before.labels, after.labels).length > 0
    );
  });
}

/** "labels and counts", for the link that opens the summary. */
export function metricsPhrase(metrics: SaveMetric[]): string {
  return metrics.map((m) => METRIC_META[m].title.toLowerCase()).join(" and ");
}
