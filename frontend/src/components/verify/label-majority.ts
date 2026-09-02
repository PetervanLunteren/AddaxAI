/**
 * The most common label in a set of detections.
 *
 * One rule for the Detections grid's Match majority (over the selected
 * cards) and the Files viewer's (over every visible box on the file), so
 * the two cannot disagree. Ties resolve to the first label encountered,
 * which is deterministic given the caller's iteration order.
 */

export interface LabelMajority {
  count: number;
  label: string;
  category: string;
  common_name: string | null;
  scientific_name: string | null;
}

interface Labelled {
  label: string | null;
  category: string;
  common_name: string | null;
  scientific_name: string | null;
}

/** Returns null when nothing in `items` carries a label. */
export function labelMajority(items: Iterable<Labelled>): LabelMajority | null {
  const counts = new Map<string, LabelMajority>();
  for (const d of items) {
    if (!d.label) continue;
    const entry = counts.get(d.label);
    if (entry) {
      entry.count += 1;
    } else {
      counts.set(d.label, {
        count: 1,
        label: d.label,
        category: d.category,
        common_name: d.common_name,
        scientific_name: d.scientific_name,
      });
    }
  }
  let mode: LabelMajority | null = null;
  for (const entry of counts.values()) {
    if (!mode || entry.count > mode.count) mode = entry;
  }
  return mode;
}
