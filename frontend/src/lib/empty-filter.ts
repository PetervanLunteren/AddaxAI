/**
 * "Empty only" and the box filters cannot both hold.
 *
 * An empty file (or event) is one where no box passes the floor; a label
 * or confidence-range filter keeps only the ones where a passing box
 * matches. Together they always return nothing, which reads as "no
 * results" rather than as two filters that contradict each other. So the
 * last one chosen wins: picking "Empty only" clears the box filters, and
 * setting a box filter while "Empty only" is on drops it.
 */

import type { EventFilterParams } from "../api/types";

const BOX_FILTER_KEYS = [
  "labels",
  "min_confidence",
  "max_confidence",
  "min_label_confidence",
  "max_label_confidence",
] as const;

function sameValue(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((v, i) => v === b[i]);
  }
  return a === b;
}

export function reconcileEmptyFilter(
  prev: EventFilterParams,
  next: EventFilterParams,
): EventFilterParams {
  if (next.empty !== "show_only") return next;

  if (prev.empty !== "show_only") {
    const cleared = { ...next };
    for (const key of BOX_FILTER_KEYS) cleared[key] = undefined;
    return cleared;
  }

  const boxFilterSet = BOX_FILTER_KEYS.some(
    (key) =>
      !sameValue(prev[key], next[key]) &&
      next[key] !== undefined &&
      !(Array.isArray(next[key]) && next[key].length === 0),
  );
  return boxFilterSet ? { ...next, empty: undefined } : next;
}
