/**
 * Turn one attribute split (sex, age or behaviour) into the segments of a
 * stacked bar, the way AddaxAI Connect draws its demographics card.
 *
 * The three largest known values keep their own segment and colour, the
 * rest fold into "Other", and Unknown always comes last in a neutral
 * colour. Colours go by position from the four-value palette because the
 * values are names, not magnitudes: "female" and "male" are not two points
 * on a ramp.
 */

import type { AttributeCount } from "../api/statistics";
import type { AttributeOption } from "./observation-attributes";

/** Palette slots by position (FRONTEND_CONVENTIONS, four distinct values). */
export const SEGMENT_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
];

/** Known values that keep their own segment before the rest fold into Other. */
const MAX_NAMED = 3;

export interface Segment {
  label: string;
  count: number;
  /** 0 to 1, of all observations in the split. */
  share: number;
  /** null draws the neutral unknown fill. */
  color: string | null;
}

export function toSegments(
  counts: AttributeCount[],
  options: readonly AttributeOption[],
): Segment[] {
  const total = counts.reduce((sum, c) => sum + c.count, 0);
  if (total === 0) return [];
  const labelOf = (value: string) =>
    options.find((o) => o.value === value)?.label ?? value;

  const known = counts
    .filter((c) => c.value !== null)
    .sort((a, b) => b.count - a.count);
  const unknown = counts
    .filter((c) => c.value === null)
    .reduce((sum, c) => sum + c.count, 0);

  // "Other" for a single leftover value would hide a name to save nothing.
  const named = known.length === MAX_NAMED + 1 ? known : known.slice(0, MAX_NAMED);
  const otherCount = known
    .slice(named.length)
    .reduce((sum, c) => sum + c.count, 0);

  const segments: Segment[] = named.map((c, i) => ({
    label: labelOf(c.value as string),
    count: c.count,
    share: c.count / total,
    color: SEGMENT_COLORS[i],
  }));
  if (otherCount > 0) {
    segments.push({
      label: "Other",
      count: otherCount,
      share: otherCount / total,
      color: SEGMENT_COLORS[MAX_NAMED],
    });
  }
  if (unknown > 0) {
    segments.push({
      label: "Unknown",
      count: unknown,
      share: unknown / total,
      color: null,
    });
  }
  return segments;
}
