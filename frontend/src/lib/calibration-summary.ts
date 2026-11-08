/**
 * The calibration curve of one label, in words.
 *
 * Shown under the graph so the result can be read without it. It states
 * what the curve does and where it passes the targets; it does not tell
 * the person what to trust or what to check. That is their call.
 */

import type { CalibrationLabel, CalibrationThreshold } from "../api/calibration";

function pct(t: CalibrationThreshold): string {
  return `${Math.round(t.target * 100)}%`;
}

/** "a", "a and b", "a, b and c". */
function list(items: string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

export function curveSummary(row: CalibrationLabel, name: string): string {
  const thresholds = row.thresholds ?? [];
  if (thresholds.length === 0) return "";
  const reached = thresholds.filter((t) => t.score !== null);
  const missed = thresholds.filter((t) => t.score === null);

  const parts: string[] = [];
  if (reached.length === 0) {
    parts.push(
      `Within the scores you answered, fewer than ${pct(thresholds[0])} of the ${name} ` +
        `labels are right at every score, so no score reaches ${list(thresholds.map(pct))}.`,
    );
  } else {
    parts.push(
      `Reading from left to right, the share of right ${name} labels rises with the score.`,
    );
    const passes = reached.map((t) => `${pct(t)} at ${t.score!.toFixed(2)}`);
    const tail = missed.length
      ? `, and does not reach ${list(missed.map(pct))} within the scores you answered`
      : "";
    parts.push(`It passes ${list(passes)}${tail}.`);
  }
  parts.push(
    `On the Labels page, the classification confidence slider under More filters ` +
      `selects the ${name} labels below or above a score.`,
  );
  parts.push(
    `Based on ${row.right + row.wrong} answers. The shaded band is the 95% uncertainty.`,
  );
  return parts.join(" ");
}
