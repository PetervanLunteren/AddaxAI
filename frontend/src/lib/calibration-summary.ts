/**
 * The calibration result of one label, in words.
 *
 * `curveSummary` is shown under the graph so the result can be read
 * without it. It states what the curve does and where it passes the
 * targets; it does not tell the person what to trust or what to check.
 * That is their call. `statusLine` is what a label shows before it has
 * a curve.
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

/** "5 more wrong needed for a curve"; null once the minimum is reached. */
export function neededForCurve(row: CalibrationLabel): string | null {
  const parts = [
    row.right_needed > 0 ? `${row.right_needed} more right` : null,
    row.wrong_needed > 0 ? `${row.wrong_needed} more wrong` : null,
  ].filter(Boolean);
  return parts.length ? `${list(parts as string[])} needed for a curve` : null;
}

/**
 * The line under a label's name while it has no curve: the share right
 * so far, the share the answers can vouch for, and what a curve still
 * needs. Null for a label with a curve or without answers.
 */
export function statusLine(row: CalibrationLabel): string | null {
  if (row.thresholds) return null;
  const judged = row.right + row.wrong;
  const bound =
    row.lower_bound === null ? "" : `: at least ${Math.floor(row.lower_bound * 100)}% right`;
  const share = judged > 0 ? `${row.right} of ${judged} right${bound}` : null;
  const needed = row.unsure + judged > 0 ? neededForCurve(row) : null;
  return [share, needed].filter(Boolean).join(". ") || null;
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
