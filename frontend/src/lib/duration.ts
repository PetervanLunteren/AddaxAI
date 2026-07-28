/**
 * Human-friendly duration formatting for progress displays.
 *
 * tqdm reports elapsed / remaining as "MM:SS" (or "H:MM:SS") strings,
 * which read like a stopwatch and churn every second. For long runs a
 * rounded, worded value ("about 35 min") is calmer and more honest
 * about how precise the estimate really is.
 */

/** Parse a tqdm time string ("MM:SS" / "H:MM:SS") into seconds, or
 * null if it doesn't look like one. */
function tqdmTimeToSeconds(raw: string): number | null {
  const parts = raw.trim().split(":");
  if (parts.length < 2 || parts.length > 3) return null;
  const nums = parts.map((p) => Number(p));
  if (nums.some((n) => Number.isNaN(n))) return null;
  return parts.length === 2
    ? nums[0] * 60 + nums[1]
    : nums[0] * 3600 + nums[1] * 60 + nums[2];
}

/** Worded duration. Under a minute shows the real seconds ("45 sec");
 * above, whole minutes / hours ("13 min", "1 h 10 min"). Floors so it
 * never overstates. Used for the exact elapsed time and, after
 * bucketing, the remaining estimate. */
export function humanizeDuration(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)} sec`;
  const mins = Math.floor(seconds / 60);
  if (mins < 60) return `${mins} min`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return m === 0 ? `${h} h` : `${h} h ${m} min`;
}

/** Snap an estimate (>= 1 min) to a stable bucket so a jittery ETA stops
 * flickering. Buckets grow with the value (nearest 1 min under 10 min,
 * 5 min under an hour, 15 min under 3 h, 30 min under 5 h, 1 h beyond):
 * precision should match how rough the guess is, and a big ETA only
 * needs to be roughly right. The displayed value then holds while the
 * true estimate wobbles within its bucket. */
function bucketEstimate(seconds: number): number {
  const mins = seconds / 60;
  const step =
    mins < 10
      ? 1
      : mins < 60
        ? 5
        : mins < 180
          ? 15
          : mins < 300
            ? 30
            : 60;
  return Math.round(mins / step) * step * 60;
}

/** Format a processing speed for the progress info card.
 *
 * The backend normalises tqdm speeds to items per second, so slow work
 * (a minute per video) arrives as 0.0166 and would render as "0.0",
 * which reads as stalled. Below 1/s the display flips to the inverse,
 * like tqdm's own "s/it" mode: "Time per video: 34 sec". Above 1/s the
 * familiar "Videos per second: 2.3" stays. */
export function formatRate(
  rate: number,
  unit: string,
): { label: string; value: string } {
  const capitalized = unit.charAt(0).toUpperCase() + unit.slice(1);
  if (rate >= 1) {
    return { label: `${capitalized} per second`, value: rate.toFixed(1) };
  }
  const secondsPer = 1 / rate;
  const value =
    secondsPer < 10
      ? `${secondsPer.toFixed(1)} sec`
      : humanizeDuration(secondsPer);
  return { label: `Time per ${unit}`, value };
}

/** Reformat a tqdm time string for display.
 *
 * Elapsed (``estimate=false``) is an exact fact, so it shows real
 * seconds under a minute ("45 sec"). The remaining estimate is a guess,
 * so it stays coarse: a magnitude-scaled bucket prefixed with "about"
 * ("about 35 min"), and just "less than a minute" in the final stretch
 * — a second-by-second countdown on a guess churns and overstates its
 * precision. Falls back to the raw string if it can't be parsed. */
export function humanizeTqdmTime(raw: string, estimate = false): string {
  const sec = tqdmTimeToSeconds(raw);
  if (sec === null) return raw;
  if (estimate) {
    return sec < 60
      ? "less than a minute"
      : `about ${humanizeDuration(bucketEstimate(sec))}`;
  }
  return humanizeDuration(sec);
}
