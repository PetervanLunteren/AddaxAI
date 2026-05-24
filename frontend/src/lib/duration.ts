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

/** Rounded, worded duration: "less than a minute", "13 min",
 * "1 h 10 min". Rounds to the nearest minute. */
export function humanizeDuration(seconds: number): string {
  if (seconds < 60) return "less than a minute";
  const mins = Math.round(seconds / 60);
  if (mins < 60) return `${mins} min`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return m === 0 ? `${h} h` : `${h} h ${m} min`;
}

/** Snap an estimate to a stable bucket so a jittery ETA stops
 * flickering. Buckets grow with the value (nearest 1 min under 10 min,
 * 5 min under an hour, 15 min under 3 h, 30 min under 5 h, 1 h beyond):
 * precision should match how rough the guess is, and a big ETA only
 * needs to be roughly right. The displayed value then holds while the
 * true estimate wobbles within its bucket. */
function bucketEstimate(seconds: number): number {
  if (seconds < 60) return seconds;
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

/** Reformat a tqdm time string for display. With ``estimate`` set, the
 * value is snapped to a stable, magnitude-scaled bucket (see
 * ``bucketEstimate``) and prefixed with "about" — a jittery ETA then
 * holds a value instead of churning every second. The sub-minute case
 * stays "less than a minute" (no "about"). Falls back to the raw
 * string if it can't be parsed. */
export function humanizeTqdmTime(raw: string, estimate = false): string {
  const sec = tqdmTimeToSeconds(raw);
  if (sec === null) return raw;
  const value = estimate ? bucketEstimate(sec) : sec;
  const human = humanizeDuration(value);
  return estimate && value >= 60 ? `about ${human}` : human;
}
