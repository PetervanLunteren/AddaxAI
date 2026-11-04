/**
 * Survey effort as a calendar: one row per year, one cell per week, the
 * value is how many cameras were running that week.
 *
 * Built from the deployment timeline's `concurrent_cameras` series, which
 * holds change points only: each point's count applies from its date up
 * to the next point's date, and the last point drops to zero. Rows are
 * years rather than GitHub's weekdays, because wildlife does not keep a
 * week: a season then lines up down the column across years.
 */

export const WEEKS_PER_YEAR = 53;

/**
 * Widest span of years the calendar draws. One corrupt EXIF year (0025,
 * 4500) stretches a camera's interval over millennia: the grid would be
 * thousands of rows, and spreading every cell into one Math.max call
 * overflowed the stack and blanked the dashboard (beta report,
 * 2026-10-31). No camera trap programme is this old, so a wider span is a
 * broken date, and the card says so instead of drawing it.
 */
export const MAX_CALENDAR_YEARS = 30;

export interface CameraCountPoint {
  /** YYYY-MM-DD */
  date: string;
  count: number;
}

const DAY_MS = 86_400_000;

const toUtc = (iso: string) => Date.parse(`${iso}T00:00:00Z`);

const yearOf = (utcMs: number) => new Date(utcMs).getUTCFullYear();

const sortByDate = (points: CameraCountPoint[]) =>
  [...points].sort((a, b) => toUtc(a.date) - toUtc(b.date));

/** Week of the year, 0-52: days 1-7 are week 0, day 365/366 is week 52. */
export function weekOfYear(utcMs: number): number {
  const d = new Date(utcMs);
  const start = Date.UTC(d.getUTCFullYear(), 0, 1);
  return Math.floor((utcMs - start) / DAY_MS / 7);
}

/**
 * First and last calendar year with a camera running, or null without
 * effort. Each point's count applies up to the next point's date, so the
 * last year comes from the day before the final drop to zero.
 */
export function effortYears(
  points: CameraCountPoint[],
): { first: number; last: number } | null {
  const sorted = sortByDate(points);
  let first: number | null = null;
  let last: number | null = null;
  for (let i = 0; i < sorted.length - 1; i++) {
    if (sorted[i].count <= 0) continue;
    first ??= yearOf(toUtc(sorted[i].date));
    last = yearOf(toUtc(sorted[i + 1].date) - DAY_MS);
  }
  return first === null || last === null ? null : { first, last };
}

/** True when the years would need more rows than the calendar draws. */
export function exceedsCalendar(span: { first: number; last: number }): boolean {
  return span.last - span.first + 1 > MAX_CALENDAR_YEARS;
}

/**
 * The most cameras running on any day of each week, per year, from the
 * first year with effort to the last. Years in between with no effort get
 * a row of zeros, so a missing season shows as a gap rather than vanishing.
 * Empty when there is no effort, or when it spans more years than the
 * calendar draws (the card shows a notice for that).
 */
export function weeklyCameras(
  points: CameraCountPoint[],
): { year: number; weeks: number[] }[] {
  const span = effortYears(points);
  if (span === null || exceedsCalendar(span)) return [];
  const sorted = sortByDate(points);
  const byYear = new Map<number, number[]>();
  const row = (year: number) => {
    let weeks = byYear.get(year);
    if (!weeks) {
      weeks = new Array(WEEKS_PER_YEAR).fill(0);
      byYear.set(year, weeks);
    }
    return weeks;
  };

  for (let i = 0; i < sorted.length - 1; i++) {
    const count = sorted[i].count;
    if (count <= 0) continue;
    const end = toUtc(sorted[i + 1].date);
    for (let day = toUtc(sorted[i].date); day < end; day += DAY_MS) {
      const weeks = row(new Date(day).getUTCFullYear());
      const w = weekOfYear(day);
      weeks[w] = Math.max(weeks[w], count);
    }
  }

  const years = [...byYear.keys()];
  const out = [];
  for (let y = Math.min(...years); y <= Math.max(...years); y++) {
    out.push({ year: y, weeks: byYear.get(y) ?? new Array(WEEKS_PER_YEAR).fill(0) });
  }
  return out;
}

/** 0 for no cameras, else 1-4 by share of the busiest week, as GitHub does. */
export function effortLevel(value: number, max: number): number {
  if (value <= 0 || max <= 0) return 0;
  return Math.min(4, Math.ceil((value / max) * 4));
}
