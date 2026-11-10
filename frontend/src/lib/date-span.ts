/**
 * One rule for every time chart: a span of capture years wider than
 * MAX_CHART_YEARS is a broken date, not a survey, so the chart shows a
 * notice instead of drawing it.
 *
 * One corrupt EXIF year (0025, 4500) stretches a project over millennia.
 * The survey effort calendar then overflowed the stack and blanked the
 * dashboard, and the Explore trend chart froze for about a minute walking
 * 1.6 million days (beta report, 2026-10-31). No camera trap programme is
 * this old.
 */
export const MAX_CHART_YEARS = 30;

export interface YearSpan {
  first: number;
  last: number;
}

/** Years of two `YYYY-MM-DD` dates (zero-padded, so year 25 is "0025"). */
export function yearSpan(from: string, to: string): YearSpan {
  return { first: Number(from.slice(0, 4)), last: Number(to.slice(0, 4)) };
}

/** True when the years would need more than a chart can sensibly draw. */
export function exceedsChartYears(span: YearSpan): boolean {
  return span.last - span.first + 1 > MAX_CHART_YEARS;
}
