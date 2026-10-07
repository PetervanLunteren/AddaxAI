// Sun bands may leave [0, 24): see DEVELOPERS.md, "Sun bands are
// ordered, not clamped to the clock". Mirrors `in_sun_band` in
// backend/app/ml/sun_time.py.

/** Whether `hour` falls in `[start, end)` on the 24 h circle (`start < end`). */
export function inSunBand(hour: number, start: number, end: number): boolean {
  return (((hour - start) % 24) + 24) % 24 < end - start;
}

/** Whether the span `[spanStart, spanEnd)` touches the band `[start, end)`. */
export function overlapsSunBand(
  spanStart: number,
  spanEnd: number,
  start: number,
  end: number,
): boolean {
  // Two arcs on a circle overlap exactly when one contains the other's start.
  return inSunBand(spanStart, start, end) || inSunBand(start, spanStart, spanEnd);
}
