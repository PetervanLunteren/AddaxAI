/**
 * Datetime rendering helpers for observational timestamps.
 *
 * Observational datetimes (File.captured_at_local, Event.event_start_local,
 * etc.) come over the wire as ISO 8601 with the project's UTC offset, e.g.
 * "2013-01-26T08:25:00+03:00". The "08:25" part is the camera's wall-clock
 * time at the deployment location, and that's what the UI must always show
 * regardless of which timezone the viewer's browser is set to.
 *
 * The naive approach `new Date(iso).toLocaleTimeString(...)` parses to a
 * UTC moment and then converts to the viewer's local tz, which silently
 * shows the wrong hour for any user not in the project's timezone. These
 * helpers strip the offset and render the local components directly.
 *
 * See DEVELOPERS.md "Datetime conventions".
 */

/**
 * Parse the local components of an ISO 8601 string with offset into a
 * Date object pinned to UTC, so subsequent `toLocaleString` calls with
 * `timeZone: "UTC"` render the camera's wall-clock time verbatim.
 *
 * Returns `null` if the input is null/undefined or doesn't look like an
 * ISO datetime.
 */
function parseLocalAsUtc(iso: string | null | undefined): Date | null {
  if (!iso) return null;
  // Strip any trailing offset (Z or ±hh:mm or ±hhmm) and append Z so the
  // browser treats the local components as UTC.
  const stripped = iso.replace(/(?:Z|[+-]\d{2}:?\d{2})$/, "");
  const d = new Date(stripped + "Z");
  return Number.isNaN(d.getTime()) ? null : d;
}

const UTC: Intl.DateTimeFormatOptions = { timeZone: "UTC" };

/**
 * Format the camera's wall-clock date portion (e.g. "26 Jan 2013").
 */
export function formatCameraDate(
  iso: string | null | undefined,
  options: Intl.DateTimeFormatOptions = { day: "numeric", month: "short", year: "numeric" },
  locale: string | string[] | undefined = undefined,
): string {
  const d = parseLocalAsUtc(iso);
  if (!d) return "";
  return d.toLocaleDateString(locale, { ...options, ...UTC });
}

/**
 * Format the camera's wall-clock time portion (e.g. "08:25").
 */
export function formatCameraTime(
  iso: string | null | undefined,
  options: Intl.DateTimeFormatOptions = { hour: "2-digit", minute: "2-digit" },
  locale: string | string[] | undefined = undefined,
): string {
  const d = parseLocalAsUtc(iso);
  if (!d) return "";
  return d.toLocaleTimeString(locale, { ...options, ...UTC });
}

/**
 * Format the camera's wall-clock date+time as a single string
 * (locale's default formatting).
 */
export function formatCameraDateTime(
  iso: string | null | undefined,
  options: Intl.DateTimeFormatOptions = {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  },
  locale: string | string[] | undefined = undefined,
): string {
  const d = parseLocalAsUtc(iso);
  if (!d) return "";
  return d.toLocaleString(locale, { ...options, ...UTC });
}

/**
 * Return a Date object whose `getHours()`, `getMinutes()`, etc. read out
 * the camera's wall-clock components. Useful when callers need to compare
 * `same date?` / `same time?` between two observational timestamps without
 * rendering through Intl.
 */
export function asCameraDate(iso: string | null | undefined): Date | null {
  return parseLocalAsUtc(iso);
}
