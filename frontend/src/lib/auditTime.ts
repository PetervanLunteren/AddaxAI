/**
 * Rendering helpers for AUDIT timestamps — the `*_utc` columns recording when
 * the server did something (a backup was taken, a folder run was last
 * touched).
 *
 * Keep these apart from the observational/camera helpers in `datetime.ts`.
 * The two kinds are not interchangeable (see DEVELOPERS.md "Datetime
 * conventions"):
 *
 * - Observational (`*_local`): the camera's wall-clock time. Must render
 *   verbatim regardless of the viewer's timezone, so `toLocale*` is WRONG
 *   there — use `datetime.ts`.
 * - Audit (`*_utc`): an absolute moment. It SHOULD render in the viewer's own
 *   timezone, so plain `toLocale*` is exactly right — use this file.
 */

/**
 * Human "when" for an audit timestamp: a relative headline
 * ("Today, 14:03" / "Yesterday, 09:12" / "3 days ago", falling back to the
 * date once it's a week old) plus the full date for a secondary line.
 */
export function formatAuditWhen(iso: string): { rel: string; abs: string } {
  const d = new Date(iso);
  const time = d.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
  const abs = d.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });

  const startOfDay = (x: Date) =>
    new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round(
    (startOfDay(new Date()) - startOfDay(d)) / 86_400_000,
  );

  let rel: string;
  if (days <= 0) rel = `Today, ${time}`;
  else if (days === 1) rel = `Yesterday, ${time}`;
  else if (days < 7) rel = `${days} days ago`;
  else rel = abs;
  return { rel, abs };
}
